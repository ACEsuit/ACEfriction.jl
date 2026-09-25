# Fast (contracted) evaluators for the ET friction backend.
#
# The generic path `evaluate(basis, Rs, Zs)` materialises every basis function as a
# Cartesian block (Rnl, Ylm -> A -> AA -> B = A2B*AA -> blocks) and only then
# contracts with the coefficients. For the *model* (Σ = Σₖ cₖ Bₖ) this is wasteful:
# the A2B map and the spherical->Cartesian output transform are both linear, so they
# are folded into the coefficients once,
#
#     W[:, q] = Σ_L Σ_{k ∈ L} c_k ⊗ (Tcart_L · A2B_L[k, q])       (NC·NR × nAA)
#     vec(Σ)  = W · AA
#
# where Tcart_L is the (NC × 2L+1) spherical->Cartesian map of channel L and NC the
# number of Cartesian components of a block. This is the friction analogue of
# ACEpotentials' `fast_evaluator` (wAA = A2Bmap' * wB): the ET tensor is used only
# for its *specification* (A spec, AA spec, symmetrisation matrices); evaluation
# runs through the kernels below. `W · AA` is fused with the AA evaluation, so AA is
# never stored. The columns of W are sparse (each AA function feeds only 2-3 of the 9
# components of a 3×3 block, 1 of 3 for vectors), so the AA functions are grouped by
# their structural nonzero components and each group accumulates only those in
# registers (`_SGroup`/`_TGroup`, `_GroupedW`). Per site the cost is radial + Ylm + A
# (as for an energy) plus the grouped fused pass.
#
# Bond (offsite) bases of atom-centred cutoffs (Spherical / SnowMan) additionally
# share the neighbour data of a centre across all its bonds (`ETBondCentre`):
#
#   * exact (`partner_in_env = false`, the original basis): the env A of bond (i,j)
#     is A_i - a_j (pooled over all neighbours, minus the partner's contribution), so
#     radial / Ylm / A are computed once per centre; the fused AA·W pass is per bond.
#   * factorised (`partner_in_env = true`): the env A is A_i for every bond, and since
#     each bond function has exactly one bond factor φ_{a0}(r_ij),
#         Σ_ij = T_i · φ(r_ij),   T_i[:, a0] = Σ_{q=(a0,e)} W[:, q] · AAenv_i[e].
#     T_i costs one fused pass per centre; each bond is an (NC·NR × n_bond1p) mat-vec.
#     The un-contracted basis (fitting path) factorises the same way,
#         B_k(ij) = Σ_{s ∈ slots(k)} φ_{b(s)}(r_ij) · P_i[s]   (`ETBasisFactorisation`),
#     with per-centre coefficients P_i, so no product basis is evaluated per bond.
#
# The fused weights, sparsity groups and factorisations are built lazily on first use
# (a model used only for fitting never builds the fused weights) and published
# atomically (`_Lazy`), so concurrent first use from several threads is safe.
#
# All per-site buffers (radial rows, Ylm, pooled A, ...) live in reusable
# workspaces (`ETSiteData`, `ETBondCentre`) so the assembly loops do not allocate
# per centre. Workspaces are owned by the caller (one per assembly call), never by
# the model: evaluating one model concurrently from several threads is safe as long
# as its coefficients are not changed at the same time.

using StaticArrays, LinearAlgebra, SparseArrays
import Polynomials4ML as P4ML
import EquivariantTensors as ET

# ----------------------------------------------------------------------
# per-site neighbour workspace

"""
    ETSiteData

Reusable neighbour workspace of a fast site: species-independent radial rows
`RN[j, n]`, spherical harmonics `Y[j, iy]`, the species-block index of each
neighbour, the neighbour indices grouped by species block, and the pooled `A`
over *all* neighbours. Only rows `1:nn` are valid; the buffers grow on demand.
Fill with [`site_data!`](@ref).
"""
mutable struct ETSiteData
   nn::Int
   Rsc::Vector{SVector{3, Float64}}
   xs::Vector{Float64}
   RN::Matrix{Float64}
   Y::Matrix{Float64}
   izs::Vector{Int}
   jz::Vector{Vector{Int}}
   A::Vector{Float64}
end

function ETSiteData(nA::Int, nR::Int, nY::Int, nz::Int; cap::Int = 0)
   return ETSiteData(0, SVector{3, Float64}[], Float64[], zeros(cap, nR), zeros(cap, nY),
                     Int[], [ Int[] for _ in 1:nz ], zeros(nA))
end

Base.length(sd::ETSiteData) = sd.nn

function _ensure_rows!(sd::ETSiteData, nn::Int)
   if size(sd.RN, 1) < nn
      cap = max(nn, 2 * size(sd.RN, 1))
      sd.RN = zeros(cap, size(sd.RN, 2))
      sd.Y = zeros(cap, size(sd.Y, 2))
   end
   resize!(sd.Rsc, nn); resize!(sd.xs, nn); resize!(sd.izs, nn)
   sd.nn = nn
   return sd
end

# ----------------------------------------------------------------------
# fast site: coefficient-independent evaluation data of a site basis

"""
    ETFastSite(basis::ETFrictionSiteBasis)

Coefficient-independent evaluation data of a site basis: the A spec grouped by
species block (for pooling from species-independent radial rows), the ET AA
basis (`SparseSymmProd`) whose spec drives the fused kernels, and a private
per-call neighbour workspaces (`ETSiteData(fs)`).
"""
struct ETFastSite{TB, TAA}
   basis::TB
   nA::Int
   nAA::Int
   apool::Vector{Vector{NTuple{3, Int}}}   # per species iz: (iA, n, iy)
   aabasis::TAA
   NC::Int
end

_nY(basis::ETFrictionSiteBasis) = length(P4ML.natural_indices(basis.ybasis))

function ETFastSite(basis::ETFrictionSiteBasis)
   rb = basis.rbasis; T = basis.tensor
   apool = [ NTuple{3, Int}[] for _ in 1:_nz(rb) ]
   for (iA, (ñ, iy)) in enumerate(T.abasis.spec)
      iz = div(ñ - 1, rb.nR) + 1; n = mod1(ñ, rb.nR)
      push!(apool[iz], (iA, n, iy))
   end
   nA = length(T.abasis.spec)
   return ETFastSite(basis, nA, length(T.aabasis), apool, T.aabasis,
                     length(block_type(basis)))
end

Base.length(fs::ETFastSite) = length(fs.basis)
block_type(fs::ETFastSite) = block_type(fs.basis)

"a fresh (empty) neighbour workspace for `fs`"
ETSiteData(fs::ETFastSite) = ETSiteData(fs.nA, fs.basis.rbasis.nR, _nY(fs.basis), _nz(fs.basis.rbasis))

# in-place radial rows: RN[j, n] = Pₙ(x(rⱼ))·env(x(rⱼ)); `xs` holds the distances on
# entry and the transformed coordinates on exit
function _radial_rows!(RN::AbstractMatrix, b::SpeciesRadialBasis, xs::AbstractVector)
   @inbounds for j in eachindex(xs)
      xs[j] = b.trans(xs[j])
   end
   P4ML.evaluate!(RN, b.polys, xs)
   @inbounds for j in eachindex(xs)
      e = env_val(b.env, xs[j])
      for n in axes(RN, 2)
         RN[j, n] *= e
      end
   end
   return RN
end

"""
    site_data!(sd, fs, Rs, Zs; scale = 1.0) -> sd

Fill the workspace `sd` for a centre with neighbour vectors `Rs` (multiplied by
`scale`, e.g. `1/rcut` for bond bases whose radial basis lives on the unit ball)
and species `Zs`.
"""
function site_data!(sd::ETSiteData, fs::ETFastSite, Rs::AbstractVector{<:SVector{3}},
                    Zs::AbstractVector; scale::Real = 1.0)
   rb = fs.basis.rbasis
   nn = length(Rs)
   _ensure_rows!(sd, nn)
   for l in sd.jz; empty!(l); end
   if nn == 0
      fill!(sd.A, 0.0)
      return sd
   end
   @inbounds for j in 1:nn
      r = Rs[j] * scale
      sd.Rsc[j] = r; sd.xs[j] = norm(r)
      iz = _z2i(rb, Zs[j]); sd.izs[j] = iz; push!(sd.jz[iz], j)
   end
   _radial_rows!(view(sd.RN, 1:nn, :), rb, sd.xs)
   P4ML.evaluate!(view(sd.Y, 1:nn, :), fs.basis.ybasis, sd.Rsc)
   _pool_A!(sd.A, fs, sd.RN, sd.Y, sd.jz)
   return sd
end

"""
    site_data(fs, Rs, Zs; scale = 1.0) -> ETSiteData

Allocating variant of [`site_data!`](@ref) (a fresh workspace).
"""
site_data(fs::ETFastSite, Rs::AbstractVector{<:SVector{3}}, Zs::AbstractVector; scale::Real = 1.0) =
      site_data!(ETSiteData(fs), fs, Rs, Zs; scale = scale)

# A[iA] = Σ_{j in species block iz} RN[j, n] * Y[j, iy]; inner loop over the
# neighbours of one species block so it vectorises.
function _pool_A!(A, fs::ETFastSite, RN, Y, jz)
   fill!(A, 0.0)
   @inbounds for iz in eachindex(fs.apool)
      js = jz[iz]
      isempty(js) && continue
      for (iA, n, iy) in fs.apool[iz]
         a = 0.0
         @simd for t in eachindex(js)
            j = js[t]
            a = muladd(RN[j, n], Y[j, iy], a)
         end
         A[iA] = a
      end
   end
   return A
end

# ----------------------------------------------------------------------
# fused AA · W kernels
#
# `Wcols[q]` is the SVector{M} of contraction weights of AA function q. The kernels
# walk the ET `SparseSymmProd` spec (tuples grouped by correlation order), form
# each AA value as a static product of A entries and accumulate `aa * Wcols[q]`
# without storing AA.

@inline _prodA(A, ϕ::NTuple{N, Int}) where {N} =
      prod(ntuple(t -> (@inbounds A[ϕ[t]]), Val(N)))

"""
    fused_contract(fs, Wcols, A) -> SVector{M}

`Σ_q Wcols[q] * AA[q](A)` with AA the ET symmetric-product basis of `fs`.
"""
function fused_contract(fs::ETFastSite, Wcols::Vector{SVector{M, T}},
                        A::AbstractVector) where {M, T}
   aab = fs.aabasis
   acc = zero(SVector{M, T})
   if aab.hasconst
      acc += Wcols[1]
   end
   return _fused_orders(acc, Wcols, aab.specs, aab.ranges, A)
end

@generated function _fused_orders(acc, Wcols, specs::NTuple{ORD, Any}, ranges, A) where {ORD}
   quote
      Base.Cartesian.@nexprs $ORD N -> (acc = _fused_order(acc, Wcols, specs[N],
                                                           first(ranges[N]) - 1, A))
      return acc
   end
end

function _fused_order(acc::SVector{M, T}, Wcols, spec::Vector{NTuple{N, Int}},
                      off::Int, A) where {M, T, N}
   @inbounds for i in eachindex(spec)
      acc = acc + _prodA(A, spec[i]) * Wcols[off + i]
   end
   return acc
end

# ----------------------------------------------------------------------
# coefficient contraction: fold c, A2B maps and Cartesian maps into W

"""
    contraction_columns(fs, c::Vector{SVector{NR}}) -> Vector{SVector{NC*NR}}

Column `q` holds the weights of AA function `q` for all Cartesian components and
replicas: `vec(Σ[r])[a] = Σ_q Wcols[q][(r-1)*NC + a] * AA[q]`.
"""
function contraction_columns(fs::ETFastSite, c::AbstractVector{SVector{NR, Float64}}) where {NR}
   NC = fs.NC; M = NC * NR
   W = zeros(M, fs.nAA)
   k0 = 0
   for (il, A2B) in enumerate(fs.basis.tensor.A2Bmaps)
      Ts = fs.basis.out.Tcart[il]
      rows, cols, vals = findnz(A2B)
      for t in eachindex(rows)
         k = k0 + rows[t]; q = cols[t]
         v = Ts * _svec(vals[t])             # NC Cartesian components of this (k, q) entry
         ck = c[k]
         for r in 1:NR, a in 1:NC
            W[(r - 1) * NC + a, q] += ck[r] * v[a]
         end
      end
      k0 += size(A2B, 1)
   end
   return [ SVector{M, Float64}(view(W, :, q)) for q in 1:fs.nAA ]
end

"split a flat SVector{NC*NR} into NR Cartesian blocks"
@inline _blocks_from(::Type{BT}, v, ::Val{NR}) where {BT, NR} =
      SVector{NR, BT}(ntuple(r -> BT(ntuple(a -> v[(r - 1) * length(BT) + a], length(BT))), NR))

# ----------------------------------------------------------------------
# sparsity-grouped contraction
#
# A fused weight column W[:, q] is sparse in its Cartesian components: an AA function
# couples to few (L, M) outputs, and each (L, M) touches at most three entries of a
# 3×3 block (one of a vector). For 3×3 blocks only a handful of distinct patterns of
# 2–3 nonzero components (out of 9) occur. The AA functions are therefore grouped by
# their *structural* (coefficient-independent) nonzero components: a group
# accumulates into a register vector of PW components per replica (PW = 3 for 3×3
# blocks, 1 for vectors) and is scattered into the block once. Wider patterns are
# split over several groups and narrower ones padded (row 0, zero weight), so the
# kernels are type-stable and exact for any basis. This divides the dominant cost —
# streaming the weights — by about NC / PW.

_pattern_width(NC::Int) = NC == 9 ? 3 : NC == 3 ? 1 : NC

"structural nonzero Cartesian components (⊆ 1:NC) of every AA function's weight column"
function _weight_patterns(fs::ETFastSite)
   pats = [ falses(fs.NC) for _ in 1:fs.nAA ]
   for (il, A2B) in enumerate(fs.basis.tensor.A2Bmaps)
      Ts = fs.basis.out.Tcart[il]
      rows, cols, vals = findnz(A2B)
      for t in eachindex(rows)
         v = Ts * _svec(vals[t])
         for a in 1:fs.NC
            iszero(v[a]) || (pats[cols[t]][a] = true)
         end
      end
   end
   return [ findall(p) for p in pats ]
end

# split a pattern into chunks of PW components, padding the last one with row 0
_pattern_chunks(p::Vector{Int}, ::Val{PW}) where {PW} =
      [ ntuple(t -> (i0 + t - 1 <= length(p) ? p[i0 + t - 1] : 0), Val(PW)) for i0 in 1:PW:length(p) ]

"""
    _SGroup{PW, TS}

AA functions sharing the Cartesian components `rows` (0 = padding): member AA
indices `qs` and their A-index tuples by correlation order (`specs`), in kernel order.
"""
struct _SGroup{PW, TS}
   rows::NTuple{PW, Int}
   qs::Vector{Int}
   specs::TS
end

_specs_empty(ORD) = ntuple(N -> NTuple{N, Int}[], ORD)
_sgroup_type(fs::ETFastSite) =
      _SGroup{_pattern_width(fs.NC), typeof(_specs_empty(length(fs.aabasis.specs)))}

function _build_sigma_groups(fs::ETFastSite)
   PW = _pattern_width(fs.NC); aab = fs.aabasis; ORD = length(aab.specs)
   pats = _weight_patterns(fs)
   G = _sgroup_type(fs); groups = G[]; bykey = Dict{NTuple{PW, Int}, Int}()
   for N in 1:ORD, (i, ϕ) in enumerate(aab.specs[N])
      q = first(aab.ranges[N]) + i - 1
      for rows in _pattern_chunks(pats[q], Val(PW))
         g = get!(bykey, rows) do
            push!(groups, G(rows, Int[], _specs_empty(ORD))); length(groups)
         end
         push!(groups[g].qs, q); push!(groups[g].specs[N], ϕ)
      end
   end
   return groups
end

"""
    _TGroup{PW, TS}

Bond-factorisation counterpart of `_SGroup`: the `(q, b, env)` entries (sorted by
bond factor `b` within each order) of the AA functions sharing the components `rows`.
"""
struct _TGroup{PW, TS}
   rows::NTuple{PW, Int}
   qs::Vector{Int}
   tspecs::TS
end

_tgroup_type(fs::ETFastSite) =
      _TGroup{_pattern_width(fs.NC), typeof(_tspecs_empty(length(fs.aabasis.specs)))}

function _build_tensor_groups(fs::ETFastSite, fb)
   PW = _pattern_width(fs.NC); ORD = length(fb.tspecs)
   pats = _weight_patterns(fs)
   G = _tgroup_type(fs); groups = G[]; bykey = Dict{NTuple{PW, Int}, Int}()
   for N in 1:ORD, (q, b, env) in fb.tspecs[N]               # sorted by b: order is kept
      for rows in _pattern_chunks(pats[q], Val(PW))
         g = get!(bykey, rows) do
            push!(groups, G(rows, Int[], _tspecs_empty(ORD))); length(groups)
         end
         push!(groups[g].qs, q); push!(groups[g].tspecs[N], (q, b, env))
      end
   end
   return groups
end

"""
    _GroupedW{M, MP}

Fused weights compressed to a group layout: per group, one `SVector{MP}` (`MP =
PW·NR`: the group's components for all replicas) per member, plus the full column
`c0` of the constant AA function (onsite bases that contain it).
"""
struct _GroupedW{M, MP}
   c0::SVector{M, Float64}
   W::Vector{Vector{SVector{MP, Float64}}}
end

function _GroupedW{M, MP}(Wd::Vector{SVector{M, Float64}}, groups, hasconst::Bool, NC::Int) where {M, MP}
   PW = _pattern_width(NC)
   W = [ [ SVector{MP, Float64}(ntuple(t -> begin
               r = (t - 1) ÷ PW + 1; row = g.rows[(t - 1) % PW + 1]
               row == 0 ? 0.0 : Wd[q][(r - 1) * NC + row]
            end, Val(MP))) for q in g.qs ] for g in groups ]
   return _GroupedW{M, MP}(hasconst ? Wd[1] : zero(SVector{M, Float64}), W)
end

# Σ over all groups: accumulate each group in registers, scatter into the block once
function _grouped_sigma(groups::Vector{_SGroup{PW, TS}}, gw::_GroupedW{M, MP}, A,
                        ::Val{NC}) where {PW, TS, M, MP, NC}
   out = MVector{M, Float64}(gw.c0)
   @inbounds for g in eachindex(groups)
      grp = groups[g]
      acc = _group_orders(zero(SVector{MP, Float64}), grp.specs, gw.W[g], A)
      for r in 1:(MP ÷ PW), p in 1:PW
         row = grp.rows[p]; row == 0 && continue
         out[(r - 1) * NC + row] += acc[(r - 1) * PW + p]
      end
   end
   return SVector(out)
end

@generated function _group_orders(acc, specs::NTuple{ORD, Any}, Wg, A) where {ORD}
   quote
      off = 0
      Base.Cartesian.@nexprs $ORD N -> begin
         acc = _fused_order(acc, Wg, specs[N], off, A)
         off += length(specs[N])
      end
      return acc
   end
end

# per-centre tensor T[b] += Σ_{q ∈ b} W[:, q] · AAenv(q), group by group
function _grouped_tensor!(T::Vector{SVector{M, Float64}}, groups::Vector{_TGroup{PW, TS}},
                          gw::_GroupedW{M, MP}, A, ::Val{NC}) where {M, PW, TS, MP, NC}
   @inbounds for g in eachindex(groups)
      _grouped_tensor_orders!(T, groups[g].rows, groups[g].tspecs, gw.W[g], A, Val(NC))
   end
   return T
end

@generated function _grouped_tensor_orders!(T, rows, tspecs::NTuple{ORD, Any}, Wg, A, vnc) where {ORD}
   quote
      off = 0
      Base.Cartesian.@nexprs $ORD N -> begin
         _grouped_tensor_order!(T, rows, tspecs[N], Wg, off, A, vnc)
         off += length(tspecs[N])
      end
      return T
   end
end

function _grouped_tensor_order!(T::Vector{SVector{M, Float64}}, rows::NTuple{PW, Int},
                                tspec::Vector{Tuple{Int, Int, NTuple{K, Int}}},
                                Wg::Vector{SVector{MP, Float64}}, off::Int, A,
                                vnc::Val{NC}) where {M, PW, K, MP, NC}
   isempty(tspec) && return T
   bcur = tspec[1][2]; acc = zero(SVector{MP, Float64})
   @inbounds for i in eachindex(tspec)
      (_, b, env) = tspec[i]
      if b != bcur
         T[bcur] = _embed_add(T[bcur], acc, rows, vnc)
         bcur = b; acc = zero(SVector{MP, Float64})
      end
      acc = acc + _prodA(A, env) * Wg[off + i]
   end
   @inbounds T[bcur] = _embed_add(T[bcur], acc, rows, vnc)
   return T
end

@inline function _embed_add(t::SVector{M, Float64}, acc::SVector{MP, Float64},
                            rows::NTuple{PW, Int}, ::Val{NC}) where {M, MP, PW, NC}
   m = MVector(t)
   @inbounds for r in 1:(MP ÷ PW), p in 1:PW
      row = rows[p]; row == 0 && continue
      m[(r - 1) * NC + row] += acc[(r - 1) * PW + p]
   end
   return SVector(m)
end

# ----------------------------------------------------------------------
# thread-safe lazily built values

"""
    _Lazy{T}

A value of type `T` built on first use by [`_getlazy!`](@ref): built at most once,
under a lock, and published atomically, so concurrent readers never see a partially
built value.
"""
mutable struct _Lazy{T}
   @atomic val::Union{Nothing, T}
   const lock::ReentrantLock
end
_Lazy{T}() where {T} = _Lazy{T}(nothing, ReentrantLock())

@inline function _getlazy!(f, l::_Lazy{T})::T where {T}
   v = @atomic :acquire l.val
   v === nothing || return v
   return lock(l.lock) do
      v2 = @atomic :acquire l.val
      v2 === nothing || return v2
      nv = f()::T
      @atomic :release l.val = nv
      nv
   end
end

# ----------------------------------------------------------------------
# contracted site model

# coefficient snapshot + its (group-compressed) fused weights, each built on first
# use; immutable, so a coefficient change publishes a new snapshot instead of mutating
# a shared one. `S`: layout of the per-site / per-bond pass; `T`: layout of the
# per-centre bond tensor (partner-in-environment bond models).
struct _FusedWeights{M, NR, MP}
   c::Vector{SVector{NR, Float64}}
   S::_Lazy{_GroupedW{M, MP}}
   T::_Lazy{_GroupedW{M, MP}}
end
_FusedWeights{M, NR, MP}(c) where {M, NR, MP} =
      _FusedWeights{M, NR, MP}(SVector{NR, Float64}[ ci for ci in c ], _Lazy{_GroupedW{M, MP}}(),
                               _Lazy{_GroupedW{M, MP}}())

"""
    ETFastModel(fs::ETFastSite, c)
    ETFastModel(basis::ETFrictionSiteBasis, c)

Coefficient-contracted evaluator of a site basis. Holds a snapshot of `c` and builds
the fused weights (`contraction_columns`, compressed to the sparsity-grouped layout
of the kernels) on first use; [`refresh!`](@ref) replaces the snapshot iff `c`
changed. The coefficient-independent layouts (sparsity groups; for bond bases the
bond factorisation and the fitting-path basis factorisation) are also built on first
use. Nothing is built at construction, so a model used only for fitting never pays
for the fused weights. All lazily built data is published atomically: concurrent
evaluation of one model from several threads is safe.
"""
mutable struct ETFastModel{M, NR, MP, TFS, TFB, TBF, TSG, TTG}
   const fs::TFS
   @atomic weights::_FusedWeights{M, NR, MP}
   const fb::_Lazy{TFB}                 # ETBondFactorisation (bond bases) / Nothing
   const bf::_Lazy{TBF}                 # ETBasisFactorisation (bond bases) / Nothing
   const sg::_Lazy{Vector{TSG}}         # sparsity groups of the per-site pass
   const tg::_Lazy{Vector{TTG}}         # sparsity groups of the bond tensor / Nothing
   const lock::ReentrantLock            # serialises coefficient updates
end

function ETFastModel(fs::ETFastSite, c::AbstractVector{SVector{NR, Float64}}) where {NR}
   @assert length(c) == length(fs) "basis length $(length(fs)) ≠ #coeffs $(length(c))"
   M = fs.NC * NR; MP = _pattern_width(fs.NC) * NR
   isb = _is_bond_basis(fs)
   TFB = isb ? _bond_factorisation_type(fs) : Nothing
   TBF = isb ? ETBasisFactorisation{fs.NC} : Nothing
   TSG = _sgroup_type(fs)
   TTG = isb ? _tgroup_type(fs) : Nothing
   return ETFastModel{M, NR, MP, typeof(fs), TFB, TBF, TSG, TTG}(
            fs, _FusedWeights{M, NR, MP}(c), _Lazy{TFB}(), _Lazy{TBF}(),
            _Lazy{Vector{TSG}}(), _Lazy{Vector{TTG}}(), ReentrantLock())
end
ETFastModel(basis::ETFrictionSiteBasis, c) = ETFastModel(ETFastSite(basis), c)

n_rep(::ETFastModel{M, NR}) where {M, NR} = NR
block_type(fm::ETFastModel) = block_type(fm.fs)

"dense fused weight columns of the current coefficients (not cached; for inspection)"
_weights(fm::ETFastModel) = contraction_columns(fm.fs, (@atomic :acquire fm.weights).c)

_sigma_groups(fm::ETFastModel) = _getlazy!(() -> _build_sigma_groups(fm.fs), fm.sg)
_tensor_groups(fm::ETFastModel) = _getlazy!(() -> _build_tensor_groups(fm.fs, _bond_fact(fm)), fm.tg)

"group-compressed weights of the per-site pass (current coefficients; built on first use)"
@inline function _sigma_weights(fm::ETFastModel{M, NR, MP}) where {M, NR, MP}
   w = @atomic :acquire fm.weights
   return _getlazy!(w.S) do
      _GroupedW{M, MP}(contraction_columns(fm.fs, w.c), _sigma_groups(fm), fm.fs.aabasis.hasconst, fm.fs.NC)
   end
end

"group-compressed weights of the bond tensor (current coefficients; built on first use)"
@inline function _tensor_weights(fm::ETFastModel{M, NR, MP}) where {M, NR, MP}
   w = @atomic :acquire fm.weights
   return _getlazy!(w.T) do
      _GroupedW{M, MP}(contraction_columns(fm.fs, w.c), _tensor_groups(fm), false, fm.fs.NC)
   end
end

"""
    refresh!(fm::ETFastModel, c)

Replace the coefficient snapshot if `c` differs from it (the fused weights are then
rebuilt on next use). Read-only when `c` is unchanged. Changing coefficients while
the model is being evaluated on other threads is not supported (those evaluations
may use either the old or the new coefficients).
"""
function refresh!(fm::ETFastModel{M, NR, MP}, c::AbstractVector) where {M, NR, MP}
   (@atomic :acquire fm.weights).c == c && return fm
   lock(fm.lock) do
      (@atomic :acquire fm.weights).c == c && return
      @atomic :release fm.weights = _FusedWeights{M, NR, MP}(c)
   end
   return fm
end

"contracted Σ (flat, all replicas) from the pooled A of an environment"
@inline _sigma_flat(fm::ETFastModel{M, NR}, A) where {M, NR} =
      _grouped_sigma(_sigma_groups(fm), _sigma_weights(fm), A, Val(M ÷ NR))

"contracted Σ blocks `SVector{NR, block}` from the pooled A of an environment"
@inline sigma_from_A(fm::ETFastModel{M, NR}, A) where {M, NR} =
      _blocks_from(block_type(fm), _sigma_flat(fm, A), Val(NR))

"a fresh neighbour workspace for evaluating `fm` (see [`evaluate!`](@ref))"
ETSiteData(fm::ETFastModel) = ETSiteData(fm.fs)

"""
    evaluate(fm::ETFastModel, Rs, Zs) -> SVector{NR, block}
    evaluate!(sd::ETSiteData, fm::ETFastModel, Rs, Zs) -> SVector{NR, block}

Contracted output `Σ[r] = Σₖ c[k][r]·B[k]` on the environment `(Rs, Zs)` (raw
vectors for onsite bases). Same result as `evaluate(::ETOnsiteModel, Rs, Zs)`.
`evaluate` uses a fresh workspace; `evaluate!` reuses the caller-owned `sd`
(one per thread / assembly call).
"""
evaluate(fm::ETFastModel, Rs::AbstractVector{<:SVector{3}}, Zs::AbstractVector) =
      evaluate!(ETSiteData(fm), fm, Rs, Zs)

evaluate!(sd::ETSiteData, fm::ETFastModel, Rs::AbstractVector{<:SVector{3}}, Zs::AbstractVector) =
      sigma_from_A(fm, site_data!(sd, fm.fs, Rs, Zs).A)

"""
    evaluate_bond(fm::ETFastModel, rrij, Rs_env, Zs_env) -> SVector{NR, block}
    evaluate_bond!(sd::ETSiteData, fm::ETFastModel, rrij, Rs_env, Zs_env)

Contracted bond output for an explicitly transformed bond environment (e.g. the
ellipsoid case): prepends the bond particle (species `BOND_Z`). The `!` variant
reuses the caller-owned workspace `sd`.
"""
evaluate_bond(fm::ETFastModel, rrij::SVector{3}, Rs_env::AbstractVector{<:SVector{3}},
              Zs_env::AbstractVector) =
      evaluate_bond!(ETSiteData(fm), fm, rrij, Rs_env, Zs_env)

evaluate_bond!(sd::ETSiteData, fm::ETFastModel, rrij::SVector{3},
               Rs_env::AbstractVector{<:SVector{3}}, Zs_env::AbstractVector) =
      evaluate!(sd, fm, vcat([rrij], Rs_env), vcat([BOND_Z], Zs_env))

# ----------------------------------------------------------------------
# un-contracted basis from the pooled A (fitting path; same ordering as
# `evaluate(basis, Rs, Zs)`)

"""
    basis_from_A(fs, A) -> Vector{block}

All basis functions as Cartesian blocks from the pooled A of an environment.
"""
function basis_from_A(fs::ETFastSite, A::AbstractVector)
   AA = zeros(eltype(A), fs.nAA)
   ET.evaluate!(AA, fs.aabasis, A)
   BB = fs.basis.tensor.A2Bmaps .* Ref(AA)
   blocks = Vector{block_type(fs)}(undef, length(fs))
   return assemble_blocks!(blocks, fs.basis.out, BB)
end

# ----------------------------------------------------------------------
# bond bases: per-centre evaluation for atom-centred (spherical / snowman) cutoffs

const _IZ_BOND = 1     # bond channel is the first species block (see `bond_basis`)

_is_bond_basis(fs::ETFastSite) = fs.basis.rbasis.zlist[_IZ_BOND] == BOND_Z

"number of bond one-particle functions of a bond basis"
_nbond1p(fs::ETFastSite) = length(fs.apool[_IZ_BOND])

"""
    ETBondFactorisation(fs)

Split of a bond basis' AA spec into (bond 1p factor) × (env product):
`AA[q] = φ_{b(q)} · AAenv[e(q)]`. `bond1p[b] = (n, iy)` are the bond one-particle
functions; `tspecs[K+1]` lists `(q, b, env A-indices)` for env order `K`, sorted
by `b`.
"""
struct ETBondFactorisation{TS}
   bond1p::Vector{NTuple{2, Int}}
   bondA::Vector{Int}                       # A index of each bond 1p function
   tspecs::TS                               # NTuple{ORD, Vector{Tuple{Int, Int, NTuple{K,Int}}}}
end

_tspecs_empty(ORD) = ntuple(K -> Tuple{Int, Int, NTuple{K - 1, Int}}[], ORD)

"the concrete `ETBondFactorisation` type of a bond basis (without building it)"
_bond_factorisation_type(fs::ETFastSite) =
      typeof(ETBondFactorisation(NTuple{2, Int}[], Int[], _tspecs_empty(length(fs.aabasis.specs))))

function ETBondFactorisation(fs::ETFastSite)
   @assert _is_bond_basis(fs) "not a bond basis"
   bondA = Int[]; bond1p = NTuple{2, Int}[]; bidx = Dict{Int, Int}()
   for (iA, n, iy) in fs.apool[_IZ_BOND]
      push!(bondA, iA); push!(bond1p, (n, iy)); bidx[iA] = length(bond1p)
   end
   aab = fs.aabasis
   ORD = length(aab.specs)
   @assert !aab.hasconst "bond basis cannot contain the constant function"
   tspecs = _tspecs_empty(ORD)
   for N in 1:ORD, (i, ϕ) in enumerate(aab.specs[N])
      q = first(aab.ranges[N]) + i - 1
      ib = findall(k -> haskey(bidx, k), ϕ)
      length(ib) == 1 || error("bond AA function $q must contain exactly one bond factor")
      env = ntuple(t -> ϕ[t < ib[1] ? t : t + 1], N - 1)
      push!(tspecs[N], (q, bidx[ϕ[ib[1]]], env))
   end
   # group by bond factor so the per-centre tensor build accumulates each column in
   # a register (see `_grouped_tensor_order!`)
   for ts in tspecs; sort!(ts, by = t -> t[2]); end
   return ETBondFactorisation(bond1p, bondA, tspecs)
end

"bond factorisation of a bond model (built on first use)"
_bond_fact(fm::ETFastModel) = _getlazy!(() -> ETBondFactorisation(fm.fs), fm.fb)

"""
    ETBasisFactorisation(fs, fb)

Fitting-path counterpart of the per-centre tensor for bond bases with the partner
in the environment. Every basis function `k` contains exactly one bond factor, so

    B_k(ij) = Σ_{s ∈ slots(k)} φ_{b(s)}(r_ij) · P_i[s],    P_i[s] = Σ_q C[s, q] · AAenv_i[q],

where the slots of `k` are the bond one-particle functions occurring in `k` (the m
values of its bond factor) and `C` holds the Cartesian coupling coefficients
(symmetrisation matrix composed with the spherical->Cartesian map). `P_i` costs one
pass over the coupling entries per centre; each bond then costs a few small
products per basis function instead of a full product-basis evaluation.
"""
struct ETBasisFactorisation{NC}
   slot_b::Vector{Int}                      # slot -> bond 1p index
   kslots::Vector{UnitRange{Int}}           # basis function -> its slots
   ent_slot::Vector{Int}                    # coupling entries, sorted by slot
   ent_q::Vector{Int}
   ent_c::Vector{SVector{NC, Float64}}
end

function ETBasisFactorisation(fs::ETFastSite, fb::ETBondFactorisation)
   NC = fs.NC
   qb = zeros(Int, fs.nAA)
   for ts in fb.tspecs, (q, b, _) in ts
      qb[q] = b
   end
   ents = Tuple{Int, Int, Int, SVector{NC, Float64}}[]        # (k, b, q, coefficient)
   k0 = 0
   for (il, A2B) in enumerate(fs.basis.tensor.A2Bmaps)
      Ts = fs.basis.out.Tcart[il]
      rows, cols, vals = findnz(A2B)
      for t in eachindex(rows)
         push!(ents, (k0 + rows[t], qb[cols[t]], cols[t], SVector{NC, Float64}(Tuple(Ts * _svec(vals[t])))))
      end
      k0 += size(A2B, 1)
   end
   sort!(ents, by = e -> (e[1], e[2]))
   slot_b = Int[]; kslots = Vector{UnitRange{Int}}(undef, k0); ent_slot = Int[]
   kstart = ones(Int, k0); kend = zeros(Int, k0)
   prev = (0, 0)
   for e in ents
      if (e[1], e[2]) != prev
         push!(slot_b, e[2]); prev = (e[1], e[2])
         s = length(slot_b)
         kend[e[1]] == 0 && (kstart[e[1]] = s)
         kend[e[1]] = s
      end
      push!(ent_slot, length(slot_b))
   end
   for k in 1:k0
      kslots[k] = kstart[k]:kend[k]
   end
   return ETBasisFactorisation{NC}(slot_b, kslots, ent_slot, [e[3] for e in ents], [e[4] for e in ents])
end

"basis factorisation of a bond model (built on first use)"
_basis_fact(fm::ETFastModel) = _getlazy!(() -> ETBasisFactorisation(fm.fs, _bond_fact(fm)), fm.bf)

"""
    ETBondCentre{M, NC}

Per-centre state of a bond model: the neighbour workspace `sd` (at `Rs/rcut`), a
scratch A vector for the exact per-bond path, the factorised per-centre tensor `T`
(`partner_in_env`, contracted mode), the per-centre basis coefficients `P` with the
env products `v` and bond features `φ` (`partner_in_env`, basis mode), the `mode`
the state was built for, and a `tag` the assembly loops use to mark which centre
the state belongs to.
"""
mutable struct ETBondCentre{M, NC}
   const sd::ETSiteData
   const Abuf::Vector{Float64}
   const T::Vector{SVector{M, Float64}}
   const P::Vector{SVector{NC, Float64}}
   const v::Vector{Float64}
   const φ::Vector{Float64}
   mode::Symbol
   tag::Int
end

function ETBondCentre(fm::ETFastModel{M}) where {M}
   nT = _is_bond_basis(fm.fs) ? _nbond1p(fm.fs) : 0
   NC = fm.fs.NC
   return ETBondCentre{M, NC}(ETSiteData(fm.fs), zeros(fm.fs.nA), zeros(SVector{M, Float64}, nT),
                              SVector{NC, Float64}[], Float64[], zeros(nT), :none, 0)
end

"concrete per-centre state type of a bond model (for typed caches in the assembly loops)"
centre_type(fm::ETFastModel{M}) where {M} = ETBondCentre{M, fm.fs.NC}

"""
    bond_centre!(ctr, fm, Rs, Zs, rcut; partner_in_env, mode = :sigma) -> ctr
    bond_centre(fm, Rs, Zs, rcut; partner_in_env, mode = :sigma)  -> new ctr

Prepare the per-centre state of the bond model `fm` for a centre with neighbour
vectors `Rs` (raw) and species `Zs`. Always fills the shared neighbour data. With
`partner_in_env = true` it also builds the factorised per-centre data: the tensor
`T_i` for contracted blocks (`mode = :sigma`, see [`bond_sigma`](@ref)) or the basis
coefficients `P_i` for the fitting path (`mode = :basis`, see
[`bond_basis_blocks`](@ref)); the basis mode never touches the fused weights.
"""
function bond_centre!(ctr::ETBondCentre{M}, fm::ETFastModel{M}, Rs::AbstractVector{<:SVector{3}},
                      Zs::AbstractVector, rcut::Real; partner_in_env::Bool,
                      mode::Symbol = :sigma) where {M}
   sd = site_data!(ctr.sd, fm.fs, Rs, Zs; scale = 1 / rcut)
   if partner_in_env
      if mode === :sigma
         fill!(ctr.T, zero(SVector{M, Float64}))
         _grouped_tensor!(ctr.T, _tensor_groups(fm), _tensor_weights(fm), sd.A, Val(M ÷ n_rep(fm)))
      elseif mode === :basis
         _centre_basis!(ctr, _basis_fact(fm), _bond_fact(fm), sd.A)
      else
         error("unknown bond-centre mode :$mode")
      end
   end
   ctr.mode = mode
   return ctr
end

bond_centre(fm::ETFastModel, Rs::AbstractVector{<:SVector{3}}, Zs::AbstractVector, rcut::Real;
            partner_in_env::Bool, mode::Symbol = :sigma) =
      bond_centre!(ETBondCentre(fm), fm, Rs, Zs, rcut; partner_in_env = partner_in_env, mode = mode)

# env product values v[q] = Π A[env(q)] for every AA function q
@generated function _env_products!(v, tspecs::NTuple{ORD, Any}, A) where {ORD}
   quote
      Base.Cartesian.@nexprs $ORD N -> (@inbounds for (q, _, env) in tspecs[N]; v[q] = _prodA(A, env); end)
      return v
   end
end

# per-centre basis coefficients P[s] = Σ_q C[s, q] · v[q]
function _centre_basis!(ctr::ETBondCentre{M, NC}, bf::ETBasisFactorisation{NC},
                        fb::ETBondFactorisation, A) where {M, NC}
   nq = sum(length, fb.tspecs)                     # = number of AA functions
   length(ctr.v) < nq && resize!(ctr.v, nq)
   length(ctr.P) != length(bf.slot_b) && resize!(ctr.P, length(bf.slot_b))
   _env_products!(ctr.v, fb.tspecs, A)
   P = ctr.P; v = ctr.v
   fill!(P, zero(SVector{NC, Float64}))
   @inbounds for t in eachindex(bf.ent_q)
      s = bf.ent_slot[t]
      P[s] = P[s] + v[bf.ent_q[t]] * bf.ent_c[t]
   end
   return ctr
end

"""
    bond_sigma(fm, ctr, j_loc; partner_in_env) -> SVector{NR, block}

Contracted block of the bond (centre -> neighbour `j_loc`) from the per-centre
state (built with `mode = :sigma`). Factorised path when `partner_in_env`, exact
per-bond path otherwise.
"""
function bond_sigma(fm::ETFastModel{M, NR}, ctr::ETBondCentre{M}, j_loc::Int;
                    partner_in_env::Bool) where {M, NR}
   if partner_in_env
      ctr.mode === :sigma || error("bond centre state was not built for contracted evaluation")
      v = _bond_sigma_factorised(_bond_fact(fm), ctr, j_loc)
   else
      v = _sigma_flat(fm, bond_A!(fm.fs, ctr, j_loc, false))
   end
   return _blocks_from(block_type(fm), v, Val(NR))
end

# Σ_ij = Σ_b φ_b(r_ij) T[b]
function _bond_sigma_factorised(fb::ETBondFactorisation, ctr::ETBondCentre{M}, j::Int) where {M}
   RN, Y = ctr.sd.RN, ctr.sd.Y
   acc = zero(SVector{M, Float64})
   @inbounds for (b, (n, iy)) in enumerate(fb.bond1p)
      acc = acc + (RN[j, n] * Y[j, iy]) * ctr.T[b]
   end
   return acc
end

"""
    bond_A!(fs, ctr, j_loc, partner_in_env) -> A

The A vector of the bond (centre -> `j_loc`): the centre's pooled env A (minus the
partner's own contribution unless `partner_in_env`) with the bond channel set to
the partner's one-particle features. Writes into the centre's scratch buffer.
"""
function bond_A!(fs::ETFastSite, ctr::ETBondCentre, j::Int, partner_in_env::Bool)
   sd = ctr.sd; A = ctr.Abuf
   copyto!(A, sd.A)
   RN, Y = sd.RN, sd.Y
   if !partner_in_env
      @inbounds for (iA, n, iy) in fs.apool[sd.izs[j]]
         A[iA] -= RN[j, n] * Y[j, iy]
      end
   end
   @inbounds for (iA, n, iy) in fs.apool[_IZ_BOND]
      A[iA] = RN[j, n] * Y[j, iy]
   end
   return A
end

"""
    bond_basis_blocks(fm::ETFastModel, ctr, j_loc; partner_in_env) -> Vector{block}
    bond_basis_blocks(fs::ETFastSite, ctr, j_loc; partner_in_env) -> Vector{block}

Un-contracted bond basis of the bond (centre -> `j_loc`) from the per-centre state
(fitting path); same ordering as `evaluate_bond(basis, ...)`. With a fast model and
`partner_in_env` the factorised path is used (the state must be built with
`mode = :basis`); otherwise one product-basis evaluation per bond.
"""
function bond_basis_blocks(fm::ETFastModel, ctr::ETBondCentre, j_loc::Int; partner_in_env::Bool)
   partner_in_env || return bond_basis_blocks(fm.fs, ctr, j_loc; partner_in_env = false)
   ctr.mode === :basis || error("bond centre state was not built for the basis (fitting) path")
   return _bond_basis_factorised(_basis_fact(fm), _bond_fact(fm), ctr, j_loc, block_type(fm))
end

bond_basis_blocks(fs::ETFastSite, ctr::ETBondCentre, j_loc::Int; partner_in_env::Bool) =
      basis_from_A(fs, bond_A!(fs, ctr, j_loc, partner_in_env))

# B_k = Σ_{s ∈ slots(k)} φ_{b(s)} P[s]
function _bond_basis_factorised(bf::ETBasisFactorisation{NC}, fb::ETBondFactorisation,
                                ctr::ETBondCentre{M, NC}, j::Int, ::Type{BT}) where {NC, M, BT}
   RN, Y = ctr.sd.RN, ctr.sd.Y
   φ = ctr.φ
   @inbounds for (b, (n, iy)) in enumerate(fb.bond1p)
      φ[b] = RN[j, n] * Y[j, iy]
   end
   blocks = Vector{BT}(undef, length(bf.kslots))
   P = ctr.P
   @inbounds for (k, sl) in enumerate(bf.kslots)
      acc = zero(SVector{NC, Float64})
      for s in sl
         acc = acc + φ[bf.slot_b[s]] * P[s]
      end
      blocks[k] = BT(Tuple(acc))
   end
   return blocks
end
