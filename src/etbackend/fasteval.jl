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
# never stored: per AA function one static product and one SVector{NC·NR} fma.
# Per site the cost is radial + Ylm + A (as for an energy) plus the fused pass.
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
#
# All per-site buffers (radial rows, Ylm, pooled A, ...) live in reusable
# workspaces (`ETSiteData`) so the assembly loops do not allocate per centre. The
# workspaces are NOT thread-safe: a multithreaded assembly needs one `ETFastModel`
# (or at least one workspace / centre state) per thread.

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
neighbour workspace.
"""
struct ETFastSite{TB, TAA}
   basis::TB
   nA::Int
   nAA::Int
   apool::Vector{Vector{NTuple{3, Int}}}   # per species iz: (iA, n, iy)
   aabasis::TAA
   NC::Int
   ws::ETSiteData
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
   ws = ETSiteData(nA, rb.nR, _nY(basis), _nz(rb))
   return ETFastSite(basis, nA, length(T.aabasis), apool, T.aabasis,
                     length(block_type(basis)), ws)
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
# contracted site model

"""
    ETFastModel(fs::ETFastSite, c)
    ETFastModel(basis::ETFrictionSiteBasis, c)

Coefficient-contracted evaluator of a site basis: caches the fused weights
`W = contraction_columns(fs, c)` together with a snapshot of `c`; [`refresh!`](@ref)
rebuilds `W` iff `c` changed. For bond bases the factorisation data is built too.
"""
mutable struct ETFastModel{M, NR, TFS, TFB}
   const fs::TFS
   const c::Vector{SVector{NR, Float64}}
   W::Vector{SVector{M, Float64}}
   const fb::TFB                       # ETBondFactorisation or nothing
end

function ETFastModel(fs::ETFastSite, c::AbstractVector{SVector{NR, Float64}}) where {NR}
   @assert length(c) == length(fs) "basis length $(length(fs)) ≠ #coeffs $(length(c))"
   W = contraction_columns(fs, c)
   fb = _is_bond_basis(fs) ? ETBondFactorisation(fs) : nothing
   return ETFastModel{fs.NC * NR, NR, typeof(fs), typeof(fb)}(fs, copy(c), W, fb)
end
ETFastModel(basis::ETFrictionSiteBasis, c) = ETFastModel(ETFastSite(basis), c)

n_rep(::ETFastModel{M, NR}) where {M, NR} = NR
block_type(fm::ETFastModel) = block_type(fm.fs)

"""
    refresh!(fm::ETFastModel, c)

Rebuild the fused weights if the coefficients `c` differ from the cached snapshot.
"""
function refresh!(fm::ETFastModel, c::AbstractVector)
   if fm.c != c
      copyto!(fm.c, c)
      fm.W = contraction_columns(fm.fs, c)
   end
   return fm
end

"contracted Σ blocks `SVector{NR, block}` from the pooled A of an environment"
@inline sigma_from_A(fm::ETFastModel{M, NR}, A) where {M, NR} =
      _blocks_from(block_type(fm), fused_contract(fm.fs, fm.W, A), Val(NR))

"""
    evaluate(fm::ETFastModel, Rs, Zs) -> SVector{NR, block}

Contracted output `Σ[r] = Σₖ c[k][r]·B[k]` on the environment `(Rs, Zs)` (raw
vectors for onsite bases). Same result as `evaluate(::ETOnsiteModel, Rs, Zs)`.
Uses the fast site's private workspace.
"""
evaluate(fm::ETFastModel, Rs::AbstractVector{<:SVector{3}}, Zs::AbstractVector) =
      sigma_from_A(fm, site_data!(fm.fs.ws, fm.fs, Rs, Zs).A)

"""
    evaluate_bond(fm::ETFastModel, rrij, Rs_env, Zs_env) -> SVector{NR, block}

Contracted bond output for an explicitly transformed bond environment (e.g. the
ellipsoid case): prepends the bond particle (species `BOND_Z`).
"""
evaluate_bond(fm::ETFastModel, rrij::SVector{3}, Rs_env::AbstractVector{<:SVector{3}},
              Zs_env::AbstractVector) =
      evaluate(fm, vcat([rrij], Rs_env), vcat([BOND_Z], Zs_env))

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

function ETBondFactorisation(fs::ETFastSite)
   @assert _is_bond_basis(fs) "not a bond basis"
   bondA = Int[]; bond1p = NTuple{2, Int}[]; bidx = Dict{Int, Int}()
   for (iA, n, iy) in fs.apool[_IZ_BOND]
      push!(bondA, iA); push!(bond1p, (n, iy)); bidx[iA] = length(bond1p)
   end
   aab = fs.aabasis
   ORD = length(aab.specs)
   @assert !aab.hasconst "bond basis cannot contain the constant function"
   tspecs = ntuple(K -> Tuple{Int, Int, NTuple{K - 1, Int}}[], ORD)
   for N in 1:ORD, (i, ϕ) in enumerate(aab.specs[N])
      q = first(aab.ranges[N]) + i - 1
      ib = findall(k -> haskey(bidx, k), ϕ)
      length(ib) == 1 || error("bond AA function $q must contain exactly one bond factor")
      env = ntuple(t -> ϕ[t < ib[1] ? t : t + 1], N - 1)
      push!(tspecs[N], (q, bidx[ϕ[ib[1]]], env))
   end
   # group by bond factor so the per-centre tensor build accumulates each column in
   # a register (see `_centre_tensor_order!`)
   for ts in tspecs; sort!(ts, by = t -> t[2]); end
   return ETBondFactorisation(bond1p, bondA, tspecs)
end

"""
    ETBondCentre{M}

Per-centre state of a bond model: the neighbour workspace `sd` (at `Rs/rcut`), a
scratch A vector for the exact per-bond path, the factorised per-centre tensor `T`
(columns indexed by bond 1p function; filled only when `partner_in_env`), and a
`tag` the assembly loops use to mark which centre the state belongs to.
"""
mutable struct ETBondCentre{M}
   const sd::ETSiteData
   const Abuf::Vector{Float64}
   const T::Vector{SVector{M, Float64}}
   tag::Int
end

function ETBondCentre(fm::ETFastModel{M}) where {M}
   nT = fm.fb === nothing ? 0 : length(fm.fb.bond1p)
   return ETBondCentre{M}(ETSiteData(fm.fs), zeros(fm.fs.nA), zeros(SVector{M, Float64}, nT), 0)
end

"concrete per-centre state type of a bond model (for typed caches in the assembly loops)"
centre_type(::ETFastModel{M}) where {M} = ETBondCentre{M}

"""
    bond_centre!(ctr, fm, Rs, Zs, rcut; partner_in_env) -> ctr
    bond_centre(fm, Rs, Zs, rcut; partner_in_env)  -> new ctr

Prepare the per-centre state of the bond model `fm` for a centre with neighbour
vectors `Rs` (raw) and species `Zs`. With `partner_in_env = true` the factorised
tensor `T_i` is built (one fused pass); otherwise only the shared neighbour data.
"""
function bond_centre!(ctr::ETBondCentre{M}, fm::ETFastModel{M}, Rs::AbstractVector{<:SVector{3}},
                      Zs::AbstractVector, rcut::Real; partner_in_env::Bool) where {M}
   sd = site_data!(ctr.sd, fm.fs, Rs, Zs; scale = 1 / rcut)
   if partner_in_env
      fill!(ctr.T, zero(SVector{M, Float64}))
      _centre_tensor!(ctr.T, fm.W, fm.fb.tspecs, sd.A)
   end
   return ctr
end

bond_centre(fm::ETFastModel, Rs::AbstractVector{<:SVector{3}}, Zs::AbstractVector, rcut::Real;
            partner_in_env::Bool) =
      bond_centre!(ETBondCentre(fm), fm, Rs, Zs, rcut; partner_in_env = partner_in_env)

@generated function _centre_tensor!(T, Wcols, tspecs::NTuple{ORD, Any}, A) where {ORD}
   quote
      Base.Cartesian.@nexprs $ORD N -> _centre_tensor_order!(T, Wcols, tspecs[N], A)
      return T
   end
end

# entries are sorted by bond factor `b`: accumulate each run in a register and
# flush once per run (avoids a store->load dependency chain through T[b])
function _centre_tensor_order!(T::Vector{SVector{M, Float64}}, Wcols,
                               tspec::Vector{Tuple{Int, Int, NTuple{K, Int}}}, A) where {M, K}
   isempty(tspec) && return T
   bcur = tspec[1][2]; acc = zero(SVector{M, Float64})
   @inbounds for (q, b, env) in tspec
      if b != bcur
         T[bcur] = T[bcur] + acc
         bcur = b; acc = zero(SVector{M, Float64})
      end
      acc = acc + _prodA(A, env) * Wcols[q]
   end
   @inbounds T[bcur] = T[bcur] + acc
   return T
end

"""
    bond_sigma(fm, ctr, j_loc; partner_in_env) -> SVector{NR, block}

Contracted block of the bond (centre -> neighbour `j_loc`) from the per-centre
state. Factorised path when `partner_in_env`, exact per-bond path otherwise.
"""
function bond_sigma(fm::ETFastModel{M, NR}, ctr::ETBondCentre{M}, j_loc::Int;
                    partner_in_env::Bool) where {M, NR}
   v = partner_in_env ? _bond_sigma_factorised(fm.fb, ctr, j_loc) :
                        fused_contract(fm.fs, fm.W, bond_A!(fm.fs, ctr, j_loc, false))
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
    bond_basis_blocks(fs, ctr, j_loc; partner_in_env) -> Vector{block}

Un-contracted bond basis of the bond (centre -> `j_loc`) from the per-centre state
(fitting path); same ordering as `evaluate_bond(basis, ...)`.
"""
bond_basis_blocks(fs::ETFastSite, ctr::ETBondCentre, j_loc::Int; partner_in_env::Bool) =
      basis_from_A(fs, bond_A!(fs, ctr, j_loc, partner_in_env))
