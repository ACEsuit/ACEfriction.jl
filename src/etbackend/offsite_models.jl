# Flattened ET offsite (bond) model + system-level pairwise-coupled (PWC) friction
# model. Mirrors PWCMatrixModel (matrixmodels/pwcmatrixmodels.jl) on the ET bond
# basis: per directed bond (i,j) an offsite Σ block; Γ = Σ Σᵀ.
#
# Two bond-environment geometries are supported (cf. the two `matrix!` methods of
# PWCMatrixModel):
#   * EllipsoidCutoff — bond-centred ellipsoid env; iterate `et_bonds`.
#   * SphericalCutoff  — atom-centred spherical env (bond = a marked neighbour j of
#                        atom i); iterate sites.

using StaticArrays, LinearAlgebra
using AtomsBase: AbstractSystem, atomic_number

"""
    ETOffsiteModel(basis, c, cutoff)
    ETOffsiteModel(basis, n_rep, cutoff)

Flattened offsite site model: a bond `ETFrictionSiteBasis` + coefficients `c` +
the cutoff (`EllipsoidCutoff` or `SphericalCutoff`) defining its bond environment.
"""
mutable struct ETOffsiteModel{NR, P, TB, CUT}
   basis::TB
   c::Vector{SVector{NR, Float64}}
   cutoff::CUT
end

function ETOffsiteModel(basis::ETFrictionSiteBasis{P}, c::Vector{SVector{NR,Float64}},
                        cutoff::CUT) where {P, NR, CUT}
   @assert length(basis) == length(c)
   return ETOffsiteModel{NR, P, typeof(basis), CUT}(basis, c, cutoff)
end

ETOffsiteModel(basis::ETFrictionSiteBasis, n_rep::Integer, cutoff) =
      ETOffsiteModel(basis, rand(SVector{n_rep,Float64}, length(basis)), cutoff)

n_rep(::ETOffsiteModel{NR}) where {NR} = NR
Base.length(m::ETOffsiteModel) = length(m.basis)
nparams(m::ETOffsiteModel) = length(m.c)
params(m::ETOffsiteModel) = m.c
function set_params!(m::ETOffsiteModel{NR}, c::Vector{SVector{NR,Float64}}) where {NR}
   @assert length(c) == length(m.c); copyto!(m.c, c); return m
end

# contract bond basis blocks with the coefficients -> SVector{NR, block}
function _contract(m::ETOffsiteModel{NR}, B) where {NR}
   TB = block_type(m.basis)
   Σ = zero(MVector{NR, TB})
   @inbounds for k in eachindex(B), r in 1:NR
      Σ[r] += m.c[k][r] * B[k]
   end
   return SVector(Σ)
end

# --- ellipsoid: bond vector + ellipsoid env (dispatch on SVector 2nd arg) ---
function evaluate_basis(m::ETOffsiteModel, rrij::SVector{3}, Rs, Zs)
   rbond, Rst, Zst = ellipsoid_env_transform(rrij, Rs, Zs, m.cutoff)
   return evaluate_bond(m.basis, rbond, Rst, Zst)
end
evaluate(m::ETOffsiteModel, rrij::SVector{3}, Rs, Zs) =
      _contract(m, evaluate_basis(m, rrij, Rs, Zs))

# --- spherical: atom-i neighbourhood + bond-partner index (dispatch on Int) ---
function evaluate_basis(m::ETOffsiteModel, j_loc::Integer, Rs, Zs)
   rbond, Rse, Zse = spherical_bond_transform(Int(j_loc), Rs, Zs, m.cutoff)
   return evaluate_bond(m.basis, rbond, Rse, Zse)
end
evaluate(m::ETOffsiteModel, j_loc::Integer, Rs, Zs) =
      _contract(m, evaluate_basis(m, j_loc, Rs, Zs))

# ---------------------------------------------------------------------------

_msort2(z1, z2) = z1 <= z2 ? (z1, z2) : (z2, z1)

"""
    ETPWCModel(offsite::Dict{Tuple{Int,Int},ETOffsiteModel}; id=:offsite)

Pairwise-coupled offsite friction model: one offsite model per (sorted) species
pair. All models must share `n_rep` and `cutoff`.
"""
struct ETPWCModel{NR, TM, CUT}
   offsite::Dict{Tuple{Int,Int}, TM}
   cutoff::CUT
   id::Symbol
end

function ETPWCModel(offsite::Dict{Tuple{Int,Int}, TM}; id::Symbol = :offsite) where {TM}
   NRs = unique(n_rep(m) for m in values(offsite))
   @assert length(NRs) == 1 "offsite models must share n_rep"
   cut = first(values(offsite)).cutoff
   return ETPWCModel{NRs[1], TM, typeof(cut)}(offsite, cut, id)
end

n_rep(::ETPWCModel{NR}) where {NR} = NR

_zz_of(Z, i, j) = _msort2(Z[i], Z[j])

# --- Σ assembly: ellipsoid (bond iterator) ---
function sigma(M::ETPWCModel{NR, TM, <:EllipsoidCutoff}, at::AbstractSystem) where {NR, TM}
   Z = _species_vec(at); N = length(at)
   Z3 = SMatrix{3,3,Float64,9}
   Σ = [ fill(zero(Z3), N, N) for _ in 1:NR ]
   for (i, j, rrij, Js, Rs, Zs) in et_bonds(at, M.cutoff)
      zz = _zz_of(Z, i, j)
      haskey(M.offsite, zz) || continue
      Σij = evaluate(M.offsite[zz], rrij, Rs, Zs)
      for r in 1:NR; Σ[r][i, j] = Σij[r]; end
   end
   return Σ
end

# --- Σ assembly: spherical (site iterator; bond = marked neighbour j) ---
function sigma(M::ETPWCModel{NR, TM, <:SphericalCutoff}, at::AbstractSystem) where {NR, TM}
   Z = _species_vec(at); N = length(at)
   Z3 = SMatrix{3,3,Float64,9}
   Σ = [ fill(zero(Z3), N, N) for _ in 1:NR ]
   for (i, neigs, Rs) in _sites_iter(at, M.cutoff.rcut)
      isempty(neigs) && continue
      Zs = Z[neigs]
      for (j_loc, j) in enumerate(neigs)
         zz = _zz_of(Z, i, j)
         haskey(M.offsite, zz) || continue
         Σij = evaluate(M.offsite[zz], j_loc, Rs, Zs)
         for r in 1:NR; Σ[r][i, j] += Σij[r]; end
      end
   end
   return Σ
end

# per-species-pair basis-function index ranges, and total length
function _offsite_ranges(M::ETPWCModel)
   ranges = Dict{Tuple{Int,Int}, UnitRange{Int}}(); off = 0
   for (zz, m) in M.offsite
      ranges[zz] = (off + 1):(off + length(m.basis)); off += length(m.basis)
   end
   return ranges, off
end
basis_length(M::ETPWCModel) = _offsite_ranges(M)[2]

"""
    basis(M, at) -> Vector{Matrix{SMatrix{3,3}}}

Un-contracted offsite basis: `B[k][i,j]` is the `k`-th basis function on bond
(i,j). `sum_k c[k][r]·B[k] == Σ[r]`.
"""
function basis(M::ETPWCModel{NR, TM, <:EllipsoidCutoff}, at::AbstractSystem) where {NR, TM}
   Z = _species_vec(at); N = length(at); Z3 = SMatrix{3,3,Float64,9}
   ranges, Ktot = _offsite_ranges(M)
   B = [ fill(zero(Z3), N, N) for _ in 1:Ktot ]
   for (i, j, rrij, Js, Rs, Zs) in et_bonds(at, M.cutoff)
      zz = _zz_of(Z, i, j); haskey(M.offsite, zz) || continue
      Bij = evaluate_basis(M.offsite[zz], rrij, Rs, Zs)
      for (k, b) in zip(ranges[zz], Bij); B[k][i, j] = b; end
   end
   return B
end

function basis(M::ETPWCModel{NR, TM, <:SphericalCutoff}, at::AbstractSystem) where {NR, TM}
   Z = _species_vec(at); N = length(at); Z3 = SMatrix{3,3,Float64,9}
   ranges, Ktot = _offsite_ranges(M)
   B = [ fill(zero(Z3), N, N) for _ in 1:Ktot ]
   for (i, neigs, Rs) in _sites_iter(at, M.cutoff.rcut)
      isempty(neigs) && continue
      Zs = Z[neigs]
      for (j_loc, j) in enumerate(neigs)
         zz = _zz_of(Z, i, j); haskey(M.offsite, zz) || continue
         Bij = evaluate_basis(M.offsite[zz], j_loc, Rs, Zs)
         for (k, b) in zip(ranges[zz], Bij); B[k][i, j] += b; end
      end
   end
   return B
end

"""
    gamma(M, at) -> Matrix{SMatrix{3,3}}  (N×N block friction matrix Γ = Σ Σᵀ)
"""
function gamma(M::ETPWCModel, at::AbstractSystem)
   Σ = sigma(M, at)
   N = size(Σ[1], 1)
   Z3 = SMatrix{3,3,Float64,9}
   Γ = fill(zero(Z3), N, N)
   for Σr in Σ, i in 1:N, k in 1:N
      acc = zero(Z3)
      for j in 1:N
         acc += Σr[i, j] * Σr[k, j]'
      end
      Γ[i, k] += acc
   end
   return Γ
end

function gamma_dense(M::ETPWCModel, at::AbstractSystem)
   Γb = gamma(M, at); N = size(Γb, 1)
   Γ = zeros(3N, 3N)
   for i in 1:N, k in 1:N
      Γ[3i-2:3i, 3k-2:3k] .= Γb[i, k]
   end
   return Γ
end
