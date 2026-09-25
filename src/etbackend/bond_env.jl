# Bond (offsite) geometry for the ET backend: ellipsoid cutoff, the ellipsoid->
# sphere environment transform, and a bond iterator over an AtomsBase system.
#
# This is pure geometry (no ACE machinery), vendored/ported from
# ACEfrictionCore/src/bonds/{bondcutoffs,ellipsoid_trans,iterator}.jl so the ET
# backend carries no dependency on ACEfrictionCore.

using StaticArrays, LinearAlgebra
using AtomsBase: AbstractSystem, position, atomic_number
using NeighbourLists: PairList, neigs
using Unitful: ustrip, @u_str

"""
    EllipsoidCutoff(rcutbond, rcutenv, zcutenv)

Bond-centred ellipsoidal cutoff: bonds with `|rij| <= rcutbond`; environment atoms
within the ellipsoid `(z/zcutenv)^2 + (r/rcutenv)^2 <= 1` around the bond midpoint
(`z` along the bond, `r` perpendicular).
"""
struct EllipsoidCutoff{T}
   rcutbond::T
   rcutenv::T
   zcutenv::T
end

env_cutoff(ec::EllipsoidCutoff) =
      max(ec.rcutbond*0.5 + ec.zcutenv, sqrt(ec.rcutenv^2 + (0.5*ec.rcutbond)^2))
env_filter(r, z, ec::EllipsoidCutoff) = ((z/ec.zcutenv)^2 + (r/ec.rcutenv)^2 <= 1)

"""
    SphericalCutoff(rcut; partner_in_env = true)

Spherical pair-environment cutoff for the *atom-centred* offsite model: the bond
environment of a pair (i,j) is the set of neighbours of atom `i` within `rcut`,
with `j` itself the bond partner. (Cf. ACEfrictionCore `SphericalCutoff`.)

`partner_in_env` selects whether the bond partner `j` is *also* pooled into the
environment features of the bond (i,j):

- `true` (default): the environment is all of `N_i` (including `j`). The environment
  features are then shared by all bonds of the centre and, since every bond basis
  function contains exactly one bond factor, the blocks factorise as
  `Σ_ij = T_i · φ(r_ij)` with a per-centre tensor `T_i` and cheap per-bond
  one-particle features `φ`. This makes the pair blocks of a centre cost about as
  much as a single onsite (energy-like) evaluation.
- `false`: the environment is `N_i \\ {j}` (all other neighbours of `i`), the
  original convention. Every bond of a centre then has a different environment, so
  evaluating the pair blocks costs one many-body evaluation *per bond*.

The two settings are different bases (spanning the same function space), so a model
must be evaluated with the setting it was fitted with; the setting is serialized, and
models saved before the option existed load as `false`. On H/Cu reference data the two
fit equally well. They coincide exactly when the partner's species is excluded from the
environment factors, and for the antisymmetric `SnowManCutoff` at `maxorder = 2`.
"""
struct SphericalCutoff{T}
   rcut::T
   partner_in_env::Bool
end
SphericalCutoff(rcut::Real; partner_in_env::Bool = true) =
      SphericalCutoff(float(rcut), partner_in_env)
SphericalCutoff{T}(rcut::Real) where {T} = SphericalCutoff{T}(T(rcut), true)
env_cutoff(sc::SphericalCutoff) = sc.rcut

"""whether the bond partner is pooled into the bond environment (see `SphericalCutoff`)."""
partner_in_env(c::SphericalCutoff) = c.partner_in_env
partner_in_env(c::EllipsoidCutoff) = false

"""
    SnowManCutoff(rcut, symmetry = :general; partner_in_env = true)

Atom-centred pair-environment cutoff combining *both* bond ends: the diffusion block
of a pair `(i,j)` evaluates the ACE basis on the spherical environment of `i` (with
`j` the bond partner) and on the spherical environment of `j` (with `i` the bond
partner) — two overlapping spheres, one per bond end (the "snowman"). Writing
`B_ij = basis(sphere_i, bond i→j)`, the combination is selected by `symmetry`:

    :general        Σ_ij = c₊ · (B_ij + B_ji) + c₋ · (B_ij - B_ji)     (default)
    :symmetric      Σ_ij = c · (B_ij + B_ji)                             Σ_ji =  Σ_ij
    :antisymmetric  Σ_ij = c · (B_ij - B_ji)                             Σ_ji = -Σ_ij

With `:symmetric` / `:antisymmetric` both ends share the coefficients, so every
off-diagonal friction block `Γ_ij = Σ_ij Σ_jiᵀ = ±Σ_ij Σ_ijᵀ` is a symmetric 3×3 matrix.
`:general` has independent coefficients for the symmetric and the antisymmetric
combination (twice as many parameters: the basis is the stacked `[B_ij + B_ji; B_ij - B_ji]`,
coefficients `[c₊; c₋]`), so `Σ_ji ≠ ±Σ_ij` and `Γ_ij` is in general not symmetric (`Γ`
itself stays symmetric positive semi-definite). It contains the other two as the
restrictions `c₋ = 0` (`:symmetric`) and `c₊ = 0` (`:antisymmetric`).

`symmetry` is carried as a (Symbol-valued) type parameter `SnowManCutoff{T, S}` so the
assembly dispatches on it. `rcut` is the per-centre spherical radius (same convention as
[`SphericalCutoff`](@ref)). The keyword `partner_in_env` (default `true`) has the same
meaning as for [`SphericalCutoff`](@ref): with `true` the bond partner is pooled into each
sphere's environment, which lets the per-centre evaluation be shared across all bonds.

Models saved before the `symmetry` tag was serialized load as `:symmetric`.
"""
struct SnowManCutoff{T, S}
   rcut::T
   partner_in_env::Bool
   function SnowManCutoff(rcut::T, symmetry::Symbol = :general;
                          partner_in_env::Bool = true) where {T}
      @assert symmetry in (:general, :symmetric, :antisymmetric) "symmetry must be :general, :symmetric or :antisymmetric (got :$symmetry)."
      return new{T, symmetry}(rcut, partner_in_env)
   end
end
env_cutoff(sc::SnowManCutoff) = sc.rcut
partner_in_env(c::SnowManCutoff) = c.partner_in_env
"the symmetry tag (`:general` / `:symmetric` / `:antisymmetric`) carried in the type parameter."
symmetry(::SnowManCutoff{T, S}) where {T, S} = S

"""
    _snowman_combine(cutoff, a, b)

Combine the two bond-end contributions of a snowman pair according to the cutoff's
symmetry: `a + b` for `:symmetric`, `a - b` for `:antisymmetric`. Dispatches on the
Symbol-valued type parameter of [`SnowManCutoff`](@ref). (`:general` combines the two
ends with different coefficients; see `_snowman_basis_combine` / `_snowman_sigma_combine`.)
"""
_snowman_combine(::SnowManCutoff{T, :symmetric}, a, b) where {T} = a + b
_snowman_combine(::SnowManCutoff{T, :antisymmetric}, a, b) where {T} = a - b

"""
    _snowman_nbasis(cutoff, K)

Number of snowman basis functions (= coefficients) built from a bond basis of length
`K`: `2K` for `:general` (stacked symmetric / antisymmetric combinations), `K` otherwise.
"""
_snowman_nbasis(::SnowManCutoff{T, :general}, K::Integer) where {T} = 2K
_snowman_nbasis(::SnowManCutoff, K::Integer) = K

"""
    _snowman_basis_combine(cutoff, Bij, Bji)

Snowman basis of a pair from the (length-`K`) bond bases of its two ends,
`Bij = basis(sphere_i, bond i→j)` and `Bji = basis(sphere_j, bond j→i)`:
`[Bij + Bji; Bij - Bji]` (length `2K`) for `:general`, `Bij ± Bji` otherwise.
"""
_snowman_basis_combine(::SnowManCutoff{T, :general}, Bij, Bji) where {T} =
      vcat(Bij .+ Bji, Bij .- Bji)
_snowman_basis_combine(sc::SnowManCutoff, Bij, Bji) = map((a, b) -> _snowman_combine(sc, a, b), Bij, Bji)

"""
    _snowman_fast_coeffs(c::AbstractVector{SVector{NR}}) -> Vector{SVector{2NR}}

Coefficients of the fused fast evaluator of a `:general` snowman model. With
`c = [c₊; c₋]` (length `2K`), `Σ_ij = a·B_ij + b·B_ji` where `a = c₊ + c₋` and
`b = c₊ - c₋`; the fused coefficients are `[a_k; b_k]`, so one contraction of a directed
bond `i→j` yields both `a·B_ij` (first `NR` entries) and `b·B_ij` (last `NR`).
"""
function _snowman_fast_coeffs(c::AbstractVector{SVector{NR, T}}) where {NR, T}
   K, r = divrem(length(c), 2)
   @assert r == 0 "a :general snowman model has an even number of coefficients (got $(length(c)))."
   return [ vcat(c[k] + c[K + k], c[k] - c[K + k]) for k in 1:K ]
end

"""
    _snowman_sigma_combine(cutoff, Vij, Vji)

Diffusion block of a snowman pair from the contracted values of its two directed bonds
(`Vij` for bond `i→j` on sphere `i`, `Vji` for `j→i` on sphere `j`), per replica. For
`:general` the values are the fused `[a·B; b·B]` (see `_snowman_fast_coeffs`) and
`Σ_ij = a·B_ij + b·B_ji`.
"""
_snowman_sigma_combine(sc::SnowManCutoff, Vij, Vji) = _snowman_combine.(Ref(sc), Vij, Vji)
@inline function _snowman_sigma_combine(::SnowManCutoff{T, :general}, Vij::SVector{M2}, Vji::SVector{M2}) where {T, M2}
   NR = M2 ÷ 2
   return SVector(ntuple(r -> Vij[r] + Vji[NR + r], Val(NR)))
end

"""
    spherical_bond_transform(j_loc, Rs, Zs, sc) -> (r̂bond, Rs_env, Zs_env)

For the (atom-centred) spherical / snowman offsite models: bond direction
`Rs[j_loc]/rcut`, environment = the neighbours of the centre (each `/rcut`),
*excluding* the bond partner unless `partner_in_env(sc)` is `true`.
Mirrors ACEfrictionCore's `env_transform(j, Rs, Zs, ::SphericalCutoff)`.
"""
function spherical_bond_transform(j_loc::Int, Rs::AbstractVector{<:SVector{3}},
                                  Zs::AbstractVector, sc::Union{SphericalCutoff,SnowManCutoff})
   rbond = Rs[j_loc] / sc.rcut
   keep = partner_in_env(sc)
   Rs_env = SVector{3,Float64}[]; Zs_env = Int[]
   for l in eachindex(Rs)
      (l == j_loc && !keep) && continue
      push!(Rs_env, Rs[l] / sc.rcut); push!(Zs_env, Zs[l])
   end
   return rbond, Rs_env, Zs_env
end

# skewed Householder reflection mapping the ellipsoid to the unit sphere
function _skew_householder(rr0::SVector{3}, zc::T, rc::T) where {T<:Real}
   r02 = sum(abs2, rr0)
   r02 == 0 && return SMatrix{3,3,T}(I) / rc
   zc_inv, rc_inv = inv(zc), inv(rc)
   return SMatrix{3,3}(rc_inv * I + (zc_inv - rc_inv)/r02 * (rr0 * transpose(rr0)))
end

"""
    ellipsoid_env_transform(rrij, Rs, Zs, ec) -> (r̂bond, Rs_t, Zs)

Map a bond environment to the normalized coordinates the bond basis expects: the
(scaled) bond direction `rrij/rcutbond` and the env vectors mapped through the
ellipsoid->sphere reflection. Mirrors ACEfrictionCore's `ellipsoid2sphere`.
"""
function ellipsoid_env_transform(rrij::SVector{3}, Rs::AbstractVector{<:SVector{3}},
                                 Zs::AbstractVector, ec::EllipsoidCutoff)
   G = _skew_householder(rrij, ec.zcutenv, ec.rcutenv)
   rbond = rrij / ec.rcutbond
   Rst = [ G * r for r in Rs ]
   return rbond, Rst, Zs
end

# ---------------------------------------------------------------------------
# Bond iterator (single ellipsoid cutoff) yielding (i, j, rrij, Js, Rs, Zs).

# (fields concretely typed: an abstract `PairList` field made every neighbour
# access in `_bond_env` dynamically dispatched, ~15 μs per bond)
struct ETBondsIterator{TPL <: PairList}
   X::Vector{SVector{3, Float64}}
   Z::Vector{Int}
   N::Int
   nlist_bond::TPL
   nlist_env::TPL
   ec::EllipsoidCutoff{Float64}
end

function et_bonds(sys::AbstractSystem, ec::EllipsoidCutoff)
   N = length(sys)
   X = SVector{3,Float64}[ SVector{3,Float64}(ustrip.(u"Å", position(sys, i))) for i in 1:N ]
   Z = Int[ Int(atomic_number(sys, i)) for i in 1:N ]
   nlist_bond = PairList(sys, ec.rcutbond * u"Å")
   nlist_env  = PairList(sys, env_cutoff(ec) * u"Å")
   return ETBondsIterator(X, Z, N, nlist_bond, nlist_env, EllipsoidCutoff{Float64}(ec.rcutbond, ec.rcutenv, ec.zcutenv))
end

function _bond_env(iter::ETBondsIterator, i, j, rrij)
   Js_i, Rs_i = neigs(iter.nlist_env, i)
   rri = iter.X[i]
   rrmid = rri + 0.5 * rrij
   ŝ = rrij / norm(rrij)
   Js = Int[]; Rs = SVector{3,Float64}[]; Zs = Int[]
   # the bond partner's own entry (same atom index and same periodic image)
   q_bond = findfirst(q -> Js_i[q] == j && Rs_i[q] ≈ rrij, eachindex(Js_i))
   for (q, rrq) in enumerate(Rs_i)
      q == q_bond && continue
      rr = rrq + rri - rrmid
      z = dot(rr, ŝ); r = norm(rr - z * ŝ)
      if env_filter(r, z, iter.ec)
         push!(Js, Js_i[q]); push!(Rs, rr); push!(Zs, iter.Z[Js_i[q]])
      end
   end
   return Js, Rs, Zs
end

function Base.iterate(iter::ETBondsIterator, state=(1, 0))
   i, q = state
   Js, Rs = neigs(iter.nlist_bond, i)
   (i >= iter.N && q >= length(Js)) && return nothing
   if q < length(Js)
      q += 1
   elseif i < iter.N
      i += 1; Js, Rs = neigs(iter.nlist_bond, i); q = 1
      while isempty(Js) && i < iter.N
         i += 1; Js, Rs = neigs(iter.nlist_bond, i)
      end
      isempty(Js) && return nothing
   else
      return nothing
   end
   j = Js[q]; rrij = Rs[q]
   Js_e, Rs_e, Zs_e = _bond_env(iter, i, j, rrij)
   return (i, j, rrij, Js_e, Rs_e, Zs_e), (i, q)
end

Base.IteratorSize(::Type{<:ETBondsIterator}) = Base.SizeUnknown()
