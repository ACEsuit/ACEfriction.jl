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
    SphericalCutoff(rcut)

Spherical pair-environment cutoff for the *atom-centred* offsite model: the bond
environment of a pair (i,j) is the set of neighbours of atom `i` within `rcut`,
with `j` itself the bond partner. (Cf. ACEfrictionCore `SphericalCutoff`.)
"""
struct SphericalCutoff{T}
   rcut::T
end
env_cutoff(sc::SphericalCutoff) = sc.rcut

"""
    SnowManCutoff(rcut, symmetry = :symmetric)

Atom-centred pair-environment cutoff combining *both* bond ends: the diffusion block
of a pair `(i,j)` evaluates the ACE basis on the spherical environment of `i` (with
`j` the bond partner) and on the spherical environment of `j` (with `i` the bond
partner) — two overlapping spheres, one per bond end (the "snowman") — and combines
them with the *same* coefficients. The combination is selected by `symmetry`:

    :symmetric      Σ_ij = c · basis(sphere_i, bond i→j) + c · basis(sphere_j, bond j→i)
    :antisymmetric  Σ_ij = c · basis(sphere_i, bond i→j) - c · basis(sphere_j, bond j→i)

`symmetry` is carried as a (Symbol-valued) type parameter `SnowManCutoff{T, S}` so the
assembly dispatches on it. `rcut` is the per-centre spherical radius (same convention as
[`SphericalCutoff`](@ref)).
"""
struct SnowManCutoff{T, S}
   rcut::T
   function SnowManCutoff(rcut::T, symmetry::Symbol = :symmetric) where {T}
      @assert symmetry in (:symmetric, :antisymmetric) "symmetry must be :symmetric or :antisymmetric (got :$symmetry)."
      return new{T, symmetry}(rcut)
   end
end
env_cutoff(sc::SnowManCutoff) = sc.rcut
"the symmetry tag (`:symmetric` / `:antisymmetric`) carried in the type parameter."
symmetry(::SnowManCutoff{T, S}) where {T, S} = S

"""
    _snowman_combine(cutoff, a, b)

Combine the two bond-end contributions of a snowman pair according to the cutoff's
symmetry: `a + b` for `:symmetric`, `a - b` for `:antisymmetric`. Dispatches on the
Symbol-valued type parameter of [`SnowManCutoff`](@ref).
"""
_snowman_combine(::SnowManCutoff{T, :symmetric}, a, b) where {T} = a + b
_snowman_combine(::SnowManCutoff{T, :antisymmetric}, a, b) where {T} = a - b

"""
    spherical_bond_transform(j_loc, Rs, Zs, sc) -> (r̂bond, Rs_env, Zs_env)

For the (atom-centred) spherical / snowman offsite models: bond direction
`Rs[j_loc]/rcut`, environment = the *other* neighbours of the centre (each `/rcut`).
Mirrors ACEfrictionCore's `env_transform(j, Rs, Zs, ::SphericalCutoff)`.
"""
function spherical_bond_transform(j_loc::Int, Rs::AbstractVector{<:SVector{3}},
                                  Zs::AbstractVector, sc::Union{SphericalCutoff,SnowManCutoff})
   rbond = Rs[j_loc] / sc.rcut
   Rs_env = SVector{3,Float64}[]; Zs_env = Int[]
   for l in eachindex(Rs)
      l == j_loc && continue
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

struct ETBondsIterator{TX}
   X::TX
   Z::Vector{Int}
   N::Int
   nlist_bond::PairList
   nlist_env::PairList
   ec::EllipsoidCutoff{Float64}
end

function et_bonds(sys::AbstractSystem, ec::EllipsoidCutoff)
   N = length(sys)
   X = SVector{3,Float64}[ SVector{3,Float64}(ustrip.(u"Å", position(sys, i))) for i in 1:N ]
   Z = Int[ Int(atomic_number(sys, i)) for i in 1:N ]
   nlist_bond = PairList(sys, ec.rcutbond * u"Å")
   nlist_env  = PairList(sys, env_cutoff(ec) * u"Å")
   return ETBondsIterator(X, Z, N, nlist_bond, nlist_env, ec)
end

function _bond_env(iter::ETBondsIterator, i, j, rrij)
   Js_i, Rs_i = neigs(iter.nlist_env, i)
   rri = iter.X[i]
   rrmid = rri + 0.5 * rrij
   ŝ = rrij / norm(rrij)
   Js = Int[]; Rs = SVector{3,Float64}[]; Zs = Int[]
   q_bond = findfirst(rrq -> rrq ≈ rrij, Rs_i)
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
