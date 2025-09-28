module BondCutoffs

# export AbstractCutoff, EllipsoidCutoff, SphericalCutoff, DSphericalCutoff
# export env_filter, env_transform, env_cutoff

using ACE
using ACE: State, DState, ForwardDiff
using JuLIP: AtomicNumber, chemical_symbol
using StaticArrays
using LinearAlgebra: norm, I

import ACE: write_dict, read_dict

export write_dict, read_dict, EllipsoidCutoff, CylindricalCutoff



"""
Every concrete subtype ConcreteBondCutoff{T} <: AbstractBondCutoff{T} must implement functions 

env_filter(r::T, z::T, env::ConcreteBondCutoff)::Bool where {T<:Real} 

that returns `true` exactly if an atom with cylindrical coordinates r, z, θ (relative to center of bond) is contained in the bond environment

env_transform(rrij::SVector, Zi::AtomicNumber, Zj::AtomicNumber, Rs::AbstractVector{<: SVector}, Zs::AbstractVector{<: AtomicNumber}, 
    env::ConcreteBondCutoff)

that transforms local coordinates into a format that is suitable for evaluyation by an (ACE) calculater.   

rrule_env_transform(rrij::SVector, Zi::AtomicNumber, Zj::AtomicNumber, Rs::AbstractVector{<: SVector}, Zs::AbstractVector{<: AtomicNumber}, 
    env::ConcreteBondCutoff)

that implements backward rrule associated with the transformation of env_transform. 

env_cutoff(env::ConcreteBondCutoff) 

that returns the maximum distance between a bond atom and a point in the environment / on the bond environment's surface.

Optionally, the following function might also be implemented. 

env_radius(env::ConcreteBondCutoff) 

returns the maximum distance between the bond center and a point in the environment / on the bond environment's surface.


"""
abstract type AbstractBondCutoff{T} end

"""
This implements a cylindrical cutoff for the bond environments: 
* A central bond (i,j) is within the cutoff if r_ij < rcutbond. 
* A neighbour atom at position rkij (relative position to midpoint) is within 
the environment if - after transformation to (r, θ, z) coordinates, it satisfies
`r <= rcutenv` and `abs(z) <= zcutenv`.

This struct implements the resulting filter under `env_filter`. 
"""
struct CylindricalCutoff{T} <: AbstractBondCutoff{T}
   rcutbond::T 
   rcutenv::T
   zcutenv::T
end

env_filter(r, z, cutoff::CylindricalCutoff) = (r <= cutoff.rcutenv) && (abs(z) <= cutoff.zcutenv)

env_cutoff(env::CylindricalCutoff) = sqrt((env.rcutbond*.5 + env.zcutenv)^2+env.rcutenv^2)

env_radius(env::CylindricalCutoff) = sqrt(env.zcutenv^2+env.rcutenv^2)

env_transform(rrij::SVector, Zi, Zj, 
      Rs::AbstractVector{<: SVector}, 
      Zs::AbstractVector{<: AtomicNumber}, 
      ::CylindricalCutoff)  = eucl2cyl(rrij, Zi, Zj, Rs, Zs)

# rrule_env_transform(rrij::SVector, Zi, Zj, 
#                 Rs::AbstractVector{<: SVector}, 
#                 Zs::AbstractVector{<: AtomicNumber}, 
#                 g_cyl::AbstractMatrix{<: DState},
#                 ::CylindricalCutoff )  = rrule_eucl2cyl(rrij::SVector, Zi, Zj, Rs, Zs, g_cyl)

# rrule_env_transform(rrij::SVector, Zi, Zj, 
#                 Rs::AbstractVector{<: SVector}, 
#                 Zs::AbstractVector{<: AtomicNumber}, 
#                 dV_cyl::AbstractVector{<: DState}, 
#                 ::CylindricalCutoff) = rrule_eucl2cyl(rrij::SVector, Zi, Zj, Rs, Zs, dV_cyl)

include("cylindrical_trans.jl") # contains rrules for eucl2cycl and other auxiliary functions

struct EllipsoidCutoff{T} <: AbstractBondCutoff{T}
    rcutbond::T 
    rcutenv::T
    zcutenv::T
end

env_filter(r, z, cutoff::EllipsoidCutoff) = ((z/cutoff.zcutenv)^2 +(r/cutoff.rcutenv)^2 <= 1)

env_cutoff(env::EllipsoidCutoff) = max(env.rcutbond*.5 + env.zcutenv, sqrt(env.rcutenv^2+(.5*env.rcutbond)^2))

env_radius(ec::EllipsoidCutoff) = max(ec.zcutenv, ec.rcutenv)


env_transform(rrij::SVector, Zi, Zj, 
    Rs::AbstractVector{<: SVector}, 
    Zs::AbstractVector{<: AtomicNumber}, 
    ec::EllipsoidCutoff) = ellipsoid2sphere(rrij, Zi, Zj, 
                            Rs,
                            Zs, 
                            ec.zcutenv, ec.rcutenv, ec.rcutbond)


# rrule_env_transform(rrij::SVector, Zi, Zj, 
#                 Rs::AbstractVector{<: SVector}, 
#                 Zs::AbstractVector{<: AtomicNumber}, 
#                 g_ell::AbstractMatrix{<: DState},
#                 ec::EllipsoidCutoff )  = rrule_ellipsoid2sphere(rrij, Zi, Zj, Rs, Zs, g_ell, 
#                                             ec.zcutenv, ec.rcutenv, ec.rcutbond)

# rrule_env_transform(rrij::SVector, Zi, Zj, 
#                 Rs::AbstractVector{<: SVector}, 
#                 Zs::AbstractVector{<: AtomicNumber}, 
#                 dV_cyl::AbstractVector{<: DState}, 
#                 ec::EllipsoidCutoff) = rrule_ellipsoid2sphere(rrij, Zi, Zj, Rs, Zs, dV_cyl,
#                                             ec.zcutenv, ec.rcutenv, ec.rcutbond)
            

include("ellipsoid_trans.jl") # contains rrules for ellipsoid2sphere and other auxiliary functions

function ACE.write_dict(cutoff::EllipsoidCutoff{T}) where {T}
    Dict("__id__" => "ACEbonds_EllipsoidCutoff",
          "rcutbond" => cutoff.rcutbond,
          "rcutenv" => cutoff.rcutenv,
          "zcutenv" => cutoff.zcutenv,
             "T" => T)         
end 

function ACE.read_dict(::Val{:ACEbonds_EllipsoidCutoff}, D::Dict)
    rcutbond = D["rcutbond"]
    rcutenv = D["rcutenv"]
    zcutenv = D["zcutenv"]
    T = getfield(Base, Symbol(D["T"]))
    return EllipsoidCutoff{T}(rcutbond,rcutenv,zcutenv)
end

end