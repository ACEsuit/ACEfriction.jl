module MatrixModels

# ET-native matrix models. The basis backend is EquivariantTensors (via
# ACEfriction.ETBackend); this module keeps the friction-specific machinery
# (markers, SiteInds, params/format plumbing, matrix!/basis! assembly) that the
# friction model + fitting layers depend on. (Replaces the ACEfrictionCore version.)

using LinearAlgebra, StaticArrays, SparseArrays
using LinearAlgebra: Diagonal
using AtomsBase: AbstractSystem, atomic_number
using NeighbourLists: PairList, sites, neigs
using Unitful: @u_str

using ACEfriction.mUtils: reinterpret
import ACEfriction.ETBackend
import ACEfriction.ETBackend: ETInvariant, ETVector, ETMatrix, ETSymMatrix, ETProperty,
       onsite_basis, bond_basis, evaluate_bond,
       SphericalCutoff, EllipsoidCutoff, SnowManCutoff, _snowman_combine,
       ellipsoid_env_transform, spherical_bond_transform, et_bonds, env_cutoff,
       _atomic_number, _chemical_symbol, block_type, output_LL
import ACEfriction.ETBackend: write_dict, read_dict

export MatrixModel, CWCMatrixModel, RWCMatrixModel, OnsiteOnlyMatrixModel, PWCMatrixModel
export OnSiteModel, OffSiteModel, BondBasis, SiteInds
export onsite_linbasis, offsite_linbasis, env_cutoff
export O3Symmetry, Invariant, VectorEquivariant, MatrixEquivariant
export Odd, Even, NoZ2Sym, SpeciesCoupled, SpeciesUnCoupled
export NeighborCentered, AtomCentered
export matrix, basis, params, nparams, set_params!, set_zero!, get_id, randf

# ---------------------------------------------------------------------------
# markers

abstract type O3Symmetry end
struct Invariant <: O3Symmetry end
struct VectorEquivariant <: O3Symmetry end
struct MatrixEquivariant <: O3Symmetry end

_o3sym(::ETInvariant) = Invariant
_o3sym(::ETVector)    = VectorEquivariant
_o3sym(::ETMatrix)    = MatrixEquivariant
_o3sym(::ETSymMatrix) = MatrixEquivariant

abstract type Z2Symmetry end
struct Odd <: Z2Symmetry end
struct Even <: Z2Symmetry end
struct NoZ2Sym <: Z2Symmetry end
_z2flag(::NoZ2Sym) = :none
_z2flag(::Odd) = :odd
_z2flag(::Even) = :even

abstract type SpeciesCoupling end
struct SpeciesCoupled <: SpeciesCoupling end
struct SpeciesUnCoupled <: SpeciesCoupling end

abstract type EvaluationCenter end
struct NeighborCentered <: EvaluationCenter end
struct AtomCentered <: EvaluationCenter end

# JuLIP-free helpers (species as Int atomic numbers; neighbour iteration)
_species(at::AbstractSystem) = Int[ Int(atomic_number(at, i)) for i in 1:length(at) ]
_sites(at::AbstractSystem, rcut::Real) = sites(PairList(at, rcut * u"Å"))
_msort(z1, z2) = z1 <= z2 ? (z1, z2) : (z2, z1)
_mreduce(z1, z2, ::SpeciesUnCoupled) = (z1, z2)
_mreduce(z1, z2, ::SpeciesCoupled) = _msort(z1, z2)
_mreduce(z1, z2, ::Type{SpeciesUnCoupled}) = (z1, z2)
_mreduce(z1, z2, ::Type{SpeciesCoupled}) = _msort(z1, z2)

# ---------------------------------------------------------------------------
# site models (flattened: ET basis + coefficients, no LinearACEModel)

"""bond basis wrapper carrying the Z2 symmetry tag."""
struct BondBasis{TB, Z2SYM}
   basis::TB
   BondBasis(basis::TB, ::Z2SYM) where {TB, Z2SYM <: Z2Symmetry} = new{TB, Z2SYM}(basis)
end
Base.length(bb::BondBasis) = length(bb.basis)

abstract type SiteModel end

struct OnSiteModel{O3S, NR, TB} <: SiteModel
   basis::TB
   c::Vector{SVector{NR, Float64}}
   cutoff::SphericalCutoff{Float64}
end
function OnSiteModel(basis::TB, cutoff::SphericalCutoff, c::Vector{SVector{NR,Float64}}) where {TB, NR}
   @assert length(basis) == length(c)
   O3S = _o3sym(basis.property)
   return OnSiteModel{O3S, NR, TB}(basis, c, SphericalCutoff{Float64}(cutoff.rcut))
end
OnSiteModel(basis, cutoff::SphericalCutoff, n_rep::Integer) =
      OnSiteModel(basis, cutoff, rand(SVector{n_rep, Float64}, length(basis)))
OnSiteModel(basis, r_cut::Real, n_rep::Integer) =
      OnSiteModel(basis, SphericalCutoff(Float64(r_cut)), n_rep)

struct OffSiteModel{O3S, Z2S, CUTOFF, NR, TB} <: SiteModel
   basis::TB
   c::Vector{SVector{NR, Float64}}
   cutoff::CUTOFF
end
function OffSiteModel(bb::BondBasis{TB, Z2S}, cutoff::CUTOFF, c::Vector{SVector{NR,Float64}}) where {TB, Z2S, CUTOFF, NR}
   @assert length(bb.basis) == length(c)
   O3S = _o3sym(bb.basis.property)
   return OffSiteModel{O3S, Z2S, CUTOFF, NR, TB}(bb.basis, c, cutoff)
end
OffSiteModel(bb::BondBasis, cutoff, n_rep::Integer) =
      OffSiteModel(bb, cutoff, rand(SVector{n_rep, Float64}, length(bb.basis)))
OffSiteModel(bb::BondBasis, r_cut::Real, n_rep::Integer) =
      OffSiteModel(bb, SphericalCutoff(Float64(r_cut)), n_rep)
OffSiteModel(bb::BondBasis, rcutbond::Real, rcutenv::Real, zcutenv::Real, n_rep::Integer) =
      OffSiteModel(bb, EllipsoidCutoff(Float64(rcutbond), Float64(rcutenv), Float64(zcutenv)), n_rep)

_n_rep(::OnSiteModel{O3S, NR}) where {O3S, NR} = NR
_n_rep(::OffSiteModel{O3S, Z2S, CUTOFF, NR}) where {O3S, Z2S, CUTOFF, NR} = NR
_o3symmetry(::OnSiteModel{O3S}) where {O3S} = O3S
_o3symmetry(::OffSiteModel{O3S}) where {O3S} = O3S
Base.length(m::SiteModel) = length(m.basis)
params(m::SiteModel) = m.c
nparams(m::SiteModel) = length(m.c)
set_params!(m::SiteModel, c) = (copyto!(m.c, c); m)

# contract ET basis blocks with the coefficients -> SVector{NR, block}
function _contract(m::SiteModel, B)
   NR = _n_rep(m); TB = block_type(m.basis)
   Σ = zero(MVector{NR, TB})
   @inbounds for k in eachindex(B), r in 1:NR
      Σ[r] += m.c[k][r] * B[k]
   end
   return SVector(Σ)
end

# onsite: raw env vectors (radial transform handles rcut)
evaluate(sm::OnSiteModel, Rs, Zs) = _contract(sm, ETBackend.evaluate(sm.basis, Rs, Zs))
evaluate_basis(sm::OnSiteModel, Rs, Zs) = ETBackend.evaluate(sm.basis, Rs, Zs)

# offsite ellipsoid: bond vector + ellipsoid env
function evaluate_basis(sm::OffSiteModel{O3S,Z2S,<:EllipsoidCutoff}, rrij::SVector{3}, Rs, Zs) where {O3S,Z2S}
   rbond, Rst, Zst = ellipsoid_env_transform(rrij, Rs, Zs, sm.cutoff)
   return evaluate_bond(sm.basis, rbond, Rst, Zst)
end
evaluate(sm::OffSiteModel{O3S,Z2S,<:EllipsoidCutoff}, rrij::SVector{3}, Rs, Zs) where {O3S,Z2S} =
      _contract(sm, evaluate_basis(sm, rrij, Rs, Zs))

# offsite spherical: atom-i neighbourhood + bond-partner local index
function evaluate_basis(sm::OffSiteModel{O3S,Z2S,<:SphericalCutoff}, j_loc::Integer, Rs, Zs) where {O3S,Z2S}
   rbond, Rse, Zse = spherical_bond_transform(Int(j_loc), Rs, Zs, sm.cutoff)
   return evaluate_bond(sm.basis, rbond, Rse, Zse)
end
evaluate(sm::OffSiteModel{O3S,Z2S,<:SphericalCutoff}, j_loc::Integer, Rs, Zs) where {O3S,Z2S} =
      _contract(sm, evaluate_basis(sm, j_loc, Rs, Zs))

# offsite snowman: single-centre spherical evaluation (same as spherical). The two
# bond ends are combined at assembly time in pwcmatrixmodels.jl (Σ_ij = c·B(env_ij)
# + c·B(env_ji)), so per-centre evaluation reuses the spherical transform.
function evaluate_basis(sm::OffSiteModel{O3S,Z2S,<:SnowManCutoff}, j_loc::Integer, Rs, Zs) where {O3S,Z2S}
   rbond, Rse, Zse = spherical_bond_transform(Int(j_loc), Rs, Zs, sm.cutoff)
   return evaluate_bond(sm.basis, rbond, Rse, Zse)
end
evaluate(sm::OffSiteModel{O3S,Z2S,<:SnowManCutoff}, j_loc::Integer, Rs, Zs) where {O3S,Z2S} =
      _contract(sm, evaluate_basis(sm, j_loc, Rs, Zs))

const OnSiteModels{O3S} = Dict{Int, <:OnSiteModel{O3S}}
const OffSiteModels{O3S, Z2S, CUTOFF} = Dict{Tuple{Int,Int}, <:OffSiteModel{O3S, Z2S, CUTOFF}}
const SiteModels = Union{OnSiteModels, OffSiteModels}

function _n_rep(models::SiteModels)
   n = unique(_n_rep(m) for m in values(models)); @assert length(n) == 1; return n[1]
end
env_cutoff(models::SiteModels) = maximum(env_cutoff(m.cutoff) for m in values(models))

# ---------------------------------------------------------------------------
# SiteInds (basis-function index ranges per species / species-pair)

struct SiteInds
   onsite::Dict{Int, UnitRange{Int}}
   offsite::Dict{Tuple{Int,Int}, UnitRange{Int}}
end
SiteInds(onsite::Dict{Int, UnitRange{Int}}) = SiteInds(onsite, Dict{Tuple{Int,Int}, UnitRange{Int}}())
SiteInds(offsite::Dict{Tuple{Int,Int}, UnitRange{Int}}) = SiteInds(Dict{Int, UnitRange{Int}}(), offsite)

Base.length(inds::SiteInds) = length(inds, :onsite) + length(inds, :offsite)
Base.length(inds::SiteInds, site::Symbol) =
      isempty(getfield(inds, site)) ? 0 : sum(length(r) for r in values(getfield(inds, site)))
get_range(inds::SiteInds, z::Int) = inds.onsite[z]
get_range(inds::SiteInds, zz::Tuple{Int,Int}) = inds.offsite[zz]

function _get_basisinds(models::Dict{Z, TM}) where {Z, TM}
   inds = Dict{Z, UnitRange{Int}}(); i0 = 1
   for (zz, mo) in models
      len = nparams(mo); inds[zz] = i0:(i0+len-1); i0 += len
   end
   return inds
end

# ---------------------------------------------------------------------------
# MatrixModel abstract + block helpers

abstract type MatrixModel{S} end

_default_id(::Type{Invariant}) = :inv
_default_id(::Type{VectorEquivariant}) = :cov
_default_id(::Type{MatrixEquivariant}) = :equ
_default_id(::Type{<:O3Symmetry}) = :equ

_block_type(::MatrixModel{Invariant}, T = Float64) = SMatrix{3, 3, T, 9}
_block_type(::MatrixModel{VectorEquivariant}, T = Float64) = SVector{3, T}
_block_type(::MatrixModel{MatrixEquivariant}, T = Float64) = SMatrix{3, 3, T, 9}

_n_rep(M::MatrixModel) = M.n_rep
get_id(M::MatrixModel) = M.id

# `Sigma(M, at)` returns one (sparse / Diagonal) matrix per replica; the per-replica
# `randf` methods (in the model files) draw an independent random force for each, and
# the model's random force is their sum.
randf(M::MatrixModel, Σ_vec::AbstractVector) = sum(randf(M, Σ) for Σ in Σ_vec)
Base.length(m::MatrixModel, args...) = length(m.inds, args...)
get_range(m::MatrixModel, args...) = get_range(m.inds, args...)
_get_model(M::MatrixModel, zz::Tuple{Int,Int}) = M.offsite[zz]
_get_model(M::MatrixModel, z::Int) = M.onsite[z]

# ---------------------------------------------------------------------------
# params / format plumbing (unchanged machinery; operates on c::Vector{SVector})

function params(mb::MatrixModel; format = :matrix, joinsites = true)
   @assert format in [:native, :matrix]
   if joinsites
      return vcat(params(mb, :onsite; format = format), params(mb, :offsite; format = format))
   else
      return (onsite = params(mb, :onsite; format = format),
              offsite = params(mb, :offsite; format = format))
   end
end

# site model dict for a model that may not have both onsite/offsite fields
_site_dict(mb::MatrixModel, site::Symbol) =
      hasfield(typeof(mb), site) ? getfield(mb, site) : Dict{Any,Any}()

function params(mb::MatrixModel, site::Symbol; format = :matrix)
   θ = zeros(SVector{mb.n_rep, Float64}, nparams(mb, site))
   for z in keys(_site_dict(mb, site))
      θ[get_range(mb, z)] = params(_get_model(mb, z))
   end
   return _transform(θ, Val(format), mb.n_rep)
end

nparams(mb::MatrixModel) = length(mb.inds, :onsite) + length(mb.inds, :offsite)
nparams(mb::MatrixModel, site::Symbol) = length(mb.inds, site)

function set_params!(mb::MatrixModel, θ)
   set_params!(mb, _split_sites(mb, θ))
end
function set_params!(mb::MatrixModel, θ::NamedTuple)
   for site in keys(θ); set_params!(mb, site, θ[site]); end
   return mb
end
function set_params!(mb::MatrixModel, site::Symbol, θ)
   hasfield(typeof(mb), site) || return mb
   θt = _rev_transform(θ, mb.n_rep)
   for z in keys(getfield(mb, site))
      set_params!(_get_model(mb, z), θt[get_range(mb, z)])
   end
   return mb
end
function set_zero!(mb::MatrixModel)
   for site in (:onsite, :offsite)
      hasfield(typeof(mb), site) || continue
      set_params!(mb, site, zeros(size(params(mb, site; format = :matrix))))
   end
   return mb
end

_join_sites(h1, h2) = vcat(h1, h2)
function _split_sites(mb::MatrixModel, h::Vector)
   i = length(mb, :onsite); return (onsite = h[1:i], offsite = h[(i+1):end])
end
function _split_sites(mb::MatrixModel, H::Matrix)
   i = length(mb, :onsite); return (onsite = H[1:i, :], offsite = H[(i+1):end, :])
end
_transform(θ, ::Val{:matrix}, n_rep) = reinterpret(Matrix{Float64}, θ)
_transform(θ, ::Val{:native}, n_rep) = reinterpret(Vector{SVector{n_rep, Float64}}, θ)
_rev_transform(θ, n_rep) = reinterpret(Vector{SVector{n_rep, Float64}}, θ)

function scaling(mb::MatrixModel, p::Int)
   scale = (onsite = ones(length(mb, :onsite)), offsite = ones(length(mb, :offsite)))
   for site in (:onsite, :offsite)
      hasfield(typeof(mb), site) || continue
      for (zz, mo) in getfield(mb, site)
         scale[site][get_range(mb, zz)] = ETBackend.scaling(mo.basis, p)
      end
   end
   return scale
end

# ---------------------------------------------------------------------------
# basis builders (delegate to ETBackend)

_z2_sym(::NoZ2Sym) = NoZ2Sym(); _z2_sym(::Odd) = Odd(); _z2_sym(::Even) = Even()

function onsite_linbasis(property::ETProperty, species;
            rcut = 5.0, maxorder = 2, maxdeg = 5, maxl = Int(floor(maxdeg)),
            r0_ratio = 0.4, rin_ratio = 0.04, pcut = 2, pin = 2, p_sel = 2,
            weight = Dict(:n => 1.0, :l => 1.0),
            species_minorder_dict = Dict{Any, Float64}(),
            species_maxorder_dict = Dict{Any, Float64}(),
            species_weight_cat = Dict(c => 1.0 for c in species),
            species_substrat = [], kwargs...)
   return onsite_basis(property, species;
            rcut = rcut, maxorder = maxorder, maxdeg = maxdeg, maxl = maxl,
            r0_ratio = r0_ratio, rin_ratio = rin_ratio, pcut = pcut, pin = pin,
            weight = weight, p_sel = p_sel,
            species_weight_cat = species_weight_cat,
            species_minorder_dict = species_minorder_dict,
            species_maxorder_dict = species_maxorder_dict)
end

function offsite_linbasis(property::ETProperty, species;
            z2symmetry = NoZ2Sym(), rcut = 1.0, maxorder = 2, maxdeg = 5,
            maxl = Int(floor(maxdeg)),
            r0_ratio = 0.4, rin_ratio = 0.04, pcut = 2, pin = 2, p_sel = 2,
            weight = Dict(:n => 1.0, :l => 1.0), bond_weight = 1.0,
            species_minorder_dict = Dict{Any, Float64}(),
            species_maxorder_dict = Dict{Any, Float64}(),
            species_weight_cat = Dict(c => 1.0 for c in species),
            species_substrat = [], isym = :mube, kwargs...)
   b = bond_basis(property, species;
            z2sym = _z2flag(z2symmetry), rcut = rcut, maxorder = maxorder,
            maxdeg = maxdeg, maxl = maxl, r0_ratio = r0_ratio, rin_ratio = rin_ratio,
            pcut = pcut, pin = pin, weight = weight, p_sel = p_sel,
            bond_weight = bond_weight, species_weight_cat = species_weight_cat,
            species_minorder_dict = species_minorder_dict,
            species_maxorder_dict = species_maxorder_dict)
   return BondBasis(b, z2symmetry)
end

# ---------------------------------------------------------------------------
# matrix / basis assembly + the concrete models

include("./onsiteonlymatrixmodels.jl")
include("./pwcmatrixmodels.jl")
include("./acmatrixmodels.jl")

end
