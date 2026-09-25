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
       SphericalCutoff, EllipsoidCutoff, SnowManCutoff, _snowman_combine, partner_in_env,
       ellipsoid_env_transform, spherical_bond_transform, et_bonds, env_cutoff,
       _atomic_number, _chemical_symbol, block_type, output_LL
# fast (coefficient-contracted, per-centre) evaluators — see etbackend/fasteval.jl
import ACEfriction.ETBackend: ETFastModel, refresh!, ETBondCentre, bond_centre, bond_centre!,
       bond_sigma, bond_basis_blocks, centre_type, ETSiteData
import ACEfriction.ETBackend: write_dict, read_dict

export MatrixModel, CWCMatrixModel, RWCMatrixModel, OnsiteOnlyMatrixModel, PWCMatrixModel
export OnSiteModel, OffSiteModel, BondBasis, SiteInds
export onsite_linbasis, offsite_linbasis, env_cutoff
export O3Symmetry, Invariant, VectorEquivariant, MatrixEquivariant
export Odd, Even, NoZ2Sym, SpeciesCoupled, SpeciesUnCoupled
export NeighborCentered, AtomCentered
export SelfImagePolicy, ExcludeSelfImages, IncludeSelfImages
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

# Whether a periodic self-image of atom `i` (a neighbour whose atom index `j == i`)
# is kept as a bond partner during Σ assembly. `ExcludeSelfImages` (the default)
# drops such bonds so the off-diagonal-only models keep Σ_ii = 0; `IncludeSelfImages`
# retains them (PWC then contributes 1·Σ_ii Σ_iiᵀ to Γ, like the generic Σ Σᵀ).
abstract type SelfImagePolicy end
struct ExcludeSelfImages <: SelfImagePolicy end
struct IncludeSelfImages <: SelfImagePolicy end
# bond-partner filter (the dispatch point); `i`/`j` are atom indices
_keep_partner(::ExcludeSelfImages, i, j) = i != j
_keep_partner(::IncludeSelfImages, i, j) = true
# (de)serialization: name <-> singleton; missing/unknown name => ExcludeSelfImages (default)
_self_image_name(p::SelfImagePolicy) = string(nameof(typeof(p)))
_self_image_from_name(s::AbstractString) = (s == "IncludeSelfImages" ? IncludeSelfImages() : ExcludeSelfImages())

# Read the self-image policy from a serialized model dict. Models serialized before this
# option existed carry no "self_images" key; warn and default to ExcludeSelfImages.
function _self_image_from_dict(D::AbstractDict)
    haskey(D, "self_images") && return _self_image_from_name(D["self_images"])
    @warn("Importing a matrix model serialized before the `include_self_images` option " *
          "was introduced (no self-image information stored). Defaulting to " *
          "ExcludeSelfImages: self-image bond partners are dropped, so PWC keeps Σ_ii = 0. " *
          "If this model was fitted with self-images included, rebuild it with " *
          "`include_self_images = true`.", model = get(D, "__id__", "?"))
    return ExcludeSelfImages()
end

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

# Every site model carries a coefficient-contracted fast evaluator (`fast`) built on
# its basis. Its fused weights are a function of `c`; `set_params!` refreshes them
# and the assembly loops call `_fast(m)` once per site model per call, which
# re-syncs iff `c` was mutated behind the model's back (e.g. through `params(m)`).

struct OnSiteModel{O3S, NR, TB, TF} <: SiteModel
   basis::TB
   c::Vector{SVector{NR, Float64}}
   cutoff::SphericalCutoff{Float64}
   fast::TF
end
function OnSiteModel(basis::TB, cutoff::SphericalCutoff, c::Vector{SVector{NR,Float64}}) where {TB, NR}
   @assert length(basis) == length(c)
   O3S = _o3sym(basis.property)
   fast = ETFastModel(basis, c)
   return OnSiteModel{O3S, NR, TB, typeof(fast)}(basis, c, SphericalCutoff{Float64}(cutoff.rcut), fast)
end
OnSiteModel(basis, cutoff::SphericalCutoff, n_rep::Integer) =
      OnSiteModel(basis, cutoff, rand(SVector{n_rep, Float64}, length(basis)))
OnSiteModel(basis, r_cut::Real, n_rep::Integer) =
      OnSiteModel(basis, SphericalCutoff(Float64(r_cut)), n_rep)

struct OffSiteModel{O3S, Z2S, CUTOFF, NR, TB, TF} <: SiteModel
   basis::TB
   c::Vector{SVector{NR, Float64}}
   cutoff::CUTOFF
   fast::TF
end
function OffSiteModel(bb::BondBasis{TB, Z2S}, cutoff::CUTOFF, c::Vector{SVector{NR,Float64}}) where {TB, Z2S, CUTOFF, NR}
   @assert length(bb.basis) == length(c)
   O3S = _o3sym(bb.basis.property)
   fast = ETFastModel(bb.basis, c)
   return OffSiteModel{O3S, Z2S, CUTOFF, NR, TB, typeof(fast)}(bb.basis, c, cutoff, fast)
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
set_params!(m::SiteModel, c) = (copyto!(m.c, c); refresh!(m.fast, m.c); m)

"the site model's fast evaluator, re-synced with `m.c` if needed"
_fast(m::SiteModel) = refresh!(m.fast, m.c)

# contract ET basis blocks with the coefficients -> SVector{NR, block}
function _contract(m::SiteModel, B)
   NR = _n_rep(m); TB = block_type(m.basis)
   Σ = zero(MVector{NR, TB})
   @inbounds for k in eachindex(B), r in 1:NR
      Σ[r] += m.c[k][r] * B[k]
   end
   return SVector(Σ)
end

# The atom-centred bond cutoffs (Spherical / SnowMan) are the ones whose bonds share
# a centre; `_partner_in_env(sm)` is the environment convention of the bond model.
const AtomCentredCutoff = Union{SphericalCutoff, SnowManCutoff}
_partner_in_env(sm::OffSiteModel) = partner_in_env(sm.cutoff)

# ---- un-contracted basis (generic ET path; the fitting-path reference) ----

# onsite: raw env vectors (radial transform handles rcut)
evaluate_basis(sm::OnSiteModel, Rs, Zs) = ETBackend.evaluate(sm.basis, Rs, Zs)

# offsite ellipsoid: bond vector + ellipsoid env
function evaluate_basis(sm::OffSiteModel{O3S,Z2S,<:EllipsoidCutoff}, rrij::SVector{3}, Rs, Zs) where {O3S,Z2S}
   rbond, Rst, Zst = ellipsoid_env_transform(rrij, Rs, Zs, sm.cutoff)
   return evaluate_bond(sm.basis, rbond, Rst, Zst)
end

# offsite spherical / snowman: atom-i neighbourhood + bond-partner local index. For
# the snowman the two bond ends are combined at assembly time (pwcmatrixmodels.jl),
# so the single-centre evaluation is the spherical one.
function evaluate_basis(sm::OffSiteModel{O3S,Z2S,<:AtomCentredCutoff}, j_loc::Integer, Rs, Zs) where {O3S,Z2S}
   rbond, Rse, Zse = spherical_bond_transform(Int(j_loc), Rs, Zs, sm.cutoff)
   return evaluate_bond(sm.basis, rbond, Rse, Zse)
end

# ---- contracted Σ blocks: fast evaluators ----

# `evaluate` re-syncs the fused weights and uses a fresh workspace (safe to call
# anywhere). The assembly loops refresh once per call (`_refresh!`) and use
# `evaluate!` with a per-call workspace (`_workspace!`): workspaces are never stored
# in the model, so concurrent `matrix` calls on one model do not share buffers.
evaluate(sm::OnSiteModel, Rs, Zs) = ETBackend.evaluate(_fast(sm), Rs, Zs)
evaluate!(sd::ETSiteData, sm::OnSiteModel, Rs, Zs) = ETBackend.evaluate!(sd, sm.fast, Rs, Zs)

function evaluate(sm::OffSiteModel{O3S,Z2S,<:EllipsoidCutoff}, rrij::SVector{3}, Rs, Zs) where {O3S,Z2S}
   return evaluate!(ETSiteData(_fast(sm)), sm, rrij, Rs, Zs)
end
function evaluate!(sd::ETSiteData, sm::OffSiteModel{O3S,Z2S,<:EllipsoidCutoff}, rrij::SVector{3},
                   Rs, Zs) where {O3S,Z2S}
   rbond, Rst, Zst = ellipsoid_env_transform(rrij, Rs, Zs, sm.cutoff)
   return ETBackend.evaluate_bond!(sd, sm.fast, rbond, Rst, Zst)
end

# per-call neighbour workspaces, one per site model (keyed like the model dict)
_workspace_cache(models::AbstractDict{K}) where {K} = Dict{K, ETSiteData}()
@inline function _workspace!(cache, key, m::SiteModel)
   sd = get(cache, key, nothing)
   if sd === nothing
      sd = ETSiteData(m.fast)
      cache[key] = sd
   end
   return sd
end

# single bond of an atom-centred model: per-centre state built for this call only.
# The assembly loops use `bond_centre` / `bond_sigma` directly to share it.
function evaluate(sm::OffSiteModel{O3S,Z2S,<:AtomCentredCutoff}, j_loc::Integer, Rs, Zs) where {O3S,Z2S}
   pie = _partner_in_env(sm)
   ctr = bond_centre(_fast(sm), Rs, Zs, sm.cutoff.rcut; partner_in_env = pie)
   return bond_sigma(sm.fast, ctr, Int(j_loc); partner_in_env = pie)
end

# ---- reference contraction through the generic basis (for cross-checks) ----
evaluate_ref(sm::SiteModel, args...) = _contract(sm, evaluate_basis(sm, args...))

# refresh the fast evaluators of all site models of a matrix model (once per call)
function _refresh!(M)
   for site in (:onsite, :offsite)
      hasfield(typeof(M), site) || continue
      for m in values(getfield(M, site)); _fast(m); end
   end
   return M
end

# Per-species-pair bond-model states, reused across all centres of one assembly
# call: the state's `tag` records the centre it was last filled for, so a state is
# (re)filled at most once per centre and its buffers are never reallocated.
_centre_cache(offsite::AbstractDict) =
      Dict{Tuple{Int,Int}, centre_type(first(values(offsite)).fast)}()

# (explicit lookup rather than `get!` with a closure: the closure would capture and
# heap-box the whole site model on every call)
@inline function _get_centre!(cache, om::OffSiteModel, zz, i::Int, Rs, Zs)
   ctr = get(cache, zz, nothing)
   if ctr === nothing
      ctr = ETBondCentre(om.fast)
      cache[zz] = ctr
   end
   if ctr.tag != i
      bond_centre!(ctr, om.fast, Rs, Zs, om.cutoff.rcut; partner_in_env = _partner_in_env(om))
      ctr.tag = i
   end
   return ctr
end

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

"""
    randf(M::MatrixModel, Σ_vec::AbstractVector)

Random force of a single matrix model from its per-replica diffusion matrices
`Σ_vec = Sigma(M, at)` (one sparse / `Diagonal` matrix per replica). Each replica draws an
independent random force via the coupling-scheme-specific `randf(M, Σ)` method (defined in
the model files), and the model's force is their sum. Its covariance equals the model's
friction tensor `Gamma(M, Σ_vec)`. Usually called through [`randf(fm, Σ)`](@ref) rather
than directly.
"""
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
            species_substrat = [], o3symmetry = true, kwargs...)
   return onsite_basis(property, species;
            rcut = rcut, maxorder = maxorder, maxdeg = maxdeg, maxl = maxl,
            r0_ratio = r0_ratio, rin_ratio = rin_ratio, pcut = pcut, pin = pin,
            weight = weight, p_sel = p_sel,
            species_weight_cat = species_weight_cat,
            species_minorder_dict = species_minorder_dict,
            species_maxorder_dict = species_maxorder_dict,
            o3symmetry = o3symmetry)
end

function offsite_linbasis(property::ETProperty, species;
            z2symmetry = NoZ2Sym(), rcut = 1.0, maxorder = 2, maxdeg = 5,
            maxl = Int(floor(maxdeg)),
            r0_ratio = 0.4, rin_ratio = 0.04, pcut = 2, pin = 2, p_sel = 2,
            weight = Dict(:n => 1.0, :l => 1.0), bond_weight = 1.0,
            species_minorder_dict = Dict{Any, Float64}(),
            species_maxorder_dict = Dict{Any, Float64}(),
            species_weight_cat = Dict(c => 1.0 for c in species),
            species_substrat = [], isym = :mube, o3symmetry = true, kwargs...)
   b = bond_basis(property, species;
            z2sym = _z2flag(z2symmetry), rcut = rcut, maxorder = maxorder,
            maxdeg = maxdeg, maxl = maxl, r0_ratio = r0_ratio, rin_ratio = rin_ratio,
            pcut = pcut, pin = pin, weight = weight, p_sel = p_sel,
            bond_weight = bond_weight, species_weight_cat = species_weight_cat,
            species_minorder_dict = species_minorder_dict,
            species_maxorder_dict = species_maxorder_dict,
            o3symmetry = o3symmetry)
   return BondBasis(b, z2symmetry)
end

# ---------------------------------------------------------------------------
# matrix / basis assembly + the concrete models

include("./onsiteonlymatrixmodels.jl")
include("./pwcmatrixmodels.jl")
include("./acmatrixmodels.jl")

end
