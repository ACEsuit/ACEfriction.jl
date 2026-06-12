using ACEfriction.MatrixModels
import ACEfriction.ETBackend: _atomic_number
import ACEfriction.MatrixModels: CWCMatrixModel, RWCMatrixModel, PWCMatrixModel, OnsiteOnlyMatrixModel
import ACEfriction.MatrixModels: OnSiteModel, OffSiteModel, BondBasis, onsite_linbasis,
       offsite_linbasis, SphericalCutoff, EllipsoidCutoff, SnowManCutoff, _o3symmetry, _default_id,
       _mreduce, NoZ2Sym, Odd, Even, SpeciesCoupled, SpeciesUnCoupled,
       AtomCentered, NeighborCentered, EvaluationCenter, _o3sym
export CWCMatrixModel, RWCMatrixModel, PWCMatrixModel, OnsiteOnlyMatrixModel

# Outer convenience constructors. The basis backend is EquivariantTensors; the
# `polytransform`/`trans` argument of the old backend is gone (the radial transform
# is a generalized Agnesi parameterised by `r0_ratio`/`rin_ratio`).

_o3id(property) = _default_id(_o3sym(property))

"""
    OnsiteOnlyMatrixModel(property, species_friction, species_env;
        maxorder=2, maxdeg=5, rcut=5.0, n_rep=1, ...)

Block-diagonal friction model (onsite only).
"""
function OnsiteOnlyMatrixModel(property, species_friction, species_env;
        id=nothing, n_rep=1, maxorder=2, maxdeg=5, rcut=5.0,
        r0_ratio=0.4, rin_ratio=0.04, pcut=2, pin=2, p_sel=2,
        weight=Dict(:n => 1.0, :l => 1.0),
        species_minorder_dict=Dict{Any,Float64}(),
        species_maxorder_dict=Dict{Any,Float64}(),
        species_weight_cat=Dict(c => 1.0 for c in species_env),
        species_substrat=[])
    onsitebasis = onsite_linbasis(property, species_env;
        rcut=rcut, maxorder=maxorder, maxdeg=maxdeg, r0_ratio=r0_ratio,
        rin_ratio=rin_ratio, pcut=pcut, pin=pin, p_sel=p_sel, weight=weight,
        species_minorder_dict=species_minorder_dict,
        species_maxorder_dict=species_maxorder_dict,
        species_weight_cat=species_weight_cat, species_substrat=species_substrat)
    onsitemodels = Dict(_atomic_number(z) => OnSiteModel(onsitebasis, SphericalCutoff(rcut), n_rep)
                        for z in species_friction)
    id = (id === nothing ? _o3id(property) : id)
    return OnsiteOnlyMatrixModel(onsitemodels, id)
end

"""
    PWCMatrixModel(property, species_friction, species_env;
        maxorder=2, maxdeg=5, rcut=5.0, n_rep=1, ...)

Pairwise-coupled (offsite) friction model with a spherical pair environment.
"""
function PWCMatrixModel(property, species_friction, species_env;
        id=nothing, n_rep=1, maxorder=2, maxdeg=5, rcut=5.0,
        z2sym=NoZ2Sym(), speciescoupling=SpeciesUnCoupled(),
        r0_ratio=0.4, rin_ratio=0.04, pcut=2, pin=2, p_sel=2,
        weight=Dict(:n => 1.0, :l => 1.0), bond_weight=1.0,
        species_minorder_dict=Dict{Any,Float64}(),
        species_maxorder_dict=Dict{Any,Float64}(),
        species_weight_cat=Dict(c => 1.0 for c in species_env),
        species_substrat=[])
    bb = offsite_linbasis(property, species_env;
        z2symmetry=z2sym, rcut=1.0, maxorder=maxorder, maxdeg=maxdeg,
        r0_ratio=r0_ratio, rin_ratio=rin_ratio, pcut=pcut, pin=pin, p_sel=p_sel,
        weight=weight, bond_weight=bond_weight,
        species_minorder_dict=species_minorder_dict,
        species_maxorder_dict=species_maxorder_dict,
        species_weight_cat=species_weight_cat, species_substrat=species_substrat)
    cutoff = SphericalCutoff(rcut)
    offsitemodels = _offsite_dict(bb, cutoff, species_friction, n_rep, speciescoupling)
    id = (id === nothing ? _o3id(property) : id)
    return PWCMatrixModel(offsitemodels, id, speciescoupling)
end

"""
    PWCMatrixModel(property, species_friction, species_env, cutoff::EllipsoidCutoff; ...)

Pairwise-coupled offsite model with an ellipsoidal (bond-centred) environment.
"""
function PWCMatrixModel(property, species_friction, species_env, cutoff::EllipsoidCutoff;
        id=nothing, n_rep=1, maxorder=2, maxdeg=5,
        z2sym=NoZ2Sym(), speciescoupling=SpeciesUnCoupled(),
        r0_ratio=0.4, rin_ratio=0.04, pcut=2, pin=2, p_sel=2,
        weight=Dict(:n => 1.0, :l => 1.0), bond_weight=1.0,
        species_minorder_dict=Dict{Any,Float64}(),
        species_maxorder_dict=Dict{Any,Float64}(),
        species_weight_cat=Dict(c => 1.0 for c in species_env),
        species_substrat=[])
    bb = offsite_linbasis(property, species_env;
        z2symmetry=z2sym, rcut=1.0, maxorder=maxorder, maxdeg=maxdeg,
        r0_ratio=r0_ratio, rin_ratio=rin_ratio, pcut=pcut, pin=pin, p_sel=p_sel,
        weight=weight, bond_weight=bond_weight,
        species_minorder_dict=species_minorder_dict,
        species_maxorder_dict=species_maxorder_dict,
        species_weight_cat=species_weight_cat, species_substrat=species_substrat)
    offsitemodels = _offsite_dict(bb, cutoff, species_friction, n_rep, speciescoupling)
    id = (id === nothing ? _o3id(property) : id)
    return PWCMatrixModel(offsitemodels, id, speciescoupling)
end

"""
    PWCMatrixModel(property, species_friction, species_env, cutoff::SnowManCutoff; ...)

Pairwise-coupled offsite model with a *symmetrised* atom-centred environment: the
block of pair `(i,j)` is `c·basis(sphere_i, bond i→j) + c·basis(sphere_j, bond j→i)`
(two overlapping spheres, one per bond end). `cutoff.rcut` is the per-centre radius.
Accepts the same selection/radial keywords as the other `PWCMatrixModel` methods.
"""
function PWCMatrixModel(property, species_friction, species_env, cutoff::SnowManCutoff;
        id=nothing, n_rep=1, maxorder=2, maxdeg=5,
        z2sym=NoZ2Sym(), speciescoupling=SpeciesUnCoupled(),
        r0_ratio=0.4, rin_ratio=0.04, pcut=2, pin=2, p_sel=2,
        weight=Dict(:n => 1.0, :l => 1.0), bond_weight=1.0,
        species_minorder_dict=Dict{Any,Float64}(),
        species_maxorder_dict=Dict{Any,Float64}(),
        species_weight_cat=Dict(c => 1.0 for c in species_env),
        species_substrat=[])
    bb = offsite_linbasis(property, species_env;
        z2symmetry=z2sym, rcut=1.0, maxorder=maxorder, maxdeg=maxdeg,
        r0_ratio=r0_ratio, rin_ratio=rin_ratio, pcut=pcut, pin=pin, p_sel=p_sel,
        weight=weight, bond_weight=bond_weight,
        species_minorder_dict=species_minorder_dict,
        species_maxorder_dict=species_maxorder_dict,
        species_weight_cat=species_weight_cat, species_substrat=species_substrat)
    offsitemodels = _offsite_dict(bb, cutoff, species_friction, n_rep, speciescoupling)
    id = (id === nothing ? _o3id(property) : id)
    return PWCMatrixModel(offsitemodels, id, speciescoupling)
end

function _offsite_dict(bb, cutoff, species_friction, n_rep, sc::SpeciesUnCoupled)
    return Dict(_atomic_number.(zz) => OffSiteModel(bb, cutoff, n_rep)
                for zz in Base.Iterators.product(species_friction, species_friction))
end
function _offsite_dict(bb, cutoff, species_friction, n_rep, sc::SpeciesCoupled)
    return Dict(_atomic_number.(zz) => OffSiteModel(bb, cutoff, n_rep)
                for zz in Base.Iterators.product(species_friction, species_friction)
                if _mreduce(zz..., SpeciesCoupled) == zz)
end

"""
    CWCMatrixModel(property, species_friction, species_env; maxorder=2, maxdeg=5, rcut=5.0, n_rep=1, ...)

Column-wise coupled friction model (onsite + spherical offsite).

!!! note "Renamed from `RWCMatrixModel`"
    This model was formerly called `RWCMatrixModel`. The coupling scheme named
    *row-wise coupling* in Appendix C of the reference paper (Sachs et al., 2024) is more
    appropriately termed *column-wise coupling*, the terminology used here. Constructing
    `RWCMatrixModel` now raises an error pointing to `CWCMatrixModel`.
"""
function CWCMatrixModel(property, species_friction, species_env;
        id=nothing, n_rep=1, maxorder=2, maxdeg=5, rcut=5.0,
        evalcenter=AtomCentered(), speciescoupling=SpeciesUnCoupled(),
        r0_ratio=0.4, rin_ratio=0.04, pcut=2, pin=2, p_sel=2,
        weight=Dict(:n => 1.0, :l => 1.0), bond_weight=1.0,
        species_minorder_dict=Dict{Any,Float64}(),
        species_maxorder_dict=Dict{Any,Float64}(),
        species_weight_cat=Dict(c => 1.0 for c in species_env),
        species_substrat=[])
    onsitebasis = onsite_linbasis(property, species_env;
        rcut=rcut, maxorder=maxorder, maxdeg=maxdeg, r0_ratio=r0_ratio,
        rin_ratio=rin_ratio, pcut=pcut, pin=pin, p_sel=p_sel, weight=weight,
        species_minorder_dict=species_minorder_dict,
        species_maxorder_dict=species_maxorder_dict,
        species_weight_cat=species_weight_cat, species_substrat=species_substrat)
    bb = offsite_linbasis(property, species_env;
        z2symmetry=NoZ2Sym(), rcut=1.0, maxorder=maxorder, maxdeg=maxdeg,
        r0_ratio=r0_ratio, rin_ratio=rin_ratio, pcut=pcut, pin=pin, p_sel=p_sel,
        weight=weight, bond_weight=bond_weight,
        species_minorder_dict=species_minorder_dict,
        species_maxorder_dict=species_maxorder_dict,
        species_weight_cat=species_weight_cat, species_substrat=species_substrat)
    onsitemodels = Dict(_atomic_number(z) => OnSiteModel(onsitebasis, SphericalCutoff(rcut), n_rep)
                        for z in species_friction)
    offsitemodels = _offsite_dict(bb, SphericalCutoff(rcut), species_friction, n_rep, speciescoupling)
    id = (id === nothing ? _o3id(property) : id)
    return CWCMatrixModel(onsitemodels, offsitemodels, id, evalcenter, speciescoupling)
end

"""
    CWCMatrixModel(property, species_friction, species_env, evalcenter::EvaluationCenter;
                   rcut_on, rcut_off, maxorder_on, maxdeg_on, ..._on/_off, bond_weight)

Backward-compatible CWC constructor with a positional `evalcenter` and separate
onsite/offsite hyperparameters (mirrors the pre-ET interface).
"""
function CWCMatrixModel(property, species_friction, species_env, evalcenter::EvaluationCenter;
        id=nothing, n_rep=1, speciescoupling=SpeciesUnCoupled(), species_substrat=[],
        rcut_on=5.0, rcut_off=rcut_on,
        maxorder_on=2, maxdeg_on=5, maxorder_off=maxorder_on, maxdeg_off=maxdeg_on,
        r0_ratio=0.4, rin_ratio=0.04, pcut=2, pin=2, p_sel=2,
        weight_on=Dict(:n => 1.0, :l => 1.0), weight_off=weight_on, bond_weight=1.0,
        species_minorder_dict_on=Dict{Any,Float64}(), species_maxorder_dict_on=Dict{Any,Float64}(),
        species_weight_cat_on=Dict(c => 1.0 for c in species_env),
        species_minorder_dict_off=Dict{Any,Float64}(), species_maxorder_dict_off=Dict{Any,Float64}(),
        species_weight_cat_off=Dict(c => 1.0 for c in species_env), kwargs...)
    onsitebasis = onsite_linbasis(property, species_env;
        rcut=rcut_on, maxorder=maxorder_on, maxdeg=maxdeg_on, r0_ratio=r0_ratio,
        rin_ratio=rin_ratio, pcut=pcut, pin=pin, p_sel=p_sel, weight=weight_on,
        species_minorder_dict=species_minorder_dict_on,
        species_maxorder_dict=species_maxorder_dict_on,
        species_weight_cat=species_weight_cat_on, species_substrat=species_substrat)
    bb = offsite_linbasis(property, species_env;
        z2symmetry=NoZ2Sym(), rcut=1.0, maxorder=maxorder_off, maxdeg=maxdeg_off,
        r0_ratio=r0_ratio, rin_ratio=rin_ratio, pcut=pcut, pin=pin, p_sel=p_sel,
        weight=weight_off, bond_weight=bond_weight,
        species_minorder_dict=species_minorder_dict_off,
        species_maxorder_dict=species_maxorder_dict_off,
        species_weight_cat=species_weight_cat_off, species_substrat=species_substrat)
    onsitemodels = Dict(_atomic_number(z) => OnSiteModel(onsitebasis, SphericalCutoff(rcut_on), n_rep)
                        for z in species_friction)
    offsitemodels = _offsite_dict(bb, SphericalCutoff(rcut_off), species_friction, n_rep, speciescoupling)
    id = (id === nothing ? _o3id(property) : id)
    return CWCMatrixModel(onsitemodels, offsitemodels, id, evalcenter, speciescoupling)
end
