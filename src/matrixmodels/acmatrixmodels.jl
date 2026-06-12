# Column-wise coupled matrix model: onsite (block-diagonal) + offsite (off-diagonal,
# atom-centred spherical pair environment). Σ is a full N×N block matrix; Γ = ΣΣᵀ.
# (Formerly RWCMatrixModel; see the deprecation stub at the bottom of this file.)

struct CWCMatrixModel{O3S, Z2S, SC, EC} <: MatrixModel{O3S}
    onsite::OnSiteModels{O3S}
    offsite::OffSiteModels{O3S, Z2S, <:SphericalCutoff}
    n_rep::Int
    inds::SiteInds
    id::Symbol
    evalcenter::EC
    function CWCMatrixModel(onsite::OnSiteModels{O3S}, offsite::OffSiteModels{O3S, Z2S, CUT},
                            id::Symbol, evalcenter::EC=AtomCentered(), sc::SC=SpeciesUnCoupled()
                            ) where {O3S, Z2S, CUT, SC, EC}
        @assert _n_rep(onsite) == _n_rep(offsite)
        inds = SiteInds(_get_basisinds(onsite), _get_basisinds(offsite))
        return new{O3S, Z2S, SC, EC}(onsite, offsite, _n_rep(onsite), inds, id, evalcenter)
    end
end

_get_SC(::CWCMatrixModel{O3S, Z2S, SC}) where {O3S, Z2S, SC} = SC
_cwc_rcut(M::CWCMatrixModel) = max(env_cutoff(M.onsite), env_cutoff(M.offsite))

function matrix(M::CWCMatrixModel{O3S, Z2S, SC}, at::AbstractSystem;
                filter=(_,_)->true, T=Float64) where {O3S, Z2S, SC}
    N = length(at); Z = _species(at)
    Is = [Int[] for _=1:M.n_rep]; Js = [Int[] for _=1:M.n_rep]; Vs = [_block_type(M,T)[] for _=1:M.n_rep]
    for (i, neigs, Rs) in _sites(at, _cwc_rcut(M))
        (filter(i, at) && length(neigs) > 0) || continue
        Zs = Z[neigs]
        if haskey(M.onsite, Z[i])
            Σi = evaluate(M.onsite[Z[i]], Rs, Zs)
            for r = 1:M.n_rep; push!(Is[r], i); push!(Js[r], i); push!(Vs[r], Σi[r]); end
        end
        for (j_loc, j) in enumerate(neigs)
            filter(j, at) || continue
            (Zi, Zj) = _mreduce(Z[i], Z[j], SC); haskey(M.offsite, (Zi, Zj)) || continue
            Σij = evaluate(M.offsite[(Zi, Zj)], j_loc, Rs, Zs)
            for r = 1:M.n_rep; push!(Is[r], i); push!(Js[r], j); push!(Vs[r], Σij[r]); end
        end
    end
    return [ sparse(Is[r], Js[r], Vs[r], N, N) for r = 1:M.n_rep ]
end

function basis(M::CWCMatrixModel{O3S, Z2S, SC}, at::AbstractSystem;
               join_sites=false, filter=(_,_)->true, T=Float64) where {O3S, Z2S, SC}
    N = length(at); Z = _species(at)
    Kon = length(M.inds, :onsite); Koff = length(M.inds, :offsite)
    Ion = [Int[] for _=1:Kon]; Jon = [Int[] for _=1:Kon]; Von = [_block_type(M,T)[] for _=1:Kon]
    Iof = [Int[] for _=1:Koff]; Jof = [Int[] for _=1:Koff]; Vof = [_block_type(M,T)[] for _=1:Koff]
    for (i, neigs, Rs) in _sites(at, _cwc_rcut(M))
        (filter(i, at) && length(neigs) > 0) || continue
        Zs = Z[neigs]
        if haskey(M.onsite, Z[i])
            Bi = evaluate_basis(M.onsite[Z[i]], Rs, Zs)
            for (k, b) in zip(get_range(M, Z[i]), Bi); push!(Ion[k], i); push!(Jon[k], i); push!(Von[k], b); end
        end
        for (j_loc, j) in enumerate(neigs)
            filter(j, at) || continue
            (Zi, Zj) = _mreduce(Z[i], Z[j], SC); haskey(M.offsite, (Zi, Zj)) || continue
            Bij = evaluate_basis(M.offsite[(Zi, Zj)], j_loc, Rs, Zs)
            for (k, b) in zip(get_range(M, (Zi, Zj)), Bij); push!(Iof[k], i); push!(Jof[k], j); push!(Vof[k], b); end
        end
    end
    Bon = [ sparse(Ion[k], Jon[k], Von[k], N, N) for k = 1:Kon ]
    Boff = [ sparse(Iof[k], Jof[k], Vof[k], N, N) for k = 1:Koff ]
    return (join_sites ? vcat(Bon, Boff) : (onsite = Bon, offsite = Boff))
end

function randf(::CWCMatrixModel, Σ::SparseMatrixCSC{SMatrix{3,3,T,9}, TI}) where {T<:Real, TI<:Int}
    return Σ * randn(SVector{3,T}, size(Σ, 2))
end

# ---- serialization ----
function write_dict(M::CWCMatrixModel{O3S, Z2S, SC, EC}) where {O3S, Z2S, SC, EC}
    return Dict("__id__" => "ACEfriction_CWCMatrixModel",
                "onsite" => write_dict(M.onsite), "offsite" => write_dict(M.offsite),
                "sc" => string(nameof(SC)), "evalcenter" => string(nameof(EC)),
                "id" => string(M.id))
end
function read_dict(::Val{:ACEfriction_CWCMatrixModel}, D::AbstractDict)
    onsite = read_dict(D["onsite"]); offsite = read_dict(D["offsite"])
    sc = getfield(@__MODULE__, Symbol(D["sc"]))()
    ec = getfield(@__MODULE__, Symbol(D["evalcenter"]))()
    return CWCMatrixModel(onsite, offsite, Symbol(D["id"]), ec, sc)
end
# Backward compatibility: models serialized under the former name `RWCMatrixModel`
# still load (into the renamed `CWCMatrixModel`).
read_dict(::Val{:ACEfriction_RWCMatrixModel}, D::AbstractDict) =
    read_dict(Val{:ACEfriction_CWCMatrixModel}(), D)

# ---- deprecation: RWCMatrixModel was renamed to CWCMatrixModel ----
# Constructing the old name now errors and points to the new name and the docs.
RWCMatrixModel(args...; kwargs...) = error(
    "RWCMatrixModel has been renamed to CWCMatrixModel (column-wise coupling). " *
    "Replace `RWCMatrixModel(...)` with `CWCMatrixModel(...)`. The coupling scheme called " *
    "\"row-wise coupling\" in Appendix C of the reference paper (Sachs et al., 2024) is more " *
    "appropriately termed \"column-wise coupling\", the terminology used in this version of " *
    "ACEfriction.jl. See the documentation section \"Matrix models and coupling schemes\".")
