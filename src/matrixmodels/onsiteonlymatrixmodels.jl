struct OnsiteOnlyMatrixModel{O3S} <: MatrixModel{O3S}
    onsite::OnSiteModels{O3S}
    n_rep::Int
    inds::SiteInds
    id::Symbol
    function OnsiteOnlyMatrixModel(onsite::OnSiteModels{O3S}, id::Symbol) where {O3S}
        @assert length(unique([_n_rep(mo) for mo in values(onsite)])) == 1
        return new{O3S}(onsite, _n_rep(onsite), SiteInds(_get_basisinds(onsite)), id)
    end
end

# Σ (diffusion coefficient matrix): per replica a block-diagonal matrix
function matrix(M::OnsiteOnlyMatrixModel, at::AbstractSystem; filter=(_,_)->true, T=Float64)
    N = length(at); Z = _species(at)
    Σ = [ Diagonal(zeros(_block_type(M,T), N)) for _ = 1:M.n_rep ]
    for (i, neigs, Rs) in _sites(at, env_cutoff(M.onsite))
        (haskey(M.onsite, Z[i]) && filter(i, at) && length(neigs) > 0) || continue
        Σi = evaluate(M.onsite[Z[i]], Rs, Z[neigs])
        for r = 1:M.n_rep
            Σ[r].diag[i] = Σi[r]
        end
    end
    return Σ
end

# un-contracted basis: per basis function a block-diagonal matrix
function basis(M::OnsiteOnlyMatrixModel, at::AbstractSystem; join_sites=false, filter=(_,_)->true, T=Float64)
    N = length(at); Z = _species(at)
    B = [ Diagonal(zeros(_block_type(M,T), N)) for _ = 1:length(M.inds, :onsite) ]
    for (i, neigs, Rs) in _sites(at, env_cutoff(M.onsite))
        (haskey(M.onsite, Z[i]) && filter(i, at) && length(neigs) > 0) || continue
        inds = get_range(M, Z[i])
        Bi = evaluate_basis(M.onsite[Z[i]], Rs, Z[neigs])
        for (k, b) in zip(inds, Bi)
            B[k].diag[i] = b
        end
    end
    return (join_sites ? B : (onsite = B,))
end

randf(::OnsiteOnlyMatrixModel, Σ::Diagonal{SMatrix{3,3,T,9}}) where {T<:Real} =
        Σ * randn(SVector{3,T}, size(Σ, 2))
randf(::OnsiteOnlyMatrixModel, Σ::Diagonal{SVector{3,T}}) where {T<:Real} =
        Σ * randn(size(Σ, 2))

# ---- serialization ----
function write_dict(m::OnSiteModel{O3S, NR}) where {O3S, NR}
    return Dict("__id__" => "ACEfriction_OnSiteModel",
                "basis" => write_dict(m.basis),
                "c" => collect(reinterpret(Vector{Float64}, m.c)),
                "n_rep" => NR,
                "rcut" => m.cutoff.rcut)
end
function read_dict(::Val{:ACEfriction_OnSiteModel}, D::AbstractDict)
    basis = read_dict(D["basis"]); NR = Int(D["n_rep"])
    c = reinterpret(Vector{SVector{NR, Float64}}, Vector{Float64}(D["c"]))
    return OnSiteModel(basis, SphericalCutoff(Float64(D["rcut"])), c)
end
function write_dict(onsite::OnSiteModels)
    return Dict("__id__" => "ACEfriction_onsitemodels",
                "zval" => Dict(string(_chemical_symbol(z)) => write_dict(v) for (z, v) in onsite))
end
read_dict(::Val{:ACEfriction_onsitemodels}, D::AbstractDict) =
        Dict(_atomic_number(Symbol(z)) => read_dict(v) for (z, v) in D["zval"])

function write_dict(M::OnsiteOnlyMatrixModel)
    return Dict("__id__" => "ACEfriction_OnsiteOnlyMatrixModel",
                "onsite" => write_dict(M.onsite), "id" => string(M.id))
end
read_dict(::Val{:ACEfriction_OnsiteOnlyMatrixModel}, D::AbstractDict) =
        OnsiteOnlyMatrixModel(read_dict(D["onsite"]), Symbol(D["id"]))
