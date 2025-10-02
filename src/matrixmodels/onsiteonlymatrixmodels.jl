struct OnsiteOnlyMatrixModel{O3S} <: MatrixModel{O3S}
    onsite::OnSiteModels{O3S}
    n_rep::Int
    inds::SiteInds
    id::Symbol
    function OnsiteOnlyMatrixModel(onsite::OnSiteModels{O3S}, id::Symbol) where {O3S}
        @assert length(unique([_n_rep(mo) for mo in values(onsite)])) == 1
        @assert length(unique([mo.cutoff for mo in values(onsite)])) == 1 
        return new{O3S}(onsite, _n_rep(onsite), SiteInds(_get_basisinds(onsite)), id)
    end
end

function ACEfrictionCore.params(mb::OnsiteOnlyMatrixModel; format=:matrix, joinsites=true) # :vector, :matrix
    @assert format in [:native, :matrix]
    if joinsites  
        return ACEfrictionCore.params(mb, :onsite; format=format)
    else 
        θ_onsite = ACEfrictionCore.params(mb, :onsite; format=format)
        return (onsite=θ_onsite, offsite=eltype(θ_offsite)[],)
    end
end

function ACEfrictionCore.set_params!(mb::OnsiteOnlyMatrixModel, θ::NamedTuple)
    ACEfrictionCore.set_params!(mb, :onsite,  θ.onsite)
end

function allocate_matrix(M::OnsiteOnlyMatrixModel, at::Atoms,  T=Float64) 
    N = length(at)
    return [Diagonal(zeros(_block_type(M,T),N)) for _ = 1:M.n_rep]
end

function matrix!(M::OnsiteOnlyMatrixModel{O3S}, at::Atoms, Σ, filter=(_,_)->true) where {O3S}
    site_filter(i,at) = (haskey(M.onsite, at.Z[i]) && filter(i, at))
    for (i, neigs, Rs) in sites(at, env_cutoff(M.onsite))
        if site_filter(i, at) && length(neigs) > 0
            Zs = at.Z[neigs]
            sm = _get_model(M, at.Z[i])
            Σ_temp = evaluate(sm, Rs, Zs)
            for r=1:M.n_rep
                Σ[r][i,i] += _val2block(M, Σ_temp[r].val)
            end
        end
    end
end

function basis(M::OnsiteOnlyMatrixModel, at::Atoms; join_sites=false, filter=(_,_)->true, T=Float64) 
    B = allocate_B(M, at, T)
    basis!(B, M, at, filter)
    return (join_sites ? B[1] : B)
end

function allocate_B(M::OnsiteOnlyMatrixModel, at::Atoms, T=Float64)
    N = length(at)
    B_onsite = [Diagonal( zeros(_block_type(M,T),N)) for _ = 1:length(M.inds,:onsite)]
    return (onsite=B_onsite,)
end

function basis!(B, M::OnsiteOnlyMatrixModel, at::Atoms, filter=(_,_)->true)
    site_filter(i,at) = (haskey(M.onsite, at.Z[i]) && filter(i, at))
    for (i, neigs, Rs) in sites(at, env_cutoff(M.onsite))
        if site_filter(i, at) && length(neigs) > 0
            # evaluate basis of onsite model
            Zs = at.Z[neigs]
            sm = _get_model(M, at.Z[i])
            inds = get_range(M, at.Z[i])
            Bii = evaluate(sm.linmodel.basis, env_transform(Rs, Zs, sm.cutoff))
            for (k,b) in zip(inds,Bii)
                B.onsite[k][i,i] += _val2block(M, b.val)
            end
        end
    end
end

function randf(::OnsiteOnlyMatrixModel, Σ::Diagonal{SMatrix{3, 3, T, 9}}) where {T<:Real}
    return Σ * randn(SVector{3,T},size(Σ,2))
end

function randf(::OnsiteOnlyMatrixModel, Σ::Diagonal{SVector{3, T}}) where {T<:Real}
    return Σ * randn(size(Σ,2))
end

function ACEfrictionCore.write_dict(M::OnsiteOnlyMatrixModel) 
    return Dict("__id__" => "ACEfriction_OnsiteOnlyMatrixModel",
            "onsite" => write_dict(M.onsite),
            #Dict(zz=>write_dict(val) for (zz,val) in M.onsite),
            "id" => string(M.id))         
end
function ACEfrictionCore.read_dict(::Val{:ACEfriction_OnsiteOnlyMatrixModel}, D::Dict)
            onsite = read_dict(D["onsite"])
            #Dict(zz=>read_dict(val) for (zz,val) in D["onsite"])
            id = Symbol(D["id"])
    return OnsiteOnlyMatrixModel(onsite, id)
end