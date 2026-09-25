struct PWCMatrixModel{O3S, CUTOFF, Z2S, SC} <: MatrixModel{O3S}
    offsite::OffSiteModels{O3S, Z2S, CUTOFF} where {Z2S, CUTOFF}
    n_rep::Int
    inds::SiteInds
    id::Symbol
    self_images::SelfImagePolicy
    function PWCMatrixModel(offsite::OffSiteModels{O3S, Z2S, CUTOFF}, id::Symbol, sc::SC,
                            self_images::SelfImagePolicy=ExcludeSelfImages()) where {O3S, Z2S, CUTOFF, SC}
        @assert length(unique([_n_rep(mo) for mo in values(offsite)])) == 1
        return new{O3S, CUTOFF, Z2S, SC}(offsite, _n_rep(offsite), SiteInds(_get_basisinds(offsite)), id, self_images)
    end
end

_get_SC(::PWCMatrixModel{O3S, TM, Z2S, SC}) where {O3S, Z2S, TM, SC} = SC
_offsite_cutoff(offsite::OffSiteModels) = first(values(offsite)).cutoff

# ---- Σ assembly (ellipsoid: bond iterator) ----
function matrix(M::PWCMatrixModel{O3S, <:EllipsoidCutoff, Z2S, SC}, at::AbstractSystem;
                filter=(_,_)->true, T=Float64) where {O3S, Z2S, SC}
    _refresh!(M)
    return _pwc_ellipsoid_matrix(M.offsite, SC, M.self_images, M.n_rep, at, filter, T)
end

function _pwc_ellipsoid_matrix(offsite::AbstractDict, ::Type{SC}, self_images, n_rep::Int, at, filter,
                               ::Type{T}) where {SC, T}
    N = length(at); Z = _species(at)
    Is = [Int[] for _=1:n_rep]; Js = [Int[] for _=1:n_rep]
    Vs = [ Vector{block_type(first(values(offsite)).basis, T)}() for _=1:n_rep ]
    for (i, j, rrij, _Js, Rs, Zs) in et_bonds(at, _offsite_cutoff(offsite))
        (filter(i, at) && filter(j, at) && _keep_partner(self_images, i, j)) || continue
        (Zi, Zj) = _mreduce(Z[i], Z[j], SC); haskey(offsite, (Zi, Zj)) || continue
        Σij = evaluate(offsite[(Zi, Zj)], rrij, Rs, Zs)
        for r = 1:n_rep; push!(Is[r], i); push!(Js[r], j); push!(Vs[r], Σij[r]); end
    end
    return [ sparse(Is[r], Js[r], Vs[r], N, N) for r = 1:n_rep ]
end

# ---- Σ assembly (spherical: site iterator, bond = marked neighbour j) ----
# The bonds of a centre share one per-centre bond state (radial / Ylm / A once per
# centre); see etbackend/fasteval.jl.
function matrix(M::PWCMatrixModel{O3S, <:SphericalCutoff, Z2S, SC}, at::AbstractSystem;
                filter=(_,_)->true, T=Float64) where {O3S, Z2S, SC}
    _refresh!(M)
    return _pwc_matrix(M.offsite, SC, M.self_images, M.n_rep, at, filter, T)
end

# function barrier: `offsite` / `self_images` arrive concretely typed (the model
# fields are abstractly typed), so the per-bond calls dispatch statically.
function _pwc_matrix(offsite::AbstractDict, ::Type{SC}, self_images, n_rep::Int, at, filter,
                     ::Type{T}) where {SC, T}
    N = length(at); Z = _species(at)
    Is = [Int[] for _=1:n_rep]; Js = [Int[] for _=1:n_rep]
    Vs = [ Vector{block_type(first(values(offsite)).basis, T)}() for _=1:n_rep ]
    ctrs = _centre_cache(offsite)
    for (i, neigs, Rs) in _sites(at, env_cutoff(offsite))
        (filter(i, at) && length(neigs) > 0) || continue
        Zs = Z[neigs]
        for (j_loc, j) in enumerate(neigs)
            (filter(j, at) && _keep_partner(self_images, i, j)) || continue
            (Zi, Zj) = _mreduce(Z[i], Z[j], SC); haskey(offsite, (Zi, Zj)) || continue
            om = offsite[(Zi, Zj)]
            ctr = _get_centre!(ctrs, om, (Zi, Zj), i, Rs, Zs)
            Σij = bond_sigma(om.fast, ctr, j_loc; partner_in_env = _partner_in_env(om))
            for r = 1:n_rep; push!(Is[r], i); push!(Js[r], j); push!(Vs[r], Σij[r]); end
        end
    end
    return [ sparse(Is[r], Js[r], Vs[r], N, N) for r = 1:n_rep ]
end

# ---- Σ assembly (snowman: symmetrised over both bond ends) ----
# Σ_ij = c·B(sphere_i, bond i→j) + c·B(sphere_j, bond j→i): the diffusion block of a
# pair combines the ACE basis on i's spherical environment (j the bond partner) and
# on j's spherical environment (i the bond partner). Needs both sites' neighbour
# data, so the per-site lists are materialised up front.

# materialise per-site neighbour data (indices + relative vectors) for O(1) lookup
function _site_nb_table(at::AbstractSystem, rcut::Real)
    tab = Dict{Int, Tuple{Vector{Int}, Vector{SVector{3,Float64}}}}()
    for (i, neigs, Rs) in _sites(at, rcut)
        tab[i] = (collect(neigs), collect(Rs))
    end
    return tab
end

# local index of the reverse bond (atom i) in atom j's neighbour list: the neighbour
# equal to `i` whose relative vector matches `-rrij` (matched by vector to pick the
# correct periodic image).
function _reverse_loc(neigs_j::Vector{Int}, Rs_j::Vector{<:SVector{3}}, i::Int, rrij::SVector{3})
    @inbounds for l in eachindex(neigs_j)
        (neigs_j[l] == i && norm(Rs_j[l] + rrij) < 1e-8) && return l
    end
    return nothing
end

# Walk every assembled snowman pair (i,j) exactly once, invoking
# `f(i, j, zz, om, Vij, Vji)` with `zz=(Zi,Zj)`, `om=M.offsite[zz]`, and the two
# per-centre directed-bond values `Vij = eval(sphere_i, bond i→j)`,
# `Vji = eval(sphere_j, bond j→i)`. The value of a directed bond is produced by
# `evalbond(om, zz, c, loc)` (contracted Σ blocks for `matrix`, basis blocks for
# `basis`); `VT` is its type.
#
# With `cache=true` the per-directed-bond evaluation is memoised by
# `(centre, local_index)` so each directed bond is evaluated only once (it is reused
# as `Vij` of pair (i,j) and as `Vji` of pair (j,i)). The key is image-specific (a
# local neighbour index, not an atom pair), which is what makes it correct under
# periodic boundary conditions. Values that depend on the model's coefficients
# (contracted Σ) must be memoised per species-pair model (`per_model=true`): under
# `SpeciesUnCoupled` the pairs (i,j) and (j,i) use different models. Basis vectors
# are model independent (all offsite models share `bb`/`cutoff`), so a single cache
# is valid across all pairs (`per_model=false`). With `cache=false` each value is
# evaluated on demand (the original two-eval path), preserved for cross-checks.
function _foreach_snowman_pair(f, M::PWCMatrixModel{O3S, <:SnowManCutoff, Z2S, SC},
                               at::AbstractSystem, evalbond, ::Type{VT};
                               filter=(_,_)->true, cache::Bool=true, per_model::Bool=true) where {O3S, Z2S, SC, VT}
    return _foreach_snowman_pair(f, M.offsite, SC, M.self_images, at, evalbond, VT, filter, cache, per_model)
end

# function barrier: `offsite` and `self_images` arrive concretely typed here
function _foreach_snowman_pair(f, offsite::AbstractDict, ::Type{SC}, self_images::SelfImagePolicy,
                               at::AbstractSystem, evalbond, ::Type{VT}, filter, cache::Bool,
                               per_model::Bool) where {SC, VT}
    N = length(at); Z = _species(at)
    nb = _site_nb_table(at, env_cutoff(offsite))
    store = Dict{Tuple{Int,Int,Tuple{Int,Int}}, VT}()
    function getV(c::Int, loc::Int, om, zz)
        cache || return evalbond(om, zz, c, loc)
        key = (c, loc, per_model ? zz : (0, 0))
        v = get(store, key, nothing)
        if v === nothing
            v = evalbond(om, zz, c, loc)
            store[key] = v
        end
        return v
    end
    for i = 1:N
        (haskey(nb, i) && filter(i, at)) || continue
        (neigs_i, Rs_i) = nb[i]
        for (j_loc, j) in enumerate(neigs_i)
            # filter before the reverse-bond lookup: also skips self-image bonds (j==i)
            # under ExcludeSelfImages, which have no well-defined reverse end.
            (filter(j, at) && _keep_partner(self_images, i, j)) || continue
            (Zi, Zj) = _mreduce(Z[i], Z[j], SC); haskey(offsite, (Zi, Zj)) || continue
            om = offsite[(Zi, Zj)]
            Vij = getV(i, j_loc, om, (Zi, Zj))                         # sphere at i, bond i→j
            (neigs_j, Rs_j) = nb[j]
            i_loc = _reverse_loc(neigs_j, Rs_j, i, Rs_i[j_loc])
            i_loc === nothing && error("snowman: reverse bond ($j,$i) not found")
            Vji = getV(j, i_loc, om, (Zi, Zj))                         # sphere at j, bond j→i
            f(i, j, (Zi, Zj), om, Vij, Vji)
        end
    end
    return nothing
end

# Per-centre bond states for the snowman walk: every centre is visited both as `i`
# and as the reverse end `j`, so the states are memoised for the whole call, keyed
# by (centre, species pair).
function _snowman_centres(offsite::AbstractDict, at::AbstractSystem)
    Z = _species(at)
    nb = _site_nb_table(at, env_cutoff(offsite))
    ctrs = Dict{Tuple{Int, Tuple{Int,Int}}, centre_type(first(values(offsite)).fast)}()
    function getctr(om, zz, c)
        ctr = get(ctrs, (c, zz), nothing)
        if ctr === nothing
            (neigs_c, Rs_c) = nb[c]
            ctr = bond_centre(om.fast, Rs_c, Z[neigs_c], om.cutoff.rcut; partner_in_env = _partner_in_env(om))
            ctrs[(c, zz)] = ctr
        end
        return ctr
    end
    return getctr
end

function matrix(M::PWCMatrixModel{O3S, <:SnowManCutoff, Z2S, SC}, at::AbstractSystem;
                filter=(_,_)->true, T=Float64, cache::Bool=true) where {O3S, Z2S, SC}
    _refresh!(M)
    return _snowman_matrix(M.offsite, SC, M.self_images, M.n_rep, at, filter, T, cache)
end

function _snowman_matrix(offsite::AbstractDict, ::Type{SC}, self_images, n_rep::Int, at, filter,
                         ::Type{T}, cache::Bool) where {SC, T}
    N = length(at)
    Is = [Int[] for _=1:n_rep]; Js = [Int[] for _=1:n_rep]
    BT = SMatrix{3,3,T,9}; Vs = [ Vector{block_type(first(values(offsite)).basis, T)}() for _=1:n_rep ]
    getctr = _snowman_centres(offsite, at)
    evalΣ(om, zz, c, loc) = bond_sigma(om.fast, getctr(om, zz, c), loc; partner_in_env = _partner_in_env(om))
    VT = SVector{n_rep, block_type(first(values(offsite)).basis)}
    _foreach_snowman_pair(offsite, SC, self_images, at, evalΣ, VT, filter, cache, true) do i, j, zz, om, Σij, Σji
        Σ = _snowman_combine.(Ref(om.cutoff), Σij, Σji)                # combine the two bond ends
        for r = 1:n_rep; push!(Is[r], i); push!(Js[r], j); push!(Vs[r], Σ[r]); end
    end
    return [ sparse(Is[r], Js[r], Vs[r], N, N) for r = 1:n_rep ]
end

# ---- un-contracted basis (ellipsoid) ----
function basis(M::PWCMatrixModel{O3S, <:EllipsoidCutoff, Z2S, SC}, at::AbstractSystem;
               join_sites=false, filter=(_,_)->true, T=Float64) where {O3S, Z2S, SC}
    N = length(at); Z = _species(at); K = length(M.inds, :offsite)
    Is = [Int[] for _=1:K]; Js = [Int[] for _=1:K]; Vs = [_block_type(M,T)[] for _=1:K]
    for (i, j, rrij, _Js, Rs, Zs) in et_bonds(at, _offsite_cutoff(M.offsite))
        (filter(i, at) && filter(j, at) && _keep_partner(M.self_images, i, j)) || continue
        (Zi, Zj) = _mreduce(Z[i], Z[j], SC); haskey(M.offsite, (Zi, Zj)) || continue
        Bij = evaluate_basis(M.offsite[(Zi, Zj)], rrij, Rs, Zs)
        for (k, b) in zip(get_range(M, (Zi, Zj)), Bij); push!(Is[k], i); push!(Js[k], j); push!(Vs[k], b); end
    end
    B = [ sparse(Is[k], Js[k], Vs[k], N, N) for k = 1:K ]
    return (join_sites ? B : (offsite = B,))
end

# ---- un-contracted basis (spherical) ----
function basis(M::PWCMatrixModel{O3S, <:SphericalCutoff, Z2S, SC}, at::AbstractSystem;
               join_sites=false, filter=(_,_)->true, T=Float64) where {O3S, Z2S, SC}
    B = _pwc_basis(M.offsite, SC, M.self_images, M.inds, at, filter, T)
    return (join_sites ? B : (offsite = B,))
end

function _pwc_basis(offsite::AbstractDict, ::Type{SC}, self_images, inds::SiteInds, at, filter,
                    ::Type{T}) where {SC, T}
    N = length(at); Z = _species(at); K = length(inds, :offsite)
    Is = [Int[] for _=1:K]; Js = [Int[] for _=1:K]
    Vs = [ Vector{block_type(first(values(offsite)).basis, T)}() for _=1:K ]
    ctrs = _centre_cache(offsite)
    for (i, neigs, Rs) in _sites(at, env_cutoff(offsite))
        (filter(i, at) && length(neigs) > 0) || continue
        Zs = Z[neigs]
        for (j_loc, j) in enumerate(neigs)
            (filter(j, at) && _keep_partner(self_images, i, j)) || continue
            (Zi, Zj) = _mreduce(Z[i], Z[j], SC); haskey(offsite, (Zi, Zj)) || continue
            om = offsite[(Zi, Zj)]
            ctr = _get_centre!(ctrs, om, (Zi, Zj), i, Rs, Zs)
            Bij = bond_basis_blocks(om.fast.fs, ctr, j_loc; partner_in_env = _partner_in_env(om))
            for (k, b) in zip(get_range(inds, (Zi, Zj)), Bij); push!(Is[k], i); push!(Js[k], j); push!(Vs[k], b); end
        end
    end
    return [ sparse(Is[k], Js[k], Vs[k], N, N) for k = 1:K ]
end

# ---- un-contracted basis (snowman: combine both bond-end spherical evaluations) ----
function basis(M::PWCMatrixModel{O3S, <:SnowManCutoff, Z2S, SC}, at::AbstractSystem;
               join_sites=false, filter=(_,_)->true, T=Float64, cache::Bool=true) where {O3S, Z2S, SC}
    B = _snowman_basis(M.offsite, SC, M.self_images, M.inds, at, filter, T, cache)
    return (join_sites ? B : (offsite = B,))
end

function _snowman_basis(offsite::AbstractDict, ::Type{SC}, self_images, inds::SiteInds, at, filter,
                        ::Type{T}, cache::Bool) where {SC, T}
    N = length(at); K = length(inds, :offsite)
    Is = [Int[] for _=1:K]; Js = [Int[] for _=1:K]
    Vs = [ Vector{block_type(first(values(offsite)).basis, T)}() for _=1:K ]
    getctr = _snowman_centres(offsite, at)
    evalB(om, zz, c, loc) = bond_basis_blocks(om.fast.fs, getctr(om, zz, c), loc; partner_in_env = _partner_in_env(om))
    VT = Vector{block_type(first(values(offsite)).basis)}
    # basis vectors are model independent -> one cache entry per directed bond
    _foreach_snowman_pair(offsite, SC, self_images, at, evalB, VT, filter, cache, false) do i, j, zz, om, Bij, Bji
        for (k, b1, b2) in zip(get_range(inds, zz), Bij, Bji)
            push!(Is[k], i); push!(Js[k], j); push!(Vs[k], _snowman_combine(om.cutoff, b1, b2))
        end
    end
    return [ sparse(Is[k], Js[k], Vs[k], N, N) for k = 1:K ]
end

# Pairwise random force. Each bond {i,j} carries one shared noise `w` and contributes
# `Σ[i,j]·w` to atom i and `Σ[j,i]·w` to atom j. The resulting covariance is exactly
# the pairwise friction tensor Γ = `_square(Σ, ::PWCMatrixModel)`: for a single bond
# the (i,j) sub-block is `[Σ[i,j]; Σ[j,i]] [Σ[i,j]; Σ[j,i]]ᵀ`. A diagonal (periodic
# self-image) entry — present only under IncludeSelfImages — draws its own noise and
# contributes `Σ[i,i]·w`, giving 1·Σ[i,i]Σ[i,i]ᵀ, matching `_square`.
#
# Why an explicit loop (not a vectorized `vec(sum(Σ .* R, dims=2))` over a symmetrized
# noise matrix `R = (sparse(I,J,Rnz) + sparse(J,I,Rnz))/√2`): that symmetrization doubles
# diagonal entries (Rₙₙ added to itself → cov 2I), which would give 2·Σ_ii Σ_iiᵀ instead
# of the required 1× under IncludeSelfImages. The loop is also ~2× faster (it avoids the
# several temporary sparse matrices the vectorized form allocates per call).
function randf(::PWCMatrixModel, Σ::SparseMatrixCSC{SMatrix{3,3,T,9}, TI}) where {T<:Real, TI<:Int}
    f = zeros(SVector{3,T}, size(Σ, 1))
    Is, Js, Vs = findnz(Σ)
    for (i, j, σij) in zip(Is, Js, Vs)
        if i < j
            w = randn(SVector{3,T})
            f[i] += σij * w
            f[j] += Σ[j,i] * w
        elseif i == j
            f[i] += σij * randn(SVector{3,T})
        end
    end
    return f
end

# vector-equivariant case: Σ blocks are SVector{3} and the per-bond noise is scalar
# (the bond sub-block of Γ is the rank-1 `[Σ[i,j]; Σ[j,i]] [Σ[i,j]; Σ[j,i]]ᵀ`).
function randf(::PWCMatrixModel, Σ::SparseMatrixCSC{SVector{3,T}, TI}) where {T<:Real, TI<:Int}
    f = zeros(SVector{3,T}, size(Σ, 1))
    Is, Js, Vs = findnz(Σ)
    for (i, j, σij) in zip(Is, Js, Vs)
        if i < j
            w = randn(T)
            f[i] += σij * w
            f[j] += Σ[j,i] * w
        elseif i == j
            f[i] += σij * randn(T)
        end
    end
    return f
end

# ---- serialization ----
function write_dict(m::OffSiteModel{O3S, Z2S, CUTOFF, NR}) where {O3S, Z2S, CUTOFF, NR}
    return Dict("__id__" => "ACEfriction_OffSiteModel",
                "basis" => write_dict(m.basis),
                "c" => collect(reinterpret(Vector{Float64}, m.c)),
                "n_rep" => NR,
                "z2sym" => string(nameof(Z2S)),
                "cutoff" => write_dict(m.cutoff))
end
function read_dict(::Val{:ACEfriction_OffSiteModel}, D::AbstractDict)
    basis = read_dict(D["basis"]); NR = Int(D["n_rep"])
    c = reinterpret(Vector{SVector{NR,Float64}}, Vector{Float64}(D["c"]))
    z2 = getfield(@__MODULE__, Symbol(D["z2sym"]))()
    cutoff = read_dict(D["cutoff"])
    return OffSiteModel(BondBasis(basis, z2), cutoff, c)
end
function write_dict(offsite::OffSiteModels)
    return Dict("__id__" => "ACEfriction_offsitemodels",
                "vals" => Dict(i => write_dict(v) for (i, v) in enumerate(values(offsite))),
                "z1" => Dict(i => string(_chemical_symbol(zz[1])) for (i, zz) in enumerate(keys(offsite))),
                "z2" => Dict(i => string(_chemical_symbol(zz[2])) for (i, zz) in enumerate(keys(offsite))))
end
read_dict(::Val{:ACEfriction_offsitemodels}, D::AbstractDict) =
        Dict((_atomic_number(Symbol(z1)), _atomic_number(Symbol(z2))) => read_dict(v)
             for (z1, z2, v) in zip(values(D["z1"]), values(D["z2"]), values(D["vals"])))

function write_dict(M::PWCMatrixModel{O3S, CUTOFF, Z2S, SC}) where {O3S, CUTOFF, Z2S, SC}
    return Dict("__id__" => "ACEfriction_PWCMatrixModel",
                "offsite" => write_dict(M.offsite), "sc" => string(nameof(SC)), "id" => string(M.id),
                "self_images" => _self_image_name(M.self_images))
end
function read_dict(::Val{:ACEfriction_PWCMatrixModel}, D::AbstractDict)
    offsite = read_dict(D["offsite"]); sc = getfield(@__MODULE__, Symbol(D["sc"]))()
    si = _self_image_from_dict(D)
    return PWCMatrixModel(offsite, Symbol(D["id"]), sc, si)
end
