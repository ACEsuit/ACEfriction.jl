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
    wss = _workspace_cache(offsite)
    for (i, j, rrij, _Js, Rs, Zs) in et_bonds(at, _offsite_cutoff(offsite))
        (filter(i, at) && filter(j, at) && _keep_partner(self_images, i, j)) || continue
        (Zi, Zj) = _mreduce(Z[i], Z[j], SC); haskey(offsite, (Zi, Zj)) || continue
        om = offsite[(Zi, Zj)]
        Σij = evaluate!(_workspace!(wss, (Zi, Zj), om), om, rrij, Rs, Zs)
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

# Directed-bond values of the snowman pairs, computed centre by centre. For each centre
# `c` the per-centre state is built once (one reusable state per species-pair model,
# refilled for every centre) and the values of all directed bonds c→n that an
# assembled pair needs are evaluated with `evalV(om, ctr, loc)` and stored. A directed
# bond c→n is needed under the model of pair (c,n) (as its first end) and under the
# model of pair (n,c) (as that pair's reverse end); the two differ only under
# `SpeciesUnCoupled`. Only the values are kept, i.e. O(#bonds) memory, never the
# per-centre states. `per_model = false` stores one value per directed bond (basis
# vectors do not depend on the coefficients; all offsite models share the basis).
# Keys are (centre, local neighbour index, model): image-specific, hence correct
# under periodic boundary conditions.
function _snowman_bond_values(offsite::AbstractDict, ::Type{SC}, self_images::SelfImagePolicy,
                              at::AbstractSystem, filter, nb, Z, mode::Symbol, evalV,
                              ::Type{VT}, per_model::Bool) where {SC, VT}
    store = Dict{Tuple{Int,Int,Tuple{Int,Int}}, VT}()
    ctrs = _centre_cache(offsite)
    for c = 1:length(at)
        (haskey(nb, c) && filter(c, at)) || continue
        (neigs_c, Rs_c) = nb[c]
        Zs_c = Z[neigs_c]
        for (loc, n) in enumerate(neigs_c)
            (filter(n, at) && _keep_partner(self_images, c, n)) || continue
            for zz in (_mreduce(Z[c], Z[n], SC), _mreduce(Z[n], Z[c], SC))
                haskey(offsite, zz) || continue
                key = (c, loc, per_model ? zz : (0, 0))
                haskey(store, key) && continue
                om = offsite[zz]
                store[key] = evalV(om, _get_centre!(ctrs, om, zz, c, Rs_c, Zs_c, mode), loc)
            end
        end
    end
    return store
end

# Walk every assembled snowman pair (i,j) exactly once, invoking
# `f(i, j, zz, om, Vij, Vji)` with `zz=(Zi,Zj)`, `om=offsite[zz]`, and the two
# directed-bond values `Vij = V(sphere_i, bond i→j)`, `Vji = V(sphere_j, bond j→i)`
# (both under the pair's model `om`), obtained from `getV(om, zz, centre, loc)`.
function _foreach_snowman_pair(f, offsite::AbstractDict, ::Type{SC}, self_images::SelfImagePolicy,
                               at::AbstractSystem, filter, nb, Z, getV) where {SC}
    for i = 1:length(at)
        (haskey(nb, i) && filter(i, at)) || continue
        (neigs_i, Rs_i) = nb[i]
        for (j_loc, j) in enumerate(neigs_i)
            # filter before the reverse-bond lookup: also skips self-image bonds (j==i)
            # under ExcludeSelfImages, which have no well-defined reverse end.
            (filter(j, at) && _keep_partner(self_images, i, j)) || continue
            zz = _mreduce(Z[i], Z[j], SC); haskey(offsite, zz) || continue
            om = offsite[zz]
            (neigs_j, Rs_j) = nb[j]
            i_loc = _reverse_loc(neigs_j, Rs_j, i, Rs_i[j_loc])
            i_loc === nothing && error("snowman: reverse bond ($j,$i) not found")
            f(i, j, zz, om, getV(om, zz, i, j_loc), getV(om, zz, j, i_loc))
        end
    end
    return nothing
end

# Shared driver of the snowman `matrix` / `basis`: with `cache = true` every directed
# bond is evaluated once (streamed per centre, see `_snowman_bond_values`); with
# `cache = false` each pair evaluates both of its ends from scratch (a fresh per-centre
# state per evaluation; the naive reference path, kept for cross-checks).
function _snowman_walk(f, offsite::AbstractDict, ::Type{SC}, self_images::SelfImagePolicy,
                       at::AbstractSystem, filter, mode::Symbol, evalV, ::Type{VT},
                       per_model::Bool, cache::Bool) where {SC, VT}
    Z = _species(at)
    nb = _site_nb_table(at, env_cutoff(offsite))
    if cache
        store = _snowman_bond_values(offsite, SC, self_images, at, filter, nb, Z, mode, evalV, VT, per_model)
        getV = (om, zz, c, loc) -> store[(c, loc, per_model ? zz : (0, 0))]
    else
        getV = function (om, zz, c, loc)
            (neigs_c, Rs_c) = nb[c]
            ctr = bond_centre(om.fast, Rs_c, Z[neigs_c], om.cutoff.rcut;
                              partner_in_env = _partner_in_env(om), mode = mode)
            return evalV(om, ctr, loc)::VT
        end
    end
    _foreach_snowman_pair(f, offsite, SC, self_images, at, filter, nb, Z, getV)
    return nothing
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
    Vs = [ Vector{block_type(first(values(offsite)).basis, T)}() for _=1:n_rep ]
    evalΣ(om, ctr, loc) = bond_sigma(om.fast, ctr, loc; partner_in_env = _partner_in_env(om))
    VT = SVector{n_rep, block_type(first(values(offsite)).basis)}
    # contracted values depend on the pair's model -> memoised per model
    _snowman_walk(offsite, SC, self_images, at, filter, :sigma, evalΣ, VT, true, cache) do i, j, zz, om, Σij, Σji
        Σ = _snowman_combine.(Ref(om.cutoff), Σij, Σji)                # combine the two bond ends
        for r = 1:n_rep; push!(Is[r], i); push!(Js[r], j); push!(Vs[r], Σ[r]); end
    end
    return [ sparse(Is[r], Js[r], Vs[r], N, N) for r = 1:n_rep ]
end

# ---- un-contracted basis (ellipsoid) ----
function basis(M::PWCMatrixModel{O3S, <:EllipsoidCutoff, Z2S, SC}, at::AbstractSystem;
               join_sites=false, filter=(_,_)->true, T=Float64) where {O3S, Z2S, SC}
    B = _pwc_ellipsoid_basis(M.offsite, SC, M.self_images, M.inds, at, filter, T)
    return (join_sites ? B : (offsite = B,))
end

function _pwc_ellipsoid_basis(offsite::AbstractDict, ::Type{SC}, self_images, inds::SiteInds, at, filter,
                              ::Type{T}) where {SC, T}
    N = length(at); Z = _species(at); K = length(inds, :offsite)
    acc = _BasisAccum{Tuple{Int,Int}, block_type(first(values(offsite)).basis, T)}()
    for (i, j, rrij, _Js, Rs, Zs) in et_bonds(at, _offsite_cutoff(offsite))
        (filter(i, at) && filter(j, at) && _keep_partner(self_images, i, j)) || continue
        (Zi, Zj) = _mreduce(Z[i], Z[j], SC); haskey(offsite, (Zi, Zj)) || continue
        _accum!(acc, (Zi, Zj), i, j, evaluate_basis(offsite[(Zi, Zj)], rrij, Rs, Zs))
    end
    return _assemble(acc, zz -> get_range(inds, zz), K, N)
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
    acc = _BasisAccum{Tuple{Int,Int}, block_type(first(values(offsite)).basis, T)}()
    ctrs = _centre_cache(offsite)
    for (i, neigs, Rs) in _sites(at, env_cutoff(offsite))
        (filter(i, at) && length(neigs) > 0) || continue
        Zs = Z[neigs]
        for (j_loc, j) in enumerate(neigs)
            (filter(j, at) && _keep_partner(self_images, i, j)) || continue
            (Zi, Zj) = _mreduce(Z[i], Z[j], SC); haskey(offsite, (Zi, Zj)) || continue
            om = offsite[(Zi, Zj)]
            ctr = _get_centre!(ctrs, om, (Zi, Zj), i, Rs, Zs, :basis)
            _accum!(acc, (Zi, Zj), i, j, bond_basis_blocks(om.fast, ctr, j_loc; partner_in_env = _partner_in_env(om)))
        end
    end
    return _assemble(acc, zz -> get_range(inds, zz), K, N)
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
    acc = _BasisAccum{Tuple{Int,Int}, block_type(first(values(offsite)).basis, T)}()
    evalB(om, ctr, loc) = bond_basis_blocks(om.fast, ctr, loc; partner_in_env = _partner_in_env(om))
    VT = Vector{block_type(first(values(offsite)).basis)}
    # basis vectors are model independent -> one stored value per directed bond
    _snowman_walk(offsite, SC, self_images, at, filter, :basis, evalB, VT, false, cache) do i, j, zz, om, Bij, Bji
        _accum!(acc, zz, i, j, map((b1, b2) -> _snowman_combine(om.cutoff, b1, b2), Bij, Bji))
    end
    return _assemble(acc, zz -> get_range(inds, zz), K, N)
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
