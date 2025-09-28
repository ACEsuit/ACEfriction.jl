
function skewedhousholderreflection(rr0::SVector{3}, zc::T, rc::T) where {T<:Real}
    r02 = sum(rr0.^2)
    if r02 == 0
        return SMatrix{3,3}(1.0/rcutbond*I)
    end
    zc_inv, rc_inv = 1.0 ./zc, 1.0 ./ rc 
    return SMatrix{3,3}(rc_inv * I + (zc_inv - rc_inv)/r02 * rr0 * transpose(rr0) )
end

function pullback_skewedhousholderreflection(rr0::SVector{3}, zc::T, rc::T) where {T<:Real}
    H = skewedhousholderreflection(rr0,zc,rc)
    sf(rr) = skewedhousholderreflection(rr,zc,rc)
    dH = ForwardDiff.jacobian(rr -> sf(rr)[:], rr0)
    dHt = SMatrix{3,9}(dH)'
    return H, g -> SMatrix{3,3}(dHt * SVector{3}(g))
 end


function ellipsoid2sphere(rrij::SVector, Zi, Zj, 
    Rs::AbstractVector{<: SVector}, 
    Zs::AbstractVector{<: AtomicNumber}, zcutenv::T, rcutenv::T, rcutbond::T) where {T<:Real}
    @assert length(Rs) == length(Zs)
    G = skewedhousholderreflection(rrij,zcutenv, rcutenv)

    Y0 = State( rr = rrij/rcutbond, mube = :bond) # Atomic species of bond atoms does not matter at this stage.
    cfg = Vector{typeof(Y0)}(undef, length(Rs)+1)
    cfg[1] = Y0
    for i = eachindex(Rs)
        cfg[i+1] = State(rr = G * Rs[i], mube = chemical_symbol(Zs[i]))
    end
    return cfg 
end

# function rrule_ellipsoid2sphere(rr0::SVector, Zi, Zj, 
#     Rs::AbstractVector{<: SVector}, 
#     Zs::AbstractVector{<: AtomicNumber}, 
#     g_ell::AbstractMatrix{<: DState}, 
#     zcutenv::T, rcutenv::T, rcutbond::T) where {T<:Real}
#     lenB = size(g_ell, 1)
#     lenR = length(Rs)
#     @assert size(g_ell, 2) == lenR + 1 
#     H = skewedhousholderreflection(rr0,zcutenv,rcutenv)
#     H, pbH = pullback_skewedhousholderreflection(rr0, zcutenv, rcutenv)

#     g_Rs = zeros(SVector{3, Float64}, lenB, lenR)
#     # g_ell[:, 1] = derivative w.r.t. rr0 only
#     g_rr0 = [ g_ell[n, 1].rr / rcutbond for n = 1:lenB ]

#     for j = 1:lenR
#         rrj = Rs[j] 
#         for n = 1:lenB
#             gj = g_ell[n, j+1].rr
#             g_rr0[n] += pbH(gj)' * rrj
#             g_Rs[n, j] = H' * gj
#         end
#     end

#     return g_rr0, g_Rs 
# end

# function rrule_ellipsoid2sphere(rr0::SVector, Zi, Zj, 
#                 Rs::AbstractVector{<: SVector}, 
#                 Zs::AbstractVector{<: AtomicNumber}, 
#                 g_ell::AbstractVector{<: DState}, 
#                 zcutenv::T, rcutenv::T, rcutbond::T) where {T<:Real}
#     lenR = length(Rs)
#     @assert length(g_ell) == lenR + 1
#     H = skewedhousholderreflection(rr0,zcutenv,rcutenv)
#     H, pbH = pullback_skewedhousholderreflection(rr0, zcutenv, rcutenv)
#     r̂0 = rr0 / norm(rr0) # ∇f(r0) = f'(r0) * r̂0

#     g_Rs = zeros(SVector{3, Float64}, lenR)
#     # deriv. of first element w.r.t. rr0 only 
#     g_rr0 = g_ell[1].rr / rcutbond 
#     for j = 1:lenR
#         rrj = Rs[j] 
#         gj = g_ell[j+1].rr
#         #gj1 = J' * SVector(gj.r, gj.θ, gj.z)
#         #@show gj
#         g_rr0 += pbH(gj)' * rrj
#         g_Rs[j] = H' * gj
#     end

#     return g_rr0, g_Rs 
# end