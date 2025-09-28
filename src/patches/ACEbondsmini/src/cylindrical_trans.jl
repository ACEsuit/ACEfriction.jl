function housholderreflection(rr0::SVector{3}) 
   r0 = norm(rr0)
   I3x3 = SMatrix{3,3}(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
   if r0 == 0
      return I3x3
   end
   v = SVector(0, 0, 1) - rr0/r0  # MS: Why is there SVector(0, 0, 1) ? 
   v̂ = v / (norm(v) + eps(eltype(rr0))^2)
   return I3x3 - 2 * v̂ * v̂' 
end

function pullback_housholderreflection(rr0::SVector{3}) 
   H = housholderreflection(rr0)
   dH = ForwardDiff.jacobian(rr -> housholderreflection(rr)[:], rr0)
   dHt = SMatrix{3,9}(dH)'
   return H, g -> SMatrix{3,3}(dHt * SVector{3}(g))
end

_xyz2rθz(ss) = SVector(sqrt(ss[1]^2 + ss[2]^2), 
                       atan(ss[2], ss[1]), 
                       ss[3])

# function _rθz2xyz_d(ss)
#    s, θ, z = _xyz2rθz(ss)
#    sθ, cθ = sincos(θ)
#    return SA[ cθ -s*sθ 0; 
#               sθ  s*cθ 0; 
#                0    0  1 ]
# end

@doc raw"""
This implements the jacobi matrix 
```math
   J = \frac{\partial (r, \theta, z)}{\partial (x, y, z)}
```
"""
function _xyz2rθz_d(ss)
   s, θ, z = _xyz2rθz(ss)
   sθ, cθ = sincos(θ)
   return SA[ cθ    sθ 0; 
              -sθ/s cθ/s 0; 
               0    0 1 ]
end

# this is just a helper function which should not be used for 
# performance critical code 
function eucl2cyl(rr0::SVector, rr::SVector) 
   H = housholderreflection(rr0)
   ss = H * rr 
   r, θ, z = _xyz2rθz(ss)
   return r, θ, z
end

"""
transforms euclidean to cylindral bond environment 

### Input 
* `rr0` : central bond 
* `Rs` : list of environment atom positions, relative to rr0/2 
* `Zs` : associated list of atomic species 

### Output 
* An ACE configuration where each state represents either the bond in terms 
of just its length; or an atom from the environment in terms of its 
cylindrical coordinates (r, θ, z) and species mu, with origin at `rr0/2`.
"""
function eucl2cyl(rrij::SVector, Zi, Zj, 
                  Rs::AbstractVector{<: SVector}, 
                  Zs::AbstractVector{<: AtomicNumber})
   @assert length(Rs) == length(Zs)
   H = housholderreflection(rrij)
   rij = norm(rrij)

   function _eucl2cyl(rr, mu)
      ss = H * rr 
      r, θ, z = _xyz2rθz(ss)
      return State( mu = mu, 
                    r = r, θ = θ, z = z, 
                    rij = rij,  
                    mui = Zi, 
                    muj = Zj, 
                    be = :env )
   end

   Y0 = State( mu = AtomicNumber(0), r = 0.0, θ = 0.0, z = 0.0, 
               rij = rij, mui = Zi, muj = Zj, be = :bond )
   cfg = Vector{typeof(Y0)}(undef, length(Rs)+1)
   cfg[1] = Y0
   for i = 1:length(Rs)
      cfg[i+1] = _eucl2cyl(Rs[i], Zs[i])
   end

   return cfg 
end

# """
# ssj = H * rrj 
# cj = xyz2rθz(ssj)
# ∂_rrj(f) = ∂_cj(f) * ∂_ssj(cj) * ∂_rrj(ssj)
# ∇_rrj(f) = ∂_rrj(ssj)' * ∂_ssj(cj)' * ∇_cj(f) 
# ∇_rrj(f) = ∂_rr0(ssj)' * ∂_ssj(cj)' * ∇_cj(f)
# """
# function rrule_eucl2cyl(rr0::SVector, Zi, Zj, 
#                         Rs::AbstractVector{<: SVector}, 
#                         Zs::AbstractVector{<: AtomicNumber}, 
#                         g_cyl::AbstractMatrix{<: DState})
#    lenB = size(g_cyl, 1)
#    lenR = length(Rs)
#    @assert size(g_cyl, 2) == lenR + 1 
#    H = housholderreflection(rr0)
#    H, pbH = pullback_housholderreflection(rr0)
#    r̂0 = rr0 / norm(rr0) # ∇f(r0) = f'(r0) * r̂0

#    g_Rs = zeros(SVector{3, Float64}, lenB, lenR)
#    # g_cyl[:, 1] = derivative w.r.t. rr0 only
#    g_rr0 = [ g_cyl[n, 1].rij * r̂0 for n = 1:lenB ]

#    for j = 1:lenR
#       rrj = Rs[j] 
#       ss = H * rrj
#       s, θ, z = _xyz2rθz(ss)
#       J = _xyz2rθz_d(ss)
#       for n = 1:lenB
#          gj = g_cyl[n, j+1] 
#          gj1 = J' * SVector(gj.r, gj.θ, gj.z)
#          g_rr0[n] += gj.rij * r̂0 + pbH(gj1)' * rrj
#          g_Rs[n, j] = H' * gj1 
#       end
#    end
   
#    return g_rr0, g_Rs 
# end

# function rrule_eucl2cyl(rr0::SVector, Zi, Zj, 
#                         Rs::AbstractVector{<: SVector}, 
#                         Zs::AbstractVector{<: AtomicNumber}, 
#                         g_cyl::AbstractVector{<: DState})
#    lenR = length(Rs)
#    @assert length(g_cyl) == lenR + 1
#    H = housholderreflection(rr0)
#    H, pbH = pullback_housholderreflection(rr0)
#    r̂0 = rr0 / norm(rr0) # ∇f(r0) = f'(r0) * r̂0

#    g_Rs = zeros(SVector{3, Float64}, lenR)
#    # deriv. of first element w.r.t. rr0 only 
#    g_rr0 = g_cyl[1].rij * r̂0

#    for j = 1:lenR
#       rrj = Rs[j] 
#       ss = H * rrj
#       s, θ, z = _xyz2rθz(ss)
#       J = _xyz2rθz_d(ss)
#       gj = g_cyl[j+1] 
#       gj1 = J' * SVector(gj.r, gj.θ, gj.z)
#       g_rr0 += gj.rij * r̂0 + pbH(gj1)' * rrj
#       g_Rs[j] = H' * gj1 
#    end

#    return g_rr0, g_Rs 
# end




# monkey-patch JuLIP since we aren't really updating it anymore 
Base.isapprox(a::AtomicNumber, b::AtomicNumber) = (a == b)

"""
For testing only. Generates a cylindrical bond environment, and returns it 
both in euclidean and cylindrical coordinates. The origin for both is the 
mid-point of the bond. 
"""
function rand_env(r0cut, rcut, zcut; Nenv = 10, species = [:Al, :Ti])
   species = AtomicNumber.(species)
   r0 = 1 + rand() 
   rr0 = randn(SVector{3, Float64})
   rr0 = r0 * (rr0/norm(rr0))
   H = housholderreflection(rr0)
   Rs = SVector{3, Float64}[] 
   Zs = AtomicNumber[] 
   Xcyl = [] 
   Zi, Zj = rand(species), rand(species)
   for _ = 1:Nenv 
      z = (rand() - 0.5) * 2 * (zcut + r0cut)
      r = rand() * rcut 
      θ = rand() * 2 * π - π
      sθ, cθ = sincos(θ)
      rr = H' * SVector(cθ * r, sθ * r, z)
      mu = rand(species)
      push!(Zs, mu)
      push!(Rs, rr)
      push!(Xcyl, State(mu = mu, r = r, θ=θ, z = z, rij = r0, mui = Zi, muj = Zj, be = :env))
   end 
   return rr0, Zi, Zj, Rs, Zs, Xcyl
end




