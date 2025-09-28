
using ACEbonds, ACE, ACEbase, Test, StaticArrays, LinearAlgebra, JuLIP
using ACEbase.Testing
using ACEbonds.BondEnvelopes: CylindricalBondEnvelope
using ACEbonds.BondSelectors: SparseCylindricalBondBasis
using ACE: discrete_jacobi, Rn1pBasis, scal1pbasis, Scal1pBasis,
           evaluate, # evaluate_d, 
           Trig1pBasis, λ, 
           Product1pBasis, Categorical1pBasis, SimpleSparseBasis, 
           SymmetricBasis, Invariant, PIBasis

using ACEbonds.BondCutoffs: housholderreflection, eucl2cyl, rand_env, rrule_eucl2cyl
## basics 

@info("Test housholderreflection")
for _ = 1:15
   rr0 = randn(SVector{3, Float64})
   H = housholderreflection(rr0)
   print_tf(@test( H * rr0 ≈ norm(rr0) * [0,0,1] ))
   print_tf(@test( H' * [0,0,1] ≈ rr0/norm(rr0) ))
   print_tf(@test H' * H ≈ I)
end
println()

##

@info("Test housholderreflection near the poles")
for _ = 1:15
   rr0 = SVector{3}([0, 0, 1]) + 1e-15 * randn(SVector{3, Float64})
   H = housholderreflection(rr0)
   print_tf(@test( H * rr0 ≈ norm(rr0) * [0,0,1] ))
   print_tf(@test( H' * [0,0,1] ≈ rr0/norm(rr0) ))
   print_tf(@test H' * H ≈ I)
end
println()

println_slim(@test( housholderreflection(SVector{3, Float64}([0, 0, 1])) == I ))

##

@info("test transformation from euclidean to cylindrical environments")

# TODO: HACK ACE States to achieve this without type piracy
#       introduce an `ace_isapprox` which defaults to `isapprox`
#       and then overload that one for Base types 
Base.isapprox(s1::Symbol, s2::Symbol) = (s1 == s2)

r0cut = 4.0 
rcut = 4.0 
zcut = 2.0 

for ntest = 1:30 
   rr0, Zi, Zj, Rs, Zs, Xs = rand_env(r0cut, rcut, zcut)
   Xenv = eucl2cyl(rr0, Zi, Zj, Rs, Zs)
   print_tf(@test( all(Xenv[2:end] .≈ Xs) ))
end
println()

##


r0cut = 4.0
rcut = 4.0
zcut = 2.0

maxdeg = 8
maxL = 5 

Jz = discrete_jacobi(maxdeg; pcut = 2, pin = 2, xcut = r0cut+zcut, xin = -r0cut-zcut)
Jr = discrete_jacobi(maxdeg; pcut = 2, pin = 2, xcut = rcut, xin = -rcut)
Jr0 = discrete_jacobi(maxdeg; pcut = 2, pin = 2, xcut = r0cut, xin = -r0cut)

Cbe = Categorical1pBasis([:bond, :env], :be, :be, "Cbe")
Zm = Scal1pBasis(:z, nothing, :m, Jz, "Zm")
Rn = Scal1pBasis(:r, nothing, :n, Jr, "Rn")
El = Trig1pBasis(maxL; varsym = :θ, lsym = :l, label = "El")
Pk = Scal1pBasis(:rij, nothing, :k, Jr0, "Jk")

B1p = Product1pBasis((Cbe, Pk, Rn, El, Zm))


##
B1p = Product1pBasis((Cbe, Pk, Rn, El, Zm))
Bsel = SparseCylindricalBondBasis(; maxorder = 3, 
                                  default_maxdeg = maxdeg, 
                                  weight = Dict{Symbol, Float64}(:m => 1.0, :n => 1.0, :k => 1.0, :l => 1.0), 
                                 )
ACE.init1pspec!(B1p, Bsel)
length(B1p)

##

# basis = PIBasis(B1p, Bsel; isreal=true)
basis = SymmetricBasis(ACE.Invariant(), B1p, ACE.NoSym(), Bsel; isreal=true)
@show length(basis)

pot = ACE.LinearACEModel(basis)
θ = randn(ACE.nparams(pot)) ./ (1:ACE.nparams(pot)).^2
ACE.set_params!(pot, θ)

##

rr0, Zi, Zj, Rs, Zs, Xs = rand_env(r0cut, rcut, zcut)
Xenv = eucl2cyl(rr0, Zi, Zj, Rs, Zs)

evaluate(B1p, Xenv)
b = evaluate(basis, ACEConfig(Xenv))
v = evaluate(pot, ACEConfig(Xenv))
println_slim(@test (dot(ACE.val.(b), θ) ≈ v.val))

##

@info("Test invariance of the new basis")
for ntest = 1:30
   local rr0, Rs, Zs, Xs, Xenv, Zi, Zj
   rr0, Zi, Zj, Rs, Zs, Xs = rand_env(r0cut, rcut, zcut)
   Xenv = eucl2cyl(rr0, Zi, Zj, Rs, Zs)
   B1 = evaluate(basis, ACEConfig(Xenv))
   Q = ACE.Random.rand_rot()
   rr0_Q = Q * rr0 
   Rs_Q = Ref(Q) .* Rs
   Xenv_Q = eucl2cyl(rr0_Q, Zi, Zj, Rs_Q, Zs)
   B2 = evaluate(basis, ACEConfig(Xenv_Q))
   print_tf(@test( B1 ≈ B2 && !all(Xenv_Q .≈ Xenv) ))
end
println()

##

# @info("Check derivatives of basis w.r.t. cylindrical coordinates")
# B = evaluate(basis, ACEConfig(Xenv))
# dB = evaluate_d(basis, ACEConfig(Xenv))
# TDX = ACE.dstate_type(Xenv[1])

# for ntest = 1:30
#    U = [ randn(TDX) for _ = 1:length(Xenv) ]
#    V = randn(length(B)) ./ (1:length(B))

#    F = t -> dot(V, ACE.val.(evaluate(basis, ACEConfig(Xenv + t * U))))
#    dF = t -> ( dB = evaluate_d(basis, ACEConfig(Xenv + t * U)); 
#                ACE.contract(sum(V[i] * dB[i, :] for i = 1:length(V)), U) )
#    F(0.0)
#    dF(0.0)

#    print_tf(@test all(ACEbase.Testing.fdtest(F, dF, 0.0; verbose=false)) )
# end
# println()

## 

# @info("Check derivatives of basis w.r.t. euclidean coordinates")

# for ntest = 1:30 
#    local rr0, Rs, Zs, Xs, Xenv, Zi, Zj 
#    rr0, Zi, Zj, Rs, Zs, Xs = rand_env(r0cut, rcut, zcut)
#    Us = randn(SVector{3, Float64}, length(Rs))
#    uu0 = randn(SVector{3, Float64})
#    V = randn(length(B)) ./ (1:length(B))

#    julip2ace = t -> ACEConfig(eucl2cyl(rr0 + t * uu0, Zi, Zj, Rs + t * Us, Zs))
#    F = t -> dot(V, ACE.val.(evaluate(basis, julip2ace(t))))

#    dF = t -> begin
#          dB = evaluate_d(basis, julip2ace(t))
#          dB0, dBenv = rrule_eucl2cyl(rr0, Zi, Zj, Rs, Zs, dB)
#          ACE.contract( sum(V[i] * dBenv[i, :] for i = 1:length(V)), Us) + 
#                   dot( sum(V[i] * dB0[i] for i = 1:length(V)), uu0 )
#       end

#    print_tf(@test all(ACEbase.Testing.fdtest(F, dF, 0.0; verbose=false)) )
# end
# println()


##

# @info("Check derivatives of a potential w.r.t. cylindrical coordinates")
# v = evaluate(pot, ACEConfig(Xenv))
# dv = ACE.grad_config(pot, ACEConfig(Xenv))
# TDX = ACE.dstate_type(Xenv[1])

# for ntest = 1:30
#    U = [ randn(TDX) for _ = 1:length(Xenv) ]

#    F = t -> evaluate(pot, ACEConfig(Xenv + t * U)).val
#    dF = t -> ( dv = ACE.grad_config(pot, ACEConfig(Xenv + t * U)); 
#                ACE.contract(dv, U) )
#    F(0.0)
#    dF(0.0)

#    print_tf(@test all(ACEbase.Testing.fdtest(F, dF, 0.0; verbose=false)) )
# end
# println()

##

# @info("Check derivatives of potential w.r.t. euclidean coordinates")

# for ntest = 1:30 
#    local rr0, Rs, Zs, Xs, Xenv, Zi, Zj 
#    rr0, Zi, Zj, Rs, Zs, Xs = rand_env(r0cut, rcut, zcut)
#    Us = randn(SVector{3, Float64}, length(Rs))
#    uu0 = randn(SVector{3, Float64})

#    julip2ace = t -> ACEConfig(eucl2cyl(rr0 + t * uu0, Zi, Zj, Rs + t * Us, Zs))
#    F = t -> evaluate(pot, julip2ace(t)).val

#    dF = t -> begin
#          dv_cyl = ACE.grad_config(pot, julip2ace(t))
#          dv0, dvenv = rrule_eucl2cyl(rr0, Zi, Zj, Rs, Zs, dv_cyl)
#          ACE.contract(dvenv, Us) + dot(dv0, uu0 )
#       end

#    print_tf(@test all(ACEbase.Testing.fdtest(F, dF, 0.0; verbose=false)) )
# end
# println()
