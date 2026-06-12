# End-to-end test: a runnable ET onsite-only friction model over a real system.
# Run:  julia --project=. test/etbackend/test_friction_models.jl
using Test
import EquivariantTensors as ET
using StaticArrays, LinearAlgebra, Random
import AtomsBuilder: bulk, rattle!

include(joinpath(@__DIR__, "..", "..", "src", "etbackend", "etbackend.jl"))
using .ETBackend

Random.seed!(6)

@testset "ETOnsiteOnlyModel (end-to-end Γ)" begin
   rcut, nrep = 5.0, 2
   basis = ETBackend.onsite_basis(ETBackend.ETMatrix(), [:Cu];
               rcut = rcut, maxorder = 2, maxdeg = 5, maxl = 3)
   sm = ETBackend.ETOnsiteModel(basis, nrep)
   # small random coefficients so blocks are well-scaled
   ETBackend.set_params!(sm, [ 0.1 .* @SVector(randn(nrep)) for _ in 1:length(basis) ])

   M = ETBackend.ETOnsiteOnlyModel(Dict(29 => sm); id = :equ)
   @test ETBackend.n_rep(M) == nrep

   # build a real Cu system
   at = rattle!(bulk(:Cu) * (2, 2, 2), 0.2)
   N = length(at)
   @info "system has $N atoms"

   Σ = ETBackend.sigma(M, at)
   @test length(Σ) == nrep
   @test all(length(Σ[r]) == N for r in 1:nrep)

   Γb = ETBackend.gamma(M, at)
   @test length(Γb) == N

   # each onsite block is symmetric PSD
   for i in 1:N
      @test norm(Γb[i] - Γb[i]') < 1e-10
      @test minimum(eigvals(Symmetric(Matrix(Γb[i])))) > -1e-9
   end

   # dense 3N×3N friction matrix is symmetric PSD
   Γ = ETBackend.gamma_dense(M, at)
   @test size(Γ) == (3N, 3N)
   @test norm(Γ - Γ') < 1e-10
   @test minimum(eigvals(Symmetric(Γ))) > -1e-9

   println("  N=$N  Γ size=$(size(Γ))  min eig=$(round(minimum(eigvals(Symmetric(Γ))), sigdigits=3))  ",
           "max eig=$(round(maximum(eigvals(Symmetric(Γ))), sigdigits=3))")
end
