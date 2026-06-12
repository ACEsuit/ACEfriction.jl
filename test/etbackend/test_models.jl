# Tests for the flattened ET onsite site model (basis + c, no linmodel).
# Run:  julia --project=. test/etbackend/test_models.jl
using Test
import EquivariantTensors as ET
using StaticArrays, LinearAlgebra, Random

include(joinpath(@__DIR__, "..", "..", "src", "etbackend", "etbackend.jl"))
using .ETBackend

Random.seed!(5)
species = [:Cu, :H]; zCu, zH = 29, 1

@testset "ETOnsiteModel (flattened)" begin
   basis = ETBackend.onsite_basis(ETBackend.ETMatrix(), species;
               rcut = 5.0, maxorder = 2, maxdeg = 5, maxl = 3)
   nrep = 3
   m = ETBackend.ETOnsiteModel(basis, nrep)

   @test ETBackend.n_rep(m) == nrep
   @test ETBackend.nparams(m) == length(basis)
   @test length(ETBackend.params(m)) == length(basis)

   # params round-trip
   c0 = [ @SVector(randn(nrep)) for _ in 1:length(basis) ]
   ETBackend.set_params!(m, c0)
   @test ETBackend.params(m) == c0

   # environment
   Nenv = 6
   Rs = [ 3.0 * (r = @SVector(randn(3)); r / norm(r)) * rand() for _ in 1:Nenv ]
   Zs = rand((zCu, zH), Nenv)

   # contracted Σ has NR replicas, each a 3x3 block
   Σ = ETBackend.evaluate(m, Rs, Zs)
   @test length(Σ) == nrep
   @test eltype(Σ) == SMatrix{3,3,Float64,9}

   # Σ equals manual contraction of the basis blocks
   B = ETBackend.evaluate_basis(m, Rs, Zs)
   for r in 1:nrep
      Σr = sum(c0[k][r] * B[k] for k in eachindex(B))
      @test Σ[r] ≈ Σr
   end

   # equivariance: each replica's Σ transforms as Q Σ Q'
   for _ in 1:3
      θ = π*rand(3); Q = ET.O3.Q_from_angles(θ)
      ΣQ = ETBackend.evaluate(m, [Q*r for r in Rs], Zs)
      err = maximum(norm(ΣQ[r] - Q*Σ[r]*Q') for r in 1:nrep)
      @test err < 1e-9
   end

   # onsite friction block Γ_ii = Σ_r Σ[r] Σ[r]' is symmetric PSD and equivariant
   Γ = sum(Σ[r] * Σ[r]' for r in 1:nrep)
   @test norm(Γ - Γ') < 1e-10
   @test minimum(eigvals(Symmetric(Matrix(Γ)))) > -1e-10
   θ = π*rand(3); Q = ET.O3.Q_from_angles(θ)
   ΣQ = ETBackend.evaluate(m, [Q*r for r in Rs], Zs)
   ΓQ = sum(ΣQ[r] * ΣQ[r]' for r in 1:nrep)
   @test norm(ΓQ - Q*Γ*Q') < 1e-9

   println("  nbasis=$(length(basis))  n_rep=$nrep  Γ eigvals=$(round.(eigvals(Symmetric(Matrix(Γ))), sigdigits=3))")
end
