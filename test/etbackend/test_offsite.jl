# End-to-end test: ET offsite/PWC friction model over a real system.
# Run:  julia --project=. test/etbackend/test_offsite.jl
using Test
import EquivariantTensors as ET
using StaticArrays, LinearAlgebra, Random
import AtomsBuilder: bulk, rattle!

include(joinpath(@__DIR__, "..", "..", "src", "etbackend", "etbackend.jl"))
using .ETBackend

Random.seed!(9)

@testset "ETPWCModel (offsite, end-to-end Γ)" begin
   ec = ETBackend.EllipsoidCutoff(3.0, 4.0, 5.0)
   nrep = 2
   bbasis = ETBackend.bond_basis(ETBackend.ETMatrix(), [:Cu];
               z2sym = :none, rcut = 1.0, maxorder = 2, maxdeg = 4, maxl = 2)
   om = ETBackend.ETOffsiteModel(bbasis, nrep, ec)
   ETBackend.set_params!(om, [ 0.1 .* @SVector(randn(nrep)) for _ in 1:length(bbasis) ])

   M = ETBackend.ETPWCModel(Dict((29, 29) => om); id = :equ_off)
   @test ETBackend.n_rep(M) == nrep

   at = rattle!(bulk(:Cu) * (2, 2, 2), 0.2)
   N = length(at)

   Σ = ETBackend.sigma(M, at)
   @test length(Σ) == nrep
   # bonds were found -> some off-diagonal Σ blocks are nonzero
   nnz = sum(count(!iszero, norm.(Σ[r])) for r in 1:nrep)
   @test nnz > 0
   # Σ is purely off-diagonal (offsite only)
   @test all(iszero(Σ[r][i, i]) for r in 1:nrep for i in 1:N)

   Γ = ETBackend.gamma_dense(M, at)
   @test size(Γ) == (3N, 3N)
   @test norm(Γ - Γ') < 1e-9
   @test minimum(eigvals(Symmetric(Γ))) > -1e-9        # PSD (Γ = ΣΣᵀ)

   println("  N=$N  #nonzero Σ-blocks=$nnz  Γ size=$(size(Γ))  ",
           "min eig=$(round(minimum(eigvals(Symmetric(Γ))), sigdigits=3))  ",
           "max eig=$(round(maximum(eigvals(Symmetric(Γ))), sigdigits=3))")
end

@testset "ETPWCModel (spherical offsite, end-to-end Γ)" begin
   sc = ETBackend.SphericalCutoff(4.0)
   nrep = 2
   bbasis = ETBackend.bond_basis(ETBackend.ETMatrix(), [:Cu];
               z2sym = :none, rcut = 1.0, maxorder = 2, maxdeg = 4, maxl = 2)
   om = ETBackend.ETOffsiteModel(bbasis, nrep, sc)
   ETBackend.set_params!(om, [ 0.1 .* @SVector(randn(nrep)) for _ in 1:length(bbasis) ])

   M = ETBackend.ETPWCModel(Dict((29, 29) => om); id = :equ_off_sph)
   at = rattle!(bulk(:Cu) * (2, 2, 2), 0.2)
   N = length(at)

   Σ = ETBackend.sigma(M, at)
   @test length(Σ) == nrep
   nnz = sum(count(!iszero, norm.(Σ[r])) for r in 1:nrep)
   @test nnz > 0
   @test all(iszero(Σ[r][i, i]) for r in 1:nrep for i in 1:N)   # offsite only

   Γ = ETBackend.gamma_dense(M, at)
   @test size(Γ) == (3N, 3N)
   @test norm(Γ - Γ') < 1e-9
   @test minimum(eigvals(Symmetric(Γ))) > -1e-9                  # PSD

   println("  [spherical] N=$N  #nonzero Σ-blocks=$nnz  ",
           "min eig=$(round(minimum(eigvals(Symmetric(Γ))), sigdigits=3))")
end
