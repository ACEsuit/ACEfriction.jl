# Tests for the onsite ET friction site basis (full container pipeline).
# Run:  julia --project=. test/etbackend/test_sitebasis.jl
using Test
import EquivariantTensors as ET
using StaticArrays, LinearAlgebra, Random

include(joinpath(@__DIR__, "..", "..", "src", "etbackend", "etbackend.jl"))
using .ETBackend

Random.seed!(4)

species = [:Cu, :H]
zCu, zH = 29, 1

transform(::ETBackend.ETInvariant, Q, b) = Q * b * Q'   # isotropic
transform(::ETBackend.ETVector,    Q, b) = Q * b
transform(::ETBackend.ETMatrix,    Q, b) = Q * b * Q'
transform(::ETBackend.ETSymMatrix, Q, b) = Q * b * Q'

@testset "ETFrictionSiteBasis (onsite)" begin
   Nenv = 7
   Rs = [ @SVector(randn(3)) for _ in 1:Nenv ]
   Rs = [ 3.0 * r / norm(r) * rand() for r in Rs ]    # inside rcut=5
   Zs = rand((zCu, zH), Nenv)

   for prop in (ETBackend.ETInvariant(), ETBackend.ETVector(),
                ETBackend.ETMatrix(), ETBackend.ETSymMatrix())
      basis = ETBackend.onsite_basis(prop, species;
                  rcut = 5.0, maxorder = 2, maxdeg = 5, maxl = 3)

      B = ETBackend.evaluate(basis, Rs, Zs)
      @test length(B) == length(basis)
      @test eltype(B) == ETBackend.block_type(basis)
      @test length(ETBackend.scaling(basis, 2)) == length(basis)

      # O(3) equivariance through the whole container
      for _ in 1:3
         θ = π * rand(3); Q = ET.O3.Q_from_angles(θ)
         BQ = ETBackend.evaluate(basis, [Q*r for r in Rs], Zs)
         err = maximum(norm(BQ[k] - transform(prop, Q, B[k])) for k in eachindex(B))
         @test err < 1e-9
      end

      # symmetric-matrix property -> symmetric blocks
      if prop isa ETBackend.ETSymMatrix
         @test maximum(norm(b - b') for b in B) < 1e-9
      end

      println("  $(typeof(prop)):  nbasis=$(length(basis))")
   end
end
