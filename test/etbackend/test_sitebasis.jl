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

# a random reflection (improper rotation, det = -1): rotation ∘ mirror
rand_reflection() =
      ET.O3.Q_from_angles(π * rand(3)) * SMatrix{3,3}(Diagonal(SA[-1.0, 1.0, 1.0]))

# max equivariance error of `basis` under transformation `Q` for `prop`
function equiv_err(basis, prop, Rs, Zs, Q)
   B  = ETBackend.evaluate(basis, Rs, Zs)
   BQ = ETBackend.evaluate(basis, [Q*r for r in Rs], Zs)
   return maximum(norm(BQ[k] - transform(prop, Q, B[k])) for k in eachindex(B))
end

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

      # SO(3) equivariance through the whole container
      for _ in 1:3
         Q = ET.O3.Q_from_angles(π * rand(3))
         @test equiv_err(basis, prop, Rs, Zs, Q) < 1e-9
      end

      # O(3): default basis (o3symmetry=true) is also equivariant under reflections
      for _ in 1:3
         @test equiv_err(basis, prop, Rs, Zs, rand_reflection()) < 1e-9
      end

      # symmetric-matrix property -> symmetric blocks
      if prop isa ETBackend.ETSymMatrix
         @test maximum(norm(b - b') for b in B) < 1e-9
      end

      println("  $(typeof(prop)):  nbasis=$(length(basis))")
   end
end

# Regression lock: with o3symmetry=false the basis is only SO(3)-equivariant. It
# stays rotation-equivariant, but parity mixing breaks reflection-equivariance —
# the bug the flag fixes. (At low correlation order some channels are accidentally
# single-parity, so we require breakage for *some* property, not every one.)
@testset "onsite o3symmetry=false is SO(3)-only" begin
   Nenv = 7
   Rs = [ 3.0 * (r = @SVector(randn(3)); r/norm(r)) * rand() for _ in 1:Nenv ]
   Zs = rand((zCu, zH), Nenv)
   props = (ETBackend.ETInvariant(), ETBackend.ETVector(),
            ETBackend.ETMatrix(), ETBackend.ETSymMatrix())
   refl_err = 0.0
   for prop in props
      basis = ETBackend.onsite_basis(prop, species;
                  rcut = 5.0, maxorder = 2, maxdeg = 5, maxl = 3, o3symmetry = false)
      # rotations still hold for every property
      @test equiv_err(basis, prop, Rs, Zs, ET.O3.Q_from_angles(π*rand(3))) < 1e-9
      refl_err = max(refl_err,
                     maximum(equiv_err(basis, prop, Rs, Zs, rand_reflection()) for _ in 1:5))
   end
   # at least one property is genuinely not reflection-equivariant without the filter
   @test refl_err > 1e-6
end
