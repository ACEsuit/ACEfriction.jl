# Tests for the production native bond basis (bond channel + Z2 parity).
# Run:  julia --project=. test/etbackend/test_bondbasis.jl
using Test
import EquivariantTensors as ET
using StaticArrays, LinearAlgebra, Random

include(joinpath(@__DIR__, "..", "..", "src", "etbackend", "etbackend.jl"))
using .ETBackend

Random.seed!(7)
species = [:Cu, :H]; zCu, zH = 29, 1

# matrix-property transform law
matT(Q, b) = Q * b * Q'

@testset "bond_basis (native Z2)" begin
   Nenv = 5
   Rs_env = [ 0.6 * (r = @SVector(randn(3)); r/norm(r)) * rand() for _ in 1:Nenv ]
   Zs_env = rand((zCu, zH), Nenv)
   rrij   = @SVector(randn(3)); rrij = 0.5 * rrij / norm(rrij)

   for z2 in (:none, :even, :odd)
      basis = ETBackend.bond_basis(ETBackend.ETMatrix(), species;
                  z2sym = z2, rcut = 1.0, maxorder = 3, maxdeg = 5, maxl = 2)
      B = ETBackend.evaluate_bond(basis, rrij, Rs_env, Zs_env)
      @test length(B) == length(basis) > 0

      # O(3) equivariance (rotate bond + env together)
      for _ in 1:3
         θ = π*rand(3); Q = ET.O3.Q_from_angles(θ)
         BQ = ETBackend.evaluate_bond(basis, Q*rrij, [Q*r for r in Rs_env], Zs_env)
         err = maximum(norm(BQ[k] - matT(Q, B[k])) for k in eachindex(B))
         @test err < 1e-9
      end

      # Z2 under bond inversion (env fixed): even unchanged, odd negated
      if z2 != :none
         Bflip = ETBackend.evaluate_bond(basis, -rrij, Rs_env, Zs_env)
         sgn = z2 == :even ? 1.0 : -1.0
         @test maximum(norm(Bflip[k] - sgn*B[k]) for k in eachindex(B)) < 1e-9
      end

      println("  z2sym=$z2:  nbasis=$(length(basis))")
   end

   # even ⊕ odd should partition the no-symmetry basis count
   nnone = length(ETBackend.bond_basis(ETBackend.ETMatrix(), species; z2sym=:none,
                     rcut=1.0, maxorder=3, maxdeg=5, maxl=2))
   neven = length(ETBackend.bond_basis(ETBackend.ETMatrix(), species; z2sym=:even,
                     rcut=1.0, maxorder=3, maxdeg=5, maxl=2))
   nodd  = length(ETBackend.bond_basis(ETBackend.ETMatrix(), species; z2sym=:odd,
                     rcut=1.0, maxorder=3, maxdeg=5, maxl=2))
   @test nnone == neven + nodd
end
