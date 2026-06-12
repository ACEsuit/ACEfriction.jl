# Round-trip tests for ET backend serialization (write_dict / read_dict).
# Run:  julia --project=. test/etbackend/test_serialization.jl
using Test
using StaticArrays, LinearAlgebra, Random

include(joinpath(@__DIR__, "..", "..", "src", "etbackend", "etbackend.jl"))
using .ETBackend

Random.seed!(8)
species = [:Cu, :H]; zCu, zH = 29, 1
randenv(N) = ([ (r=@SVector(randn(3)); 3.0*r/norm(r)*rand()) for _ in 1:N ], rand((zCu,zH), N))

@testset "serialization round-trip" begin
   Rs, Zs = randenv(6)

   @testset "onsite basis ($p)" for p in (ETBackend.ETInvariant(), ETBackend.ETVector(),
                                          ETBackend.ETMatrix(), ETBackend.ETSymMatrix())
      b = ETBackend.onsite_basis(p, species; rcut=5.0, maxorder=2, maxdeg=4, maxl=3)
      b2 = ETBackend.read_dict(ETBackend.write_dict(b))
      @test length(b2) == length(b)
      B1 = ETBackend.evaluate(b, Rs, Zs)
      B2 = ETBackend.evaluate(b2, Rs, Zs)
      @test maximum(norm(B1[k] - B2[k]) for k in eachindex(B1)) < 1e-12
   end

   @testset "bond basis (z2=$z2)" for z2 in (:none, :even, :odd)
      b = ETBackend.bond_basis(ETBackend.ETMatrix(), species; z2sym=z2,
                  rcut=1.0, maxorder=2, maxdeg=4, maxl=2)
      b2 = ETBackend.read_dict(ETBackend.write_dict(b))
      @test length(b2) == length(b)
      rrij = (v=@SVector(randn(3)); 0.5*v/norm(v))
      Re, Ze = randenv(4); Re = [0.6*r/norm(r)*rand() for r in Re]
      B1 = ETBackend.evaluate_bond(b, rrij, Re, Ze)
      B2 = ETBackend.evaluate_bond(b2, rrij, Re, Ze)
      @test maximum(norm(B1[k] - B2[k]) for k in eachindex(B1)) < 1e-12
   end

   @testset "onsite model + coefficients" begin
      b = ETBackend.onsite_basis(ETBackend.ETMatrix(), species; rcut=5.0, maxorder=2, maxdeg=4, maxl=3)
      m = ETBackend.ETOnsiteModel(b, 3)
      ETBackend.set_params!(m, [ @SVector(randn(3)) for _ in 1:length(b) ])
      m2 = ETBackend.read_dict(ETBackend.write_dict(m))
      @test ETBackend.n_rep(m2) == 3
      @test ETBackend.params(m2) == ETBackend.params(m)
      Σ1 = ETBackend.evaluate(m, Rs, Zs)
      Σ2 = ETBackend.evaluate(m2, Rs, Zs)
      @test maximum(norm(Σ1[r] - Σ2[r]) for r in 1:3) < 1e-12
   end
   println("  serialization round-trip OK")
end
