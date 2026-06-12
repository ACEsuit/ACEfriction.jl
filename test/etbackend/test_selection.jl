# Tests for the per-species selection controls (CategorySparseBasis features).
# Run:  julia --project=. test/etbackend/test_selection.jl
using Test
using StaticArrays, LinearAlgebra, Random

include(joinpath(@__DIR__, "..", "..", "src", "etbackend", "etbackend.jl"))
using .ETBackend

Random.seed!(13)
species = [:Cu, :H]; zCu, zH = 29, 1
P = ETBackend.ETMatrix()

@testset "selection controls" begin
   base = length(ETBackend.onsite_basis(P, species; rcut=5.0, maxorder=3, maxdeg=6, maxl=3))

   # forbidding H (maxorder 0) shrinks the basis
   noH = length(ETBackend.onsite_basis(P, species; rcut=5.0, maxorder=3, maxdeg=6, maxl=3,
                  species_maxorder_dict = Dict(:H => 0)))
   @test 0 < noH < base

   # requiring at least one H (minorder 1) also shrinks, and is disjoint-ish from noH
   needH = length(ETBackend.onsite_basis(P, species; rcut=5.0, maxorder=3, maxdeg=6, maxl=3,
                   species_minorder_dict = Dict(:H => 1)))
   @test 0 < needH < base
   # "no H" + "≥1 H" partition the unrestricted (per-H-count) basis
   @test noH + needH == base

   # up-weighting a species' category increases its effective degree -> fewer fns
   heavyH = length(ETBackend.onsite_basis(P, species; rcut=5.0, maxorder=3, maxdeg=6, maxl=3,
                    species_weight_cat = Dict(:H => 2.0, :Cu => 1.0)))
   @test heavyH < base

   # weight Dict(:n,:l) and p_sel are accepted and change the count
   wcount = length(ETBackend.onsite_basis(P, species; rcut=5.0, maxorder=3, maxdeg=6, maxl=3,
                    weight = Dict(:n => 1.5, :l => 1.0), p_sel = 2))
   @test wcount != base

   # bond: bond_weight up-weights the bond factor -> fewer bond fns
   bbase = length(ETBackend.bond_basis(P, species; z2sym=:none, rcut=1.0, maxorder=3, maxdeg=6, maxl=2))
   bheavy = length(ETBackend.bond_basis(P, species; z2sym=:none, rcut=1.0, maxorder=3, maxdeg=6, maxl=2,
                    bond_weight = 2.0))
   @test 0 < bheavy < bbase

   # serialization round-trips the selection controls
   b = ETBackend.onsite_basis(P, species; rcut=5.0, maxorder=3, maxdeg=6, maxl=3,
          species_maxorder_dict = Dict(:H => 1), species_weight_cat = Dict(:H => 1.5),
          weight = Dict(:n => 1.0, :l => 1.5), p_sel = 2)
   b2 = ETBackend.read_dict(ETBackend.write_dict(b))
   @test length(b2) == length(b)
   Rs = [ 3.0*(r=@SVector(randn(3)); r/norm(r))*rand() for _ in 1:6 ]; Zs = rand((zCu,zH), 6)
   @test maximum(norm(x-y) for (x,y) in zip(ETBackend.evaluate(b,Rs,Zs), ETBackend.evaluate(b2,Rs,Zs))) < 1e-12

   println("  base=$base noH=$noH needH=$needH heavyH=$heavyH | bond base=$bbase heavy=$bheavy")
end
