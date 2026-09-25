# Tests for the fast (coefficient-contracted, per-centre) evaluators in
# src/etbackend/fasteval.jl against the generic ET reference path.
# Run:  julia --project=. test/etbackend/test_fasteval.jl
using Test
import EquivariantTensors as ET
using StaticArrays, LinearAlgebra, Random

include(joinpath(@__DIR__, "..", "..", "src", "etbackend", "etbackend.jl"))
using .ETBackend

Random.seed!(11)
species = [:Cu, :H]; zCu, zH = 29, 1

randenv(n, r) = [ r * (v = @SVector(randn(3)); v / norm(v)) * (0.3 + 0.7 * rand()) for _ in 1:n ]
maxblockerr(a, b) = maximum(norm.(a .- b))

# contract a basis block vector with coefficients -> Vector of per-replica blocks
contract(c, B) = [ sum(c[k][r] * B[k] for k in eachindex(B)) for r in eachindex(c[1]) ]

@testset "fast onsite evaluator == generic path" begin
   for prop in (ETBackend.ETMatrix(), ETBackend.ETSymMatrix(), ETBackend.ETVector(), ETBackend.ETInvariant()),
       (maxorder, maxdeg) in ((2, 5), (3, 4))
      basis = ETBackend.onsite_basis(prop, species; rcut = 5.0, maxorder = maxorder, maxdeg = maxdeg)
      nrep = 2
      c = [ @SVector(randn(nrep)) for _ in 1:length(basis) ]
      fm = ETBackend.ETFastModel(basis, c)
      for _ in 1:3
         Rs = randenv(rand(3:12), 5.0); Zs = rand((zCu, zH), length(Rs))
         B = ETBackend.evaluate(basis, Rs, Zs)                    # generic ET path
         Σref = contract(c, B)
         Σ = ETBackend.evaluate(fm, Rs, Zs)                       # fused W·AA
         @test maximum(norm(Σ[r] - Σref[r]) for r in 1:nrep) < 1e-11 * max(1, maximum(norm, Σref))
         # basis blocks from the pooled A reproduce the generic path
         sd = ETBackend.site_data(fm.fs, Rs, Zs)
         @test maxblockerr(ETBackend.basis_from_A(fm.fs, sd.A), B) < 1e-12
      end
      # refresh! tracks coefficient changes
      c2 = [ 2 .* ck for ck in c ]
      ETBackend.refresh!(fm, c2)
      Rs = randenv(6, 5.0); Zs = rand((zCu, zH), 6)
      Σ2 = ETBackend.evaluate(fm, Rs, Zs); Σ1 = contract(c, ETBackend.evaluate(basis, Rs, Zs))
      @test maximum(norm(Σ2[r] - 2 * Σ1[r]) for r in 1:nrep) < 1e-11
   end
end

@testset "per-centre bond evaluator (exact + factorised) == generic path" begin
   rcut = 4.0
   for prop in (ETBackend.ETMatrix(), ETBackend.ETVector()), z2 in (:none, :odd)
      bb = ETBackend.bond_basis(prop, species; z2sym = z2, rcut = 1.0, maxorder = 3, maxdeg = 4)
      nrep = 2
      c = [ @SVector(randn(nrep)) for _ in 1:length(bb) ]
      fm = ETBackend.ETFastModel(bb, c)
      Rs = randenv(9, rcut); Zs = rand((zCu, zH), 9)
      for pie in (false, true)
         sc = ETBackend.SphericalCutoff(rcut; partner_in_env = pie)
         ctr = ETBackend.bond_centre(fm, Rs, Zs, rcut; partner_in_env = pie)
         for j in eachindex(Rs)
            rbond, Rse, Zse = ETBackend.spherical_bond_transform(j, Rs, Zs, sc)
            @test length(Rse) == length(Rs) - (pie ? 0 : 1)        # env convention
            Bref = ETBackend.evaluate_bond(bb, rbond, Rse, Zse)      # generic ET path
            Σref = contract(c, Bref)
            Σ = ETBackend.bond_sigma(fm, ctr, j; partner_in_env = pie)
            @test maximum(norm(Σ[r] - Σref[r]) for r in 1:nrep) < 1e-11 * max(1, maximum(norm, Σref))
            B = ETBackend.bond_basis_blocks(fm.fs, ctr, j; partner_in_env = pie)
            @test maxblockerr(B, Bref) < 1e-12
            # fitting path through the fast model: factorised per-centre basis
            # coefficients when `partner_in_env` (state built in basis mode)
            ctrB = ETBackend.bond_centre(fm, Rs, Zs, rcut; partner_in_env = pie, mode = :basis)
            BF = ETBackend.bond_basis_blocks(fm, ctrB, j; partner_in_env = pie)
            @test maxblockerr(BF, Bref) < 1e-12 * max(1, maximum(norm, Bref))
         end
      end
      # the basis mode never builds the fused (coefficient-dependent) weights
      fmB = ETBackend.ETFastModel(bb, c)
      ETBackend.bond_centre(fmB, Rs, Zs, rcut; partner_in_env = true, mode = :basis)
      w = @atomic fmB.weights
      @test w.S.val === nothing && w.T.val === nothing
      # the two conventions are genuinely different bases
      ctr0 = ETBackend.bond_centre(fm, Rs, Zs, rcut; partner_in_env = false)
      ctr1 = ETBackend.bond_centre(fm, Rs, Zs, rcut; partner_in_env = true)
      @test norm(ETBackend.bond_sigma(fm, ctr0, 1; partner_in_env = false)[1] -
                 ETBackend.bond_sigma(fm, ctr1, 1; partner_in_env = true)[1]) > 1e-6
   end
end

@testset "factorised bond model: O(3) equivariance and Z2" begin
   rcut = 4.0
   bb = ETBackend.bond_basis(ETBackend.ETMatrix(), species; z2sym = :odd, rcut = 1.0, maxorder = 3, maxdeg = 4)
   c = [ @SVector(randn(1)) for _ in 1:length(bb) ]
   fm = ETBackend.ETFastModel(bb, c)
   Rs = randenv(8, rcut); Zs = rand((zCu, zH), 8)
   ctr = ETBackend.bond_centre(fm, Rs, Zs, rcut; partner_in_env = true)
   Σ = [ ETBackend.bond_sigma(fm, ctr, j; partner_in_env = true)[1] for j in eachindex(Rs) ]
   reflection() = ET.O3.Q_from_angles(π * rand(3)) * SMatrix{3,3}(Diagonal(SA[-1.0, 1.0, 1.0]))
   for Q in (ET.O3.Q_from_angles(π * rand(3)), reflection())
      ctrQ = ETBackend.bond_centre(fm, [Q * r for r in Rs], Zs, rcut; partner_in_env = true)
      ΣQ = [ ETBackend.bond_sigma(fm, ctrQ, j; partner_in_env = true)[1] for j in eachindex(Rs) ]
      @test maximum(norm(ΣQ[j] - Q * Σ[j] * Q') for j in eachindex(Rs)) < 1e-10
   end
   # Z2 (odd bond parity): flipping the bond direction with the env fixed negates Σ.
   # With the partner in the env this is only exact if the env is symmetric under the
   # flip, so test it directly on the bond basis with an explicit environment.
   rrij = Rs[1] / rcut; Rse = Rs[2:end] ./ rcut; Zse = Zs[2:end]
   B = ETBackend.evaluate_bond(bb, rrij, Rse, Zse); Bf = ETBackend.evaluate_bond(bb, -rrij, Rse, Zse)
   @test maxblockerr(Bf, -B) < 1e-10
end

@testset "cutoff option round trip" begin
   for c in (ETBackend.SphericalCutoff(3.5; partner_in_env = true), ETBackend.SphericalCutoff(3.5),
             ETBackend.SnowManCutoff(3.5, :antisymmetric; partner_in_env = true), ETBackend.SnowManCutoff(3.5))
      c2 = ETBackend.read_dict(ETBackend.write_dict(c))
      @test typeof(c2) == typeof(c) && c2.rcut == c.rcut && c2.partner_in_env == c.partner_in_env
   end
   # dicts written before the option existed default to the partner-excluded env
   D = Dict{String,Any}("__id__" => "ETBackend_SphericalCutoff", "rcut" => 4.0)
   @test ETBackend.read_dict(D).partner_in_env == false
end
