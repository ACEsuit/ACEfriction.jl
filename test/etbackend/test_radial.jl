# Tests for the fixed species-channel radial basis.
# Run:  julia --project=. test/etbackend/test_radial.jl
using Test
import Polynomials4ML as P4ML
using StaticArrays, LinearAlgebra

include(joinpath(@__DIR__, "..", "..", "src", "etbackend", "etbackend.jl"))
using .ETBackend

@testset "SpeciesRadialBasis" begin
   species = [:Cu, :H]          # atomic numbers 29, 1
   rcut, maxn = 5.0, 4
   rb = ETBackend.RnYlm_radial(species; rcut = rcut, maxn = maxn)
   nR = rb.nR
   NZ = length(species)

   @test nR == maxn + 1
   @test ETBackend.nchannels(rb) == NZ * nR
   @test length(ETBackend.radial_spec(rb)) == NZ * nR

   # channel_info round-trips n in 1:nR and species in zlist
   for ñ in 1:ETBackend.nchannels(rb)
      ci = ETBackend.channel_info(rb, ñ)
      @test 1 <= ci.n <= nR
      @test ci.z in (29, 1)
   end

   # evaluate on a mixed environment
   zCu, zH = 29, 1
   rs = [1.5, 2.0, 3.0, 4.0]
   Zs = [zCu, zH, zCu, zH]
   Rnl = ETBackend.evaluate_batched(rb, rs, Zs)
   @test size(Rnl) == (4, NZ * nR)

   # species-channel structure: a Cu neighbour only populates Cu channels,
   # a H neighbour only H channels (the other species block is exactly zero)
   iCu = ETBackend._z2i(rb, zCu)
   iH  = ETBackend._z2i(rb, zH)
   colsCu = ((iCu-1)*nR+1):(iCu*nR)
   colsH  = ((iH-1)*nR+1):(iH*nR)
   for j in 1:length(rs)
      if Zs[j] == zCu
         @test all(Rnl[j, colsH] .== 0)
         @test any(Rnl[j, colsCu] .!= 0)
      else
         @test all(Rnl[j, colsCu] .== 0)
         @test any(Rnl[j, colsH] .!= 0)
      end
   end

   # envelope: radial values vanish at/after rcut
   Rcut = ETBackend.evaluate_batched(rb, [rcut, rcut + 1.0], [zCu, zCu])
   @test all(abs.(Rcut) .< 1e-10)

   # the Cu-channel values equal the bare radial Pn(x)·env(x)
   rn = ETBackend._rn(rb, 1.5)
   @test Rnl[1, colsCu] ≈ rn

   println("  nR=$nR  NZ=$NZ  nchannels=$(ETBackend.nchannels(rb))")
end

@testset "r0_ratio/rin_ratio scale-free equivalence" begin
   # the radial basis depends on r only through r/rcut: a neighbour at distance r
   # in a spherical env of radius R must give the same radial values as the same
   # neighbour normalised to the unit sphere (r/R) with rcut=1. This is exactly
   # why r0_ratio/rin_ratio mean the same thing for onsite (physical rcut) and
   # offsite (rcut=1 after the ellipsoid->sphere transform).
   R = 5.0
   rb_abs  = ETBackend.RnYlm_radial([:Cu]; rcut = R,   maxn = 4, r0_ratio = 0.4, rin_ratio = 0.04)
   rb_norm = ETBackend.RnYlm_radial([:Cu]; rcut = 1.0, maxn = 4, r0_ratio = 0.4, rin_ratio = 0.04)
   for r in range(0.05*R, 0.99*R; length = 20)
      @test ETBackend._rn(rb_abs, r) ≈ ETBackend._rn(rb_norm, r / R)
   end

   # defaults reproduce the old backend's r0_ratio=0.4, rin_ratio=0.04
   rb_def = ETBackend.RnYlm_radial([:Cu]; rcut = R, maxn = 4)
   @test ETBackend._rn(rb_def, 2.0) ≈ ETBackend._rn(rb_abs, 2.0)
   # rin = rin_ratio*rcut: values for r < rin are clamped (transform constant there)
   @test ETBackend._rn(rb_abs, 0.04*R) ≈ ETBackend._rn(rb_abs, 0.02*R)
   println("  scale-free r/rcut equivalence OK (onsite rcut=$R ≡ offsite rcut=1)")
end
