# Standalone equivariance tests for the ET-backend output transforms.
# Run with:  julia --project=. test/etbackend/test_output_transforms.jl
using Test
import EquivariantTensors as ET
import Polynomials4ML as P4ML
using StaticArrays, LinearAlgebra, Random

include(joinpath(@__DIR__, "..", "..", "src", "etbackend", "output_transforms.jl"))

Random.seed!(2)

maxn, maxl, ORD, maxlevel = 4, 2, 2, 4
level = bb -> sum(b.n + b.l for b in bb; init = 0)
mb_spec = ET.sparse_nnll_set(; ORD = ORD, minn = 0, maxn = maxn, maxl = maxl,
                               level = level, maxlevel = maxlevel)
rbasis = P4ML.legendre_basis(maxn + 1)
Rnl_spec = P4ML.natural_indices(rbasis)
ybasis = P4ML.real_sphericalharmonics(maxl)
Ylm_spec = P4ML.natural_indices(ybasis)

build_tensor(prop) = ET.sparse_equivariant_tensors(;
      LL = output_LL(prop), mb_spec = mb_spec,
      Rnl_spec = Rnl_spec, Ylm_spec = Ylm_spec, basis = real)

function eval_blocks(tensor, out, Rs)
   rs = norm.(Rs)
   Rnl = P4ML.evaluate(rbasis, rs)
   Ylm = P4ML.evaluate(ybasis, Rs)
   BB = ET.evaluate(tensor, Rnl, Ylm)
   blocks = Vector{block_type(out.property)}(undef, nblocks(BB))
   return assemble_blocks!(blocks, out, BB)
end

# transformation law for each property under Q ∈ O(3)
transform(::ETInvariant, Q, b) = Q * b * Q'   # = b (isotropic)
transform(::ETVector,    Q, b) = Q * b
transform(::ETMatrix,    Q, b) = Q * b * Q'
transform(::ETSymMatrix, Q, b) = Q * b * Q'

@testset "ET-backend output transforms" begin
   nneig = 6
   Rs = [ @SVector(randn(3)) for _ in 1:nneig ]
   for prop in (ETInvariant(), ETVector(), ETMatrix(), ETSymMatrix())
      tensor = build_tensor(prop)
      out = ETOutput(prop)
      for _ in 1:5
         θ = π * rand(3); Q = ET.O3.Q_from_angles(θ)
         RsQ = [ Q * r for r in Rs ]
         B  = eval_blocks(tensor, out, Rs)
         BQ = eval_blocks(tensor, out, RsQ)
         err = maximum(norm(BQ[k] - transform(prop, Q, B[k])) for k in eachindex(B))
         @test err < 1e-10
      end
      # symmetric-matrix property must produce symmetric blocks
      if prop isa ETSymMatrix
         B = eval_blocks(tensor, out, Rs)
         @test maximum(norm(b - b') for b in B) < 1e-10
      end
      println("  $(typeof(prop)):  LL=$(output_LL(prop))  nbasis=$(length(eval_blocks(tensor, out, Rs)))")
   end
end
