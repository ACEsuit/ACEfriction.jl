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

# O(3): restrict mb_spec to the property's required total-l parity (the selection
# that `onsite_basis`/`bond_basis` apply when o3symmetry=true), then build the
# tensor + matched output transform, dropping any empty L channel.
function build_tensor(prop)
   spec = filter(bb -> _parity_ok(bb, required_parity(prop)), mb_spec)
   return build_equivariant_tensor(prop, spec, Rnl_spec, Ylm_spec)
end

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

# proper rotations (det +1) and improper rotations / reflections (det -1)
rand_rotation()   = ET.O3.Q_from_angles(π * rand(3))
rand_reflection() = rand_rotation() * SMatrix{3,3}(Diagonal(SA[-1.0, 1.0, 1.0]))

@testset "ET-backend output transforms" begin
   nneig = 6
   Rs = [ @SVector(randn(3)) for _ in 1:nneig ]
   for prop in (ETInvariant(), ETVector(), ETMatrix(), ETSymMatrix())
      tensor, out = build_tensor(prop)
      # full O(3): both proper rotations and reflections
      for Q in (rand_rotation(), rand_rotation(), rand_reflection(), rand_reflection())
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
      println("  $(typeof(prop)):  LL=$(out.LL)  nbasis=$(length(eval_blocks(tensor, out, Rs)))")
   end
end
