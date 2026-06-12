# Prototype: native ET bond basis with Z2 parity grading.
#
# Construction: the bond direction r̂_bond is particle #1 in a pooled config;
# env atoms are particles 2..N. The bond lives in its OWN radial channel, so the
# pooled A over the bond channel receives only the bond particle (=> "bond
# appears exactly once" falls out). Z2 parity = parity of the bond factor's l.
#
# Tests:
#  (1) O(3) equivariance of the assembled 3x3 matrix blocks (Q M Q').
#  (2) Z2: flip r̂_bond -> -r̂_bond (env fixed) => Even blocks unchanged,
#      Odd blocks negated.

import EquivariantTensors as ET
import Polynomials4ML as P4ML
using StaticArrays, LinearAlgebra, Random
Random.seed!(3)

# with-replacement combinations of `xs` of length k (non-decreasing index order)
function with_replacement_combinations(xs, k)
   n = length(xs)
   out = Vector{eltype(xs)}[]
   idx = ones(Int, k)
   while true
      push!(out, [xs[i] for i in idx])
      p = k
      while p >= 1 && idx[p] == n; p -= 1; end
      p == 0 && break
      idx[p] += 1
      for q = p+1:k; idx[q] = idx[p]; end
   end
   return out
end

include(joinpath(@__DIR__, "..", "..", "src", "etbackend", "output_transforms.jl"))

maxn, maxl, maxorder, maxlevel = 3, 2, 3, 4

rbasis = P4ML.legendre_basis(maxn + 1)
ybasis = P4ML.real_sphericalharmonics(maxl)
Ylm_spec = P4ML.natural_indices(ybasis)
nR = length(P4ML.natural_indices(rbasis))         # radial length (= maxn+1)

# radial channels: ñ = 1..nR -> bond ; ñ = nR+1..2nR -> env
Rnl_spec = [ (n = ñ,) for ñ = 1:(2*nR) ]
deg_r(ñ) = (ñ - 1) % nR                            # underlying Legendre degree n
is_bond(ñ) = ñ <= nR

NT = NamedTuple{(:n, :l), Tuple{Int, Int}}
bond_factors = [ NT((ñ, l)) for ñ = 1:nR        for l = 0:maxl if deg_r(ñ)+l <= maxlevel ]
env_factors  = [ NT((ñ, l)) for ñ = (nR+1):(2nR) for l = 0:maxl if deg_r(ñ)+l <= maxlevel ]
fdeg(b) = deg_r(b.n) + b.l

# generate mb_spec with exactly one bond factor + (0..maxorder-1) env factors,
# env factors as sorted multisets; optional parity filter on bond l.
function gen_mb_spec(; even::Union{Nothing,Bool})
   specs = Vector{NT}[]
   for bf in bond_factors
      (!isnothing(even) && iseven(bf.l) != even) && continue
      for k = 0:(maxorder-1)
         combos = k == 0 ? [NT[]] : collect(with_replacement_combinations(env_factors, k))
         for envc in combos
            d = fdeg(bf) + sum(fdeg(e) for e in envc; init = 0)
            d <= maxlevel || continue
            push!(specs, sort(vcat([bf], envc)))
         end
      end
   end
   return unique(specs)
end

build_tensor(spec) = ET.sparse_equivariant_tensors(;
      LL = (0, 1, 2), mb_spec = spec,
      Rnl_spec = Rnl_spec, Ylm_spec = Ylm_spec, basis = real)

function embed(r_bond, Rs_env)
   Rs = vcat([r_bond], Rs_env)
   rn = P4ML.evaluate(rbasis, norm.(Rs))      # Npt x nR
   Ylm = P4ML.evaluate(ybasis, Rs)            # Npt x nY
   Npt = length(Rs)
   Rnl = zeros(Npt, 2nR)
   Rnl[1, 1:nR] .= rn[1, :]                    # bond channels (particle 1)
   for j = 2:Npt
      Rnl[j, nR+1:2nR] .= rn[j, :]            # env channels
   end
   return Rnl, Ylm
end

out = ETOutput(ETMatrix())
function mat_blocks(tensor, r_bond, Rs_env)
   Rnl, Ylm = embed(r_bond, Rs_env)
   BB = ET.evaluate(tensor, Rnl, Ylm)
   blocks = Vector{block_type(out.property)}(undef, nblocks(BB))
   return assemble_blocks!(blocks, out, BB)
end

# environment + bond
Nenv = 5
Rs_env  = [ @SVector(randn(3)) for _ in 1:Nenv ]
r_bond  = @SVector(randn(3))
θ = π*rand(3); Q = ET.O3.Q_from_angles(θ)

for (name, even) in (("Even", true), ("Odd", false))
   spec = gen_mb_spec(; even = even)
   tensor = build_tensor(spec)
   # (1) O(3) equivariance
   B  = mat_blocks(tensor, r_bond, Rs_env)
   BQ = mat_blocks(tensor, Q*r_bond, [Q*r for r in Rs_env])
   err_o3 = isempty(B) ? 0.0 : maximum(norm(BQ[k] - Q*B[k]*Q') for k in eachindex(B))
   # (2) Z2 under bond inversion (env fixed)
   Bflip = mat_blocks(tensor, -r_bond, Rs_env)
   sgn = even ? 1.0 : -1.0
   err_z2 = isempty(B) ? 0.0 : maximum(norm(Bflip[k] - sgn*B[k]) for k in eachindex(B))
   println("$name bond basis:  nspec=$(length(spec))  nbasis=$(length(B))  ",
           "O3 err=$(round(err_o3, sigdigits=3))  Z2 err=$(round(err_z2, sigdigits=3))")
end
