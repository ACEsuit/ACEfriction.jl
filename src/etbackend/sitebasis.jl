# Onsite ET friction site basis: a thin ETACE-style container bundling the
# species-channel radial embedding, the spherical-harmonic embedding, the ET
# equivariant tensor, and the Cartesian output transform. Forward-only (the
# friction fitting path precomputes B and differentiates only over the linear
# coefficients), so no Lux ps/st or ChainRules plumbing is implemented here.
#
# Equivariance is carried by the tensor's LL (derived from the property), not a
# type parameter.

import Polynomials4ML as P4ML
import EquivariantTensors as ET
using StaticArrays, LinearAlgebra

# ----------------------------------------------------------------------
# mb_spec generation (reproduces the role of SparseBasis/CategorySparseBasis).
#
# A many-body basis function is an ordered multiset of (n = ñ, l) factors, where
# ñ is a radial *channel* (species, poly-degree). Selection rules:
#   - correlation order (number of factors)  <= maxorder
#   - weighted total degree  Σ (wn·degₙ(ñ) + wl·l)  <= maxdeg
# This is the standard p=1 ACE total-degree sparsification; `weight` mirrors
# ACEfrictionCore's `Dict(:n=>…, :l=>…)`.

const _NL = NamedTuple{(:n, :l), Tuple{Int, Int}}

# with-replacement combinations of `xs` of length k (non-decreasing index order)
function _wr_combinations(xs, k)
   k == 0 && return [eltype(xs)[]]
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

# normalize a species-keyed Dict (symbols or atomic numbers) to Int atomic numbers
_int_keyed(d::AbstractDict) = Dict{Int,Any}(_atomic_number(k) => v for (k, v) in d)
_wn_wl(weight::AbstractDict) = (Float64(get(weight, :n, 1.0)), Float64(get(weight, :l, 1.0)))

"""
    generate_mb_spec(rbasis, maxl; maxorder, maxdeg, weight, p,
                     weight_cat, minorder_dict, maxorder_dict)

Generate the many-body `(n=ñ, l)` specification over the radial channels of
`rbasis` (a `SpeciesRadialBasis`) and angular degrees `0:maxl`, reproducing
ACEfrictionCore's `CategorySparseBasis` selection:

- per-factor weighted degree `deg(b) = (wₙ·n + wₗ·l) · weight_cat[species(b)]`
  (`weight = Dict(:n=>…, :l=>…)`);
- product level `level(bb) = ‖[deg(b) for b in bb]‖_p`, admissible iff `≤ maxdeg`;
- correlation order `≤ maxorder`, plus per-species limits
  `minorder_dict[z] ≤ #{factors of species z} ≤ maxorder_dict[z]`.

`weight_cat`/`minorder_dict`/`maxorder_dict` may be keyed by element symbol or
atomic number.
"""
function generate_mb_spec(rbasis::SpeciesRadialBasis, maxl::Integer;
                          maxorder::Integer, maxdeg::Real,
                          weight = Dict(:n => 1.0, :l => 1.0), p::Real = 1,
                          weight_cat::AbstractDict = Dict{Int,Float64}(),
                          minorder_dict::AbstractDict = Dict{Int,Int}(),
                          maxorder_dict::AbstractDict = Dict{Int,Int}())
   wn, wl = _wn_wl(weight)
   wcat = _int_keyed(weight_cat)
   mnd = _int_keyed(minorder_dict); mxd = _int_keyed(maxorder_dict)
   zof(ñ) = channel_info(rbasis, ñ).z
   degn(ñ) = channel_info(rbasis, ñ).n - 1
   fdeg(b) = (wn * degn(b.n) + wl * b.l) * Float64(get(wcat, zof(b.n), 1.0))
   level(bb) = isempty(bb) ? 0.0 : norm(Float64[fdeg(b) for b in bb], p)
   function ok_orders(bb)
      for (z, mn) in mnd
         count(b -> zof(b.n) == z, bb) >= mn || return false
      end
      for (z, mx) in mxd
         count(b -> zof(b.n) == z, bb) <= mx || return false
      end
      return true
   end
   factors = _NL[ (n = ñ, l = l)
                  for ñ in 1:nchannels(rbasis) for l in 0:maxl
                  if fdeg((n = ñ, l = l)) <= maxdeg ]
   spec = Vector{_NL}[]
   for k in 1:maxorder, bb in _wr_combinations(factors, k)
      (level(bb) <= maxdeg && ok_orders(bb)) && push!(spec, sort(bb))
   end
   return unique(spec)
end

# ----------------------------------------------------------------------
# Site basis container

"""
    ETFrictionSiteBasis(property, rbasis, ybasis, tensor, out, meta)

Forward-only ET site basis. Build via [`onsite_basis`](@ref). `property` is one
of `ETInvariant`/`ETVector`/`ETMatrix`/`ETSymMatrix`; `rbasis` the species-channel
radial; `ybasis` the P4ML spherical harmonics; `tensor` the ET `SparseACEbasis`;
`out` the cached output transform.
"""
struct ETFrictionSiteBasis{P, TR, TY, TT, TO}
   property::P
   rbasis::TR
   ybasis::TY
   tensor::TT
   out::TO
   meta::Dict{String, Any}
end

Base.length(b::ETFrictionSiteBasis) = sum(b.tensor.lens)
_o3property(b::ETFrictionSiteBasis) = b.property
block_type(b::ETFrictionSiteBasis, T = Float64) = block_type(b.property, T)

# build a serializable recipe Dict for the selection controls
function _selection_recipe(weight, p_sel, weight_cat, minorder_dict, maxorder_dict)
   wn, wl = _wn_wl(weight)
   return Dict{String, Any}(
      "weight" => Dict("n" => wn, "l" => wl), "p_sel" => Float64(p_sel),
      "weight_cat" => Dict{Int,Any}(_int_keyed(weight_cat)),
      "minorder_dict" => Dict{Int,Any}(_int_keyed(minorder_dict)),
      "maxorder_dict" => Dict{Int,Any}(_int_keyed(maxorder_dict)))
end

"""
    onsite_basis(property, species; rcut, maxorder, maxdeg, maxl,
                 weight, p_sel, species_weight_cat,
                 species_minorder_dict, species_maxorder_dict, radial_kwargs...)

Construct an onsite ET friction site basis for the given `property` and `species`
(list of element symbols / atomic numbers). Mirrors the inputs of
ACEfrictionCore's `onsite_linbasis`, including the `CategorySparseBasis` selection
controls (`weight=Dict(:n,:l)`, `p_sel`, per-species `species_weight_cat` /
`species_minorder_dict` / `species_maxorder_dict`).
"""
function onsite_basis(property::ETProperty, species;
                      rcut::Real, maxorder::Integer, maxdeg::Real,
                      maxl::Integer = Int(floor(maxdeg)),
                      weight = Dict(:n => 1.0, :l => 1.0), p_sel::Real = 1,
                      species_weight_cat::AbstractDict = Dict{Int,Float64}(),
                      species_minorder_dict::AbstractDict = Dict{Int,Int}(),
                      species_maxorder_dict::AbstractDict = Dict{Int,Int}(),
                      radial_kwargs...)
   rbasis = RnYlm_radial(species; rcut = rcut, maxn = Int(floor(maxdeg)),
                         radial_kwargs...)
   ybasis = P4ML.real_sphericalharmonics(maxl)
   Ylm_spec = P4ML.natural_indices(ybasis)
   mb_spec = generate_mb_spec(rbasis, maxl; maxorder = maxorder, maxdeg = maxdeg,
                  weight = weight, p = p_sel, weight_cat = species_weight_cat,
                  minorder_dict = species_minorder_dict,
                  maxorder_dict = species_maxorder_dict)
   tensor = ET.sparse_equivariant_tensors(;
            LL = output_LL(property), mb_spec = mb_spec,
            Rnl_spec = radial_spec(rbasis), Ylm_spec = Ylm_spec, basis = real)
   out = ETOutput(property)
   recipe = Dict{String, Any}(
      "kind" => "onsite",
      "property" => _property_str(property),
      "species" => [ _atomic_number(s) for s in species ],
      "rcut" => Float64(rcut), "maxorder" => Int(maxorder),
      "maxdeg" => Float64(maxdeg), "maxl" => Int(maxl),
      "selection" => _selection_recipe(weight, p_sel, species_weight_cat,
                                       species_minorder_dict, species_maxorder_dict),
      "radial" => Dict{String, Any}(string(k) => v for (k, v) in radial_kwargs))
   meta = Dict{String, Any}("recipe" => recipe)
   return ETFrictionSiteBasis(property, rbasis, ybasis, tensor, out, meta)
end

# ----------------------------------------------------------------------
# Forward evaluation

"""
    evaluate(basis, Rs, Zs) -> Vector{block}

Evaluate every basis function on an environment given by neighbour vectors `Rs`
(raw, un-scaled; the radial transform handles the cutoff) and neighbour species
`Zs`. Returns a `Vector` of Cartesian blocks (one per basis function), the array
that fills `B` for fitting and is contracted in `Gamma`.
"""
function evaluate(basis::ETFrictionSiteBasis,
                  Rs::AbstractVector{<:SVector{3}}, Zs::AbstractVector)
   rs = norm.(Rs)
   Rnl = evaluate_batched(basis.rbasis, rs, Zs)
   Ylm = P4ML.evaluate(basis.ybasis, Rs)
   BB = ET.evaluate(basis.tensor, Rnl, Ylm)
   blocks = Vector{block_type(basis)}(undef, nblocks(BB))
   return assemble_blocks!(blocks, basis.out, BB)
end

# empty environment -> all-zero blocks (matches "no neighbours" -> zero basis)
function evaluate(basis::ETFrictionSiteBasis,
                  Rs::AbstractVector{<:SVector{3}}, Zs::AbstractVector,
                  ::Val{:empty})
   z = zero(block_type(basis))
   return fill(z, length(basis))
end

# ----------------------------------------------------------------------
# Regularization scaling: per-basis-function weight based on its (n,l) degree.

"""
    scaling(basis, p) -> Vector

Per-basis-function scaling `(1 + Σ(n+l))^p` over each basis function's (nn,ll)
spec, used as a smoothness prior (mirrors `ACEfrictionCore.scaling`).
"""
function scaling(basis::ETFrictionSiteBasis, p::Real)
   sc = Float64[]
   for il in eachindex(basis.tensor.LL)
      nnll = ET.get_nnll_spec(basis.tensor, il)
      for bb in nnll
         deg = isempty(bb) ? 0 : sum(b.n + b.l for b in bb)
         push!(sc, (1 + deg)^p)
      end
   end
   return sc
end
