# Flattened ET site models.
#
# This realizes the Phase-2 design decision: there is NO LinearACEModel / wrapper
# layer. A site model holds the basis and the coefficient vector `c` directly, and
# the contracted evaluation Σₖ cₖ·Bₖ is a one-liner. `params`/`nparams`/
# `set_params!` operate on `c`. (ACEfrictionCore's LinearACEModel contracts at the
# AA level via ProductEvaluator, but ET has no such fast path and the fitting path
# materializes B anyway, so the wrapper carries no value.)
#
# `c` is `Vector{SVector{NR,Float64}}` where `NR` = number of replicas (n_rep):
# each basis function carries NR linear coefficients (the friction model fits NR
# independent linear maps that are squared/contracted into Γ).

using StaticArrays, LinearAlgebra
import Random

"""
    ETOnsiteModel(basis, c)
    ETOnsiteModel(basis, n_rep::Int)

Flattened onsite site model: the site basis plus per-basis-function coefficients
`c::Vector{SVector{NR,Float64}}` (`NR` = number of replicas). No nested linear
model. Build with an explicit `c`, or with `n_rep` for random initialisation.
"""
mutable struct ETOnsiteModel{NR, P, TB}
   basis::TB
   c::Vector{SVector{NR, Float64}}
end

function ETOnsiteModel(basis::ETFrictionSiteBasis{P},
                       c::Vector{SVector{NR, Float64}}) where {P, NR}
   @assert length(basis) == length(c) "basis length $(length(basis)) ≠ #coeffs $(length(c))"
   return ETOnsiteModel{NR, P, typeof(basis)}(basis, c)
end

ETOnsiteModel(basis::ETFrictionSiteBasis, n_rep::Integer) =
      ETOnsiteModel(basis, rand(SVector{n_rep, Float64}, length(basis)))

n_rep(::ETOnsiteModel{NR}) where {NR} = NR
Base.length(m::ETOnsiteModel) = length(m.basis)
_o3property(m::ETOnsiteModel) = _o3property(m.basis)
block_type(m::ETOnsiteModel, T = Float64) = block_type(m.basis, T)

# ---- params API (operates directly on c) ----
nparams(m::ETOnsiteModel) = length(m.c)
params(m::ETOnsiteModel) = m.c

function set_params!(m::ETOnsiteModel{NR}, c::Vector{SVector{NR, Float64}}) where {NR}
   @assert length(c) == length(m.c)
   copyto!(m.c, c)
   return m
end

# ---- evaluation ----

"""
    evaluate_basis(m, Rs, Zs) -> Vector{block}

The (un-contracted) basis blocks `B` on environment `(Rs, Zs)` — the array that
fills `B` for fitting.
"""
evaluate_basis(m::ETOnsiteModel, Rs, Zs) = evaluate(m.basis, Rs, Zs)

"""
    evaluate(m, Rs, Zs) -> SVector{NR, block}

Contracted output Σ for each replica: `Σ[r] = Σₖ c[k][r]·B[k]`. Replaces the old
`evaluate(linmodel, cfg)`.
"""
function evaluate(m::ETOnsiteModel{NR}, Rs, Zs) where {NR}
   B = evaluate(m.basis, Rs, Zs)
   TB = block_type(m.basis)
   Σ = zero(MVector{NR, TB})
   @inbounds for k in eachindex(B)
      ck = m.c[k]; Bk = B[k]
      for r in 1:NR
         Σ[r] += ck[r] * Bk
      end
   end
   return SVector(Σ)
end
