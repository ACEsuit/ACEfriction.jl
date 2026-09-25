# Flattened ET site models.
#
# This realizes the Phase-2 design decision: there is NO LinearACEModel / wrapper
# layer. A site model holds the basis and the coefficient vector `c` directly.
# `params`/`nparams`/`set_params!` operate on `c`. The contracted evaluation
# Σₖ cₖ·Bₖ goes through the coefficient-folded fast evaluator (`fast`, see
# fasteval.jl), which is re-synced with `c` on every call.
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
mutable struct ETOnsiteModel{NR, P, TB, TF}
   basis::TB
   c::Vector{SVector{NR, Float64}}
   fast::TF                            # ETFastModel: contracted evaluation (fasteval.jl)
end

function ETOnsiteModel(basis::ETFrictionSiteBasis{P},
                       c::Vector{SVector{NR, Float64}}) where {P, NR}
   @assert length(basis) == length(c) "basis length $(length(basis)) ≠ #coeffs $(length(c))"
   fast = ETFastModel(basis, c)
   return ETOnsiteModel{NR, P, typeof(basis), typeof(fast)}(basis, c, fast)
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
evaluate(m::ETOnsiteModel, Rs, Zs) = evaluate(refresh!(m.fast, m.c), Rs, Zs)
