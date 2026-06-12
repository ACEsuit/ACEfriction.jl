# Fixed species-channel radial embedding for the ET friction backend.
#
#   Rnl[j, ñ] = δ_{z(ñ), Zⱼ} · Pₙ(ñ)(x(rⱼ)) · env(x(rⱼ))
#
# where x = normalized distance transform (generalized Agnesi), P = orthogonal
# polynomials (Polynomials4ML), env = polynomial cutoff envelope. The species is
# folded into the radial channel index ñ (the ACEpotentials "onehot"/ACE-linear
# radial), so no separate categorical 1-particle basis is needed.
#
# This is the *fixed* (non-learnable) specialisation of ACEpotentials'
# `LearnableRnlrzzBasis` with `Winit=:onehot`; friction never trains the radial
# basis (only the linear coefficients are fit), so the learnable 4-tensor machinery
# is omitted. The transform / envelope are ported from ACEpotentials
# (src/models/radial_transforms.jl, radial_envelopes.jl).

import Polynomials4ML as P4ML
using StaticArrays

# ----------------------------------------------------------------------
# Distance transform: generalized Agnesi, normalized to x ∈ [-1, 1].

struct GeneralizedAgnesiTransform{T}
   p::Int
   q::Int
   a::T
   rin::T
   r0::T
end

function (t::GeneralizedAgnesiTransform{T})(r::Number) where {T}
   r <= t.rin && return one(promote_type(T, typeof(r)))
   a, r0, q, p, rin = t.a, t.r0, t.q, t.p, t.rin
   s = (r - rin) / (r0 - rin)
   return 1 / (1 + a * s^q / (1 + s^(q - p)))
end

struct NormalizedTransform{T, TT}
   trans::TT
   yin::T
   ycut::T
   rin::T
   rcut::T
end

function NormalizedTransform(trans, rin::Number, rcut::Number)
   return NormalizedTransform(trans, trans(rin), trans(rcut), rin, rcut)
end

function (t::NormalizedTransform)(r::Number)
   y = t.trans(r)
   𝟙 = one(typeof(y))
   return min(max(-𝟙, -𝟙 + 2 * (y - t.yin) / (t.ycut - t.yin)), 𝟙)
end

"""
    agnesi_transform(r0, rcut, p, q; rin, a)

Generalized Agnesi transform normalized to map `[rin, rcut] -> [-1, 1]`. Default
`a` maximizes `|x'(r)|` at `r = r0`. Recommended `p = 2, q = 2` (the friction /
ACEfrictionCore-style default) or `p = 2, q = 4`.
"""
function agnesi_transform(r0, rcut, p, q;
            rin = zero(r0),
            a = (-2 * q + p * (-2 + 4 * q)) / (p + p^2 + q + q^2))
   @assert q >= p > 0 && a > 0 && 0 <= rin < r0 < rcut
   return NormalizedTransform(GeneralizedAgnesiTransform(p, q, a, rin, r0),
                              rin, rcut)
end

# ----------------------------------------------------------------------
# Envelope in x-space: vanishes (with multiplicity) at x = ±1, i.e. at rin/rcut.

struct PolyEnvelope2sX{T}
   x1::T
   x2::T
   p1::Int
   p2::Int
   s::T
end

function PolyEnvelope2sX(x1, x2, p1::Integer, p2::Integer)
   x1 == x2 && error("x1 and x2 must differ")
   x1 > x2 && ((x1, x2, p1, p2) = (x2, x1, p2, p1))
   x1, x2 = promote(float(x1), float(x2))
   s = 1 / (abs(x2 - x1) / 2)^(p1 + p2)
   return PolyEnvelope2sX(x1, x2, p1, p2, s)
end

function env_val(env::PolyEnvelope2sX, x::T) where {T}
   (env.x1 < x < env.x2) || return zero(T)
   return env.s * (x - env.x1)^env.p1 * (env.x2 - x)^env.p2
end

# ----------------------------------------------------------------------
# Species-channel radial basis.

"""
    SpeciesRadialBasis(zlist, polys, trans, env)

Fixed radial embedding with species folded into the channel index. `zlist` is a
tuple of `Int` atomic numbers; `polys` a Polynomials4ML basis on `[-1,1]`;
`trans` a (normalized) distance transform; `env` an x-space envelope. Produces an
`Rnl` matrix with `NZ * nR` channels (`nR = length(polys)`), channel
`ñ = (iz-1)*nR + n` carrying `δ_{zlist[iz], Zⱼ} · Pₙ(x) · env(x)`.
"""
struct SpeciesRadialBasis{NZ, TP, TT, TE}
   zlist::NTuple{NZ, Int}
   polys::TP
   trans::TT
   env::TE
   nR::Int
end

function SpeciesRadialBasis(zlist, polys, trans, env)
   nR = length(P4ML.natural_indices(polys))
   return SpeciesRadialBasis(Tuple(Int.(zlist)), polys, trans, env, nR)
end

_nz(b::SpeciesRadialBasis{NZ}) where {NZ} = NZ
nchannels(b::SpeciesRadialBasis) = _nz(b) * b.nR
Base.length(b::SpeciesRadialBasis) = nchannels(b)

function _z2i(b::SpeciesRadialBasis, z::Integer)
   @inbounds for i in 1:_nz(b)
      b.zlist[i] == z && return i
   end
   error("species $z not in radial basis zlist $(b.zlist)")
end

"""channel index ñ -> (n, z): underlying polynomial degree-index and species."""
function channel_info(b::SpeciesRadialBasis, ñ::Integer)
   iz = div(ñ - 1, b.nR) + 1
   n = mod1(ñ, b.nR)
   return (n = n, z = b.zlist[iz])
end

"""ET `Rnl_spec`: one `(n = ñ,)` entry per channel."""
radial_spec(b::SpeciesRadialBasis) = [ (n = ñ,) for ñ in 1:nchannels(b) ]

"the per-channel radial values Pₙ(x)·env(x) at transformed coordinate x (length nR)"
function _rn(b::SpeciesRadialBasis, r::Real)
   x = b.trans(r)
   P = P4ML.evaluate(b.polys, x)
   e = env_val(b.env, x)
   return P .* e
end

"""
    evaluate_batched(basis, rs, Zs) -> Rnl

`Rnl[j, ñ] = δ_{z(ñ), Zⱼ} · Pₙ(ñ)(x(rⱼ)) · env(x(rⱼ))`, a
`length(rs) × nchannels(basis)` matrix.
"""
function evaluate_batched(b::SpeciesRadialBasis, rs::AbstractVector, Zs::AbstractVector)
   @assert length(rs) == length(Zs)
   T = promote_type(eltype(rs), Float64)
   Rnl = zeros(T, length(rs), nchannels(b))
   nR = b.nR
   @inbounds for j in eachindex(rs)
      iz = _z2i(b, Zs[j])
      rn = _rn(b, rs[j])
      off = (iz - 1) * nR
      for n in 1:nR
         Rnl[j, off + n] = rn[n]
      end
   end
   return Rnl
end

"""
    RnYlm_radial(species; rcut, maxn, r0_ratio, rin_ratio, pin, pcut, p, q, polys)

Convenience constructor for a `SpeciesRadialBasis` mirroring ACEfrictionCore's
`RnYlm_1pbasis` radial part: Agnesi transform on `[rin, rcut]`, Legendre
polynomials of length `maxn+1`, and a `(pin, pcut)` x-space envelope.

The "typical" radius `r0` and inner cutoff `rin` are given as **scale-free
ratios** of `rcut` (matching ACEfrictionCore's `r0_ratio`/`rin_ratio`):
`r0 = r0_ratio * rcut`, `rin = rin_ratio * rcut`. This keeps the parameterisation
identical for spherical (onsite, `rcut` = physical Å) and ellipsoid (offsite,
`rcut = 1.0` after the ellipsoid→sphere transform) environments: in both cases the
radial basis sees its argument normalised by its own `rcut`, so `r0_ratio` /
`rin_ratio` mean the same thing. Defaults `0.4` / `0.04` match the old backend.
"""
function RnYlm_radial(species;
            rcut, maxn::Integer,
            r0_ratio = 0.4, rin_ratio = 0.04,
            pin = 2, pcut = 2, p = 2, q = 2,
            polys = P4ML.legendre_basis(maxn + 1))
   r0 = r0_ratio * rcut
   rin = rin_ratio * rcut
   trans = agnesi_transform(r0, rcut, p, q; rin = rin)
   env = PolyEnvelope2sX(-1.0, 1.0, pin, pcut)
   zlist = Tuple(_atomic_number(s) for s in species)
   return SpeciesRadialBasis(zlist, polys, trans, env)
end
