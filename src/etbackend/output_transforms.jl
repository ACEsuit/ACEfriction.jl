# Output transforms for the EquivariantTensors backend.
#
# An ET `SparseACEbasis` built with `LL = (L1, L2, ...)` returns, per call, a
# tuple `BB` of per-L outputs; `BB[i][k]` is the k-th L_i-equivariant basis
# function value as an `SVector{2L_i+1}` (or `Float64` for L=0). The friction
# models need each basis function expressed as a Cartesian block:
#   - scalar invariant  -> 3x3 diagonal block
#   - vector equivariant -> SVector{3}
#   - matrix equivariant -> 3x3 matrix (general or symmetric)
#
# This file maps the ET property choice to the right `LL` and converts the
# per-L outputs into the corresponding Cartesian blocks. The conversions reuse
# ET's own O3 building blocks (`cgmatrix`, `TYVec2CartVec`) and are validated by
# O(3)-equivariance tests (see test/etbackend/).

using StaticArrays
using LinearAlgebra: I
import EquivariantTensors as ET

# ----------------------------------------------------------------------
# Property markers: the O(3) character of a single output block.

abstract type ETProperty end

"""scalar invariant output; realised as a 3x3 diagonal (isotropic) block."""
struct ETInvariant <: ETProperty end
"""vector equivariant output; an `SVector{3}` transforming as `v -> Q v`."""
struct ETVector <: ETProperty end
"""general 3x3 matrix equivariant output transforming as `M -> Q M Qᵀ`."""
struct ETMatrix <: ETProperty end
"""symmetric 3x3 matrix equivariant output (drops the antisymmetric L=1 part)."""
struct ETSymMatrix <: ETProperty end

# The list of equivariance orders L needed to span each output type.
#   3x3 general matrix = L0 ⊕ L1 ⊕ L2 (1+3+5 = 9)
#   3x3 symmetric      = L0 ⊕ L2      (1+5   = 6)
#   3-vector           = L1
#   scalar             = L0
output_LL(::ETInvariant) = (0,)
output_LL(::ETVector)    = (1,)
output_LL(::ETMatrix)    = (0, 1, 2)
output_LL(::ETSymMatrix) = (0, 2)

# Required total-l parity of a basis function for an O(3)-equivariant block.
# A many-body basis function with angular factors (l_1,…,l_k) picks up a sign
# (-1)^Σl under spatial inversion (Yₗᵐ(-r̂) = (-1)ˡ Yₗᵐ(r̂)). Restricting mb_spec to
# a single parity per property makes each Σ block transform as a definite-parity
# object, so the friction tensor Γ = ΣΣᵀ is a true (even) rank-2 tensor:
# Γ(Q·r) = Q·Γ(r)·Qᵀ for all Q ∈ O(3), reflections included.
#   - scalar invariant : even (true scalar, no pseudoscalar)
#   - vector           : odd  (polar vector ⇒ ΣΣᵀ even)
#   - matrix/symmatrix : even (true tensor; the L=1 antisym part is an axial vector)
required_parity(::ETInvariant) = :even
required_parity(::ETVector)    = :odd
required_parity(::ETMatrix)    = :even
required_parity(::ETSymMatrix) = :even

# Parity predicate for a single many-body basis function `bb` (a vector of (n,l)
# factors). `parity === nothing` disables the filter (SO(3)-only selection).
_parity_ok(bb, ::Nothing) = true
_parity_ok(bb, p::Symbol) =
      (p == :even) == iseven(sum(b.l for b in bb; init = 0))

# property <-> string (for serialization recipes)
_property_str(::ETInvariant) = "invariant"
_property_str(::ETVector)    = "vector"
_property_str(::ETMatrix)    = "matrix"
_property_str(::ETSymMatrix) = "symmatrix"

function _property_from_str(s::AbstractString)
   s == "invariant" && return ETInvariant()
   s == "vector"    && return ETVector()
   s == "matrix"    && return ETMatrix()
   s == "symmatrix" && return ETSymMatrix()
   error("unknown property string $s")
end

# ----------------------------------------------------------------------
# Public-facing property aliases (the names ACEfriction exports). Backed by the ET
# markers; the legacy `EuclideanMatrix(Float64)` call form is accepted (the element
# type argument is retained only for backward compatibility and is a no-op).
const Invariant               = ETInvariant
const EuclideanVector         = ETVector
const EuclideanMatrix         = ETMatrix
const SymmetricEuclideanMatrix = ETSymMatrix

ETInvariant(::Type) = ETInvariant()
ETVector(::Type)    = ETVector()
ETMatrix(::Type)    = ETMatrix()
ETSymMatrix(::Type) = ETSymMatrix()

# The Cartesian block type produced for each property.
block_type(::ETInvariant, T = Float64) = SMatrix{3, 3, T, 9}
block_type(::ETVector,    T = Float64) = SVector{3, T}
block_type(::ETMatrix,    T = Float64) = SMatrix{3, 3, T, 9}
block_type(::ETSymMatrix, T = Float64) = SMatrix{3, 3, T, 9}

# ----------------------------------------------------------------------
# Per-L converters from an ET output (SVector{2L+1} / Float64) to a block.

# Permutation taking ET's spherical (1,1)-matrix ordering to Cartesian ordering;
# matches `ET.O3.TYVec2CartMat` (`P * Hy * P'`). ET's real L=1 harmonics are stored
# in m = (-1,0,+1) order ~ (y,z,x); `_Pcart` relabels them to (x,y,z), so that a
# block transforms as Q*M*Q' (Cartesian) rather than D1*M*D1' (spherical basis).
const _Pcart = SMatrix{3, 3}(0, 1, 0, 0, 0, 1, 1, 0, 0)

"""
    ETOutput(property, LL)

Converts the per-L ET outputs into Cartesian blocks for `property`. Besides the
Clebsch-Gordan matrices (matrix case) it precomputes, per L channel, the *dense*
linear map `Tcart[il]::SMatrix{NC, 2L+1}` from the spherical output `y` to the
flattened Cartesian block `vec(block)` (`NC = length(block_type(property))`).
All runtime conversions are then a single small static mat-vec; the same maps
are folded into the model coefficients by the fast evaluators (fasteval.jl).
"""
struct ETOutput{P <: ETProperty, NL, TCG, TT}
   property::P
   LL::NTuple{NL, Int}
   cgs::TCG          # tuple of cgmatrix(1,1,L) for matrix properties; () otherwise
   Tcart::TT         # NTuple{NL, SMatrix{NC, 2L+1, Float64}} spherical -> vec(Cartesian)
end

function ETOutput(property::ETProperty)
   LL = output_LL(property)
   return ETOutput(property, LL)
end

# build for an explicit (possibly reduced) LL — see `build_equivariant_tensor`.
function ETOutput(property::ETProperty, LL::NTuple{N, Int}) where {N}
   cgs = _build_cgs(property, LL)
   Tcart = ntuple(il -> _cart_map(property, cgs, LL, il), N)
   return ETOutput(property, LL, cgs, Tcart)
end

# ET's per-channel output element is a Float64 for L = 0 and an SVector{2L+1}
# otherwise; `_sph_unit(L, m)` is the m-th unit element of that type.
_sph_dim(L::Int) = 2L + 1
_sph_unit(L::Int, m::Int) = L == 0 ? 1.0 :
      SVector{2L + 1, Float64}(ntuple(i -> i == m ? 1.0 : 0.0, 2L + 1))
_svec(x::Number) = SVector(x)
_svec(v::SVector) = v

# dense (NC × 2L+1) matrix of the linear spherical -> vec(Cartesian block) map of
# channel `il`, obtained by pushing unit spherical inputs through `_to_block`.
function _cart_map(property::ETProperty, cgs, LL, il::Int)
   L = LL[il]; D = _sph_dim(L); NC = length(block_type(property))
   cols = ntuple(m -> SVector{NC, Float64}(Tuple(_to_block(property, cgs, L, il, _sph_unit(L, m)))), D)
   return SMatrix{NC, D, Float64}(hcat(cols...))
end

_build_cgs(::Union{ETMatrix, ETSymMatrix}, LL) =
      tuple((ET.O3.cgmatrix(1, 1, L) for L in LL)...)
_build_cgs(::Union{ETInvariant, ETVector}, LL) = ()

"""
    build_equivariant_tensor(property, mb_spec, Rnl_spec, Ylm_spec) -> (tensor, out)

Build the ET equivariant tensor for `property` over `mb_spec`, keeping only the L
channels whose symmetrisation is non-empty. A channel can be empty after parity
selection: e.g. the axial-vector L=1 part of a matrix (even parity) is unbuildable
from 2-body real-harmonic products and needs correlation order ≥ 3. ET cannot
represent an empty channel, so we drop it; `out` is matched to the tensor's actual
LL. With no parity filter all channels in `output_LL(property)` are non-empty, so
this reproduces the previous behaviour.
"""
function build_equivariant_tensor(property::ETProperty, mb_spec, Rnl_spec, Ylm_spec)
   keep = Int[]
   for L in output_LL(property)
      symm, _ = ET.symmetrisation_matrix(L, mb_spec; prune = true, PI = true,
                                         basis = real)
      size(symm, 1) > 0 && push!(keep, L)
   end
   isempty(keep) && error("no non-empty O(3) channels for property " *
                  "'$(_property_str(property))' — increase maxorder/maxdeg/maxl")
   LL = tuple(keep...)
   tensor = ET.sparse_equivariant_tensors(; LL = LL, mb_spec = mb_spec,
                  Rnl_spec = Rnl_spec, Ylm_spec = Ylm_spec, basis = real)
   return tensor, ETOutput(property, LL)
end

# scalar invariant: L=0 only, y is a scalar -> isotropic 3x3 block
_to_block(::ETInvariant, ::Tuple{}, ::Int, il::Int, y) =
      SMatrix{3, 3}(y * I)

# vector: L=1 only, y is SVector{3} (spherical order) -> Cartesian SVector{3}
const _tcv = ET.O3.TYVec2CartVec(real)
_to_block(::ETVector, ::Tuple{}, L::Int, il::Int, y) = _tcv(y)

# matrix / symmetric matrix: per-L block.
#   L=0 -> isotropic, L=2 -> symmetric traceless: via cgmatrix(1,1,L) then perm.
#   L=1 -> antisymmetric part. NB: ET's real `cgmatrix(1,1,1)` is identically zero
#   (the antisymmetric 1⊗1->1 coupling has imaginary CG coeffs that a real-valued
#   cgmatrix cannot represent — so `TYVec2CartMat` only covers the symmetric part).
#   We therefore build the antisymmetric block as the hat-map of the axial vector
#   `TYVec2CartVec(y)` (the same validated L=1 -> Cartesian-vector transform).
function _to_block(::ETMatrix, cgs, L::Int, il::Int, y)
   if L == 1
      v = _tcv(y)        # axial (Cartesian) vector
      return @SMatrix [  zero(eltype(v))  -v[3]              v[2];
                         v[3]              zero(eltype(v))  -v[1];
                        -v[2]              v[1]              zero(eltype(v)) ]
   end
   Hy = SMatrix{3, 3}(cgs[il] * y)
   return _Pcart * Hy * _Pcart'
end

# symmetric-matrix property never has an L=1 channel (LL=(0,2))
function _to_block(::ETSymMatrix, cgs, L::Int, il::Int, y)
   Hy = SMatrix{3, 3}(cgs[il] * y)
   return _Pcart * Hy * _Pcart'
end

# ----------------------------------------------------------------------
# Assemble the full basis from the ET multi-L outputs `BB`.

"""
    assemble_blocks!(blocks, out::ETOutput, BB)

Fill `blocks` (a `Vector{block}` of length `sum(length, BB)`) with the Cartesian
block for every (L, k) basis function, ordered by L channel then within-channel
index. Each block is one static mat-vec with the precomputed `Tcart` map of its
channel (type-stable recursion over the channel tuple). Returns `blocks`.
"""
assemble_blocks!(blocks, out::ETOutput, BB) = _assemble_rec(blocks, 0, out.Tcart, BB)

_assemble_rec(blocks, k0, ::Tuple{}, ::Tuple{}) = blocks
function _assemble_rec(blocks, k0, Tcart::Tuple, BB::Tuple)
   k1 = _fill_blocks!(blocks, k0, Tcart[1], BB[1])
   return _assemble_rec(blocks, k1, Base.tail(Tcart), Base.tail(BB))
end

function _fill_blocks!(blocks::AbstractVector{BT}, k0::Int, Ts::SMatrix, y::AbstractVector) where {BT}
   @inbounds for k in eachindex(y)
      blocks[k0 + k] = BT(Tuple(Ts * _svec(y[k])))
   end
   return k0 + length(y)
end

"""number of basis functions = total over all L channels."""
nblocks(BB) = sum(length, BB)
