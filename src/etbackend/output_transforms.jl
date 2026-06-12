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

# PERF TODO (production container): instead of conjugating every (L,k) block with
# `_Pcart` at runtime (`_Pcart * Hy * _Pcart'` in `_to_block`), fold `_Pcart` into
# the Clebsch-Gordan matrices once at construction. I.e. precompute `cg̃_L` such
# that `reshape(cg̃_L * y, 3, 3)` is already in Cartesian ordering
# (`cg̃_L = kron(_Pcart, _Pcart) * cgmatrix(1,1,L)`, accounting for column-major
# reshape). Each block then costs a single mat-vec with no per-call conjugation.

"""
    ETOutput(property, LL)

Callable that converts a single per-L ET output `(L, y)` into the Cartesian
block for `property`. Precomputes the Clebsch-Gordan matrices for the matrix
case and the cart-vector transform for the vector case.
"""
struct ETOutput{P <: ETProperty, NL, TCG}
   property::P
   LL::NTuple{NL, Int}
   cgs::TCG          # tuple of cgmatrix(1,1,L) for matrix properties; () otherwise
end

function ETOutput(property::ETProperty)
   LL = output_LL(property)
   cgs = _build_cgs(property, LL)
   return ETOutput(property, LL, cgs)
end

_build_cgs(::Union{ETMatrix, ETSymMatrix}, LL) =
      tuple((ET.O3.cgmatrix(1, 1, L) for L in LL)...)
_build_cgs(::Union{ETInvariant, ETVector}, LL) = ()

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
index. Returns `blocks`.
"""
function assemble_blocks!(blocks, out::ETOutput, BB)
   i = 0
   for (il, L) in enumerate(out.LL)
      BBl = BB[il]
      @inbounds for k in eachindex(BBl)
         i += 1
         blocks[i] = _to_block(out.property, out.cgs, L, il, BBl[k])
      end
   end
   return blocks
end

"""number of basis functions = total over all L channels."""
nblocks(BB) = sum(length, BB)
