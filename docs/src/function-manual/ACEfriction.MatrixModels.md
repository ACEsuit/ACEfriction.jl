```@meta
CurrentModule = ACEfriction
```

```@docs
CWCMatrixModel
```
```@docs
PWCMatrixModel
```
```@docs
OnsiteOnlyMatrixModel
```

The first argument of every constructor is an equivariant *block property*
(`Invariant`, `EuclideanVector`, `EuclideanMatrix`, or `SymmetricEuclideanMatrix`) fixing
the $O(3)$ transformation law of the blocks ${\bm \Sigma}_{ij}$; see
[Equivariant block properties](@ref block-properties). The block sparsity and assembly of
${\bm \Sigma}$ (the *coupling scheme*) are summarised in [Matrix models and coupling
schemes](@ref matrix-models).

### Local environments (cutoffs)

The local environment entering each block is delimited by one of the following cutoffs
(see also [Local environments (cutoffs)](@ref cutoffs)):

```@docs
SphericalCutoff
```
```@docs
EllipsoidCutoff
```
```@docs
SnowManCutoff
```