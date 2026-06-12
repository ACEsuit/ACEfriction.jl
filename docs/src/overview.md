## Model Overview

The package `ACEfriction` provides an ACE-based implementation of the size-transferrable, E(3)-equivariant models introduced in [Sachs et al., (2024)](@ref ACEfriction-paper) for configuration-dependent friction or diffusion tensors.

In a nutshell, the package provides utilities to efficiently learn and evaluate E(3)-equivariant symmetric positive semi-definite matrix functions of the form
```math
{\bm \Gamma} \left ( ({\bm r}_{i},z_i)_{i=1}^{N_{\rm at}} \right ) \in \mathbb{R}^{3 N_{\rm at} \times 3N_{\rm at}},
```
where the ${\bm r}_{i}$s are the positions and the $z_{i}$s are the atomic element types of atoms in an atomic configuration comprised of $N_{\rm at}$ atoms.

The underlying model is based on an equivariance-preserving matrix square root decomposition,
```math
{\bm \Gamma} = {\bm \Sigma}{\bm \Sigma}^T,
```
where block entries of the matrix square root ${\bm \Sigma}\left ( ({\bm r}_{i},z_i)_{i=1}^{N_{\rm at}} \right ) \in \mathbb{R}^{3 N_{\rm at} \times m}$ with some $m \in \mathbb{N}$, are linearly expanded using an equivariant linear atomic cluster expansion. The identity ${\bm \Gamma}={\bm \Sigma}{\bm \Sigma}^{T}$ holds verbatim for the onsite/column-wise coupling schemes; for the pair-wise coupling the friction tensor is assembled block-wise from ${\bm \Sigma}$ via a pair-coupled fluctuation–dissipation relation (equivalently, ${\bm \Gamma}=\tilde{\bm \Sigma}\tilde{\bm \Sigma}^{T}$ for a transformed square root $\tilde{\bm \Sigma}$). See [Assembly of the friction tensor](@ref gamma-assembly).

## Code Overview

The package `ACEfriction` is comprised of three main sub-modules:

1. The sub-module `FrictionModels` implements the structure `FrictionModel`, which facilitates the specification of and evaluation of friction models. The module implements the functions `Gamma(fm::FrictionModel, at::Atoms)`, `Sigma(fm::FrictionModel, at::Atoms)` which evaluate the friction model `fm` at the atomic configuration `at` to the correspong friction tensor ${\bm \Gamma}$ and  diffusion coefficient matrix ${\bm \Sigma}$, respectively. Moreover, it provides the functions `Gamma(fm::FrictionModel, Σ)`, `randf(fm::FrictionModel, Σ)` for efficient computation of the friction tensor and generation of ${\rm Normal}({\bm 0}, {\bm \Gamma})$-distributed Gaussian random numbers from a precomputed diffusion coeffiient matrix `Σ`.

2. The sub-module `MatrixModels` implements various matrix models, which make up a friction model and, in essence, specify (i) properties of the ACE-basis used to evaluate blocks ${\bm \Sigma}_{ij}$ of the difffusion matrix, and (ii) how blocks  ${\bm \Sigma}_{ij}$ are combined in the assembly of the friction tensor ${\bm \Gamma}$. The assembly of the friction tensor is governed by what is referred to in [Sachs et al., (2024)](@ref ACEfriction-paper) as the coupling scheme and implements versions of the the pair-wise coupling and column-wise coupling described therein (the latter is named *row-wise coupling* in Appendix C of the paper; see the warning in [Matrix models and coupling schemes](@ref matrix-models)).

3. The sub-module `FrictionFit` provides utility functions for training of friction models using the julia machine learning library `Flux.jl`. 

!!! note "ACE backend"
    As of the current version, the equivariant ACE basis used to expand the blocks ${\bm \Sigma}_{ij}$ is provided by [`EquivariantTensors.jl`](https://github.com/ACEsuit/EquivariantTensors.jl) (the shared ACEsuit kernel), and atomic configurations are represented through the [`AtomsBase.jl`](https://github.com/JuliaMolSim/AtomsBase.jl) interface. This replaces the former in-house `ACEfrictionCore` backend; the user-facing API documented here is unchanged in spirit, but model files saved with the old backend are not loadable and must be re-fit.

## [Equivariant block properties](@id block-properties)

Each matrix model fixes the equivariance symmetry of its blocks through a *property* type, passed as the first argument to the model constructors. Writing $Q \in O(3)$ for an orthogonal transformation acting on the input configuration, the corresponding $3\times 3$ block ${\bm \Sigma}_{ij}$ transforms as follows:

| Property | Block shape | Transformation law |
|---|---|---|
| `Invariant` | isotropic, ${\bm \Sigma}_{ij} = s\,{\bm I}_3$ | $s \mapsto s$ |
| `EuclideanVector` | vector ${\bm v} \in \mathbb{R}^3$ | ${\bm v} \mapsto Q{\bm v}$ |
| `EuclideanMatrix` | general $3\times3$ | ${\bm M} \mapsto Q{\bm M}Q^{T}$ |
| `SymmetricEuclideanMatrix` | symmetric $3\times3$, ${\bm M}={\bm M}^{T}$ | ${\bm M} \mapsto Q{\bm M}Q^{T}$ |

Internally these are realised by decomposing a block into its $O(3)$-irreducible parts: a general matrix carries the angular momenta $L\in\{0,1,2\}$ (with $9 = 1 + 3 + 5$), a symmetric matrix only $L\in\{0,2\}$ (it drops the antisymmetric $L=1$ component), a vector $L=1$, and an invariant $L=0$. The ACE expansion produces the equivariant $Y$-vectors for each $L$, which are mapped to Cartesian blocks via the standard real Clebsch–Gordan/coupling coefficients.

## [Matrix models and coupling schemes](@id matrix-models)

A friction model is a sum over one or more *matrix models*. Each matrix model specifies (i) an equivariant ACE expansion of the blocks ${\bm \Sigma}_{ij}$ of a diffusion (matrix square-root) matrix ${\bm \Sigma}$, and (ii) a *coupling scheme* that fixes the block sparsity of ${\bm \Sigma}$ and how the friction tensor ${\bm \Gamma}$ is assembled from it (see [Assembly of the friction tensor](@ref gamma-assembly) below); the total friction tensor is the sum of the per-model contributions. The package provides:

!!! warning "Renamed from `RWCMatrixModel`"
    What was formerly `RWCMatrixModel` is now `CWCMatrixModel`. The coupling scheme named *row-wise coupling* in Appendix C of [Sachs et al., (2024)](@ref ACEfriction-paper) is more appropriately named *column-wise coupling* — the terminology used in this version of ACEfriction.jl. Constructing `RWCMatrixModel` now raises an error pointing to `CWCMatrixModel`.

- **`OnsiteOnlyMatrixModel`** — block-diagonal. Only ${\bm \Sigma}_{ii}$ is non-zero, expanded from the spherical environment of atom $i$:
  ```math
  {\bm \Sigma}_{ij} = \delta_{ij}\,{\bm \Sigma}_{ii}\big(({\bm r}_k,z_k)_{k\in\mathcal{N}_i}\big).
  ```
- **`PWCMatrixModel`** (pair-wise coupling) — the diffusion matrix is purely off-diagonal, with each ${\bm \Sigma}_{ij}$ ($i\neq j$) expanded from a bond environment of the pair $(i,j)$:
  ```math
  {\bm \Sigma}_{ij} = {\bm \Sigma}_{ij}\big({\bm r}_{ij}, ({\bm r}_k,z_k)_{k}\big), \qquad {\bm \Sigma}_{ii} = {\bm 0}.
  ```
  (The friction tensor ${\bm \Gamma}$ nevertheless acquires a non-zero onsite diagonal; see the pair-wise assembly below.)
- **`CWCMatrixModel`** (column-wise coupling) — full matrix with an *independently fitted* onsite block ${\bm \Sigma}_{ii}$ plus atom-centred off-diagonal blocks ${\bm \Sigma}_{ij}$ ($i\neq j$). (Formerly `RWCMatrixModel`; see the warning above.)

### [Assembly of the friction tensor](@id gamma-assembly)

How ${\bm \Gamma}$ is built from ${\bm \Sigma}$ depends on the coupling scheme — it is **not** in general the plain product ${\bm \Sigma}{\bm \Sigma}^{T}$:

- **Onsite / column-wise** coupling (`OnsiteOnlyMatrixModel`, `CWCMatrixModel`): the friction tensor is the matrix square ${\bm \Gamma} = {\bm \Sigma}{\bm \Sigma}^{T}$, i.e.
  ```math
  {\bm \Gamma}_{ij} = \sum_{k} {\bm \Sigma}_{ik}\,{\bm \Sigma}_{jk}^{T}.
  ```
- **Pair-wise** coupling (`PWCMatrixModel`): the off-diagonal diffusion blocks are combined through the pair-coupled fluctuation–dissipation relation (eq. (7) of [Sachs et al., (2024)](@ref ACEfriction-paper)),
  ```math
  {\bm \Gamma}_{ij} = \begin{cases} {\bm \Sigma}_{ij}\,({\bm \Sigma}_{ji})^{T}, & i\neq j,\\[2pt] \sum_{k} {\bm \Sigma}_{ik}\,({\bm \Sigma}_{ik})^{T}, & i = j. \end{cases}
  ```
  so ${\bm \Gamma}$ has a non-zero onsite diagonal even though ${\bm \Sigma}_{ii}={\bm 0}$.

In both cases the resulting ${\bm \Gamma}$ is symmetric positive semi-definite, and the friction tensor of a multi-model friction model is the sum of the per-model contributions.

## [Local environments (cutoffs)](@id cutoffs)

The local environment entering each block is delimited by a cutoff:

- **`SphericalCutoff(rcut)`** — used for onsite blocks and for the atom-centred offsite blocks of `CWCMatrixModel`: the environment of $i$ is $\mathcal{N}_i = \{\,k : \|{\bm r}_{ik}\| \le r_{\rm cut}\,\}$, and for a pair $(i,j)$ the bond partner is $j\in\mathcal{N}_i$.
- **`EllipsoidCutoff(rcutbond, rcutenv, zcutenv)`** — a bond-centred ellipsoidal environment for `PWCMatrixModel`: bonds with $\|{\bm r}_{ij}\|\le r_{\rm cut}^{\rm bond}$, and environment atoms inside $(z/z_{\rm cut}^{\rm env})^2 + (r/r_{\rm cut}^{\rm env})^2 \le 1$ around the bond midpoint ($z$ along the bond, $r$ perpendicular).
- **`SnowManCutoff(rcut, symmetry)`** — an atom-centred alternative for the pair model: the block of $(i,j)$ combines the ACE basis evaluated on the spherical environment of $i$ (bond $i\to j$) and of $j$ (bond $j\to i$),
  ```math
  {\bm \Sigma}_{ij} = c\cdot B(\text{sphere}_i, i\to j) \;\pm\; c\cdot B(\text{sphere}_j, j\to i),
  ```
  with the sign set by `symmetry` (`:symmetric` $\to +$, giving ${\bm \Sigma}_{ij}={\bm \Sigma}_{ji}$; `:antisymmetric` $\to -$, giving ${\bm \Sigma}_{ij}=-{\bm \Sigma}_{ji}$).

## [Model fitting](@id model-fitting)

The free parameters of a friction model are the scalar ACE coefficients of every block, collected into a single vector ${\bm \theta} = ({\bm c}^{(l,p)})_{l,p}$. The model is fitted to a data set of $N_{\rm obs}$ observations $\big(R^{(n)}, {\bm \Gamma}^{(n)}\big)_{n=1}^{N_{\rm obs}}$ — atomic configurations $R^{(n)} = ({\bm r}^{(n)}_i, z^{(n)}_i)_i$ and reference friction tensors ${\bm \Gamma}^{(n)}$ — by minimising the weighted least-squares loss
```math
\mathcal{L}({\bm \theta}) = \sum_{n=1}^{N_{\rm obs}} \sum_{i,j} \big\| {\bm \Gamma}^{(n)}_{ij} - {\bm W}_{ij}\odot{\bm \Gamma}^{\bm \theta}_{ij}\big(R^{(n)}\big) \big\|_{F}^{2},
```
where $\|\cdot\|_F$ is the Frobenius norm, $\odot$ the entrywise (Hadamard) product, and ${\bm W}$ a weight matrix (assembled by [`flux_assemble`](@ref), with separate weights for diagonal / sub-diagonal / off-diagonal blocks and per observation).

### A quartic, basis-precomputed optimisation problem

The key structural property exploited for fast training is that the diffusion matrix is **linear** in the parameters, while the friction tensor is **quadratic**:
```math
{\bm \Sigma}^{\bm \theta}_{ij} = \sum_{k} {\bm \theta}_{k}\, {\bm B}_{k}(\text{env}_{ij}), \qquad {\bm \Gamma}^{\bm \theta} = \text{(coupling assembly of } {\bm \Sigma}^{\bm \theta}{} \text{)},
```
where ${\bm B}_k$ are the (fixed) equivariant ACE basis functions. Consequently the loss $\mathcal{L}({\bm \theta})$ is a **quartic polynomial** in ${\bm \theta}$.

Crucially, the basis functions ${\bm B}_k(\text{env}_{ij})$ do **not** depend on ${\bm \theta}$, so they are evaluated **once** — during the call to [`flux_assemble`](@ref), which pre-computes and caches the basis array ${\bm B}$ for every observation. During optimisation the (expensive) ACE evaluation is therefore never repeated: each loss/gradient evaluation only re-runs the cheap tensor contractions ${\bm \Sigma}^{\bm \theta}={\bm B}\!\cdot\!{\bm \theta}$ and the quadratic assembly of ${\bm \Gamma}^{\bm \theta}$ (implemented as `@tullio` contractions). This reduces fitting to a fast, smooth polynomial optimisation problem whose per-iteration cost is dominated by dense tensor products rather than basis evaluation.

### Optimiser and autodiff backend

Optimisation is currently carried out with [`Flux.jl`](https://github.com/FluxML/Flux.jl) (e.g. the `Adam` optimiser), differentiating the quartic loss by reverse-mode automatic differentiation via [`Zygote.jl`](https://github.com/FluxML/Zygote.jl). The parameters are wrapped in a [`FluxFrictionModel`](@ref), pre-processed data is produced by [`flux_assemble`](@ref), and ready-made loss functions [`weighted_l2_loss`](@ref) / [`weighted_l1_loss`](@ref) are provided; the workflow is GPU-compatible via CUDA. See the [workflow examples](fitting-eft.md) for the full training loop.

!!! note "Planned migration to Lux"
    The fitting layer is slated to migrate from Flux/Zygote to [`Lux.jl`](https://github.com/LuxDL/Lux.jl), which will allow it to integrate seamlessly with the native automatic-differentiation features of `EquivariantTensors.jl` (the ACE backend). The user-facing fitting API is expected to remain largely unchanged.

## Prototypical Applications

Learned models of ${\bm \Gamma}$ (and the corresponding matrix root ${\bm \Sigma}$) can be used to parametrize tensor-valued coefficients in an Itô diffusion process such as a configuration-dependent friction tensor in a kinetic Langevin equation,
```math
\begin{aligned}
\dot{{\bm r}} &= - M^{-1}{\bm p},\\
\dot{{\bm p}} &= - \nabla U({\bm r}) - {\bm \Gamma}({\bm r})M^{-1}{\bm p} + \sqrt{2 \beta^{-1}} {\bm \Sigma} \dot{{\bm W}},
\end{aligned}
```
or a configuration-dependent diffusion tensor in an overdamped Langevin equation,
```math
\dot{{\bm r}} = - {\bm\Gamma}({\bm r}) \nabla U({\bm r})  + \sqrt{2 \beta^{-1}} {\bm\Sigma}\circ \dot{{\bm W}}. %+ \beta^{-1}{\rm div}({\bm \Gamma}(r)).
```

The model and code allows imposing additional symmetry constraints on the matrix ${\bm \Gamma}$. In particular, the learned friction-tensor ${\bm \Gamma}$ can be specified to satisfy relevant symmetries for the dynamics to be momentum-conserving.