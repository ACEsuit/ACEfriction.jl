# [Extending ACEfriction](@id extending)

New matrix-model types can be defined in other packages, on top of the models provided
here. A matrix model is a subtype of `ACEfriction.MatrixModels.MatrixModel` that produces
the diffusion matrix ${\bm \Sigma}$ (one sparse `N×N` block matrix per replica) of a
configuration, and its un-contracted basis for fitting. The names below are the public
extension API of `ACEfriction.MatrixModels` (declared `public`, not exported).

## What a model type provides

- the fields `n_rep`, `inds::SiteInds`, `id`, and its site-model dictionaries `onsite`
  and/or `offsite`: the generic parameter plumbing (`params`, `set_params!`, `nparams`,
  `scaling`, …) reads these;
- methods for `matrix(M, at; filter, T)` and `basis(M, at; join_sites, filter, T)`;
  `basis` returns one sparse block matrix per basis function such that
  `Σ[r] = Σ_k c[k][r] * B[k]`;
- a method for `randf(M, Σ)` (random force with covariance ${\bm \Gamma}$);
- `write_dict(M)` and `read_dict(::Val{:<id>}, D)` (ACEbase FIO) for serialization;
- a [`sigma_structure`](@ref ACEfriction.MatrixModels.sigma_structure) method if its
  friction tensor is not ${\bm \Gamma} = {\bm \Sigma}{\bm \Sigma}^T$ of a general block
  matrix (the default). It selects both the friction-tensor assembly and the layout of the
  fitting data, so a model that declares it is fitted with `flux_assemble` /
  `FluxFrictionModel` without further code.

## Building blocks

Atom-centred pair models are built from a bond basis (`offsite_linbasis`) and one
`OffSiteModel` per species pair, and assembled with the same fast per-centre evaluation
as the built-in coupling schemes:

```@docs
ACEfriction.MatrixModels.offsite_models
ACEfriction.MatrixModels.offsite_matrix
ACEfriction.MatrixModels.offsite_basis
ACEfriction.MatrixModels.site_inds
ACEfriction.MatrixModels.sigma_structure
ACEfriction.MatrixModels.SigmaStructure
ACEfriction.MatrixModels.n_rep
ACEfriction.MatrixModels.default_id
ACEfriction.MatrixModels.self_image_policy
```

`test/test_extension_api.jl` defines a complete example model type (atom-centred pair
blocks only) using nothing but this API, and checks it against the built-in models,
serialization and fitting.
