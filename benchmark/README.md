# Benchmarks

Standalone scripts, run from the repository root with the package environment:

```bash
julia --project=. benchmark/<script>.jl
```

| script | what it measures |
|---|---|
| `bench_evaluation.jl` | Cost of `matrix` (Σ) and `basis` (fitting path) per atom for every coupling scheme (onsite, CWC, PWC spherical with and without the bond partner in the environment, snowman, ellipsoid), relative to an energy-like evaluation of an ACE basis of the same size. Settings (system size, body order / degree, `n_rep`) at the top of the file. |
| `bench_snowman.jl` | Snowman pair model: the default assembly (each directed bond evaluated once) against the naive two-evaluations-per-pair path, after checking that both agree. |
| `fit_partner_in_env.jl` | Fit quality of the two bond-environment conventions (`partner_in_env = true`, the default, vs `false`) on the H/Cu data `test/test-data-100.h5`: paired fits with identical seeds and settings, standard split or k-fold CV, absolute or scale-balanced loss. Options are documented in the file header; results are appended to a CSV file and summarised as paired differences. |

The timing scripts are single-threaded and report minimum wall-clock times; run them
on an otherwise idle machine for comparable numbers.
