# Form of `Γ` produced by `_square`

This documents the explicit block form of the friction tensor
`Γ = Gamma(fm, at)` as assembled by `_square` in
[`src/frictionmodels.jl`](../src/frictionmodels.jl).

Each block `Γ_ij` is a 3×3 matrix indexed by atoms `i, j ∈ {1, …, N}`.
`Σ` denotes a single replica's diffusion matrix; a matrix model carries
`n_rep` replicas `Σ⁽¹⁾, …, Σ⁽ⁿ⁾` and its `Γ` is the sum over replicas.
A `FrictionModel`'s `Γ` is in turn the sum over its matrix models.

## Generic models (`OnsiteOnly`, `CWC`)

`_square(Σ, ::MatrixModel) = Σ * transpose(Σ)`:

$$
\Gamma \;=\; \sum_{r=1}^{n_{\mathrm{rep}}} \Sigma^{(r)} \,\bigl(\Sigma^{(r)}\bigr)^{\!\top},
\qquad
\Gamma_{ij} \;=\; \sum_{r}\sum_{k=1}^{N} \Sigma^{(r)}_{ik}\,\bigl(\Sigma^{(r)}_{jk}\bigr)^{\!\top}.
$$

This is just the symmetric square, so `Γ` is positive semidefinite by construction.

## PWC model

`_square(Σ, ::PWCMatrixModel)` builds `Γ` bond-by-bond. For a single replica
`Σ`, it loops over stored entries with `i ≤ j`, reads the reverse block `Σ_ji`,
and accumulates four contributions per bond.

### Off-diagonal pair `{i, j}`, `i ≠ j` (a bond)

$$
\Gamma_{ii} \mathrel{+}= \Sigma_{ij}\Sigma_{ij}^{\top},\qquad
\Gamma_{jj} \mathrel{+}= \Sigma_{ji}\Sigma_{ji}^{\top},\qquad
\Gamma_{ij} \mathrel{+}= \Sigma_{ij}\Sigma_{ji}^{\top},\qquad
\Gamma_{ji} \mathrel{+}= \Sigma_{ji}\Sigma_{ij}^{\top}.
$$

Equivalently, the `{i, j}` 6×6 sub-block is a rank-(≤3) symmetric square:

$$
\begin{pmatrix}\Gamma_{ii} & \Gamma_{ij}\\ \Gamma_{ji} & \Gamma_{jj}\end{pmatrix}
\;=\;
\begin{pmatrix}\Sigma_{ij}\\ \Sigma_{ji}\end{pmatrix}
\begin{pmatrix}\Sigma_{ij}\\ \Sigma_{ji}\end{pmatrix}^{\!\top}.
$$

### Diagonal entry `i = j` (periodic self-image bond)

Whether a diagonal `Σ_ii` block exists at all is governed by the model's
`include_self_images` flag (the `SelfImagePolicy` field):

- **`ExcludeSelfImages` (default):** self-image bond partners (`j == i`) are
  dropped during assembly, so `Σ_ii = 0` and there is no diagonal contribution —
  matching PWC's definition `Σ_ii = 0`.
- **`IncludeSelfImages`:** the diagonal block is kept and contributes **once**,
  exactly like the generic `Σ Σᵀ`:
  $$
  \Gamma_{ii} \mathrel{+}= \Sigma_{ii}\Sigma_{ii}^{\top}.
  $$

### Collected over all bonds and replicas

$$
\Gamma_{ii} \;=\; \sum_{r}\Biggl(\;\sum_{j\,:\,\{i,j\}\ \mathrm{bond}} \Sigma^{(r)}_{ij}\bigl(\Sigma^{(r)}_{ij}\bigr)^{\!\top} \;+\; \Sigma^{(r)}_{ii}\bigl(\Sigma^{(r)}_{ii}\bigr)^{\!\top}\Biggr),
$$

$$
\Gamma_{ij} \;=\; \sum_{r}\,\Sigma^{(r)}_{ij}\bigl(\Sigma^{(r)}_{ji}\bigr)^{\!\top}
\qquad (i \neq j).
$$

with the `Σ_ii Σ_iiᵀ` term present only under `IncludeSelfImages` (it vanishes,
`Σ_ii = 0`, by default).

## Notes on the structure

- **Diagonal blocks use the *row* of `Σ`** (`Σ_ij` summed over bond partners
  `j`), not the column contraction of `Σ Σᵀ`. This is the key difference from
  the generic case, and what the corrected `randf` reproduces.
- **Only bonds with `i ≤ j` are walked**, so a bond stored solely as `(j, i)`
  with `j > i` contributes nothing — `_square` and `randf` are consistent on
  this.
- **Self-image diagonals** are off by default (`Σ_ii = 0`); when
  `include_self_images = true` the `i == j` case is special-cased to contribute
  `1·Σ_ii Σ_iiᵀ` (the same single-square rule as the generic `Σ Σᵀ`), and `randf`
  draws one shared noise for it.
- The **vector-equivariant** case is identical, with each 3×3 product replaced
  by the rank-1 outer product of the `SVector{3}` blocks (e.g.
  `Σ_ij Σ_ji^⊤`).

## Self-image handling compared across models

A *self-image* is a periodic image of atom `i` that falls within atom `i`'s own
cutoff, so the neighbour's atom index `j` equals `i`. As a **bond partner** such
a neighbour would produce a diagonal block `Σ_ii`. Whether it does is controlled
by the `include_self_images` flag (default `false`), which acts on bond partners
only — descriptor *environments* are never filtered by it.

### Where a diagonal `Σ_ii` comes from, and the effect of the flag

| model | `Σ` structure | diagonal source | effect of `include_self_images` |
| --- | --- | --- | --- |
| `OnsiteOnly` | `Diagonal` (only `(i,i)`) | the onsite descriptor (always present) | **no-op** — no bond-partner loop |
| `CWC` | onsite `(i,i)` + offsite `(i,j)` | onsite block, plus any self-image offsite partner `j == i` merged into `(i,i)` | toggles the offsite self-image contribution |
| `PWC` | offsite only | a self-image offsite partner `j == i` | toggles the whole diagonal: `Σ_ii = 0` (off) vs nonzero (on) |

### How the kept diagonal is squared

**`OnsiteOnly` and `CWC` use the generic `_square(Σ) = Σ Σᵀ`.** A diagonal block
is then just an ordinary matrix entry:

$$
\Gamma_{ij} \;=\; \sum_{r}\sum_{k} \Sigma^{(r)}_{ik}\bigl(\Sigma^{(r)}_{jk}\bigr)^{\!\top},
$$

so `Σ_ii` contributes only through the single `k = i` term — **counted exactly
once**. Because `randf = Σ · w` always satisfies `Cov(Σ w) = Σ Σᵀ`, the random
force is automatically consistent for *any* `Σ`. For `CWC`, a kept self-image
offsite block is **added into the same `(i,i)` block as the onsite block before
squaring**, so the two are coupled and the combined block is squared as a unit.

**`PWC` uses the bespoke pairwise `_square`.** When self-images are excluded
(default) there are no diagonal entries and the question is moot. When included,
the `i == j` case is special-cased to contribute `1 · Σ_ii Σ_iiᵀ` — the **same
single-square rule** as the generic case — and `randf` draws one shared noise for
it. (This replaced an earlier non-physical `4×` that came from running the
off-diagonal four-push code on a diagonal entry.)

In short: with the default `ExcludeSelfImages`, `PWC` is purely off-diagonal as
its definition intends; with `IncludeSelfImages`, all three models treat a
diagonal block the same way — squared exactly once — so `Cov(randf) = Γ` holds in
every case.
