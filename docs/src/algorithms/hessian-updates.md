# Hessian Updates

The `hessian` keyword selects how the Hessian (or inverse Hessian) matrix is
stored and updated at each optimization step.

## Available Methods

### `direct_psb` — Powell-Symmetric-Broyden (Default)

```
hessian = direct_psb
```

Stores the **direct Hessian** $\mathbf{H}$ (units: Ha/Å²) and solves the linear
system $\mathbf{H} \cdot \mathbf{d}_k = -\mathbf{g}$ at each step via LU
decomposition.

The PSB update formula preserves the secant condition while maintaining
symmetry:

$$\mathbf{H}_{k+1} = \mathbf{H}_k + \frac{(\mathbf{y} - \mathbf{H}_k \mathbf{s})\mathbf{s}^\top + \mathbf{s}(\mathbf{y} - \mathbf{H}_k \mathbf{s})^\top}{\mathbf{s}^\top \mathbf{s}} - \frac{(\mathbf{y} - \mathbf{H}_k \mathbf{s})^\top \mathbf{s}}{\left(\mathbf{s}^\top \mathbf{s}\right)^2}\mathbf{s}\mathbf{s}^\top$$

where $\mathbf{s} = \mathbf{R}_{k+1} - \mathbf{R}_k$ and
$\mathbf{y} = \mathbf{g}_{k+1} - \mathbf{g}_k$.

**Required** when `use_gediis = blend`. **Recommended for all production runs.**

---

### `inverse_bfgs` — Inverse Hessian BFGS

```
hessian = inverse_bfgs
```

Stores the **inverse Hessian** $\mathbf{H}^{-1}$ (units: Å²/Ha) and computes
steps as $\mathbf{d}_k = -\mathbf{H}^{-1} \mathbf{g}$ via matrix-vector
multiply (no linear solve required).

Uses the standard BFGS rank-2 update:

$$\mathbf{H}^{-1}_{k+1} = \left(\mathbf{I} - \rho_k \mathbf{s}\mathbf{y}^\top\right) \mathbf{H}^{-1}_k \left(\mathbf{I} - \rho_k \mathbf{y}\mathbf{s}^\top\right) + \rho_k \mathbf{s}\mathbf{s}^\top, \quad \rho_k = \frac{1}{\mathbf{y}^\top \mathbf{s}}$$

**Note**: Incompatible with `use_gediis = blend`. Use `direct_psb` instead for
the blend optimizer.

---

### `bofill` — Bofill Update *(Experimental)*

```
hessian = bofill
```

Blends the Powell (SR1) and Murtagh–Sargent updates using a Bofill-style
weighting parameter $\phi \in [0, 1]$:

$$\mathbf{H}_{k+1} = \phi \cdot \mathbf{H}^\text{Powell}_{k+1} + (1-\phi) \cdot \mathbf{H}^\text{MS}_{k+1}$$

The weight $\phi$ is computed from the curvature: $\phi$ is close to 1 when the
gradient change is well-described by a symmetric rank-1 correction (Powell),
and close to 0 otherwise (Murtagh–Sargent).

**Recommended for**: TS-like crossing points where the Hessian may have
negative eigenvalues.

---

### `powell` — Powell Symmetric Rank-1 *(Experimental)*

```
hessian = powell
```

Uses the symmetric SR1 update. Unlike BFGS, SR1 can build up negative
curvature, which is useful for crossing points with Hessians that have
negative eigenvalues.

---

### `bfgs_powell_mix` — Adaptive Blend *(Experimental)*

```
hessian = bfgs_powell_mix
```

Adaptively switches between BFGS and Powell updates using Bofill-style
weighting at each step. Inherits the robustness of Powell near negative
curvature and the superlinear convergence of BFGS near positive-definite
regions.

---

## Choosing a Method

| Scenario | Recommended |
|---|---|
| Standard MECP optimization | `direct_psb` *(default)* |
| GDIIS_blend optimizer | `direct_psb` *(required)* |
| TS-like crossing point | `bofill` |
| Very flat or ill-conditioned PES | `powell` |
| Uncertain curvature | `bfgs_powell_mix` |
| Legacy / simple system | `inverse_bfgs` |
