# Optimizers

OpenMECP provides several geometry optimizers. All optimizers work on the MECP
effective gradient described in the [MECP Algorithm](README.md) chapter.

## BFGS (Initial Steps)

The Broyden–Fletcher–Goldfarb–Shanno quasi-Newton method provides a stable
warm-up for the first `switch_step` iterations (default: 3). It builds up
curvature information in the approximate Hessian before DIIS takes over.

**When to use BFGS exclusively**: Set `switch_step = 999` (or any value ≥
`max_steps`) to run pure BFGS throughout — most stable but slower.

Relevant keywords: `hessian`, `bfgs_rho`, `max_step_size`

---

## GDIIS — Geometry DIIS (Default)

`use_gediis = false` (default)

The Geometry Direct Inversion in the Iterative Subspace method constructs the
next geometry as an optimal linear combination of previous Newton-step
corrections:

$$\mathbf{e}_i = \mathbf{H}^{-1} \cdot (\mathbf{g}_i + \mathbf{f}_i)$$

$$\mathbf{B}_{jk} = \langle \mathbf{e}_j \mid \mathbf{e}_k \rangle$$

$$\begin{pmatrix} \mathbf{B} & \mathbf{1} \\ \mathbf{1}^\top & 0 \end{pmatrix}
  \begin{pmatrix} \mathbf{c} \\ \lambda \end{pmatrix} =
  \begin{pmatrix} \mathbf{0} \\ 1 \end{pmatrix}$$

$$\mathbf{x}^* = \sum_i c_i \mathbf{x}_i, \qquad
  \mathbf{x}_\text{GDIIS} = \mathbf{x}^* - \mathbf{H}^{-1} \cdot P(\mathbf{x}^*)$$

GDIIS minimizes residual Newton-step forces and delivers robust quadratic
convergence. Proven reliable for ~10-step convergence on production systems.

**Performance**: ~2–3× faster than pure BFGS.

---

## GEDIIS — Energy-Informed DIIS

`use_gediis = true`, `use_hybrid_gediis = false`

Extends GDIIS by adding an energy-based diagonal bias to the B-matrix:

$$B_{ii} \mathrel{+}= \alpha \cdot E_i^2, \qquad \text{RHS}_i = -E_i$$

where $E_i = |E_{1,i} - E_{2,i}|$ is the energy gap at step $i$. The diagonal
term biases interpolation toward geometries with small energy gaps (i.e., close
to the crossing seam). The constraint is $c_i > 0$ — GEDIIS only interpolates,
never extrapolates.

**Best for**: systems where the crossing seam needs to be approached from a
geometry far from it.

---

## Sequential Hybrid GEDIIS/GDIIS

`use_gediis = true`, `use_hybrid_gediis = true`

Implements the 3-phase switching strategy of Li & Frisch (*JCTC* 2006):

| Phase | Condition | Algorithm | Rationale |
|---|---|---|---|
| 1 | RMS grad > `gediis_switch_rms` | GDIIS | Robust convergence from rough starting geometry |
| 2 | RMS grad ≤ `gediis_switch_rms` | GEDIIS | Energy-guided approach to the crossing seam |
| 3 | RMS disp < `gediis_switch_step` | GDIIS | Clean quadratic final convergence |

Default thresholds: `gediis_switch_rms = 0.005` Ha/Å, `gediis_switch_step = 0.001` Å.

**Performance**: ~2–4× faster than pure GDIIS on most systems.

---

## GDIIS_blend

`use_gediis = "blend"`

A per-step blend of a GDIIS step and an EDIIS step, governed by a trust radius:

$$\mathbf{x}_\text{new} = w \cdot \mathbf{x}_\text{EDIIS} + (1 - w) \cdot \mathbf{x}_\text{GDIIS}$$

When `use_hybrid_gediis = false`, no EDIIS component is added (pure GDIIS with
trust radius control). When `use_hybrid_gediis = true`, the blend weight $w$ is
determined by `gediis_blend_mode`:

| `gediis_blend_mode` | Weight $w$ | Behavior |
|---|---|---|
| `fixed` | 0.5 constant | Equal GDIIS+EDIIS; may plateau |
| `fixed_sequential` *(default)* | 0.5 → 0 near convergence | 50/50 far out, pure GDIIS for final convergence |
| `gradient` | $w = \frac{\text{RMS}_g}{\text{RMS}_g + \text{switch\_rms}}$ | Smooth sigmoid; EDIIS-heavy far, GDIIS-heavy near |
| `sequential` | 1 or 0 per step | Binary switching based on displacement trend |

The trust radius automatically contracts when energy increases and expands when
energy decreases, preventing wild steps.

**Requirement**: A direct Hessian update method (`direct_psb`, `bofill`, `powell`,
or `bfgs_powell_mix`). `inverse_bfgs` is incompatible.

---

## Optimizer Switching

The `switch_step` parameter controls when the warm-up BFGS steps hand off to DIIS:

```
switch_step = 3     # Default: 3 BFGS steps, then DIIS
switch_step = 0     # Pure DIIS from step 1 (fastest, needs good geometry)
switch_step = 10    # 10 BFGS steps, then DIIS (better for difficult cases)
switch_step = 999   # Pure BFGS (most stable, slowest)
```

The `smart_history` option retains the most useful geometry history points
rather than using a simple FIFO queue, which can accelerate convergence.

---

## Performance Comparison

| Optimizer | Convergence Speed | Stability | Best Use |
|---|---|---|---|
| BFGS | Baseline | Highest | Very difficult cases |
| GDIIS | ~2–3× BFGS | High | Most production runs |
| Sequential Hybrid | ~2–4× GDIIS | High | Default recommended |
| GEDIIS | Similar to hybrid | Medium | Crossing-seam-distant starts |
| GDIIS_blend | Adaptive | Medium–High | Systems with trust radius issues |
