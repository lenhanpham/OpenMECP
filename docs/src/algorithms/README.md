# MECP Algorithm

OpenMECP implements the MECP optimization algorithm of Harvey et al.
(*Theor. Chem. Acc.* **99**, 95–99, 1998) with several modern enhancements.

## Problem Statement

Given two electronic states with energies $E_1(\mathbf{R})$ and $E_2(\mathbf{R})$
and gradients $\mathbf{f}_1(\mathbf{R})$ and $\mathbf{f}_2(\mathbf{R})$, find the
geometry $\mathbf{R}^*$ that simultaneously satisfies:

1. **Degeneracy**: $E_1(\mathbf{R}^*) = E_2(\mathbf{R}^*)$
2. **Minimum energy**: $\mathbf{R}^*$ is the lowest-energy point on the crossing seam

## Effective Gradient

Harvey et al. construct an effective gradient that encodes both conditions.
First, define the normalized gradient difference:

$$\hat{\mathbf{x}} = \frac{\mathbf{f}_1 - \mathbf{f}_2}{|\mathbf{f}_1 - \mathbf{f}_2|}$$

The effective gradient has two orthogonal components:

**f-vector** — drives the energy gap to zero:

$$\mathbf{f} = (E_1 - E_2)\,\hat{\mathbf{x}}$$

**g-vector** — minimizes the average energy in the intersection seam:

$$\mathbf{g} = \mathbf{f}_1 - (\hat{\mathbf{x}} \cdot \mathbf{f}_1)\,\hat{\mathbf{x}}$$

The total effective gradient is:

$$\mathbf{G}_{\text{eff}} = \mathbf{f} + \mathbf{g}$$

## Convergence Criteria

All five criteria must be satisfied simultaneously:

| Criterion | Keyword | Default |
|---|---|---|
| Energy difference $|E_1 - E_2|$ | `delta_e` | 5.0 × 10⁻⁵ Ha |
| RMS atomic displacement | `rms_dis` | 0.0025 Å |
| Maximum atomic displacement | `max_dis` | 0.004 Å |
| Maximum gradient component | `max_grad` | 1.323 × 10⁻³ Ha/Å |
| RMS gradient | `rms_grad` | 9.45 × 10⁻⁴ Ha/Å |

## Optimization Strategy

OpenMECP uses a hybrid strategy combining stability and speed:

```
Step 1–switch_step:  BFGS (builds Hessian curvature information)
Step switch_step+1:  GDIIS / GEDIIS (quadratic-convergence DIIS acceleration)
```

The default `switch_step = 3` provides a 3-step BFGS warm-up before engaging DIIS.
See [Optimizers](optimizers.md) for a full description of each algorithm.

## Two-State QM Calculation

At each MECP optimization step, OpenMECP:

1. Writes the QM input files for **state A** and **state B** from the current geometry.
2. Runs the QM program for state A, reads $E_1$ and $\mathbf{f}_1$.
3. Runs the QM program for state B, reads $E_2$ and $\mathbf{f}_2$.
4. Evaluates the MECP effective gradient $\mathbf{G}_{\text{eff}}$.
5. Updates the geometry using the selected optimizer.
6. Checks convergence; writes checkpoint file if requested.

## Unit Conventions

| Quantity | Unit used internally |
|---|---|
| Energies | Hartree |
| Coordinates (geometry) | Ångström |
| Gradients | Hartree/Ångström |
| Hessian (direct) | Hartree/Ångström² |
| Hessian (inverse) | Ångström²/Hartree |
| Step size | Bohr (max_step_size) |
