# Robust DIIS

Standard GDIIS/GEDIIS can occasionally produce unreliable steps when the DIIS
matrix is ill-conditioned — for example, near convergence or when the history
contains redundant points. The Robust DIIS system adds configurable sanity
checks that detect and reject bad steps before they are applied.

## Enabling Robust DIIS

```
use_robust_diis = true
```

This is the **master switch**. All options described on this page are inactive
unless `use_robust_diis = true`.

---

## Cosine Check

```
gdiis_cosine_check = standard    # default
```

Computes the cosine between the proposed DIIS step and the reference gradient.
A low cosine means DIIS is stepping in a direction inconsistent with the local
gradient — a sign that the step is unreliable and should be rejected.

| Value | Threshold | Description |
|---|---|---|
| `none` | — | Skip cosine check |
| `zero` | cos > 0 | Reject only steps pointing uphill |
| `standard` | cos ≥ 0.71 | 45° angle limit — balances safety and speed |
| `variable` | 0.1 → 0.7 | Adaptive: permissive early, tight near convergence |
| `strict` | cos ≥ 0.866 | 30° angle limit — conservative for fragile systems |

---

## Coefficient Check

```
gdiis_coeff_check = regular    # default
```

Validates that the DIIS interpolation coefficients $c_i$ are physically
reasonable. Well-conditioned DIIS produces coefficients that sum to 1 with
individual values near $[0, 1]$.

| Value | Description |
|---|---|
| `none` | Accept any coefficients |
| `regular` | Reject extreme deviations; allows moderate negative $c_i$ |
| `force_recent` | Strongly bias toward the latest point; useful after restart |
| `combined` | Both `regular` bounds and cosine threshold simultaneously |

---

## GEDIIS Variants

```
gediis_variant = auto    # default
```

Only active when `use_robust_diis = true` **and** `use_gediis = true`.

| Variant | B-matrix | Best for |
|---|---|---|
| `auto` | Switches automatically | General use; starts with `energy`, switches to `rfo` |
| `rfo` | Quadratic step overlaps | Near the minimum; quadratic PES approximation |
| `energy` | Gradient-coordinate products | Far from minimum; non-quadratic PES |
| `simultaneous` | Combined energy + quadratic | Most sophisticated; blends `rfo` and `energy` |

The `gediis_sim_switch` parameter (default: `0.0025`) controls the RMS error
threshold at which `auto` switches from `energy` to `rfo`.

---

## Negative Eigenvalue Control

```
n_neg = 0    # default (minimum search)
n_neg = 1    # transition-state search (one imaginary mode)
```

Set `n_neg = 1` for TS-like MECP searches where the crossing point is a
saddle point with one negative Hessian eigenvalue.

---

## Worked Examples

### Oscillating Convergence

If the optimizer oscillates or takes wild steps near convergence:

```
use_robust_diis    = true
gdiis_cosine_check = standard
gdiis_coeff_check  = regular
```

### Fragile System (Flat PES / Near-Dissociation)

```
use_robust_diis    = true
gdiis_cosine_check = strict
```

### Restart from Checkpoint

```
use_robust_diis   = true
gdiis_coeff_check = force_recent
restart           = true
```

### Full Robust GEDIIS

```
use_robust_diis    = true
use_gediis         = true
gediis_variant     = energy
gdiis_cosine_check = standard
gdiis_coeff_check  = regular
```
