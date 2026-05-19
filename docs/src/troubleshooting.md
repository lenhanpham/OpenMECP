# Troubleshooting

## Common Errors

### "No such file or directory"

**Cause**: QM program not found in PATH, or input file is missing.

**Solutions**:
- Verify the QM program is installed and accessible in your `$PATH`.
- Check that the input file path is spelled correctly.

---

### "QM calculation failed"

**Cause**: The QM program encountered an internal error.

**Solutions**:
- Open the `running_dir/*.log` (or `*.out`) files and look for program-specific error messages.
- Confirm that the method and basis set are valid for the chosen program.
- Check that `nprocs` and `mem` do not exceed available resources.
- Switch to `mode = noread` to force a fresh SCF each step and avoid reading a corrupted wavefunction:

```
mode = noread
```

**Program-specific hints**:

| Program | Symptom | Remedy |
|---|---|---|
| Gaussian | "Convergence failure" | Add `scf=(xqc,qc,nofermi)` to the `*TAIL` section |
| ORCA | "SCF not converged" | Add `%scf SOSCF true end` to the `*TAIL` section |
| Custom | Unexpected output | Verify the `energy_parser` / `forces_parser` regex patterns in the JSON interface |

---

### "Failed to parse output"

**Cause**: The QM output format was not recognized — often because the calculation did not finish normally.

**Solutions**:
- Confirm the QM calculation completed without errors before OpenMECP tries to read it.
- Check that the program version is supported.

**Parsing-specific tips**:

| Issue | Action |
|---|---|
| Energy not found | Check the `energy_parser` regex in the custom interface |
| Forces not found | Verify `forces_parser` matches the output format |
| Geometry incomplete | Ensure the optimization step wrote a complete geometry block |
| State extraction failed (TD-DFT) | Confirm `state_a` / `state_b` indices are valid (0 = ground, 1+ = excited) |

---

### "Maximum steps exceeded"

**Cause**: The optimizer did not converge within `max_steps` (default 100).

**Solutions**:
- Increase the step limit:
  ```
  max_steps = 300
  ```
- Verify that the starting geometry is chemically reasonable.
- Use [LST interpolation](advanced/lst.md) to generate a better initial guess.
- Check for recurring SCF convergence failures — if the QM energy/gradient is wrong the optimizer cannot converge.

---

## SCF Convergence Issues

**Symptoms**: QM calculations fail repeatedly, or the energy oscillates step-to-step.

**Solutions**:

1. **Use `mode = noread`** — starts every SCF from scratch, avoids corrupted guess orbitals:
   ```
   mode = noread
   ```

2. **Use `mode = stable`** — checks wavefunction stability before proceeding:
   ```
   mode = stable
   ```

3. **Use `mode = inter_read`** — recommended for open-shell singlets, reads MOs from the previous step to maintain spin symmetry:
   ```
   mode = inter_read
   ```

4. **Increase the SCF iteration limit** via the `*TAIL` section:
   - Gaussian:
     ```
     scf=(maxcycle=200,xqc)
     ```
   - ORCA:
     ```
     %scf maxiter 200 end
     ```

See [Run Modes](run-modes.md) for a full comparison of `mode` options.

---

## Slow Convergence

**Symptoms**: Optimization requires an unusually large number of steps.

### Step 1 — Increase the step limit

```
max_steps = 300
```

### Step 2 — Try a different optimizer

The examples below go from fastest/least-robust to most-robust. Pick the first
one that converges for your system.

#### Pure GDIIS (default)

No changes needed — GDIIS is the default. Explicitly:

```
use_gediis = false   # GDIIS only (default)
switch_step = 3      # 3 BFGS warm-up steps, then DIIS (default)
```

---

#### Sequential Hybrid GEDIIS/GDIIS

Three-phase switching: GDIIS → GEDIIS (near seam) → GDIIS (final). Typically
2–4× faster than pure GDIIS.

```
use_gediis = true
use_hybrid_gediis = true
gediis_switch_rms  = 0.005   # GDIIS → GEDIIS when RMS grad < 0.005 Ha/Å (default)
gediis_switch_step = 0.001   # GEDIIS → GDIIS when RMS disp < 0.001 Å (default)
```

---

#### GDIIS\_blend — Pure (trust-radius GDIIS, no EDIIS)

GDIIS protected by an adaptive trust radius. Very robust; no EDIIS component.

```
switch_step = 3
hessian = direct_psb      # required for blend mode
use_gediis = blend
use_hybrid_gediis = false  # default for blend; can be omitted
```

---

#### GDIIS\_blend — Fixed-Sequential Hybrid (recommended blend default)

50/50 GDIIS+EDIIS far from the minimum, transitions to pure GDIIS near
convergence. Best overall blend mode for production runs.

```
switch_step = 3
hessian = direct_psb
use_gediis = blend
use_hybrid_gediis = true
gediis_blend_mode = fixed_sequential   # default when use_hybrid_gediis = true
```

---

#### GDIIS\_blend — Gradient-Weighted Hybrid

Smooth sigmoid blend driven by the RMS gradient. EDIIS-heavy when forces are
large, GDIIS-heavy near convergence. Useful for systems with energy plateaus.

```
switch_step = 3
hessian = direct_psb
use_gediis = blend
use_hybrid_gediis = true
gediis_blend_mode = gradient
```

---

#### GDIIS\_blend — Sequential Hybrid

Per-step binary switching (GDIIS or EDIIS) based on the RMS displacement trend.

```
switch_step = 3
hessian = direct_psb
use_gediis = blend
use_hybrid_gediis = true
gediis_blend_mode = sequential
```

---

### Optimizer Quick-Reference

| Optimizer | Keywords | Speed | Stability |
|---|---|---|---|
| Pure BFGS | `switch_step ≥ max_steps` | Slowest | Highest |
| GDIIS (default) | `use_gediis = false` | Good | High |
| Sequential Hybrid | `use_gediis = true`, `use_hybrid_gediis = true` | ~2–4× GDIIS | High |
| GDIIS\_blend Pure | `use_gediis = blend`, `use_hybrid_gediis = false` | Good | High |
| GDIIS\_blend Fixed-Seq | `use_gediis = blend`, `gediis_blend_mode = fixed_sequential` | Good | Medium–High |
| GDIIS\_blend Gradient | `use_gediis = blend`, `gediis_blend_mode = gradient` | Good | Medium–High |
| GDIIS\_blend Sequential | `use_gediis = blend`, `gediis_blend_mode = sequential` | Good | Medium |

> **Note**: All `blend` modes require a direct Hessian update method:
> `direct_psb` (default), `bofill`, `powell`, or `bfgs_powell_mix`.
> `inverse_bfgs` is incompatible with blend mode.

---

### Hessian Update Methods

The `hessian` keyword affects both step quality and compatibility with blend
mode. Choose a method suited to your system:

| Value | Description | Blend Compatible |
|---|---|---|
| `direct_psb` | Powell–Symmetric–Broyden (default) | Yes |
| `bofill` | Bofill update — better for TS-like crossings | Yes |
| `powell` | Powell update | Yes |
| `bfgs_powell_mix` | Mixed BFGS/Powell | Yes |
| `inverse_bfgs` | Classic inverse BFGS | **No** |

For transition-state-like crossings, `bofill` is recommended:

```
hessian = bofill
```

---

### Robust DIIS

For particularly difficult cases, enable the [Robust DIIS](algorithms/robust-diis.md)
sanity-check layer, which detects and rejects ill-conditioned DIIS steps:

```
use_robust_diis = true
```

Combine with any optimizer above for extra protection. See the
[Robust DIIS](algorithms/robust-diis.md) page for full options.
