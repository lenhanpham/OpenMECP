# Gaussian Examples

All examples locate the triplet/singlet (S=3 / S=1) MECP of the phenylcation
(C₆H₅⁺, charge = 1). See [Overview](overview.md) for the geometry and how to
run.

---

## Example 1 — Default GDIIS

The simplest production-ready input. Uses the default GDIIS optimizer with
3 warm-up BFGS steps.

```
nprocs = 10
mem = 40GB
method = n b3lyp/6-31g**
charge = 1
mult_a = 3
mult_b = 1
mode = normal
program = gaussian

*geom
C     0.000000000000      1.396613000000      0.000000000000
C     1.209503000000      0.698307000000      0.000000000000
C     1.209503000000     -0.698307000000      0.000000000000
C     0.000000000000     -1.396613000000      0.000000000000
C    -1.209503000000     -0.698307000000      0.000000000000
C    -1.209503000000      0.698307000000      0.000000000000
H     2.150882000000      1.241812000000      0.000000000000
H     2.150882000000     -1.241812000000      0.000000000000
H     0.000000000000     -2.483625000000      0.000000000000
H    -2.150882000000     -1.241812000000      0.000000000000
H    -2.150882000000      1.241812000000      0.000000000000
*

*tail_a
*

*tail_b
*
```

**Key settings**: `switch_step = 3` (default), `hessian = direct_psb`
(default), `use_gediis = false` (default).

---

## Example 2 — Sequential Hybrid (GEDIIS → GDIIS)

Activates the 3-phase switching strategy of Li & Frisch (*JCTC* 2006). Adds
only two lines to Example 1 and typically converges ~2–4× faster.

```
nprocs = 10
mem = 40GB
method = n b3lyp/6-31g**
charge = 1
mult_a = 3
mult_b = 1
mode = normal
program = gaussian

# --- optimizer ---
use_gediis = true
use_hybrid_gediis = true

*geom
C     0.000000000000      1.396613000000      0.000000000000
C     1.209503000000      0.698307000000      0.000000000000
C     1.209503000000     -0.698307000000      0.000000000000
C     0.000000000000     -1.396613000000      0.000000000000
C    -1.209503000000     -0.698307000000      0.000000000000
C    -1.209503000000      0.698307000000      0.000000000000
H     2.150882000000      1.241812000000      0.000000000000
H     2.150882000000     -1.241812000000      0.000000000000
H     0.000000000000     -2.483625000000      0.000000000000
H    -2.150882000000     -1.241812000000      0.000000000000
H    -2.150882000000      1.241812000000      0.000000000000
*

*tail_a
*

*tail_b
*
```

**Phase behaviour**: starts with GDIIS, switches to GEDIIS when
`rms_grad ≤ gediis_switch_rms` (0.005 Ha/Å), then back to GDIIS for the
final quadratic convergence when `rms_disp < gediis_switch_step` (0.001 Å).

---

## Example 3 — GDIIS_blend with Trust Radius

Uses the `blend` optimizer with the `fixed_sequential` blend mode and a live
trust radius. Suitable when the default GDIIS takes excessively large or
oscillating steps.

```
nprocs = 10
mem = 40GB
method = n b3lyp/6-31g**
charge = 1
mult_a = 3
mult_b = 1
mode = normal
program = gaussian

# --- optimizer ---
use_gediis = blend
use_hybrid_gediis = true
gediis_blend_mode = fixed_sequential

*geom
C     0.000000000000      1.396613000000      0.000000000000
C     1.209503000000      0.698307000000      0.000000000000
C     1.209503000000     -0.698307000000      0.000000000000
C     0.000000000000     -1.396613000000      0.000000000000
C    -1.209503000000     -0.698307000000      0.000000000000
C    -1.209503000000      0.698307000000      0.000000000000
H     2.150882000000      1.241812000000      0.000000000000
H     2.150882000000     -1.241812000000      0.000000000000
H     0.000000000000     -2.483625000000      0.000000000000
H    -2.150882000000     -1.241812000000      0.000000000000
H    -2.150882000000      1.241812000000      0.000000000000
*

*tail_a
*

*tail_b
*
```

**Blend behaviour**: 50 % GDIIS + 50 % EDIIS far from convergence, transitions
to pure GDIIS for the final steps. Trust radius contracts when energy rises
and expands when energy falls.

> **Note**: `blend` mode requires a direct Hessian method
> (`hessian = direct_psb`, `bofill`, `powell`, or `bfgs_powell_mix`).
> `inverse_bfgs` is incompatible.

---

## Real-World Example — Cu Complex with Solvent

For a larger, more demanding system the `to-4-tep-1` examples in
`examples/gaussian/` show a 29-atom Cu complex optimised at the
ωB97X-D/def2-SVP level with SMD solvation in benzene:

```
nprocs = 48
mem = 190GB
method = n scf(maxcycle=300,xqc) UWB97XD/DEF2SVPP scrf(smd,solvent=Benzene)
charge = 1
mult_a = 3
mult_b = 1
mode = normal
program = gaussian
use_gediis = blend
use_hybrid_gediis = true
gediis_blend_mode = fixed_sequential
```

The geometry is read from the external file `to-4-tep-1-blend-gdiis.xyz`
(included in the repository) via `*geom\n@to-4-tep-1-blend-gdiis.xyz\n*`.
