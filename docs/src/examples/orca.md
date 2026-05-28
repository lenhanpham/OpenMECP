# ORCA Examples

All examples locate the triplet/singlet (S=3 / S=1) MECP of the phenylcation
(C₆H₅⁺, charge = 1). See [Overview](overview.md) for the geometry and how to
run.

> **ORCA memory**: pass the `MaxCore` value in MB, not the total job memory.
> For example, `mem = 4000` means 4 000 MB per core.

---

## Example 1 — Default GDIIS

```
nprocs = 10
mem = 4000
method = B3LYP 6-31G(d,p) VeryTightSCF
charge = 1
mult_a = 3
mult_b = 1
mode = normal
program = orca
orca_comm = orca

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

**Key settings**: default GDIIS, `hessian = direct_psb`, `switch_step = 3`.

> On HPC systems, set `orca_comm` to the full path of the ORCA binary, e.g.:
> ```
> orca_comm = /apps/orca/6.1.1/orca
> ```

---

## Example 2 — GDIIS_blend with Gradient-Based Weight

Uses `use_gediis = blend` with `gediis_blend_mode = gradient`. The blend weight
$w$ varies smoothly as $w = \text{RMS}_g / (\text{RMS}_g + \text{switch\_rms})$,
so the optimizer is EDIIS-heavy far from convergence and GDIIS-heavy close to it.

```
nprocs = 10
mem = 4000
method = B3LYP 6-31G(d,p) VeryTightSCF
charge = 1
mult_a = 3
mult_b = 1
mode = normal
program = orca
orca_comm = orca

# --- optimizer ---
use_gediis = blend
use_hybrid_gediis = true
gediis_blend_mode = gradient

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

---

## TD-DFT with State Selection

For excited-state MECPs, add `%tddft` blocks in the tail sections and set
`state_a` / `state_b` to select which excited state to track:

```
nprocs = 10
mem = 4000
method = B3LYP def2-SVP VeryTightSCF
charge = 0
mult_a = 1
mult_b = 1
state_a = 1
state_b = 2
mode = normal
program = orca
orca_comm = orca

*geom
...
*

*tail_a
%tddft
  nroots 5
  iroot 1
end
*

*tail_b
%tddft
  nroots 5
  iroot 2
end
*
```
