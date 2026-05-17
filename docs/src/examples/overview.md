# Examples Overview

This section provides complete, ready-to-run examples for locating MECPs with
OpenMECP. All examples are also available in the `examples/` directory of the
repository.

## Test System: Phenylcation (C₆H₅⁺)

The primary test case used throughout these examples is the **phenylcation**,
a well-studied 11-atom aromatic cation. The goal is to locate the minimum
energy crossing point between the **triplet (S = 3)** and **open-shell singlet
(S = 1)** spin states.

This system is an excellent benchmark: it is small enough for rapid testing on
any hardware, yet exercises all parts of the MECP optimizer, including
gradient combination, Hessian updates, and DIIS extrapolation.

### Geometry

Save the following as `phenylcation.xyz` (or embed it directly in the
`*geom` section of your input):

```
11

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
```

## How to Run

```bash
omecp input.inp > output.log
```

OpenMECP writes a summary line to stdout at each step. Key fields:

| Column | Meaning |
|---|---|
| `Step` | Iteration number |
| `ΔE` | Energy difference between the two states (target: → 0) |
| `RMS grad` | RMS effective gradient (target: < `rms_grad` threshold) |
| `RMS disp` | RMS Cartesian displacement from previous step |
| `Max disp` | Largest single atomic displacement |

## Output Files

| File | Contents |
|---|---|
| `{basename}_mecp.xyz` | Final optimized MECP geometry |
| `running_dir/` | All intermediate QM input/output files |
| `{basename}_omecp_debug.log` | Step-by-step debug log (if `file_logging = true`) |

## Available Examples

| Page | System | Program | Optimizer |
|---|---|---|---|
| [Gaussian Examples](gaussian.md) | C₆H₅⁺ | Gaussian | GDIIS, Sequential Hybrid, GDIIS_blend |
| [ORCA Examples](orca.md) | C₆H₅⁺ | ORCA | GDIIS, GDIIS_blend |
