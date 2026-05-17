# PES Scan

OpenMECP can perform 1D and 2D potential energy surface scans by running a
constrained MECP optimization at each point on a grid.

## 1D Bond Scan

Specify a scan in the `*CONSTR` section:

```
*CONSTR
s r atom1 atom2 start num_points step_size
*
```

**Example** — scan a C–C bond from 1.0 to 1.95 Å in 20 steps of 0.05 Å:

```
*CONSTR
s r 1 2 1.0 20 0.05
*
```

Final geometry at each scan point: `scan_1.00.xyz`, `scan_1.05.xyz`, …

---

## 1D Angle Scan

```
*CONSTR
s a atom1 atom2 atom3 start num_points step_size
*
```

**Example** — scan a C–C–H angle from 90° to 135° in 10 steps of 5°:

```
*CONSTR
s a 1 2 3 90 10 5
*
```

---

## 1D Dihedral Scan

```
*CONSTR
s d atom1 atom2 atom3 atom4 start num_points step_size
*
```

**Example** — full dihedral rotation from 0° to 350° in 36 steps of 10°:

```
*CONSTR
s d 1 2 3 4 0 36 10
*
```

---

## 2D Scan

Specify two scan lines in `*CONSTR`:

```
*CONSTR
s r 1 2 1.0 10 0.1    # First dimension: C–C bond, 10 points
s a 1 2 3 90 5 10     # Second dimension: C–C–H angle, 5 points
*
```

This performs 10 × 5 = 50 MECP optimizations on a grid. Output geometries are
named `scan_{val1}_{val2}.xyz`.

---

## Output

| File | Content |
|---|---|
| `scan_{value}.xyz` (1D) | MECP geometry at each scan point |
| `scan_{v1}_{v2}.xyz` (2D) | MECP geometry at each grid point |
| stdout | Energies and scan coordinate values for each point |

The energy values printed to stdout can be used directly to plot the PES scan.

---

## Tips

- For 2D scans, reduce `max_steps` per point to keep the total runtime
  manageable: `max_steps = 50`.
- Use `print_level = 0` to suppress per-step output and only log the final
  energies for each scan point.
- Converged MECP geometries from one scan point are used as the starting
  geometry for the next, which generally speeds up convergence along the scan.
