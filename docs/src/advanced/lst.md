# LST Interpolation

Linear Synchronous Transit (LST) interpolation generates an initial MECP guess
by linearly mixing between two endpoint geometries. This is useful when the
MECP lies somewhere between a known reactant geometry and a known product
geometry.

## Setup

Add `*LST_A` and `*LST_B` sections alongside the standard `*GEOM`:

```
*GEOM
C  -4.10   -0.27   -0.11
H  -4.46    0.46    0.62
*

*LST_A
C  0.0  0.0  0.0
H  1.0  0.0  0.0
*

*LST_B
C  0.0  0.0  0.5
H  1.2  0.0  0.5
*
```

When both `*LST_A` and `*LST_B` sections are present, OpenMECP:

1. Applies the **Kabsch algorithm** to optimally align both structures to the
   `*GEOM` reference (removes translational and rotational differences).
2. Generates a sequence of **10 interpolation points** along the LST path.
3. Prints the energy profile along the interpolated path.
4. Asks for interactive confirmation before proceeding with MECP optimization.

---

## Interpolation Methods

| Method | Description |
|---|---|
| **LST** (default) | Linear Synchronous Transit — linear mixing of coordinates |
| **QST** | Quadratic Synchronous Transit — uses midpoint approximation |

**Note**: QST currently uses a midpoint approximation for the third geometry.
Full QST with an explicit user-supplied midpoint is planned for a future release.

---

## Kabsch Alignment

The Kabsch algorithm finds the optimal rotation matrix $\mathbf{R}$ that
minimizes the RMSD between corresponding atoms of the two structures. This
ensures that the interpolation path follows the shortest physically meaningful
route between the two geometries, avoiding spurious rotations.

No manual pre-alignment of the structures is required — OpenMECP handles this
automatically.

---

## Output

After LST interpolation:

| Output | Description |
|---|---|
| Interpolation energy profile | Printed to stdout: coordinate vs. E₁, E₂ |
| `running_dir/lst_*.inp` | Input files for each interpolation point |
| `{basename}_mecp.xyz` | Final MECP geometry after optimization |

---

## Tips

- LST is most effective when the two endpoint geometries differ mainly in bond
  lengths or angles, not overall topology.
- If the energy profile shows a clear minimum-energy crossing along the LST
  path, the starting geometry for MECP optimization will be close to the MECP,
  leading to faster convergence.
- Use `print_level = 2` to see the full interpolated path during the LST stage.
