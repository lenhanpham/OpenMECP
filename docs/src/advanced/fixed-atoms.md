# Fixed Atoms

Selected atoms can be frozen during MECP optimization, allowing partial
geometry optimization. Fixed atoms do not contribute to the MECP gradient
and their coordinates are never updated by the optimizer.

## Usage

Use the `fixedatoms` keyword (not inside `*CONSTR`). Specify atom indices
using comma-separated values and ranges:

```
fixedatoms = 1,3-5,7    # Fix atoms 1, 3, 4, 5, and 7
```

Atom indices are **1-based** and correspond to the order of atoms in the
`*GEOM` section.

---

## Syntax

| Syntax | Meaning |
|---|---|
| `1` | Atom 1 only |
| `1,2,3` | Atoms 1, 2, and 3 |
| `3-7` | Atoms 3, 4, 5, 6, and 7 |
| `1,3-5,7` | Atoms 1, 3, 4, 5, and 7 |

---

## Use Cases

**Partial optimization** — optimize only the active site of a large molecule:

```
fixedatoms = 10-50    # Freeze the molecular scaffold; optimize only atoms 1–9
```

**QM/MM boundaries** — freeze the MM region atoms while optimizing the QM core:

```
fixedatoms = 15-100
```

**Surface calculations** — fix the surface slab atoms while relaxing the
adsorbate:

```
fixedatoms = 1-48    # 48-atom surface slab; adsorbate is atoms 49–54
```

---

## Interaction with Constraints

Fixed atoms and geometric constraints (`*CONSTR`) can be combined in the same
calculation. A fixed atom's coordinate is not optimized, but it can still
appear as part of a constraint definition (e.g., as the anchor atom of a bond
constraint).

---

## Notes

- Fixing a large fraction of atoms significantly reduces the effective degrees
  of freedom, which can speed up convergence.
- The DIIS history and Hessian are built only in the subspace of free atoms.
- Fixed atoms are included in the geometry written to output files for visual
  inspection.
