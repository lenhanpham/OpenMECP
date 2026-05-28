# Constraints & Scans

Constraints and PES scan specifications go inside the `*CONSTR` section.
Constraints use Lagrange multipliers via the augmented Hessian method, with
analytical gradients for all constraint types.

## Bond Constraint

Fixes a bond length between two atoms at a target distance (Ångström).

```
R atom1 atom2 target_distance
```

**Example:**

```
*CONSTR
R 1 2 1.54    # Fix C–C bond at 1.54 Å
*
```

---

## Angle Constraint

Fixes a bond angle defined by three atoms at a target value (degrees).

```
A atom1 atom2 atom3 target_angle
```

`atom2` is the central atom.

**Example:**

```
*CONSTR
A 1 2 3 109.5    # Fix angle at 109.5°
*
```

---

## Dihedral Constraint

Fixes a dihedral angle defined by four atoms at a target value (degrees).

```
D atom1 atom2 atom3 atom4 target_dihedral
```

**Example:**

```
*CONSTR
D 1 2 3 4 180.0    # Fix dihedral at 180°
*
```

---

## Fixed Atoms

Individual atoms can be frozen using the `fixedatoms` keyword (not inside
`*CONSTR`). Ranges are supported.

```
fixedatoms = 1,3-5,7    # Fix atoms 1, 3, 4, 5, and 7
```

---

## PES Scans

Scan specifications are placed inside `*CONSTR` and trigger a PES scan run
instead of a single MECP optimization.

### Bond Scan

```
s r atom1 atom2 start num_points step_size
```

**Example** — scan a C–H bond from 1.0 to 1.95 Å in 20 steps of 0.05 Å:

```
*CONSTR
s r 1 2 1.0 20 0.05
*
```

### Angle Scan

```
s a atom1 atom2 atom3 start num_points step_size
```

**Example** — scan from 90° to 135° in 10 steps of 5°:

```
*CONSTR
s a 1 2 3 90 10 5
*
```

### Dihedral Scan

```
s d atom1 atom2 atom3 atom4 start num_points step_size
```

**Example** — full dihedral rotation from 0° to 350° in 36 steps of 10°:

```
*CONSTR
s d 1 2 3 4 0 36 10
*
```

### 2D Scan

Specify two scan lines in `*CONSTR` for a 2D scan:

```
*CONSTR
s r 1 2 1.0 10 0.1    # First dimension: bond
s a 1 2 3 90 5 10     # Second dimension: angle
*
```

**Output**: Geometries saved as `scan_{value1}_{value2}.xyz`.

---

## Combining Constraints

Multiple constraints of different types can be combined freely:

```
*CONSTR
R 1 2 1.54      # Fix bond
A 1 2 3 109.5   # Fix angle
D 1 2 3 4 60.0  # Fix dihedral
*
```
