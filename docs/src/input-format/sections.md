# Input Sections

## *GEOM Section

Provides the initial molecular geometry in Cartesian coordinates (Ångström).

**Format:**

```
*GEOM
Element  X  Y  Z
...
*
```

**Example:**

```
*GEOM
C     -4.108383   -0.269570   -0.105531
C     -3.086048   -0.004314   -0.907012
H     -4.464908    0.456695    0.621271
H     -4.622302   -1.226624   -0.151874
C     -2.326155    1.302381   -0.898892
H     -2.751025   -0.749763   -1.624593
*
```

You can also reference an external XYZ file:

```
*GEOM
@geometry.xyz
*
```

The XYZ file must use standard format (Ångström units, with or without the
atom-count / comment header lines).

---

## *TAIL_A and *TAIL_B Sections

These sections inject extra text directly into the QM program's input for
state A and state B respectively. Their content is program-specific.

**Gaussian** — append additional route section keywords:

```
*TAIL_A
scf(maxcycle=500,xqc)
*

*TAIL_B
scf(maxcycle=500,xqc)
*
```

**ORCA** — append additional input blocks (e.g., `%tddft`, `%scf`):

```
*TAIL_A
%tddft
  nroots 5
  iroot  1
end
*

*TAIL_B
%tddft
  nroots 5
  iroot  2
end
*
```

If no extra keywords are needed, include empty sections:

```
*TAIL_A
*

*TAIL_B
*
```

---

## *CONSTR Section

Specifies geometric constraints (fixed internal coordinates) or PES scan
specifications. See [Constraints & Scans](constraints.md) for the full
syntax reference.

**Example — fix a bond and an angle:**

```
*CONSTR
R 1 2 1.54
A 1 2 3 109.5
*
```

**Example — 1D bond scan:**

```
*CONSTR
s r 1 2 1.0 20 0.05
*
```

---

## *LST_A and *LST_B Sections

Provide the endpoint geometry for Linear Synchronous Transit (LST) interpolation.
When both sections are present, OpenMECP generates an interpolated initial guess
by linearly mixing the `*GEOM` geometry and the LST target.

**Format** — same as `*GEOM` (Cartesian coordinates, Ångström):

```
*LST_A
C  0.0  0.0  0.0
H  1.0  0.0  0.0
*

*LST_B
C  0.0  0.0  0.5
H  1.2  0.0  0.5
*
```

Atom ordering must match between `*GEOM`, `*LST_A`, and `*LST_B`. OpenMECP
applies the Kabsch algorithm to find the optimal rigid-body alignment before
interpolating, so pre-alignment of the structures is not required.

See [LST Interpolation](../advanced/lst.md) for full details.
