# Input File Format

OpenMECP input files use a custom section-based format. Sections are delimited
by a `*SECTION_NAME` opening marker and a bare `*` closing marker. Keywords are
specified as `key = value` pairs outside sections (or they can appear after the
geometry section).

## File Layout

```
*GEOM
  <Cartesian coordinates>
*

*TAIL_A
  <extra QM keywords for state A>
*

*TAIL_B
  <extra QM keywords for state B>
*

keyword1 = value1
keyword2 = value2
...
```

Section names are **case-insensitive** (`*GEOM`, `*geom`, and `*Geom` are all
equivalent).

## Section Reference

| Section | Required | Purpose |
|---|---|---|
| [`*GEOM`](sections.md#geom-section) | Yes | Initial geometry in Cartesian coordinates |
| [`*TAIL_A`](sections.md#tail-sections) | Recommended | Extra QM input for state A |
| [`*TAIL_B`](sections.md#tail-sections) | Recommended | Extra QM input for state B |
| [`*CONSTR`](sections.md#constr-section) | No | Geometric constraints and PES scan specs |
| [`*LST_A`](sections.md#lst-sections) | No | Target geometry for LST interpolation (state A) |
| [`*LST_B`](sections.md#lst-sections) | No | Target geometry for LST interpolation (state B) |

## Keywords

All parameters controlling the run are specified as `key = value` pairs after
the geometry section. See the [Keywords Reference](keywords.md) for the full list.

Minimum required keywords:

```
program  = gaussian    # or orca
method   = B3LYP/6-31G*
nprocs   = 4
mem      = 4GB
charge   = 0
mult_a   = 1
mult_b   = 3
```
