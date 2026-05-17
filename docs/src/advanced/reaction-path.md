# Reaction Path & NEB

## Coordinate Driving

Coordinate driving systematically varies an internal coordinate (bond, angle,
or dihedral) in a series of constrained MECP optimizations. This traces a
reaction path on the crossing seam.

### Keywords

```
drive_type       = bond        # bond | angle | dihedral
drive_atoms      = 1,2         # comma-separated atom indices
drive_start      = 1.0         # starting value (Å or degrees)
drive_end        = 2.5         # ending value
drive_steps      = 20          # number of steps
```

### Example — C–N Bond Elongation

```
drive_type   = bond
drive_atoms  = 1,2
drive_start  = 1.4
drive_end    = 2.4
drive_steps  = 20
```

At each of the 20 steps, OpenMECP runs a constrained MECP optimization with the
C–N bond fixed at the current driving value. The converged geometry is then used
as the starting point for the next step.

**Output**: Energy profile along the driven coordinate printed to stdout;
geometries saved to `running_dir/`.

---

## Nudged Elastic Band (NEB)

NEB optimizes an entire reaction path simultaneously by connecting a chain of
geometries ("images") with spring forces that keep them evenly spaced along the
path.

### Setup

```
drive_type   = bond
drive_atoms  = 1,2
drive_start  = 1.0
drive_end    = 2.0
drive_steps  = 10    # number of images on the path
```

When the `run_mode` is set to `path_optimization`, OpenMECP:

1. Creates an initial path via coordinate driving.
2. Applies simplified NEB spring forces between adjacent images.
3. Optimizes each image's position on the crossing seam.

**Current Limitations**: The NEB implementation uses a simplified spring model.
A full NEB with perpendicular energy gradient projection is planned for a future
release.

**Output**: Smooth energy profile along the optimized path; geometry files for
each image.

---

## Choosing Between Methods

| Method | Use When |
|---|---|
| Coordinate driving | Tracing a known reaction path step by step |
| NEB | Optimizing an entire path simultaneously |
| LST interpolation | Generating an initial MECP guess from two endpoints |
| PES scan | Mapping energy as a function of one or two coordinates |
