# Run Modes

The `mode` keyword controls how OpenMECP manages QM wavefunction checkpoints
between optimization steps. Choosing the right run mode can significantly
affect SCF convergence speed and stability.

```
mode = normal    # default
```

---

## Normal Mode (Default)

```
mode = normal
```

Standard MECP optimization. OpenMECP reads the wavefunction checkpoint from the
previous step as the SCF initial guess.

- Reads `.chk` (Gaussian) or `.gbw` (ORCA) from the previous step
- Recommended for most calculations
- Fastest per-step runtime

---

## Read Mode

```
mode = read
```

Reads existing wavefunction checkpoints from disk without running pre-point
calculations. Use this when restarting a calculation where checkpoints already
exist from a previous run.

- Skips pre-point calculations
- Requires existing checkpoint files
- Useful for restarting after a crash or job time limit

---

## NoRead Mode

```
mode = noread
```

Runs a fresh SCF at every step without reading any checkpoint. This is slower
but more robust for systems with difficult SCF convergence or when the
wavefunction character changes significantly between steps.

- Does not read or write checkpoint files
- Slower (full SCF from scratch each step)
- Use when checkpoint reading causes SCF divergence

---

## Stable Mode

```
mode = stable
```

Runs pre-point calculations at the initial geometry before starting the MECP
optimization. This checks SCF stability (`stable=opt` in Gaussian) and
produces reliable initial checkpoints.

- Runs `stable=opt` or equivalent before step 1
- Ensures stable wavefunction from the start
- Recommended for systems with unstable ground-state wavefunctions (e.g., open-shell systems)

---

## InterRead Mode

```
mode = inter_read
```

**Essential for open-shell singlet (OSS) states.**

InterRead runs state B (typically the triplet) first at each step, then copies
the triplet wavefunction as the initial guess for state A (the open-shell
singlet) and adds `guess=(read,mix)` (Gaussian) or equivalent. This provides
a physically correct broken-symmetry starting point for the singlet.

**Example — open-shell singlet/triplet MECP:**

```
mode   = inter_read
mult_a = 1    # Open-shell singlet
mult_b = 3    # Triplet (reference for broken-symmetry guess)
```

---

## Mode Comparison

| Mode | SCF Stability | Speed | Use Case |
|---|---|---|---|
| `normal` | Good | Fastest | Standard production runs |
| `read` | Good | Fast | Restart from existing checkpoints |
| `noread` | Best | Slowest | Difficult SCF convergence |
| `stable` | Best | Moderate | Unstable initial wavefunction |
| `inter_read` | Best for OSS | Moderate | Open-shell singlet calculations |
