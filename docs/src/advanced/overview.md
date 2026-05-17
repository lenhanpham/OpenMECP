# Advanced Features

This chapter covers OpenMECP's advanced capabilities beyond basic MECP optimization.

| Feature | Description |
|---|---|
| [PES Scan](pes-scan.md) | 1D and 2D potential energy surface scans |
| [LST Interpolation](lst.md) | Generate initial MECP guess from two endpoint geometries |
| [Reaction Path & NEB](reaction-path.md) | Systematic coordinate driving and Nudged Elastic Band |
| [Fixed Atoms](fixed-atoms.md) | Freeze selected atoms during optimization |

## State Selection for TD-DFT

Choose specific excited states for TD-DFT calculations using `state_a` and
`state_b`:

```
state_a = 1    # Use first excited state for state A
state_b = 2    # Use second excited state for state B
td_a = TD(nstates=5,root=1)
td_b = TD(nstates=5,root=2)
```

This is particularly important when:

- The ground state is contaminated by near-degeneracy
- Targeting a specific conical intersection between excited states
- Studying multi-state photochemical reactivity

## Pre-point Calculations

Pre-point calculations run a single-point QM job at the starting geometry
before the first optimization step. Enable with `mode = stable` (see
[Run Modes](../run-modes.md)). This checks SCF stability and ensures that
wavefunction checkpoints are available from step 1.

## Restart from Checkpoint

```
restart = true
```

OpenMECP saves a JSON checkpoint (`{basename}_checkpoint.json`) after each step
that records the full optimization state: geometry, DIIS history, Hessian, and
configuration. Setting `restart = true` picks up from the last saved checkpoint.

Enable checkpoint writing:

```
print_checkpoint = true
restart          = true
```

## Debug Logging

Enable file-based debug logging via `omecp_config.cfg`:

```ini
[logging]
file_logging = true
```

Log files are written to `omecp_debug_{basename}.log` in the working directory.
