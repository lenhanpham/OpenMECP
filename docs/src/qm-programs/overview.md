# QM Program Support

OpenMECP communicates with external quantum chemistry programs through a unified
interface. Each program has its own input/output format; OpenMECP handles all
the file writing and parsing automatically.

## Supported Programs

| Program | `program` value | Methods | Notes |
|---|---|---|---|
| Gaussian 16 | `gaussian` | DFT, TD-DFT, MP2, CASSCF | Recommended |
| ORCA 5+ | `orca` | DFT, TD-DFT, CASSCF | Full gradient support |
| Custom | `custom` | User-defined | JSON configuration |

## Selecting a Program

Set the `program` keyword in your input file:

```
program = gaussian    # or orca / custom
```

## Common Configuration

Both programs share these keywords:

```
nprocs  = 8       # Number of parallel CPU cores
charge  = 0       # Molecular charge
mult_a  = 1       # Multiplicity of state A
mult_b  = 3       # Multiplicity of state B
```

See individual program pages for `method` and `mem` format details:

- [Gaussian](gaussian.md)
- [ORCA](orca.md)
- [Custom Interface](custom.md)

## Wavefunction Checkpoints

OpenMECP saves wavefunction files from each QM step to use as a starting guess
for the next step. This dramatically speeds up SCF convergence over a
multi-step optimization.

| Program | State A | State B |
|---|---|---|
| Gaussian | `state_A.chk` | `state_B.chk` |
| ORCA | `state_A.gbw` | `state_B.gbw` |

The [Run Mode](../run-modes.md) controls whether these checkpoints are read.

## File Organization

All QM input/output files are written to a `running_dir/` subdirectory,
keeping your working directory clean. The final MECP geometry is written to
`{basename}_mecp.xyz` in the current directory.
