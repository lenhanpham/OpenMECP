# ORCA

## Setting Up an ORCA Calculation

```
program = orca
method  = B3LYP def2-SVP
nprocs  = 8
mem     = 8000
charge  = 0
mult_a  = 1
mult_b  = 3
```

### `method` Format

The `method` value is inserted after the `!` on the ORCA simple input line.
Include all keywords that would normally follow `!` in your ORCA input:

```
method = B3LYP def2-SVP CPCM(2-octanone) VeryTightSCF
```

### `mem` Format

Specify per-core memory in **MB** — this maps to `%maxcore` in the ORCA input:

```
mem = 8000    # 8000 MB per core
```

### Executable

By default, OpenMECP calls `orca`. Override with:

```
orca_comm = /opt/orca5/orca    # full path recommended for ORCA
```

---

## TD-DFT and Excited States

Use the `*TAIL_A` / `*TAIL_B` sections to inject `%tddft` blocks:

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

Combined with:

```
state_a = 1
state_b = 2
```

---

## CASSCF

Put CASSCF blocks in the tail sections:

```
*TAIL_A
%casscf
  nel    8
  norb   8
  nroots 3
  iroot  0
end
*
```

---

## Extra Input via *TAIL Sections

Anything in `*TAIL_A` / `*TAIL_B` is appended as an additional input block for
the respective state. Common uses:

```
*TAIL_A
%scf
  MaxIter 500
  Convergence VeryTight
end
*
```

---

## Output Files

| File | Content |
|---|---|
| `running_dir/{basename}_N_state_A.inp` | ORCA input for state A at step N |
| `running_dir/{basename}_N_state_A.out` | ORCA output for state A at step N |
| `running_dir/{basename}_N_state_A.engrad` | Gradient file for state A at step N |
| `state_A.gbw` | Wavefunction for state A (updated each step) |
| `state_B.gbw` | Wavefunction for state B (updated each step) |
| `{basename}_mecp.xyz` | Final MECP geometry |

---

## Run Modes

ORCA reads wavefunction from `.gbw` files. The `read` and `noread` run modes
control whether OpenMECP passes a `MORead` keyword to ORCA. See
[Run Modes](../run-modes.md) for details.

---

## Notes

- Ensure the `orca` executable is on your `PATH` or specify the full path via `orca_comm`.
- ORCA requires an exclusive MPI launch; do not wrap it in `mpirun` in your job script — ORCA handles parallel execution internally.
- The `.engrad` file is written only when ORCA is asked for gradients (the default for MECP runs).
