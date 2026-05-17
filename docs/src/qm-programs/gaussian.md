# Gaussian

## Setting Up a Gaussian Calculation

```
program = gaussian
method  = B3LYP/6-31G*
nprocs  = 8
mem     = 16GB
charge  = 0
mult_a  = 1
mult_b  = 3
```

### `method` Format

The `method` value is inserted directly into the Gaussian route section (`#`
line). You can include any valid Gaussian route keywords:

```
method = n scf(maxcycle=500,xqc) uwb97xd/def2svpp scrf=(smd,solvent=acetonitrile)
```

### `mem` Format

Specify memory using Gaussian's syntax — e.g. `4GB`, `8GB`, `4000MB`.

### Executable

By default, OpenMECP calls `g16`. Override with:

```
gau_comm = g09    # or full path: /opt/g16/g16
```

---

## TD-DFT and Excited States

Use `td_a` / `td_b` to inject TD-DFT keywords into the route section for each
state, and `state_a` / `state_b` to select which root to use for the gradient:

```
td_a   = TD(nstates=5,root=1)
td_b   = TD(nstates=5,root=2)
state_a = 1
state_b = 2
```

---

## MP2

Enable MP2 gradients (instead of DFT) with:

```
mp2 = true
```

---

## CASSCF and Multi-Reference Methods

Put CASSCF keywords in the `*TAIL_A` / `*TAIL_B` sections as extra route or
additional input blocks:

```
*TAIL_A
casscf(8,8)/cc-pVDZ
*

*TAIL_B
casscf(8,8)/cc-pVDZ
*
```

---

## Extra Input via *TAIL Sections

Anything in `*TAIL_A` / `*TAIL_B` is appended to the Gaussian route section
keywords for the respective state. Common uses:

```
*TAIL_A
scf(maxcycle=500,xqc) stable=opt
*
```

---

## ONIOM (QM/MM)

Multi-layer ONIOM calculations are supported. Mark atom layers in `*GEOM`
using Gaussian's `H` / `M` / `L` layer notation:

```
*GEOM
C  0.0  0.0  0.0  H
H  1.0  0.0  0.0  L
H  0.0  1.0  0.0  L
*

isoniom = true
chargeandmultforoniom1 = 0 1 0 1   # state A: QM and MM charges/mults
chargeandmultforoniom2 = 0 3 0 1   # state B
```

---

## Run Modes and Checkpoints

OpenMECP automatically manages Gaussian checkpoint files (`.chk`) to pass the
wavefunction between steps. The `mode` keyword controls this behavior — see
[Run Modes](../run-modes.md) for details.

The `inter_read` mode is **essential for open-shell singlet (OSS)** calculations:
it copies the triplet wavefunction to the singlet and adds `guess=(read,mix)`.

---

## Output Files

| File | Content |
|---|---|
| `running_dir/{basename}_N_state_A.log` | Gaussian output for state A at step N |
| `running_dir/{basename}_N_state_B.log` | Gaussian output for state B at step N |
| `state_A.chk` | Checkpoint for state A (updated each step) |
| `state_B.chk` | Checkpoint for state B (updated each step) |
| `{basename}_mecp.xyz` | Final MECP geometry |
