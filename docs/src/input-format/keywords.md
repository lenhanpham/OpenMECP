# Keywords Reference

All keywords are case-insensitive and use `key = value` syntax. Boolean values
accept `true`/`false`.

## Required Keywords

| Keyword | Type | Description | Example |
|---|---|---|---|
| `program` | string | QM program to use | `gaussian`, `orca` |
| `method` | string | QM method and basis set | `B3LYP/6-31G*` |
| `nprocs` | integer | Number of processors | `4` |
| `mem` | string | Memory allocation | `4GB` (Gaussian), `8000` (ORCA MB/core) |
| `charge` | integer | Molecular charge | `0` |
| `mult_a` | integer | Spin multiplicity of state A | `1` |
| `mult_b` | integer | Spin multiplicity of state B | `3` |

### Program-Specific Method Format

**Gaussian** — method goes in the route section:

```
method = n scf(maxcycle=500,xqc) uwb97xd/def2svpp scrf=(smd,solvent=acetonitrile)
mem    = 8GB
```

**ORCA** — method is the `!` line (without the `!`):

```
method = B3LYP def2-SVP CPCM(2-octanone) VeryTightSCF
mem    = 8000
```

For ORCA, `mem` is the per-core memory in MB (`%maxcore` in the ORCA input).

---

## TD-DFT and State Selection

| Keyword | Type | Default | Description |
|---|---|---|---|
| `td_a` | string | `""` | TD-DFT keywords for state A |
| `td_b` | string | `""` | TD-DFT keywords for state B |
| `state_a` | integer | `0` | Excited state index for state A (0 = ground state) |
| `state_b` | integer | `0` | Excited state index for state B |
| `mp2` | boolean | `false` | Use MP2 gradients (Gaussian only) |

`td_state_a` and `td_state_b` are accepted as aliases for `td_a` and `td_b`.

---

## Run Mode and Basic Optimization

| Keyword | Type | Default | Description |
|---|---|---|---|
| `mode` | string | `normal` | Run mode — see [Run Modes](../run-modes.md) |
| `max_steps` | integer | `100` | Maximum optimization steps |
| `max_step_size` | float | `0.1` | Maximum step size (Bohr) |
| `restart` | boolean | `false` | Restart from checkpoint file |

---

## Optimizer Settings

| Keyword | Type | Default | Description |
|---|---|---|---|
| `hessian` | string | `direct_psb` | Hessian update method (see [Hessian Updates](../algorithms/hessian-updates.md)) |
| `use_gediis` | bool/string | `false` | DIIS variant: `false`=GDIIS, `true`/`sequential`=GEDIIS, `blend`=GDIIS_blend |
| `use_hybrid_gediis` | boolean | `false` | Enable sequential hybrid GDIIS/GEDIIS switching |
| `gediis_blend_mode` | string | `fixed_sequential` | Blend strategy: `fixed`, `fixed_sequential`, `gradient`, `sequential` |
| `gediis_switch_rms` | float | `0.005` | RMS gradient threshold for GDIIS → GEDIIS switch (Ha/Å) |
| `gediis_switch_step` | float | `0.001` | RMS displacement threshold for GEDIIS → GDIIS switch (Å) |
| `switch_step` | integer | `3` | Steps of BFGS before switching to DIIS |
| `max_history` | integer | `4` | DIIS history length |
| `smart_history` | boolean | `false` | Smart DIIS history (retains most useful points) |
| `reduced_factor` | float | `0.5` | Step size reduction factor for GDIIS |
| `step_reduction_multiplier` | float | `10.0` | RMS gradient multiplier for step reduction |
| `steepest_descent_step` | float | `0.01` | Fallback steepest-descent step size (Å) |
| `bfgs_rho` | float | `15.0` | BFGS step size scaling factor |
| `print_level` | integer | `1` | Verbosity: `0`=quiet, `1`=normal, `2`=verbose |
| `print_checkpoint` | boolean | `false` | Write checkpoint JSON at each step |

### Trust Radius (GDIIS_blend)

| Keyword | Type | Default | Description |
|---|---|---|---|
| `trust_reduction_factor` | float | `0.5` | Contraction factor when energy increases |
| `trust_increase_factor` | float | `1.2` | Expansion factor when energy decreases |
| `trust_inc_threshold` | float | `0.0001` | Energy increase threshold (Ha) for contraction |
| `trust_dec_threshold` | float | `0.0001` | Energy decrease threshold (Ha) for expansion |
| `trust_min_radius` | float | `0.01` | Minimum trust radius (Å) |
| `trust_max_radius` | float | `1.0` | Maximum trust radius (Å) |

---

## Convergence Thresholds

| Keyword | Type | Default | Description |
|---|---|---|---|
| `delta_e` | float | `0.000050` | Energy difference threshold (hartree) |
| `rms_dis` | float | `0.0025` | RMS displacement threshold (Å) |
| `max_dis` | float | `0.004` | Maximum displacement threshold (Å) |
| `max_grad` | float | `0.001323` | Maximum gradient component threshold (Ha/Å) |
| `rms_grad` | float | `0.000945` | RMS gradient threshold (Ha/Å) |

---

## Robust DIIS (Step Validation)

These options are only active when `use_robust_diis = true`.

| Keyword | Type | Default | Description |
|---|---|---|---|
| `use_robust_diis` | boolean | `false` | **Master switch** for DIIS step validation |
| `gediis_variant` | string | `auto` | GEDIIS variant: `auto`, `rfo`, `energy`, `simultaneous` |
| `gdiis_cosine_check` | string | `standard` | Cosine threshold: `none`, `zero`, `standard`, `variable`, `strict` |
| `gdiis_coeff_check` | string | `regular` | Coefficient check: `none`, `regular`, `force_recent`, `combined` |
| `n_neg` | integer | `0` | Negative eigenvalues: `0`=minimum, `1`=TS search |
| `gediis_sim_switch` | float | `0.0025` | RMS error threshold for GEDIIS variant switching |

See [Robust DIIS](../algorithms/robust-diis.md) for detailed descriptions.

---

## Optimizer Switching Control

| `switch_step` value | Behavior |
|---|---|
| `3` (default) | BFGS for 3 steps, then DIIS |
| `0` | Pure DIIS from step 1 (fastest, needs good geometry) |
| `10` | Extended 10-step BFGS warm-up, then DIIS |
| `≥ max_steps` | Pure BFGS throughout (most stable) |

---

## Constraints and Fixed Atoms

| Keyword | Type | Default | Description |
|---|---|---|---|
| `fixedatoms` | string | `""` | Fixed atom indices, e.g. `1,3-5,7` |

---

## Coordinate Driving

| Keyword | Type | Default | Description |
|---|---|---|---|
| `drive_type` | string | `""` | Coordinate type: `bond`, `angle`, `dihedral` |
| `drive_atoms` | string | `""` | Atom indices (comma-separated) |
| `drive_start` | float | `0.0` | Starting coordinate value |
| `drive_end` | float | `0.0` | Ending coordinate value |
| `drive_steps` | integer | `10` | Number of drive steps |
| `drive_coordinate` | string | `""` | Reaction coordinate specification |

---

## Program Commands

| Keyword | Type | Default | Description |
|---|---|---|---|
| `gau_comm` | string | `g16` | Gaussian executable name or path |
| `orca_comm` | string | `orca` | ORCA executable name or path |

---

## ONIOM (QM/MM)

| Keyword | Type | Default | Description |
|---|---|---|---|
| `isoniom` | boolean | `false` | Enable ONIOM multi-layer calculation |
| `chargeandmultforoniom1` | string | `""` | Charge/multiplicity specification for state A |
| `chargeandmultforoniom2` | string | `""` | Charge/multiplicity specification for state B |

---

## Advanced / Miscellaneous

| Keyword | Type | Default | Description |
|---|---|---|---|
| `charge_b` | integer | — | Separate charge for state B (overrides `charge`) |
| `basis` | string | `""` | Basis set specification |
| `solvent` | string | `""` | Solvent model |
| `dispersion` | string | `""` | Dispersion correction |
| `custom_interface_file` | string | `""` | Path to custom QM interface JSON config |
| `fix_de` | float | `0.0` | Target energy difference for fix-dE mode (eV) *(temporarily disabled)* |
