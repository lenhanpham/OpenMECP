<p align="center">
  <a href="https://github.com/lenhanpham/OpenMECP">
    <picture>
      <img src="resources/omecp-logo.svg" alt="OpenMECP" style="width: 50%;">
    </picture>
  </a>
</p>
<p align="center">
  <a href="https://www.rust-lang.org/">
    <img src="https://img.shields.io/badge/Rust-1.70+-orange.svg" alt="Rust">
  </a>
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  </a>
  <a href="https://github.com/lenhanpham/OpenMECP/actions/workflows/build.yml">
    <img src="https://github.com/lenhanpham/OpenMECP/actions/workflows/build.yml/badge.svg" alt="Build Status">
  </a>
</p>

A high-performance Rust implementation of the MECP (Minimum Energy Crossing Point) optimizer for locating crossing points between two potential energy surfaces in quantum chemistry calculations.

## Note

- **OpenMECP is highly robust and ready for production runs to locate MECPs now.** 

- **The program is under active development. All supported features are being tested and improved.**

**Status**: Beta testing phase

Current features include:

- Template generation from geometry files (`omecp ci file.xyz`)
- Comprehensive built-in help system (`omecp --help`)
- Full MECP optimization capabilities

## Overview

OpenMECP locates the minimum energy crossing point between two potential energy surfaces (PES), which is crucial for understanding:

- Photochemical reactions
- Spin-forbidden processes
- Intersystem crossing
- Conical intersections
- Non-adiabatic dynamics

The program implements the algorithm reported by Harvey et al. in Theor Chem Acc 99, 95–99 (1998) with modern optimizers and supports multiple quantum chemistry programs.

### Key Algorithm

The MECP gradient combines two components:

1. **f-vector**: Drives energy difference to zero: `f = (E1 - E2) * x_norm`
2. **g-vector**: Minimizes energy perpendicular to gradient difference: `g = f1 - (x_norm · f1) * x_norm`

Where `x_norm = (f1 - f2) / |f1 - f2|` is the normalized gradient difference.

## Features

### User Interface

- **Template Generation**: Create input files from geometry files (XYZ, GJF, LOG)
- **Comprehensive Help**: Built-in help system with keyword, method, and feature documentation
- **Command Line Interface**: Simple `omecp ci` and `omecp --help` commands

### Supported Algorithms

- **MECP Optimization**: Harvey et al. algorithm implementation
- **Direct Hessian + PSB (Recommended)**: Default algorithm using direct Hessian with Powell-Symmetric-Broyden update. Proven on hundreds of MECP calculations.
- **Inverse Hessian + BFGS**: Algorithm using inverse Hessian with BFGS update.
- **GDIIS Optimizer**: Geometry-based DIIS for 2-3x faster convergence
- **GEDIIS Optimizer**: Energy-informed DIIS for enhanced convergence (Li & Frisch JCTC 2006)
- **Sequential Hybrid Strategy**: 3-phase switching: GDIIS → GEDIIS → GDIIS (Li & Frisch JCTC 2006)
- **GDIIS_blend**: Pure GDIIS with trust region and inverted mean true Hessian
- **GDIIS_blend + EDIIS Blend**: Four blend strategies combining GDIIS and EDIIS with trust radius:
  - `fixed`: 50/50 equal blend
  - `fixed_sequential` (default): 50/50 → pure GDIIS near convergence
  - `gradient`: Smooth sigmoid blend weighted by RMS gradient
  - `sequential`: Per-step GDIIS/EDIIS switching based on step trend
- **Robust DIIS (Experimental)**: Enhanced GDIIS/GEDIIS with SR1 updates, cosine validation, and coefficient checks
- **Multiple Hessian Updates (Experimental)**: BFGS, Powell, Bofill, and adaptive BFGS/Powell blend

### Constraints

- **Bond Constraints**: Fix bond lengths during optimization.
- **Angle Constraints**: Fix bond angles during optimization.
- **Dihedral Constraints**: Fix dihedral angles during optimization using four-atom definitions with analytical gradients.
- **Lagrange Multipliers**: Enforces constraints with high precision and efficiency using a full implementation of the augmented Hessian method with analytical gradients for all constraint types (bond, angle, dihedral).
- **Fixed Atoms**: Partial geometry optimization by freezing selected atoms.

### QM Program Support

- **Gaussian** (recommended): DFT, TD-DFT, MP2, CASSCF
- **ORCA**: DFT, TD-DFT, CASSCF
- **Custom**: User-defined via JSON configuration

### Advanced Features

- **PES Scan**: 1D and 2D potential energy surface scans
- **LST Interpolation**: Linear synchronous transit with Kabsch alignment
- **Coordinate Driving**: Drive reaction coordinates systematically
- **Path Optimization**: Nudged Elastic Band (NEB) method
- **Fix-dE Optimization**: Constrain energy difference to target value. (*Note: This feature is currently being updated to use the new constraint system and is temporarily disabled.*)
- **State Selection**: Choose specific excited states for TD-DFT
- **External Geometry**: Read geometries from external files
- **ONIOM Support**: Multi-layer QM/MM calculations
- **Run Modes**: Normal, Read, NoRead, Stable, InterRead
- **Pre-point Calculations**: For difficult SCF convergence

### Configuration Management

- **Hierarchical Configuration**: Support for multiple configuration file locations with precedence
- **Configuration File**: `omecp_config.cfg` (avoiding confusion with other programs' settings.ini)
- **Configuration Sources** (in order of precedence):
  1. `./omecp_config.cfg` (local directory - highest priority)
  2. `~/.config/omecp/omecp_config.cfg` (user configuration)
  3. `/etc/omecp/omecp_config.cfg` (system configuration)
  4. Built-in defaults (fallback)
- **Customizable Settings**:
  - File extensions for different QM programs
  - Memory and processor defaults
  - Logging levels and debug file configuration
  - Cleanup behavior and frequency
- **Parameter Display**: Prints all configuration parameters and settings source at startup
- **Debug Logging**: Optional file-based debug logging with dynamic filenames (`omecp_debug_<input_basename>.log`)

To create a configuration template:

```bash
omecp ci omecp_config.cfg
```

## Installation

### Prerequisites

- Rust 1.70 or later
- One or more QM programs (Gaussian or ORCA)

### Building from Source

```bash
# Clone the repository
git clone https://github.com/lenhanpham/OpenMECP.git
cd OpenMECP

# Build release version (optimized)
cargo build --release

# Binary will be at: target/release/omecp
```

### Quick Test

```bash
# Run with example input
./target/release/omecp --help 
```

## Quick Start

### Option 1: Generate Template from Geometry File (Recommended)

Create a template input file from an existing geometry:

```bash
# From XYZ file
omecp ci molecule.xyz
# Creates: molecule.inp

# From Gaussian output (automatically adds suffix to prevent overwriting)
omecp ci calculation.log
# Creates: calculation_omecp.inp (protects original calculation.log)

# From Gaussian input
omecp ci input.gjf
# Creates: input.inp

# Custom output name
omecp ci molecule.xyz my_calculation.inp
```

**Note**: When creating templates from QM output files (`.log`, `.out`, `.json`), 
OpenMECP automatically adds an `_omecp` suffix to the input filename. This prevents 
the MECP optimization run from overwriting your original QM output files.

Then edit the generated template to customize parameters.

### Option 2: Manual Input File

Create a file named `input.inp`:

```
*geom
C     -4.108383   -0.269570   -0.105531
C     -3.086048   -0.004314   -0.907012
H     -4.464908    0.456695    0.621271
H     -4.622302   -1.226624   -0.151874
C     -2.326155    1.302381   -0.898892
H     -2.751025   -0.749763   -1.624593
H     -2.733108    1.996398   -0.159695
H     -1.274400    1.129883   -0.660641
H     -2.379567    1.780215   -1.879473
*

*tail_a
# Additional Gaussian keywords for state a
*

*tail_b
# Additional Gaussian keywords for state b
*

program = gaussian
method = B3LYP/6-31G*
nprocs = 4
mem = 4GB
charge = 0
mult_a = 1
mult_b = 3
```

### Get Help

```bash
# General help
omecp --help

# Keyword reference
omecp --help keywords

# Method reference
omecp --help methods

# Features and examples
omecp --help features
omecp --help examples

# CI command help
omecp ci --help
```

### Run OpenMECP

- Copy omecp to your preferred directory, and chmod 700 omecp before exporting this dir in your shellscript

- Prepare a shell script

```bash
# export directory where omecp is copied
export /path/to/your/dir/
module load gaussian (or Orca) // Check your HPC system
omecp input.inp > output.log 
```

### Check Results

- Optimized geometry: `{input_basename}_mecp.xyz` (e.g., `compound_xyz_123_mecp.xyz`)
- Intermediate geometries: `running_dir/` directory
- Convergence information: stdout

## Configuration File

### Create Configuration File

OpenMECP uses a configuration file `omecp_config.cfg` to customize program behavior. Create a template:

```bash
omecp ci omecp_config.cfg
```

This creates a comprehensive configuration template with all available options and detailed comments.

### Configuration File Format

The configuration file uses INI format with the following sections:

```ini
[extensions]
# Output file extensions for different QM programs
gaussian = log
orca = out
custom = log

[general]
# General program settings
max_memory = 4GB
default_nprocs = 4
print_level = 0

[logging]
# Logging configuration
level = info
file_logging = false

[cleanup]
# Automatic file cleanup configuration
enabled = true
preserve_extensions = xyz,backup
verbose = 1
cleanup_frequency = 5
```

### Configuration Precedence

Configuration files are loaded in hierarchical order (local settings override system settings):

1. **Local**: `./omecp_config.cfg` (current directory)
2. **User**: `~/.config/omecp/omecp_config.cfg` (Unix) or `%APPDATA%/omecp/omecp_config.cfg` (Windows)
3. **System**: `/etc/omecp/omecp_config.cfg` (Unix) or `%PROGRAMDATA%/omecp/omecp_config.cfg` (Windows)
4. **Built-in defaults**: Used if no configuration file is found

### Key Configuration Options

- **File Logging**: Enable with `file_logging = true` to create debug log files with dynamic names (`omecp_debug_<input_basename>.log`)
- **Print Level**: Control verbosity (`0=quiet`, `1=normal`, `2=verbose`)
- **Cleanup Frequency**: Set how often to clean up temporary files during optimization
- **Preserve Extensions**: Add custom file extensions to preserve during cleanup

### Configuration Display

When OpenMECP starts, it automatically displays:

- The source of configuration (which file was loaded, or built-in defaults)
- All input parameters with their values
- All convergence thresholds
- All configuration file settings

This ensures users always know what settings are being used.

## Input File Format

The input file consists of several sections marked by `*SECTION` and terminated by `*`.

### Required Sections

#### *GEOM Section

Initial geometry in Cartesian coordinates (Angstroms):

```
*GEOM
Element  X  Y  Z
C     -4.108383   -0.269570   -0.105531
C     -3.086048   -0.004314   -0.907012
H     -4.464908    0.456695    0.621271
H     -4.622302   -1.226624   -0.151874
C     -2.326155    1.302381   -0.898892
H     -2.751025   -0.749763   -1.624593
H     -2.733108    1.996398   -0.159695
H     -1.274400    1.129883   -0.660641
H     -2.379567    1.780215   -1.879473
*
```

#### *TAIL_a and *TAIL_b Sections

Additional keywords for each state:

```
*TAIL_a
# Gaussian: Additional route section keywords
# ORCA: Additional input blocks
*

*TAIL_b
# Same format as TAIL_a
*
```

### Optional Sections

#### *CONSTR Section

Constraints for optimization:

```
*CONSTR
R 1 2 1.5        # Fix bond between atoms 1-2 at 1.5 Å
A 1 2 3 90.0     # Fix angle 1-2-3 at 90 degrees
s r 1 2 1.0 20 0.05   # Scan bond from 1.0 to 1.95 Å (20 points)
s a 1 2 3 90 10 5     # Scan angle from 90° to 135° (10 points)
*
```

#### *LST_a and *LST_b Sections

For LST interpolation:

```
*LST_a
C  0.0  0.0  0.0
H  1.0  0.0  0.0
*

*LST_b
C  0.0  0.0  0.5
H  1.2  0.0  0.5
*
```

## Keywords Reference

### Required Keywords

| Keyword   | Type    | Description                  | Example            |
| --------- | ------- | ---------------------------- | ------------------ |
| `program` | string  | QM program to use            | `gaussian`, `orca` |
| `method`  | string  | QM method and basis set      | `B3LYP/6-31G*`     |
| `nprocs`  | integer | Number of processors         | `4`                |
| `mem`     | string  | Memory allocation            | `4GB` `4000`       |
| `charge`  | integer | Molecular charge             | `0`                |
| `mult_a`  | integer | Spin multiplicity of state A | `1`                |
| `mult_b`  | integer | Spin multiplicity of state B | `3`                |

**Note**: method and mem should be specific for Gaussian and Orca. For example: 

- Gaussian: 
  
  ```
  mem = 8GB
  method = n scf(maxcycle=500,xqc) uwb97xd/def2svpp scrf=(smd,solvent=acetonitrile)
  ```

- Orca: 
  
  ```
  method = B3LYP def2-SVP CPCM(2-octanone) VeryTightSCF  --> All keywords after ! in Orca input
  mem = 8000 --> (memory 8000 MB for each core <=> %maxcore 8000)
  ```

### Optional Keywords

#### TD-DFT and State Selection

| Keyword   | Type    | Default | Description                                |
| --------- | ------- | ------- | ------------------------------------------ |
| `td_a`    | string  | `""`    | TD-DFT keywords for state A (short form)   |
| `td_b`    | string  | `""`    | TD-DFT keywords for state B (short form)   |
| `state_a` | integer | `0`     | Excited state index for state A (0=ground) |
| `state_b` | integer | `0`     | Excited state index for state B (0=ground) |
| `mp2`     | boolean | `false` | Use MP2 instead of DFT (Gaussian only)     |

**Note**: You can also use long forms `td_state_a` and `td_state_b` instead of `td_a` and `td_b`.

#### Run Mode and Optimization

| Keyword         | Type    | Default  | Description                            |
| --------------- | ------- | -------- | -------------------------------------- |
| `mode`          | string  | `normal` | Run mode (see [Run Modes](#run-modes)) |
| `max_steps`     | integer | `100`    | Maximum optimization steps             |
| `max_step_size` | float   | `0.1`    | Maximum step size (Bohr)               |
| `restart`       | boolean | `false`  | Enable restart from checkpoint file    |

#### Optimizer Settings

| Keyword                     | Type        | Default            | Description                                                                                   |
| --------------------------- | ----------- | ------------------ | --------------------------------------------------------------------------------------------- |
| `hessian`                   | string      | `direct_psb`       | Hessian update method: `direct_psb`, `inverse_bfgs`, `bofill`, `powell`, `bfgs_powell_mix`    |
| `use_gediis`                | bool/string | `false`            | `false`/`none`=GDIIS, `true`/`sequential`=GEDIIS, `blend`=GDIIS_blend with trust region       |
| `use_hybrid_gediis`         | boolean     | `false`            | Sequential hybrid GDIIS/GEDIIS (requires `use_gediis=true` or `sequential`)                   |
| `gediis_blend_mode`         | string      | `fixed_sequential` | Blend strategy when `use_gediis=blend`: `fixed`, `fixed_sequential`, `gradient`, `sequential` |
| `gediis_switch_rms`         | float       | `0.005`            | RMS gradient threshold for GDIIS→GEDIIS switch (paper: 10⁻² au)                               |
| `gediis_switch_step`        | float       | `0.001`            | RMS step threshold for GEDIIS→GDIIS switch (paper: 2.5×10⁻³ au)                               |
| `switch_step`               | integer     | `3`                | Step to switch from BFGS to DIIS optimizers                                                   |
| `max_history`               | integer     | `4`                | Max iterations used for DIIS extrapolation                                                    |
| `smart_history`             | boolean     | `false`            | Smart history instead of first in first out                                                   |
| `reduced_factor`            | float       | `0.5`              | Step size reduction factor for GDIIS                                                          |
| `step_reduction_multiplier` | float       | `10.0`             | RMS gradient multiplier for step reduction threshold                                          |
| `steepest_descent_step`     | float       | `0.01`             | Small steepest descent step size (Å) for fallback when DIIS/BFGS fails                        |
| `bfgs_rho`                  | float       | `15.0`             | Scaling factor for BFGS step size                                                             |
| `trust_reduction_factor`    | float       | `0.5`              | Trust radius contraction factor when energy increases                                         |
| `trust_increase_factor`     | float       | `1.2`              | Trust radius expansion factor when energy decreases                                           |
| `trust_inc_threshold`       | float       | `0.0001`           | Energy increase threshold (Ha) for trust radius reduction                                     |
| `trust_dec_threshold`       | float       | `0.0001`           | Energy decrease threshold (Ha) for trust radius increase                                      |
| `trust_min_radius`          | float       | `0.01`             | Minimum trust radius in Å                                                                     |
| `trust_max_radius`          | float       | `1.0`              | Maximum trust radius in Å                                                                     |
| `print_level`               | integer     | `1`                | Output verbosity: `0`=quiet, `1`=normal, `2`=verbose (DIIS debug)                             |
| `print_checkpoint`          | boolean     | `false`            | Enable/disable checkpoint JSON file generation                                                |

#### Advanced DIIS Options (Step Validation)

Standard GDIIS/GEDIIS can occasionally produce wild extrapolations when the DIIS matrix is ill-conditioned (e.g., near-convergence or redundant history). These options activate additional sanity checks that reject bad steps.

**Master switch:** `use_robust_diis = true` — the options below only take effect when this is enabled.

| Keyword              | Type    | Default      | Activation                                            | Description                                              |
| -------------------- | ------- | ------------ | ----------------------------------------------------- | -------------------------------------------------------- |
| `use_robust_diis`    | boolean | `false`      | standalone                                            | **Master switch** for DIIS step validation               |
| `gediis_variant`     | string  | `"auto"`     | requires `use_robust_diis=true` AND `use_gediis=true` | `auto`, `rfo`, `energy`, `simultaneous`                  |
| `gdiis_cosine_check` | string  | `"standard"` | requires `use_robust_diis=true`                       | Cosine threshold for step acceptance (see below)         |
| `gdiis_coeff_check`  | string  | `"regular"`  | requires `use_robust_diis=true`                       | DIIS coefficient bounds (see below)                      |
| `n_neg`              | integer | `0`          | requires `use_robust_diis=true`                       | `0` for minimum search, `1` for transition-state search  |
| `gediis_sim_switch`  | float   | `0.0025`     | requires `use_robust_diis=true`                       | RMS error threshold controlling GEDIIS variant switching |

**How step validation works:**

When a DIIS extrapolation produces a candidate step, the robust DIIS system checks:

1. **Cosine check** (`gdiis_cosine_check`): Computes the cosine between the DIIS step and the reference gradient. A low cosine (< threshold) indicates that DIIS is extrapolating in a direction inconsistent with the local gradient, suggesting the step is unreliable.
   
   - `none`: Skip cosine check entirely
   - `zero`: Reject when cosine is negative (step points uphill)
   - `standard` (default): Require cosine ≥ 0.71 (45° angle) — balances safety with convergence speed
   - `variable`: Use adaptive threshold: 0.1 far from minimum, tightening to 0.7 near convergence. More permissive in early steps where gradients are large.
   - `strict`: Require cosine ≥ 0.866 (30° angle) — conservative, good for fragile systems

2. **Coefficient check** (`gdiis_coeff_check`): Validates that the DIIS interpolation coefficients are physically reasonable. DIIS ideally interpolates (coefficients sum to 1, individual cᵢ between 0 and 1). Deviation signals ill-conditioning.
   
   - `none`: Accept any coefficients (even wildly negative or > 1)
   - `regular` (default): Reject if coefficients deviate beyond reasonable bounds. Accepts moderate negative coefficients but flags extreme values.
   - `force_recent`: Strongly bias toward the latest point. Useful when history contains outdated geometries.
   - `combined`: Both regular bounds and cosine threshold applied together — the most restrictive.

3. **n_neg control**: Set `n_neg = 1` to treat the optimization as a **transition-state search** (seeking a saddle point with one negative Hessian eigenvalue). Default `n_neg = 0` is for minimum-energy crossing point search.

**GEDIIS Variants** (only active when `use_robust_diis=true` AND `use_gediis=true`):

- `auto`: Automatically selects the best variant based on RMS error and energy trend. Starts with `energy` for robust early convergence, switches to `rfo` when close to the minimum.
- `rfo`: Rational Function Optimization with DIIS. Uses quadratic step overlaps for the B-matrix. Performs best near the minimum where the quadratic approximation is accurate.
- `energy`: Energy-DIIS using gradient-coordinate products for the B-matrix. More robust far from the minimum, where the PES is less quadratic.
- `simultaneous`: Combines Energy-DIIS with a quadratic energy approximation. The most sophisticated variant, blending the advantages of both `rfo` and `energy`.

**When to use robust DIIS:**

- **Difficult convergence**: If the optimizer oscillates or takes wild steps near convergence, enable `use_robust_diis = true` with `standard` settings.
- **Fragile calculations**: For systems with ill-conditioned Hessians (e.g., flat PES, near-dissociation), use `gdiis_cosine_check = strict`.
- **Restart calculations**: On restart, the history may contain stale points. Use `gdiis_coeff_check = force_recent` to favour the latest geometry.

**Example — activating step validation:**

```
use_robust_diis = true
use_gediis = true
gediis_variant = energy
gdiis_cosine_check = standard
```

#### Hessian Update Methods

The `hessian` keyword controls how the Hessian matrix is stored and updated:

- `direct_psb` **(default, recommended)**: Direct Hessian + PSB (Powell-Symmetric-Broyden) update. Stores the full Hessian H (Ha/Å²) and solves `H·dk = -g` via LU decomposition. Required when `use_gediis = blend`.
- `inverse_bfgs`**: Stores inverse Hessian H⁻¹ (Å²/Ha) and computes steps as `H⁻¹·g` via simple matrix-vector multiply. **Incompatible with the blend optimizer**.
- `bofill` **(experimental)**: Bofill weighted update blending Powell and Murtagh-Sargent formulas. Recommended for TS-like crossing points. Direct Hessian.
- `powell` **(experimental)**: Symmetric rank-one (SR1) update. Tolerates negative curvature. Direct Hessian.
- `bfgs_powell_mix` **(experimental)**: Adaptive blend switching between BFGS and Powell using Bofill-style weighting. Direct Hessian.

**Example — switching to Bofill:**

```
hessian = bofill
```

#### Convergence Thresholds

| Keyword    | Type  | Default    | Description                           |
| ---------- | ----- | ---------- | ------------------------------------- |
| `delta_e`  | float | `0.000050` | Energy difference threshold (hartree) |
| `rms_dis`  | float | `0.0025`   | RMS displacement threshold (Å)        |
| `max_dis`  | float | `0.004`    | Max displacement threshold (Å)        |
| `max_grad` | float | `0.001323` | Max gradient threshold (hartree/Å)    |
| `rms_grad` | float | `0.000945` | RMS gradient threshold (hartree/Å)    |

#### Constraints and Fixed Atoms

| Keyword      | Type   | Default | Description                          |
| ------------ | ------ | ------- | ------------------------------------ |
| `fixedatoms` | string | `""`    | Fixed atom indices (e.g., `1,3-5,7`) |

#### Coordinate Driving

| Keyword            | Type    | Default | Description                                           |
| ------------------ | ------- | ------- | ----------------------------------------------------- |
| `drive_type`       | string  | `""`    | Coordinate type (`bond`, `angle`, `dihedral`)         |
| `drive_atoms`      | string  | `""`    | Atom indices for coordinate driving (comma-separated) |
| `drive_start`      | float   | `0.0`   | Starting value for coordinate driving                 |
| `drive_end`        | float   | `0.0`   | Ending value for coordinate driving                   |
| `drive_steps`      | integer | `10`    | Number of steps for coordinate driving                |
| `drive_coordinate` | string  | `""`    | Reaction coordinate specification                     |

#### Fix-dE Mode

| Keyword  | Type  | Default | Description                   |
| -------- | ----- | ------- | ----------------------------- |
| `fix_de` | float | `0.0`   | Target energy difference (eV) |

#### Program Commands

| Keyword     | Type   | Default  | Description      |
| ----------- | ------ | -------- | ---------------- |
| `gau_comm`  | string | `"g16"`  | Gaussian command |
| `orca_comm` | string | `"orca"` | ORCA command     |

#### ONIOM-Specific

| Keyword                  | Type    | Default | Description                             |
| ------------------------ | ------- | ------- | --------------------------------------- |
| `isoniom`                | boolean | `false` | Enable ONIOM (QM/MM) calculation        |
| `chargeandmultforoniom1` | string  | `""`    | Charge/multiplicity for state A (ONIOM) |
| `chargeandmultforoniom2` | string  | `""`    | Charge/multiplicity for state B (ONIOM) |

#### Advanced Settings

| Keyword                 | Type    | Default | Description                             |
| ----------------------- | ------- | ------- | --------------------------------------- |
| `charge_b`              | integer | —       | Separate charge for state B             |
| `basis`                 | string  | `""`    | Basis set specification                 |
| `solvent`               | string  | `""`    | Solvent model specification             |
| `dispersion`            | string  | `""`    | Dispersion correction                   |
| `custom_interface_file` | string  | `""`    | Path to custom QM interface JSON config |
| `fixedatoms`            | string  | `""`    | Fixed atom indices (e.g., `1,3-5,7`)    |

#### Optimizer Switching Control

The `switch_step` parameter provides full control over the optimizer switching strategy used in MECP optimization. OpenMECP uses a hybrid approach that combines the stability of BFGS with the speed of DIIS methods.

**Three Optimization Modes:**

1. **Hybrid Mode (Default)**: `switch_step = 3`
   
   - Uses BFGS for the first 3 steps to build curvature information
   - Switches to DIIS (GDIIS or GEDIIS) for faster convergence
   - Recommended for most calculations

2. **DIIS-Only Mode**: `switch_step = 0`
   
   - Uses DIIS from the first step
   - Faster but may be less stable for difficult cases
   - Requires good initial geometry

3. **BFGS-Only Mode**: `switch_step >= max_steps`
   
   - Uses only BFGS throughout the optimization
   - Most stable but slower convergence
   - Recommended for very difficult optimizations

4. **`smart_history`** can speed up MECP calculations for most of the cases 

**Examples:**

```
switch_step = 3     # Default: BFGS for 3 steps, then DIIS
switch_step = 0     # Pure DIIS (fastest)
switch_step = 10    # Extended BFGS for 10 steps, then DIIS
switch_step = 999   # Pure BFGS (most stable)
```

**Performance Comparison:**

- BFGS: Baseline convergence rate
- GDIIS: ~2-3x faster than BFGS
- Sequential Hybrid: ~2-4x faster than GDIIS (default, uses GEDIIS at optimal stage)
- Pure GEDIIS: Comparable to hybrid, may converge differently on some systems

### Constraint Syntax

#### Bond Constraint

```
R atom1 atom2 target_distance
```

Example: `R 1 2 1.5` (fix C-H bond at 1.5 Å)

#### Angle Constraint

```
A atom1 atom2 atom3 target_angle
```

Example: `A 1 2 3 90.0` (fix angle at 90°)

#### Bond Scan

```
s r atom1 atom2 start num_points step_size
```

Example: `s r 1 2 1.0 20 0.05` (scan from 1.0 to 1.95 Å)

#### Angle Scan

```
s a atom1 atom2 atom3 start num_points step_size
```

Example: `s a 1 2 3 90 10 5` (scan from 90° to 135°)

#### Dihedral Constraint

```
D atom1 atom2 atom3 atom4 target_dihedral
```

Example: `D 1 2 3 4 180.0` (fix dihedral at 180°)

#### Dihedral Scan

```
s d atom1 atom2 atom3 atom4 start num_points step_size
```

Example: `s d 1 2 3 4 0 36 10` (scan from 0° to 350°)

## Supported QM Programs

### Gaussian

**Supported Methods**: DFT, TD-DFT, MP2, CASSCF

**Input Format**: Gaussian .gjf files

**Example**:

```
program = gaussian
method = B3LYP/6-31G*
td_a = TD(nstates=5,root=1)
td_b = TD(nstates=5,root=2)
```

**Output Files**: `.log` files in `running_dir/` directory

**Checkpoint Files**: `state_A.chk` (state A), `state_B.chk` (state B)

### ORCA

**Supported Methods**: DFT, TD-DFT, CASSCF

**Input Format**: ORCA .inp files

**Example**:

```
program = orca
method = B3LYP def2-SVP

*TAIL_a
%tddft
  nroots 5
  iroot 1
end
*

*TAIL_b
%tddft
  nroots 5
  iroot 2
end
*
```

**Output Files**: `.log` and `.engrad` files in `running_dir/` directory

**Checkpoint Files**: `state_A.gbw` (state A), `state_B.gbw` (state B)

## Run Modes

### Normal Mode (Default)

```
mode = normal
```

- Standard MECP optimization
- Reads checkpoint files from previous steps
- Recommended for most calculations

### Read Mode

```
mode = read
```

- Reads existing checkpoint files
- Skips pre-point calculations
- Use for restarting calculations

### NoRead Mode

```
mode = noread
```

- Fresh SCF at each step
- Does not read checkpoint files
- Slower but more robust
- Use for difficult SCF convergence

### Stable Mode

```
mode = stable
```

- Runs pre-point calculations first
- Checks SCF stability
- Recommended for unstable wavefunctions

### InterRead Mode

```
mode = inter_read
```

- Runs state B first
- Copies B wavefunction to A
- Adds `guess=(read,mix)` for state A
- **Essential for open-shell singlet (OSS) states**

**Example for OSS**:

```
mode = inter_read
mult_a = 1    # Open-shell singlet
mult_b = 3    # Triplet reference
```

## Advanced Features

### PES Scan

Perform 1D or 2D potential energy surface scans:

**1D Bond Scan**:

```
*CONSTR
s r 1 2 1.0 20 0.05    # Scan C-H bond from 1.0 to 1.95 Å
*
```

**2D Scan**:

```
*CONSTR
s r 1 2 1.0 10 0.1    # First dimension: bond
s a 1 2 3 90 5 10     # Second dimension: angle
*
```

**Output**: Geometries saved as `scan_{value1}_{value2}.xyz`

### LST Interpolation

Generate initial MECP guess by interpolating between two geometries:

```
*LST_a
C  0.0  0.0  0.0
H  1.0  0.0  0.0
*

*LST_b
C  0.0  0.0  0.5
H  1.2  0.0  0.5
*
```

**Features**:

- Kabsch algorithm for optimal alignment
- Linear (LST) and Quadratic (QST) interpolation methods
- 10 interpolation points (default)
- Energy profile calculation
- Interactive confirmation

**Note**: QST interpolation uses a midpoint approximation for the third geometry. Full quadratic interpolation with explicit midpoint geometry is planned for a future release.

**Output**: Input files in `running_dir/` directory, energy profile to stdout

### Fixed Atoms

Freeze specific atoms during optimization:

```
fixedatoms = 1,3-5,7    # Fix atoms 1, 3, 4, 5, and 7
```

**Use Cases**:

- Partial optimization
- QM/MM boundaries
- Surface calculations

### GDIIS and GEDIIS Optimizers

**GDIIS (Geometry-based DIIS) — Default** (`use_gediis = false`):

The geometry DIIS method solves for interpolation coefficients **c** that minimize a
linear combination of error vectors subject to ∑cᵢ = 1:

```
Error vectors:      eᵢ = H⁻¹ · (gᵢ + fᵢ)         [Newton-step on combined forces]
B-matrix:           Bⱼₖ = ⟨eⱼ | eₖ⟩
Linear system:      ┌        ┐ ┌    ┐   ┌  ┐
                    │ B   1  │ │ c  │   │ 0│
                    │ 1ᵀ  0  │ │ λ  │ = │ 1│
                    └        ┘ └    ┘   └  ┘
Interpolated geom:  x* = ∑cᵢ·xᵢ
Newton correction:  x_GDIIS = x* - H⁻¹ · P(x*)
```

where P(x*) is the interpolated combined force. GDIIS minimizes residual forces
via Newton-step corrections, giving robust quadratic convergence. Proven reliable
for ~10-step convergence on test systems.

**GEDIIS (Energy-informed DIIS)** (`use_gediis = true` + `use_hybrid_gediis = false`):

Extends GDIIS by adding an energy-based diagonal bias:

```
Bⱼₖ = ⟨eⱼ | eₖ⟩                           (same as GDIIS)
Bᵢᵢ += α · Eᵢ²                              (energy diagonal coupling)
RHSᵢ  = -Eᵢ                                  (RHS now includes energy)
```

where Eᵢ = |E1ᵢ - E2ᵢ| is the energy gap at point i. The diagonal term `α·Eᵢ²`
biases the interpolation toward low-gap points (where the MECP lies), making it
"energy-informed." The modified RHS `-Eᵢ` shifts the interpolation toward
geometries with smaller energy gaps.

Key constraint: `cᵢ > 0` (interpolation only, no extrapolation beyond convex hull).

**Sequential Hybrid GEDIIS/GDIIS** (`use_gediis = true` + `use_hybrid_gediis = true`):

A 3-phase switching strategy (Li & Frisch JCTC 2006 Sec II.B):

1. **Phase 1 — GDIIS**: Active while RMS gradient > `gediis_switch_rms` (0.005 Ha/Å).
   GDIIS converges robustly from poor starting geometries using Newton-step
   force minimization.

2. **Phase 2 — GEDIIS**: Activates when RMS gradient falls below `gediis_switch_rms`.
   The energy-diagonal bias guides interpolation toward the crossing seam,
   improving the energy gap while maintaining gradient convergence.

3. **Phase 3 — GDIIS**: Activates when RMS displacement < `gediis_switch_step`
   (0.001 Å). Switches back to pure GDIIS for clean quadratic final convergence,
   avoiding the plateau problem where 50/50 blends stall.

**GDIIS_blend Optimizer** (`use_gediis = "blend"`):

A per-step blend of GDIIS and EDIIS steps with a trust region. At each step,
both the GDIIS geometry (`x_GDIIS`, Newton-corrected interpolation) and EDIIS
geometry (`x_EDIIS`, energy-informed interpolation) are computed independently,
then blended together:

```
x_new = w · x_EDIIS  +  (1 - w) · x_GDIIS
```

where `w` is the EDIIS weight (0 = pure GDIIS, 1 = pure EDIIS). The blend
strategy is selected by `gediis_blend_mode` and only activates when
`use_hybrid_gediis = true`. When `use_hybrid_gediis = false`, the optimizer
uses pure GDIIS_blend (GDIIS step with an inverted mean true Hessian and
trust radius, no EDIIS component at all).

**Blend Modes** (all require `use_gediis = "blend"` + `use_hybrid_gediis = true`):

| Mode                         | Weight formula                                                     | Behavior                                                                                                                                                                  |
| ---------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fixed`                      | `w = 0.5`                                                          | Constant 50/50 GDIIS+EDIIS. Simple but may plateau                                                                                                                        |
| `fixed_sequential` (default) | `w = 0.5` while `RMS_disp ≥ switch_step`, then `w = 0`             | 50/50 blend far from minimum, transitions to pure GDIIS near convergence for quadratic finishing                                                                          |
| `gradient`                   | `w = RMS_g / (RMS_g + switch_rms)`                                 | Smooth sigmoid: EDIIS-dominated when forces are large (uses energy info to navigate toward seam), smoothly transitions to GDIIS as forces decrease for Newton convergence |
| `sequential`                 | Stepwise: `w = 1` (EDIIS) if step decreasing, else `w = 0` (GDIIS) | Binary per-step selection based on RMS displacement trend                                                                                                                 |

where `switch_rms = gediis_switch_rms` (default 0.005 Ha/Å) and
`switch_step = gediis_switch_step` (default 0.001 Å).

**Gradient blend form** (`gediis_blend_mode = gradient`):

```
w = RMS_grad / (RMS_grad + gediis_switch_rms)
```

Unlike `fixed`, this smoothly and continuously varies the EDIIS weight from
~1 (dominant EDIIS when far) to ~0 (dominant GDIIS near convergence) with no
hard switching threshold. The `gediis_switch_rms` parameter controls the
transition midpoint (w = 0.5 when RMS_grad = switch_rms).

- **Large forces (far)**: `RMS_g ≫ switch_rms` → `w ≈ 1` → mostly EDIIS,
  using energy information to find the crossing seam
- **Small forces (near)**: `RMS_g ≪ switch_rms` → `w ≈ 0` → mostly GDIIS,
  Newton-step minimization for clean final convergence

**Trust radius**: All blend modes use a trust radius that:

- Contracts to `trust_reduction_factor × radius` when energy increases
- Expands by `trust_increase_factor × radius` when energy decreases
- Bounds between `trust_min_radius` and `trust_max_radius`
- Caps the displacement norm to prevent wild steps
- Bootstraps from the BFGS `max_step_size`

**Requirements**: Blend mode requires a direct Hessian method
(`direct_psb`, `bofill`, `powell`, or `bfgs_powell_mix`).
`inverse_bfgs` is incompatible and raises an error.

### State Selection for TD-DFT

Choose specific excited states for TD-DFT calculations:

```
state_a = 1    # Use first excited state for state 1
state_b = 2    # Use second excited state for state 2
td_a = TD(nstates=5,root=1)
td_b = TD(nstates=5,root=2)
```

**Use Cases**:

- Avoid ground state contamination
- Target specific conical intersections
- Study multi-state reactivity

### Coordinate Driving

Systematically explore reaction coordinates:

```
drive_type = bond
drive_atoms = 1,2
drive_start = 1.0
drive_end = 2.0
drive_steps = 20
```

**Supported coordinate types**:

- `bond`: Bond length between two atoms
- `angle`: Bond angle between three atoms
- `dihedral`: Dihedral angle between four atoms

**Output**: Energy profiles and geometries along the reaction path

### Path Optimization (NEB)

Optimize reaction paths using the Nudged Elastic Band method:

```
run_mode = path_optimization
drive_type = bond
drive_atoms = 1,2
drive_start = 1.0
drive_end = 2.0
drive_steps = 10
```

**Features**:

- Creates initial path via coordinate driving
- Optimizes path using simplified NEB algorithm with spring forces
- Provides smooth energy profiles
- Identifies transition states and intermediates

**Note**: Current implementation uses a simplified NEB model based on spring forces. Full NEB with perpendicular energy gradient projection is planned for a future release.

### Fix-dE Optimization

Constrain the energy difference to a target value:

```
run_mode = fix_de
fix_de = 0.1    # Target ΔE = 0.1 eV
```

**⚠️ Currently Unavailable**: Fix-dE optimization is temporarily disabled and needs to be reimplemented with the new constraint handling system. This feature will be available in a future release.

**Planned Use Cases**:

- Study avoided crossings
- Generate diabatic potential energy surfaces
- Investigate spin-orbit coupling effects

### External Geometry Reader

Read geometries from external XYZ files:

```
*GEOM
@geometry.xyz
*
```

**Format**: Standard XYZ format with Angstrom units

### ONIOM Support

Multi-layer QM/MM calculations with Gaussian:

```
*GEOM
C  0.0  0.0  0.0  H
H  1.0  0.0  0.0  L
H  0.0  1.0  0.0  L
*

oniom_layer_info = H,L
charge_and_mult_oniom1 = 0 1 H 0 1 L
charge_and_mult_oniom2 = 0 3 H 0 3 L
```

**Layer labels**: H (High), M (Medium), L (Low)

## Output Files

### Standard Output (stdout)

```
**** OpenMECP in Rust****
****By Le Nhan Pham****

Parsed 3 atoms
Program: Gaussian, Mode: Normal

****Running initial calculations****

****Step 1****
Using BFGS optimizer
E1 = -40.123456, E2 = -40.234567, ΔE = 0.111111

****Step 2****
...

Converged at step 8
****Congrats! MECP has converged****
```

### Final MECP Geometry File

Final optimized geometry in XYZ format: `{input_basename}_mecp.xyz`

For example, if your input file is `compound_xyz_123.input`, the final geometry will be saved as `compound_xyz_123_mecp.xyz`.

```
10

C     -0.004656   -0.096904    0.392462
C      1.463039   -0.021272    0.493374
H      1.796216   -0.608655    1.352230
H      1.896006   -0.477168   -0.399749
C      1.974429    1.428606    0.618264
N      1.541011    2.045670    1.884895
H      3.067296    1.425355    0.583735
H      1.604312    2.023629   -0.221614
H      1.907833    1.518644    2.675008
H      1.907848    2.992783    1.956802
```

### {input_basename}/ Directory

Contains all intermediate calculations, where `{input_basename}` is the stem of your input file (e.g., `compound_x` for `compound_x.inp`):

```
{input_basename}/
├── 0_A.{ext}          # Initial state A input
├── 0_A.{log,out}      # Initial state A output
├── 0_B.{ext}          # Initial state B input
├── 0_B.{log,out}      # Initial state B output
├── 1_A.{ext}          # Step 1 state A input
├── 1_A.{log,out}      # Step 1 state A output
├── 0_A.gbw            # ORCA wavefunction (chain-linked)
├── {input_basename}_state_A.chk    # Gaussian state A checkpoint
└── {input_basename}_state_B.chk    # Gaussian state B checkpoint
```

### Scan Output

For PES scans:

```
scan_1.0000_0.0000.xyz
scan_1.0500_0.0000.xyz
...
scan_1.9500_0.0000.xyz
```

## Convergence Criteria

All five criteria must be satisfied:

| Criterion        | Threshold          | Description                    |
| ---------------- | ------------------ | ------------------------------ |
| ΔE               | < 0.000050 hartree | Energy difference              |
| RMS gradient     | < 0.000945         | Root mean square gradient      |
| Max gradient     | < 0.001323         | Maximum gradient component     |
| RMS displacement | < 0.0025           | Root mean square displacement  |
| Max displacement | < 0.004            | Maximum displacement component |

**Note**: These are the same criteria used in Gaussian optimizations.

## Examples

### Example 1: Basic MECP (Singlet-Triplet, pure GDIIS)

```
*GEOM
C     -0.004656   -0.096904    0.392462
C      1.463039   -0.021272    0.493374
H      1.796216   -0.608655    1.352230
H      1.896006   -0.477168   -0.399749
C      1.974429    1.428606    0.618264
N      1.541011    2.045670    1.884895
H      3.067296    1.425355    0.583735
H      1.604312    2.023629   -0.221614
H      1.907833    1.518644    2.675008
H      1.907848    2.992783    1.956802
*

*TAIL_a
*

*TAIL_b
*

program = gaussian
method = B3LYP/6-31G*
nprocs = 4
mem = 4GB
charge = 0
mult_a = 1
mult_b = 3
```

### Example 2: TD-DFT Excited States

```
*GEOM
C     -0.004656   -0.096904    0.392462
C      1.463039   -0.021272    0.493374
H      1.796216   -0.608655    1.352230
H      1.896006   -0.477168   -0.399749
C      1.974429    1.428606    0.618264
N      1.541011    2.045670    1.884895
H      3.067296    1.425355    0.583735
H      1.604312    2.023629   -0.221614
H      1.907833    1.518644    2.675008
H      1.907848    2.992783    1.956802
*

*TAIL_a
*

*TAIL_b
*

program = gaussian
method = B3LYP/6-31G*
td_a = TD(nstates=5,root=1)
td_b = TD(nstates=5,root=2)
nprocs = 8
mem = 8GB
charge = 0
mult_a = 1
mult_b = 1
```

### Example 3: Open-Shell Singlet with ORCA

```
*GEOM
C     0.003353000000     -0.614939000000     -0.311569000000
C     1.191330000000     -1.267961000000      0.432978000000
C     1.907393000000      0.845669000000      0.107379000000
C     0.477702000000      0.869192000000     -0.513933000000
H    -0.186004000000     -1.091217000000     -1.276563000000
H     0.541768000000      1.092853000000     -1.586485000000
C     1.716461000000     -0.108556000000      1.295995000000
H     2.657461000000     -0.353238000000      1.797365000000
H     1.000889000000      0.253006000000      2.042673000000
C     2.331293000000     -1.439998000000     -0.591244000000
H     1.987535000000     -1.925810000000     -1.508845000000
C     2.817464000000      0.018948000000     -0.822489000000
H     0.928068000000     -2.190212000000      0.955676000000
H     2.288053000000      1.843003000000      0.329772000000
H     2.733104000000      0.335451000000     -1.866092000000
S    -1.529023000000     -0.875105000000      0.662585000000
S    -3.035841000000     -0.450291000000     -0.538423000000
S    -0.535554000000      2.209136000000      0.139010000000
H     3.124215000000     -2.058494000000     -0.161005000000
H     3.861645000000      0.140669000000     -0.519951000000
*

*TAIL_a
%scf
  maxiter 200
end
*

*TAIL_b
%scf
  maxiter 200
end
*

program = orca
method = B3LYP def2-SVP
mode = inter_read
nprocs = 8
mem = 8GB
charge = 0
mult_a = 1
mult_b = 3
```

### Example 4: PES Scan with Constraints

```
*GEOM
O      0.074139    0.074139    0.000000
H      1.136608   -0.210747    0.000000
H     -0.210747    1.136608    0.000000
*

*CONSTR
s r 1 2 1.0 20 0.05    # Scan O-H bond
R 1 3 1.1              # Fix other O-H bond
*

*TAIL_a
*

*TAIL_b
*

program = gaussian
method = B3LYP/6-31G*
nprocs = 4
mem = 4GB
charge = 0
mult_a = 1
mult_b = 3
```

### Example 5: LST Interpolation

```
*LST_a
O      0.074139    0.074139    0.000000
H      1.136608   -0.210747    0.000000
H     -0.210747    1.136608    0.000000
*

*LST_b
O      0.074139    0.074139    0.000000
H      1.136608   -0.210747    0.000000
H     -0.210747    1.136608    0.000000
*

*TAIL_a
*

*TAIL_b
*

program = gaussian
method = B3LYP/6-31G*
nprocs = 4
mem = 4GB
charge = 0
mult_a = 1
mult_b = 3
```

### Example 6: Sequential Hybrid GEDIIS

```
*GEOM
C     0.000000000000      1.396613000000      0.000000000000
C     1.209503000000      0.698307000000      0.000000000000
C     1.209503000000     -0.698307000000      0.000000000000
C     0.000000000000     -1.396613000000      0.000000000000
C    -1.209503000000     -0.698307000000      0.000000000000
C    -1.209503000000      0.698307000000      0.000000000000
H     2.150882000000      1.241812000000      0.000000000000
H     2.150882000000     -1.241812000000      0.000000000000
H     0.000000000000     -2.483625000000      0.000000000000
H    -2.150882000000     -1.241812000000      0.000000000000
H    -2.150882000000      1.241812000000      0.000000000000
*

*TAIL_a
*

*TAIL_b
*

program = gaussian
method = B3LYP/6-31G**
nprocs = 4
mem = 4GB
charge = 1
mult_a = 1
mult_b = 3
use_gediis = true          # Enable GEDIIS
use_hybrid_gediis = true   # Sequential hybrid (3-phase switching)
gediis_switch_rms = 0.008  # GDIIS→GEDIIS when rms_g < 0.008 (default: 0.005)
gediis_switch_step = 0.002 # GEDIIS→GDIIS when rms_disp < 0.002 (default: 0.001)
```

### Example 7: TD-DFT State Selection

```
*GEOM
C      0.037739    0.000000   -0.000000
C      1.362261    0.000000   -0.000000
H     -0.524574    0.930562    0.000000
H     -0.524574   -0.930562   -0.000000
H      1.924574    0.930562   -0.000000
H      1.924574   -0.930562    0.000000
*

*TAIL_a
*

*TAIL_b
*

program = gaussian
method = B3LYP/6-31G*
td_a = TD(nstates=5,root=1)
td_b = TD(nstates=5,root=2)
state_a = 1
state_b = 2
nprocs = 8
mem = 8GB
charge = 0
mult_a = 1
mult_b = 1
```

### Example 8: Optimizer Switching Control

```
*GEOM
C      0.037739    0.000000   -0.000000
C      1.362261    0.000000   -0.000000
H     -0.524574    0.930562    0.000000
H     -0.524574   -0.930562   -0.000000
H      1.924574    0.930562   -0.000000
H      1.924574   -0.930562    0.000000
*

*TAIL_a
*

*TAIL_b
*

program = gaussian
method = B3LYP/6-31G*
nprocs = 4
mem = 4GB
charge = 0
mult_a = 1
mult_b = 3
switch_step = 0      # Pure DIIS mode (fastest)
use_gediis = true    # Use sequential hybrid GEDIIS/GDIIS (default)
# Phase switching thresholds:
gediis_switch_rms = 0.01    # GDIIS→GEDIIS when rms_g < 0.01 Ha/Å
gediis_switch_step = 0.002  # GEDIIS→GDIIS when rms_disp < 0.002 Å
```

### Example 9: Coordinate Driving

```
*GEOM
C      0.037739    0.000000   -0.000000
C      1.362261    0.000000   -0.000000
H     -0.524574    0.930562    0.000000
H     -0.524574   -0.930562   -0.000000
H      1.924574    0.930562   -0.000000
H      1.924574   -0.930562    0.000000
*

*TAIL_a
*

*TAIL_b
*

program = gaussian
method = B3LYP/6-31G*
nprocs = 4
mem = 4GB
charge = 0
mult_a = 1
mult_b = 3
run_mode = coordinate_drive
drive_type = bond
drive_atoms = 1,2
drive_start = 1.0
drive_end = 2.0
drive_steps = 20
```

### Example 10: Fix-dE Optimization

```
*GEOM
C      0.037739    0.000000   -0.000000
C      1.362261    0.000000   -0.000000
H     -0.524574    0.930562    0.000000
H     -0.524574   -0.930562   -0.000000
H      1.924574    0.930562   -0.000000
H      1.924574   -0.930562    0.000000
*

*TAIL_a
*

*TAIL_b
*

program = gaussian
method = B3LYP/6-31G*
nprocs = 4
mem = 4GB
charge = 0
mult_a = 1
mult_b = 3
run_mode = fix_de
fix_de = 0.1
```

### Example 11: External Geometry File

```
*GEOM
@my_molecule.xyz
*

*TAIL_a
*

*TAIL_b
*

program = gaussian
method = B3LYP/6-31G*
nprocs = 4
mem = 4GB
charge = 0
mult_a = 1
mult_b = 3
```

**my_molecule.xyz**:

```
5

C     -3.006349    0.690453    0.000000
H     -1.975345    0.980885   -0.212471
H     -3.317685   -0.086238   -0.701630
H     -3.073701    0.307432    1.020403
H     -3.658665    1.559733   -0.106302
```

### Example 12: ONIOM Calculation

```
*GEOM
C     -3.006349    0.690453    0.000000 H
H     -1.975345    0.980885   -0.212471 L
H     -3.317685   -0.086238   -0.701630 L
H     -3.073701    0.307432    1.020403 L
H     -3.658665    1.559733   -0.106302 L
*

*TAIL_a
ONIOM(B3LYP/6-31G*:AM1)
*

*TAIL_b
ONIOM(B3LYP/6-31G*:AM1)
*

program = gaussian
method = ONIOM
nprocs = 4
mem = 4GB
charge = 0
mult_a = 1
mult_b = 3
oniom_layer_info = H,L
charge_and_mult_oniom1 = 0 1 H 0 1 L
charge_and_mult_oniom2 = 0 3 H 0 3 L
```

### Example 13: Dihedral Constraints

```
*GEOM
C     -3.061368    0.418316    0.054793
C     -1.871810    1.377432   -0.152416
H     -3.207769   -0.194298   -0.837855
H     -2.879128   -0.247755    0.901544
H     -3.985045    0.972493    0.236909
H     -1.001811    0.793046   -0.461758
H     -2.098489    2.070786   -0.966564
C     -1.504279    2.169506    1.123817
H     -1.251550    1.476786    1.930302
C     -2.628250    3.116411    1.590645
H     -0.609919    2.763669    0.919378
H     -3.499808    2.553251    1.930802
H     -2.935420    3.782372    0.780755
H     -2.278773    3.729907    2.424254
*

*CONSTR
D 1 2 8 10 180.0    # Fix dihedral angle
R 1 2 1.5          # Fix bond length
*

*TAIL_a
*

*TAIL_b
*

program = gaussian
method = B3LYP/6-31G*
nprocs = 4
mem = 4GB
charge = 0
mult_a = 1
mult_b = 3
```

### Example 14: Robust DIIS with Energy-DIIS Variant (Experimental)

Use the experimental GEDIIS implementation with Energy-DIIS variant for difficult convergence cases.
Note: The sequential hybrid (default) is recommended for most cases.

```
*GEOM
C      0.037739    0.000000   -0.000000
C      1.362261    0.000000   -0.000000
H     -0.524574    0.930562    0.000000
H     -0.524574   -0.930562   -0.000000
H      1.924574    0.930562   -0.000000
H      1.924574   -0.930562    0.000000
*

*TAIL_a
*

*TAIL_b
*

program = gaussian
method = B3LYP/6-31G*
nprocs = 4
mem = 4GB
charge = 0
mult_a = 1
mult_b = 3
# Enable robust DIIS (experimental)
use_robust_diis = true
use_gediis = true
gediis_variant = energy
gdiis_cosine_check = standard
```

### Example 15: Advanced Hessian Update for TS-like Crossings

Use Bofill Hessian update method for transition state-like crossing points:

```
*GEOM
C      0.037739    0.000000   -0.000000
C      1.362261    0.000000   -0.000000
H     -0.524574    0.930562    0.000000
H     -0.524574   -0.930562   -0.000000
H      1.924574    0.930562   -0.000000
H      1.924574   -0.930562    0.000000
*

*TAIL_a
*

*TAIL_b
*

program = gaussian
method = B3LYP/6-31G*
nprocs = 4
mem = 4GB
charge = 0
mult_a = 1
mult_b = 3
# Use Bofill Hessian update for TS-like crossings
hessian = bofill
# Combine with robust DIIS for best results
use_robust_diis = true
use_gediis = true
n_neg = 1
```

### Example 16: GDIIS_blend with Gradient-Weighted Blend

Use the new GDIIS_blend optimizer with gradient-weighted blend mode for difficult cases with plateaus:

```
*GEOM
C     0.000000000000      1.396613000000      0.000000000000
C     1.209503000000      0.698307000000      0.000000000000
C     1.209503000000     -0.698307000000      0.000000000000
C     0.000000000000     -1.396613000000      0.000000000000
C    -1.209503000000     -0.698307000000      0.000000000000
C    -1.209503000000      0.698307000000      0.000000000000
H     2.150882000000      1.241812000000      0.000000000000
H     2.150882000000     -1.241812000000      0.000000000000
H     0.000000000000     -2.483625000000      0.000000000000
H    -2.150882000000     -1.241812000000      0.000000000000
H    -2.150882000000      1.241812000000      0.000000000000
*

*TAIL_a
*

*TAIL_b
*

program = gaussian
method = B3LYP/6-31G*
nprocs = 4
mem = 4GB
charge = 1
mult_a = 1
mult_b = 3
# Enable GDIIS_blend optimizer
use_gediis = blend
# Enable hybrid blend with gradient-weighted mode
use_hybrid_gediis = true
gediis_blend_mode = gradient
```

### Example 17: GDIIS_blend with Sequential Blend

Per-step switching between GDIIS and EDIIS based on RMS displacement trend:

```
*GEOM
C     0.000000000000      1.396613000000      0.000000000000
C     1.209503000000      0.698307000000      0.000000000000
C     1.209503000000     -0.698307000000      0.000000000000
C     0.000000000000     -1.396613000000      0.000000000000
C    -1.209503000000     -0.698307000000      0.000000000000
C    -1.209503000000      0.698307000000      0.000000000000
H     2.150882000000      1.241812000000      0.000000000000
H     2.150882000000     -1.241812000000      0.000000000000
H     0.000000000000     -2.483625000000      0.000000000000
H    -2.150882000000     -1.241812000000      0.000000000000
H    -2.150882000000      1.241812000000      0.000000000000
*

*TAIL_a
*

*TAIL_b
*

program = gaussian
method = B3LYP/6-31G*
nprocs = 4
mem = 4GB
charge = 1
mult_a = 1
mult_b = 3
# Enable GDIIS_blend optimizer
use_gediis = blend
# Enable hybrid blend with per-step switching
use_hybrid_gediis = true
gediis_blend_mode = sequential
```

### Example 18: GDIIS_blend with Fixed Sequential Blend (Default Blend)

Recommended blend mode: 50/50 blend far from minimum, pure GDIIS near convergence.
This is the default `gediis_blend_mode` when `use_gediis = blend` + `use_hybrid_gediis = true`:

```
*GEOM
C     0.000000000000      1.396613000000      0.000000000000
C     1.209503000000      0.698307000000      0.000000000000
C     1.209503000000     -0.698307000000      0.000000000000
C     0.000000000000     -1.396613000000      0.000000000000
C    -1.209503000000     -0.698307000000      0.000000000000
C    -1.209503000000      0.698307000000      0.000000000000
H     2.150882000000      1.241812000000      0.000000000000
H     2.150882000000     -1.241812000000      0.000000000000
H     0.000000000000     -2.483625000000      0.000000000000
H    -2.150882000000     -1.241812000000      0.000000000000
H    -2.150882000000      1.241812000000      0.000000000000
*

*TAIL_a
*

*TAIL_b
*

program = gaussian
method = B3LYP/6-31G*
nprocs = 4
mem = 4GB
charge = 1
mult_a = 1
mult_b = 3
# Enable GDIIS_blend optimizer
use_gediis = blend
# Enable hybrid blend with default fixed_sequential mode
use_hybrid_gediis = true
gediis_blend_mode = fixed_sequential  # explicit; this is also the default
```

### Example 19: Pure GDIIS_blend (no EDIIS, very robust)

Pure GDIIS with trust region and inverted mean true Hessian. No EDIIS component.
Set `use_hybrid_gediis = false` (the default for blend mode):

```
*GEOM
C     0.000000000000      1.396613000000      0.000000000000
C     1.209503000000      0.698307000000      0.000000000000
C     1.209503000000     -0.698307000000      0.000000000000
C     0.000000000000     -1.396613000000      0.000000000000
C    -1.209503000000     -0.698307000000      0.000000000000
C    -1.209503000000      0.698307000000      0.000000000000
H     2.150882000000      1.241812000000      0.000000000000
H     2.150882000000     -1.241812000000      0.000000000000
H     0.000000000000     -2.483625000000      0.000000000000
H    -2.150882000000     -1.241812000000      0.000000000000
H    -2.150882000000      1.241812000000      0.000000000000
*

*TAIL_a
*

*TAIL_b
*

program = gaussian
method = B3LYP/6-31G*
nprocs = 4
mem = 4GB
charge = 1
mult_a = 1
mult_b = 3
# Enable GDIIS_blend optimizer
use_gediis = blend
# Pure GDIIS_blend: no EDIIS component
use_hybrid_gediis = false  # default for blend, can be omitted
```

## Troubleshooting

### Common Errors

#### Error: "No such file or directory"

**Cause**: QM program not found or input file missing
**Solution**:

- Check QM program is installed and in PATH
- Verify input file path is correct

#### Error: "QM calculation failed"

**Cause**: QM program encountered an error
**Solution**:

- Check `running_dir/*.log` files for QM program errors
- Verify method and basis set are valid
- Check memory and processor settings
- Try `mode = noread` for SCF convergence issues

**Common QM-Specific Errors**:

- **Gaussian**: "Convergence failure" → Try `scf=(xqc,qc,nofermi)` in TAIL
- **ORCA**: "SCF not converged" → Try `%scf SOSCF true end` in TAIL
- **Custom**: Check JSON configuration and program-specific error messages

#### Error: "Failed to parse output"

**Cause**: QM output format not recognized
**Solution**:

- Verify QM program version is supported
- Check output files are complete
- Ensure calculation finished successfully

**Parsing-Specific Issues**:

- **Energy not found**: Check energy_parser regex pattern in custom interface
- **Forces not found**: Verify forces_parser regex matches output format
- **Geometry incomplete**: Ensure calculation reached geometry optimization completion
- **State extraction failed**: For TD-DFT, verify state indices are valid (0=ground, 1+=excited)

#### Error: "Maximum steps exceeded"

**Cause**: Optimization did not converge
**Solution**:

- Increase `max_steps` (default: 100)
- Check initial geometry is reasonable
- Try different starting geometry
- Use LST interpolation for better initial guess
- Check for SCF convergence issues

### SCF Convergence Issues

**Symptoms**: QM calculations fail or oscillate

**Solutions**:

1. Use `mode = noread` for fresh SCF each step
2. Use `mode = stable` to check wavefunction stability
3. Use `mode = inter_read` for open-shell singlets
4. Add SCF convergence keywords in TAIL sections:
   - Gaussian: `scf=(maxcycle=200,xqc)`
   - ORCA: `%scf maxiter 200 end`

### Slow Convergence

**Symptoms**: Many optimization steps required

**Solutions**: add more steps; try other optimizers

## Cite

Cite this project if you use OpenThermo for your research:

```bibtex
@article{pham_omecp_2026,
    title = {OpenMECP: A High-Performance Rust Implementation for the Rigorous Location of Minimum Energy Crossing Points in Chemical Dynamics},
    url = {https://doi.org/10.13140/RG.2.2.21309.73443},
    doi = {10.13140/RG.2.2.21309.73443},
    urldate = {2026},
    author = {Pham, Le Nhan},
    year = {2026},
    note = {Publisher: Preprint},
}
```

## License

OpenMECP is licensed under the **MIT License**.

### Acknowledgments

- **Le Nhan Pham**: Developer and maintainer
- **Open-source community**: For contributions and feedback
- **Quantum chemistry community**: For validation and testing

---

**OpenMECP v0.0.5** - A Rust implementation of the MECP optimizer
Developed by Le Nhan Pham | [GitHub](https://github.com/lenhanpham/OpenMECP)

For more information, visit the project documentation or use `omecp --help`
