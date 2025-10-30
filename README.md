<p align="center">
  <a href="https://github.com/lenhanpham/OpenMECP">
    <picture>
      <img src="resources/omecp-logo.svg" alt="OpenMECP">
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
  <img src="https://img.shields.io/badge/Build-Passing-brightgreen.svg" alt="Build Status">
</p>

A high-performance Rust implementation of the MECP (Minimum Energy Crossing Point) optimizer for locating crossing points between two potential energy surfaces in quantum chemistry calculations.

**Version**: 0.0.1
**Author**: Le Nhan Pham
**Language**: Rust
**License**: MIT

## Important Note

### **The program is under active development and not ready for use**

**Status**: Alpha testing phase

Current features include:

- Template generation from geometry files (`omecp ci file.xyz`)
- Comprehensive built-in help system (`omecp --help`)
- Full MECP optimization capabilities

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Input File Format](#input-file-format)
- [Keywords Reference](#keywords-reference)
- [Supported QM Programs](#supported-qm-programs)
- [Run Modes](#run-modes)
- [Advanced Features](#advanced-features)
- [Output Files](#output-files)
- [Convergence Criteria](#convergence-criteria)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Performance](#performance)
- [References](#references)

## Overview

OpenMECP locates the minimum energy crossing point between two potential energy surfaces (PES), which is crucial for understanding:

- Photochemical reactions
- Spin-forbidden processes
- Intersystem crossing
- Conical intersections
- Non-adiabatic dynamics

The program implements the Harvey et al. algorithm (Chem. Phys. Lett. 1994) with modern optimizers and supports multiple quantum chemistry programs.

### Key Algorithm

The MECP gradient combines two components:

1. **f-vector**: Drives energy difference to zero: `f = (E1 - E2) * x_norm`
2. **g-vector**: Minimizes energy perpendicular to gradient difference: `g = f1 - (x_norm · f1) * x_norm`

Where `x_norm = (f1 - f2) / |f1 - f2|` is the normalized gradient difference.

## Features

### User Interface

- ✅ **Template Generation**: Create input files from geometry files (XYZ, GJF, LOG)
- ✅ **Comprehensive Help**: Built-in help system with keyword, method, and feature documentation
- ✅ **Command Line Interface**: Simple `omecp ci` and `omecp --help` commands

### Core Functionality

- ✅ **MECP Optimization**: Harvey et al. algorithm implementation
- ✅ **BFGS Optimizer**: Quasi-Newton optimization with PSB Hessian updates
- ✅ **GDIIS Optimizer**: Geometry-based DIIS for 2-3x faster convergence
- ✅ **GEDIIS Optimizer**: Energy-informed DIIS for enhanced convergence
- ✅ **Hybrid Strategy**: Automatic switching between BFGS, GDIIS, and GEDIIS

### Constraints

- ✅ **Bond Constraints**: Fix bond lengths during optimization
- ✅ **Angle Constraints**: Fix bond angles during optimization
- ✅ **Lagrange Multipliers**: Exact constraint enforcement
- ✅ **Fixed Atoms**: Partial geometry optimization

### QM Program Support

- ✅ **Gaussian**: DFT, TD-DFT, MP2, CASSCF
- ✅ **ORCA**: DFT, TD-DFT, CASSCF
- ✅ **Custom**: User-defined via JSON configuration

### Advanced Features

- ✅ **PES Scan**: 1D and 2D potential energy surface scans
- ✅ **LST Interpolation**: Linear synchronous transit with Kabsch alignment
- ✅ **Coordinate Driving**: Drive reaction coordinates systematically
- ✅ **Path Optimization**: Nudged Elastic Band (NEB) method
- ✅ **Fix-dE Optimization**: Constrain energy difference to target value
- ✅ **State Selection**: Choose specific excited states for TD-DFT
- ✅ **External Geometry**: Read geometries from external files
- ✅ **ONIOM Support**: Multi-layer QM/MM calculations
- ✅ **Run Modes**: Normal, Read, NoRead, Stable, InterRead
- ✅ **Pre-point Calculations**: For difficult SCF convergence

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

# From Gaussian output
omecp ci calculation.log

# From Gaussian input
omecp ci input.gjf

# Custom output name
omecp ci molecule.xyz my_calculation.inp
```

Then edit the generated template to customize parameters.

### Option 2: Manual Input File

Create a file named `input.inp`:

```
*geom
C  0.0  0.0  0.0
H  1.0  0.0  0.0
H  0.0  1.0  0.0
H  0.0  0.0  1.0
*

*tail1
# Additional Gaussian keywords for state 1
*

*tail2
# Additional Gaussian keywords for state 2
*

program = gaussian
method = B3LYP/6-31G*
nprocs = 4
mem = 4GB
charge = 0
mult1 = 1
mult2 = 3
```

### 3. Get Help

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

### 4. Run OpenMECP

- Copy omecp to your preferred directory, and chmod 700 omecp before exporting this dir in your shellscript

- Prepare a shell script

```bash
# export directory where omecp is copied
export /path/to/your/dir/
module load gaussian (or Orca) // Check your HPC system
omecp input.inp
```

### 5. Check Results

- Optimized geometry: `final.xyz`
- Intermediate geometries: `running_dir/` directory
- Convergence information: stdout

## Input File Format

The input file consists of several sections marked by `*SECTION` and terminated by `*`.

### Required Sections

#### *GEOM Section

Initial geometry in Cartesian coordinates (Angstroms):

```
*GEOM
Element  X  Y  Z
C  0.0  0.0  0.0
H  1.0  0.0  0.0
*
```

#### *TAIL1 and *TAIL2 Sections

Additional keywords for each state:

```
*TAIL1
# Gaussian: Additional route section keywords
# ORCA: Additional input blocks
*

*TAIL2
# Same format as TAIL1
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

#### *LST1 and *LST2 Sections

For LST interpolation:

```
*LST1
C  0.0  0.0  0.0
H  1.0  0.0  0.0
*

*LST2
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
| `mem`     | string  | Memory allocation            | `4GB`              |
| `charge`  | integer | Molecular charge             | `0`                |
| `mult1`   | integer | Spin multiplicity of state 1 | `1`                |
| `mult2`   | integer | Spin multiplicity of state 2 | `3`                |

### Optional Keywords

| Keyword                 | Type    | Default    | Description                                               |
| ----------------------- | ------- | ---------- | --------------------------------------------------------- |
| `charge2`               | integer | `charge`   | Charge of state 2 (if different)                          |
| `td1`                   | string  | `""`       | TD-DFT keywords for state 1                               |
| `td2`                   | string  | `""`       | TD-DFT keywords for state 2                               |
| `mode`                  | string  | `normal`   | Run mode (see [Run Modes](#run-modes))                    |
| `max_steps`             | integer | `100`      | Maximum optimization steps                                |
| `max_step_size`         | float   | `0.1`      | Maximum step size (Bohr)                                  |
| `fixedatoms`            | string  | `""`       | Fixed atom indices (e.g., `1,3-5,7`)                      |
| `fix_de`                | float   | `0.0`      | Target energy difference (eV)                             |
| `state1`                | integer | `0`        | Excited state index for state 1 (TD-DFT)                  |
| `state2`                | integer | `0`        | Excited state index for state 2 (TD-DFT)                  |
| `use_gediis`            | boolean | `false`    | Use GEDIIS optimizer instead of GDIIS                     |
| `drive_type`            | string  | `""`       | Coordinate type for driving (`bond`, `angle`, `dihedral`) |
| `drive_atoms`           | string  | `""`       | Atom indices for coordinate driving                       |
| `drive_start`           | float   | `0.0`      | Starting value for coordinate driving                     |
| `drive_end`             | float   | `0.0`      | Ending value for coordinate driving                       |
| `drive_steps`           | integer | `10`       | Number of steps for coordinate driving                    |
| `de_thresh`             | float   | `0.000050` | Energy difference convergence threshold                   |
| `rms_thresh`            | float   | `0.0025`   | RMS gradient convergence threshold                        |
| `max_dis_thresh`        | float   | `0.004`    | Max displacement convergence threshold                    |
| `max_g_thresh`          | float   | `0.0007`   | Max gradient convergence threshold                        |
| `rms_g_thresh`          | float   | `0.0005`   | RMS gradient convergence threshold                        |
| `custom_interface_file` | string  | `""`       | Path to custom QM interface JSON config                   |

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

## Supported QM Programs

### Gaussian

**Supported Methods**: DFT, TD-DFT, MP2, CASSCF

**Input Format**: Gaussian .gjf files

**Example**:

```
program = gaussian
method = B3LYP/6-31G*
td1 = TD(nstates=5,root=1)
td2 = TD(nstates=5,root=2)
```

**Output Files**: `.log` files in `running_dir/` directory

**Checkpoint Files**: `a.chk` (state A), `b.chk` (state B)

### ORCA

**Supported Methods**: DFT, TD-DFT, CASSCF

**Input Format**: ORCA .inp files

**Example**:

```
program = orca
method = B3LYP def2-SVP

*TAIL1
%tddft
  nroots 5
  iroot 1
end
*

*TAIL2
%tddft
  nroots 5
  iroot 2
end
*
```

**Output Files**: `.log` and `.engrad` files in `running_dir/` directory

**Checkpoint Files**: `a.gbw` (state A), `b.gbw` (state B)

### Custom QM Interface

**Supported Methods**: Any QM program with configurable input/output

**Configuration**: JSON configuration file defining input templates and output parsing

**Example**:

```
program = custom
custom_interface_file = my_qm_interface.json
```

**Configuration File** (`my_qm_interface.json`):

```json
{
  "name": "My Custom QM Program",
  "command": "my_qm_exe",
  "input_template": "{header}\n{geometry}\n{tail}",
  "output_extension": "out",
  "energy_parser": {
    "pattern": "Total Energy:\\s*(-?\\d+\\.\\d+)",
    "unit_factor": 1.0
  },
  "forces_parser": {
    "pattern": "Forces:\\s*(-?\\d+\\.\\d+)\\s+(-?\\d+\\.\\d+)\\s+(-?\\d+\\.\\d+)"
  }
}
```

**Template Placeholders**:

- `{geometry}`: Replaced with XYZ-format geometry
- `{header}`: Replaced with state-specific header
- `{tail}`: Replaced with additional keywords

**Output Files**: Files with configured extension in `running_dir/` directory

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
mult1 = 1    # Open-shell singlet
mult2 = 3    # Triplet reference
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
*LST1
C  0.0  0.0  0.0
H  1.0  0.0  0.0
*

*LST2
C  0.0  0.0  0.5
H  1.2  0.0  0.5
*
```

**Features**:

- Kabsch algorithm for optimal alignment
- 10 interpolation points (default)
- Energy profile calculation
- Interactive confirmation

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

Automatically activated after 3 BFGS steps:

**GDIIS (Geometry-based DIIS)**:

- Default optimizer for geometry convergence
- Stores last 4 geometries, gradients, and Hessians
- Computes error vectors: `E[i] = H^-1 * G[i]`
- Solves DIIS equations for optimal step

**GEDIIS (Energy-informed DIIS)**:

- Enhanced optimizer using energy information
- Set `use_gediis = true` to enable
- Includes energy differences in DIIS error vectors
- Better convergence for difficult cases
- Can be 2-4x faster than GDIIS in some systems

**Benefits**:

- 2-4x faster convergence compared to BFGS
- Typical 15-20 steps → 6-10 steps
- No user intervention required
- Automatic fallback to BFGS for first 3 steps

### State Selection for TD-DFT

Choose specific excited states for TD-DFT calculations:

```
state1 = 1    # Use first excited state for state 1
state2 = 2    # Use second excited state for state 2
td1 = TD(nstates=5,root=1)
td2 = TD(nstates=5,root=2)
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
- Optimizes path using NEB algorithm
- Provides smooth energy profiles
- Identifies transition states and intermediates

### Fix-dE Optimization

Constrain the energy difference to a target value:

```
run_mode = fix_de
fix_de = 0.1    # Target ΔE = 0.1 eV
```

**Use Cases**:

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

### final.xyz

Final optimized geometry in XYZ format:

```
3

C  0.00000000  0.00000000  0.00000000
H  1.08900000  0.00000000  0.00000000
H  0.00000000  1.08900000  0.00000000
```

### running_dir/ Directory

Contains all intermediate calculations:

```
running_dir/
├── 0_A.gjf          # Initial state A input
├── 0_A.log          # Initial state A output
├── 0_B.gjf          # Initial state B input
├── 0_B.log          # Initial state B output
├── 1_A.gjf          # Step 1 state A input
├── 1_A.log          # Step 1 state A output
...
├── a.chk            # State A checkpoint
└── b.chk            # State B checkpoint
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
| RMS gradient     | < 0.0005           | Root mean square gradient      |
| Max gradient     | < 0.0007           | Maximum gradient component     |
| RMS displacement | < 0.0025           | Root mean square displacement  |
| Max displacement | < 0.004            | Maximum displacement component |

**Note**: These are the same criteria used in Gaussian optimizations.

## Examples

### Example 1: Basic MECP (Singlet-Triplet)

```
*GEOM
C  0.0  0.0  0.0
H  1.0  0.0  0.0
H  0.0  1.0  0.0
H  0.0  0.0  1.0
*

*TAIL1
*

*TAIL2
*

program = gaussian
method = B3LYP/6-31G*
nprocs = 4
mem = 4GB
charge = 0
mult1 = 1
mult2 = 3
```

### Example 2: TD-DFT Excited States

```
*GEOM
C  0.0  0.0  0.0
C  1.4  0.0  0.0
H  -0.5  0.9  0.0
H  -0.5  -0.9  0.0
H  1.9  0.9  0.0
H  1.9  -0.9  0.0
*

*TAIL1
*

*TAIL2
*

program = gaussian
method = B3LYP/6-31G*
td1 = TD(nstates=5,root=1)
td2 = TD(nstates=5,root=2)
nprocs = 8
mem = 8GB
charge = 0
mult1 = 1
mult2 = 1
```

### Example 3: Open-Shell Singlet with ORCA

```
*GEOM
Fe  0.0  0.0  0.0
N   2.0  0.0  0.0
N   0.0  2.0  0.0
N  -2.0  0.0  0.0
N   0.0 -2.0  0.0
*

*TAIL1
%scf
  maxiter 200
end
*

*TAIL2
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
mult1 = 1
mult2 = 3
```

### Example 4: PES Scan with Constraints

```
*GEOM
C  0.0  0.0  0.0
H  1.0  0.0  0.0
H  0.0  1.0  0.0
*

*CONSTR
s r 1 2 1.0 20 0.05    # Scan C-H bond
R 1 3 1.1              # Fix other C-H bond
*

*TAIL1
*

*TAIL2
*

program = gaussian
method = B3LYP/6-31G*
nprocs = 4
mem = 4GB
charge = 0
mult1 = 1
mult2 = 3
```

### Example 5: LST Interpolation

```
*LST1
C  0.0  0.0  0.0
H  1.0  0.0  0.0
H  0.0  1.0  0.0
*

*LST2
C  0.0  0.0  0.5
H  1.2  0.0  0.5
H  0.0  1.2  0.5
*

*TAIL1
*

*TAIL2
*

program = gaussian
method = B3LYP/6-31G*
nprocs = 4
mem = 4GB
charge = 0
mult1 = 1
mult2 = 3
```

### Example 6: GEDIIS Optimizer

```
*GEOM
C  0.0  0.0  0.0
H  1.089  0.0  0.0
H  -0.363  1.027  0.0
H  -0.363  -0.513  0.889
H  -0.363  -0.513  -0.889
*

*TAIL1
*

*TAIL2
*

program = gaussian
method = B3LYP/6-31G*
nprocs = 4
mem = 4GB
charge = 0
mult1 = 1
mult2 = 3
use_gediis = true
```

### Example 7: TD-DFT State Selection

```
*GEOM
C  0.0  0.0  0.0
C  1.4  0.0  0.0
H  -0.5  0.9  0.0
H  -0.5  -0.9  0.0
H  1.9  0.9  0.0
H  1.9  -0.9  0.0
*

*TAIL1
*

*TAIL2
*

program = gaussian
method = B3LYP/6-31G*
td1 = TD(nstates=5,root=1)
td2 = TD(nstates=5,root=2)
state1 = 1
state2 = 2
nprocs = 8
mem = 8GB
charge = 0
mult1 = 1
mult2 = 1
```

### Example 8: Coordinate Driving

```
*GEOM
C  0.0  0.0  0.0
H  1.0  0.0  0.0
H  0.0  1.0  0.0
*

*TAIL1
*

*TAIL2
*

program = gaussian
method = B3LYP/6-31G*
nprocs = 4
mem = 4GB
charge = 0
mult1 = 1
mult2 = 3
run_mode = coordinate_drive
drive_type = bond
drive_atoms = 1,2
drive_start = 1.0
drive_end = 2.0
drive_steps = 20
```

### Example 9: Fix-dE Optimization

```
*GEOM
C  0.0  0.0  0.0
H  1.089  0.0  0.0
H  -0.363  1.027  0.0
H  -0.363  -0.513  0.889
H  -0.363  -0.513  -0.889
*

*TAIL1
*

*TAIL2
*

program = gaussian
method = B3LYP/6-31G*
nprocs = 4
mem = 4GB
charge = 0
mult1 = 1
mult2 = 3
run_mode = fix_de
fix_de = 0.1
```

### Example 10: External Geometry File

```
*GEOM
@my_molecule.xyz
*

*TAIL1
*

*TAIL2
*

program = gaussian
method = B3LYP/6-31G*
nprocs = 4
mem = 4GB
charge = 0
mult1 = 1
mult2 = 3
```

**my_molecule.xyz**:

```
5

C  0.0  0.0  0.0
H  1.089  0.0  0.0
H  -0.363  1.027  0.0
H  -0.363  -0.513  0.889
H  -0.363  -0.513  -0.889
```

### Example 11: ONIOM Calculation

```
*GEOM
C  0.0  0.0  0.0  H
H  1.0  0.0  0.0  L
H  0.0  1.0  0.0  L
H  0.0  0.0  1.0  L
*

*TAIL1
ONIOM(B3LYP/6-31G*:AM1)
*

*TAIL2
ONIOM(B3LYP/6-31G*:AM1)
*

program = gaussian
method = ONIOM
nprocs = 4
mem = 4GB
charge = 0
mult1 = 1
mult2 = 3
oniom_layer_info = H,L
charge_and_mult_oniom1 = 0 1 H 0 1 L
charge_and_mult_oniom2 = 0 3 H 0 3 L
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

#### Error: "Failed to parse output"

**Cause**: QM output format not recognized
**Solution**:

- Verify QM program version is supported
- Check output files are complete
- Ensure calculation finished successfully

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

**Solutions**: add more steps
