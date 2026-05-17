# Quick Start

This page walks through the two ways to set up a calculation and how to run it.

## Option 1: Generate a Template (Recommended)

OpenMECP can generate a ready-to-edit input file from any existing geometry file.

```bash
# From an XYZ file
omecp ci molecule.xyz
# Creates: molecule.inp

# From a Gaussian output file
# (adds _omecp suffix to avoid overwriting your output)
omecp ci calculation.log
# Creates: calculation_omecp.inp

# From a Gaussian input file
omecp ci input.gjf
# Creates: input.inp

# Specify a custom output name
omecp ci molecule.xyz my_run.inp
```

Then open the generated `.inp` file and fill in the required keywords
(`program`, `method`, `nprocs`, `mem`, `charge`, `mult_a`, `mult_b`).

## Option 2: Write an Input File Manually

Create a file named `input.inp` with the following structure:

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
# Extra Gaussian route keywords for state A
*

*tail_b
# Extra Gaussian route keywords for state B
*

program  = gaussian
method   = B3LYP/6-31G*
nprocs   = 4
mem      = 4GB
charge   = 0
mult_a   = 1
mult_b   = 3
```

See the [Input File Format](input-format/input-format.md) chapter for the full section
reference and the [Keywords Reference](input-format/keywords.md) for all available
parameters.

## Running OpenMECP

```bash
omecp input.inp > output.log
```

On an HPC cluster, put the above in a job submission script and load your QM
program module before calling `omecp`.

## Checking the Results

| Output | Location |
|---|---|
| Converged MECP geometry | `{basename}_mecp.xyz` |
| Intermediate geometries | `running_dir/` |
| Convergence summary | stdout / `output.log` |
| Checkpoint (restart) | `{basename}_checkpoint.json` |

## Built-in Help

OpenMECP includes a comprehensive built-in help system:

```bash
omecp --help               # General usage
omecp --help keywords      # All input keywords with descriptions
omecp --help methods       # Supported QM methods
omecp --help features      # Feature overview
omecp --help examples      # Worked examples
omecp ci --help            # Template generation help
```

## Minimal Gaussian Example

```
*geom
Fe    0.000000    0.000000    0.000000
N     0.000000    0.000000    2.100000
N     0.000000    2.100000    0.000000
*

*tail_a
*

*tail_b
*

program  = gaussian
method   = B3LYP/def2-SVP
nprocs   = 8
mem      = 16GB
charge   = 0
mult_a   = 1
mult_b   = 3
```

## Minimal ORCA Example

```
*geom
Fe    0.000000    0.000000    0.000000
N     0.000000    0.000000    2.100000
*

*tail_a
%scf
  MaxIter 500
end
*

*tail_b
*

program  = orca
method   = B3LYP def2-SVP
nprocs   = 8
mem      = 8000
charge   = 0
mult_a   = 1
mult_b   = 3
```
