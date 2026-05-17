# Custom QM Interface

OpenMECP supports user-defined QM programs via a JSON configuration file.
This allows connecting any quantum chemistry code that can produce energies
and gradients to the MECP optimizer.

## Enabling Custom Interface

```
program              = custom
custom_interface_file = my_program.json
```

## JSON Configuration Format

Create a JSON file describing how to write inputs, run the program, and parse
its outputs:

```json
{
  "name": "my_qm_program",
  "executable": "/path/to/qm_program",
  "input_template": "template_state.inp",
  "output_extension": "out",
  "gradient_extension": "grad",
  "energy_pattern": "Total Energy:\\s+([\\-0-9\\.]+)",
  "gradient_pattern": "Gradient:\\s+([\\-0-9\\.eE+\\-]+)",
  "extra_args": []
}
```

### Fields

| Field | Type | Description |
|---|---|---|
| `name` | string | Human-readable program name |
| `executable` | string | Path to the QM program executable |
| `input_template` | string | Path to an input template file |
| `output_extension` | string | Extension of the output file (e.g. `out`) |
| `gradient_extension` | string | Extension of the gradient file (e.g. `grad`) |
| `energy_pattern` | string | Regex to extract the total energy (first capture group) |
| `gradient_pattern` | string | Regex to extract gradient values |
| `extra_args` | array | Extra command-line arguments passed to the executable |

## Creating a Template

Generate a starting JSON template with:

```bash
omecp ci custom_interface.json
```

Edit the generated file to match your program's input/output format.

## Notes

- The regex patterns use Rust regex syntax.
- The first capturing group in `energy_pattern` must match a floating-point
  number (hartree).
- The `gradient_pattern` is applied repeatedly to extract all gradient
  components in the order: x₁ y₁ z₁ x₂ y₂ z₂ … for N atoms.
- OpenMECP writes a separate input file for state A and state B and calls the
  executable twice per optimization step.
