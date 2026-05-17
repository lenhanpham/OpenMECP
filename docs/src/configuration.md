# Configuration File

OpenMECP reads a configuration file `omecp_config.cfg` at startup to customize
program-wide behavior. Settings in the config file apply to all calculations
run from that directory.

## Creating a Config Template

```bash
omecp ci omecp_config.cfg
```

This generates a fully commented template with all available options.

---

## File Locations and Precedence

Configuration files are loaded in hierarchical order. A setting in a higher-priority
file overrides the same setting in a lower-priority file.

| Priority | Location |
|---|---|
| 1 (highest) | `./omecp_config.cfg` — current working directory |
| 2 | `~/.config/omecp/omecp_config.cfg` (Unix) or `%APPDATA%\omecp\omecp_config.cfg` (Windows) |
| 3 | `/etc/omecp/omecp_config.cfg` (Unix) or `%PROGRAMDATA%\omecp\omecp_config.cfg` (Windows) |
| 4 (lowest) | Built-in defaults |

---

## File Format

The config file uses INI format with four sections:

```ini
[extensions]
gaussian = log
orca     = out
custom   = log

[general]
max_memory     = 4GB
default_nprocs = 4
print_level    = 0

[logging]
level        = info
file_logging = false

[cleanup]
enabled              = true
preserve_extensions  = xyz,backup
verbose              = 1
cleanup_frequency    = 5
```

---

## `[extensions]` Section

Configures the file extensions that OpenMECP looks for when parsing QM output.

| Key | Default | Description |
|---|---|---|
| `gaussian` | `log` | Gaussian output file extension |
| `orca` | `out` | ORCA output file extension |
| `custom` | `log` | Custom program output extension |

---

## `[general]` Section

| Key | Default | Description |
|---|---|---|
| `max_memory` | `4GB` | Default memory allocation (overridden by input `mem`) |
| `default_nprocs` | `4` | Default number of processors (overridden by input `nprocs`) |
| `print_level` | `0` | Output verbosity: `0`=quiet, `1`=normal, `2`=verbose |

---

## `[logging]` Section

| Key | Default | Description |
|---|---|---|
| `level` | `info` | Log level: `error`, `warn`, `info`, `debug`, `trace` |
| `file_logging` | `false` | Write debug log to `omecp_debug_{basename}.log` |

Enabling `file_logging = true` writes a detailed debug log named
`omecp_debug_{basename}.log` (where `{basename}` is derived from the input file
name). This is useful for diagnosing convergence issues.

---

## `[cleanup]` Section

OpenMECP accumulates many intermediate QM input/output files during a long run.
The cleanup system automatically removes unnecessary files to prevent disk
saturation.

| Key | Default | Description |
|---|---|---|
| `enabled` | `true` | Enable automatic file cleanup |
| `preserve_extensions` | `xyz,backup` | Comma-separated extensions to always keep |
| `verbose` | `1` | Cleanup verbosity: `0`=silent, `1`=summary, `2`=each file |
| `cleanup_frequency` | `5` | Run cleanup every N optimization steps |

**Cleanup strategy**: OpenMECP keeps the latest `.engrad` files and all
`.log` / input files; it removes older intermediate files. Add custom extensions
to `preserve_extensions` to protect specific file types.

---

## Configuration Display at Startup

When OpenMECP starts, it prints:

- Which configuration file was loaded (or "built-in defaults" if none found)
- All input parameters and their values
- All convergence thresholds
- All configuration file settings

This ensures you always know exactly what settings are being used for a given run.
