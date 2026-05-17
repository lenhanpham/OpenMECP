# API Reference

The full Rust API documentation for the `omecp` crate is generated from source
code comments and is available at:

**[View API Reference →](api/omecp/index.html)**

---

## Module Overview

| Module | Purpose |
|---|---|
| [`config`](api/omecp/config/index.html) | Configuration structures: `Config`, `QMProgram`, `RunMode`, `Thresholds` |
| [`parser`](api/omecp/parser/index.html) | Input file parser — section-based format with key-value parameters |
| [`optimizer`](api/omecp/optimizer/index.html) | BFGS, GDIIS, GEDIIS, hybrid optimization algorithms |
| [`geometry`](api/omecp/geometry/index.html) | Core types: `Geometry`, `State`, unit conversion utilities |
| [`gdiis`](api/omecp/gdiis/index.html) | GDIIS algorithm with SR1 updates and step validation |
| [`gediis`](api/omecp/gediis/index.html) | GEDIIS algorithm — RFO, energy, and simultaneous variants |
| [`constraints`](api/omecp/constraints/index.html) | Geometric constraint system with Lagrange multipliers |
| [`hessian_update`](api/omecp/hessian_update/index.html) | BFGS, Bofill, Powell, PSB, and adaptive Hessian updates |
| [`qm_interface`](api/omecp/qm_interface/index.html) | Unified `QMInterface` trait and program adapters |
| [`io`](api/omecp/io/index.html) | File I/O: XYZ, Gaussian formats, checkpoint files |
| [`lst`](api/omecp/lst/index.html) | LST/QST interpolation with Kabsch alignment |
| [`pes_scan`](api/omecp/pes_scan/index.html) | 1D and 2D PES scan implementation |
| [`reaction_path`](api/omecp/reaction_path/index.html) | Coordinate driving and NEB optimization |
| [`checkpoint`](api/omecp/checkpoint/index.html) | JSON checkpoint system for restart capability |
| [`cleanup`](api/omecp/cleanup/index.html) | Automated temporary file management |
| [`settings`](api/omecp/settings/index.html) | Hierarchical INI configuration system |
| [`naming`](api/omecp/naming/index.html) | Dynamic file naming based on input basename |
| [`template_generator`](api/omecp/template_generator/index.html) | Input template generation from geometry files |
| [`validation`](api/omecp/validation/index.html) | Run mode and program combination validation |
| [`help`](api/omecp/help/index.html) | Built-in help system text |

---

## Generating API Docs Locally

```bash
cargo doc --no-deps --all-features --open
```

This builds and opens the full API documentation in your browser.
