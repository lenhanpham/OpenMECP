#![deny(missing_docs)]

//! OpenMECP - High-Performance Minimum Energy Crossing Point Optimizer
//!
//! OpenMECP is a Rust implementation of the Harvey et al. algorithm for locating
//! Minimum Energy Crossing Points (MECP) between two potential energy surfaces
//! in quantum chemistry calculations.
//!
//! # Overview
//!
//! The MECP is the geometry where two electronic states have the same energy but
//! different wavefunctions. This is crucial for understanding:
//! - Photochemical reactions
//! - Spin-forbidden processes
//! - Intersystem crossing
//! - Conical intersections
//! - Non-adiabatic dynamics
//!
//! # Algorithm
//!
//! OpenMECP implements the Harvey et al. algorithm (Chem. Phys. Lett. 1994) with
//! modern optimization strategies. The MECP gradient combines two components:
//!
//! 1. **f-vector**: Drives energy difference to zero
//!    ```text
//!    f = (E1 - E2) * x_norm
//!    ```
//!
//! 2. **g-vector**: Minimizes energy perpendicular to gradient difference
//!    ```text
//!    g = f1 - (x_norm · f1) * x_norm
//!    ```
//!
//! Where `x_norm = (f1 - f2) / |f1 - f2|` is the normalized gradient difference.
//!
//! # Features
//!
//! - **Multiple Optimizers**: BFGS, GDIIS, GEDIIS with automatic switching
//! - **QM Program Support**: Gaussian, ORCA, Bagel, XTB, Custom interfaces
//! - **Constraint System**: Bond, angle, and fixed atom constraints with Lagrange multipliers
//! - **Advanced Methods**: LST interpolation, NEB, PES scans, coordinate driving
//! - **Run Modes**: Normal, Read, NoRead, Stable, InterRead modes for various scenarios
//! - **Energy Scanning**: 1D and 2D potential energy surface scans
//! - **Restart Capability**: Checkpoint system for restarting calculations
//! - **Multi-layer QM/MM**: ONIOM support for hybrid calculations
//!
//! # Quick Start
//!
//! ```no_run
//! use omecp::parser::parse_input;
//! use std::path::Path;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create configuration from file
//!     let input_data = parse_input(Path::new("input.inp"))?;
//!     let config = input_data.config;
//!
//!     // Run MECP optimization
//!     // let result = config.run()?;
//!
//!     // println!("MECP converged at step: {}", result.steps);
//!     Ok(())
//! }
//! ```
//!
//! # Supported QM Programs
//!
//! | Program | Methods | Features |
//! |---------|---------|----------|
//! | Gaussian | DFT, TD-DFT, MP2, CASSCF | Full feature support, checkpoints |
//! | ORCA | DFT, TD-DFT, CASSCF | Energy/gradient files, GBW checkpoints |
//! | XTB | GFN2-xTB | Fast semi-empirical calculations |
//! | Bagel | CASSCF, MRCI | Advanced quantum chemistry methods |
//! | Custom | Any | JSON-configurable interface |
//!
//! # Optimizers
//!
//! ## BFGS (Broyden-Fletcher-Goldfarb-Shanno)
//! - Quasi-Newton method with PSB Hessian updates
//! - Default for first 3 steps
//! - Good general-purpose optimizer
//!
//! ## GDIIS (Geometry-based Direct Inversion in Iterative Subspace)
//! - Automatically activates after 3 BFGS steps
//! - 2-3x faster convergence than BFGS
//! - Stores last 4 geometries and gradients
//! - Computes error vectors for optimal step direction
//!
//! ## GEDIIS (Energy-Informed DIIS)
//! - Enhanced version of GDIIS using energy information
//! - Set `use_gediis = true` to enable
//! - 2-4x faster convergence than GDIIS for difficult cases
//! - Better handling of energy-difference minimization
//!
//! # Modules
//!
//! - [`config`](config/index.html) - Configuration structures and parsing
//! - [`geometry`](geometry/index.html) - Core geometry and state data structures
//! - [`parser`](parser/index.html) - Input file parsing
//! - [`qm_interface`](qm_interface/index.html) - QM program interfaces
//! - [`optimizer`](optimizer/index.html) - MECP optimization algorithms
//! - [`constraints`](constraints/index.html) - Geometric constraint system
//! - [`io`](io/index.html) - File I/O utilities
//! - [`lst`](lst/index.html) - Linear synchronous transit interpolation
//! - [`checkpoint`](checkpoint/index.html) - Restart functionality
//! - [`reaction_path`](reaction_path/index.html) - NEB and path optimization
//! - [`template_generator`](template_generator/index.html) - Input file templates
//! - [`help`](help/index.html) - Built-in help system
//!
//! # Input File Format
//!
//! OpenMECP uses a custom input format with section-based syntax:
//!
//! ```text
//! *GEOM
//! C  0.0  0.0  0.0
//! H  1.0  0.0  0.0
//! *
//!
//! *TAIL1
//! # Additional keywords for state 1
//! *
//!
//! *TAIL2
//! # Additional keywords for state 2
//! *
//!
//! program = gaussian
//! method = B3LYP/6-31G*
//! nprocs = 4
//! mem = 4GB
//! charge = 0
//! mult_state_a = 1  # or mult_a = 1
//! mult_state_b = 3  # or mult_b = 3
//! ```
//!
//! # Examples
//!
//! See the [project repository](https://github.com/lenhanpham/OpenMECP) for
//! complete examples including:
//! - Basic singlet-triplet MECP
//! - TD-DFT excited states
//! - Open-shell singlet calculations
//! - PES scans with constraints
//! - LST interpolation
//! - Coordinate driving
//! - ONIOM calculations
//!
//! # References
//!
//! - Harvey, J. N.; Aschi, M.; Schwarz, H.; Koch, W.
//!   *Theoret. Chim. Acta* **1994**, 90, 189-194.
//!   [DOI: 10.1007/BF01120148](https://doi.org/10.1007/BF01120148)
//!
//! # License
//!
//! MIT License - see [LICENSE](../LICENSE) file for details
//!
//! # Version
//!
//! 0.0.1 (Alpha)

/// Restart functionality
pub mod checkpoint;
/// Automated file cleanup for quantum chemistry calculations
pub mod cleanup;
pub mod config;
pub mod constraints;
pub mod geometry;
/// Built-in help system
pub mod help;
pub mod io;
/// Linear synchronous transit interpolation
pub mod lst;
/// Dynamic file naming based on input file basename
pub mod naming;
pub mod optimizer;
pub mod parser;
/// PES scanning functionality
pub mod pes_scan;
pub mod qm_interface;
/// NEB and path optimization
pub mod reaction_path;
/// Configuration management system
pub mod settings;
/// Input file templates
pub mod template_generator;
/// Run mode validation and compatibility checking
pub mod validation;

pub use config::Config;
pub use geometry::Geometry;
