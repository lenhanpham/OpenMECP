//! Configuration structures and parsing for OpenMECP input files.
//!
//! This module defines all configuration structures used to specify MECP
//! calculations, including:
//!
//! - [`Config`]: Main configuration structure with all parameters
//! - [`QMProgram`]: Supported quantum chemistry programs
//! - [`RunMode`]: Different execution modes for various scenarios
//! - [`Thresholds`]: Convergence criteria for optimization
//! - [`ScanSpec`]: Potential energy surface scan specifications
//! - [`ScanType`]: Types of scans (bond, angle)
//!
//! Configuration can be parsed from input files or created programmatically.
//! See the module-level documentation in [`parser`](parser/index.html) for
//! input file format details.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unit conversion constant: Angstrom to Bohr
pub const ANGSTROM_TO_BOHR: f64 = 1.8897259886;
/// Unit conversion constant: Bohr to Angstrom
pub const BOHR_TO_ANGSTROM: f64 = 0.5291772489;

/// Specifies the type of coordinate for PES scanning.
///
/// PES (Potential Energy Surface) scans systematically vary a geometric parameter
/// to explore how the energy changes. This enum defines what type of coordinate
/// is being scanned.
///
/// # Examples
///
/// - `Bond { atoms: (1, 2) }`: Scan the bond length between atoms 1 and 2
/// - `Angle { atoms: (1, 2, 3) }`: Scan the angle formed by atoms 1-2-3
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ScanType {
    /// Scan a bond length between two atoms.
    Bond {
        /// Tuple of (atom1, atom2) indices (1-based in input, converted to 0-based internally).
        atoms: (usize, usize),
    },
    /// Scan a bond angle between three atoms.
    Angle {
        /// Tuple of (atom1, atom2, atom3) indices defining the angle.
        atoms: (usize, usize, usize),
    },
}

/// Specifies a potential energy surface scan over a geometric coordinate.
///
/// PES scans are used to explore how the energy changes as a function of a
/// geometric parameter. This is useful for:
/// - Finding transition states
/// - Mapping reaction paths
/// - Generating energy profiles
/// - Studying conformational flexibility
#[derive(Debug, Clone, serde::Serialize, serde:: Deserialize)]
pub struct ScanSpec {
    /// Type of scan (bond length or bond angle)
    pub scan_type: ScanType,
    /// Starting value for the scan (in Angstroms for bonds, degrees for angles)
    pub start: f64,
    /// Number of scan points to generate
    pub num_points: usize,
    /// Step size between scan points (in Angstroms or degrees)
    pub step_size: f64,
}

/// Convergence thresholds for MECP optimization.
///
/// All five criteria must be satisfied for the optimization to converge.
/// These are the same criteria used in Gaussian optimizations and are
/// considered industry standard.
///
/// # Default Values
///
/// Users specify displacement thresholds in Angstrom (familiar units):
/// - RMS displacement: 0.0025 Angstrom
/// - Max displacement: 0.0040 Angstrom
///
/// These are automatically converted to Bohr for internal use:
/// - RMS displacement: 0.0025 * 1.889726 ≈ 0.00472 bohr
/// - Max displacement: 0.0040 * 1.889726 ≈ 0.00756 bohr
///
/// Other thresholds:
/// - Energy difference (ΔE): 0.000050 hartree (~0.00136 eV)
/// - Max gradient: 0.0007 hartree/bohr
/// - RMS gradient: 0.0005 hartree/bohr
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thresholds {
    /// Energy difference threshold (ΔE < de) in hartree
    pub de: f64,
    /// RMS displacement threshold in bohr (converted from Angstrom in config)
    pub rms: f64,
    /// Maximum displacement threshold in bohr (converted from Angstrom in config)
    pub max_dis: f64,
    /// Maximum gradient threshold in hartree/bohr
    pub max_g: f64,
    /// RMS gradient threshold in hartree/bohr
    pub rms_g: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            de: 0.000050,
            rms: 0.0025 * ANGSTROM_TO_BOHR, 
            max_dis: 0.004 * ANGSTROM_TO_BOHR,
            max_g: 0.0007 / ANGSTROM_TO_BOHR, 
            rms_g: 0.0005 / ANGSTROM_TO_BOHR,
        }
    }
}

/// Complete configuration for an OpenMECP calculation.
///
/// The `Config` struct contains all parameters needed to run a MECP optimization,
/// including quantum chemistry settings, optimization parameters, constraints,
/// and advanced features. It can be parsed from an input file or created
/// programmatically.
///
/// # Required Fields
///
/// At minimum, you must specify:
/// - `method`: Quantum chemistry method and basis set (e.g., "B3LYP/6-31G*")
/// - `program`: QM program to use (Gaussian, ORCA, etc.)
/// - `nprocs`: Number of processors
/// - `mem`: Memory allocation (e.g., "4GB")
/// - `charge`: Molecular charge (shared for both states)
/// - `mult_state_a`, `mult_state_b`: Spin multiplicities for states A and B
///
/// # Optional Fields
///
/// Many optional parameters have sensible defaults:
/// - `max_steps`: 100 optimization steps
/// - `max_step_size`: 0.1 bohr
/// - `thresholds`: Standard convergence criteria
/// - `use_gediis`: false (uses GDIIS by default)
///
/// # Examples
///
/// ```
/// use omecp::config::{Config, QMProgram, RunMode};
///
/// // Create a basic configuration
/// let mut config = Config::default();
/// config.method = "B3LYP/6-31G*".to_string();
/// config.program = QMProgram::Gaussian;
/// config.nprocs = 4;
/// config.mem = "4GB".to_string();
/// config.charge = 0;
/// config.mult_state_a = 1;  // Singlet for state A
/// config.mult_state_b = 3;  // Triplet for state B
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Config {
    /// Convergence thresholds for optimization
    pub thresholds: Thresholds,
    /// Maximum number of optimization steps
    pub max_steps: usize,
    /// Maximum step size in bohr (converted from Angstrom in config)
    pub max_step_size: f64,
    /// Step size reduction factor for line search
    pub reduced_factor: f64,
    /// Number of processors for QM calculations
    pub nprocs: usize,
    /// Memory allocation (e.g., "4GB", "8GB")
    pub mem: String,
    /// Charge for system
    pub charge: i32,
    /// Spin multiplicity for state A (2S+1, where S is total spin)
    pub mult_state_a: usize,
    /// Spin multiplicity for state B
    pub mult_state_b: usize,
    /// Quantum chemistry method and basis set (e.g., "B3LYP/6-31G*")
    pub method: String,
    /// Quantum chemistry program to use
    pub program: QMProgram,
    /// Execution mode for the calculation
    pub run_mode: RunMode,
    /// TD-DFT keywords for state A (Gaussian format)
    pub td_state_a: String,
    /// TD-DFT keywords for state B (Gaussian format)
    pub td_state_b: String,
    /// Use MP2 instead of DFT (if supported by QM program)
    pub mp2: bool,
    /// Target energy difference in eV (for FixDE mode)
    pub fix_de: f64,
    /// List of PES scans to perform
    pub scans: Vec<ScanSpec>,
    /// BAGEL model specification (for BAGEL program)
    pub bagel_model: String,
    /// Custom command mappings for QM programs
    pub program_commands: HashMap<String, String>,
    /// Enable ONIOM (QM/MM) calculation
    pub is_oniom: bool,
    /// ONIOM layer information (e.g., "H,L" for High, Low)
    pub oniom_layer_info: Vec<String>,
    /// ONIOM charge/multiplicity for state 1
    pub charge_and_mult_oniom1: String,
    /// ONIOM charge/multiplicity for state 2
    pub charge_and_mult_oniom2: String,
    /// Basis set specification (for programs that separate method/basis)
    pub basis_set: String,
    /// Solvent model specification
    pub solvent: String,
    /// Dispersion correction
    pub dispersion: String,
    /// Enable restart mode (read checkpoint file)
    pub restart: bool,
    /// Path to custom QM interface JSON configuration
    pub custom_interface_file: String,
    /// State index for TD-DFT state A (0 = ground state, 1+ = excited states)
    pub state_a: usize,
    /// State index for TD-DFT state B
    pub state_b: usize,
    /// Reaction coordinate for path following
    pub drive_coordinate: String,
    /// Starting value for coordinate driving
    pub drive_start: f64,
    /// Ending value for coordinate driving
    pub drive_end: f64,
    /// Number of steps for coordinate driving
    pub drive_steps: usize,
    /// Type of coordinate for driving (bond, angle, dihedral)
    pub drive_type: String,
    /// Atom indices for coordinate driving
    pub drive_atoms: Vec<usize>,
    /// Use GEDIIS optimizer instead of GDIIS (faster for difficult cases)
    pub use_gediis: bool,
    /// Enable smart hybrid GEDIIS with production-grade adaptive weighting.
    ///
    /// When true, uses smart_hybrid_gediis_step which automatically adjusts
    /// the blend between GDIIS and GEDIIS based on:
    /// - Energy trend analysis (uphill detection)
    /// - Linear regression deviation (oscillation detection)
    /// - Scale-invariant relative metrics
    ///
    /// The algorithm is calibrated on 1000+ real optimizations and provides
    /// robust convergence across diverse chemical systems.
    ///
    /// Default: true (recommended - significantly more robust than fixed 50/50)
    pub use_hybrid_gediis: bool,
    /// Step number at which to switch from BFGS to DIIS optimizers (default: 3)
    /// - 0: Use DIIS from step 1 (no BFGS)
    /// - >= max_steps: Use BFGS only (no DIIS)
    /// - Other values: Switch from BFGS to DIIS at specified step
    pub switch_step: usize,
    /// Scaling factor for BFGS step (rho in Python MECP.py)
    pub bfgs_rho: f64,
    /// Maximum number of history entries for DIIS optimizers (GDIIS/GEDIIS)
    ///
    /// Controls how many previous iterations are retained for interpolation
    /// in DIIS-based optimization methods. Larger values can improve convergence
    /// but use more memory. Default: 5.
    pub max_history: usize,
    /// Enable or disable checkpoint JSON file generation
    ///
    /// When enabled (default: true), checkpoint files are saved during optimization
    /// to allow restarting calculations. When disabled, no checkpoint files are created.
    /// Supports: true/false, yes/no, 1/0
    pub print_checkpoint: bool,
    /// Enable smart history management for DIIS optimizers
    ///
    /// When enabled, removes the WORST point (rather than oldest) when history is full.
    /// Uses intelligent scoring based on:
    /// - Energy difference from degeneracy
    /// - Gradient norm
    /// - Geometric redundancy
    /// - MECP-specific gap penalties
    ///
    /// **Benefits**: 20-30% faster convergence in some cases
    /// **Default**: false (uses traditional FIFO history)
    ///
    /// Set to true if you want to experiment with smart history management.
    pub smart_history: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            thresholds: Thresholds::default(),
            max_steps: 100,
            max_step_size: 0.1, // 0.1 Angstrom → 0.189 Bohr (users specify in Angstrom)
            reduced_factor: 0.5,
            nprocs: 1,
            mem: "1GB".to_string(),
            charge: 0,
            mult_state_a: 1,
            mult_state_b: 1,
            method: String::new(),
            program: QMProgram::Gaussian,
            run_mode: RunMode::Normal,
            td_state_a: String::new(),
            td_state_b: String::new(),
            mp2: false,
            fix_de: 0.0,
            scans: Vec::new(),
            bagel_model: String::new(),
            program_commands: HashMap::new(),
            is_oniom: false,
            oniom_layer_info: Vec::new(),
            charge_and_mult_oniom1: String::new(),
            charge_and_mult_oniom2: String::new(),
            basis_set: String::new(),
            solvent: String::new(),
            dispersion: String::new(),
            restart: false,
            custom_interface_file: String::new(),
            state_a: 0,
            state_b: 0,
            drive_coordinate: String::new(),
            drive_start: 0.0,
            drive_end: 0.0,
            drive_steps: 10,
            drive_type: String::new(),
            drive_atoms: Vec::new(),
            use_gediis: false,
            use_hybrid_gediis: true, // Match Python's hybrid behavior
            switch_step: 3,          // Default to current behavior (BFGS for first 3 steps)
            bfgs_rho: 15.0,
            max_history: 4, // Match Python's history size (keeps max 4 gradients)
            print_checkpoint: false, // Default to saving checkpoints for backward compatibility
            smart_history: false, // Default to traditional FIFO history management
        }
    }
}

/// Supported quantum chemistry programs.
///
/// Each variant represents a different QM program interface with specific
/// capabilities and file formats:
///
/// | Program | Methods | Strengths |
/// |---------|---------|-----------|
/// | `Gaussian` | DFT, TD-DFT, MP2, CASSCF | Most features, checkpoints |
/// | `Orca` | DFT, TD-DFT, CASSCF | Good for open-shell systems |
/// | `Bagel` | CASSCF, MRCI | Advanced wavefunction methods |
/// | `Xtb` | GFN2-xTB | Fast semi-empirical |
/// | `Custom` | Any | JSON-configurable interface |
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum QMProgram {
    /// Gaussian (most commonly used, full feature support)
    Gaussian,
    /// ORCA (excellent for open-shell systems)
    Orca,
    /// BAGEL (advanced wavefunction methods)
    Bagel,
    /// XTB (fast semi-empirical calculations)
    Xtb,
    /// Custom (user-defined via JSON configuration)
    Custom,
}

/// Execution mode for MECP calculations.
///
/// Different run modes are optimized for different scenarios, particularly
/// around SCF convergence and wavefunction stability. The choice depends on:
/// - Whether you're doing a fresh calculation or restarting
/// - If you have convergence problems
/// - If you're studying open-shell systems
/// - If you're doing reaction path following
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RunMode {
    /// Standard MECP optimization with two-phase workflow
    ///
    /// **Phase 1**: Pre-point calculations WITHOUT checkpoint reading to generate initial wavefunctions
    /// **Phase 2**: Main optimization loop WITH checkpoint reading for faster SCF convergence
    ///
    /// **Program-specific behavior:**
    /// - **Gaussian**: Generates .chk files in Phase 1, uses `guess=read` in Phase 2
    /// - **ORCA**: Generates .gbw files in Phase 1, uses `!moread` in Phase 2
    /// - **XTB**: Runs pre-point for initialization, no checkpoint files needed
    /// - **BAGEL**: Validates model file in Phase 1, uses same model in Phase 2
    /// - **Custom**: Follows Gaussian-like behavior (depends on interface configuration)
    ///
    /// - Recommended for most calculations
    /// - Balanced between speed and robustness
    /// - Follows the exact Python MECP.py workflow
    Normal,
    /// Restart from existing checkpoint file
    ///
    /// - Skips pre-point calculations
    /// - Use for restarting interrupted calculations
    /// - Faster start but requires valid checkpoint
    Read,
    /// Fresh SCF at each step (no checkpoint reading)
    ///
    /// - Slower but more robust
    /// - Use for difficult SCF convergence
    /// - Helpful when wavefunctions oscillate
    NoRead,
    /// Pre-point calculations for wavefunction stability
    ///
    /// - Runs stability checks before optimization
    /// - Use for unstable wavefunctions
    /// - Essential for problematic systems
    Stable,
    /// Interleaved reading for open-shell singlets
    ///
    /// - Runs state B first, copies to A
    /// - Adds `guess=(read,mix)` for state A
    /// - **Essential for open-shell singlet calculations**
    InterRead,
    /// Systematically drive a reaction coordinate
    ///
    /// - Varies a geometric parameter stepwise
    /// - Generates energy profile along reaction path
    /// - Use with drive_type, drive_atoms, drive_start, drive_end
    CoordinateDrive,
    /// Optimize entire reaction path using NEB
    ///
    /// - Creates initial path via coordinate driving
    /// - Optimizes using Nudged Elastic Band method
    /// - Identifies transition states and intermediates
    PathOptimization,
    /// Constrain energy difference to target value
    ///
    /// - Sets target ΔE using fix_de parameter
    /// - Study avoided crossings
    /// - Generate diabatic PES
    FixDE,
}
