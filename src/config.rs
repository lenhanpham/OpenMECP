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
/// # Units
///
/// - **Displacement thresholds** (`rms`, `max_dis`): Angstrom (Å)
/// - **Gradient thresholds** (`max_g`, `rms_g`): Hartree/Angstrom (Ha/Å)
/// - **Energy threshold** (`de`): Hartree (Ha)
///
/// # Default Values
///
/// - Energy difference (ΔE): 0.000050 Ha (~0.00136 eV)
/// - RMS displacement: 0.0025 Å
/// - Max displacement: 0.004 Å
/// - Max gradient: 0.001323 Ha/Å
/// - RMS gradient: 0.000945 Ha/Å
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thresholds {
    /// Energy difference threshold (ΔE < delta_e) in Hartree.
    pub delta_e: f64,
    /// RMS displacement threshold in Angstrom (Å).
    pub rms_dis: f64,
    /// Maximum displacement threshold in Angstrom (Å).
    pub max_dis: f64,
    /// Maximum gradient threshold in Hartree/Angstrom (Ha/Å).
    pub max_grad: f64,
    /// RMS gradient threshold in Hartree/Angstrom (Ha/Å).
    pub rms_grad: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            delta_e: 0.000050,
            rms_dis: 0.0025,
            max_dis: 0.004,
            // All gradients are in Ha/Å (standardized internal unit).
            // Python KST48 reference uses Ha/Bohr (0.0007 / 0.0005).
            // Converted: 0.0007 × 1.8897 = 0.001323, 0.0005 × 1.8897 = 0.000945
            max_grad: 0.001323,
            rms_grad: 0.000945,
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
/// - `max_step_size`: 0.1 Å
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
    /// Maximum step size in Angstrom (Å).
    pub max_step_size: f64,
    /// Step size reduction factor for line search
    pub reduced_factor: f64,
    /// Number of processors for QM calculations
    pub nprocs: usize,
    /// Memory allocation (e.g., "4GB", "8GB")
    pub mem: String,
    /// Charge for system (shared for both states)
    pub charge: i32,
    /// Separate charge for state B (None = same as charge)
    pub charge_b: Option<i32>,
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
    /// RMS gradient threshold for phase 1→2 switch in sequential hybrid (default: 0.005 Ha/Å)
    ///
    /// Li & Frisch JCTC 2006: "switches to GEDIIS when the root-mean-square force
    /// of the latest point is smaller than 10⁻² au"
    /// 10⁻² Ha/Bohr ≈ 0.005 Ha/Å
    pub gediis_switch_rms: f64,
    /// RMS displacement threshold for phase 2→3 switch in sequential hybrid (default: 0.001 Å)
    ///
    /// Li & Frisch JCTC 2006: "when the root-mean-square RFO step of the latest
    /// point is less than 2.5×10⁻³ au, GDIIS is used until convergence"
    /// 2.5×10⁻³ Bohr ≈ 0.0013 Å, rounded to 0.001 Å
    pub gediis_switch_step: f64,
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

    /// Verbosity level for output:
    /// - `0`: quiet (only convergence info)
    /// - `1`: normal (step details)
    /// - `2`: verbose (debug messages including DIIS internals)
    pub print_level: usize,

    // ========== New Fortran-ported DIIS Options ==========

    /// Use the new Fortran-ported DIIS implementations (more robust).
    ///
    /// When enabled, uses `GdiisOptimizer` and `GediisOptimizer` classes
    /// which include:
    /// - SR1 inverse matrix updates for GDIIS
    /// - Multiple GEDIIS variants (RFO, Energy, Simultaneous)
    /// - Cosine and coefficient validation
    /// - Energy rise tracking and adaptive variant selection
    ///
    /// **Default**: false (for backward compatibility)
    pub use_robust_diis: bool,

    /// GEDIIS variant selection (only used when use_robust_diis = true).
    ///
    /// Options:
    /// - "auto": Automatically select based on RMS error and energy trend
    /// - "rfo": RFO-DIIS using quadratic step overlaps
    /// - "energy": Energy-DIIS using gradient-coordinate products
    /// - "simultaneous": Combines Energy-DIIS with quadratic terms
    ///
    /// When `use_robust_diis = false` (standard mode):
    /// - "sequential" (default): Standard GDIIS/GEDIIS using mean inverse Hessian
    /// - "blend": New Python-faithful GDIIS_blend using mean true Hessian inversion
    ///
    /// Also settable via `use_gediis = blend` / `use_gediis = sequential` in input.
    ///
    /// **Default**: "auto"
    pub gediis_variant: String,

    /// Blend mode for hybrid GEDIIS+GDIIS blend methods (default: "fixed_sequential").
    ///
    /// - `"fixed"`: 50/50 fixed blend of GDIIS and EDIIS geometries.
    /// - `"fixed_sequential"`: 50/50 blend far, pure GDIIS near convergence (default).
    /// - `"gradient"`: Gradient-weighted blend (EDIIS when far, GDIIS near minimum).
    /// - `"sequential"`: Phased switching (GDIIS → weighted → GDIIS).
    ///
    /// Only used when `use_hybrid_gediis = true` and `use_gediis = "blend"`.
    pub gediis_blend_mode: String,

    /// Cosine check mode for GDIIS step validation.
    ///
    /// Controls how the GDIIS step direction is validated against the last error vector.
    /// Options:
    /// - "none": No cosine check
    /// - "zero": CosLim = 0.0
    /// - "standard": CosLim = 0.71 (recommended)
    /// - "variable": Variable limit based on number of vectors
    /// - "strict": CosLim = 0.866
    ///
    /// **Default**: "standard"
    pub gdiis_cosine_check: String,

    /// Coefficient check mode for GDIIS validation.
    ///
    /// Controls validation of DIIS coefficients to prevent excessive extrapolation.
    /// Options:
    /// - "none": No coefficient check
    /// - "regular": Standard coefficient check
    /// - "force_recent": Force recent vectors to have larger weight
    /// - "combined": Regular + ForceRecent
    ///
    /// **Default**: "regular"
    pub gdiis_coeff_check: String,

    /// Number of negative Hessian eigenvalues for saddle point search.
    ///
    /// - 0: Minimum search (default)
    /// - 1: Transition state search
    /// - >1: Higher-order saddle points (rare)
    ///
    /// Affects GEDIIS variant selection and TS scaling.
    /// **Default**: 0
    pub n_neg: usize,

    /// RMS error threshold for GEDIIS variant switching (SimSw in Fortran).
    ///
    /// When RMS error > this threshold, Energy-DIIS is preferred.
    /// When RMS error <= this threshold, RFO-DIIS is used.
    ///
    /// **Default**: 0.0025 (matching Fortran)
    pub gediis_sim_switch: f64,

    // ========== Hessian Update Method ==========

    /// Hessian update method selection.
    ///
    /// Options:
    /// - `DirectPsb`: Direct Hessian + PSB update (default, recommended).
    ///   Solves `B × dk = -g` via LU decomposition. Better for saddle-point-like MECP.
    /// - `InverseBfgs`: Inverse H⁻¹ + BFGS (Fortran-style).
    /// - `Bofill`: Weighted Powell/MS for saddle points. Direct Hessian.
    /// - `Powell`: Symmetric rank-one (SR1) update. Direct Hessian.
    /// - `BfgsPowellMix`: Adaptive BFGS/Powell blend. Direct Hessian.
    ///
    /// **Default**: DirectPsb
    ///
    /// **Compatibility**:
    /// - All direct Hessian methods (DirectPsb, Bofill, Powell, BfgsPowellMix)
    ///   are compatible with `use_gediis = blend`.
    /// - `InverseBfgs` is **incompatible** with `use_gediis = blend`.
    pub hessian_method: HessianMethod,

    /// Multiplier for RMS gradient threshold in step reduction activation.
    /// Step reduction triggers when history norm < rms_g × this value.
    /// Default: 10.0
    pub step_reduction_multiplier: f64,

    /// Trust radius reduction factor when energy increases.
    /// Default: 0.5
    pub trust_reduction_factor: f64,

    /// Trust radius increase factor when energy decreases.
    /// Default: 1.2
    pub trust_increase_factor: f64,

    /// Energy increase threshold (Ha) for trust radius reduction.
    /// Default: 0.0001
    pub trust_inc_threshold: f64,

    /// Energy decrease threshold (Ha) for trust radius increase.
    /// Default: 0.0001
    pub trust_dec_threshold: f64,

    /// Minimum trust radius in Angstrom.
    /// Default: 0.01
    pub trust_min_radius: f64,

    /// Maximum trust radius in Angstrom.
    /// Default: 1.0
    pub trust_max_radius: f64,

    /// Steepest descent step size in Angstrom for fallback when DIIS/BFGS fails.
    /// Default: 0.01
    pub steepest_descent_step: f64,
}

/// Hessian update method for geometry optimization.
///
/// Determines how the Hessian matrix (or its inverse) is stored and updated
/// at each optimization step. The choice affects convergence behavior,
/// especially near crossing points.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HessianMethod {
    /// Direct Hessian + PSB (Powell-Symmetric-Broyden) update.
    /// Stores the full Hessian H (Ha/Å²), solves H·dk = -g via LU.
    /// Recommended for MECP. Required for blend optimizer.
    DirectPsb,
    /// Inverse H⁻¹ + BFGS update (Fortran-style).
    /// Stores H⁻¹ (Å²/Ha), computes step as H⁻¹·g via multiply.
    /// Incompatible with blend optimizer.
    InverseBfgs,
    /// Bofill weighted update for saddle-point-like crossings.
    /// Direct Hessian. Blends Powell and Murtagh-Sargent.
    Bofill,
    /// Powell symmetric rank-one (SR1) update.
    /// Direct Hessian.
    Powell,
    /// Adaptive BFGS/Powell blend with Bofill weighting.
    /// Direct Hessian.
    BfgsPowellMix,
}

impl HessianMethod {
    /// Returns `true` if this method stores a direct Hessian H (not inverse).
    ///
    /// Direct Hessian methods solve linear systems `H·e = F` to compute
    /// error vectors. They are required for the blend optimizer.
    pub fn is_direct(&self) -> bool {
        matches!(
            self,
            HessianMethod::DirectPsb
                | HessianMethod::Bofill
                | HessianMethod::Powell
                | HessianMethod::BfgsPowellMix
        )
    }

    /// Returns `true` if this method stores an inverse Hessian H⁻¹.
    pub fn is_inverse(&self) -> bool {
        matches!(self, HessianMethod::InverseBfgs)
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            thresholds: Thresholds::default(),
            max_steps: 100,
            // Python MAX_STEP_SIZE = 0.1 — this is in ANGSTROM (not Bohr!)
            // Verified: Python log shows step = 0.1 Å, RMS = 0.1/sqrt(3N) in Å
            max_step_size: 0.1,
            reduced_factor: 0.5,
            nprocs: 1,
            mem: "1GB".to_string(),
            charge: 0,
            charge_b: None,
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
            use_gediis: false, // GDIIS is the default (robust and well-tested)
            use_hybrid_gediis: false, // Enable for Li & Frisch JCTC 2006 sequential hybrid
            gediis_switch_rms: 0.005, // 10⁻² Ha/Bohr → Ha/Å for phase 1→2 switch
            gediis_switch_step: 0.001, // 2.5×10⁻³ Bohr → Å for phase 2→3 switch
            switch_step: 3,          // Default to current behavior (BFGS for first 3 steps)
            bfgs_rho: 15.0,
            max_history: 4, // Match Python's history size (keeps max 4 gradients)
            print_checkpoint: false, // Default to saving checkpoints for backward compatibility
            smart_history: false, // Default to traditional FIFO history management
            print_level: 1,      // Default: normal output level
            // New Fortran-ported DIIS options
            use_robust_diis: false,
            gediis_variant: "sequential".to_string(),
            gediis_blend_mode: "fixed_sequential".to_string(),
            gdiis_cosine_check: "standard".to_string(),
            gdiis_coeff_check: "regular".to_string(),
            n_neg: 0,
            gediis_sim_switch: 0.0025,
            // Hessian update method (default: direct PSB)
            hessian_method: HessianMethod::DirectPsb,
            // Step reduction and trust radius parameters
            step_reduction_multiplier: 10.0,
            trust_reduction_factor: 0.5,
            trust_increase_factor: 1.2,
            trust_inc_threshold: 0.0001,
            trust_dec_threshold: 0.0001,
            trust_min_radius: 0.01,
            trust_max_radius: 1.0,
            steepest_descent_step: 0.01,
        }
    }
}

impl Config {
    /// Returns `true` when the GEDIIS variant is set to `"blend"`.
    ///
    /// `"blend"` activates the new Python-faithful GDIIS_blend implementation
    /// with blended mean Hessian error vectors and trust-region step control.
    pub fn gediis_variant_is_blend(&self) -> bool {
        self.gediis_variant == "blend"
    }

    /// Validates compatibility between the Hessian method and optimizer variant.
    ///
    /// The blend optimizer (`use_gediis = blend`) requires a direct Hessian method
    /// because it solves `H·e = F` via LU decomposition. Inverse BFGS is incompatible.
    ///
    /// # Returns
    ///
    /// - `Ok(())` if the combination is valid
    /// - `Err(String)` with a descriptive error message if invalid
    pub fn validate_hessian_optimizer_compatibility(&self) -> Result<(), String> {
        if self.gediis_variant_is_blend() && self.hessian_method.is_inverse() {
            Err(
                "Incompatible settings: use_gediis = blend requires a direct Hessian method\n\
                 (direct_psb, bofill, powell, or bfgs_powell_mix), but hessian = inverse_bfgs was selected.\n\
                 To fix: set hessian = direct_psb (recommended) or use use_gediis = false/sequential."
                    .to_string(),
            )
        } else {
            Ok(())
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
