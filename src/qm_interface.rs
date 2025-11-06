//! Quantum chemistry program interfaces for MECP calculations.
//!
//! This module provides a unified interface for running calculations with different
//! quantum chemistry programs. It abstracts away the differences between programs
//! (Gaussian, ORCA, Bagel, XTB, custom interfaces) and provides a consistent API
//! for:
//!
//! - Writing input files
//! - Executing calculations
//! - Parsing output files
//! - Extracting energies, forces, and geometries
//!
//! # Supported Programs
//!
//! - **Gaussian**: Full feature support with checkpoint files, TD-DFT, MP2, etc.
//! - **ORCA**: Energy and gradient files with GBW checkpoints
//! - **Bagel**: CASSCF and MRCI calculations with JSON model files
//! - **XTB**: Fast semi-empirical calculations using GFN2-xTB
//! - **Custom**: JSON-configurable interface for any program
//!
//! # Interface Design
//!
//! The [`QMInterface`] trait defines the contract that all QM programs must implement.
//! Each program has its own implementation that handles:
//! - Program-specific input file format
//! - Program execution commands and flags
//! - Output parsing and data extraction
//! - Error handling specific to the program
//!
//! # Usage Pattern
//!
//! ```rust
//! use omecp::qm_interface::{QMInterface, GaussianInterface};
//!
//! let gaussian = GaussianInterface::new("g16".to_string(), false);
//! gaussian.write_input(&geometry, &header, &tail, Path::new("input.gjf"))?;
//! gaussian.run_calculation(Path::new("input.gjf"))?;
//! let state = gaussian.read_output(Path::new("output.log"), 0)?;
//! ```
//!
//! # Error Handling
//!
//! All operations return a [`QMError`] result that can be:
//! - `IO`: File system errors (missing files, permission issues)
//! - `Calculation`: QM program execution failures
//! - `Parse`: Output parsing errors (malformed or unexpected output)
//!
//! # State Extraction
//!
//! Each call to `read_output` extracts a [`State`] containing:
//! - Energy (in hartree)
//! - Forces/gradients (in hartree/bohr)
//! - Final geometry

use crate::geometry::{Geometry, State};
use crate::io;
use lazy_static::lazy_static;
use nalgebra::DVector;
use regex::Regex;

use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};
use thiserror::Error;

/// Error type for QM interface operations.
///
/// QM calculations can fail at three stages:
/// 1. **I/O**: File operations (reading/writing input/output files)
/// 2. **Calculation**: Program execution (segfaults, convergence failures, etc.)
/// 3. **Parsing**: Output file interpretation (malformed data, unexpected format)
#[derive(Error, Debug)]
pub enum QMError {
    /// File system or I/O operation failed
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// Quantum chemistry program execution failed
    #[error("QM calculation failed: {0}")]
    Calculation(String),
    /// Failed to parse program output
    #[error("Parse error: {0}")]
    Parse(String),
}

/// Type alias for QM operation results
type Result<T> = std::result::Result<T, QMError>;

/// Trait that defines the interface for all quantum chemistry programs.
///
/// This trait must be implemented by each QM program to provide a uniform
/// interface for MECP calculations. It handles the complete lifecycle of
/// a single-point calculation: input generation, execution, and output parsing.
pub trait QMInterface {
    /// Writes a calculation input file for the quantum chemistry program.
    ///
    /// # Arguments
    ///
    /// * `geom` - Molecular geometry with element types and coordinates
    /// * `header` - Program-specific header section (methods, basis sets, options)
    /// * `tail` - Additional keywords or sections specific to the calculation
    /// * `path` - Output path for the input file
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or a `QMError` if file writing fails.
    fn write_input(&self, geom: &Geometry, header: &str, tail: &str, path: &Path) -> Result<()>;

    /// Executes the quantum chemistry calculation.
    ///
    /// Runs the external QM program with the input file and waits for completion.
    /// The program output is captured and checked for success status.
    ///
    /// # Arguments
    ///
    /// * `input_path` - Path to the input file to execute
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the calculation succeeds, or a `QMError` if:
    /// - The program cannot be executed (not found, permissions)
    /// - The calculation fails (non-zero exit status)
    /// - Runtime errors occur during execution
    fn run_calculation(&self, input_path: &Path) -> Result<()>;

    /// Parses the calculation output file to extract results.
    ///
    /// Reads and parses the program output file to extract the electronic state
    /// properties needed for MECP optimization.
    ///
    /// # Arguments
    ///
    /// * `output_path` - Path to the output/log file to parse
    /// * `state` - Electronic state index to extract (0 = ground state, >0 = excited state)
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing:
    /// - `Ok(State)` - Successfully parsed state with energy, forces, and geometry
    /// - `Err(QMError::Parse)` - Failed to parse required data from output
    /// - `Err(QMError::IO)` - Cannot read output file
    ///
    /// # State Selection
    ///
    /// The `state` parameter selects which electronic state to extract:
    /// - `state = 0`: Ground state (S0 for singlet, S1 for triplet, etc.)
    /// - `state = n`: n-th excited state (for TD-DFT calculations)
    ///
    /// Different programs interpret this differently:
    /// - **Gaussian**: State index for TD-DFT excited states
    /// - **ORCA**: State index for excited state calculations
    /// - **Other programs**: May ignore this parameter
    fn read_output(&self, output_path: &Path, state: usize) -> Result<State>;
}

/// Gaussian quantum chemistry program interface.
///
/// Provides support for Gaussian calculations including:
/// - DFT, TD-DFT, MP2, and post-HF methods
/// - Excited state calculations via TD-DFT
/// - Checkpoint files for wavefunction continuity
/// - Both energy-only and gradient calculations
///
/// # Examples
///
/// ```
/// use omecp::qm_interface::GaussianInterface;
/// use std::path::Path;
///
/// // Create interface with Gaussian 16
/// let gaussian = GaussianInterface::new("g16".to_string(), false);
///
/// // Create interface with MP2 enabled
/// let gaussian_mp2 = GaussianInterface::new("g16".to_string(), true);
/// ```
pub struct GaussianInterface {
    /// Gaussian command to execute (e.g., "g16", "g09", "gview")
    pub command: String,
    /// Enable MP2 energy extraction from checkpoint archive
    pub mp2: bool,
}

impl GaussianInterface {
    /// Creates a new Gaussian interface.
    ///
    /// # Arguments
    ///
    /// * `command` - Gaussian executable command (e.g., "g16", "/path/to/g16")
    /// * `mp2` - If true, extracts MP2 energy from checkpoint archive instead of SCF
    ///
    /// # Examples
    ///
    /// ```
    /// use omecp::qm_interface::GaussianInterface;
    ///
    /// // Standard DFT calculation
    /// let g16 = GaussianInterface::new("g16".to_string(), false);
    ///
    /// // MP2 calculation
    /// let g16_mp2 = GaussianInterface::new("g16".to_string(), true);
    /// ```
    pub fn new(command: String, mp2: bool) -> Self {
        Self { command, mp2 }
    }
}

lazy_static! {
    // Robust floating-point regex: handles 1.23, -0.032, 1.2e-4, .123, etc.
    static ref FLOAT_RE: String = r"[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?".to_string();

    // Excited state line: " Excited State   1: Singlet-A 3.9741 eV ..."
    static ref EXCITED_RE: Regex = Regex::new(&format!(
        r"^\s*Excited\s+State\s+(\d+)\s*:\s*.*?({0})\s+eV",
        *FLOAT_RE
    )).unwrap();

    // CIS/TDA total energy: " Total Energy, E(CIS/TDA) = -2823.11144861"
    static ref CIS_TOTAL_RE: Regex = Regex::new(&format!(
        r"Total\s+Energy,\s+E\(CIS/TDA\)\s*=\s*({0})",
        *FLOAT_RE
    )).unwrap();

    // Force line: " 1 5 -0.032351682 0.050284933 0.089439223"
    static ref FORCE_RE: Regex = Regex::new(&format!(
        r"^\s*\d+\s+\d+\s+({0})\s+({0})\s+({0})",
        *FLOAT_RE
    )).unwrap();

    // Geometry line: " 1 5 0 -0.032351682 0.050284933 0.089439223"
    static ref GEOM_RE: Regex = Regex::new(&format!(
        r"^\s*\d+\s+(\d+)\s+\d+\s+({0})\s+({0})\s+({0})",
        *FLOAT_RE
    )).unwrap();
}

impl QMInterface for GaussianInterface {
    fn write_input(&self, geom: &Geometry, header: &str, tail: &str, path: &Path) -> Result<()> {
        let mut content = String::new();
        content.push_str(header);
        content.push('\n');

        for i in 0..geom.num_atoms {
            let coords = geom.get_atom_coords(i);
            content.push_str(&format!(
                "{}  {:.8}  {:.8}  {:.8}\n",
                geom.elements[i], coords[0], coords[1], coords[2]
            ));
        }

        content.push('\n');
        content.push_str(tail);
        content.push('\n');

        fs::write(path, content)?;
        Ok(())
    }

    fn run_calculation(&self, input_path: &Path) -> Result<()> {
        let output = Command::new(&self.command).arg(input_path).output()?;

        if !output.status.success() {
            return Err(QMError::Calculation(
                String::from_utf8_lossy(&output.stderr).to_string(),
            ));
        }
        Ok(())
    }

    fn read_output(&self, output_path: &Path, state: usize) -> Result<State> {
        let content = fs::read_to_string(output_path)
            .map_err(|e| QMError::Parse(format!("Failed to read file: {}", e)))?;
        let mut energy = 0.0_f64;
        let mut forces = Vec::new();
        let mut geom_coords = Vec::new();
        let mut elements = Vec::new();
        let mut in_forces = false;
        let mut in_geom = false;
        let mut archive_part = String::new();
        for line in content.lines() {
            let trimmed = line.trim();
            // === 1. SCF Energy ===
            if line.contains("SCF Done") {
                if let Some(eq_pos) = line.find('=') {
                    if let Some(val) = line[eq_pos + 1..].split_whitespace().next() {
                        energy = val.parse().unwrap_or(0.0);
                    }
                }
            }
            // === 2. TD-DFT / CIS / TDA Energy ===
            else if line.contains("E(TD-HF/TD-DFT)") || line.contains("E(CIS/TDA)") {
                if state == 0 {
                    // Ground state
                    if line.contains("E(TD-HF/TD-DFT)") {
                        if let Some(eq) = line.find('=') {
                            if let Some(val) = line[eq + 1..].split_whitespace().next() {
                                energy = val.parse().unwrap_or(0.0);
                            }
                        }
                    } else if let Some(caps) = CIS_TOTAL_RE.captures(line) {
                        energy = caps[1].parse().unwrap_or(0.0);
                    }
                } else {
                    // Excited state
                    if let Some(caps) = EXCITED_RE.captures(line) {
                        let line_state: usize = caps[1].parse().expect("state number");
                        if line_state == state {
                            let ev: f64 = caps[2].parse().expect("eV value");
                            energy = ev * 0.0367493; // eV → Hartree
                        }
                    }
                }
            }
            // === 3. MP2 Archive Line ===
            else if self.mp2 && trimmed.starts_with('\\') {
                archive_part = line.to_string(); // Keep last archive line
            }
            // === 4. Forces Block ===
            else if line.contains("Forces (Hartrees/Bohr)") {
                in_forces = true;
                forces.clear();
            } else if line.contains("Cartesian Forces: Max") {
                in_forces = false;
            } else if in_forces {
                if let Some(caps) = FORCE_RE.captures(line) {
                    forces.push(caps[1].parse().unwrap_or(0.0));
                    forces.push(caps[2].parse().unwrap_or(0.0));
                    forces.push(caps[3].parse().unwrap_or(0.0));
                }
            }
            // === 5. Geometry Block ===
            else if line.contains("Input orientation") || line.contains("Standard orientation") {
                in_geom = true;
                geom_coords.clear();
                elements.clear();
            } else if (line.contains("Distance matrix") || line.contains("Rotational constants"))
                && in_geom
            {
                in_geom = false;
            } else if in_geom {
                if let Some(caps) = GEOM_RE.captures(line) {
                    let atomic_num: usize = caps[1].parse().unwrap_or(0);
                    elements.push(atomic_number_to_symbol(atomic_num));
                    geom_coords.push(caps[2].parse().unwrap_or(0.0));
                    geom_coords.push(caps[3].parse().unwrap_or(0.0));
                    geom_coords.push(caps[4].parse().unwrap_or(0.0));
                }
            }
        }
        // === 6. MP2 Energy from Archive ===
        if self.mp2 && !archive_part.is_empty() {
            if let Some(mp2_pos) = archive_part.find("MP2=") {
                let after = &archive_part[mp2_pos + 4..];
                if let Some(end) = after.find('\\') {
                    let mp2_str = after[..end].trim();
                    if let Ok(mp2_e) = mp2_str.parse::<f64>() {
                        energy = mp2_e;
                    }
                }
            }
        }
        // === 7. Final Validation ===
        let n_atoms = elements.len();
        if n_atoms == 0 {
            return Err(QMError::Parse("No atoms found in geometry".into()));
        }
        if forces.len() != 3 * n_atoms {
            return Err(QMError::Parse(format!(
                "Expected {} force components, got {}",
                3 * n_atoms,
                forces.len()
            )));
        }
        if geom_coords.len() != 3 * n_atoms {
            return Err(QMError::Parse(format!(
                "Expected {} coordinates, got {}",
                3 * n_atoms,
                geom_coords.len()
            )));
        }
        
        let state = State {
            energy,
            forces: DVector::from_vec(forces),
            geometry: Geometry::new(elements, geom_coords),
        };

        // Validate the state to ensure meaningful data
        state.validate().map_err(|e| QMError::Parse(format!(
            "Gaussian state validation failed: {}. Check that the calculation completed successfully and produced valid energy/gradient data.",
            e
        )))?;

        Ok(state)
    }
}

fn atomic_number_to_symbol(num: usize) -> String {
    match num {
        1 => "H",    // Hydrogen
        2 => "He",   // Helium
        3 => "Li",   // Lithium
        4 => "Be",   // Beryllium
        5 => "B",    // Boron
        6 => "C",    // Carbon
        7 => "N",    // Nitrogen
        8 => "O",    // Oxygen
        9 => "F",    // Fluorine
        10 => "Ne",  // Neon
        11 => "Na",  // Sodium
        12 => "Mg",  // Magnesium
        13 => "Al",  // Aluminum
        14 => "Si",  // Silicon
        15 => "P",   // Phosphorus
        16 => "S",   // Sulfur
        17 => "Cl",  // Chlorine
        18 => "Ar",  // Argon
        19 => "K",   // Potassium
        20 => "Ca",  // Calcium
        21 => "Sc",  // Scandium
        22 => "Ti",  // Titanium
        23 => "V",   // Vanadium
        24 => "Cr",  // Chromium
        25 => "Mn",  // Manganese
        26 => "Fe",  // Iron
        27 => "Co",  // Cobalt
        28 => "Ni",  // Nickel
        29 => "Cu",  // Copper
        30 => "Zn",  // Zinc
        31 => "Ga",  // Gallium
        32 => "Ge",  // Germanium
        33 => "As",  // Arsenic
        34 => "Se",  // Selenium
        35 => "Br",  // Bromine
        36 => "Kr",  // Krypton
        37 => "Rb",  // Rubidium
        38 => "Sr",  // Strontium
        39 => "Y",   // Yttrium
        40 => "Zr",  // Zirconium
        41 => "Nb",  // Niobium
        42 => "Mo",  // Molybdenum
        43 => "Tc",  // Technetium
        44 => "Ru",  // Ruthenium
        45 => "Rh",  // Rhodium
        46 => "Pd",  // Palladium
        47 => "Ag",  // Silver
        48 => "Cd",  // Cadmium
        49 => "In",  // Indium
        50 => "Sn",  // Tin
        51 => "Sb",  // Antimony
        52 => "Te",  // Tellurium
        53 => "I",   // Iodine
        54 => "Xe",  // Xenon
        55 => "Cs",  // Cesium
        56 => "Ba",  // Barium
        57 => "La",  // Lanthanum
        58 => "Ce",  // Cerium
        59 => "Pr",  // Praseodymium
        60 => "Nd",  // Neodymium
        61 => "Pm",  // Promethium
        62 => "Sm",  // Samarium
        63 => "Eu",  // Europium
        64 => "Gd",  // Gadolinium
        65 => "Tb",  // Terbium
        66 => "Dy",  // Dysprosium
        67 => "Ho",  // Holmium
        68 => "Er",  // Erbium
        69 => "Tm",  // Thulium
        70 => "Yb",  // Ytterbium
        71 => "Lu",  // Lutetium
        72 => "Hf",  // Hafnium
        73 => "Ta",  // Tantalum
        74 => "W",   // Tungsten
        75 => "Re",  // Rhenium
        76 => "Os",  // Osmium
        77 => "Ir",  // Iridium
        78 => "Pt",  // Platinum
        79 => "Au",  // Gold
        80 => "Hg",  // Mercury
        81 => "Tl",  // Thallium
        82 => "Pb",  // Lead
        83 => "Bi",  // Bismuth
        84 => "Po",  // Polonium
        85 => "At",  // Astatine
        86 => "Rn",  // Radon
        87 => "Fr",  // Francium
        88 => "Ra",  // Radium
        89 => "Ac",  // Actinium
        90 => "Th",  // Thorium
        91 => "Pa",  // Protactinium
        92 => "U",   // Uranium
        93 => "Np",  // Neptunium
        94 => "Pu",  // Plutonium
        95 => "Am",  // Americium
        96 => "Cm",  // Curium
        97 => "Bk",  // Berkelium
        98 => "Cf",  // Californium
        99 => "Es",  // Einsteinium
        100 => "Fm", // Fermium
        101 => "Md", // Mendelevium
        102 => "No", // Nobelium
        103 => "Lr", // Lawrencium
        104 => "Rf", // Rutherfordium
        105 => "Db", // Dubnium
        106 => "Sg", // Seaborgium
        107 => "Bh", // Bohrium
        108 => "Hs", // Hassium
        109 => "Mt", // Meitnerium
        110 => "Ds", // Darmstadtium
        111 => "Rg", // Roentgenium
        112 => "Cn", // Copernicium
        113 => "Nh", // Nihonium
        114 => "Fl", // Flerovium
        115 => "Mc", // Moscovium
        116 => "Lv", // Livermorium
        117 => "Ts", // Tennessine
        118 => "Og", // Oganesson
        _ => "X",    // Unknown or invalid
    }
    .to_string()
}

/// ORCA quantum chemistry program interface.
///
/// Provides support for ORCA calculations including:
/// - DFT, TD-DFT, and CASSCF methods
/// - Energy and gradient parsing from .engrad files
/// - Checkpoint files (.gbw) for wavefunction continuity
///
/// # Examples
///
/// ```
/// use omecp::qm_interface::OrcaInterface;
///
/// // Create interface with ORCA
/// let orca = OrcaInterface::new("orca".to_string());
/// ```
pub struct OrcaInterface {
    /// ORCA executable command (e.g., "orca", "/path/to/orca")
    pub command: String,
}

impl OrcaInterface {
    /// Creates a new ORCA interface.
    ///
    /// # Arguments
    ///
    /// * `command` - ORCA executable command (e.g., "orca", "/path/to/orca")
    ///
    /// # Examples
    ///
    /// ```
    /// use omecp::qm_interface::OrcaInterface;
    ///
    /// let orca = OrcaInterface::new("orca".to_string());
    /// ```
    pub fn new(command: String) -> Self {
        Self { command }
    }
}

impl QMInterface for OrcaInterface {
    fn write_input(&self, geom: &Geometry, header: &str, tail: &str, path: &Path) -> Result<()> {
        let mut content = String::new();
        content.push_str(header);
        content.push('\n');

        for i in 0..geom.num_atoms {
            let coords = geom.get_atom_coords(i);
            content.push_str(&format!(
                "{}  {:.8}  {:.8}  {:.8}\n",
                geom.elements[i], coords[0], coords[1], coords[2]
            ));
        }

        content.push_str("*\n");
        content.push_str(tail);
        content.push('\n');

        fs::write(path, content)?;
        Ok(())
    }

    fn run_calculation(&self, input_path: &Path) -> Result<()> {
        let output_path = input_path.with_extension("out");
        let output_file = fs::File::create(&output_path)?;
        let status = Command::new(&self.command)
            .arg(input_path)
            .stdout(Stdio::from(output_file))
            .status()?;

        if !status.success() {
            return Err(QMError::Calculation("ORCA failed".into()));
        }
        Ok(())
    }

    fn read_output(&self, output_path: &Path, _state: usize) -> Result<State> {
        let base = output_path.with_extension("");
        let engrad_path = base.with_extension("engrad");
        let log_path = output_path;

        // Check if required files exist before attempting to read
        if !engrad_path.exists() {
            return Err(QMError::Parse(format!(
                "ORCA engrad file not found: {}. Check that ORCA calculation completed successfully and produced gradient output.",
                engrad_path.display()
            )));
        }

        if !log_path.exists() {
            return Err(QMError::Parse(format!(
                "ORCA output file not found: {}. Check that ORCA calculation completed successfully.",
                log_path.display()
            )));
        }

        let engrad_content = fs::read_to_string(&engrad_path)
            .map_err(|e| QMError::Parse(format!(
                "Failed to read ORCA engrad file {}: {}",
                engrad_path.display(), e
            )))?;
        let log_content = fs::read_to_string(&log_path)
            .map_err(|e| QMError::Parse(format!(
                "Failed to read ORCA output file {}: {}",
                log_path.display(), e
            )))?;

        let mut energy = 0.0;
        let mut forces = Vec::new();
        let mut geom_coords = Vec::new();
        let mut elements = Vec::new();

        let mut in_geom = false;
        let mut in_forces = false;

        for line in engrad_content.lines() {
            if line.contains("The atomic numbers and current coordinates in Bohr") {
                in_geom = true;
            } else if line.starts_with('#') && !geom_coords.is_empty() {
                in_geom = false;
            } else if line.contains("The current gradient") {
                in_forces = true;
            } else if line.starts_with('#') && !forces.is_empty() {
                in_forces = false;
            } else if in_geom
                && line
                    .trim()
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_digit())
            {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 4 {
                    let atomic_num: usize = parts[0].parse().unwrap_or(0);
                    elements.push(atomic_number_to_symbol(atomic_num));
                    geom_coords.push(parts[1].parse::<f64>().unwrap_or(0.0) * 0.52918);
                    geom_coords.push(parts[2].parse::<f64>().unwrap_or(0.0) * 0.52918);
                    geom_coords.push(parts[3].parse::<f64>().unwrap_or(0.0) * 0.52918);
                }
            } else if in_forces
                && line
                    .trim()
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_digit() || c == '-')
            {
                forces.push(-line.trim().parse::<f64>().unwrap_or(0.0));
            }
        }

        for line in log_content.lines() {
            if line.contains("FINAL SINGLE POINT ENERGY") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 5 {
                    energy = parts[4].parse().unwrap_or(0.0);
                }
            }
        }

        // Validate that we successfully parsed required data
        if forces.is_empty() {
            return Err(QMError::Parse(format!(
                "Failed to parse forces from ORCA engrad file: {}. Check that gradient calculation completed successfully.",
                engrad_path.display()
            )));
        }

        if geom_coords.is_empty() {
            return Err(QMError::Parse(format!(
                "Failed to parse geometry from ORCA engrad file: {}. Check that the file contains valid atomic coordinates.",
                engrad_path.display()
            )));
        }

        // Validate that we got meaningful energy (prevent zero energy failures)
        if energy == 0.0 {
            return Err(QMError::Parse(format!(
                "Failed to extract energy from ORCA output file: {}. Check that the calculation completed successfully and contains 'FINAL SINGLE POINT ENERGY'.",
                log_path.display()
            )));
        }

        // Validate force/geometry consistency
        let n_atoms = elements.len();
        if forces.len() != 3 * n_atoms {
            return Err(QMError::Parse(format!(
                "Force/geometry mismatch in ORCA output: expected {} force components for {} atoms, got {}",
                3 * n_atoms, n_atoms, forces.len()
            )));
        }

        let state = State {
            energy,
            forces: DVector::from_vec(forces),
            geometry: Geometry::new(elements, geom_coords),
        };

        // Validate the state to ensure meaningful data
        state.validate().map_err(|e| QMError::Parse(format!(
            "ORCA state validation failed: {}. Check that the calculation completed successfully and produced valid energy/gradient data.",
            e
        )))?;

        Ok(state)
    }
}

/// BAGEL quantum chemistry program interface.
///
/// Provides support for BAGEL calculations including:
/// - CASSCF and MRCI methods
/// - JSON-based input files
/// - Parsing energies and forces from BAGEL output
///
/// # Examples
///
/// ```
/// use omecp::qm_interface::BagelInterface;
///
/// // Create interface with BAGEL
/// let bagel = BagelInterface::new(
///     "bagel".to_string(),
///     "{\"job\":{\"method\":\"caspt2\"}}".to_string(),
/// );
/// ```
pub struct BagelInterface {
    /// BAGEL executable command (e.g., "bagel", "/path/to/bagel")
    pub command: String,
    /// JSON template for BAGEL input (geometry placeholder will be replaced)
    pub model_template: String,
}

/// XTB semi-empirical program interface.
///
/// Provides support for XTB calculations using GFN2-xTB method:
/// - Fast semi-empirical energies and gradients
/// - XYZ input format
/// - Parsing from .engrad files
///
/// # Examples
///
/// ```
/// use omecp::qm_interface::XtbInterface;
///
/// // Create interface with XTB
/// let xtb = XtbInterface::new("xtb".to_string());
/// ```
pub struct XtbInterface {
    /// XTB executable command (e.g., "xtb", "/path/to/xtb")
    pub command: String,
}

impl BagelInterface {
    /// Creates a new BAGEL interface.
    ///
    /// # Arguments
    ///
    /// * `command` - BAGEL executable command (e.g., "bagel", "/path/to/bagel")
    /// * `model_template` - JSON string for BAGEL input, with "geometry" placeholder
    ///
    /// # Examples
    ///
    /// ```
    /// use omecp::qm_interface::BagelInterface;
    ///
    /// let bagel = BagelInterface::new(
    ///     "bagel".to_string(),
    ///     "{\"job\":{\"method\":\"caspt2\"}}".to_string(),
    /// );
    /// ```
    pub fn new(command: String, model_template: String) -> Self {
        Self {
            command,
            model_template,
        }
    }
}

impl QMInterface for BagelInterface {
    fn write_input(&self, geom: &Geometry, _header: &str, _tail: &str, path: &Path) -> Result<()> {
        let mut content = String::new();

        // Read model template and replace placeholders
        for line in self.model_template.lines() {
            if line.contains("geometry") {
                content.push_str(&geometry_to_json(geom));
            } else {
                content.push_str(line);
                content.push('\n');
            }
        }

        fs::write(path, content)?;
        Ok(())
    }

    fn run_calculation(&self, input_path: &Path) -> Result<()> {
        let output = Command::new(&self.command).arg(input_path).output()?;

        if !output.status.success() {
            return Err(QMError::Calculation(
                String::from_utf8_lossy(&output.stderr).to_string(),
            ));
        }
        Ok(())
    }

    fn read_output(&self, output_path: &Path, state: usize) -> Result<State> {
        // Check if output file exists before attempting to read
        if !output_path.exists() {
            return Err(QMError::Parse(format!(
                "BAGEL output file not found: {}. Check that BAGEL calculation completed successfully.",
                output_path.display()
            )));
        }

        let content = fs::read_to_string(output_path)
            .map_err(|e| QMError::Parse(format!(
                "Failed to read BAGEL output file {}: {}",
                output_path.display(), e
            )))?;

        let mut energy = 0.0;
        let mut forces = Vec::new();
        let mut geom_coords = Vec::new();
        let mut elements = Vec::new();

        let mut in_geom = false;
        let mut in_forces = false;
        let mut in_energy = false;

        for line in content.lines() {
            if line.contains("*** Geometry ***") {
                in_geom = true;
                geom_coords.clear();
                elements.clear();
            } else if line.contains("Number of auxiliary basis functions") {
                in_geom = false;
            } else if line.contains("Nuclear energy gradient") {
                in_forces = true;
            } else if line.contains("* Gradient computed with") {
                in_forces = false;
            } else if line.contains("=== FCI iteration ===") {
                in_energy = true;
            } else if in_energy
                && line
                    .trim()
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_digit())
            {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 4 && parts[1].parse::<usize>().ok() == Some(state) {
                    energy = parts[parts.len() - 3].parse().unwrap_or(0.0);
                }
            } else if line.contains("MS-CASPT2 energy : state") {
                let state_str = line.split("state").nth(1).unwrap_or("");
                let parts: Vec<&str> = state_str.split_whitespace().collect();
                if parts.len() >= 2 && parts[0].parse::<usize>().ok() == Some(state) {
                    energy = parts[1].parse().unwrap_or(0.0);
                }
            } else if in_geom && line.contains("{ \"atom\" :") {
                // Parse: { "atom" : "C", "xyz" : [ 1.0, 2.0, 3.0 ] }
                if let Some(atom_part) = line.split("\"atom\"").nth(1) {
                    if let Some(elem) = atom_part.split('"').nth(2) {
                        elements.push(elem.to_string());
                    }
                }
                if let Some(xyz_part) = line.split('[').nth(1) {
                    if let Some(coords_str) = xyz_part.split(']').next() {
                        for coord in coords_str.split(',') {
                            if let Ok(val) = coord.trim().parse::<f64>() {
                                geom_coords.push(val * 0.52918); // Bohr to Angstrom
                            }
                        }
                    }
                }
            } else if in_forces && line.trim().starts_with(['x', 'y', 'z']) {
                if let Some(val) = line.split_whitespace().last() {
                    forces.push(-val.parse::<f64>().unwrap_or(0.0)); // Gradient to force
                }
            }
        }

        // Validate that we successfully parsed required data
        if forces.is_empty() {
            return Err(QMError::Parse(format!(
                "Failed to parse forces from BAGEL output file: {}. Check that gradient calculation completed successfully.",
                output_path.display()
            )));
        }

        if geom_coords.is_empty() {
            return Err(QMError::Parse(format!(
                "Failed to parse geometry from BAGEL output file: {}. Check that the file contains valid atomic coordinates.",
                output_path.display()
            )));
        }

        // Validate that we got meaningful energy (prevent zero energy failures)
        if energy == 0.0 {
            return Err(QMError::Parse(format!(
                "Failed to extract energy from BAGEL output file: {}. Check that the calculation completed successfully and contains valid energy data for state {}.",
                output_path.display(), state
            )));
        }

        // Validate force/geometry consistency
        let n_atoms = elements.len();
        if forces.len() != 3 * n_atoms {
            return Err(QMError::Parse(format!(
                "Force/geometry mismatch in BAGEL output: expected {} force components for {} atoms, got {}",
                3 * n_atoms, n_atoms, forces.len()
            )));
        }

        let state = State {
            energy,
            forces: DVector::from_vec(forces),
            geometry: Geometry::new(elements, geom_coords),
        };

        // Validate the state to ensure meaningful data
        state.validate().map_err(|e| QMError::Parse(format!(
            "BAGEL state validation failed: {}. Check that the calculation completed successfully and produced valid energy/gradient data.",
            e
        )))?;

        Ok(state)
    }
}

impl XtbInterface {
    /// Creates a new XTB interface.
    ///
    /// # Arguments
    ///
    /// * `command` - XTB executable command (e.g., "xtb", "/path/to/xtb")
    ///
    /// # Examples
    ///
    /// ```
    /// use omecp::qm_interface::XtbInterface;
    ///
    /// let xtb = XtbInterface::new("xtb".to_string());
    /// ```
    pub fn new(command: String) -> Self {
        Self { command }
    }
}

impl QMInterface for XtbInterface {
    fn write_input(&self, geom: &Geometry, _header: &str, _tail: &str, path: &Path) -> Result<()> {
        // xTB uses XYZ format
        io::write_xyz(geom, path)?;
        Ok(())
    }

    fn run_calculation(&self, input_path: &Path) -> Result<()> {
        let output = Command::new(&self.command)
            .arg(input_path)
            .arg("--engrad")
            .output()?;

        if !output.status.success() {
            return Err(QMError::Calculation(
                String::from_utf8_lossy(&output.stderr).to_string(),
            ));
        }
        Ok(())
    }

    fn read_output(&self, output_path: &Path, _state: usize) -> Result<State> {
        // xTB outputs energy and gradients to .engrad file
        let engrad_path = output_path.with_extension("engrad");
        
        // Check if required files exist before attempting to read
        if !engrad_path.exists() {
            return Err(QMError::Parse(format!(
                "XTB engrad file not found: {}. Check that XTB calculation completed successfully and produced gradient output.",
                engrad_path.display()
            )));
        }

        let content = fs::read_to_string(&engrad_path)
            .map_err(|e| QMError::Parse(format!(
                "Failed to read XTB engrad file {}: {}",
                engrad_path.display(), e
            )))?;

        let mut energy = 0.0;
        let mut forces = Vec::new();
        let mut geom_coords = Vec::new();
        let mut elements = Vec::new();

        let mut in_energy = false;
        let mut in_geom = false;
        let mut in_forces = false;

        for line in content.lines() {
            let line = line.trim();
            if line.starts_with("$energy") {
                in_energy = true;
                continue;
            } else if line.starts_with("$gradients") {
                in_forces = true;
                in_energy = false;
                continue;
            } else if line.starts_with("$geometry") {
                in_geom = true;
                in_forces = false;
                continue;
            } else if line.starts_with("$") {
                in_energy = false;
                in_geom = false;
                in_forces = false;
                continue;
            }

            if in_energy && !line.is_empty() {
                energy = line.parse().unwrap_or(0.0);
            } else if in_geom && !line.is_empty() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 4 {
                    elements.push(parts[0].to_string());
                    geom_coords.push(parts[1].parse().unwrap_or(0.0));
                    geom_coords.push(parts[2].parse().unwrap_or(0.0));
                    geom_coords.push(parts[3].parse().unwrap_or(0.0));
                }
            } else if in_forces && !line.is_empty() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 {
                    forces.push(parts[0].parse().unwrap_or(0.0));
                    forces.push(parts[1].parse().unwrap_or(0.0));
                    forces.push(parts[2].parse().unwrap_or(0.0));
                }
            }
        }

        // Validate that we successfully parsed required data
        if forces.is_empty() {
            return Err(QMError::Parse(format!(
                "Failed to parse forces from XTB engrad file: {}. Check that gradient calculation completed successfully.",
                engrad_path.display()
            )));
        }

        if geom_coords.is_empty() {
            return Err(QMError::Parse(format!(
                "Failed to parse geometry from XTB engrad file: {}. Check that the file contains valid atomic coordinates.",
                engrad_path.display()
            )));
        }

        // Validate that we got meaningful energy (prevent zero energy failures)
        if energy == 0.0 {
            return Err(QMError::Parse(format!(
                "Failed to extract energy from XTB engrad file: {}. Check that the calculation completed successfully and contains valid energy data.",
                engrad_path.display()
            )));
        }

        // Validate force/geometry consistency
        let n_atoms = elements.len();
        if forces.len() != 3 * n_atoms {
            return Err(QMError::Parse(format!(
                "Force/geometry mismatch in XTB output: expected {} force components for {} atoms, got {}",
                3 * n_atoms, n_atoms, forces.len()
            )));
        }

        let state = State {
            energy,
            forces: DVector::from_vec(forces),
            geometry: Geometry::new(elements, geom_coords),
        };

        // Validate the state to ensure meaningful data
        state.validate().map_err(|e| QMError::Parse(format!(
            "XTB state validation failed: {}. Check that the calculation completed successfully and produced valid energy/gradient data.",
            e
        )))?;

        Ok(state)
    }
}

fn geometry_to_json(geom: &Geometry) -> String {
    let mut result = String::from("\"geometry\" : [\n");

    for i in 0..geom.num_atoms {
        let coords = geom.get_atom_coords(i);
        result.push_str(&format!(
            "{{ \"atom\" : \"{}\", \"xyz\" : [ {:.8}, {:.8}, {:.8} ] }}",
            geom.elements[i], coords[0], coords[1], coords[2]
        ));

        if i < geom.num_atoms - 1 {
            result.push(',');
        }
        result.push('\n');
    }

    result.push_str("]\n");
    result
}

/// Configuration for custom QM interfaces
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CustomInterfaceConfig {
    /// Name of the QM program
    pub name: String,
    /// Command to run the program
    pub command: String,
    /// Input file template (supports placeholders: {geometry}, {header}, {tail})
    pub input_template: String,
    /// Output file extension (e.g., "log", "out")
    pub output_extension: String,
    /// Energy parsing configuration
    pub energy_parser: EnergyParser,
    /// Forces parsing configuration (optional)
    pub forces_parser: Option<ForcesParser>,
}

/// Configuration for parsing energy from custom QM program output.
///
/// Specifies the regular expression pattern to locate the energy value
/// in the output file and a unit conversion factor to convert it to Hartree.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EnergyParser {
    /// Regex pattern to find energy (should contain a capture group for the value)
    pub pattern: String,
    /// Unit conversion factor (multiply by this to get hartree)
    pub unit_factor: f64,
}

/// Configuration for parsing forces from custom QM program output.
///
/// Specifies the regular expression pattern to locate the force components
/// (Fx, Fy, Fz) for each atom in the output file.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ForcesParser {
    /// Regex pattern to find forces (should capture Fx, Fy, Fz for each atom)
    pub pattern: String,
}

/// Custom QM interface that reads configuration from JSON
pub struct CustomInterface {
    config: CustomInterfaceConfig,
    energy_regex: Regex,
    forces_regex: Option<Regex>,
}

impl CustomInterface {
    /// Creates a new `CustomInterface` by loading configuration from a JSON file.
    ///
    /// This function reads a JSON configuration file that defines how to interact
    /// with a custom quantum chemistry program. The configuration includes:
    /// - Program command
    /// - Input file template with placeholders
    /// - Output file extension
    /// - Regular expressions for parsing energy and forces
    ///
    /// # Arguments
    ///
    /// * `config_path` - Path to the JSON configuration file
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing:
    /// - `Ok(Self)` - Successfully created `CustomInterface`
    /// - `Err(QMError::Parse)` - Failed to read or parse the configuration file
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use omecp::qm_interface::CustomInterface;
    /// use std::path::Path;
    ///
    /// let custom_interface = CustomInterface::from_file(Path::new("my_qm_config.json"))?;
    /// ```
    pub fn from_file(config_path: &Path) -> Result<Self> {
        let content = fs::read_to_string(config_path).map_err(|e| {
            QMError::Parse(format!("Failed to read custom interface config: {}", e))
        })?;

        let config: CustomInterfaceConfig = serde_json::from_str(&content).map_err(|e| {
            QMError::Parse(format!("Failed to parse custom interface config: {}", e))
        })?;

        // Compile regexes
        let energy_regex = Regex::new(&config.energy_parser.pattern)
            .map_err(|e| QMError::Parse(format!("Invalid energy regex: {}", e)))?;

        let forces_regex = if let Some(ref forces_parser) = config.forces_parser {
            Some(
                Regex::new(&forces_parser.pattern)
                    .map_err(|e| QMError::Parse(format!("Invalid forces regex: {}", e)))?,
            )
        } else {
            None
        };

        Ok(Self {
            config,
            energy_regex,
            forces_regex,
        })
    }
}

impl QMInterface for CustomInterface {
    fn write_input(&self, geom: &Geometry, header: &str, tail: &str, path: &Path) -> Result<()> {
        // Generate geometry string in XYZ format
        let mut geometry_lines = Vec::new();
        for i in 0..geom.num_atoms {
            let coords = geom.get_atom_coords(i);
            geometry_lines.push(format!(
                "{:>2} {:>12.8} {:>12.8} {:>12.8}",
                geom.elements[i], coords[0], coords[1], coords[2]
            ));
        }
        let geometry_str = geometry_lines.join("\n");

        // Replace placeholders in template
        let input_content = self
            .config
            .input_template
            .replace("{geometry}", &geometry_str)
            .replace("{header}", header)
            .replace("{tail}", tail);

        fs::write(path, input_content)?;
        Ok(())
    }

    fn run_calculation(&self, input_path: &Path) -> Result<()> {
        let output = Command::new(&self.config.command)
            .arg(input_path)
            .output()?;

        if !output.status.success() {
            return Err(QMError::Calculation(
                String::from_utf8_lossy(&output.stderr).to_string(),
            ));
        }
        Ok(())
    }

    fn read_output(&self, output_path: &Path, _state: usize) -> Result<State> {
        // Check if output file exists before attempting to read
        if !output_path.exists() {
            return Err(QMError::Parse(format!(
                "Custom QM program ({}) output file not found: {}. Check that {} calculation completed successfully.",
                self.config.name, output_path.display(), self.config.name
            )));
        }

        let content = fs::read_to_string(output_path)
            .map_err(|e| QMError::Parse(format!(
                "Failed to read {} output file {}: {}",
                self.config.name, output_path.display(), e
            )))?;

        // Parse energy
        let energy = if let Some(caps) = self.energy_regex.captures(&content) {
            if let Some(energy_match) = caps.get(1) {
                let energy_val: f64 = energy_match
                    .as_str()
                    .parse()
                    .map_err(|_| QMError::Parse(format!(
                        "Failed to parse energy value '{}' from {} output file: {}",
                        energy_match.as_str(), self.config.name, output_path.display()
                    )))?;
                energy_val * self.config.energy_parser.unit_factor
            } else {
                return Err(QMError::Parse(format!(
                    "Energy regex pattern '{}' must have a capture group for {} output file: {}",
                    self.config.energy_parser.pattern, self.config.name, output_path.display()
                )));
            }
        } else {
            return Err(QMError::Parse(format!(
                "Energy pattern '{}' not found in {} output file: {}. Check that the calculation completed successfully and the pattern matches the output format.",
                self.config.energy_parser.pattern, self.config.name, output_path.display()
            )));
        };

        // Parse forces if available
        let forces = if let Some(ref forces_regex) = self.forces_regex {
            let mut forces_vec = Vec::new();
            for caps in forces_regex.captures_iter(&content) {
                if caps.len() >= 4 {
                    let fx: f64 = caps[1].parse().map_err(|_| QMError::Parse(format!(
                        "Failed to parse force component Fx '{}' from {} output file: {}",
                        &caps[1], self.config.name, output_path.display()
                    )))?;
                    let fy: f64 = caps[2].parse().map_err(|_| QMError::Parse(format!(
                        "Failed to parse force component Fy '{}' from {} output file: {}",
                        &caps[2], self.config.name, output_path.display()
                    )))?;
                    let fz: f64 = caps[3].parse().map_err(|_| QMError::Parse(format!(
                        "Failed to parse force component Fz '{}' from {} output file: {}",
                        &caps[3], self.config.name, output_path.display()
                    )))?;
                    forces_vec.push(fx);
                    forces_vec.push(fy);
                    forces_vec.push(fz);
                }
            }
            if forces_vec.is_empty() {
                return Err(QMError::Parse(format!(
                    "No forces found using pattern '{}' in {} output file: {}. Check that gradient calculation completed successfully and the pattern matches the output format.",
                    self.forces_regex.as_ref().unwrap().as_str(), self.config.name, output_path.display()
                )));
            }
            DVector::from_vec(forces_vec)
        } else {
            // No forces available - return zero forces
            DVector::zeros(3)
        };

        // Validate that we got meaningful energy (prevent zero energy failures)
        if energy == 0.0 {
            return Err(QMError::Parse(format!(
                "Extracted zero energy from {} output file: {}. Check that the calculation completed successfully and the energy pattern is correct.",
                self.config.name, output_path.display()
            )));
        }

        // Return a simple geometry (could be enhanced)
        let geometry = Geometry::new(vec!["H".to_string()], vec![0.0, 0.0, 0.0]);

        let state = State {
            energy,
            forces,
            geometry,
        };

        // Validate the state to ensure meaningful data
        state.validate().map_err(|e| QMError::Parse(format!(
            "Custom interface ({}) state validation failed: {}. Check that the calculation completed successfully and produced valid energy/gradient data.",
            self.config.name, e
        )))?;

        Ok(state)
    }
}
/// Returns the output file extension for the given QM program.
///
/// This function maps each QM program to its expected output file extension:
/// - Gaussian: `.log`
/// - ORCA: `.out`
/// - XTB: `.out`
/// - BAGEL: `.json`
/// - Custom: `.log`
///
/// # Arguments
///
/// * `program` - The QM program type
///
/// # Returns
///
/// The file extension (without the dot) as a string slice
///
/// # Examples
///
/// ```
/// use omecp::config::QMProgram;
/// use omecp::qm_interface::get_output_file_base;
///
/// assert_eq!(get_output_file_base(QMProgram::Gaussian), "log");
/// assert_eq!(get_output_file_base(QMProgram::Orca), "out");
/// ```
pub fn get_output_file_base(program: crate::config::QMProgram) -> &'static str {
    match program {
        crate::config::QMProgram::Gaussian => "log",
        crate::config::QMProgram::Orca => "out",
        crate::config::QMProgram::Xtb => "out",
        crate::config::QMProgram::Bagel => "json",
        crate::config::QMProgram::Custom => "log",
    }
}