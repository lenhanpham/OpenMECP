//! Input file parsing for OpenMECP configuration.
//!
//! This module provides parsing functionality for OpenMECP input files, which
//! use a custom section-based format. The parser handles:
//!
//! - **Geometries**: Main geometry and LST interpolation geometries
//! - **Configuration**: All calculation parameters (method, program, etc.)
//! - **Constraints**: Bond lengths, angles, and fixed atoms
//! - **Tail sections**: Additional keywords for each state
//!
//! # Input File Format
//!
//! OpenMECP input files use a section-based syntax marked with `*SECTION` markers
//! and terminated with `*`. The format is designed to be human-readable and
//! flexible for different types of calculations.
//!
//! ## Required Sections
//!
//! ### *GEOM Section
//!
//! Contains the initial molecular geometry in Cartesian coordinates:
//!
//! ```text
//! *GEOM
//! C  0.0  0.0  0.0
//! H  1.0  0.0  0.0
//! H  0.0  1.0  0.0
//! *
//! ```
//!
//! Can also reference external files:
//!
//! ```text
//! *GEOM
//! @geometry.xyz
//! *
//! ```
//!
//! ### Key-Value Parameters
//!
//! After sections, specify calculation parameters:
//!
//! ```text
//! program = gaussian
//! method = B3LYP/6-31G*
//! nprocs = 4
//! mem = 4GB
//! charge = 0
//! mult1 = 1
//! mult2 = 3
//! ```
//!
//! ## Optional Sections
//!
//! - `*TAIL1` / `*TAIL2`: Additional keywords for each electronic state
//! - `*CONSTR`: Geometric constraints (bonds, angles, scans)
//! - `*LST1` / `*LST2`: Geometries for linear synchronous transit interpolation
//!
//! # Examples
//!
//! ```
//! use omecp::parser::{parse_input, InputData};
//! use std::path::Path;
//!
//! // Parse an input file
//! let input_path = Path::new("input.inp");
//! let input_data: InputData = parse_input(input_path)?;
//!
//! // Access parsed data
//! let config = input_data.config;
//! let geometry = input_data.geometry;
//! let constraints = input_data.constraints;
//! ```

use crate::config::{Config, QMProgram, RunMode, ScanSpec, ScanType};
use crate::constraints::Constraint;
use crate::geometry::Geometry;
use regex::Regex;
use std::fs;
use std::path::Path;
use thiserror::Error;

/// Error type for parsing operations.
///
/// Parsing can fail due to:
/// - File I/O errors (missing files, permission issues)
/// - Format errors (malformed input, invalid values)
/// - Unsupported file formats
#[derive(Error, Debug)]
pub enum ParseError {
    /// I/O error when reading files
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// Parse error with descriptive message
    #[error("Parse error: {0}")]
    Parse(String),
}

/// Type alias for parse operation results
type Result<T> = std::result::Result<T, ParseError>;

/// Complete parsed input data from an OpenMECP input file.
///
/// This struct contains all information parsed from an input file, organized
/// into logical components:
///
/// # Examples
///
/// ```
/// use omecp::parser::InputData;
/// use omecp::config::Config;
/// use omecp::geometry::Geometry;
/// use omecp::constraints::Constraint;
///
/// // Access parsed components
/// let input_data: InputData = /* ... parsed from file ... */;
/// let config = &input_data.config;
/// let geometry = &input_data.geometry;
/// let constraints = &input_data.constraints;
/// let tail1 = &input_data.tail1;
/// let tail2 = &input_data.tail2;
/// ```
pub struct InputData {
    /// Complete calculation configuration
    pub config: Config,
    /// Initial molecular geometry
    pub geometry: Geometry,
    /// List of geometric constraints
    pub constraints: Vec<Constraint>,
    /// Additional keywords for electronic state 1
    pub tail1: String,
    /// Additional keywords for electronic state 2
    pub tail2: String,
    /// List of fixed atom indices (0-based)
    pub fixed_atoms: Vec<usize>,
    /// Optional geometry for LST interpolation (first endpoint)
    pub lst1: Option<Geometry>,
    /// Optional geometry for LST interpolation (second endpoint)
    pub lst2: Option<Geometry>,
}

/// Parse an OpenMECP input file.
///
/// This function reads and parses a complete OpenMECP input file, extracting
/// all configuration parameters, geometries, constraints, and additional
/// keywords. It supports the custom section-based input format with support
/// for external geometry files.
///
/// # Arguments
///
/// * `path` - Path to the input file (e.g., "input.inp")
///
/// # Returns
///
/// Returns `Ok(InputData)` on successful parsing, or `Err(ParseError)` if:
/// - The file cannot be read (I/O error)
/// - The file format is invalid
/// - Required sections are missing
/// - External geometry files cannot be read
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use omecp::parser::parse_input;
/// use std::path::Path;
///
/// let input_path = Path::new("input.inp");
/// match parse_input(input_path) {
///     Ok(input_data) => {
///         println!("Parsed {} atoms", input_data.geometry.num_atoms);
///         println!("QM program: {:?}", input_data.config.program);
///     }
///     Err(e) => eprintln!("Parse error: {}", e),
/// }
/// ```
///
/// Parsing with external geometry file:
///
/// ```text
/// // input.inp
/// *GEOM
/// @molecule.xyz
/// *
///
/// program = gaussian
/// method = B3LYP/6-31G*
/// // ... other parameters ...
/// ```
///
/// ```
/// use omecp::parser::parse_input;
/// use std::path::Path;
///
/// let input_data = parse_input(Path::new("input.inp"))?;
/// // Geometry will be read from molecule.xyz
/// ```
pub fn parse_input(path: &Path) -> Result<InputData> {
    let content = fs::read_to_string(path)?;
    let mut config = Config::default();
    let mut elements = Vec::new();
    let mut coords = Vec::new();
    let mut constraints = Vec::new();
    let mut tail1 = String::new();
    let mut tail2 = String::new();
    let mut fixed_atoms = Vec::new();
    let mut lst1_elements = Vec::new();
    let mut lst1_coords = Vec::new();
    let mut lst2_elements = Vec::new();
    let mut lst2_coords = Vec::new();
    let mut oniom_layer_info = Vec::new();
    
    let mut in_geom = false;
    let mut in_tail1 = false;
    let mut in_tail2 = false;
    let mut in_constr = false;
    let mut in_lst1 = false;
    let mut in_lst2 = false;
    
    let geom_re = Regex::new(r"^\s*(\S+)\s+(-?\d+\.?\d*)").unwrap();
    
    for line in content.lines() {
        let line_lower = line.to_lowercase();
        let trimmed = line_lower.trim();
        
        if trimmed.starts_with('#') {
            continue;
        }
        
        if trimmed.contains("*geom") {
            in_geom = true;
            continue;
        } else if trimmed.contains("*tail1") {
            in_tail1 = true;
            continue;
        } else if trimmed.contains("*tail2") {
            in_tail2 = true;
            continue;
        } else if trimmed.contains("*constr") {
            in_constr = true;
            continue;
        } else if trimmed.contains("*lst1") {
            in_lst1 = true;
            continue;
        } else if trimmed.contains("*lst2") {
            in_lst2 = true;
            continue;
        } else if trimmed == "*" {
            in_geom = false;
            in_tail1 = false;
            in_tail2 = false;
            in_constr = false;
            in_lst1 = false;
            in_lst2 = false;
            continue;
        }
        
        if in_geom {
            if line.trim().starts_with('@') {
                // External geometry file
                let filename = line.trim().strip_prefix('@').unwrap().trim();
                let external_path = Path::new(filename);
                let (ext_elements, ext_coords) = read_external_geometry(external_path)?;
                elements = ext_elements;
                coords = ext_coords;
            } else if geom_re.is_match(line) {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 4 {
                    elements.push(parts[0].to_string());
                    coords.push(parts[1].parse().map_err(|_| ParseError::Parse("Invalid coordinate".into()))?);
                    coords.push(parts[2].parse().map_err(|_| ParseError::Parse("Invalid coordinate".into()))?);
                    coords.push(parts[3].parse().map_err(|_| ParseError::Parse("Invalid coordinate".into()))?);
                    if parts.len() > 4 {
                        oniom_layer_info.push(parts[4..].join(" "));
                    } else {
                        oniom_layer_info.push(String::new());
                    }
                }
            }
        } else if in_lst1 && geom_re.is_match(line) {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                lst1_elements.push(parts[0].to_string());
                lst1_coords.push(parts[1].parse().map_err(|_| ParseError::Parse("Invalid coordinate".into()))?);
                lst1_coords.push(parts[2].parse().map_err(|_| ParseError::Parse("Invalid coordinate".into()))?);
                lst1_coords.push(parts[3].parse().map_err(|_| ParseError::Parse("Invalid coordinate".into()))?);
            }
        } else if in_lst2 && geom_re.is_match(line) {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                lst2_elements.push(parts[0].to_string());
                lst2_coords.push(parts[1].parse().map_err(|_| ParseError::Parse("Invalid coordinate".into()))?);
                lst2_coords.push(parts[2].parse().map_err(|_| ParseError::Parse("Invalid coordinate".into()))?);
                lst2_coords.push(parts[3].parse().map_err(|_| ParseError::Parse("Invalid coordinate".into()))?);
            }
        } else if in_tail1 {
            tail1.push_str(line);
            tail1.push('\n');
        } else if in_tail2 {
            tail2.push_str(line);
            tail2.push('\n');
        } else if in_constr && !trimmed.is_empty() {
            if trimmed.starts_with('s') {
                parse_scan(line, &mut config)?;
            } else {
                parse_constraint(line, &mut constraints)?;
            }
        } else if trimmed.contains('=') {
            parse_parameter(line, &mut config, &mut fixed_atoms)?;
        }
    }
    
    config.oniom_layer_info = oniom_layer_info;

    let geometry = Geometry::new(elements, coords);
    let lst1 = if !lst1_elements.is_empty() {
        Some(Geometry::new(lst1_elements, lst1_coords))
    } else {
        None
    };
    let lst2 = if !lst2_elements.is_empty() {
        Some(Geometry::new(lst2_elements, lst2_coords))
    } else {
        None
    };

    Ok(InputData {
        config,
        geometry,
        constraints,
        tail1,
        tail2,
        fixed_atoms,
        lst1,
        lst2,
    })
}

fn parse_constraint(line: &str, constraints: &mut Vec<Constraint>) -> Result<()> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.is_empty() {
        return Ok(());
    }
    
    match parts[0].to_lowercase().as_str() {
        "r" if parts.len() >= 4 => {
            let a = parts[1].parse::<usize>().map_err(|_| ParseError::Parse("Invalid atom index".into()))? - 1;
            let b = parts[2].parse::<usize>().map_err(|_| ParseError::Parse("Invalid atom index".into()))? - 1;
            let target = parts[3].parse().map_err(|_| ParseError::Parse("Invalid target value".into()))?;
            constraints.push(Constraint::Bond { atoms: (a, b), target });
        }
        "a" if parts.len() >= 5 => {
            let a = parts[1].parse::<usize>().map_err(|_| ParseError::Parse("Invalid atom index".into()))? - 1;
            let b = parts[2].parse::<usize>().map_err(|_| ParseError::Parse("Invalid atom index".into()))? - 1;
            let c = parts[3].parse::<usize>().map_err(|_| ParseError::Parse("Invalid atom index".into()))? - 1;
            let target = parts[4].parse::<f64>().map_err(|_| ParseError::Parse("Invalid target value".into()))?.to_radians();
            constraints.push(Constraint::Angle { atoms: (a, b, c), target });
        }
        _ => {}
    }
    Ok(())
}

fn parse_scan(line: &str, config: &mut Config) -> Result<()> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 6 {
        return Ok(());
    }
    
    let scan_type = match parts[1].to_lowercase().as_str() {
        "r" if parts.len() >= 7 => {
            let a = parts[2].parse::<usize>().map_err(|_| ParseError::Parse("Invalid atom".into()))? - 1;
            let b = parts[3].parse::<usize>().map_err(|_| ParseError::Parse("Invalid atom".into()))? - 1;
            ScanType::Bond { atoms: (a, b) }
        }
        "a" if parts.len() >= 8 => {
            let a = parts[2].parse::<usize>().map_err(|_| ParseError::Parse("Invalid atom".into()))? - 1;
            let b = parts[3].parse::<usize>().map_err(|_| ParseError::Parse("Invalid atom".into()))? - 1;
            let c = parts[4].parse::<usize>().map_err(|_| ParseError::Parse("Invalid atom".into()))? - 1;
            ScanType::Angle { atoms: (a, b, c) }
        }
        _ => return Ok(()),
    };
    
    let offset = if matches!(scan_type, ScanType::Bond { .. }) { 4 } else { 5 };
    let start = parts[offset].parse().map_err(|_| ParseError::Parse("Invalid start".into()))?;
    let num_points = parts[offset + 1].parse().map_err(|_| ParseError::Parse("Invalid num".into()))?;
    let step_size = parts[offset + 2].parse().map_err(|_| ParseError::Parse("Invalid step".into()))?;
    
    config.scans.push(ScanSpec { scan_type, start, num_points, step_size });
    Ok(())
}

fn parse_parameter(line: &str, config: &mut Config, fixed_atoms: &mut Vec<usize>) -> Result<()> {
    let parts: Vec<&str> = line.splitn(2, '=').collect();
    if parts.len() != 2 {
        return Ok(());
    }
    
    let key = parts[0].trim().to_lowercase();
    let value = parts[1].trim();
    
    match key.as_str() {
        "nprocs" => config.nprocs = value.parse().unwrap_or(1),
        "mem" => config.mem = value.to_string(),
        "charge" => config.charge1 = value.parse().unwrap_or(0),
        "charge2" => config.charge2 = value.parse().unwrap_or(config.charge1),
        "mult1" => config.mult1 = value.parse().unwrap_or(1),
        "mult2" => config.mult2 = value.parse().unwrap_or(1),
        "method" => config.method = value.to_string(),
        "program" => {
            config.program = match value.to_lowercase().as_str() {
                "gaussian" => QMProgram::Gaussian,
                "orca" => QMProgram::Orca,
                "bagel" => QMProgram::Bagel,
                "xtb" => QMProgram::Xtb,
                _ => QMProgram::Gaussian,
            };
        }
        "mode" => {
            config.run_mode = match value.to_lowercase().as_str() {
                "read" => RunMode::Read,
                "noread" => RunMode::NoRead,
                "stable" => RunMode::Stable,
                "inter_read" => RunMode::InterRead,
                _ => RunMode::Normal,
            };
        }
        "td1" => config.td1 = value.to_string(),
        "td2" => config.td2 = value.to_string(),
        "mp2" => config.mp2 = value.to_lowercase() == "true",
        "max_steps" => config.max_steps = value.parse().unwrap_or(100),
        "max_step_size" => config.max_step_size = value.parse().unwrap_or(0.1),
        "fix_de" => config.fix_de = value.parse().unwrap_or(0.0),
        "de_thresh" => config.thresholds.de = value.parse().unwrap_or(0.000050),
        "rms_thresh" => config.thresholds.rms = value.parse().unwrap_or(0.0025),
        "max_dis_thresh" => config.thresholds.max_dis = value.parse().unwrap_or(0.004),
        "max_g_thresh" => config.thresholds.max_g = value.parse().unwrap_or(0.0007),
        "rms_g_thresh" => config.thresholds.rms_g = value.parse().unwrap_or(0.0005),
        "bagel_model" => config.bagel_model = value.to_string(),
        "custom_interface_file" => config.custom_interface_file = value.to_string(),
        "drive_coordinate" => config.drive_coordinate = value.to_string(),
        "drive_start" => config.drive_start = value.parse().unwrap_or(0.0),
        "drive_end" => config.drive_end = value.parse().unwrap_or(0.0),
        "drive_steps" => config.drive_steps = value.parse().unwrap_or(10),
        "drive_type" => config.drive_type = value.to_string(),
        "use_gediis" => config.use_gediis = value.to_lowercase() == "true",
        "drive_atoms" => {
            config.drive_atoms = value.split(',')
                .filter_map(|s| s.trim().parse::<usize>().ok())
                .map(|x| x.saturating_sub(1)) // Convert to 0-based indexing
                .collect();
        },
        "fixedatoms" => {
            for group in value.split(',') {
                if group.contains('-') {
                    let range: Vec<&str> = group.split('-').collect();
                    if range.len() == 2 {
                        if let (Ok(start), Ok(end)) = (range[0].parse::<usize>(), range[1].parse::<usize>()) {
                            for i in start..=end {
                                fixed_atoms.push(i - 1);
                            }
                        }
                    }
                } else if let Ok(atom) = group.parse::<usize>() {
                    fixed_atoms.push(atom - 1);
                }
            }
        }
        "gau_comm" => {
            config.program_commands.insert("gaussian".to_string(), value.to_string());
        }
        "orca_comm" => {
            config.program_commands.insert("orca".to_string(), value.to_string());
        }
        "xtb_comm" => {
            config.program_commands.insert("xtb".to_string(), value.to_string());
        }
        "bagel_comm" => {
            config.program_commands.insert("bagel".to_string(), value.to_string());
        }
        "isoniom" => config.is_oniom = value.to_lowercase() == "true",
        "chargeandmultforoniom1" => config.charge_and_mult_oniom1 = value.to_string(),
        "chargeandmultforoniom2" => config.charge_and_mult_oniom2 = value.to_string(),
        "basis" => config.basis_set = value.to_string(),
        "solvent" => config.solvent = value.to_string(),
        "dispersion" => config.dispersion = value.to_string(),
        "checkpoint" => config.checkpoint_file = value.to_string(),
        "restart" => config.restart = value.to_lowercase() == "true",
        _ => {}
    }
    Ok(())
}

fn read_geom_from_xyz(path: &Path) -> Result<(Vec<String>, Vec<f64>)> {
    let content = fs::read_to_string(path)?;
    let mut elements = Vec::new();
    let mut coords = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || !line.chars().next().is_some_and(|c| c.is_alphabetic()) {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 4 {
            elements.push(parts[0].to_string());
            for &coord_str in &parts[1..4] {
                coords.push(coord_str.parse().map_err(|_| ParseError::Parse("Invalid coordinate in XYZ file".into()))?);
            }
        }
    }
    Ok((elements, coords))
}

fn read_geom_from_gjf(path: &Path) -> Result<(Vec<String>, Vec<f64>)> {
    let content = fs::read_to_string(path)?;
    let mut elements = Vec::new();
    let mut coords = Vec::new();
    let mut in_geom = false;

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line.contains("0 1") || line.contains("0 2") || line.contains("0 3") {
            in_geom = true;
            continue;
        }
        if in_geom && line.chars().next().is_some_and(|c| c.is_alphabetic()) {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                elements.push(parts[0].to_string());
                for &coord_str in &parts[1..4] {
                    coords.push(coord_str.parse().map_err(|_| ParseError::Parse("Invalid coordinate in GJF file".into()))?);
                }
            }
        } else if in_geom && line.chars().next().is_some_and(|c| c.is_ascii_digit()) {
            break;
        }
    }
    Ok((elements, coords))
}

fn read_geom_from_log(path: &Path) -> Result<(Vec<String>, Vec<f64>)> {
    let content = fs::read_to_string(path)?;
    let mut elements = Vec::new();
    let mut coords = Vec::new();
    let mut in_geom = false;

    for line in content.lines() {
        if line.contains("Input orientation") {
            in_geom = true;
            elements.clear();
            coords.clear();
            continue;
        } else if line.contains("Distance matrix") || line.contains("Rotational constants") {
            in_geom = false;
            continue;
        }
        if in_geom {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 6 && parts[0].chars().all(|c| c.is_ascii_digit()) {
                // Atomic number, atomic number, 0, x, y, z
                elements.push(parts[1].to_string());
                for &coord_str in &parts[3..6] {
                    coords.push(coord_str.parse().map_err(|_| ParseError::Parse("Invalid coordinate in LOG file".into()))?);
                }
            }
        }
    }
    Ok((elements, coords))
}

fn read_external_geometry(path: &Path) -> Result<(Vec<String>, Vec<f64>)> {
    let path_str = path.to_string_lossy();
    if path_str.ends_with(".xyz") {
        read_geom_from_xyz(path)
    } else if path_str.ends_with(".gjf") {
        read_geom_from_gjf(path)
    } else if path_str.ends_with(".log") {
        read_geom_from_log(path)
    } else {
        Err(ParseError::Parse("Unsupported external geometry file format".into()))
    }
}
