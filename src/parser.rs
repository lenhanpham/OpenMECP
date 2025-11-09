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
use crate::geometry::{angstrom_to_bohr, Geometry};
use nalgebra::DVector;
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
/// - Invalid tail section content
/// - Gaussian syntax errors
#[derive(Error, Debug)]
pub enum ParseError {
    /// I/O error when reading files
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// Parse error with descriptive message
    #[error("Parse error: {0}")]
    Parse(String),
    /// Invalid tail section content with detailed context
    #[error("Invalid tail section '{section}'{}: {message}", line_number.map(|n| format!(" at line {}", n)).unwrap_or_default())]
    InvalidTailSection {
        /// Name of the tail section (e.g., "TAIL1", "TAIL2")
        section: String,
        /// Descriptive error message
        message: String,
        /// Optional line number where the error occurred
        line_number: Option<usize>,
    },
    /// Gaussian syntax error with detailed context
    #[error("Gaussian syntax error in {section}{}: {details}", line_number.map(|n| format!(" at line {}", n)).unwrap_or_default())]
    GaussianSyntaxError {
        /// Name of the section where the error occurred
        section: String,
        /// Detailed error information
        details: String,
        /// Optional line number where the error occurred
        line_number: Option<usize>,
    },
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
                    coords.push(
                        parts[1]
                            .parse()
                            .map_err(|_| ParseError::Parse("Invalid coordinate".into()))?,
                    );
                    coords.push(
                        parts[2]
                            .parse()
                            .map_err(|_| ParseError::Parse("Invalid coordinate".into()))?,
                    );
                    coords.push(
                        parts[3]
                            .parse()
                            .map_err(|_| ParseError::Parse("Invalid coordinate".into()))?,
                    );
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
                lst1_coords.push(
                    parts[1]
                        .parse()
                        .map_err(|_| ParseError::Parse("Invalid coordinate".into()))?,
                );
                lst1_coords.push(
                    parts[2]
                        .parse()
                        .map_err(|_| ParseError::Parse("Invalid coordinate".into()))?,
                );
                lst1_coords.push(
                    parts[3]
                        .parse()
                        .map_err(|_| ParseError::Parse("Invalid coordinate".into()))?,
                );
            }
        } else if in_lst2 && geom_re.is_match(line) {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                lst2_elements.push(parts[0].to_string());
                lst2_coords.push(
                    parts[1]
                        .parse()
                        .map_err(|_| ParseError::Parse("Invalid coordinate".into()))?,
                );
                lst2_coords.push(
                    parts[2]
                        .parse()
                        .map_err(|_| ParseError::Parse("Invalid coordinate".into()))?,
                );
                lst2_coords.push(
                    parts[3]
                        .parse()
                        .map_err(|_| ParseError::Parse("Invalid coordinate".into()))?,
                );
            }
        } else if in_tail1 {
            if !is_comment_line(line) && !line.trim().is_empty() {
                if !tail1.is_empty() {
                    tail1.push(' '); // Add space separator between keywords
                }
                tail1.push_str(line.trim());
            }
        } else if in_tail2 {
            if !is_comment_line(line) && !line.trim().is_empty() {
                if !tail2.is_empty() {
                    tail2.push(' '); // Add space separator between keywords
                }
                tail2.push_str(line.trim());
            }
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

    // Validate tail sections after parsing with enhanced error reporting
    // Provide context about filtering if sections became empty
    validate_tail_section_with_filtering_context(&tail1, "TAIL1")?;
    validate_tail_section_with_filtering_context(&tail2, "TAIL2")?;

    let geometry = Geometry::new(
        elements,
        angstrom_to_bohr(&DVector::from_vec(coords))
            .data
            .as_vec()
            .clone(),
    );
    let lst1 = if !lst1_elements.is_empty() {
        Some(Geometry::new(
            lst1_elements,
            angstrom_to_bohr(&DVector::from_vec(lst1_coords))
                .data
                .as_vec()
                .clone(),
        ))
    } else {
        None
    };
    let lst2 = if !lst2_elements.is_empty() {
        Some(Geometry::new(
            lst2_elements,
            angstrom_to_bohr(&DVector::from_vec(lst2_coords))
                .data
                .as_vec()
                .clone(),
        ))
    } else {
        None
    };

    // Validate constraints against the parsed geometry
    validate_constraints_against_geometry(&constraints, &geometry)?;

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

/// Parses a single constraint line from the *CONSTR section.
///
/// Supports the following constraint types:
/// - `r atom1 atom2 target_distance` - Bond length constraint
/// - `a atom1 atom2 atom3 target_angle` - Bond angle constraint (degrees)
/// - `d atom1 atom2 atom3 atom4 target_dihedral` - Dihedral angle constraint (degrees)
///
/// # Arguments
///
/// * `line` - The constraint line to parse
/// * `constraints` - Mutable vector to add parsed constraints to
///
/// # Returns
///
/// `Ok(())` if parsing succeeds, `Err(ParseError)` with detailed error information if parsing fails.
///
/// # Examples
///
/// ```
/// // Bond constraint: keep atoms 1-2 at 1.5 Å
/// r 1 2 1.5
///
/// // Angle constraint: keep angle 1-2-3 at 120 degrees
/// a 1 2 3 120.0
///
/// // Dihedral constraint: keep dihedral 1-2-3-4 at 180 degrees
/// d 1 2 3 4 180.0
/// ```
fn parse_constraint(line: &str, constraints: &mut Vec<Constraint>) -> Result<()> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return Ok(());
    }

    let parts: Vec<&str> = trimmed.split_whitespace().collect();
    if parts.is_empty() {
        return Ok(());
    }

    let constraint_type = parts[0].to_lowercase();

    match constraint_type.as_str() {
        "r" | "bond" => {
            if parts.len() < 4 {
                return Err(ParseError::Parse(format!(
                    "Bond constraint requires 4 parameters: 'r atom1 atom2 target_distance'. Got: '{}'", 
                    line
                )));
            }

            let a = parse_atom_index(parts[1], "first atom", line)?;
            let b = parse_atom_index(parts[2], "second atom", line)?;
            let target = parse_distance_target(parts[3], line)?;

            if a == b {
                return Err(ParseError::Parse(format!(
                    "Bond constraint cannot have the same atom twice: atom {} in line '{}'",
                    a + 1,
                    line
                )));
            }

            constraints.push(Constraint::Bond {
                atoms: (a, b),
                target,
            });
        }
        "a" | "angle" => {
            if parts.len() < 5 {
                return Err(ParseError::Parse(format!(
                    "Angle constraint requires 5 parameters: 'a atom1 atom2 atom3 target_angle'. Got: '{}'", 
                    line
                )));
            }

            let a = parse_atom_index(parts[1], "first atom", line)?;
            let b = parse_atom_index(parts[2], "second atom", line)?;
            let c = parse_atom_index(parts[3], "third atom", line)?;
            let target_degrees = parse_angle_target(parts[4], line)?;
            let target = target_degrees.to_radians();

            // Validate atom indices are unique
            if a == b || b == c || a == c {
                return Err(ParseError::Parse(format!(
                    "Angle constraint atoms must be unique. Got atoms {}, {}, {} in line '{}'",
                    a + 1,
                    b + 1,
                    c + 1,
                    line
                )));
            }

            constraints.push(Constraint::Angle {
                atoms: (a, b, c),
                target,
            });
        }
        "d" | "dihedral" => {
            if parts.len() < 6 {
                return Err(ParseError::Parse(format!(
                    "Dihedral constraint requires 6 parameters: 'd atom1 atom2 atom3 atom4 target_dihedral'. Got: '{}'", 
                    line
                )));
            }

            let a = parse_atom_index(parts[1], "first atom", line)?;
            let b = parse_atom_index(parts[2], "second atom", line)?;
            let c = parse_atom_index(parts[3], "third atom", line)?;
            let d = parse_atom_index(parts[4], "fourth atom", line)?;
            let target_degrees = parse_angle_target(parts[5], line)?;
            let target = target_degrees.to_radians();

            // Validate atom indices are unique
            let atoms = [a, b, c, d];
            for i in 0..4 {
                for j in (i + 1)..4 {
                    if atoms[i] == atoms[j] {
                        return Err(ParseError::Parse(format!(
                            "Dihedral constraint atoms must be unique. Got atoms {}, {}, {}, {} in line '{}'", 
                            a + 1, b + 1, c + 1, d + 1, line
                        )));
                    }
                }
            }

            constraints.push(Constraint::Dihedral {
                atoms: (a, b, c, d),
                target,
            });
        }
        _ => {
            return Err(ParseError::Parse(format!(
                "Unknown constraint type '{}'. Supported types: 'r' (bond), 'a' (angle), 'd' (dihedral). Line: '{}'", 
                constraint_type, line
            )));
        }
    }
    Ok(())
}

/// Parses an atom index from a string with detailed error reporting.
fn parse_atom_index(s: &str, atom_description: &str, full_line: &str) -> Result<usize> {
    let index = s.parse::<usize>()
        .map_err(|_| ParseError::Parse(format!(
            "Invalid {} index '{}' in constraint line '{}'. Atom indices must be positive integers.", 
            atom_description, s, full_line
        )))?;

    if index == 0 {
        return Err(ParseError::Parse(format!(
            "Atom indices must be 1-based (starting from 1), got {} for {} in line '{}'",
            index, atom_description, full_line
        )));
    }

    Ok(index - 1) // Convert to 0-based indexing
}

/// Parses a distance target value with validation.
fn parse_distance_target(s: &str, full_line: &str) -> Result<f64> {
    let distance = s.parse::<f64>()
        .map_err(|_| ParseError::Parse(format!(
            "Invalid distance target '{}' in constraint line '{}'. Distance must be a positive number.", 
            s, full_line
        )))?;

    if distance <= 0.0 {
        return Err(ParseError::Parse(format!(
            "Distance target must be positive, got {} in line '{}'",
            distance, full_line
        )));
    }

    if distance > 20.0 {
        return Err(ParseError::Parse(format!(
            "Distance target {} Å seems unreasonably large in line '{}'. Maximum allowed: 20.0 Å",
            distance, full_line
        )));
    }

    Ok(distance)
}

/// Parses an angle target value (in degrees) with validation.
fn parse_angle_target(s: &str, full_line: &str) -> Result<f64> {
    let angle = s.parse::<f64>().map_err(|_| {
        ParseError::Parse(format!(
            "Invalid angle target '{}' in constraint line '{}'. Angle must be a number in degrees.",
            s, full_line
        ))
    })?;

    if !(0.0..=360.0).contains(&angle) {
        return Err(ParseError::Parse(format!(
            "Angle target {} degrees is outside valid range [0, 360] in line '{}'",
            angle, full_line
        )));
    }

    Ok(angle)
}

/// Validates parsed constraints against the molecular geometry.
///
/// This function performs comprehensive validation of constraints including:
/// - Atom index bounds checking
/// - Duplicate constraint detection
/// - Geometric feasibility checks
/// - Target value reasonableness
///
/// # Arguments
///
/// * `constraints` - List of parsed constraints to validate
/// * `geometry` - Molecular geometry to validate against
///
/// # Returns
///
/// `Ok(())` if all constraints are valid, `Err(ParseError)` with detailed error information otherwise.
fn validate_constraints_against_geometry(
    constraints: &[Constraint],
    geometry: &Geometry,
) -> Result<()> {
    let num_atoms = geometry.num_atoms;

    // Validate each constraint
    for (i, constraint) in constraints.iter().enumerate() {
        match constraint {
            Constraint::Bond {
                atoms: (a, b),
                target,
            } => {
                // Check atom indices
                if *a >= num_atoms {
                    return Err(ParseError::Parse(format!(
                        "Bond constraint {}: atom index {} exceeds number of atoms ({})",
                        i + 1,
                        a + 1,
                        num_atoms
                    )));
                }
                if *b >= num_atoms {
                    return Err(ParseError::Parse(format!(
                        "Bond constraint {}: atom index {} exceeds number of atoms ({})",
                        i + 1,
                        b + 1,
                        num_atoms
                    )));
                }

                // Check target reasonableness
                if *target > 10.0 {
                    return Err(ParseError::Parse(format!(
                        "Bond constraint {}: target distance {:.3} Å is unreasonably large",
                        i + 1,
                        target
                    )));
                }
            }
            Constraint::Angle {
                atoms: (a, b, c),
                target: _,
            } => {
                // Check atom indices
                for (atom_idx, atom_name) in [(*a, "first"), (*b, "second"), (*c, "third")] {
                    if atom_idx >= num_atoms {
                        return Err(ParseError::Parse(format!(
                            "Angle constraint {}: {} atom index {} exceeds number of atoms ({})",
                            i + 1,
                            atom_name,
                            atom_idx + 1,
                            num_atoms
                        )));
                    }
                }
            }
            Constraint::Dihedral {
                atoms: (a, b, c, d),
                target: _,
            } => {
                // Check atom indices
                for (atom_idx, atom_name) in
                    [(*a, "first"), (*b, "second"), (*c, "third"), (*d, "fourth")]
                {
                    if atom_idx >= num_atoms {
                        return Err(ParseError::Parse(format!(
                            "Dihedral constraint {}: {} atom index {} exceeds number of atoms ({})",
                            i + 1,
                            atom_name,
                            atom_idx + 1,
                            num_atoms
                        )));
                    }
                }
            }
        }
    }

    // Check for duplicate constraints
    for i in 0..constraints.len() {
        for j in (i + 1)..constraints.len() {
            if constraints_are_equivalent(&constraints[i], &constraints[j]) {
                return Err(ParseError::Parse(format!(
                    "Duplicate constraints found: constraint {} and {} define the same geometric parameter", 
                    i + 1, j + 1
                )));
            }
        }
    }

    Ok(())
}

/// Checks if two constraints are equivalent (define the same geometric parameter).
fn constraints_are_equivalent(c1: &Constraint, c2: &Constraint) -> bool {
    match (c1, c2) {
        (
            Constraint::Bond {
                atoms: (a1, b1), ..
            },
            Constraint::Bond {
                atoms: (a2, b2), ..
            },
        ) => (a1 == a2 && b1 == b2) || (a1 == b2 && b1 == a2),
        (
            Constraint::Angle {
                atoms: (a1, b1, c1),
                ..
            },
            Constraint::Angle {
                atoms: (a2, b2, c2),
                ..
            },
        ) => (a1 == a2 && b1 == b2 && c1 == c2) || (a1 == c2 && b1 == b2 && c1 == a2),
        (
            Constraint::Dihedral {
                atoms: (a1, b1, c1, d1),
                ..
            },
            Constraint::Dihedral {
                atoms: (a2, b2, c2, d2),
                ..
            },
        ) => {
            (a1 == a2 && b1 == b2 && c1 == c2 && d1 == d2)
                || (a1 == d2 && b1 == c2 && c1 == b2 && d1 == a2)
        }
        _ => false,
    }
}

fn parse_scan(line: &str, config: &mut Config) -> Result<()> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 6 {
        return Ok(());
    }

    let scan_type = match parts[1].to_lowercase().as_str() {
        "r" if parts.len() >= 7 => {
            let a = parts[2]
                .parse::<usize>()
                .map_err(|_| ParseError::Parse("Invalid atom".into()))?
                - 1;
            let b = parts[3]
                .parse::<usize>()
                .map_err(|_| ParseError::Parse("Invalid atom".into()))?
                - 1;
            ScanType::Bond { atoms: (a, b) }
        }
        "a" if parts.len() >= 8 => {
            let a = parts[2]
                .parse::<usize>()
                .map_err(|_| ParseError::Parse("Invalid atom".into()))?
                - 1;
            let b = parts[3]
                .parse::<usize>()
                .map_err(|_| ParseError::Parse("Invalid atom".into()))?
                - 1;
            let c = parts[4]
                .parse::<usize>()
                .map_err(|_| ParseError::Parse("Invalid atom".into()))?
                - 1;
            ScanType::Angle { atoms: (a, b, c) }
        }
        _ => return Ok(()),
    };

    let offset = if matches!(scan_type, ScanType::Bond { .. }) {
        4
    } else {
        5
    };
    let start = parts[offset]
        .parse()
        .map_err(|_| ParseError::Parse("Invalid start".into()))?;
    let num_points = parts[offset + 1]
        .parse()
        .map_err(|_| ParseError::Parse("Invalid num".into()))?;
    let step_size = parts[offset + 2]
        .parse()
        .map_err(|_| ParseError::Parse("Invalid step".into()))?;

    config.scans.push(ScanSpec {
        scan_type,
        start,
        num_points,
        step_size,
    });
    Ok(())
}

fn parse_parameter(line: &str, config: &mut Config, fixed_atoms: &mut Vec<usize>) -> Result<()> {
    let parts: Vec<&str> = line.splitn(2, '=').collect();
    if parts.len() != 2 {
        return Ok(());
    }

    let key = parts[0].trim().to_lowercase();
    let raw_value = parts[1].trim();

    // Remove inline comments from the value
    let value = if let Some(comment_pos) = raw_value.find('#') {
        raw_value[..comment_pos].trim()
    } else {
        raw_value
    };

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
        "use_hybrid_gediis" => config.use_hybrid_gediis = value.to_lowercase() == "true",
        "switch_step" => {
            config.switch_step = value.parse().unwrap_or_else(|_| {
                eprintln!(
                    "Warning: Invalid switch_step value '{}', using default (3)",
                    value
                );
                3
            });
        }
        "drive_atoms" => {
            config.drive_atoms = value
                .split(',')
                .filter_map(|s| s.trim().parse::<usize>().ok())
                .map(|x| x.saturating_sub(1)) // Convert to 0-based indexing
                .collect();
        }
        "fixedatoms" => {
            for group in value.split(',') {
                if group.contains('-') {
                    let range: Vec<&str> = group.split('-').collect();
                    if range.len() == 2 {
                        if let (Ok(start), Ok(end)) =
                            (range[0].parse::<usize>(), range[1].parse::<usize>())
                        {
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
            config
                .program_commands
                .insert("gaussian".to_string(), value.to_string());
        }
        "orca_comm" => {
            config
                .program_commands
                .insert("orca".to_string(), value.to_string());
        }
        "xtb_comm" => {
            config
                .program_commands
                .insert("xtb".to_string(), value.to_string());
        }
        "bagel_comm" => {
            config
                .program_commands
                .insert("bagel".to_string(), value.to_string());
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

/// Filters comments and cleans tail section content.
///
/// This function processes raw tail section content by:
/// - Removing lines that start with '#' (full-line comments)
/// - Removing inline comments (text after '#' on the same line as valid keywords)
/// - Trimming leading and trailing whitespace from remaining lines
/// - Joining valid lines with single spaces
/// - Removing empty lines after trimming
///
/// # Arguments
///
/// * `raw_content` - The raw tail section content that may contain comments
///
/// # Returns
///
/// A cleaned string containing only valid Gaussian keywords separated by spaces
///
/// # Examples
///
/// ```
/// use omecp::parser::filter_tail_content;
///
/// let input = "# This is a comment\nTD(NStates=5)\n# Another comment\nRoot=1";
/// let result = filter_tail_content(input);
/// assert_eq!(result, "TD(NStates=5) Root=1");
///
/// // Inline comments are also removed
/// let inline = "TD(NStates=5) # inline comment\nRoot=1 # another comment";
/// let result = filter_tail_content(inline);
/// assert_eq!(result, "TD(NStates=5) Root=1");
/// ```
pub fn filter_tail_content(raw_content: &str) -> String {
    raw_content
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();

            // Skip empty lines
            if trimmed.is_empty() {
                return None;
            }

            // Skip lines that start with '#' (full-line comments)
            if is_comment_line(line) {
                return None;
            }

            // Remove inline comments (everything after '#' on the same line)
            let cleaned_line = if let Some(comment_pos) = trimmed.find('#') {
                trimmed[..comment_pos].trim()
            } else {
                trimmed
            };

            // Return the cleaned line if it's not empty after comment removal
            if cleaned_line.is_empty() {
                None
            } else {
                Some(cleaned_line)
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Checks if a line is a comment line.
///
/// A line is considered a comment if it starts with '#' after trimming
/// leading whitespace.
///
/// # Arguments
///
/// * `line` - The line to check
///
/// # Returns
///
/// `true` if the line is a comment, `false` otherwise
///
/// # Examples
///
/// ```
/// use omecp::parser::is_comment_line;
///
/// assert_eq!(is_comment_line("# This is a comment"), true);
/// assert_eq!(is_comment_line("  # Indented comment"), true);
/// assert_eq!(is_comment_line("TD(NStates=5)"), false);
/// assert_eq!(is_comment_line(""), false);
/// ```
pub fn is_comment_line(line: &str) -> bool {
    line.trim().starts_with('#')
}

/// Validates Gaussian keyword syntax.
///
/// Performs basic validation to ensure the content contains valid Gaussian
/// syntax after comment filtering. Checks for:
/// - Invalid characters that would break Gaussian input
/// - Proper keyword formatting
///
/// # Arguments
///
/// * `content` - The cleaned content to validate
///
/// # Returns
///
/// `Ok(())` if the content is valid, or `Err(ParseError)` with details
/// about the validation failure
///
/// # Examples
///
/// ```
/// use omecp::parser::validate_gaussian_keywords;
///
/// // Valid content
/// assert!(validate_gaussian_keywords("TD(NStates=5) Root=1").is_ok());
///
/// // Invalid content with newlines
/// assert!(validate_gaussian_keywords("TD(NStates=5)\nRoot=1").is_err());
/// ```
pub fn validate_gaussian_keywords(content: &str) -> Result<()> {
    validate_gaussian_syntax(content, "tail")
}

/// Validates basic Gaussian syntax patterns.
///
/// Implements comprehensive validation for Gaussian input syntax to ensure
/// that content will not break Gaussian execution. This function checks for:
/// - Invalid characters that would cause parsing errors
/// - Malformed keyword structures
/// - Remaining comment artifacts
///
/// # Arguments
///
/// * `content` - The content to validate for Gaussian compatibility
/// * `section_name` - The name of the section being validated for error reporting
///
/// # Returns
///
/// `Ok(())` if the content is valid, or `Err(ParseError::GaussianSyntaxError)`
/// with detailed information about the validation failure
///
/// # Examples
///
/// ```
/// use omecp::parser::validate_gaussian_syntax;
///
/// // Valid Gaussian keywords
/// assert!(validate_gaussian_syntax("TD(NStates=5) Root=1", "TAIL1").is_ok());
///
/// // Invalid content with newlines
/// assert!(validate_gaussian_syntax("TD(NStates=5)\nRoot=1", "TAIL1").is_err());
///
/// // Invalid content with comment characters
/// assert!(validate_gaussian_syntax("TD(NStates=5) # comment", "TAIL1").is_err());
/// ```
pub fn validate_gaussian_syntax(content: &str, section_name: &str) -> Result<()> {
    if content.is_empty() {
        return Ok(());
    }

    // Check for invalid characters that would break Gaussian input
    let invalid_chars = ['\n', '\r', '\t'];

    for ch in invalid_chars {
        if content.contains(ch) {
            let char_name = match ch {
                '\n' => "newline",
                '\r' => "carriage return",
                '\t' => "tab",
                _ => "unknown",
            };

            let suggestion = match ch {
                '\n' | '\r' => "Put all keywords on a single line or separate them with spaces instead of line breaks",
                '\t' => "Replace tabs with single spaces between keywords",
                _ => "Remove this character and use spaces to separate keywords",
            };

            return Err(ParseError::GaussianSyntaxError {
                section: section_name.to_string(),
                details: format!("Invalid {} character found in Gaussian keywords. Gaussian route sections must be on a single line. Suggestion: {}", char_name, suggestion),
                line_number: None,
            });
        }
    }

    // Check for remaining comment characters (should not happen after filtering)
    if content.contains('#') {
        return Err(ParseError::GaussianSyntaxError {
            section: section_name.to_string(),
            details: "Comment character '#' found in keywords. Comments are not allowed in Gaussian route sections. Suggestion: Move comments to separate lines in your OpenMECP input file, or remove them entirely from the tail section.".to_string(),
            line_number: None,
        });
    }

    // Check for potentially problematic characters that might break Gaussian parsing
    let problematic_chars = ['|', '&', ';', '>', '<'];
    for ch in problematic_chars {
        if content.contains(ch) {
            let suggestion = match ch {
                '|' => "Use parentheses or commas to separate options instead of pipes",
                '&' => "Remove ampersands - they are not valid in Gaussian keywords",
                ';' => "Use spaces or commas to separate keywords instead of semicolons",
                '>' | '<' => {
                    "Remove redirection operators - they are not valid in Gaussian keywords"
                }
                _ => "Remove this character and check Gaussian manual for proper keyword syntax",
            };

            return Err(ParseError::GaussianSyntaxError {
                section: section_name.to_string(),
                details: format!("Potentially problematic character '{}' found. This may cause Gaussian parsing errors. Suggestion: {}", ch, suggestion),
                line_number: None,
            });
        }
    }

    // Check for unbalanced parentheses which are common in Gaussian keywords
    let open_parens = content.chars().filter(|&c| c == '(').count();
    let close_parens = content.chars().filter(|&c| c == ')').count();
    if open_parens != close_parens {
        let suggestion = if open_parens > close_parens {
            format!("Add {} closing parenthesis ')'", open_parens - close_parens)
        } else {
            format!(
                "Remove {} extra closing parenthesis ')' or add matching opening parenthesis '('",
                close_parens - open_parens
            )
        };

        return Err(ParseError::GaussianSyntaxError {
            section: section_name.to_string(),
            details: format!("Unbalanced parentheses in Gaussian keywords: {} opening '(', {} closing ')'. Suggestion: {}", open_parens, close_parens, suggestion),
            line_number: None,
        });
    }

    // Check for excessive whitespace that might cause issues
    if content.contains("  ") {
        return Err(ParseError::GaussianSyntaxError {
            section: section_name.to_string(),
            details: "Multiple consecutive spaces found between keywords. Gaussian prefers single spaces. Suggestion: Replace multiple spaces with single spaces between keywords.".to_string(),
            line_number: None,
        });
    }

    // Check for common Gaussian keyword formatting issues
    if content.contains("=,") || content.contains(",=") {
        return Err(ParseError::GaussianSyntaxError {
            section: section_name.to_string(),
            details: "Invalid syntax with equals sign and comma combination. Suggestion: Check Gaussian manual for proper keyword=value syntax.".to_string(),
            line_number: None,
        });
    }

    // Check for trailing or leading commas which can cause issues
    let trimmed = content.trim();
    if trimmed.starts_with(',') || trimmed.ends_with(',') {
        return Err(ParseError::GaussianSyntaxError {
            section: section_name.to_string(),
            details: "Keywords start or end with comma, which may cause parsing issues. Suggestion: Remove leading/trailing commas and ensure proper keyword syntax.".to_string(),
            line_number: None,
        });
    }

    Ok(())
}

/// Validates tail section content with enhanced context and error reporting.
///
/// This function provides comprehensive validation with informational messages
/// about empty sections and detailed error reporting for formatting issues.
/// It's designed to give users clear guidance on how to fix their input files.
///
/// The function gracefully handles empty tail sections after comment filtering
/// by providing informational context without causing errors. This is important
/// because users may have tail sections that contain only comments, which become
/// empty after filtering but should not cause the parsing to fail.
///
/// # Arguments
///
/// * `content` - The tail section content to validate
/// * `section_name` - The name of the section (e.g., "TAIL1", "TAIL2") for error reporting
///
/// # Returns
///
/// `Ok(())` if the content is valid (including empty content), or `Err(ParseError)`
/// with detailed error information and suggestions for fixing issues
///
/// # Examples
///
/// ```
/// use omecp::parser::validate_tail_section_with_context;
///
/// // Empty content is handled gracefully
/// assert!(validate_tail_section_with_context("", "TAIL1").is_ok());
///
/// // Valid content passes validation
/// assert!(validate_tail_section_with_context("TD(NStates=5)", "TAIL1").is_ok());
/// ```
pub fn validate_tail_section_with_context(content: &str, section_name: &str) -> Result<()> {
    // Handle empty content gracefully with informational context
    if content.is_empty() {
        // Log informational message about empty tail section
        // This is acceptable and doesn't cause errors, but provides user awareness
        log_empty_tail_section_info(section_name);
        return Ok(());
    }

    // Delegate to the main validation function for non-empty content
    validate_tail_section(content, section_name)
}

/// Validates tail section content with filtering context awareness.
///
/// This function provides enhanced validation that takes into account the filtering
/// process. It validates the content and provides informational messages when
/// sections are empty, helping users understand the filtering behavior.
///
/// # Arguments
///
/// * `content` - The filtered tail section content to validate
/// * `section_name` - The name of the section (e.g., "TAIL1", "TAIL2") for error reporting
///
/// # Returns
///
/// `Ok(())` if the content is valid (including empty content), or `Err(ParseError)`
/// with detailed error information and suggestions for fixing issues
pub fn validate_tail_section_with_filtering_context(
    content: &str,
    section_name: &str,
) -> Result<()> {
    // Use the existing validation function which already handles empty content gracefully
    validate_tail_section_with_context(content, section_name)
}

/// Logs informational message about empty tail sections after filtering.
///
/// This function provides user-friendly information when tail sections become
/// empty after comment removal. It helps users understand that this is acceptable
/// behavior and provides guidance on when they might want to add content.
///
/// # Arguments
///
/// * `section_name` - The name of the empty section for context
///
/// # Implementation Note
///
/// Currently uses `eprintln!` for immediate user feedback. In a production
/// environment, this could be replaced with a proper logging framework
/// or structured logging system.
fn log_empty_tail_section_info(section_name: &str) {
    // Provide informational message to stderr (non-blocking for normal operation)
    // This gives users awareness without causing errors or stopping execution
    eprintln!(
        "INFO: {} section is empty after comment filtering. This is acceptable behavior.",
        section_name
    );
    eprintln!(
        "      If you intended to include Gaussian keywords in {}, ensure they are not commented out.",
        section_name
    );
    eprintln!(
        "      Common {} keywords: TD(NStates=5,Root=1), TD(NStates=10), etc.",
        section_name
    );
}

/// Validates tail section content for Gaussian compatibility.
///
/// This function performs comprehensive validation of tail section content
/// after comment filtering to ensure it will work correctly with Gaussian.
/// It checks for proper syntax and provides meaningful error messages with
/// specific suggestions for fixing common formatting issues.
///
/// # Arguments
///
/// * `content` - The tail section content to validate
/// * `section_name` - The name of the section (e.g., "TAIL1", "TAIL2") for error reporting
///
/// # Returns
///
/// `Ok(())` if the content is valid, or `Err(ParseError)` with details
/// about the validation failure including the section name and helpful suggestions
///
/// # Examples
///
/// ```
/// use omecp::parser::validate_tail_section;
///
/// // Valid content
/// assert!(validate_tail_section("TD(NStates=5) Root=1", "TAIL1").is_ok());
///
/// // Empty content is acceptable
/// assert!(validate_tail_section("", "TAIL1").is_ok());
///
/// // Invalid content with comments (should be filtered before validation)
/// assert!(validate_tail_section("TD(NStates=5) # comment", "TAIL1").is_err());
/// ```
pub fn validate_tail_section(content: &str, section_name: &str) -> Result<()> {
    // Empty content after filtering is acceptable - provide informational context
    if content.is_empty() {
        return Ok(());
    }

    // Perform additional tail-section specific validations before general syntax validation

    // Check for common TD-DFT keyword patterns and provide specific guidance
    if content.to_uppercase().contains("TD") && !content.contains('(') {
        let suggestion_message = format!(
            "TD keyword found without required parentheses in {} section.\n\nProblem: TD-DFT calculations require parameters in parentheses.\n\nSuggestions:\n  • Use 'TD(NStates=N)' where N is the number of excited states (e.g., TD(NStates=5))\n  • For specific roots: 'TD(NStates=N,Root=M)' where M is the state number\n  • Common examples:\n    - TD(NStates=5,Root=1)  # First excited state, calculate 5 states\n    - TD(NStates=10)        # Calculate 10 excited states",
            section_name
        );

        return Err(ParseError::InvalidTailSection {
            section: section_name.to_string(),
            message: suggestion_message,
            line_number: None,
        });
    }

    // Check for ROOT keyword without TD
    if content.to_uppercase().contains("ROOT") && !content.to_uppercase().contains("TD") {
        let suggestion_message = format!(
            "ROOT keyword found without TD keyword in {} section.\n\nProblem: ROOT is typically used with TD-DFT calculations to specify which excited state to optimize.\n\nSuggestions:\n  • Use 'TD(NStates=N,Root=M)' for combined TD-DFT with root specification\n  • Example: 'TD(NStates=5,Root=1)' to optimize the first excited state\n  • If you need separate keywords: 'TD(NStates=5) Root=1'",
            section_name
        );

        return Err(ParseError::InvalidTailSection {
            section: section_name.to_string(),
            message: suggestion_message,
            line_number: None,
        });
    }

    // Check for common misspellings or formatting issues
    let common_issues = [
        (
            "NSTATES",
            "Use 'NStates' instead of 'NSTATES' (case-sensitive)",
        ),
        (
            "nstates",
            "Use 'NStates' instead of 'nstates' (case-sensitive)",
        ),
        ("root=", "Use 'Root=' instead of 'root=' (case-sensitive)"),
        ("ROOT=", "Use 'Root=' instead of 'ROOT=' (case-sensitive)"),
        ("td(", "Use 'TD(' instead of 'td(' (case-sensitive)"),
    ];

    for (wrong, suggestion) in &common_issues {
        if content.contains(wrong) {
            return Err(ParseError::InvalidTailSection {
                section: section_name.to_string(),
                message: format!(
                    "Potential keyword formatting issue: found '{}'. Suggestion: {}",
                    wrong, suggestion
                ),
                line_number: None,
            });
        }
    }

    // Check for missing equals signs in keyword=value pairs
    if content.contains("NStates") && !content.contains("NStates=") {
        let suggestion_message = format!(
            "NStates keyword found without value assignment in {} section.\n\nProblem: NStates requires a numeric value to specify how many excited states to calculate.\n\nSuggestions:\n  • Use 'NStates=N' where N is the number of excited states\n  • Common values: NStates=5, NStates=10, NStates=20\n  • Example: 'TD(NStates=5,Root=1)' or 'TD(NStates=5) Root=1'",
            section_name
        );

        return Err(ParseError::InvalidTailSection {
            section: section_name.to_string(),
            message: suggestion_message,
            line_number: None,
        });
    }

    if content.contains("Root") && !content.contains("Root=") {
        let suggestion_message = format!(
            "Root keyword found without value assignment in {} section.\n\nProblem: Root requires a numeric value to specify which excited state to optimize.\n\nSuggestions:\n  • Use 'Root=N' where N is the excited state number (starting from 1)\n  • Example: 'Root=1' for first excited state, 'Root=2' for second excited state\n  • Complete example: 'TD(NStates=5,Root=1)' or 'TD(NStates=5) Root=1'",
            section_name
        );

        return Err(ParseError::InvalidTailSection {
            section: section_name.to_string(),
            message: suggestion_message,
            line_number: None,
        });
    }

    // Validate the content using the existing Gaussian keyword validation
    validate_gaussian_keywords(content).map_err(|e| match e {
        ParseError::GaussianSyntaxError {
            details,
            line_number,
            ..
        } => {
            // Create a user-friendly error message with specific suggestions
            let user_friendly_message =
                create_user_friendly_tail_error(&details, section_name, content);

            ParseError::InvalidTailSection {
                section: section_name.to_string(),
                message: user_friendly_message,
                line_number,
            }
        }
        _ => {
            // Handle other types of validation errors
            let fallback_message = format!(
                "Validation failed for {} section: {}\n\n{}",
                section_name,
                e,
                get_tail_section_suggestions(content, section_name)
            );

            ParseError::InvalidTailSection {
                section: section_name.to_string(),
                message: fallback_message,
                line_number: None,
            }
        }
    })
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
                coords.push(
                    coord_str
                        .parse()
                        .map_err(|_| ParseError::Parse("Invalid coordinate in XYZ file".into()))?,
                );
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
                    coords.push(
                        coord_str.parse().map_err(|_| {
                            ParseError::Parse("Invalid coordinate in GJF file".into())
                        })?,
                    );
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
                    coords.push(
                        coord_str.parse().map_err(|_| {
                            ParseError::Parse("Invalid coordinate in LOG file".into())
                        })?,
                    );
                }
            }
        }
    }
    Ok((elements, coords))
}

/// Provides suggestions for fixing common tail section formatting issues.
///
/// This function analyzes the content and provides specific, actionable
/// suggestions based on common mistakes users make when writing tail sections.
///
/// # Arguments
///
/// * `content` - The problematic content to analyze
/// * `section_name` - The name of the section for context
///
/// # Returns
///
/// A string containing specific suggestions for fixing the issues
fn get_tail_section_suggestions(content: &str, section_name: &str) -> String {
    let mut suggestions = Vec::new();

    // Check for common patterns and provide specific advice
    if content.contains('\n') {
        suggestions.push("Put all keywords on a single line separated by spaces".to_string());
    }

    if content.contains('#') {
        suggestions.push("Move comments to separate lines outside the tail section".to_string());
    }

    if content.to_uppercase().contains("TD") && !content.contains('(') {
        suggestions.push("Add parentheses to TD keyword: TD(NStates=5,Root=1)".to_string());
    }

    if content.contains("  ") {
        suggestions.push("Replace multiple spaces with single spaces between keywords".to_string());
    }

    if content.chars().filter(|&c| c == '(').count()
        != content.chars().filter(|&c| c == ')').count()
    {
        suggestions.push("Check that all parentheses are properly balanced".to_string());
    }

    // Add general guidance if no specific issues found
    if suggestions.is_empty() {
        suggestions.push(format!("Ensure {} contains valid Gaussian keywords (e.g., 'TD(NStates=5,Root=1)' for TD-DFT calculations)", section_name));
        suggestions.push("Check the Gaussian manual for proper keyword syntax".to_string());
        suggestions.push("Remove any shell command characters or special symbols".to_string());
    }

    format!(
        "Suggestions for fixing {} section:\n  • {}",
        section_name,
        suggestions.join("\n  • ")
    )
}

/// Creates a user-friendly error message for tail section validation failures.
///
/// This function takes technical validation errors and converts them into
/// clear, actionable error messages that help users understand and fix
/// their input file formatting issues.
///
/// # Arguments
///
/// * `error_details` - The technical error details from validation
/// * `section_name` - The name of the section where the error occurred
/// * `content` - The problematic content for generating suggestions
///
/// # Returns
///
/// A formatted error message with context and suggestions
fn create_user_friendly_tail_error(
    error_details: &str,
    section_name: &str,
    content: &str,
) -> String {
    let suggestions = get_tail_section_suggestions(content, section_name);

    format!(
        "Problem in {} section: {}\n\n{}\n\nExample of valid {} content:\n  TD(NStates=5,Root=1)\n  or\n  TD(NStates=10) Root=2",
        section_name,
        error_details,
        suggestions,
        section_name
    )
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
        Err(ParseError::Parse(
            "Unsupported external geometry file format".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhanced_error_messages_td_without_parentheses() {
        let result = validate_tail_section("TD NStates=5", "TAIL1");
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("TD keyword found without required parentheses"));
        assert!(error_msg.contains("TD(NStates=N)"));
    }

    #[test]
    fn test_enhanced_error_messages_unbalanced_parentheses() {
        let result = validate_tail_section("TD(NStates=5", "TAIL1");
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Unbalanced parentheses"));
        assert!(error_msg.contains("Add 1 closing parenthesis"));
    }

    #[test]
    fn test_enhanced_error_messages_comment_character() {
        let result = validate_tail_section("TD(NStates=5) # comment", "TAIL1");
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Comment character '#' found"));
        assert!(error_msg.contains("Comments are not allowed in Gaussian route sections"));
    }

    #[test]
    fn test_enhanced_error_messages_case_sensitivity() {
        let result = validate_tail_section("td(nstates=5)", "TAIL1");
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("case-sensitive"));
    }

    #[test]
    fn test_enhanced_error_messages_root_without_td() {
        let result = validate_tail_section("Root=1", "TAIL2");
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("ROOT keyword found without TD keyword"));
        assert!(error_msg.contains("TD(NStates=N,Root=M)"));
    }

    #[test]
    fn test_enhanced_error_messages_nstates_without_equals() {
        let result = validate_tail_section("TD(NStates)", "TAIL1");
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("NStates keyword found without value assignment"));
        assert!(error_msg.contains("NStates=N"));
    }

    #[test]
    fn test_enhanced_error_messages_valid_content() {
        let result = validate_tail_section("TD(NStates=5,Root=1)", "TAIL1");
        assert!(result.is_ok());
    }

    #[test]
    fn test_enhanced_error_messages_empty_content() {
        let result = validate_tail_section("", "TAIL1");
        assert!(result.is_ok());
    }

    #[test]
    fn test_graceful_empty_tail_section_handling() {
        // Test that empty content is handled gracefully
        let result = validate_tail_section_with_context("", "TAIL1");
        assert!(result.is_ok());

        let result = validate_tail_section_with_context("", "TAIL2");
        assert!(result.is_ok());
    }

    #[test]
    fn test_graceful_empty_tail_section_with_filtering_context() {
        // Test the filtering context function
        let result = validate_tail_section_with_filtering_context("", "TAIL1");
        assert!(result.is_ok());

        let result = validate_tail_section_with_filtering_context("", "TAIL2");
        assert!(result.is_ok());
    }

    #[test]
    fn test_filter_tail_content_empty_result() {
        // Test that filtering only comments results in empty string
        let only_comments = "# This is a comment\n# Another comment\n# More comments";
        let result = filter_tail_content(only_comments);
        assert_eq!(result, "");

        // Test mixed content with some valid keywords
        let mixed_content = "# Comment\nTD(NStates=5)\n# Another comment\nRoot=1";
        let result = filter_tail_content(mixed_content);
        assert_eq!(result, "TD(NStates=5) Root=1");

        // Test empty input
        let empty_input = "";
        let result = filter_tail_content(empty_input);
        assert_eq!(result, "");

        // Test whitespace only
        let whitespace_only = "   \n\t\n   ";
        let result = filter_tail_content(whitespace_only);
        assert_eq!(result, "");

        // Test inline comments are removed
        let inline_comments =
            "TD(NStates=5) # This is an inline comment\nRoot=1 # Another inline comment";
        let result = filter_tail_content(inline_comments);
        assert_eq!(result, "TD(NStates=5) Root=1");

        // Test line that becomes empty after removing inline comment
        let only_inline = "   # This entire line is a comment";
        let result = filter_tail_content(only_inline);
        assert_eq!(result, "");

        // Test mixed inline and full-line comments
        let mixed_inline = "TD(NStates=5) # inline comment\n# full line comment\nRoot=1";
        let result = filter_tail_content(mixed_inline);
        assert_eq!(result, "TD(NStates=5) Root=1");
    }

    #[test]
    fn test_enhanced_error_messages_multiple_spaces() {
        let result = validate_tail_section("TD(NStates=5)  Root=1", "TAIL1");
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Multiple consecutive spaces"));
        assert!(error_msg.contains("single spaces"));
    }

    #[test]
    fn test_user_friendly_suggestions() {
        let result = validate_tail_section("TD(NStates=5) # This is a comment", "TAIL1");
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Suggestions for fixing TAIL1 section"));
        assert!(error_msg.contains("Example of valid TAIL1 content"));
    }

    #[test]
    fn test_parameter_parsing_with_inline_comments() {
        // Test that inline comments are properly removed from parameter values
        let mut config = Config::default();
        let mut fixed_atoms = Vec::new();

        // Test nprocs with inline comment
        let result = parse_parameter("nprocs = 30 #processors", &mut config, &mut fixed_atoms);
        assert!(result.is_ok());
        assert_eq!(config.nprocs, 30);

        // Test mem with inline comment
        let result = parse_parameter(
            "mem = 120GB # memory to be used",
            &mut config,
            &mut fixed_atoms,
        );
        assert!(result.is_ok());
        assert_eq!(config.mem, "120GB");

        // Test method with inline comment
        let result = parse_parameter(
            "method = B3LYP/6-31G* # method comment",
            &mut config,
            &mut fixed_atoms,
        );
        assert!(result.is_ok());
        assert_eq!(config.method, "B3LYP/6-31G*");

        // Test charge with inline comment
        let result = parse_parameter(
            "charge = 1 # molecular charge",
            &mut config,
            &mut fixed_atoms,
        );
        assert!(result.is_ok());
        assert_eq!(config.charge1, 1);

        // Test mult1 with inline comment
        let result = parse_parameter("mult1 = 3 # triplet state", &mut config, &mut fixed_atoms);
        assert!(result.is_ok());
        assert_eq!(config.mult1, 3);
    }
}
