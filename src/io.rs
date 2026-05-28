//! File I/O utilities for geometry and checkpoint files.
//!
//! This module provides functions for reading and writing molecular geometries
//! in various formats including XYZ, Gaussian input/output, and checkpoint files.

use crate::geometry::Geometry;
use std::fs;
use std::io::Result;
use std::path::Path;

/// Writes a molecular geometry to an XYZ file.
///
/// The XYZ format is a simple plain-text format for molecular geometries,
/// widely used in chemistry software. It consists of:
/// 1. Number of atoms
/// 2. A comment line (empty in this implementation)
/// 3. Lines for each atom: Element X Y Z
///
/// # Arguments
///
/// * `geom` - The molecular geometry to write
/// * `path` - The path to the output XYZ file
///
/// # Returns
///
/// Returns `Ok(())` on success, or an `std::io::Error` if file writing fails.
///
/// # Examples
///
/// ```
/// use omecp::geometry::Geometry;
/// use omecp::io;
/// use std::path::Path;
///
/// fn main() -> std::io::Result<()> {
///     let elements = vec!["C".to_string(), "H".to_string()];
///     let coords = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
///     let geometry = Geometry::new(elements, coords);
///
///     io::write_xyz(&geometry, Path::new("molecule.xyz"))?;
///     std::fs::remove_file("molecule.xyz")?;
///     Ok(())
/// }
/// ```
pub fn write_xyz(geom: &Geometry, path: &Path) -> Result<()> {
    let mut content = format!("{}\n\n", geom.num_atoms);

    for i in 0..geom.num_atoms {
        let coords = geom.get_atom_coords(i);
        // Coordinates are already in Angstrom - write directly
        content.push_str(&format!(
            "{}  {:.8}  {:.8}  {:.8}\n",
            geom.elements[i], coords[0], coords[1], coords[2]
        ));
    }

    fs::write(path, content)
}

/// Cleans Gaussian keywords by removing comments and extra whitespace.
///
/// This function processes multi-line keyword strings to remove:
/// - Lines starting with '#' (full-line comments)
/// - Inline comments (text after '#' on the same line as valid keywords)
/// - Empty lines
/// - Leading and trailing whitespace from each line
///
/// The remaining valid keywords are joined with single spaces to create
/// a clean string suitable for Gaussian route sections. If all content
/// is filtered out (e.g., only comments), an empty string is returned,
/// which is handled gracefully by the Gaussian header generation.
///
/// # Arguments
///
/// * `keywords` - The raw keyword string that may contain comments and extra whitespace
///
/// # Returns
///
/// Returns a `String` containing cleaned keywords joined with single spaces.
/// Returns an empty string if no valid keywords remain after filtering.
///
/// # Examples
///
/// ```
/// use omecp::io;
///
/// let raw_keywords = "# This is a comment\nTD(NStates=5)\n# Another comment\nRoot=1\n\n";
/// let cleaned = io::clean_gaussian_keywords(raw_keywords);
/// assert_eq!(cleaned, "TD(NStates=5) Root=1");
///
/// // Empty result when only comments are present
/// let only_comments = "# Only comments\n# More comments";
/// let empty_result = io::clean_gaussian_keywords(only_comments);
/// assert_eq!(empty_result, "");
///
/// // Inline comments are removed
/// let inline_comments = "TD(NStates=5) # This is an inline comment\nRoot=1 # Another inline comment";
/// let cleaned_inline = io::clean_gaussian_keywords(inline_comments);
/// assert_eq!(cleaned_inline, "TD(NStates=5) Root=1");
/// ```
pub fn clean_gaussian_keywords(keywords: &str) -> String {
    let result = keywords
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();

            // Skip empty lines
            if trimmed.is_empty() {
                return None;
            }

            // Skip lines that start with '#' (full-line comments)
            if trimmed.starts_with('#') {
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
        .join(" ");

    // The result may be empty if all content was filtered out (e.g., only comments)
    // This is acceptable and will be handled gracefully by the caller
    result
}

/// Cleans keywords by removing comments and extra whitespace (generic version).
///
/// This function works for any quantum chemistry program by removing:
/// - Lines starting with '#' (full-line comments)
/// - Inline comments (text after '#' on the same line as valid keywords)
/// - Empty lines
/// - Leading and trailing whitespace from each line
///
/// The remaining valid keywords are joined with single spaces.
///
/// # Arguments
///
/// * `keywords` - The raw keyword string that may contain comments and extra whitespace
///
/// # Returns
///
/// Returns a `String` containing cleaned keywords joined with single spaces.
pub fn clean_keywords(keywords: &str) -> String {
    keywords
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();

            // Skip empty lines
            if trimmed.is_empty() {
                return None;
            }

            // Skip lines that start with '#' (full-line comments)
            if trimmed.starts_with('#') {
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

/// Builds a Gaussian input file header string (legacy interface).
///
/// This function is maintained for backward compatibility. New code should use
/// `build_program_header()` which includes dynamic method modification.
///
/// # Arguments
///
/// * `config` - The global configuration for the MECP calculation
/// * `charge` - Molecular charge for the current state
/// * `mult` - Spin multiplicity for the current state
/// * `td` - TD-DFT keywords (e.g., "TD(NStates=5,Root=1)"), may contain comments
///
/// # Returns
///
/// Returns a `String` containing the formatted Gaussian input header with clean route section.
pub fn build_gaussian_header(
    config: &crate::config::Config,
    charge: i32,
    mult: usize,
    td: &str,
) -> String {
    // Use the dynamic method modification for consistency
    let modified_method =
        modify_method_for_run_mode(&config.method, config.program, config.run_mode);

    let mut temp_config = config.clone();
    temp_config.method = modified_method;

    build_gaussian_header_internal(&temp_config, charge, mult, td)
}

/// Internal Gaussian header builder that doesn't modify the method string.
///
/// This function constructs the route section and title card for a Gaussian
/// input file based on the provided configuration and state-specific parameters.
/// It assumes the method string has already been modified by `modify_method_for_run_mode()`.
///
/// # Arguments
///
/// * `config` - The global configuration with pre-modified method string
/// * `charge` - Molecular charge for the current state
/// * `mult` - Spin multiplicity for the current state
/// * `td` - TD-DFT keywords (e.g., "TD(NStates=5,Root=1)"), may contain comments
///
/// # Returns
///
/// Returns a `String` containing the formatted Gaussian input header.
fn build_gaussian_header_internal(
    config: &crate::config::Config,
    charge: i32,
    mult: usize,
    td: &str,
) -> String {
    build_gaussian_header_internal_with_chk(config, charge, mult, td, "calc.chk")
}

fn build_gaussian_header_internal_with_chk(
    config: &crate::config::Config,
    charge: i32,
    mult: usize,
    td: &str,
    chk_file: &str,
) -> String {
    // Use the method string as-is (already modified by modify_method_for_run_mode)
    let method_str = &config.method;

    // Clean TD-DFT keywords to remove comments and extra whitespace
    let clean_td = clean_gaussian_keywords(td);

    // Build route section 
    let route_section = if clean_td.is_empty() {
        format!("# {} nosymm", method_str)
    } else {
        format!("# {} {} nosymm", method_str, clean_td)
    };

    format!(
        "%chk={}\n%nprocshared={} \n%mem={} \n{}\n\n Title Card \n\n{} {}",
        chk_file, config.nprocs, config.mem, route_section, charge, mult
    )
}

/// Builds an ORCA input file header string with basename.
///
/// This function constructs the header for an ORCA input file based on the
/// provided configuration and state-specific parameters. It requires a
/// basename for proper .gbw file path construction.
///
/// # Arguments
///
/// * `config` - The global configuration for the MECP calculation
/// * `charge` - Molecular charge for the current state
/// * `mult` - Spin multiplicity for the current state
/// * `tail` - Additional ORCA keywords (tail section content)
/// * `input_basename` - Full path prefix for .gbw files
///
/// # Returns
///
/// Returns a `String` containing the formatted ORCA input header.

pub fn build_orca_header(
    config: &crate::config::Config,
    charge: i32,
    mult: usize,
    tail: &str,
    input_basename: &str,
) -> String {
    // Use the dynamic method modification for consistency
    let modified_method =
        modify_method_for_run_mode(&config.method, config.program, config.run_mode);

    let mut temp_config = config.clone();
    temp_config.method = modified_method;

    build_orca_header_internal(&temp_config, charge, mult, tail, input_basename, None)
}


fn build_orca_header_internal(
    config: &crate::config::Config,
    charge: i32,
    mult: usize,
    tail: &str,
    input_basename: &str,
    chk_file: Option<&str>,
) -> String {
    // Use the method string as-is (already modified by modify_method_for_run_mode)
    let method_str = &config.method;

    // Clean tail keywords to remove comments
    let clean_tail = clean_keywords(tail);

    // Build the method line
    let method_line = if clean_tail.is_empty() {
        format!("! {}", method_str)
    } else {
        format!("! {} {}", method_str, clean_tail)
    };

    // If chk_file is provided (e.g. for chain-linking steps), use it directly.
    // Otherwise, fall back to constructing it from input_basename.
    let method_line = if method_line.contains("***") {
        let gbw_file = if let Some(chk) = chk_file {
            chk.to_string()
        } else if mult == config.mult_state_a {
            format!("{}_state_A.gbw", input_basename)
        } else {
            format!("{}_state_B.gbw", input_basename)
        };
        method_line.replace("***", &gbw_file)
    } else {
        method_line
    };

    format!(
        "%pal nprocs {} end\n%maxcore {} \n{}\n\n *xyz {} {}",
        config.nprocs, config.mem, method_line, charge, mult
    )
}

/// Builds an XTB input file header string.
///
/// XTB uses a simple format with just charge and multiplicity information.
/// The method is typically specified via command line arguments.
///
/// # Arguments
///
/// * `config` - The global configuration for the MECP calculation
/// * `charge` - Molecular charge for the current state
/// * `mult` - Spin multiplicity for the current state
/// * `_tail` - Additional keywords (unused for XTB)
///
/// # Returns
///
/// Returns a `String` containing the formatted XTB input header.
pub fn build_xtb_header(
    _config: &crate::config::Config,
    charge: i32,
    mult: usize,
    _tail: &str,
) -> String {
    // XTB uses a simple format - just charge and multiplicity
    // The method is specified via command line arguments
    format!("$chrg {}\n$uhf {}", charge, mult - 1)
}

/// Builds a BAGEL input file header string.
///
/// BAGEL uses JSON format and requires a model file. This function creates
/// the basic structure that will be filled with geometry and other parameters.
///
/// # Arguments
///
/// * `config` - The global configuration for the MECP calculation
/// * `charge` - Molecular charge for the current state
/// * `mult` - Spin multiplicity for the current state
/// * `state` - Electronic state index for multireference calculations
///
/// # Returns
///
/// Returns a `String` containing the formatted BAGEL input header.
pub fn build_bagel_header(
    config: &crate::config::Config,
    charge: i32,
    mult: usize,
    state: usize,
) -> String {
    // BAGEL uses JSON format - this is a basic template
    // The actual geometry will be inserted by the writeBAGEL equivalent function
    let basis = if config.basis_set.is_empty() {
        "cc-pVDZ"
    } else {
        &config.basis_set
    };
    let df_basis = if config.basis_set.is_empty() {
        "cc-pVDZ-jkfit".to_string()
    } else {
        format!("{}-jkfit", config.basis_set)
    };

    format!(
        r#"{{
  "bagel" : [
    {{
      "title" : "molecule",
      "basis" : "{}",
      "df_basis" : "{}",
      "charge" : {},
      "nspin" : {},
      "target" : {},
      "geometry" : [
        // Geometry will be inserted here
      ]
    }}
  ]
}}"#,
        basis,
        df_basis,
        charge,
        mult - 1, // nspin = 2S where mult = 2S+1
        state
    )
}

/// Dynamically modifies a QM method string based on run mode and program.
///
/// This function implements the core logic,
/// adding program-specific keywords and run mode-specific modifications to the method string.
/// This ensures that calculations use the correct keywords for each scenario.
///
/// # Method Modification Logic
///
/// 1. **Syntax Correction** (added for ORCA):
///    - Replaces Gaussian-style basis set separators (`/`) with spaces.
///    - E.g., "B3LYP/6-31G*" -> "B3LYP 6-31G*"
///
/// 2. **Program-specific keywords** (always added):
///    - Gaussian: `force` (for gradient calculations)
///    - ORCA: `engrad` (for energy and gradient calculations)
///    - XTB/BAGEL: No modification needed
///
/// 3. **Stability keywords** (added for `Stable` mode):
///    - Gaussian: `stable=opt` (perform stability analysis and reoptimize if unstable)
///    - ORCA: `%scf stabperform true StabRestartUHFifUnstable true end` (stability analysis)
///
/// 4. **Guess keywords** (added for all modes except `NoRead`):
///    - Gaussian: `guess=read` (read initial guess from checkpoint)
///    - ORCA: `!moread` with `%moinp "***"` (read molecular orbitals)
///
/// # Arguments
///
/// * `method` - The base QM method string (e.g., "B3LYP/6-31G*")
/// * `program` - The quantum chemistry program being used
/// * `run_mode` - The execution mode for the calculation
///
/// # Returns
///
/// Returns a `String` containing the modified method with appropriate keywords added.
///
/// # Examples
///
/// ```
/// use omecp::config::{QMProgram, RunMode};
/// use omecp::io;
///
/// // Normal mode with Gaussian
/// let modified = io::modify_method_for_run_mode("B3LYP/6-31G*", QMProgram::Gaussian, RunMode::Normal);
/// assert_eq!(modified, "B3LYP/6-31G* force guess=read");
///
/// // Normal mode with ORCA (handling Gaussian syntax)
/// let modified = io::modify_method_for_run_mode("B3LYP/6-31G*", QMProgram::Orca, RunMode::Normal);
/// assert!(modified.contains("B3LYP 6-31G*"));
/// assert!(modified.contains("engrad"));
/// ```
pub fn modify_method_for_run_mode(
    method: &str,
    program: crate::config::QMProgram,
    run_mode: crate::config::RunMode,
) -> String {
    let mut modified_method = method.to_string();

    // Fix Gaussian-style method/basis syntax (e.g., "B3LYP/6-31G*") for ORCA
    if program == crate::config::QMProgram::Orca && modified_method.contains('/') {
        modified_method = modified_method.replace('/', " ");
    }

    // Add program-specific keywords
    match program {
        crate::config::QMProgram::Gaussian | crate::config::QMProgram::Custom => {
            if !modified_method.is_empty() {
                modified_method.push_str(" force");
            }
        }
        crate::config::QMProgram::Orca => {
            if !modified_method.is_empty() {
                modified_method.push_str(" engrad");
            }
        }
        // XTB and BAGEL don't need method modification
        _ => {}
    }

    // Add stability keywords for stable mode
    if run_mode == crate::config::RunMode::Stable && !modified_method.is_empty() {
        match program {
            crate::config::QMProgram::Gaussian | crate::config::QMProgram::Custom => {
                modified_method.push_str(" stable=opt");
            }
            crate::config::QMProgram::Orca => {
                modified_method
                    .push_str("\n %scf stabperform true StabRestartUHFifUnstable true end \n");
            }
            _ => {}
        }
    }

    // Add guess keywords
    if run_mode != crate::config::RunMode::NoRead && !modified_method.is_empty() {
        match program {
            crate::config::QMProgram::Gaussian | crate::config::QMProgram::Custom => {
                modified_method.push_str(" guess=read");
            }
            crate::config::QMProgram::Orca => {
                modified_method.push_str("\n!moread \n %moinp \"***\"\n");
            }
            _ => {}
        }
    }

    modified_method
}

/// Builds a program-specific input file header string.
///
/// This function dispatches to the appropriate header building function
/// based on the quantum chemistry program specified in the configuration.
/// It now uses dynamic method modification to ensure run mode compatibility.
///
/// **Note**: For ORCA programs, this function will panic if no input basename is provided.
/// Use `build_program_header_with_basename()` instead for ORCA calculations.
///
/// # Arguments
///
/// * `config` - The global configuration for the MECP calculation
/// * `charge` - Molecular charge for the current state
/// * `mult` - Spin multiplicity for the current state
/// * `td_or_tail` - TD-DFT keywords (Gaussian) or tail section content (other programs)
/// * `state` - Electronic state index (used for BAGEL)
///
/// # Returns
///
/// Returns a `String` containing the formatted input header for the specified program.
///
/// # Panics
///
/// Panics if `config.program` is `QMProgram::Orca` since ORCA requires an input basename.
///
/// # Examples
///
/// ```
/// use omecp::config::{Config, QMProgram, RunMode};
/// use omecp::io;
///
/// let mut config = Config::default();
/// config.program = QMProgram::Gaussian; // Works for Gaussian
/// config.method = "B3LYP/6-31G*".to_string();
/// config.run_mode = RunMode::Normal;
///
/// let header = io::build_program_header(&config, 0, 1, "", 0);
/// println!("{}", header);
/// ```
pub fn build_program_header(
    config: &crate::config::Config,
    charge: i32,
    mult: usize,
    td_or_tail: &str,
    state: usize,
) -> String {
    if config.program == crate::config::QMProgram::Orca {
        panic!("ORCA requires input basename for .gbw file paths. Use build_program_header_with_basename() instead.");
    }
    build_program_header_with_chk(config, charge, mult, td_or_tail, state, None, None)
}

/// Builds a program-specific input file header string with input basename for ORCA .gbw paths.
///
/// This function is specifically designed for cases where you need to specify the basename
/// of the input file for ORCA calculations, which is used to construct proper .gbw file paths
/// (e.g., "calc/state_A.gbw" instead of "running_dir/state_A.gbw").
///
/// # Arguments
///
/// * `config` - The global configuration for the MECP calculation
/// * `charge` - Molecular charge for the current state
/// * `mult` - Spin multiplicity for the current state
/// * `td_or_tail` - TD-DFT keywords (Gaussian) or tail section content (ORCA)
/// * `state` - State index for multi-reference calculations (BAGEL)
/// * `input_basename` - Basename of input file for ORCA .gbw paths (e.g., "calc" for "calc.inp")
///
/// # Returns
///
/// Returns a `String` containing the formatted input header for the specified program.
///
/// # Examples
///
/// ```
/// use omecp::config::{Config, QMProgram, RunMode};
/// use omecp::io;
///
/// let mut config = Config::default();
/// config.program = QMProgram::Orca;
/// config.method = "B3LYP def2-SVP".to_string();
/// config.run_mode = RunMode::Read;
///
/// // This will generate ORCA header with "calc/compound_xyz_123_state_A.gbw" paths
/// let header = io::build_program_header_with_basename(&config, 0, 1, "", 0, "calc/compound_xyz_123");
/// ```
pub fn build_program_header_with_basename(
    config: &crate::config::Config,
    charge: i32,
    mult: usize,
    td_or_tail: &str,
    state: usize,
    input_basename: &str,
) -> String {
    build_program_header_with_chk(
        config,
        charge,
        mult,
        td_or_tail,
        state,
        None,
        Some(input_basename),
    )
}

/// Builds a program-specific input file header with custom checkpoint/GBW file support.
///
/// This function generates headers for any supported QM program, allowing precise control
/// over the checkpoint or wavefunction file usage. This is critical for:
/// - Gaussian: Specifying the `%chk` file path.
/// - ORCA: Specifying the `%moinp` file path for chain-linked restarts.
///
/// It automatically applies run-mode specific modifications (like `force`, `guess=read`,
/// or `!moread`) to the method string.
///
/// # Arguments
///
/// * `config` - The global configuration.
/// * `charge` - Molecular charge.
/// * `mult` - Spin multiplicity.
/// * `td_or_tail` - TD-DFT keywords (Gaussian) or tail content (ORCA).
/// * `state` - Electronic state index (for BAGEL).
/// * `chk_file` - Optional custom checkpoint/GBW file path.
///   - For Gaussian: Sets `%chk={chk_file}`.
///   - For ORCA: Sets `%moinp "{chk_file}"` if `!moread` is active.
/// * `input_basename` - Optional basename for default naming fallback (required for ORCA if chk_file is None).
///
/// # Returns
///
/// Returns the formatted input header string.

pub fn build_program_header_with_chk(
    config: &crate::config::Config,
    charge: i32,
    mult: usize,
    td_or_tail: &str,
    state: usize,
    chk_file: Option<&str>,
    input_basename: Option<&str>,
) -> String {
    // Get dynamically modified method based on run mode and program
    let modified_method =
        modify_method_for_run_mode(&config.method, config.program, config.run_mode);

    // Create temporary config with modified method for header generation
    let mut temp_config = config.clone();
    temp_config.method = modified_method;

    // Determine checkpoint file name
    let checkpoint_file = chk_file.unwrap_or(
        // Default checkpoint file names based on charge/mult
        if mult == config.mult_state_a {
            "state_A.chk"
        } else {
            "state_B.chk"
        },
    );

    match config.program {
        crate::config::QMProgram::Gaussian => build_gaussian_header_internal_with_chk(
            &temp_config,
            charge,
            mult,
            td_or_tail,
            checkpoint_file,
        ),
        crate::config::QMProgram::Orca => {
            let basename =
                input_basename.expect("ORCA requires input_basename parameter for .gbw file paths");
            // Pass chk_file explicitly to build_orca_header_internal.
            // In Gaussian this parameter is 'checkpoint_file' (unwrapped), but for Orca
            // we want the Option to allow fallback to basename logic if not provided.
            // However, the 'checkpoint_file' variable above unwrap_or's it with "state_A.chk".
            // We should pass the ORIGINAL chk_file option to Orca builder to preserve None.
            build_orca_header_internal(&temp_config, charge, mult, td_or_tail, basename, chk_file)
        }
        crate::config::QMProgram::Xtb => build_xtb_header(&temp_config, charge, mult, td_or_tail),
        crate::config::QMProgram::Bagel => build_bagel_header(&temp_config, charge, mult, state),
        crate::config::QMProgram::Custom => {
            // For custom programs, fall back to Gaussian format
            // Users can override this via custom interface files
            build_gaussian_header_internal_with_chk(
                &temp_config,
                charge,
                mult,
                td_or_tail,
                checkpoint_file,
            )
        }
    }
}

