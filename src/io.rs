
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
    let modified_method = modify_method_for_run_mode(
        &config.method,
        config.program,
        config.run_mode,
    );
    
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
    
    // Build route section following Python format exactly
    // Python: f'# {Method} {Td1} nosymm'
    let route_section = if clean_td.is_empty() {
        format!("# {} nosymm", method_str)
    } else {
        format!("# {} {} nosymm", method_str, clean_td)
    };

    // Format following Python exactly: 
    // f'%chk=a.chk\n%nprocshared={NProcs} \n%mem={Mem} \n# {Method} {Td1} nosymm\n\n Title Card \n\n{Charge1} {Mult1}'
    format!(
        "%chk={}\n%nprocshared={} \n%mem={} \n{}\n\n Title Card \n\n{} {}",
        chk_file, config.nprocs, config.mem, route_section, charge, mult
    )
}

/// Builds an ORCA input file header string (legacy interface).
///
/// This function is maintained for backward compatibility. New code should use
/// `build_program_header()` which includes dynamic method modification.
///
/// # Arguments
///
/// * `config` - The global configuration for the MECP calculation
/// * `charge` - Molecular charge for the current state
/// * `mult` - Spin multiplicity for the current state
/// * `tail` - Additional ORCA keywords (tail section content)
///
/// # Returns
///
/// Returns a `String` containing the formatted ORCA input header.
pub fn build_orca_header(
    config: &crate::config::Config,
    charge: i32,
    mult: usize,
    tail: &str,
) -> String {
    // Use the dynamic method modification for consistency
    let modified_method = modify_method_for_run_mode(
        &config.method,
        config.program,
        config.run_mode,
    );
    
    let mut temp_config = config.clone();
    temp_config.method = modified_method;
    
    build_orca_header_internal(&temp_config, charge, mult, tail)
}

/// Internal ORCA header builder that doesn't modify the method string.
///
/// This function constructs the header for an ORCA input file based on the 
/// provided configuration and state-specific parameters. It assumes the method
/// string has already been modified by `modify_method_for_run_mode()`.
///
/// # Arguments
///
/// * `config` - The global configuration with pre-modified method string
/// * `charge` - Molecular charge for the current state
/// * `mult` - Spin multiplicity for the current state
/// * `tail` - Additional ORCA keywords (tail section content)
///
/// # Returns
///
/// Returns a `String` containing the formatted ORCA input header.
fn build_orca_header_internal(
    config: &crate::config::Config,
    charge: i32,
    mult: usize,
    tail: &str,
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

    // Replace *** with proper .gbw file paths (following Python logic)
    let method_line = if method_line.contains("***") {
        let gbw_file = if charge == config.charge1 && mult == config.mult1 {
            "running_dir/a.gbw"
        } else {
            "running_dir/b.gbw"
        };
        method_line.replace("***", gbw_file)
    } else {
        method_line
    };

    // Format following Python exactly:
    // f'%pal nprocs {NProcs} end\n%maxcore {Mem} \n! {Method} \n\n *xyz {Charge1} {Mult1}'
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
    let basis = if config.basis_set.is_empty() { "cc-pVDZ" } else { &config.basis_set };
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
/// This function implements the core logic from Python MECP.py's modifyMETHOD function,
/// adding program-specific keywords and run mode-specific modifications to the method string.
/// This ensures that calculations use the correct keywords for each scenario.
///
/// # Method Modification Logic
///
/// 1. **Program-specific keywords** (always added):
///    - Gaussian: `force` (for gradient calculations)
///    - ORCA: `engrad` (for energy and gradient calculations)
///    - XTB/BAGEL: No modification needed
///
/// 2. **Stability keywords** (added for `Stable` mode):
///    - Gaussian: `stable=opt` (perform stability analysis and reoptimize if unstable)
///    - ORCA: `%scf stabperform true StabRestartUHFifUnstable true end` (stability analysis)
///
/// 3. **Guess keywords** (added for all modes except `NoRead`):
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
/// // Stable mode with ORCA
/// let modified = io::modify_method_for_run_mode("B3LYP def2-SVP", QMProgram::Orca, RunMode::Stable);
/// assert!(modified.contains("engrad"));
/// assert!(modified.contains("stabperform"));
///
/// // NoRead mode (no guess keywords)
/// let modified = io::modify_method_for_run_mode("B3LYP/6-31G*", QMProgram::Gaussian, RunMode::NoRead);
/// assert_eq!(modified, "B3LYP/6-31G* force");
/// ```
pub fn modify_method_for_run_mode(
    method: &str,
    program: crate::config::QMProgram,
    run_mode: crate::config::RunMode,
) -> String {
    let mut modified_method = method.to_string();
    
    // Add program-specific keywords (following Python MECP.py modifyMETHOD logic)
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
    
    // Add stability keywords for stable mode (following Python logic)
    if run_mode == crate::config::RunMode::Stable && !modified_method.is_empty() {
        match program {
            crate::config::QMProgram::Gaussian | crate::config::QMProgram::Custom => {
                modified_method.push_str(" stable=opt");
            }
            crate::config::QMProgram::Orca => {
                modified_method.push_str("\n %scf stabperform true StabRestartUHFifUnstable true end \n");
            }
            _ => {}
        }
    }
    
    // Add guess keywords (except for noread mode, following Python logic)
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
/// # Examples
///
/// ```
/// use omecp::config::{Config, QMProgram, RunMode};
/// use omecp::io;
///
/// let mut config = Config::default();
/// config.program = QMProgram::Orca;
/// config.method = "B3LYP def2-SVP".to_string();
/// config.run_mode = RunMode::Stable;
/// config.nprocs = 8;
/// config.mem = "8000".to_string();
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
    build_program_header_with_chk(config, charge, mult, td_or_tail, state, None)
}

/// Builds a program-specific input file header string with custom checkpoint file name.
///
/// This function creates headers for quantum chemistry programs with the ability to specify
/// a custom checkpoint file name. It automatically applies method modifications based on the
/// run mode (e.g., adding "force" and "guess=read" for read mode).
///
/// # Arguments
///
/// * `config` - The global configuration for the MECP calculation
/// * `charge` - Molecular charge for the current state
/// * `mult` - Spin multiplicity for the current state  
/// * `td_or_tail` - TD-DFT keywords (Gaussian) or tail section content (ORCA)
/// * `state` - State index for multi-reference calculations (BAGEL)
/// * `chk_file` - Optional custom checkpoint file name. If None, uses default naming
///
/// # Returns
///
/// Returns a `String` containing the formatted input header for the specified program.
pub fn build_program_header_with_chk(
    config: &crate::config::Config,
    charge: i32,
    mult: usize,
    td_or_tail: &str,
    state: usize,
    chk_file: Option<&str>,
) -> String {
    // Get dynamically modified method based on run mode and program
    let modified_method = modify_method_for_run_mode(
        &config.method,
        config.program,
        config.run_mode,
    );
    
    // Create temporary config with modified method for header generation
    let mut temp_config = config.clone();
    temp_config.method = modified_method;
    
    // Determine checkpoint file name
    let checkpoint_file = chk_file.unwrap_or_else(|| {
        // Default checkpoint file names based on charge/mult
        if charge == config.charge1 && mult == config.mult1 {
            "state_A.chk"
        } else {
            "state_B.chk"
        }
    });
    
    match config.program {
        crate::config::QMProgram::Gaussian => {
            build_gaussian_header_internal_with_chk(&temp_config, charge, mult, td_or_tail, checkpoint_file)
        }
        crate::config::QMProgram::Orca => {
            build_orca_header_internal(&temp_config, charge, mult, td_or_tail)
        }
        crate::config::QMProgram::Xtb => {
            build_xtb_header(&temp_config, charge, mult, td_or_tail)
        }
        crate::config::QMProgram::Bagel => {
            build_bagel_header(&temp_config, charge, mult, state)
        }
        crate::config::QMProgram::Custom => {
            // For custom programs, fall back to Gaussian format
            // Users can override this via custom interface files
            build_gaussian_header_internal_with_chk(&temp_config, charge, mult, td_or_tail, checkpoint_file)
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, RunMode};

    #[test]
    fn test_clean_gaussian_keywords_empty_result() {
        // Test that only comments result in empty string
        let only_comments = "# This is a comment\n# Another comment";
        let result = clean_gaussian_keywords(only_comments);
        assert_eq!(result, "");
    }

    #[test]
    fn test_clean_gaussian_keywords_mixed_content() {
        // Test mixed comments and keywords
        let mixed = "# Comment\nTD(NStates=5)\n# Another comment\nRoot=1";
        let result = clean_gaussian_keywords(mixed);
        assert_eq!(result, "TD(NStates=5) Root=1");
    }

    #[test]
    fn test_clean_gaussian_keywords_inline_comments() {
        // Test inline comments are removed
        let inline = "TD(NStates=5) # This is an inline comment\nRoot=1 # Another inline comment";
        let result = clean_gaussian_keywords(inline);
        assert_eq!(result, "TD(NStates=5) Root=1");
        
        // Test line that becomes empty after removing inline comment
        let only_inline = "# This entire line is a comment";
        let result = clean_gaussian_keywords(only_inline);
        assert_eq!(result, "");
        
        // Test mixed inline and full-line comments
        let mixed_inline = "TD(NStates=5) # inline comment\n# full line comment\nRoot=1";
        let result = clean_gaussian_keywords(mixed_inline);
        assert_eq!(result, "TD(NStates=5) Root=1");
    }

    #[test]
    fn test_clean_gaussian_keywords_empty_input() {
        // Test empty input
        let empty = "";
        let result = clean_gaussian_keywords(empty);
        assert_eq!(result, "");
    }

    #[test]
    fn test_clean_gaussian_keywords_whitespace_only() {
        // Test whitespace and empty lines only
        let whitespace = "   \n\t\n   \n";
        let result = clean_gaussian_keywords(whitespace);
        assert_eq!(result, "");
    }

    #[test]
    fn test_build_gaussian_header_empty_td() {
        // Test that empty TD keywords are handled gracefully
        let mut config = Config::default();
        config.method = "B3LYP".to_string();
        config.nprocs = 4;
        config.mem = "4GB".to_string();
        config.run_mode = RunMode::Normal;

        let header = build_gaussian_header(&config, 0, 1, "");
        
        // Should contain proper formatting with spaces after resource specifications
        assert!(header.contains("%nprocshared=4 "));
        assert!(header.contains("%mem=4GB "));
        assert!(header.contains("# B3LYP force guess=read nosymm"));
        assert!(header.contains("0 1")); // Charge and multiplicity
    }

    #[test]
    fn test_build_gaussian_header_comment_only_td() {
        // Test that TD with only comments results in clean header
        let mut config = Config::default();
        config.method = "B3LYP".to_string();
        config.nprocs = 4;
        config.mem = "4GB".to_string();
        config.run_mode = RunMode::Normal;

        let td_with_comments = "# This is a comment\n# Another comment";
        let header = build_gaussian_header(&config, 0, 1, td_with_comments);
        
        // Should contain proper formatting
        assert!(header.contains("%nprocshared=4 "));
        assert!(header.contains("%mem=4GB "));
        assert!(header.contains("# B3LYP force guess=read nosymm"));
        // The route section should start with # but not contain comment content
        let lines: Vec<&str> = header.lines().collect();
        let route_line = lines.iter().find(|line| line.starts_with('#')).unwrap();
        assert!(!route_line.contains("This is a comment"));
        assert!(!route_line.contains("Another comment"));
    }

    #[test]
    fn test_build_gaussian_header_mixed_td_content() {
        // Test that mixed TD content is cleaned properly
        let mut config = Config::default();
        config.method = "B3LYP".to_string();
        config.nprocs = 4;
        config.mem = "4GB".to_string();
        config.run_mode = RunMode::Normal;

        let mixed_td = "# Comment\nTD(NStates=5)\n# Another comment\nRoot=1";
        let header = build_gaussian_header(&config, 0, 1, mixed_td);
        
        // Should contain proper formatting and cleaned TD keywords
        assert!(header.contains("%nprocshared=4 "));
        assert!(header.contains("%mem=4GB "));
        assert!(header.contains("# B3LYP force guess=read TD(NStates=5) Root=1 nosymm"));
        // The route section should start with # but not contain comment content
        let lines: Vec<&str> = header.lines().collect();
        let route_line = lines.iter().find(|line| line.starts_with('#')).unwrap();
        assert!(!route_line.contains("Comment"));
        assert!(!route_line.contains("Another comment"));
    }

    #[test]
    fn test_clean_keywords() {
        // Test the generic keyword cleaning function
        let mixed = "# Comment\nTD(NStates=5) # inline\n# Another comment\nRoot=1";
        let result = clean_keywords(mixed);
        assert_eq!(result, "TD(NStates=5) Root=1");
        
        // Test empty result
        let only_comments = "# Only comments\n# More comments";
        let result = clean_keywords(only_comments);
        assert_eq!(result, "");
    }

    #[test]
    fn test_build_orca_header() {
        let mut config = Config::default();
        config.program = crate::config::QMProgram::Orca;
        config.method = "B3LYP def2-SVP".to_string();
        config.nprocs = 8;
        config.mem = "8000".to_string();
        config.run_mode = RunMode::Normal;

        let header = build_orca_header(&config, 0, 1, "");
        
        // Should contain ORCA-specific formatting
        assert!(header.contains("%pal nprocs 8 end"));
        assert!(header.contains("%maxcore 8000"));
        assert!(header.contains("! B3LYP def2-SVP engrad"));
        assert!(header.contains("!moread"));
        assert!(header.contains("*xyz 0 1"));
    }

    #[test]
    fn test_build_orca_header_with_tail() {
        let mut config = Config::default();
        config.program = crate::config::QMProgram::Orca;
        config.method = "B3LYP def2-SVP".to_string();
        config.nprocs = 4;
        config.mem = "4000".to_string();
        config.run_mode = RunMode::Normal;

        let tail_with_comments = "# Comment\n%tddft\n  nroots 5\nend\n# Another comment";
        let header = build_orca_header(&config, -1, 2, tail_with_comments);
        
        
        // Should contain cleaned tail content - the method is now modified by dynamic function
        assert!(header.contains("B3LYP def2-SVP engrad"));
        assert!(header.contains("%tddft nroots 5 end"));
        assert!(header.contains("!moread"));
        assert!(header.contains("*xyz -1 2"));
        // Should not contain comments
        assert!(!header.contains("Comment"));
        assert!(!header.contains("Another comment"));
    }

    #[test]
    fn test_build_orca_header_noread_mode() {
        let mut config = Config::default();
        config.program = crate::config::QMProgram::Orca;
        config.method = "B3LYP def2-SVP".to_string();
        config.nprocs = 4;
        config.mem = "4000".to_string();
        config.run_mode = crate::config::RunMode::NoRead;

        let header = build_orca_header(&config, 0, 1, "");
        
        // Should not contain moread in noread mode
        assert!(!header.contains("!moread"));
        assert!(header.contains("! B3LYP def2-SVP engrad"));
    }

    #[test]
    fn test_build_xtb_header() {
        let config = Config::default();
        let header = build_xtb_header(&config, 1, 3, "");
        
        // XTB format: charge and unpaired electrons (mult-1)
        assert!(header.contains("$chrg 1"));
        assert!(header.contains("$uhf 2")); // mult=3 -> uhf=2
    }

    #[test]
    fn test_build_bagel_header() {
        let mut config = Config::default();
        config.program = crate::config::QMProgram::Bagel;
        config.basis_set = "cc-pVTZ".to_string();

        let header = build_bagel_header(&config, 0, 1, 2);
        
        // Should contain BAGEL JSON format
        assert!(header.contains("\"bagel\""));
        assert!(header.contains("\"charge\" : 0"));
        assert!(header.contains("\"nspin\" : 0")); // mult=1 -> nspin=0
        assert!(header.contains("\"target\" : 2"));
        assert!(header.contains("\"basis\" : \"cc-pVTZ\""));
    }

    #[test]
    fn test_build_bagel_header_default_basis() {
        let config = Config::default();
        let header = build_bagel_header(&config, -1, 2, 0);
        
        // Should use default basis when none specified
        assert!(header.contains("\"basis\" : \"cc-pVDZ\""));
        assert!(header.contains("\"charge\" : -1"));
        assert!(header.contains("\"nspin\" : 1")); // mult=2 -> nspin=1
    }

    #[test]
    fn test_modify_method_for_run_mode_gaussian() {
        // Test Gaussian method modification for different run modes
        
        // Normal mode: should add force and guess=read
        let result = modify_method_for_run_mode(
            "B3LYP/6-31G*", 
            crate::config::QMProgram::Gaussian, 
            crate::config::RunMode::Normal
        );
        assert_eq!(result, "B3LYP/6-31G* force guess=read");
        
        // NoRead mode: should add force but not guess=read
        let result = modify_method_for_run_mode(
            "B3LYP/6-31G*", 
            crate::config::QMProgram::Gaussian, 
            crate::config::RunMode::NoRead
        );
        assert_eq!(result, "B3LYP/6-31G* force");
        
        // Stable mode: should add force, guess=read, and stable=opt
        let result = modify_method_for_run_mode(
            "B3LYP/6-31G*", 
            crate::config::QMProgram::Gaussian, 
            crate::config::RunMode::Stable
        );
        assert_eq!(result, "B3LYP/6-31G* force stable=opt guess=read");
        
        // Read mode: should add force and guess=read
        let result = modify_method_for_run_mode(
            "B3LYP/6-31G*", 
            crate::config::QMProgram::Gaussian, 
            crate::config::RunMode::Read
        );
        assert_eq!(result, "B3LYP/6-31G* force guess=read");
        
        // InterRead mode: should add force and guess=read
        let result = modify_method_for_run_mode(
            "B3LYP/6-31G*", 
            crate::config::QMProgram::Gaussian, 
            crate::config::RunMode::InterRead
        );
        assert_eq!(result, "B3LYP/6-31G* force guess=read");
    }

    #[test]
    fn test_modify_method_for_run_mode_orca() {
        // Test ORCA method modification for different run modes
        
        // Normal mode: should add engrad and moread
        let result = modify_method_for_run_mode(
            "B3LYP def2-SVP", 
            crate::config::QMProgram::Orca, 
            crate::config::RunMode::Normal
        );
        assert!(result.contains("B3LYP def2-SVP engrad"));
        assert!(result.contains("!moread"));
        assert!(result.contains("%moinp \"***\""));
        
        // NoRead mode: should add engrad but not moread
        let result = modify_method_for_run_mode(
            "B3LYP def2-SVP", 
            crate::config::QMProgram::Orca, 
            crate::config::RunMode::NoRead
        );
        assert_eq!(result, "B3LYP def2-SVP engrad");
        
        // Stable mode: should add engrad, moread, and stability keywords
        let result = modify_method_for_run_mode(
            "B3LYP def2-SVP", 
            crate::config::QMProgram::Orca, 
            crate::config::RunMode::Stable
        );
        assert!(result.contains("B3LYP def2-SVP engrad"));
        assert!(result.contains("stabperform true"));
        assert!(result.contains("StabRestartUHFifUnstable true"));
        assert!(result.contains("!moread"));
    }

    #[test]
    fn test_modify_method_for_run_mode_xtb_bagel() {
        // Test that XTB and BAGEL don't modify method strings
        
        let result = modify_method_for_run_mode(
            "GFN2-xTB", 
            crate::config::QMProgram::Xtb, 
            crate::config::RunMode::Normal
        );
        assert_eq!(result, "GFN2-xTB");
        
        let result = modify_method_for_run_mode(
            "CASSCF", 
            crate::config::QMProgram::Bagel, 
            crate::config::RunMode::Stable
        );
        assert_eq!(result, "CASSCF");
    }

    #[test]
    fn test_modify_method_for_run_mode_empty_method() {
        // Test behavior with empty method string
        
        let result = modify_method_for_run_mode(
            "", 
            crate::config::QMProgram::Gaussian, 
            crate::config::RunMode::Normal
        );
        assert_eq!(result, "");
        
        let result = modify_method_for_run_mode(
            "", 
            crate::config::QMProgram::Orca, 
            crate::config::RunMode::Normal
        );
        assert_eq!(result, "");
    }

    #[test]
    fn test_build_program_header_with_dynamic_modification() {
        // Test that build_program_header uses dynamic method modification
        
        // Gaussian with stable mode
        let mut config = Config::default();
        config.program = crate::config::QMProgram::Gaussian;
        config.method = "B3LYP/6-31G*".to_string();
        config.run_mode = crate::config::RunMode::Stable;
        config.nprocs = 4;
        config.mem = "4GB".to_string();
        
        let header = build_program_header(&config, 0, 1, "", 0);
        assert!(header.contains("B3LYP/6-31G* force stable=opt guess=read"));
        
        // ORCA with noread mode
        config.program = crate::config::QMProgram::Orca;
        config.method = "B3LYP def2-SVP".to_string();
        config.run_mode = crate::config::RunMode::NoRead;
        
        let header = build_program_header(&config, 0, 1, "", 0);
        assert!(header.contains("B3LYP def2-SVP engrad"));
        assert!(!header.contains("!moread"));
    }

    #[test]
    fn test_build_program_header_dispatch() {
        // Test that the dispatcher works correctly for different programs
        
        // Gaussian
        let mut config = Config::default();
        config.program = crate::config::QMProgram::Gaussian;
        config.method = "B3LYP".to_string();
        let header = build_program_header(&config, 0, 1, "", 0);
        assert!(header.contains("%chk=calc.chk"));
        assert!(header.contains("%nprocshared="));
        
        // ORCA
        config.program = crate::config::QMProgram::Orca;
        let header = build_program_header(&config, 0, 1, "", 0);
        assert!(header.contains("%pal nprocs"));
        assert!(header.contains("*xyz"));
        
        // XTB
        config.program = crate::config::QMProgram::Xtb;
        let header = build_program_header(&config, 1, 2, "", 0);
        assert!(header.contains("$chrg 1"));
        assert!(header.contains("$uhf 1"));
        
        // BAGEL
        config.program = crate::config::QMProgram::Bagel;
        let header = build_program_header(&config, 0, 1, "", 1);
        assert!(header.contains("\"bagel\""));
        assert!(header.contains("\"target\" : 1"));
    }

    #[test]
    fn test_orca_header_gbw_replacement() {
        // Test that ORCA headers properly replace *** with .gbw file paths
        
        let mut config = Config::default();
        config.program = crate::config::QMProgram::Orca;
        config.method = "B3LYP def2-SVP".to_string();
        config.run_mode = crate::config::RunMode::Normal;
        config.charge1 = 0;
        config.mult1 = 1;
        config.charge2 = 0;
        config.mult2 = 3;
        
        // Test state A (should use a.gbw)
        let header = build_program_header(&config, 0, 1, "", 0);
        assert!(header.contains("running_dir/a.gbw"));
        assert!(!header.contains("***"));
        
        // Test state B (should use b.gbw)
        let header = build_program_header(&config, 0, 3, "", 0);
        assert!(header.contains("running_dir/b.gbw"));
        assert!(!header.contains("***"));
    }
}