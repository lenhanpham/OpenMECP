//! Run mode validation and compatibility checking for OpenMECP.
//!
//! This module implements comprehensive validation of run mode and program
//! combinations to ensure calculations are set up correctly and provide
//! helpful error messages and warnings to users.
//!
//! # Features
//!
//! - Run mode and QM program compatibility validation
//! - Wavefunction file existence checking
//! - Clear error messages for invalid combinations
//! - Enhanced user guidance and warnings
//! - Logging for mode transitions and file operations

use crate::config::{Config, QMProgram, RunMode};
use std::path::Path;

/// Result type for validation operations.
pub type ValidationResult<T> = Result<T, ValidationError>;

/// Comprehensive validation error with detailed user guidance.
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Error category for programmatic handling
    pub category: ErrorCategory,
    /// Human-readable error message
    pub message: String,
    /// Optional suggestion for fixing the issue
    pub suggestion: Option<String>,
    /// Optional reference to documentation or examples
    pub reference: Option<String>,
}

/// Categories of validation errors for better error handling.
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorCategory {
    /// Incompatible run mode and QM program combination
    IncompatibleCombination,
    /// Missing required wavefunction files
    MissingWavefunctionFiles,
    /// Invalid configuration parameters
    InvalidConfiguration,
    /// Missing required files or dependencies
    MissingDependencies,
    /// Unsupported feature for the selected program
    UnsupportedFeature,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)?;
        if let Some(suggestion) = &self.suggestion {
            write!(f, "\n\nSuggestion: {}", suggestion)?;
        }
        if let Some(reference) = &self.reference {
            write!(f, "\n\nFor more information: {}", reference)?;
        }
        Ok(())
    }
}

impl std::error::Error for ValidationError {}

/// Validates run mode compatibility with the selected QM program and configuration.
///
/// This function performs comprehensive validation following the Python MECP.py logic
/// and provides detailed error messages and suggestions for fixing issues.
///
/// # Arguments
///
/// * `config` - The configuration to validate
///
/// # Returns
///
/// Returns `Ok(())` if the configuration is valid, or a `ValidationError` with
/// detailed information about the issue and how to fix it.
///
/// # Examples
///
/// ```
/// use omecp::config::{Config, QMProgram, RunMode};
/// use omecp::validation::validate_run_mode_compatibility;
///
/// let mut config = Config::default();
/// config.program = QMProgram::Gaussian;
/// config.run_mode = RunMode::Read;
///
/// match validate_run_mode_compatibility(&config) {
///     Ok(()) => println!("Configuration is valid"),
///     Err(e) => println!("Validation error: {}", e),
/// }
/// ```
pub fn validate_run_mode_compatibility(config: &Config) -> ValidationResult<()> {
    // Check basic program/mode compatibility
    validate_program_mode_compatibility(config)?;

    // Check wavefunction file requirements
    validate_wavefunction_files(config)?;

    // Check program-specific requirements
    validate_program_specific_requirements(config)?;

    // Check run mode specific requirements
    validate_run_mode_requirements(config)?;

    Ok(())
}

/// Validates basic compatibility between QM program and run mode.
fn validate_program_mode_compatibility(config: &Config) -> ValidationResult<()> {
    match (config.program, config.run_mode) {
        // ORCA-specific validations
        (QMProgram::Orca, RunMode::Stable) => {
            // ORCA stability mode has limitations with RI
            if config.method.contains("RI") || config.method.contains("ri") {
                return Err(ValidationError {
                    category: ErrorCategory::IncompatibleCombination,
                    message: "ORCA stability analysis is incompatible with RI (Resolution of Identity) approximations".to_string(),
                    suggestion: Some("Use the 'read' mode instead and manually obtain the correct wavefunction, or remove RI keywords from the method".to_string()),
                    reference: Some("See ORCA manual section on stability analysis for details".to_string()),
                });
            }
        }

        // XTB-specific validations
        (QMProgram::Xtb, RunMode::Stable) => {
            return Err(ValidationError {
                category: ErrorCategory::UnsupportedFeature,
                message: "XTB does not support explicit stability analysis".to_string(),
                suggestion: Some("Use 'normal' or 'noread' mode for XTB calculations. XTB handles stability internally".to_string()),
                reference: Some("XTB automatically uses stable wavefunctions".to_string()),
            });
        }

        (QMProgram::Xtb, RunMode::InterRead) => {
            return Err(ValidationError {
                category: ErrorCategory::UnsupportedFeature,
                message: "XTB does not support inter_read mode (no wavefunction files)".to_string(),
                suggestion: Some("Use 'normal' or 'noread' mode for XTB calculations".to_string()),
                reference: Some("XTB uses internal wavefunction handling".to_string()),
            });
        }

        // BAGEL-specific validations
        (QMProgram::Bagel, RunMode::Read) | (QMProgram::Bagel, RunMode::InterRead) => {
            if config.bagel_model.is_empty() {
                return Err(ValidationError {
                    category: ErrorCategory::InvalidConfiguration,
                    message: "BAGEL calculations require a model file specification".to_string(),
                    suggestion: Some(
                        "Set the 'bagel_model' parameter to point to your BAGEL JSON template file"
                            .to_string(),
                    ),
                    reference: Some("See BAGEL documentation for model file format".to_string()),
                });
            }
        }

        // Custom program validations
        (QMProgram::Custom, _) => {
            if config.custom_interface_file.is_empty() {
                return Err(ValidationError {
                    category: ErrorCategory::InvalidConfiguration,
                    message: "Custom QM program requires interface configuration file".to_string(),
                    suggestion: Some("Set the 'custom_interface_file' parameter to point to your JSON interface configuration".to_string()),
                    reference: Some("See OpenMECP documentation for custom interface format".to_string()),
                });
            }
        }

        // Coordinate driving validations
        (_, RunMode::CoordinateDrive) => {
            if config.drive_type.is_empty() || config.drive_atoms.is_empty() {
                return Err(ValidationError {
                    category: ErrorCategory::InvalidConfiguration,
                    message: "Coordinate driving requires drive_type and drive_atoms parameters".to_string(),
                    suggestion: Some("Set drive_type (bond/angle/dihedral) and drive_atoms (list of atom indices)".to_string()),
                    reference: Some("See coordinate driving examples in OpenMECP documentation".to_string()),
                });
            }
        }

        // Path optimization validations
        (_, RunMode::PathOptimization) => {
            if config.drive_type.is_empty() || config.drive_atoms.is_empty() {
                return Err(ValidationError {
                    category: ErrorCategory::InvalidConfiguration,
                    message: "Path optimization requires initial path specification via drive_type and drive_atoms".to_string(),
                    suggestion: Some("Set drive_type and drive_atoms to define the initial reaction coordinate".to_string()),
                    reference: Some("Path optimization uses coordinate driving to create the initial path".to_string()),
                });
            }
        }

        // FixDE mode validations
        (_, RunMode::FixDE) => {
            if config.fix_de == 0.0 {
                return Err(ValidationError {
                    category: ErrorCategory::InvalidConfiguration,
                    message: "FixDE mode requires a target energy difference specification"
                        .to_string(),
                    suggestion: Some(
                        "Set the 'fix_de' parameter to your target energy difference in eV"
                            .to_string(),
                    ),
                    reference: Some(
                        "FixDE mode constrains the energy difference to the specified value"
                            .to_string(),
                    ),
                });
            }
        }

        // All other combinations are valid
        _ => {}
    }

    Ok(())
}

/// Validates wavefunction file requirements for read-based modes.
fn validate_wavefunction_files(config: &Config) -> ValidationResult<()> {
    match config.run_mode {
        RunMode::Read | RunMode::InterRead => {
            match config.program {
                QMProgram::Gaussian => {
                    // Check for Gaussian checkpoint files
                    let chk_files = [
                        "state_A.chk",
                        "state_B.chk",
                        "running_dir/state_A.chk",
                        "running_dir/state_B.chk",
                    ];
                    let mut found_files = Vec::new();

                    for file in &chk_files {
                        if Path::new(file).exists() {
                            found_files.push(*file);
                        }
                    }

                    if found_files.is_empty() {
                        return Err(ValidationError {
                            category: ErrorCategory::MissingWavefunctionFiles,
                            message: "Read mode requires Gaussian checkpoint files (state_A.chk, state_B.chk) but none were found".to_string(),
                            suggestion: Some("Run a calculation in 'normal' or 'noread' mode first to generate checkpoint files, or switch to 'noread' mode".to_string()),
                            reference: Some("Checkpoint files are created automatically during normal calculations".to_string()),
                        });
                    }

                    println!("Found Gaussian checkpoint files: {:?}", found_files);
                }

                QMProgram::Orca => {
                    // Check for ORCA wavefunction files
                    let gbw_files = [
                        "state_A.gbw",
                        "state_B.gbw",
                        "running_dir/state_A.gbw",
                        "running_dir/state_B.gbw",
                    ];
                    let mut found_files = Vec::new();

                    for file in &gbw_files {
                        if Path::new(file).exists() {
                            found_files.push(*file);
                        }
                    }

                    if found_files.is_empty() {
                        return Err(ValidationError {
                            category: ErrorCategory::MissingWavefunctionFiles,
                            message: "Read mode requires ORCA wavefunction files (state_A.gbw, state_B.gbw) but none were found".to_string(),
                            suggestion: Some("Run a calculation in 'normal' or 'noread' mode first to generate .gbw files, or switch to 'noread' mode".to_string()),
                            reference: Some("ORCA .gbw files contain the molecular orbitals and are created during calculations".to_string()),
                        });
                    }

                    println!("Found ORCA wavefunction files: {:?}", found_files);
                }

                QMProgram::Xtb => {
                    // XTB doesn't use persistent wavefunction files in the same way
                    // This is handled in the program compatibility check
                }

                QMProgram::Bagel => {
                    // BAGEL wavefunction handling is different - check model file instead
                    if !Path::new(&config.bagel_model).exists() {
                        return Err(ValidationError {
                            category: ErrorCategory::MissingDependencies,
                            message: format!("BAGEL model file '{}' not found", config.bagel_model),
                            suggestion: Some(
                                "Ensure the BAGEL model file path is correct and the file exists"
                                    .to_string(),
                            ),
                            reference: Some(
                                "BAGEL model files define the calculation template".to_string(),
                            ),
                        });
                    }
                }

                QMProgram::Custom => {
                    // Custom interface validation is handled elsewhere
                    if !Path::new(&config.custom_interface_file).exists() {
                        return Err(ValidationError {
                            category: ErrorCategory::MissingDependencies,
                            message: format!("Custom interface file '{}' not found", config.custom_interface_file),
                            suggestion: Some("Ensure the custom interface file path is correct and the file exists".to_string()),
                            reference: Some("Custom interface files define how to interact with your QM program".to_string()),
                        });
                    }
                }
            }
        }

        RunMode::Normal | RunMode::NoRead | RunMode::Stable => {
            // These modes don't require existing wavefunction files
        }

        _ => {
            // Other modes (CoordinateDrive, PathOptimization, FixDE) don't have specific wavefunction requirements
        }
    }

    Ok(())
}

/// Validates program-specific requirements and configurations.
fn validate_program_specific_requirements(config: &Config) -> ValidationResult<()> {
    match config.program {
        QMProgram::Gaussian => {
            // Validate Gaussian-specific settings
            if config.mp2 && config.method.contains("DFT") {
                return Err(ValidationError {
                    category: ErrorCategory::InvalidConfiguration,
                    message: "MP2 flag is incompatible with DFT methods in Gaussian".to_string(),
                    suggestion: Some(
                        "Either remove the MP2 flag or use a wavefunction method like HF or MP2"
                            .to_string(),
                    ),
                    reference: Some("MP2 is a post-HF method, not compatible with DFT".to_string()),
                });
            }
        }

        QMProgram::Orca => {
            // Validate ORCA-specific settings
            if config.mp2 {
                println!("Warning: MP2 flag may not be applicable for ORCA calculations");
            }
        }

        QMProgram::Bagel => {
            // BAGEL requires model file
            if config.bagel_model.is_empty() {
                return Err(ValidationError {
                    category: ErrorCategory::InvalidConfiguration,
                    message: "BAGEL calculations require a model file specification".to_string(),
                    suggestion: Some(
                        "Set the 'bagel_model' parameter to your BAGEL JSON template file"
                            .to_string(),
                    ),
                    reference: Some(
                        "BAGEL model files define the quantum chemistry method and basis set"
                            .to_string(),
                    ),
                });
            }
        }

        QMProgram::Xtb => {
            // XTB has limited method options
            if !config.method.is_empty() && !config.method.contains("GFN") {
                println!("Warning: XTB typically uses GFN methods (GFN1-xTB, GFN2-xTB). Method '{}' may not be recognized", config.method);
            }
        }

        QMProgram::Custom => {
            // Custom program validation is handled in compatibility check
        }
    }

    Ok(())
}

/// Validates run mode specific requirements.
fn validate_run_mode_requirements(config: &Config) -> ValidationResult<()> {
    match config.run_mode {
        RunMode::InterRead => {
            // Inter-read mode is specifically for open-shell singlets
            if config.mult1 != 1 || config.mult2 != 1 {
                println!("Warning: Inter-read mode is typically used for open-shell singlet calculations (mult1=1, mult2=1)");
                println!(
                    "Current multiplicities: mult1={}, mult2={}",
                    config.mult1, config.mult2
                );
            }
        }

        RunMode::Stable => {
            // Stability mode warnings
            match config.program {
                QMProgram::Orca => {
                    println!("ORCA Stability Mode Guidance:");
                    println!(
                        "- RHF calculations will not restart automatically if instability is found"
                    );
                    println!("- Remember to use UKS for singlet state calculations");
                    println!("- RI approximations are not supported in stability analysis");
                    println!("- Consider using 'read' mode with manually converged wavefunctions for RI calculations");
                }
                QMProgram::Gaussian => {
                    println!(
                        "Gaussian Stability Mode: Will automatically handle wavefunction stability"
                    );
                }
                _ => {}
            }
        }

        RunMode::CoordinateDrive => {
            // Validate coordinate driving parameters
            if config.drive_start == config.drive_end {
                return Err(ValidationError {
                    category: ErrorCategory::InvalidConfiguration,
                    message: "Coordinate driving requires different start and end values".to_string(),
                    suggestion: Some("Set drive_start and drive_end to different values to define the scan range".to_string()),
                    reference: Some("Coordinate driving varies a parameter from start to end value".to_string()),
                });
            }

            if config.drive_steps == 0 {
                return Err(ValidationError {
                    category: ErrorCategory::InvalidConfiguration,
                    message: "Coordinate driving requires at least one step".to_string(),
                    suggestion: Some(
                        "Set drive_steps to a positive integer (typically 10-50)".to_string(),
                    ),
                    reference: Some(
                        "More steps give higher resolution but take longer".to_string(),
                    ),
                });
            }
        }

        _ => {
            // Other modes don't have specific additional requirements
        }
    }

    Ok(())
}

/// Provides enhanced user guidance and warnings for specific configurations.
///
/// This function prints helpful information and warnings to guide users
/// toward optimal configurations and avoid common pitfalls.
///
/// # Arguments
///
/// * `config` - The configuration to provide guidance for
pub fn provide_user_guidance(config: &Config) {
    println!("\n****Configuration Guidance****");

    // Program-specific guidance
    match config.program {
        QMProgram::Orca => {
            if config.run_mode == RunMode::InterRead {
                println!("ORCA Inter-Read Mode Guidance:");
                println!("- The inter_read mode is set for ORCA");
                println!(
                    "- Unlike Gaussian, ORCA will not automatically add guess=mix for state A"
                );
                println!("- For open-shell singlet convergence, add convergence control keywords to your tail section:");
                println!("  %scf");
                println!("    MaxIter 200");
                println!("    ConvForced true");
                println!("  end");
            }

            if config.run_mode == RunMode::Stable {
                println!("ORCA Stability Mode Limitations:");
                println!("- RI approximations are not supported in stability analysis");
                println!("- Consider using 'read' mode with pre-converged wavefunctions for RI calculations");
                println!("- UKS is recommended for singlet state calculations");
            }
        }

        QMProgram::Gaussian => {
            if config.run_mode == RunMode::InterRead {
                println!("Gaussian Inter-Read Mode:");
                println!("- Will automatically add guess=(read,mix) for state A");
                println!("- Optimal for open-shell singlet calculations");
            }
        }

        QMProgram::Xtb => {
            println!("XTB Calculation Notes:");
            println!("- XTB handles wavefunction stability internally");
            println!("- Use 'normal' or 'noread' modes for best performance");
            println!("- Method should typically be GFN1-xTB or GFN2-xTB");
        }

        _ => {}
    }

    // Run mode specific guidance
    match config.run_mode {
        RunMode::Normal => {
            println!("Normal Mode: Balanced speed and robustness with checkpoint reading");
        }
        RunMode::Read => {
            println!("Read Mode: Fast restart using existing wavefunction files");
        }
        RunMode::NoRead => {
            println!("NoRead Mode: Fresh SCF at each step - slower but more robust");
        }
        RunMode::Stable => {
            println!("Stable Mode: Includes wavefunction stability analysis");
        }
        RunMode::InterRead => {
            println!("Inter-Read Mode: Optimized for open-shell singlet calculations");
        }
        _ => {}
    }

    println!("****End Configuration Guidance****\n");
}

/// Logs mode transitions and file operations for debugging and user information.
///
/// # Arguments
///
/// * `from_mode` - The original run mode
/// * `to_mode` - The new run mode after transition
/// * `reason` - Reason for the mode transition
pub fn log_mode_transition(from_mode: RunMode, to_mode: RunMode, reason: &str) {
    if from_mode != to_mode {
        println!("****Mode Transition: {:?} -> {:?}****", from_mode, to_mode);
        println!("Reason: {}", reason);

        match (from_mode, to_mode) {
            (RunMode::Stable, RunMode::Read) => {
                println!("Stability analysis completed, switching to read mode for optimization");
            }
            (RunMode::InterRead, RunMode::Read) => {
                println!(
                    "Inter-read initialization completed, switching to read mode for optimization"
                );
            }
            _ => {}
        }

        println!("****Mode Transition Complete****\n");
    }
}

/// Validates and logs file operations for wavefunction management.
///
/// # Arguments
///
/// * `operation` - Description of the file operation
/// * `source` - Source file path
/// * `destination` - Destination file path (optional)
/// * `print_level` - Print level (0=quiet, 1=normal, 2=verbose)
pub fn log_file_operation(
    operation: &str,
    source: &str,
    destination: Option<&str>,
    print_level: u32,
) {
    // Only print file operations if print_level is 2 (verbose)
    if print_level >= 2 {
        match destination {
            Some(dest) => {
                println!("File Operation: {} - {} -> {}", operation, source, dest);
                if !Path::new(source).exists() {
                    println!("Warning: Source file '{}' does not exist", source);
                }
            }
            None => {
                println!("File Operation: {} - {}", operation, source);
                if !Path::new(source).exists() {
                    println!("Warning: File '{}' does not exist", source);
                }
            }
        }
    }
}

/// Logs file operations for debugging and validation purposes (legacy version).
///
/// This function maintains backward compatibility by defaulting to verbose output.
/// New code should use `log_file_operation` with explicit print_level.
///
/// # Arguments
///
/// * `operation` - Description of the file operation
/// * `source` - Source file path
/// * `destination` - Destination file path (optional)
#[deprecated(note = "Use log_file_operation with explicit print_level parameter")]
pub fn log_file_operation_legacy(operation: &str, source: &str, destination: Option<&str>) {
    log_file_operation(operation, source, destination, 2); // Default to verbose for backward compatibility
}
