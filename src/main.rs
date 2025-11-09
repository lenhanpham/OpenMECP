//! OpenMECP Command-Line Interface
//!
//! This module contains the main entry point for the OpenMECP program and handles
//! command-line argument parsing, help system integration, and orchestration of
//! different calculation modes.
//!
//! # Usage
//!
//! OpenMECP supports two main commands:
//!
//! 1. **Input Creation** (`omecp ci <geometry_file> [output_file]`):
//!    Creates a template input file from a geometry file
//!
//! 2. **MECP Optimization** (`omecp <input_file>`):
//!    Runs MECP optimization using the specified input file
//!
//! # Examples
//!
//! ```bash
//! # Create template input file from XYZ geometry
//! omecp ci molecule.xyz
//!
//! # Create template with custom output name
//! omecp ci molecule.xyz custom.inp
//!
//! # Run MECP optimization
//! omecp my_calculation.inp
//! ```
//!
//! # Help System
//!
//! Built-in help is available through the `--help` or `-h` flags:
//!
//! - `omecp --help` - General help
//! - `omecp --help keywords` - All input file keywords
//! - `omecp --help methods` - QM methods and programs
//! - `omecp --help features` - MECP features and modes
//! - `omecp ci --help` - Input creation help
//!
//! # Supported Geometry Formats
//!
//! - `.xyz` - XYZ coordinate files
//! - `.log` - Gaussian output files (with final geometry)
//! - `.gjf` - Gaussian input files

use nalgebra::DMatrix;
use omecp::qm_interface::get_output_file_base;
use omecp::*;
use omecp::{checkpoint, lst, validation};
use std::env;
use std::path::Path;
use std::process;

/// Prints comprehensive convergence criteria status for a single optimization step.
///
/// This function displays all 5 convergence criteria with:
/// - Current value (with appropriate units)
/// - Threshold value
/// - Pass/fail status using ✓/✗ symbols
/// - Automatic unit conversion (Bohr → Angstrom for displacements)
///
/// # Arguments
///
/// * `conv` - ConvergenceStatus struct with boolean flags for each criterion
/// * `de` - Current energy difference (Hartree)
/// * `rms_grad` - Current RMS gradient (Hartree/Bohr)
/// * `max_grad` - Current max gradient (Hartree/Bohr)
/// * `rms_disp` - Current RMS displacement (Bohr)
/// * `max_disp` - Current max displacement (Bohr)
/// * `config` - Configuration with threshold values
fn print_convergence_status(
    conv: &optimizer::ConvergenceStatus,
    de: f64,
    rms_grad: f64,
    max_grad: f64,
    rms_disp: f64,
    max_disp: f64,
    config: &config::Config,
) {
    const BOHR_TO_ANGSTROM: f64 = 0.529177;

    println!(" Criteria                            Current           Threshold        Pass");
    println!("----------------------------------------------------------------------------");

    println!(
        "  1. Energy difference             {:>12.8}      {:>12.8}       {}   ",
        de,
        config.thresholds.de,
        if conv.de_converged { "YES" } else { "NO " }
    );

    println!(
        "  2. RMS gradient                  {:>12.8}      {:>12.8}       {}   ",
        rms_grad,
        config.thresholds.rms_g,
        if conv.rms_grad_converged {
            "YES"
        } else {
            "NO "
        }
    );

    println!(
        "  3. Max gradient                  {:>12.8}      {:>12.8}       {}   ",
        max_grad,
        config.thresholds.max_g,
        if conv.max_grad_converged {
            "YES"
        } else {
            "NO "
        }
    );

    println!(
        "  4. RMS displacement              {:>12.8}      {:>12.8}       {}   ",
        rms_disp * BOHR_TO_ANGSTROM,
        config.thresholds.rms,
        if conv.rms_disp_converged {
            "YES"
        } else {
            "NO "
        }
    );

    println!(
        "  5. Max displacement              {:>12.8}      {:>12.8}       {}   ",
        max_disp * BOHR_TO_ANGSTROM,
        config.thresholds.max_dis,
        if conv.max_disp_converged {
            "YES"
        } else {
            "NO "
        }
    );

    println!("----------------------------------------------------------------------------");
    println!();
}

/// Returns the appropriate input file extension for the given QM program.
///
/// # Arguments
///
/// * `program` - The QM program type
///
/// # Returns
///
/// Returns the file extension string (without the dot) for input files.
///
/// # Examples
///
/// ```
/// use omecp::config::QMProgram;
///
/// assert_eq!(get_input_file_extension(QMProgram::Gaussian), "gjf");
/// assert_eq!(get_input_file_extension(QMProgram::Orca), "inp");
/// ```
fn get_input_file_extension(program: config::QMProgram) -> &'static str {
    match program {
        config::QMProgram::Gaussian => "gjf",
        config::QMProgram::Orca => "inp",
        config::QMProgram::Xtb => "inp",
        config::QMProgram::Bagel => "json",
        config::QMProgram::Custom => "inp",
    }
}

/// Returns the output file base name for a given QM program.
///
/// Different QM programs create output files with different extensions.
/// This function returns the base name that should be passed to the
// Use get_output_file_base from qm_interface module
/// Main entry point for OpenMECP program.
///
/// Initializes the logger, parses command-line arguments, and dispatches to the
/// appropriate calculation mode based on the command provided.
///
/// # Command-Line Arguments
///
/// - `omecp ci <geometry_file> [output_file]`: Create template input file
/// - `omecp <input_file>`: Run MECP optimization
/// - `omecp --help [topic]`: Display help information
///
/// # Errors
///
/// Exits with code 1 if:
/// - Insufficient arguments provided
/// - Invalid command specified
/// - File operations fail
/// - Calculation errors occur
///
/// # Examples
///
/// ```
/// use std::env;
/// use std::process;
///
/// fn main() {
///     // OpenMECP initialization and execution happens here
///     // See implementation below for details
/// }
/// ```
fn main() {
    // Initialize console logger for all commands
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .target(env_logger::Target::Stdout)
        .format_timestamp_millis()
        .init();

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage(&args[0]);
        process::exit(1);
    }

    // Check for help flags before processing commands
    check_help_flags(&args);

    let command = &args[1];

    match command.as_str() {
        "ci" => {
            // Create Input/Settings template command
            if args.len() < 3 {
                eprintln!("Error: Missing file argument");
                eprintln!("Usage:");
                eprintln!(
                    "  {} ci <geometry_file> [output_file]  - Create input template",
                    args[0]
                );
                eprintln!(
                    "  {} ci omecp_config.cfg               - Create settings template",
                    args[0]
                );
                process::exit(1);
            }

            let file_arg = &args[2];

            // Check if user wants to create settings template
            if file_arg == "omecp_config.cfg" {
                match run_create_settings_template() {
                    Ok(()) => {
                        println!("✓ Settings template created successfully!");
                        println!("  Output file: omecp_config.cfg");
                        println!("\nNext steps:");
                        println!("  1. Review and edit the omecp_config.cfg file");
                        println!("  2. Customize file extensions and other parameters as needed");
                        println!("  3. The settings will be automatically loaded by OpenMECP");
                    }
                    Err(e) => {
                        eprintln!("Error creating settings template: {}", e);
                        process::exit(1);
                    }
                }
            } else {
                // Original input template creation
                let geometry_path = Path::new(file_arg);
                let output_path = args.get(3).map(Path::new);

                match run_create_input(geometry_path, output_path) {
                    Ok(output_file) => {
                        println!("✓ Template input file created successfully!");
                        println!("  Output file: {}", output_file.display());
                        println!("\nNext steps:");
                        println!("  1. Review and edit the generated template file");
                        println!("  2. Update parameters (method, basis set, etc.) as needed");
                        println!("  3. Run OpenMECP: {} {}", args[0], output_file.display());
                    }
                    Err(e) => {
                        eprintln!("Error creating template: {}", e);
                        process::exit(1);
                    }
                }
            }
        }
        _ => {
            // Default OpenMECP optimization command
            if !command.starts_with('-') {
                let input_path = Path::new(&args[1]);
                match run_mecp(input_path) {
                    Ok(()) => println!("\n****Congrats! MECP has converged****"),
                    Err(e) => {
                        eprintln!("Error: {}", e);
                        process::exit(1);
                    }
                }
            } else {
                eprintln!("Error: Unknown command: {}", command);
                print_usage(&args[0]);
                process::exit(1);
            }
        }
    }
}

/// Check for help flags and print appropriate help
fn check_help_flags(args: &[String]) {
    use omecp::help::*;

    // Check for global help flags
    if args.len() >= 3 && (args[1] == "--help" || args[1] == "-h") {
        let topic = &args[2];

        match topic.as_str() {
            "keywords" => {
                print_keyword_help();
                process::exit(0);
            }
            "methods" => {
                print_method_help();
                process::exit(0);
            }
            "features" => {
                print_feature_help();
                process::exit(0);
            }
            "examples" => {
                print_examples();
                process::exit(0);
            }
            _ => {
                print_global_help();
                process::exit(0);
            }
        }
    }

    // Check for help without topic (omecp --help or omecp -h)
    if args.len() == 2 && (args[1] == "--help" || args[1] == "-h") {
        print_global_help();
        process::exit(0);
    }

    // Check for command-specific help (omecp ci --help)
    if args.len() >= 3 && args[1] == "ci" && (args[2] == "--help" || args[2] == "-h") {
        print_ci_help();
        process::exit(0);
    }

    // Check for help in MECP mode (omecp input.inp --help)
    if args.len() >= 3 && !args[1].starts_with('-') && (args[2] == "--help" || args[2] == "-h") {
        print_mecp_help(&args[1]);
        process::exit(0);
    }
}

/// Print help for MECP optimization mode
fn print_mecp_help(input_file: &str) {
    println!("MECP Optimization - Input File Help");
    println!("═══════════════════════════════════════");
    println!();
    println!("Input file: {}", input_file);
    println!();
    println!("This file contains all parameters and settings for the MECP optimization.");
    println!("For complete documentation, use:");
    println!();
    println!("  omecp --help keywords   # All input file keywords");
    println!("  omecp --help methods    # QM methods and programs");
    println!("  omecp --help features   # MECP features and modes");
    println!("  omecp --help examples   # Usage examples");
    println!();
}

/// Prints usage information to stderr.
///
/// # Arguments
///
/// * `program_name` - The name/path of the program binary (typically argv[0])
///
/// # Examples
///
/// ```
/// print_usage("omecp");
/// // Prints:
/// // OpenMECP - Minimum Energy Crossing Point optimization
/// // Usage:
/// //   omecp ci <geometry_file> [output_file]
/// //                         Create a template input file from a geometry file
/// //   ...
/// ```
fn print_usage(program_name: &str) {
    eprintln!("OpenMECP - Minimum Energy Crossing Point optimization");
    eprintln!("--------------Developed by Le Nhan Pham--------------");
    eprintln!("           https://github.com/lenhanpham");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  {} ci <geometry_file> [output_file]", program_name);
    eprintln!("                    Create a template input file from a geometry file");
    eprintln!();
    eprintln!("  {} ci omecp_config.cfg", program_name);
    eprintln!("                    Create a settings template file for configuration");
    eprintln!();
    eprintln!("  {} <input_file>", program_name);
    eprintln!("                    Run MECP optimization using the input file");
    eprintln!();
    eprintln!("Supported geometry formats:");
    eprintln!("  .xyz  - XYZ coordinate file");
    eprintln!("  .log  - Gaussian output file");
    eprintln!("  .gjf  - Gaussian input file");
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  {} ci molecule.xyz", program_name);
    eprintln!("  {} ci molecule.xyz custom.inp", program_name);
    eprintln!("  {} ci omecp_config.cfg", program_name);
    eprintln!("  {} my_calculation.inp", program_name);
}

/// Creates a template input file from a geometry file.
///
/// This function reads a geometry from the provided file and generates a template
/// OpenMECP input file with placeholders for all required parameters. The template
/// includes the geometry section, standard configuration parameters, and optional
/// sections for constraints and advanced features.
///
/// # Arguments
///
/// * `geometry_file` - Path to the input geometry file (.xyz, .log, or .gjf)
/// * `output_path` - Optional custom output path; if None, uses default naming
///
/// # Returns
///
/// Returns a `Result` containing:
/// - `Ok(PathBuf)` - Path to the created template file
/// - `Err(Box<dyn Error>)` - Error details for any failure
///
/// # Supported Input Formats
///
/// - **.xyz**: Standard XYZ coordinate format
/// - **.log**: Gaussian output file with final geometry
/// - **.gjf**: Gaussian input file
///
/// # Examples
///
/// ```
/// use std::path::Path;
///
/// // Create default template
/// let output = run_create_input(Path::new("molecule.xyz"), None)?;
/// println!("Template created: {}", output.display());
///
/// // Create with custom name
/// let output = run_create_input(
///     Path::new("molecule.xyz"),
///     Some(Path::new("custom.inp"))
/// )?;
/// ```
///
/// # Errors
///
/// Returns an error if:
/// - Geometry file does not exist or cannot be read
/// - File format is not supported
/// - Template generation fails
/// - Output file cannot be written
fn run_create_input<P: AsRef<Path>>(
    geometry_file: P,
    output_path: Option<P>,
) -> Result<std::path::PathBuf, Box<dyn std::error::Error>>
{
    use omecp::template_generator::*;

    let geometry_file = geometry_file.as_ref();
    let geometry_path = geometry_file.canonicalize()?;

    // Validate file exists and has supported extension
    if !geometry_path.exists() {
        return Err(format!("Geometry file not found: {}", geometry_path.display()).into());
    }

    if !is_supported_format(&geometry_path) {
        return Err("Unsupported file format. Supported formats: .xyz, .log, .gjf".to_string().into());
    }

    // Extract geometry to validate the file
    println!("Reading geometry from: {}", geometry_path.display());
    let template_content = generate_template_from_file(&geometry_path)?;

    // Determine output path
    let output_path = match output_path {
        Some(p) => p.as_ref().to_path_buf(),
        None => get_default_output_path(&geometry_path),
    };

    // Write template to file
    write_template_to_file(&template_content, &output_path)?;

    Ok(output_path)
}

/// Creates a omecp_config.cfg template file with all available configuration options.
///
/// This function generates a comprehensive configuration file template that includes:
/// - All available configuration sections (extensions, general, logging)
/// - Default values for each parameter with explanations
/// - Detailed comments explaining each option
/// - Examples of common customizations
///
/// The template is created in the current working directory as 'omecp_config.cfg'.
///
/// # Returns
///
/// Returns a `Result` indicating success or failure of template creation.
///
/// # Errors
///
/// Returns an error if:
/// - The omecp_config.cfg file already exists (to prevent accidental overwrite)
/// - File cannot be written due to permissions or disk space
/// - Template generation fails
fn run_create_settings_template() -> Result<(), Box<dyn std::error::Error>> {
    use omecp::settings::SettingsManager;

    let settings_path = Path::new("omecp_config.cfg");

    // Check if file already exists to prevent accidental overwrite
    if settings_path.exists() {
        return Err(
            "omecp_config.cfg already exists. Please remove it first or choose a different location."
                .into(),
        );
    }

    // Create the settings template
    SettingsManager::create_template(settings_path)?;

    Ok(())
}

/// Runs MECP optimization using the specified input file.
///
/// This is the main optimization loop that orchestrates the entire MECP calculation.
/// It handles:
/// - Input file parsing and validation
/// - QM program interface initialization
/// - Initial calculation execution
/// - Iterative optimization with BFGS/GDIIS/GEDIIS
/// - Convergence checking and checkpointing
/// - Multiple run modes (normal, restart, LST, PES scan, etc.)
///
/// # Arguments
///
/// * `input_path` - Path to the OpenMECP input file
///
/// # Returns
///
/// Returns a `Result` containing:
/// - `Ok(())` - Optimization completed successfully
/// - `Err(Box<dyn Error>)` - Error details for any failure
///
/// # Optimization Process
///
/// 1. **Initialization**: Parse input file and create QM interface
/// 2. **Initial Calculations**: Run single-point calculations for both states
/// 3. **Main Loop**: For each optimization step:
///    - Compute MECP gradient
///    - Choose optimizer (BFGS → GDIIS/GEDIIS)
///    - Take optimization step
///    - Run new calculations
///    - Check convergence
///    - Update Hessian and history
///    - Save checkpoint
/// 4. **Completion**: Save final geometry to `final.xyz`
///
/// # Run Modes
///
/// - **Normal**: Standard MECP optimization
/// - **Restart**: Continue from checkpoint file
/// - **LST**: Linear synchronous transit interpolation
/// - **PES Scan**: 1D/2D potential energy surface scans
/// - **Coordinate Drive**: Single coordinate driving
/// - **Path Optimization**: Nudged elastic band optimization
/// - **Fix-dE**: Optimize at fixed energy difference
///
/// # Errors
///
/// Returns an error if:
/// - Input file cannot be parsed
/// - QM calculations fail
/// - Optimization diverges
/// - Maximum steps exceeded
/// - File I/O errors occur
///
/// Prints all configuration parameters and settings information to the output.
/// This function displays:
///   1. Settings file location and source (if loaded)
///   2. All input configuration parameters
///   3. Settings from omecp_config.cfg
///   4. Debug log file information (if file logging is enabled)
///
/// This helps users understand what parameters are being used and where they come from.
fn print_configuration(
    input_config: &config::Config,
    settings_manager: &Option<omecp::settings::SettingsManager>,
    debug_log_file: Option<&str>,
) {
    println!("{}", "=".repeat(76));
    println!("CONFIGURATION AND SETTINGS");
    println!("{}", "=".repeat(76));
    println!();

    // Print settings file information if loaded
    if let Some(ref settings) = settings_manager {
        println!("Settings Configuration:");
        println!("  Source: {}", settings.config_source());
        println!();
    }

    // Print input file configuration parameters
    println!("Input File Parameters:");
    println!("  Program:                    {:?}", input_config.program);
    println!("  Method:                     {}", input_config.method);
    println!("  Memory:                     {}", input_config.mem);
    println!("  Processors:                 {}", input_config.nprocs);
    println!("  Charge (State 1):           {}", input_config.charge1);
    println!("  Charge (State 2):           {}", input_config.charge2);
    println!("  Multiplicity (State 1):     {}", input_config.mult1);
    println!("  Multiplicity (State 2):     {}", input_config.mult2);
    println!("  Run Mode:                   {:?}", input_config.run_mode);

    if !input_config.td1.is_empty() {
        println!("  TD-DFT (State 1):           {}", input_config.td1);
    }
    if !input_config.td2.is_empty() {
        println!("  TD-DFT (State 2):           {}", input_config.td2);
    }

    println!("  Max Steps:                  {}", input_config.max_steps);
    println!(
        "  Max Step Size (Bohr):       {}",
        input_config.max_step_size
    );
    println!("  Use GEDIIS:                 {}", input_config.use_gediis);
    println!("  Switch Step:                {}", input_config.switch_step);
    println!("  Restart Mode:               {}", input_config.restart);

    if !input_config.bagel_model.is_empty() {
        println!("  BAGEL Model:                {}", input_config.bagel_model);
    }

    if !input_config.custom_interface_file.is_empty() {
        println!(
            "  Custom Interface File:      {}",
            input_config.custom_interface_file
        );
    }

    if input_config.state1 > 0 || input_config.state2 > 0 {
        println!("  TD-DFT State (State 1):     {}", input_config.state1);
        println!("  TD-DFT State (State 2):     {}", input_config.state2);
    }

    if input_config.is_oniom {
        println!(
            "  ONIOM Layer Info:           {:?}",
            input_config.oniom_layer_info.join(",")
        );
        println!(
            "  ONIOM Charge/Mult 1:        {}",
            input_config.charge_and_mult_oniom1
        );
        println!(
            "  ONIOM Charge/Mult 2:        {}",
            input_config.charge_and_mult_oniom2
        );
    }

    if !input_config.drive_type.is_empty() {
        println!("  Drive Type:                 {}", input_config.drive_type);
        println!(
            "  Drive Atoms:                {:?}",
            input_config.drive_atoms
        );
        println!("  Drive Start:                {}", input_config.drive_start);
        println!("  Drive End:                  {}", input_config.drive_end);
        println!("  Drive Steps:                {}", input_config.drive_steps);
    }

    // Print thresholds
    println!("\nConvergence Thresholds:");
    println!(
        "  Energy Difference (ΔE):     {:>12.8} hartree",
        input_config.thresholds.de
    );
    println!(
        "  RMS Gradient:               {:>12.8} hartree/bohr",
        input_config.thresholds.rms_g
    );
    println!(
        "  Max Gradient:               {:>12.8} hartree/bohr",
        input_config.thresholds.max_g
    );
    println!(
        "  RMS Displacement:           {:>12.8} bohr",
        input_config.thresholds.rms
    );
    println!(
        "  Max Displacement:           {:>12.8} bohr",
        input_config.thresholds.max_dis
    );

    // Print settings from omecp_config.cfg if loaded
    if let Some(ref settings) = settings_manager {
        println!("\nConfiguration File Settings (omecp_config.cfg):");

        // Print file extensions
        println!("  Output File Extensions:");
        println!(
            "    Gaussian:                  {}",
            settings.extensions().gaussian
        );
        println!(
            "    ORCA:                      {}",
            settings.extensions().orca
        );
        //println!("    XTB:                       {}", settings.extensions().xtb);
        //println!("    BAGEL:                     {}", settings.extensions().bagel);
        //println!("    Custom:                    {}", settings.extensions().custom);

        // Print general settings
        println!("  General Settings:");
        println!(
            "    Max Memory:                {}",
            settings.general().max_memory
        );
        println!(
            "    Default Processors:        {}",
            settings.general().default_nprocs
        );
        println!(
            "    Print Level:               {}",
            settings.general().print_level
        );

        // Print cleanup settings
        println!("  Cleanup Settings:");
        println!(
            "    Enabled:                   {}",
            settings.cleanup().enabled
        );
        println!(
            "    Verbose:                   {}",
            settings.cleanup().verbose
        );
        println!(
            "    Cleanup Frequency:         {} steps",
            settings.cleanup().cleanup_frequency
        );
        if !settings.cleanup().preserve_extensions.is_empty() {
            println!(
                "    Preserve Extensions:       {:?}",
                settings.cleanup().preserve_extensions
            );
        }

        // Print logging settings
        println!("  Logging Settings:");
        println!(
            "    Level:                     {}",
            settings.logging().level
        );
        println!(
            "    File Logging Enabled:      {}",
            settings.logging().file_logging
        );
        if let Some(log_file) = debug_log_file {
            println!("    Debug Log File:            {}", log_file);
        }
    }

    println!();
    println!("{}", "=".repeat(80));
    println!();
}

fn run_mecp(input_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("**** OpenMECP: Minimum Energy Crossing Point Optimizer****");
    println!("              Version {}  Release date: 2025", env!("CARGO_PKG_VERSION"));
    println!("               ****Developer Le Nhan Pham****             ");
    println!("           https://github.com/lenhanpham/OpenMECP        \n");

    // Parse input
    let input_data = parser::parse_input(input_path)?;

    // Load settings (for print_level, cleanup, and parameter display)
    let settings_manager = omecp::settings::SettingsManager::load().ok();

    let print_level = if let Some(settings) = settings_manager.as_ref() {
        settings.general().print_level
    } else {
        // If settings can't be loaded, use default (quiet mode)
        0
    };

    // Generate debug log filename if file logging is enabled
    let debug_log_file = if let Some(settings) = settings_manager.as_ref() {
        if settings.logging().file_logging {
            let input_stem = input_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("job");
            Some(format!("omecp_debug_{}.log", input_stem))
        } else {
            None
        }
    } else {
        None
    };

    // Set up file-based logging if enabled in settings
    if let Some(settings) = settings_manager.as_ref() {
        if let Some(log_file) = debug_log_file.as_ref() {
            if settings.logging().file_logging {
                // Write a startup message to the debug log file
                use std::io::Write;
                let mut file = std::fs::File::create(log_file)
                    .map_err(|e| format!("Failed to create log file {}: {}", log_file, e))?;
                writeln!(file, "OpenMECP debug log started")
                    .map_err(|e| format!("Failed to write to log file: {}", e))?;
            }
        }
    }

    // Print all configuration and settings information
    print_configuration(
        &input_data.config,
        &settings_manager,
        debug_log_file.as_deref(),
    );

    // Extract directory name from input file (e.g., "compound_x.inp" -> "compound_x")
    let job_dir = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("mecp_job");

    println!("Parsed {} atoms", input_data.geometry.num_atoms);
    println!(
        "Program: {:?}, Mode: {:?}",
        input_data.config.program, input_data.config.run_mode
    );
    println!();

    // Validate run mode compatibility (Task 7.1)
    if let Err(e) = validation::validate_run_mode_compatibility(&input_data.config) {
        eprintln!("Configuration Error: {}", e);
        return Err(e.into());
    }

    // Provide user guidance and warnings (Task 7.2)
    validation::provide_user_guidance(&input_data.config);

    // Create cleanup manager if settings were loaded
    let cleanup_manager = if let Some(settings) = settings_manager.as_ref() {
        omecp::cleanup::CleanupManager::new(
            omecp::cleanup::CleanupConfig::from_settings_manager(
                settings,
                input_data.config.program,
            ),
            input_data.config.program,
        )
    } else {
        omecp::cleanup::CleanupManager::new(
            omecp::cleanup::CleanupConfig::default(),
            input_data.config.program,
        )
    };

    // Create QM interface
    let qm: Box<dyn qm_interface::QMInterface> = match input_data.config.program {
        config::QMProgram::Gaussian => {
            let command = input_data
                .config
                .program_commands
                .get("gaussian")
                .cloned()
                .unwrap_or_else(|| "g16".to_string());
            Box::new(qm_interface::GaussianInterface::new(
                command,
                input_data.config.mp2,
            ))
        }
        config::QMProgram::Orca => {
            let command = input_data
                .config
                .program_commands
                .get("orca")
                .cloned()
                .unwrap_or_else(|| "orca".to_string());
            Box::new(qm_interface::OrcaInterface::new(command))
        }
        config::QMProgram::Bagel => {
            let command = input_data
                .config
                .program_commands
                .get("bagel")
                .cloned()
                .unwrap_or_else(|| "bagel".to_string());
            // Read BAGEL model template
            let model_path = &input_data.config.bagel_model;
            if model_path.is_empty() {
                return Err("BAGEL requires bagel_model parameter".into());
            }
            let model_template = std::fs::read_to_string(model_path)
                .map_err(|e| format!("Failed to read BAGEL model: {}", e))?;
            Box::new(qm_interface::BagelInterface::new(command, model_template))
        }
        config::QMProgram::Xtb => {
            let command = input_data
                .config
                .program_commands
                .get("xtb")
                .cloned()
                .unwrap_or_else(|| "xtb".to_string());
            Box::new(qm_interface::XtbInterface::new(command))
        }
        config::QMProgram::Custom => {
            if input_data.config.custom_interface_file.is_empty() {
                return Err("Custom QM program requires custom_interface_file parameter".into());
            }
            let interface_path = Path::new(&input_data.config.custom_interface_file);
            Box::new(qm_interface::CustomInterface::from_file(interface_path)?)
        }
    };

    // Create job directory
    std::fs::create_dir_all(job_dir)?;

    // Check if restart is requested
    if input_data.config.restart {
        return run_restart(input_data, &*qm, job_dir);
    }

    // Check if LST interpolation is requested
    if input_data.lst1.is_some() && input_data.lst2.is_some() {
        return run_lst_interpolation(input_data, &*qm, job_dir);
    }

    // Check if PES scan is requested
    if !input_data.config.scans.is_empty() {
        return run_pes_scan(input_data, &*qm, job_dir);
    }

    // Check if coordinate driving is requested
    if input_data.config.run_mode == config::RunMode::CoordinateDrive {
        return run_coordinate_driving(input_data, &*qm, job_dir);
    }

    // Check if path optimization is requested
    if input_data.config.run_mode == config::RunMode::PathOptimization {
        return run_path_optimization(input_data, &*qm);
    }

    // Check if fix-dE optimization is requested
    if input_data.config.run_mode == config::RunMode::FixDE {
        return run_fix_de(input_data, &*qm);
    }

    // Build headers
    let header_a = io::build_program_header_with_basename(
        &input_data.config,
        input_data.config.charge1,
        input_data.config.mult1,
        &input_data.config.td1,
        input_data.config.state1,
        job_dir,
    );

    let header_b = io::build_program_header_with_basename(
        &input_data.config,
        input_data.config.charge2,
        input_data.config.mult2,
        &input_data.config.td2,
        input_data.config.state2,
        job_dir,
    );

    // Run pre-point calculations for stable/inter_read modes
    let original_run_mode = input_data.config.run_mode;
    let mut config = input_data.config.clone();
    let mut header_a = header_a;
    let mut header_b = header_b;

    if original_run_mode == config::RunMode::Normal {
        // Normal Mode: Phase 1 - Pre-point calculations (following Python MECP.py logic)
        println!(
            "****Normal Mode: Phase 1 - Pre-point calculations to generate checkpoint files****"
        );

        // Build RAW headers WITHOUT any modifications (matching Python buildHeader)
        // This means NO force, NO guess=read, just the basic method
        let pre_header_a = build_raw_program_header(
            &input_data.config,
            input_data.config.charge1,
            input_data.config.mult1,
            &input_data.config.td1,
            input_data.config.state1,
            "state_A.chk",
        );
        let pre_header_b = build_raw_program_header(
            &input_data.config,
            input_data.config.charge2,
            input_data.config.mult2,
            &input_data.config.td2,
            input_data.config.state2,
            "state_B.chk",
        );

        println!("Pre-point headers (raw, no modifications):");
        println!("State A: {}", pre_header_a.lines().nth(2).unwrap_or(""));
        println!("State B: {}", pre_header_b.lines().nth(2).unwrap_or(""));

        // Write and run pre-point calculations
        let ext = get_input_file_extension(input_data.config.program);
        let pre_a_path = format!("{}/pre_A.{}", job_dir, ext);
        let pre_b_path = format!("{}/pre_B.{}", job_dir, ext);

        qm.write_input(
            &input_data.geometry,
            &pre_header_a,
            &input_data.tail1,
            Path::new(&pre_a_path),
        )?;
        qm.write_input(
            &input_data.geometry,
            &pre_header_b,
            &input_data.tail2,
            Path::new(&pre_b_path),
        )?;

        println!("Running pre-point calculation for state B...");
        let pre_b_result = qm.run_calculation(Path::new(&pre_b_path));
        if let Err(e) = &pre_b_result {
            println!("  Warning: Pre-point calculation for state B failed: {}", e);
            println!("  This is expected if the QM program is not installed");
        }

        println!("Running pre-point calculation for state A...");
        let pre_a_result = qm.run_calculation(Path::new(&pre_a_path));
        if let Err(e) = &pre_a_result {
            println!(" Warning: Pre-point calculation for state A failed: {}", e);
            println!("  This is expected if the QM program is not installed");
        }

        // Check if both calculations succeeded
        let pre_calculations_successful = pre_a_result.is_ok() && pre_b_result.is_ok();
        if pre_calculations_successful {
            println!("✓ Pre-point calculations completed successfully");
        } else {
            println!("  Pre-point calculations failed - continuing without checkpoint files");
        }

        // Check checkpoint files exactly like Python MECP.py
        match input_data.config.program {
            config::QMProgram::Gaussian => {
                println!("Checking Gaussian checkpoint files...");
                // Gaussian creates state_A.chk and state_B.chk in root directory (current working directory)
                let state_a_chk = "state_A.chk";
                let state_b_chk = "state_B.chk";

                if Path::new(state_a_chk).exists() && Path::new(state_b_chk).exists() {
                    println!(
                        "✓ Gaussian checkpoint files found: {} and {}",
                        state_a_chk, state_b_chk
                    );
                } else {
                    return Err(format!(
                        "Error: Gaussian checkpoint files not found after pre-point calculations.\n\
                         Expected: {} and {} in current directory.\n\
                         Pre-point calculations may have failed. Please check the calculation setup.",
                        state_a_chk, state_b_chk
                    ).into());
                }
            }
            config::QMProgram::Orca => {
                if print_level >= 1 {
                    println!("Renaming ORCA wavefunction files...");
                }
                let pre_a_gbw = format!("{}/pre_A.gbw", job_dir);
                let pre_b_gbw = format!("{}/pre_B.gbw", job_dir);
                let state_a_gbw = format!("{}/state_A.gbw", job_dir);
                let state_b_gbw = format!("{}/state_B.gbw", job_dir);

                // Rename ORCA files (more efficient than copying)
                if Path::new(&pre_a_gbw).exists() && Path::new(&pre_b_gbw).exists() {
                    // Remove existing destination files if they exist
                    if Path::new(&state_a_gbw).exists() {
                        validation::log_file_operation("Delete", &state_a_gbw, None, print_level);
                        std::fs::remove_file(&state_a_gbw)?;
                    }
                    if Path::new(&state_b_gbw).exists() {
                        validation::log_file_operation("Delete", &state_b_gbw, None, print_level);
                        std::fs::remove_file(&state_b_gbw)?;
                    }

                    // Rename the files
                    validation::log_file_operation(
                        "Rename",
                        &pre_a_gbw,
                        Some(&state_a_gbw),
                        print_level,
                    );
                    std::fs::rename(&pre_a_gbw, &state_a_gbw)?;
                    validation::log_file_operation(
                        "Rename",
                        &pre_b_gbw,
                        Some(&state_b_gbw),
                        print_level,
                    );
                    std::fs::rename(&pre_b_gbw, &state_b_gbw)?;

                    if print_level >= 1 {
                        println!(
                            "✓ ORCA wavefunction files renamed: {} -> {}, {} -> {}",
                            pre_a_gbw, state_a_gbw, pre_b_gbw, state_b_gbw
                        );
                    }
                } else {
                    return Err(format!(
                        "Error: ORCA wavefunction files not found after pre-point calculations.\n\
                         Expected: {} and {}\n\
                         Pre-point calculations may have failed. Please check the calculation setup.",
                        pre_a_gbw, pre_b_gbw
                    ).into());
                }
            }
            config::QMProgram::Xtb => {
                println!("XTB pre-point calculations completed");
                println!(
                    "✓ XTB doesn't require checkpoint files - ready for main optimization loop"
                );
            }
            _ => {
                println!(
                    "Pre-point calculations completed for {:?}",
                    input_data.config.program
                );
            }
        }

        println!("****Normal Mode: Phase 2 - Main optimization loop with checkpoint reading****");

        // If we reach here, checkpoint files exist, so switch to read mode like Python MECP.py
        validation::log_mode_transition(
            original_run_mode,
            config::RunMode::Read,
            "Pre-point calculations completed successfully, switching to read mode for main optimization"
        );
        config.run_mode = config::RunMode::Read;

        // Rebuild headers with read mode (includes force + guess=read)
        header_a = io::build_program_header_with_chk(
            &config,
            config.charge1,
            config.mult1,
            &config.td1,
            config.state1,
            Some("state_A.chk"),
            Some(job_dir),
        );

        header_b = io::build_program_header_with_chk(
            &config,
            config.charge2,
            config.mult2,
            &config.td2,
            config.state2,
            Some("state_B.chk"),
            Some(job_dir),
        );

        println!("Headers rebuilt for read mode with force and guess=read");
    } else if original_run_mode == config::RunMode::Stable
        || original_run_mode == config::RunMode::InterRead
    {
        run_pre_point(
            &input_data.geometry,
            &header_a,
            &header_b,
            &input_data,
            &*qm,
            original_run_mode,
            job_dir,
        )?;

        // CRITICAL: Switch to read mode after pre-point (following Python MECP.py logic)
        validation::log_mode_transition(
            original_run_mode,
            config::RunMode::Read,
            "Pre-point calculations completed, switching to read mode for main optimization",
        );
        config.run_mode = config::RunMode::Read;

        // Rebuild headers with new run mode (read mode)
        header_a = io::build_program_header_with_basename(
            &config,
            config.charge1,
            config.mult1,
            &config.td1,
            config.state1,
            job_dir,
        );

        header_b = io::build_program_header_with_basename(
            &config,
            config.charge2,
            config.mult2,
            &config.td2,
            config.state2,
            job_dir,
        );

        println!("****Headers rebuilt for read mode****");

        // Handle stability mode post-processing (following Python MECP.py logic)
        if original_run_mode == config::RunMode::Stable {
            match input_data.config.program {
                config::QMProgram::Orca => {
                    println!("In an RHF calculation in ORCA, it will not restart automatically if an instability is found.");
                    println!("Remember to write UKS when you are handling singlet state!");
                    println!("RI is unsupported for stability analysis. It is recommended to MANUALLY obtain the correct wavefunction,");
                    println!("and then use the read model of OpenMECP, rather than the stable mode, in order to use RI.");

                    // Copy wavefunction files for subsequent calculations
                    let pre_a_gbw = format!("{}/pre_A.gbw", job_dir);
                    let pre_b_gbw = format!("{}/pre_B.gbw", job_dir);
                    let state_a_gbw = format!("{}/state_A.gbw", job_dir);
                    let state_b_gbw = format!("{}/state_B.gbw", job_dir);

                    if Path::new(&pre_a_gbw).exists() {
                        std::fs::copy(&pre_a_gbw, &state_a_gbw)?;
                    }
                    if Path::new(&pre_b_gbw).exists() {
                        std::fs::copy(&pre_b_gbw, &state_b_gbw)?;
                    }
                }
                _ => {
                    // Gaussian handles stability automatically
                    println!("Stability analysis completed. Continuing with read mode.");
                }
            }
        }
    } else {
        config = input_data.config;
    }

    let mut geometry = input_data.geometry;
    let constraints = &input_data.constraints;
    let fixed_atoms = &input_data.fixed_atoms;

    // Run initial calculations
    println!("\n****Running initial calculations****");
    let ext = get_input_file_extension(config.program);
    let initial_a_path = format!("{}/0_state_A.{}", job_dir, ext);
    let initial_b_path = format!("{}/0_state_B.{}", job_dir, ext);

    qm.write_input(
        &geometry,
        &header_a,
        &input_data.tail1,
        Path::new(&initial_a_path),
    )?;
    qm.write_input(
        &geometry,
        &header_b,
        &input_data.tail2,
        Path::new(&initial_b_path),
    )?;

    qm.run_calculation(Path::new(&initial_a_path))?;
    qm.run_calculation(Path::new(&initial_b_path))?;

    let output_ext = get_output_file_base(config.program);
    let initial_a_output = format!("{}/0_state_A.{}", job_dir, output_ext);
    let initial_b_output = format!("{}/0_state_B.{}", job_dir, output_ext);
    let state1 = qm.read_output(Path::new(&initial_a_output), config.state1)?;
    let state2 = qm.read_output(Path::new(&initial_b_output), config.state2)?;

    // FIX: Synchronize geometry from pre-point calculation
    geometry.coords = state1.geometry.coords.clone();

    // Initialize optimization
    let mut opt_state = optimizer::OptimizationState::new();
    let mut x_old = geometry.coords.clone();
    let mut hessian = DMatrix::identity(geometry.coords.len(), geometry.coords.len());

    // Main optimization loop
    for step in 0..config.max_steps {
        println!("\n****Step {}****", step + 1);

        // Compute MECP gradient
        let mut grad = optimizer::compute_mecp_gradient(&state1, &state2, fixed_atoms);

        // Apply constraint forces if constraints are present
        if !constraints.is_empty() {
            println!("Applying constraint forces using Lagrange multipliers");
            match constraints::add_constraint_lagrange(
                &geometry,
                grad.clone(),
                constraints,
                &mut opt_state.lambdas,
            ) {
                Ok(constrained_grad) => {
                    grad = constrained_grad;
                    println!("Constraint forces applied successfully");

                    // Report constraint status
                    constraints::report_constraint_status(
                        &geometry,
                        constraints,
                        &opt_state.lambdas,
                        step + 1,
                    );
                }
                Err(e) => {
                    println!("Warning: Failed to apply constraint forces: {}. Using unconstrained gradient.", e);
                }
            }
        }

        // Choose optimizer based on switch_step configuration
        let use_bfgs = if config.switch_step >= config.max_steps {
            // Always BFGS if switch_step >= max_steps (BFGS-only mode)
            true
        } else if config.switch_step == 0 {
            // Never BFGS if switch_step = 0 (DIIS-only mode)
            false
        } else {
            // Use BFGS until switch_step, then switch to DIIS
            step < config.switch_step
        };

        let x_new = if use_bfgs || !opt_state.has_enough_history() {
            if config.switch_step >= config.max_steps {
                println!("Using BFGS optimizer (BFGS-only mode)");
            } else {
                println!(
                    "Using BFGS optimizer (step {} < switch point {})",
                    step + 1,
                    config.switch_step
                );
            }
            let adaptive_scale = if step == 0 {
                1.0 // First step, no previous energy to compare
            } else {
                let energy_current = state1.energy - state2.energy;
                let energy_previous = opt_state.energy_history.back().unwrap_or(&energy_current);
                optimizer::compute_adaptive_scale(energy_current, *energy_previous, grad.norm(), step)
            };
            optimizer::bfgs_step(&x_old, &grad, &hessian, &config, adaptive_scale)
        } else if config.use_gediis {
            println!(
                "Using GEDIIS optimizer (step {} >= switch point {})",
                step + 1,
                config.switch_step
            );
            optimizer::gediis_step(&opt_state, &config)
        } else {
            println!(
                "Using GDIIS optimizer (step {} >= switch point {})",
                step + 1,
                config.switch_step
            );
            optimizer::gdiis_step(&opt_state, &config)
        };

        // Update geometry
        geometry.coords = x_new.clone();

        // Run calculations based on program type (following Python MECP.py runEachStep logic)
        match config.program {
            config::QMProgram::Gaussian | config::QMProgram::Orca | config::QMProgram::Custom => {
                // Standard Gaussian/ORCA workflow
                let ext = get_input_file_extension(config.program);
                let step_name_a = format!("{}/{}_state_A.{}", job_dir, step + 1, ext);
                let step_name_b = format!("{}/{}_state_B.{}", job_dir, step + 1, ext);

                qm.write_input(
                    &geometry,
                    &header_a,
                    &input_data.tail1,
                    Path::new(&step_name_a),
                )?;
                qm.write_input(
                    &geometry,
                    &header_b,
                    &input_data.tail2,
                    Path::new(&step_name_b),
                )?;

                // Run calculations sequentially
                qm.run_calculation(Path::new(&step_name_a))?;
                qm.run_calculation(Path::new(&step_name_b))?;
            }
            config::QMProgram::Xtb => {
                // XTB-specific workflow
                run_xtb_step(
                    &geometry,
                    step + 1,
                    &header_a,
                    &header_b,
                    &input_data.tail1,
                    &input_data.tail2,
                    qm.as_ref(),
                )?;
            }
            config::QMProgram::Bagel => {
                // BAGEL-specific workflow
                run_bagel_step(
                    &geometry,
                    step + 1,
                    &config,
                    &geometry.elements,
                    qm.as_ref(),
                )?;
            }
        }

        // Read output files based on program type
        let output_ext = get_output_file_base(config.program);
        let state1_new = qm.read_output(
            Path::new(&format!("{}/{}_state_A.{}", job_dir, step + 1, output_ext)),
            config.state1,
        )?;
        let state2_new = qm.read_output(
            Path::new(&format!("{}/{}_state_B.{}", job_dir, step + 1, output_ext)),
            config.state2,
        )?;

        // CRITICAL FIX: Update main geometry with the actual geometry from QM output
        geometry.coords = state1_new.geometry.coords.clone();

        // Manage ORCA wavefunction files (following Python MECP.py logic)
        manage_orca_wavefunction_files(step + 1, &config, &geometry, job_dir)?;

        // Compute new gradient for Hessian update
        let mut grad_new = optimizer::compute_mecp_gradient(&state1_new, &state2_new, fixed_atoms);

        // Apply constraint forces to new gradient as well
        if !constraints.is_empty() {
            match constraints::add_constraint_lagrange(
                &geometry,
                grad_new.clone(),
                constraints,
                &mut opt_state.lambdas,
            ) {
                Ok(constrained_grad_new) => {
                    grad_new = constrained_grad_new;
                }
                Err(e) => {
                    println!(
                        "Warning: Failed to apply constraint forces to new gradient: {}",
                        e
                    );
                }
            }
        }

        // Check convergence
        let conv = optimizer::check_convergence(
            state1_new.energy,
            state2_new.energy,
            &x_old,
            &x_new,
            &grad_new,
            &config,
        );

        // Compute current values for display
        let de = (state1_new.energy - state2_new.energy).abs();
        let disp_vec = &x_new - &x_old;
        let rms_disp = disp_vec.norm() / (disp_vec.len() as f64).sqrt();
        let max_disp = disp_vec.iter().map(|x| x.abs()).fold(0.0, f64::max);
        let rms_grad = grad_new.norm() / (grad_new.len() as f64).sqrt();
        let max_grad = grad_new.iter().map(|x| x.abs()).fold(0.0, f64::max);

        // Print energy and convergence status
        println!(
            "E1 = {:.8}, E2 = {:.8}, ΔE = {:.8}",
            state1_new.energy,
            state2_new.energy,
            de
        );
        print_convergence_status(&conv, de, rms_grad, max_grad, rms_disp, max_disp, &config);

        if conv.is_converged() {
            println!("\nConverged at step {}", step + 1);
            io::write_xyz(&geometry, Path::new("final.xyz"))?;

            // Clean up temporary files after successful convergence
            if let Err(e) = cleanup_manager.cleanup_directory(Path::new(job_dir)) {
                println!("Warning: Failed to clean up temporary files: {}", e);
            }

            return Ok(());
        }

        // Update Hessian
        let sk = &x_new - &x_old;
        let yk = &grad_new - &grad;
        hessian = optimizer::update_hessian_psb(&hessian, &sk, &yk);

        // Add to history for GDIIS/GEDIIS
        let energy_diff = state1_new.energy - state2_new.energy;
        opt_state.add_to_history(
            state1_new.geometry.coords.clone(),
            grad_new.clone(),
            hessian.clone(),
            energy_diff,
        );

        // Save checkpoint with dynamic filename based on input file
        let checkpoint_filename = format!("{}.json", job_dir);
        let checkpoint =
            checkpoint::Checkpoint::new(step, &geometry, &state1_new.geometry.coords, &hessian, &opt_state, &config);
        checkpoint.save(Path::new(&checkpoint_filename))?;

        // Periodic cleanup during optimization to prevent file accumulation
        let cleanup_freq = cleanup_manager.config().cleanup_frequency();
        if cleanup_freq > 0 && (step + 1) % cleanup_freq as usize == 0 {
            println!(
                "Performing periodic cleanup (every {} steps)...",
                cleanup_freq
            );
            if let Err(e) = cleanup_manager.cleanup_directory(Path::new(job_dir)) {
                println!("Warning: Failed to clean up temporary files: {}", e);
            }
        }

        x_old = state1_new.geometry.coords.clone();
    }

    // Clean up temporary files even if optimization didn't converge
    if let Err(e) = cleanup_manager.cleanup_directory(Path::new(job_dir)) {
        println!("Warning: Failed to clean up temporary files: {}", e);
    }

    Err("Maximum steps exceeded".into())
}

/// Manages ORCA wavefunction files (.gbw) based on run mode.
///
/// This function implements the Python MECP.py logic for ORCA wavefunction file management:
/// - NoRead mode: Deletes .gbw files to prevent reuse
/// - Normal/Read modes: Renames .gbw files for reuse in subsequent calculations (more efficient than copying)
/// - Writes XYZ files for ORCA (following Python logic)
///
/// # Arguments
///
/// * `step` - Current optimization step number
/// * `config` - Configuration containing run mode and program settings
/// * `geometry` - Current molecular geometry for XYZ file writing
///
/// # Returns
///
/// Returns `Ok(())` on success, or an error if file operations fail.
fn manage_orca_wavefunction_files(
    step: usize,
    config: &config::Config,
    geometry: &geometry::Geometry,
    job_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Only manage files for ORCA program
    if config.program != config::QMProgram::Orca {
        return Ok(());
    }

    // Load settings to get print_level
    let settings = match omecp::settings::SettingsManager::load() {
        Ok(s) => s,
        Err(_) => {
            // If settings can't be loaded, use default (quiet mode)
            return Ok(());
        }
    };
    let print_level = settings.general().print_level;

    let delete_gbw = config.run_mode == config::RunMode::NoRead;

    if delete_gbw {
        // Delete .gbw files for noread mode (following Python logic)
        let gbw_a = format!("{}/{}_state_A.gbw", job_dir, step);
        let gbw_b = format!("{}/{}_state_B.gbw", job_dir, step);

        if Path::new(&gbw_a).exists() {
            validation::log_file_operation("Delete", &gbw_a, None, print_level);
            std::fs::remove_file(&gbw_a)?;
            if print_level >= 1 {
                println!("Deleted {} for noread mode", gbw_a);
            }
        }
        if Path::new(&gbw_b).exists() {
            validation::log_file_operation("Delete", &gbw_b, None, print_level);
            std::fs::remove_file(&gbw_b)?;
            if print_level >= 1 {
                println!("Deleted {} for noread mode", gbw_b);
            }
        }
    } else {
        // Rename .gbw files for reuse (more efficient than copying)
        let gbw_a = format!("{}/{}_state_A.gbw", job_dir, step);
        let gbw_b = format!("{}/{}_state_B.gbw", job_dir, step);

        if Path::new(&gbw_a).exists() {
            let dest_a = format!("{}/state_A.gbw", job_dir);

            // Remove existing destination file if it exists
            if Path::new(&dest_a).exists() {
                validation::log_file_operation("Delete", &dest_a, None, print_level);
                std::fs::remove_file(&dest_a)?;
            }

            validation::log_file_operation("Rename", &gbw_a, Some(&dest_a), print_level);
            std::fs::rename(&gbw_a, &dest_a)?;
            if print_level >= 1 {
                println!("Renamed {} → {}", gbw_a, dest_a);
            }
        }
        if Path::new(&gbw_b).exists() {
            let dest_b = format!("{}/state_B.gbw", job_dir);

            // Remove existing destination file if it exists
            if Path::new(&dest_b).exists() {
                validation::log_file_operation("Delete", &dest_b, None, print_level);
                std::fs::remove_file(&dest_b)?;
            }

            validation::log_file_operation("Rename", &gbw_b, Some(&dest_b), print_level);
            std::fs::rename(&gbw_b, &dest_b)?;
            if print_level >= 1 {
                println!("Renamed {} → {}", gbw_b, dest_b);
            }
        }
    }

    // Write XYZ file for ORCA (following Python logic)
    let xyz_file = format!("{}/{}.xyz", job_dir, step);
    io::write_xyz(geometry, Path::new(&xyz_file))?;
    if print_level >= 1 {
        println!("Wrote XYZ file: {}", xyz_file);
    }

    Ok(())
}

/// Executes PES (Potential Energy Surface) scans following Python MECP.py logic.
///
/// This function implements the complete PES scan workflow from Python MECP.py:
/// - Supports 1D and 2D scans with proper grid generation
/// - Applies scan constraints temporarily during optimization
/// - Runs constrained MECP optimization at each scan point
/// - Saves results with proper naming convention
/// - Handles constraint management (add/remove scan constraints)
///
/// # Python MECP.py Implementation Details:
/// - SCANS format: [ [[r,A,B], [start, num, size] ], ... ]
/// - Supports up to 2D scans (automatically adds dummy 2nd dimension for 1D)
/// - Uses constraint system with temporary constraint addition/removal
/// - Saves results with format: {val1:4f}_{val2:4f}.{ext}
///
/// # Arguments
///
/// * `input_data` - Parsed input data containing scan specifications
/// * `qm` - QM interface for running calculations
///
/// # Returns
///
/// Returns `Ok(())` on successful completion of all scan points, or error if any scan point fails.
fn run_pes_scan(
    input_data: parser::InputData,
    qm: &dyn qm_interface::QMInterface,
    job_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use config::ScanType;

    println!("\n****Running PES Scan****");
    println!("Following Python MECP.py scan logic");

    let config = &input_data.config;
    let mut geometry = input_data.geometry;
    let mut constraints = input_data.constraints.clone();

    // Store initial number of constraints (following Python MECP.py)
    let _initial_cons_num = constraints.len();

    // Get scan specifications (following Python MECP.py SCANS format)
    let scan1 = &config.scans[0];
    let mut scans = config.scans.clone();

    // Add dummy 2nd dimension for 1D scans (following Python MECP.py logic)
    if scans.len() == 1 {
        scans.push(config::ScanSpec {
            scan_type: ScanType::Bond { atoms: (0, 0) }, // Dummy scan
            start: 0.0,
            num_points: 0,
            step_size: 0.0,
        });
    }

    let scan2 = &scans[1];

    // Generate scan grid values (following Python MECP.py scanVars logic)
    let mut scan_vars = [Vec::new(), Vec::new()];

    // First dimension values
    for i in 0..scan1.num_points {
        scan_vars[0].push(scan1.start + i as f64 * scan1.step_size);
    }

    // Second dimension values
    for i in 0..scan2.num_points {
        scan_vars[1].push(scan2.start + i as f64 * scan2.step_size);
    }

    // Ensure second dimension has at least one value (following Python MECP.py)
    if scan_vars[1].is_empty() {
        scan_vars[1].push(-1.0); // Dummy value for 1D scans
    }

    println!(
        "Scan grid: {} x {} points",
        scan_vars[0].len(),
        scan_vars[1].len()
    );

    // Collect scan results for analysis (Task 6.2)
    let mut scan_results = Vec::new();

    // Execute scan grid (following Python MECP.py nested loop structure)
    for &val1 in &scan_vars[0] {
        // Add first scan constraint (following Python MECP.py logic)
        let constraint1 = create_scan_constraint(&scan1.scan_type, val1);
        constraints.push(constraint1);

        for &val2 in &scan_vars[1] {
            // Add second scan constraint if it's a 2D scan
            if scan2.num_points > 0 {
                let constraint2 = create_scan_constraint(&scan2.scan_type, val2);
                constraints.push(constraint2);
            }

            // Print scan cycle info (following Python MECP.py format)
            println!("\n****Scan Cycle {:.4}_{:.4}****", val1, val2);

            // Debug: print constraints (following Python MECP.py)
            println!("constraints after pop and append");
            for (i, constraint) in constraints.iter().enumerate() {
                println!("  {}: {:?}", i, constraint);
            }

            // Run MECP optimization with scan constraints
            let scan_start_time = std::time::Instant::now();
            let optimization_result = execute_pes_scan_point(
                config,
                &mut geometry,
                &constraints,
                &input_data.tail1,
                &input_data.tail2,
                &input_data.fixed_atoms,
                qm,
                job_dir,
            );

            let (converged_step, final_geometry, converged) = match optimization_result {
                Ok((step, geom)) => {
                    println!("****Congrats! MECP has converged****");
                    (step, geom, true)
                }
                Err(e) => {
                    println!("****Warning: MECP optimization failed: {}****", e);
                    println!("Continuing with unconverged geometry");
                    (0, geometry.clone(), false)
                }
            };

            geometry = final_geometry.clone();

            // Read final energies for analysis (Task 6.2)
            let (energy_a, energy_b) = if converged {
                // Read energies from the converged calculation
                let output_ext = get_output_file_base(config.program);
                match (
                    qm.read_output(
                        Path::new(&format!("job_dir/{}_A.{}", converged_step, output_ext)),
                        config.state1,
                    ),
                    qm.read_output(
                        Path::new(&format!("job_dir/{}_B.{}", converged_step, output_ext)),
                        config.state2,
                    ),
                ) {
                    (Ok(state_a), Ok(state_b)) => (state_a.energy, state_b.energy),
                    _ => {
                        println!("Warning: Could not read final energies, using dummy values");
                        (0.0, 0.0)
                    }
                }
            } else {
                // Use dummy values for failed optimizations
                (0.0, 0.0)
            };

            // Create scan point result for analysis (Task 6.2)
            let scan_result = pes_scan::ScanPointResult {
                coord1: val1,
                coord2: val2,
                energy_a,
                energy_b,
                energy_diff: energy_a - energy_b,
                converged,
                num_steps: converged_step,
                geometry: final_geometry.clone(),
            };
            scan_results.push(scan_result);

            // Save scan results with Python MECP.py naming convention
            if converged {
                save_scan_results(config, &geometry, converged_step, val1, val2)?;
            }

            let scan_duration = scan_start_time.elapsed();
            println!(
                "Scan point completed in {:.2}s",
                scan_duration.as_secs_f64()
            );

            // Remove second scan constraint if it was added
            if scan2.num_points > 0 {
                constraints.pop();
            }
        }

        // Remove first scan constraint (following Python MECP.py)
        constraints.pop();
    }

    println!("\n****PES Scan completed successfully****");

    // Perform comprehensive scan analysis (Task 6.2)
    println!("\n****Analyzing PES Scan Results****");
    pes_scan::analyze_scan_results(&scan_results, "pes_scan_analysis.txt")?;

    Ok(())
}

/// Creates a scan constraint from scan type and value.
///
/// This helper function converts scan specifications into constraint objects
/// following the Python MECP.py constraint format.
///
/// # Arguments
///
/// * `scan_type` - The type of scan (bond or angle)
/// * `value` - The target value for the constraint
///
/// # Returns
///
/// Returns a `Constraint` object for the scan point.
fn create_scan_constraint(scan_type: &config::ScanType, value: f64) -> constraints::Constraint {
    match scan_type {
        config::ScanType::Bond { atoms } => constraints::Constraint::Bond {
            atoms: *atoms,
            target: value,
        },
        config::ScanType::Angle { atoms } => constraints::Constraint::Angle {
            atoms: *atoms,
            target: value.to_radians(), // Convert degrees to radians
        },
    }
}

/// Executes a single PES scan point with constrained MECP optimization.
///
/// This function implements the core optimization logic for each scan point,
/// following the Python MECP.py `runOpt()` call within the scan loop.
///
/// # Arguments
///
/// * `config` - Configuration parameters
/// * `geometry` - Starting geometry for this scan point
/// * `constraints` - All constraints including scan constraints
/// * `tail1` - Tail section for state A
/// * `tail2` - Tail section for state B
/// * `fixed_atoms` - List of fixed atom indices
/// * `qm` - QM interface for running calculations
///
/// # Returns
///
/// Returns a tuple of (converged_step, final_geometry) on successful convergence.
#[allow(clippy::too_many_arguments)]
fn execute_pes_scan_point(
    config: &config::Config,
    geometry: &mut geometry::Geometry,
    constraints: &[constraints::Constraint],
    tail1: &str,
    tail2: &str,
    fixed_atoms: &[usize],
    qm: &dyn qm_interface::QMInterface,
    job_dir: &str,
) -> Result<(usize, geometry::Geometry), Box<dyn std::error::Error>> {
    // Delegate the actual work to the single-optimization routine which performs
    // the full constrained MECP optimization. This keeps logic centralized.
    //
    // The clippy warning about too many arguments is suppressed for this wrapper
    // because callers pass the same separate pieces of data that map naturally
    // to the underlying optimization routine. Grouping these would require
    // broader changes across call sites; suppressing the lint here is the
    // minimal, local fix to address the diagnostics.
    run_single_optimization(
        config,
        geometry,
        constraints,
        tail1,
        tail2,
        fixed_atoms,
        qm,
        job_dir,
    )?;

    // The single-optimization routine returns () on success. It performs the
    // full optimization and writes outputs (including any checkpoint files).
    // For compatibility with the scan loop we return a conservative converged
    // step value. If more detailed step information is needed in future, the
    // single-optimization interface should be extended to report it.
    Ok((1, geometry.clone()))
}

/// Saves scan results with proper naming convention following Python MECP.py.
///
/// This function saves the converged geometry and calculation files using
/// the same naming convention as Python MECP.py: {val1:4f}_{val2:4f}.{ext}
///
/// # Arguments
///
/// * `config` - Configuration parameters
/// * `geometry` - Final converged geometry
/// * `converged_step` - Step number where convergence was achieved
/// * `val1` - First scan coordinate value
/// * `val2` - Second scan coordinate value (may be dummy for 1D scans)
///
/// # Returns
///
/// Returns `Ok(())` on successful file operations.
fn save_scan_results(
    config: &config::Config,
    geometry: &geometry::Geometry,
    converged_step: usize,
    val1: f64,
    val2: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    // Save geometry file
    let xyz_filename = format!("scan_{:.4}_{:.4}.xyz", val1, val2);
    io::write_xyz(geometry, Path::new(&xyz_filename))?;

    // Copy calculation files based on program type (following Python MECP.py)
    match config.program {
        config::QMProgram::Gaussian => {
            // Copy Gaussian files
            let input_ext = get_input_file_extension(config::QMProgram::Gaussian);
            let output_ext = get_output_file_base(config::QMProgram::Gaussian);
            let gjf_a = format!("job_dir/{}_A.{}", converged_step, input_ext);
            let gjf_b = format!("job_dir/{}_B.{}", converged_step, input_ext);
            let log_a = format!("job_dir/{}_A.{}", converged_step, output_ext);
            let log_b = format!("job_dir/{}_B.{}", converged_step, output_ext);

            if Path::new(&gjf_a).exists() {
                std::fs::copy(&gjf_a, format!("{:.4}_{:.4}_A.{}", val1, val2, input_ext))?;
            }
            if Path::new(&gjf_b).exists() {
                std::fs::copy(&gjf_b, format!("{:.4}_{:.4}_B.{}", val1, val2, input_ext))?;
            }
            if Path::new(&log_a).exists() {
                std::fs::copy(&log_a, format!("{:.4}_{:.4}_A.{}", val1, val2, output_ext))?;
            }
            if Path::new(&log_b).exists() {
                std::fs::copy(&log_b, format!("{:.4}_{:.4}_B.{}", val1, val2, output_ext))?;
            }
        }
        config::QMProgram::Orca => {
            // Copy ORCA files
            let input_ext = get_input_file_extension(config::QMProgram::Orca);
            let output_ext = get_output_file_base(config::QMProgram::Orca);
            let inp_a = format!("job_dir/{}_A.{}", converged_step, input_ext);
            let inp_b = format!("job_dir/{}_B.{}", converged_step, input_ext);
            let out_a = format!("job_dir/{}_A.{}", converged_step, output_ext);
            let out_b = format!("job_dir/{}_B.{}", converged_step, output_ext);

            if Path::new(&inp_a).exists() {
                std::fs::copy(&inp_a, format!("{:.4}_{:.4}_A.{}", val1, val2, input_ext))?;
            }
            if Path::new(&inp_b).exists() {
                std::fs::copy(&inp_b, format!("{:.4}_{:.4}_B.{}", val1, val2, input_ext))?;
            }
            if Path::new(&out_a).exists() {
                std::fs::copy(&out_a, format!("{:.4}_{:.4}_A.{}", val1, val2, output_ext))?;
            }
            if Path::new(&out_b).exists() {
                std::fs::copy(&out_b, format!("{:.4}_{:.4}_B.{}", val1, val2, output_ext))?;
            }
        }
        config::QMProgram::Xtb => {
            // Copy XTB files
            let out_a = format!("job_dir/{}_A.out", converged_step);
            let out_b = format!("job_dir/{}_B.out", converged_step);

            if Path::new(&out_a).exists() {
                std::fs::copy(&out_a, format!("{:.4}_{:.4}_A.out", val1, val2))?;
            }
            if Path::new(&out_b).exists() {
                std::fs::copy(&out_b, format!("{:.4}_{:.4}_B.out", val1, val2))?;
            }
        }
        config::QMProgram::Bagel => {
            // Copy BAGEL files
            let output_ext = get_output_file_base(config::QMProgram::Bagel);
            let out_a = format!("job_dir/{}_A.{}", converged_step, output_ext);
            let out_b = format!("job_dir/{}_B.{}", converged_step, output_ext);

            if Path::new(&out_a).exists() {
                std::fs::copy(&out_a, format!("{:.4}_{:.4}_A.{}", val1, val2, output_ext))?;
            }
            if Path::new(&out_b).exists() {
                std::fs::copy(&out_b, format!("{:.4}_{:.4}_B.{}", val1, val2, output_ext))?;
            }
        }
        config::QMProgram::Custom => {
            // Copy custom program files
            let output_ext = get_output_file_base(config::QMProgram::Custom);
            let out_a = format!("job_dir/{}_A.{}", converged_step, output_ext);
            let out_b = format!("job_dir/{}_B.{}", converged_step, output_ext);

            if Path::new(&out_a).exists() {
                std::fs::copy(&out_a, format!("{:.4}_{:.4}_A.{}", val1, val2, output_ext))?;
            }
            if Path::new(&out_b).exists() {
                std::fs::copy(&out_b, format!("{:.4}_{:.4}_B.{}", val1, val2, output_ext))?;
            }
        }
    }

    println!("Saved scan results: {}", xyz_filename);
    Ok(())
}

/// Builds a raw program header without any method modifications.
///
/// This function matches Python MECP.py's `buildHeader` function exactly,
/// providing the original method string without any additional keywords
/// like `force`, `guess=read`, `stable=opt`, etc.
///
/// This is used for pre-point calculations in Normal mode where we need
/// simple SCF calculations to generate initial checkpoint files.
///
/// # Arguments
///
/// * `config` - Configuration containing program and method information
/// * `charge` - Charge for this state
/// * `mult` - Multiplicity for this state
/// * `td` - TD-DFT keywords (if any)
/// * `state` - State index for TD-DFT calculations
///
/// # Returns
///
/// Returns a header string with the original method, no modifications
fn build_raw_program_header(
    config: &config::Config,
    charge: i32,
    mult: usize,
    td: &str,
    _state: usize,
    chk_file: &str,
) -> String {
    match config.program {
        config::QMProgram::Gaussian => {
            // Match Python: f'%chk=a.chk\n%nprocshared={NProcs} \n%mem={Mem} \n# {Method} {Td1} nosymm\n\n Title Card \n\n{Charge1} {Mult1}'
            format!(
                "%chk={}\n%nprocshared={}\n%mem={}\n# {} {} nosymm\n\nTitle Card\n\n{} {}",
                chk_file,
                config.nprocs,
                config.mem,
                config.method, // Original method, NO modifications
                td,
                charge,
                mult
            )
        }
        config::QMProgram::Orca => {
            // Match Python: f'%pal nprocs {NProcs} end\n%maxcore {Mem} \n! {Method} \n\n *xyz {Charge1} {Mult1}'
            format!(
                "%pal nprocs {} end\n%maxcore {}\n! {}\n\n*xyz {} {}",
                config.nprocs,
                config.mem,
                config.method, // Original method, NO modifications
                charge,
                mult
            )
        }
        config::QMProgram::Xtb => {
            // XTB doesn't need complex headers for pre-point
            format!(
                "$chrg {}\n$uhf {}\n$end",
                charge,
                mult.saturating_sub(1) // UHF = multiplicity - 1
            )
        }
        config::QMProgram::Bagel => {
            // BAGEL uses JSON format, but for pre-point we can use a simple structure
            format!(
                "{{\n  \"bagel\": [\n    {{\n      \"title\": \"molecule\",\n      \"charge\": {},\n      \"nspin\": {}\n    }}\n  ]\n}}",
                charge,
                mult.saturating_sub(1)
            )
        }
        config::QMProgram::Custom => {
            // Custom programs follow Gaussian-like format but with original method
            format!(
                "%chk=calc.chk\n%nprocshared={}\n%mem={}\n# {} nosymm\n\nTitle Card\n\n{} {}",
                config.nprocs,
                config.mem,
                config.method, // Original method, NO modifications
                charge,
                mult
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_single_optimization(
    config: &config::Config,
    geometry: &mut geometry::Geometry,
    constraints: &[constraints::Constraint],
    tail1: &str,
    tail2: &str,
    fixed_atoms: &[usize],
    qm: &dyn qm_interface::QMInterface,
    job_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Load settings to get print_level
    let print_level = match omecp::settings::SettingsManager::load() {
        Ok(s) => s.general().print_level,
        Err(_) => 0, // Default to quiet mode if settings can't be loaded
    };

    // Phase 1: Pre-point calculations for Normal mode (following Python MECP.py logic)
    if config.run_mode == config::RunMode::Normal {
        println!(
            "****Normal Mode: Phase 1 - Pre-point calculations to generate checkpoint files****"
        );

        // Build RAW headers WITHOUT any modifications (matching Python buildHeader)
        // This means NO force, NO guess=read, just the basic method
        let pre_header_a = build_raw_program_header(
            config,
            config.charge1,
            config.mult1,
            &config.td1,
            config.state1,
            "a.chk",
        );
        let pre_header_b = build_raw_program_header(
            config,
            config.charge2,
            config.mult2,
            &config.td2,
            config.state2,
            "b.chk",
        );

        println!("Pre-point headers (raw, no modifications):");
        println!("State A: {}", pre_header_a.lines().nth(2).unwrap_or(""));
        println!("State B: {}", pre_header_b.lines().nth(2).unwrap_or(""));

        // Write and run pre-point calculations
        let ext = get_input_file_extension(config.program);
        let pre_a_path = format!("{}/pre_A.{}", job_dir, ext);
        let pre_b_path = format!("{}/pre_B.{}", job_dir, ext);

        qm.write_input(geometry, &pre_header_a, tail1, Path::new(&pre_a_path))?;
        qm.write_input(geometry, &pre_header_b, tail2, Path::new(&pre_b_path))?;

        println!("Running pre-point calculation for state B...");
        qm.run_calculation(Path::new(&pre_b_path))?;

        println!("Running pre-point calculation for state A...");
        qm.run_calculation(Path::new(&pre_a_path))?;

        // Copy checkpoint/wavefunction files to standard locations (following Python logic)
        match config.program {
            config::QMProgram::Gaussian => {
                println!("Copying Gaussian checkpoint files...");
                let pre_a_chk = format!("{}/pre_A.chk", job_dir);
                let pre_b_chk = format!("{}/pre_B.chk", job_dir);
                let a_chk = format!("{}/a.chk", job_dir);
                let b_chk = format!("{}/b.chk", job_dir);

                std::fs::copy(&pre_a_chk, &a_chk)?;
                std::fs::copy(&pre_b_chk, &b_chk)?;
                // Also copy to root directory for compatibility
                let _ = std::fs::copy(&a_chk, "a.chk");
                let _ = std::fs::copy(&b_chk, "b.chk");
                println!("✓ Gaussian checkpoint files ready for main optimization loop");
            }
            config::QMProgram::Orca => {
                if print_level >= 1 {
                    println!("Preparing ORCA wavefunction files...");
                }
                let pre_a_gbw = format!("{}/pre_A.gbw", job_dir);
                let pre_b_gbw = format!("{}/pre_B.gbw", job_dir);
                let a_gbw = format!("{}/a.gbw", job_dir);
                let b_gbw = format!("{}/b.gbw", job_dir);

                // Copy pre_ files to a.gbw and b.gbw (these are template files, so copy is appropriate)
                validation::log_file_operation("Copy", &pre_a_gbw, Some(&a_gbw), print_level);
                std::fs::copy(&pre_a_gbw, &a_gbw)?;
                validation::log_file_operation("Copy", &pre_b_gbw, Some(&b_gbw), print_level);
                std::fs::copy(&pre_b_gbw, &b_gbw)?;

                // Also copy to root directory for compatibility
                validation::log_file_operation("Copy", &a_gbw, Some("a.gbw"), print_level);
                let _ = std::fs::copy(&a_gbw, "a.gbw");
                validation::log_file_operation("Copy", &b_gbw, Some("b.gbw"), print_level);
                let _ = std::fs::copy(&b_gbw, "b.gbw");

                if print_level >= 1 {
                    println!("✓ ORCA wavefunction files ready for main optimization loop");
                }
            }
            config::QMProgram::Xtb => {
                println!("XTB pre-point calculations completed");
                println!(
                    "✓ XTB doesn't require checkpoint files - ready for main optimization loop"
                );
                // XTB doesn't use persistent checkpoint files like Gaussian/ORCA
                // The pre-point calculations establish the initial geometry and energy
            }
            config::QMProgram::Bagel => {
                println!("BAGEL pre-point calculations completed");
                println!("✓ BAGEL uses model-based approach - ready for main optimization loop");
                // BAGEL uses JSON model files rather than binary checkpoint files
                // The pre-point calculations validate the model and establish initial state
            }
            config::QMProgram::Custom => {
                println!("Custom program pre-point calculations completed");
                println!("✓ Custom program checkpoint handling depends on implementation");
                // Custom programs may or may not use checkpoint files
                // The behavior depends on the specific program's interface configuration
            }
        }

        println!("****Normal Mode: Phase 2 - Main optimization loop with checkpoint reading****");
    }

    // Phase 2: Main optimization loop with proper headers (including guess=read for Normal mode)
    let header_a = io::build_program_header(
        config,
        config.charge1,
        config.mult1,
        &config.td1,
        config.state1,
    );
    let header_b = io::build_program_header(
        config,
        config.charge2,
        config.mult2,
        &config.td2,
        config.state2,
    );

    // For Normal mode, we start from step 0 but with checkpoint reading enabled
    // For other modes, this is the initial calculation
    let ext = get_input_file_extension(config.program);
    let initial_a_path = format!("{}/0_A.{}", job_dir, ext);
    let initial_b_path = format!("{}/0_B.{}", job_dir, ext);

    qm.write_input(geometry, &header_a, tail1, Path::new(&initial_a_path))?;
    qm.write_input(geometry, &header_b, tail2, Path::new(&initial_b_path))?;
    qm.run_calculation(Path::new(&initial_a_path))?;
    qm.run_calculation(Path::new(&initial_b_path))?;

    let mut opt_state = optimizer::OptimizationState::new();
    let mut x_old = geometry.coords.clone();
    let mut hessian = DMatrix::identity(geometry.coords.len(), geometry.coords.len());

    for step in 0..config.max_steps {
        let output_ext = get_output_file_base(config.program);
        let state1 = qm.read_output(
            Path::new(&format!("job_dir/{}_A.{}", step, output_ext)),
            config.state1,
        )?;
        let state2 = qm.read_output(
            Path::new(&format!("job_dir/{}_B.{}", step, output_ext)),
            config.state2,
        )?;

        let grad = optimizer::compute_mecp_gradient(&state1, &state2, fixed_atoms);

        let x_new = if !constraints.is_empty() {
            println!("Using Lagrange multiplier constrained optimization");
            // Constrained step
            let violations = constraints::evaluate_constraints(geometry, constraints);
            let jacobian = constraints::build_constraint_jacobian(geometry, constraints);

            if let Some((delta_x, lambdas)) =
                optimizer::solve_constrained_step(&hessian, &grad, &jacobian, &violations)
            {
                opt_state.lambdas = lambdas.iter().cloned().collect();

                // Apply step size limit
                let step_norm = delta_x.norm();
                if step_norm > config.max_step_size {
                    let scale = config.max_step_size / step_norm;
                    println!("current stepsize: {} is reduced to max_size {}", step_norm, config.max_step_size);
                    x_old.clone() + delta_x * scale
                } else {
                    x_old.clone() + delta_x
                }
            } else {
                // Fallback to BFGS if solver fails
                println!("Warning: Constrained step solver failed. Falling back to BFGS.");
                let adaptive_scale = if step == 0 {
                    1.0
                } else {
                    let energy_current = state1.energy - state2.energy;
                    let energy_previous = opt_state.energy_history.back().unwrap_or(&energy_current);
                    optimizer::compute_adaptive_scale(energy_current, *energy_previous, grad.norm(), step)
                };
                optimizer::bfgs_step(&x_old, &grad, &hessian, config, adaptive_scale)
            }
        } else {
            // Choose optimizer based on switch_step configuration
            let use_bfgs = if config.switch_step >= config.max_steps {
                // Always BFGS if switch_step >= max_steps (BFGS-only mode)
                true
            } else if config.switch_step == 0 {
                // Never BFGS if switch_step = 0 (DIIS-only mode)
                false
            } else {
                // Use BFGS until switch_step, then switch to DIIS
                step < config.switch_step
            };

            if use_bfgs || !opt_state.has_enough_history() {
                let adaptive_scale = if step == 0 {
                    1.0
                } else {
                    let energy_current = state1.energy - state2.energy;
                    let energy_previous = opt_state.energy_history.back().unwrap_or(&energy_current);
                    optimizer::compute_adaptive_scale(energy_current, *energy_previous, grad.norm(), step)
                };
                optimizer::bfgs_step(&x_old, &grad, &hessian, config, adaptive_scale)
            } else if config.use_gediis {
                optimizer::gediis_step(&opt_state, config)
            } else {
                optimizer::gdiis_step(&opt_state, config)
            }
        };

        geometry.coords = x_new.clone();

        let ext = get_input_file_extension(config.program);
        let step_a_path = format!("job_dir/{}_A.{}", step + 1, ext);
        let step_b_path = format!("job_dir/{}_B.{}", step + 1, ext);

        qm.write_input(geometry, &header_a, tail1, Path::new(&step_a_path))?;
        qm.write_input(geometry, &header_b, tail2, Path::new(&step_b_path))?;
        qm.run_calculation(Path::new(&step_a_path))?;
        qm.run_calculation(Path::new(&step_b_path))?;

        let output_ext = get_output_file_base(config.program);
        let state1_new = qm.read_output(
            Path::new(&format!("job_dir/{}_A.{}", step + 1, output_ext)),
            config.state1,
        )?;
        let state2_new = qm.read_output(
            Path::new(&format!("job_dir/{}_B.{}", step + 1, output_ext)),
            config.state2,
        )?;
        let grad_new = optimizer::compute_mecp_gradient(&state1_new, &state2_new, fixed_atoms);

        let conv = optimizer::check_convergence(
            state1_new.energy,
            state2_new.energy,
            &x_old,
            &x_new,
            &grad_new,
            config,
        );

        // Compute current values for display
        let de = (state1_new.energy - state2_new.energy).abs();
        let disp_vec = &x_new - &x_old;
        let rms_disp = disp_vec.norm() / (disp_vec.len() as f64).sqrt();
        let max_disp = disp_vec.iter().map(|x| x.abs()).fold(0.0, f64::max);
        let rms_grad = grad_new.norm() / (grad_new.len() as f64).sqrt();
        let max_grad = grad_new.iter().map(|x| x.abs()).fold(0.0, f64::max);

        // Print convergence status
        print_convergence_status(&conv, de, rms_grad, max_grad, rms_disp, max_disp, config);

        if conv.is_converged() {
            // Clean up temporary files after successful convergence
            let settings_manager = omecp::settings::SettingsManager::load().ok();
            let cleanup_manager = if let Some(ref settings) = settings_manager {
                omecp::cleanup::CleanupManager::new(
                    omecp::cleanup::CleanupConfig::from_settings_manager(settings, config.program),
                    config.program,
                )
            } else {
                omecp::cleanup::CleanupManager::new(
                    omecp::cleanup::CleanupConfig::default(),
                    config.program,
                )
            };
            if let Err(e) = cleanup_manager.cleanup_directory(Path::new(job_dir)) {
                println!("Warning: Failed to clean up temporary files: {}", e);
            }
            return Ok(());
        }

        let sk = &x_new - &x_old;
        let yk = &grad_new - &grad;
        hessian = optimizer::update_hessian_psb(&hessian, &sk, &yk);
        let energy_diff = state1_new.energy - state2_new.energy;
        opt_state.add_to_history(
            state1_new.geometry.coords.clone(),
            grad_new.clone(),
            hessian.clone(),
            energy_diff,
        );
        x_old = state1_new.geometry.coords.clone();
    }

    Ok(())
}

fn run_lst_interpolation(
    input_data: parser::InputData,
    qm: &dyn qm_interface::QMInterface,
    _job_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n****Running Advanced LST Interpolation****");

    let geom1 = input_data.lst1.as_ref().unwrap();
    let geom2 = input_data.lst2.as_ref().unwrap();
    let config = &input_data.config;

    // Choose interpolation method
    println!("Available LST methods:");
    println!("1. Linear Synchronous Transit (LST) - Simple linear interpolation");
    println!("2. Quadratic Synchronous Transit (QST) - Smooth quadratic interpolation");
    println!("Select method (1-2) [1]: ");

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let method_choice = input.trim().parse::<usize>().unwrap_or(1);

    let method = match method_choice {
        2 => lst::InterpolationMethod::Quadratic,
        _ => lst::InterpolationMethod::Linear,
    };

    // Get number of interpolation points
    println!("Number of interpolation points [10]: ");
    input.clear();
    std::io::stdin().read_line(&mut input)?;
    let num_points = input.trim().parse::<usize>().unwrap_or(10);

    println!(
        "Using {:?} interpolation with {} points",
        method, num_points
    );

    // Generate geometries
    let geometries = lst::interpolate(geom1, geom2, num_points, method);

    // Validate geometries
    match lst::validate_geometries(&geometries) {
        Ok(_) => println!("✓ All geometries validated successfully"),
        Err(e) => {
            println!("✗ Geometry validation failed: {}", e);
            println!("Do you want to continue anyway? (y/n) [n]: ");
            input.clear();
            std::io::stdin().read_line(&mut input)?;
            if input.trim().to_lowercase() != "y" {
                return Ok(());
            }
        }
    }

    // Show geometry preview
    lst::print_geometry_preview(&geometries);

    // Build headers
    let header_a = io::build_program_header(
        config,
        config.charge1,
        config.mult1,
        &config.td1,
        config.state1,
    );
    let header_b = io::build_program_header(
        config,
        config.charge2,
        config.mult2,
        &config.td2,
        config.state2,
    );

    // Write input files
    for (i, geom) in geometries.iter().enumerate() {
        let num = i + 1;
        let ext = get_input_file_extension(input_data.config.program);
        let step_a_path = format!("job_dir/{}_A.{}", num, ext);
        let step_b_path = format!("job_dir/{}_B.{}", num, ext);

        qm.write_input(geom, &header_a, &input_data.tail1, Path::new(&step_a_path))?;
        qm.write_input(geom, &header_b, &input_data.tail2, Path::new(&step_b_path))?;
    }

    println!(
        "\nGenerated {} input files in job_dir/",
        geometries.len() * 2
    );

    // Enhanced confirmation
    println!("\n****Confirmation****");
    println!("Interpolation method: {:?}", method);
    println!("Number of points: {}", num_points);
    println!("Total input files: {}", geometries.len() * 2);
    println!("QM program: {:?}", config.program);
    println!("Method: {}", config.method);

    println!("\nDo you want to run the calculations? (y/n) [n]: ");
    input.clear();
    std::io::stdin().read_line(&mut input)?;

    if input.trim().to_lowercase() != "y" {
        println!("LST interpolation completed. Input files are ready in job_dir/ directory.");
        println!("You can run them manually or modify the geometries as needed.");
        return Ok(());
    }

    // Run calculations and collect energies
    let mut energies_a = Vec::new();
    let mut energies_b = Vec::new();
    let mut successful_points = 0;

    for (i, _) in geometries.iter().enumerate() {
        let num = i + 1;
        println!("\n****Running Point {}/{}****", num, geometries.len());

        // Run state A
        let ext = get_input_file_extension(input_data.config.program);
        let step_a_path = format!("job_dir/{}_A.{}", num, ext);
        match qm.run_calculation(Path::new(&step_a_path)) {
            Ok(_) => {
                let output_ext = get_output_file_base(input_data.config.program);
                match qm.read_output(
                    Path::new(&format!("job_dir/{}_A.{}", num, output_ext)),
                    config.state1,
                ) {
                    Ok(state_a) => {
                        energies_a.push(state_a.energy);
                        println!("✓ State A completed: E = {:.8} hartree", state_a.energy);
                    }
                    Err(e) => {
                        println!("✗ Failed to read state A output: {}", e);
                        energies_a.push(f64::NAN);
                    }
                }
            }
            Err(e) => {
                println!("✗ Failed to run state A: {}", e);
                energies_a.push(f64::NAN);
                continue;
            }
        }

        // Run state B
        let step_b_path = format!("job_dir/{}_B.{}", num, ext);
        match qm.run_calculation(Path::new(&step_b_path)) {
            Ok(_) => {
                let output_ext = get_output_file_base(input_data.config.program);
                match qm.read_output(
                    Path::new(&format!("job_dir/{}_B.{}", num, output_ext)),
                    config.state2,
                ) {
                    Ok(state_b) => {
                        energies_b.push(state_b.energy);
                        println!("✓ State B completed: E = {:.8} hartree", state_b.energy);
                    }
                    Err(e) => {
                        println!("✗ Failed to read state B output: {}", e);
                        energies_b.push(f64::NAN);
                    }
                }
            }
            Err(e) => {
                println!("✗ Failed to run state B: {}", e);
                energies_b.push(f64::NAN);
                continue;
            }
        }

        if !energies_a.last().unwrap().is_nan() && !energies_b.last().unwrap().is_nan() {
            successful_points += 1;
        }
    }

    // Print energy profile
    println!("\n****Energy Profile****");
    println!(
        "Successful points: {}/{}",
        successful_points,
        geometries.len()
    );

    let mut min_de = f64::INFINITY;
    let mut min_de_idx = 0;

    for (i, (ea, eb)) in energies_a.iter().zip(energies_b.iter()).enumerate() {
        if ea.is_finite() && eb.is_finite() {
            let de = ea - eb;
            println!(
                "Point {:2}: EA = {:.8}, EB = {:.8}, ΔE = {:.8}",
                i + 1,
                ea,
                eb,
                de
            );

            if de.abs() < min_de.abs() {
                min_de = de;
                min_de_idx = i;
            }
        } else {
            println!("Point {:2}: Calculation failed", i + 1);
        }
    }

    if successful_points > 0 {
        println!("\n****Summary****");
        println!(
            "Minimum |ΔE| found at point {}: {:.8} hartree ({:.3} eV)",
            min_de_idx + 1,
            min_de,
            min_de * 27.211386
        );
        let ext = get_input_file_extension(input_data.config.program);
        println!(
            "Suggested MECP starting geometry: job_dir/{}_A.{}",
            min_de_idx + 1,
            ext
        );
    }

    Ok(())
}

/// Runs pre-point calculations based on program and run mode.
///
/// This function implements the Python MECP.py runPrePoint logic with program-specific
/// dispatch. Pre-point calculations are essential for:
/// - Stable mode: Running stability analysis before optimization
/// - Inter_read mode: Proper wavefunction initialization for open-shell singlets
/// - Normal mode: Standard pre-point calculations for both states
///
/// # Arguments
///
/// * `geometry` - The molecular geometry for calculations
/// * `header_a` - Header string for state A
/// * `header_b` - Header string for state B
/// * `input_data` - Complete input data including tails and config
/// * `qm` - QM interface for running calculations
/// * `run_mode` - The run mode determining pre-point behavior
///
/// # Returns
///
/// Returns `Ok(())` on success, or an error if calculations fail.
fn run_pre_point(
    geometry: &geometry::Geometry,
    header_a: &str,
    header_b: &str,
    input_data: &parser::InputData,
    qm: &dyn qm_interface::QMInterface,
    run_mode: config::RunMode,
    job_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n****Initialization: running the first single point calculations according to the mode****");

    // Dispatch to program-specific pre-point implementations
    match input_data.config.program {
        config::QMProgram::Gaussian => {
            run_pre_point_gaussian(
                geometry, header_a, header_b, input_data, qm, run_mode, job_dir,
            )?;
        }
        config::QMProgram::Orca => {
            run_pre_point_orca(
                geometry, header_a, header_b, input_data, qm, run_mode, job_dir,
            )?;
        }
        config::QMProgram::Xtb => {
            run_pre_point_xtb(
                geometry, header_a, header_b, input_data, qm, run_mode, job_dir,
            )?;
        }
        config::QMProgram::Bagel => {
            run_pre_point_bagel(
                geometry, header_a, header_b, input_data, qm, run_mode, job_dir,
            )?;
        }
        config::QMProgram::Custom => {
            // For custom programs, fall back to Gaussian-style pre-point
            run_pre_point_gaussian(
                geometry, header_a, header_b, input_data, qm, run_mode, job_dir,
            )?;
        }
    }

    println!("****Initialization OK, now entering main loop****");
    Ok(())
}

/// Gaussian-specific pre-point calculations.
///
/// Implements the Python MECP.py logic for Gaussian:
/// - Runs state B first (B→A order)
/// - For inter_read mode: copies b.chk → a.chk and adds guess=(read,mix) to state A
/// - Handles all run modes appropriately
fn run_pre_point_gaussian(
    geometry: &geometry::Geometry,
    header_a: &str,
    header_b: &str,
    input_data: &parser::InputData,
    qm: &dyn qm_interface::QMInterface,
    run_mode: config::RunMode,
    job_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Load settings to get print_level
    let settings = match omecp::settings::SettingsManager::load() {
        Ok(s) => s,
        Err(_) => {
            // If settings can't be loaded, use default (quiet mode)
            return Ok(());
        }
    };
    let print_level = settings.general().print_level;

    // Write and run state B first (following Python MECP.py B→A order)
    let ext = get_input_file_extension(input_data.config.program);
    let pre_b_path = format!("{}/pre_B.{}", job_dir, ext);

    qm.write_input(
        geometry,
        header_b,
        &input_data.tail2,
        Path::new(&pre_b_path),
    )?;
    qm.run_calculation(Path::new(&pre_b_path))?;

    // Handle inter_read mode special case
    if run_mode == config::RunMode::InterRead {
        println!("Inter-read mode: copying state B wavefunction to state A");

        // Ensure proper wavefunction copying (b.chk → a.chk for Gaussian)
        let pre_b_chk = format!("{}/pre_B.chk", job_dir);
        let a_chk = format!("{}/a.chk", job_dir);

        if Path::new(&pre_b_chk).exists() {
            validation::log_file_operation("Copy", &pre_b_chk, Some(&a_chk), print_level);
            std::fs::copy(&pre_b_chk, &a_chk)?;
        } else if Path::new("b.chk").exists() {
            validation::log_file_operation("Copy", "b.chk", Some("a.chk"), print_level);
            std::fs::copy("b.chk", "a.chk")?;
        } else {
            println!("Warning: No checkpoint file found for inter_read mode. Continuing without wavefunction copying.");
        }

        // Modify header A to add guess=(read,mix) for inter_read mode
        let mut header_a_modified = header_a.to_string();
        if header_a_modified.contains("guess=read") {
            header_a_modified = header_a_modified.replace("guess=read", "guess=(read,mix)");
            println!("Modified state A header to use guess=(read,mix) for inter_read mode");
        } else {
            // Add guess=(read,mix) if not present (fallback)
            header_a_modified = header_a_modified.replace("# ", "# guess=(read,mix) ");
            println!("Added guess=(read,mix) to state A header for inter_read mode");
        }

        let pre_a_path = format!("{}/pre_A.{}", job_dir, ext);
        qm.write_input(
            geometry,
            &header_a_modified,
            &input_data.tail1,
            Path::new(&pre_a_path),
        )?;
    } else {
        // Normal case: use header as-is
        let pre_a_path = format!("{}/pre_A.{}", job_dir, ext);
        qm.write_input(
            geometry,
            header_a,
            &input_data.tail1,
            Path::new(&pre_a_path),
        )?;
    }

    // Run state A
    let pre_a_path = format!("{}/pre_A.{}", job_dir, ext);
    qm.run_calculation(Path::new(&pre_a_path))?;

    Ok(())
}

/// ORCA-specific pre-point calculations.
///
/// Implements the Python MECP.py logic for ORCA:
/// - Runs state B first and manages .gbw files
/// - For inter_read mode: copies .gbw files and provides user guidance
/// - Handles ORCA-specific wavefunction file management
fn run_pre_point_orca(
    geometry: &geometry::Geometry,
    header_a: &str,
    header_b: &str,
    input_data: &parser::InputData,
    qm: &dyn qm_interface::QMInterface,
    run_mode: config::RunMode,
    _job_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Load settings to get print_level
    let settings = match omecp::settings::SettingsManager::load() {
        Ok(s) => s,
        Err(_) => {
            // If settings can't be loaded, use default (quiet mode)
            return Ok(());
        }
    };
    let print_level = settings.general().print_level;

    // Write and run state B first
    qm.write_input(
        geometry,
        header_b,
        &input_data.tail2,
        Path::new("job_dir/pre_B.inp"),
    )?;
    qm.run_calculation(Path::new("job_dir/pre_B.inp"))?;

    // Copy B wavefunction file if it exists
    if Path::new("job_dir/pre_B.gbw").exists() {
        validation::log_file_operation(
            "Copy",
            "job_dir/pre_B.gbw",
            Some("job_dir/b.gbw"),
            print_level,
        );
        std::fs::copy("job_dir/pre_B.gbw", "job_dir/b.gbw")?;
    }

    // Handle inter_read mode
    if run_mode == config::RunMode::InterRead {
        println!("Inter-read mode: copying state B wavefunction to state A");

        // Ensure proper wavefunction copying for ORCA
        if Path::new("job_dir/pre_B.gbw").exists() {
            validation::log_file_operation(
                "Copy",
                "job_dir/pre_B.gbw",
                Some("job_dir/a.gbw"),
                print_level,
            );
            std::fs::copy("job_dir/pre_B.gbw", "job_dir/a.gbw")?;
            if print_level >= 1 {
                println!("Copied pre_B.gbw → a.gbw for inter_read mode");
            }
        } else {
            println!("Warning: No .gbw file found for inter_read mode. Continuing without wavefunction copying.");
        }

        // Provide comprehensive user guidance for ORCA inter_read mode
        println!("\n****ORCA Inter-Read Mode Guidance****");
        println!("Note: The inter_read mode is set for ORCA. In Gaussian, the program automatically adds guess=mix for state A,");
        println!("but this will not be done for ORCA. If you want to converge to correct OSS wavefunction from a triplet wavefunction,");
        println!("guess=(read,mix) is always beneficial. So please do not forget to add relevant convergence controlling in your ORCA tail part.");
        println!("Recommended ORCA tail keywords for inter_read mode:");
        println!("  %scf");
        println!("    MaxIter 200");
        println!("    ConvForced true");
        println!("  end");
        println!("****End of ORCA Guidance****\n");
    }

    // Write and run state A
    qm.write_input(
        geometry,
        header_a,
        &input_data.tail1,
        Path::new("job_dir/pre_A.inp"),
    )?;
    qm.run_calculation(Path::new("job_dir/pre_A.inp"))?;

    // Copy A wavefunction file if it exists
    if Path::new("job_dir/pre_A.gbw").exists() {
        std::fs::copy("job_dir/pre_A.gbw", "job_dir/a.gbw")?;
    }

    Ok(())
}

/// XTB-specific pre-point calculations.
///
/// XTB typically doesn't need complex pre-point calculations, but we follow
/// the same pattern for consistency.
/// XTB-specific pre-point calculations.
///
/// Implements the Python MECP.py logic for XTB:
/// - Writes XYZ files for both states
/// - Runs XTB calculations with appropriate command line arguments
/// - XTB doesn't require complex wavefunction management like Gaussian/ORCA
///
/// # Arguments
///
/// * `geometry` - Current molecular geometry
/// * `header_a` - XTB header for state A (contains charge/multiplicity info)
/// * `header_b` - XTB header for state B (contains charge/multiplicity info)
/// * `input_data` - Parsed input data containing configuration
/// * `qm` - QM interface for running calculations
/// * `run_mode` - Current run mode (affects calculation type)
///
/// # Returns
///
/// Returns `Ok(())` on success, or an error if XTB calculations fail.
fn run_pre_point_xtb(
    geometry: &geometry::Geometry,
    header_a: &str,
    header_b: &str,
    input_data: &parser::InputData,
    qm: &dyn qm_interface::QMInterface,
    run_mode: config::RunMode,
    _job_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Running XTB pre-point calculations");

    // Create running directory if it doesn't exist
    std::fs::create_dir_all("job_dir")?;

    // Write XYZ files for both states (following Python MECP.py pattern)
    qm.write_input(
        geometry,
        header_a,
        &input_data.tail1,
        Path::new("job_dir/pre_A.xyz"),
    )?;
    qm.write_input(
        geometry,
        header_b,
        &input_data.tail2,
        Path::new("job_dir/pre_B.xyz"),
    )?;

    // Run XTB calculations for both states
    // Following Python MECP.py: run state B first, then state A
    println!("Running XTB calculation for state B");
    qm.run_calculation(Path::new("job_dir/pre_B.xyz"))?;

    println!("Running XTB calculation for state A");
    qm.run_calculation(Path::new("job_dir/pre_A.xyz"))?;

    // XTB doesn't need special handling for different run modes like Gaussian/ORCA
    // The run mode affects mainly the method modification, which is handled elsewhere
    match run_mode {
        config::RunMode::NoRead => {
            println!("XTB pre-point completed (noread mode - no wavefunction files to manage)");
        }
        config::RunMode::Stable => {
            println!("XTB pre-point completed (stable mode - XTB handles stability internally)");
        }
        config::RunMode::InterRead => {
            println!(
                "XTB pre-point completed (inter_read mode - no special handling needed for XTB)"
            );
        }
        _ => {
            println!("XTB pre-point completed");
        }
    }

    Ok(())
}

/// BAGEL-specific pre-point calculations.
///
/// Implements the Python MECP.py logic for BAGEL:
/// - Uses JSON input format with model file substitution
/// - Handles state-specific targeting and geometry insertion
/// - Manages multireference calculations with proper state indexing
///
/// # Arguments
///
/// * `geometry` - Current molecular geometry
/// * `header_a` - BAGEL header for state A (JSON template)
/// * `header_b` - BAGEL header for state B (JSON template)
/// * `input_data` - Parsed input data containing configuration and BAGEL model
/// * `qm` - QM interface for running calculations
/// * `run_mode` - Current run mode (affects calculation type)
///
/// # Returns
///
/// Returns `Ok(())` on success, or an error if BAGEL calculations fail.
fn run_pre_point_bagel(
    geometry: &geometry::Geometry,
    _header_a: &str,
    _header_b: &str,
    input_data: &parser::InputData,
    qm: &dyn qm_interface::QMInterface,
    run_mode: config::RunMode,
    _job_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Running BAGEL pre-point calculations");

    // Create running directory if it doesn't exist
    std::fs::create_dir_all("job_dir")?;

    // BAGEL requires a model file - check if it's specified
    if input_data.config.bagel_model.is_empty() {
        return Err("BAGEL model file not specified. Please set bagel_model parameter.".into());
    }

    // Check if model file exists
    if !Path::new(&input_data.config.bagel_model).exists() {
        return Err(format!(
            "BAGEL model file '{}' not found",
            input_data.config.bagel_model
        )
        .into());
    }

    println!("Using BAGEL model file: {}", input_data.config.bagel_model);

    // Write BAGEL JSON input files for both states
    // Following Python MECP.py: run state B first, then state A
    write_bagel_input(
        geometry,
        &input_data.config.bagel_model,
        "job_dir/pre_B.json",
        input_data.config.mult2 as i32,
        input_data.config.state2,
        &geometry.elements,
    )?;

    write_bagel_input(
        geometry,
        &input_data.config.bagel_model,
        "job_dir/pre_A.json",
        input_data.config.mult1 as i32,
        input_data.config.state1,
        &geometry.elements,
    )?;

    // Run BAGEL calculations for both states
    println!("Running BAGEL calculation for state B");
    qm.run_calculation(Path::new("job_dir/pre_B.json"))?;

    println!("Running BAGEL calculation for state A");
    qm.run_calculation(Path::new("job_dir/pre_A.json"))?;

    // Write XYZ file for geometry tracking (following Python MECP.py)
    io::write_xyz(geometry, Path::new("job_dir/pre.xyz"))?;

    // BAGEL doesn't need special run mode handling like Gaussian/ORCA
    match run_mode {
        config::RunMode::NoRead => {
            println!("BAGEL pre-point completed (noread mode)");
        }
        config::RunMode::Stable => {
            println!("BAGEL pre-point completed (stable mode - handled by BAGEL internally)");
        }
        config::RunMode::InterRead => {
            println!("BAGEL pre-point completed (inter_read mode - no special handling needed)");
        }
        _ => {
            println!("BAGEL pre-point completed");
        }
    }

    Ok(())
}

/// Writes a BAGEL JSON input file with geometry and parameter substitution.
///
/// This function implements the Python MECP.py `writeBAGEL()` function logic:
/// - Reads the BAGEL model file
/// - Substitutes geometry using `geom2Json()` equivalent
/// - Substitutes target state and spin multiplicity
/// - Writes the complete JSON input file
///
/// # Arguments
///
/// * `geometry` - Molecular geometry to insert
/// * `model_file` - Path to the BAGEL model file
/// * `output_file` - Path for the output JSON file
/// * `mult` - Spin multiplicity
/// * `state` - Target electronic state index
/// * `elements` - Element symbols for atoms
///
/// # Returns
///
/// Returns `Ok(())` on success, or an error if file operations fail.
fn write_bagel_input(
    geometry: &geometry::Geometry,
    model_file: &str,
    output_file: &str,
    mult: i32,
    state: usize,
    elements: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    // Read the BAGEL model file
    let model_content = std::fs::read_to_string(model_file)
        .map_err(|e| format!("Failed to read BAGEL model file '{}': {}", model_file, e))?;

    let mut bagel_content = String::new();

    // Process each line of the model file (following Python MECP.py logic)
    for line in model_content.lines() {
        if line.contains("geometry") {
            // Replace with actual geometry in JSON format
            bagel_content.push_str(&geometry_to_json(elements, geometry));
        } else if line.contains("target") {
            // Replace with target state
            bagel_content.push_str(&format!("\"target\" : {},\n", state));
        } else if line.contains("nspin") {
            // Replace with spin multiplicity (nspin = mult - 1)
            bagel_content.push_str(&format!("\"nspin\": {},\n", mult - 1));
        } else {
            // Keep the line as-is
            bagel_content.push_str(line);
            bagel_content.push('\n');
        }
    }

    // Write the complete BAGEL input file
    std::fs::write(output_file, bagel_content)
        .map_err(|e| format!("Failed to write BAGEL input file '{}': {}", output_file, e))?;

    println!("Written BAGEL input file: {}", output_file);
    Ok(())
}

/// Converts molecular geometry to BAGEL JSON format.
///
/// This function implements the Python MECP.py `geom2Json()` function:
/// - Formats geometry as JSON array with atom symbols and coordinates
/// - Uses Angstrom units (BAGEL's default)
/// - Follows BAGEL's JSON schema for geometry specification
///
/// # Arguments
///
/// * `elements` - Element symbols for each atom
/// * `geometry` - Molecular geometry with coordinates
///
/// # Returns
///
/// Returns a `String` containing the JSON-formatted geometry.
fn geometry_to_json(elements: &[String], geometry: &geometry::Geometry) -> String {
    // Use iterator with enumerate to satisfy clippy's needless_range_loop diagnostic.
    // Ensure we only iterate over the minimum of provided element labels and geometry atoms.
    let n = std::cmp::min(geometry.num_atoms, elements.len());
    let mut result = String::from("\"geometry\" : [\n");

    for (i, elem) in elements.iter().enumerate().take(n) {
        let coords = geometry.get_atom_coords(i);
        // Convert from Bohrs to Angstroms for JSON output
        let angstrom_coords = geometry::bohr_to_angstrom(&nalgebra::DVector::from_vec(vec![coords[0], coords[1], coords[2]]));
        result.push_str(&format!(
            "{{ \"atom\" : \"{}\", \"xyz\" : [ {:.6}, {:.6}, {:.6} ]}}",
            elem, angstrom_coords[0], angstrom_coords[1], angstrom_coords[2]
        ));

        // Add comma for all but the last atom
        if i != n - 1 {
            result.push(',');
        }
        result.push('\n');
    }

    result.push_str("]\n");
    result
}

/// Runs a single optimization step for XTB calculations.
///
/// This function implements the XTB-specific part of Python MECP.py's `runEachStep()`:
/// - Writes XYZ input files for both states
/// - Runs XTB calculations with appropriate command line arguments
/// - Handles XTB's simple file format requirements
///
/// # Arguments
///
/// * `geometry` - Current molecular geometry
/// * `step` - Current optimization step number
/// * `header_a` - XTB header for state A
/// * `header_b` - XTB header for state B
/// * `input_data` - Input data containing tail sections
/// * `qm` - QM interface for running calculations
///
/// # Returns
///
/// Returns `Ok(())` on success, or an error if calculations fail.
fn run_xtb_step(
    geometry: &geometry::Geometry,
    step: usize,
    header_a: &str,
    header_b: &str,
    tail1: &str,
    tail2: &str,
    qm: &dyn qm_interface::QMInterface,
) -> Result<(), Box<dyn std::error::Error>> {
    // XTB uses XYZ format - following Python MECP.py pattern
    let step_name_a = format!("job_dir/{}_A.xyz", step);
    let step_name_b = format!("job_dir/{}_B.xyz", step);

    // Write XYZ input files
    qm.write_input(geometry, header_a, tail1, Path::new(&step_name_a))?;
    qm.write_input(geometry, header_b, tail2, Path::new(&step_name_b))?;

    // Run XTB calculations (following Python MECP.py order: B first, then A)
    qm.run_calculation(Path::new(&step_name_b))?;
    qm.run_calculation(Path::new(&step_name_a))?;

    Ok(())
}

/// Runs a single optimization step for BAGEL calculations.
///
/// This function implements the BAGEL-specific part of Python MECP.py's `runEachStep()`:
/// - Writes JSON input files using model file substitution
/// - Runs BAGEL calculations with proper state targeting
/// - Handles BAGEL's JSON format and multireference requirements
/// - Writes XYZ file for geometry tracking
///
/// # Arguments
///
/// * `geometry` - Current molecular geometry
/// * `step` - Current optimization step number
/// * `input_data` - Input data containing configuration and model file
/// * `qm` - QM interface for running calculations
///
/// # Returns
///
/// Returns `Ok(())` on success, or an error if calculations fail.
fn run_bagel_step(
    geometry: &geometry::Geometry,
    step: usize,
    config: &config::Config,
    elements: &[String],
    qm: &dyn qm_interface::QMInterface,
) -> Result<(), Box<dyn std::error::Error>> {
    // BAGEL uses JSON format - following Python MECP.py pattern
    let step_name_a = format!("job_dir/{}_A.json", step);
    let step_name_b = format!("job_dir/{}_B.json", step);

    // Write BAGEL JSON input files with model substitution
    write_bagel_input(
        geometry,
        &config.bagel_model,
        &step_name_a,
        config.mult1 as i32,
        config.state1,
        elements,
    )?;

    write_bagel_input(
        geometry,
        &config.bagel_model,
        &step_name_b,
        config.mult2 as i32,
        config.state2,
        elements,
    )?;

    // Run BAGEL calculations (following Python MECP.py order: B first, then A)
    qm.run_calculation(Path::new(&step_name_b))?;
    qm.run_calculation(Path::new(&step_name_a))?;

    // Write XYZ file for geometry tracking (following Python MECP.py)
    io::write_xyz(geometry, Path::new(&format!("job_dir/{}.xyz", step)))?;

    Ok(())
}

fn run_restart(
    input_data: parser::InputData,
    qm: &dyn qm_interface::QMInterface,
    job_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n****Restarting from checkpoint****");

    // Load checkpoint with dynamic filename based on input file
    let checkpoint_filename = format!("{}.json", job_dir);
    let checkpoint_path = Path::new(&checkpoint_filename);
    let checkpoint_load = checkpoint::Checkpoint::load(checkpoint_path)?;
    let step = checkpoint_load.step;
    let mut geometry = checkpoint_load.geometry;
    let x_old = checkpoint_load.x_old;
    let hessian = checkpoint_load.hessian;
    let mut opt_state = checkpoint_load.opt_state;
    let config = checkpoint_load.config;

    println!("Loaded checkpoint from step {}", step);

    let constraints = &input_data.constraints;
    let fixed_atoms = &input_data.fixed_atoms;

    // Build headers
    let header_a = io::build_program_header(
        &config,
        config.charge1,
        config.mult1,
        &config.td1,
        config.state1,
    );
    let header_b = io::build_program_header(
        &config,
        config.charge2,
        config.mult2,
        &config.td2,
        config.state2,
    );

    // Continue optimization from the next step
    let start_step = step + 1;
    for step in start_step..config.max_steps {
        println!("\n****Step {}****", step + 1);

        // Run calculations with current geometry
        let ext = get_input_file_extension(input_data.config.program);
        let step_name_a = format!("job_dir/{}_A.{}", step, ext);
        qm.write_input(
            &geometry,
            &header_a,
            &input_data.tail1,
            Path::new(&step_name_a),
        )?;
        qm.run_calculation(Path::new(&step_name_a))?;

        let step_name_b = format!("job_dir/{}_B.{}", step, ext);
        qm.write_input(
            &geometry,
            &header_b,
            &input_data.tail2,
            Path::new(&step_name_b),
        )?;
        qm.run_calculation(Path::new(&step_name_b))?;

        let output_ext = get_output_file_base(config.program);
        let state1 = qm.read_output(
            Path::new(&format!("job_dir/{}_A.{}", step, output_ext)),
            config.state1,
        )?;
        let state2 = qm.read_output(
            Path::new(&format!("job_dir/{}_B.{}", step, output_ext)),
            config.state2,
        )?;

        // Compute MECP gradient
        let grad = optimizer::compute_mecp_gradient(&state1, &state2, fixed_atoms);

        // Choose optimizer
        let x_new = if !constraints.is_empty() {
            println!("Using Lagrange multiplier constrained optimization");
            // Constrained step
            let violations = constraints::evaluate_constraints(&geometry, constraints);
            let jacobian = constraints::build_constraint_jacobian(&geometry, constraints);

            if let Some((delta_x, lambdas)) =
                optimizer::solve_constrained_step(&hessian, &grad, &jacobian, &violations)
            {
                opt_state.lambdas = lambdas.iter().cloned().collect();

                // Apply step size limit
                let step_norm = delta_x.norm();
                if step_norm > config.max_step_size {
                    let scale = config.max_step_size / step_norm;
                    println!("current stepsize: {} is reduced to max_size {}", step_norm, config.max_step_size);
                    x_old.clone() + delta_x * scale
                } else {
                    x_old.clone() + delta_x
                }
            } else {
                // Fallback to BFGS if solver fails
                println!("Warning: Constrained step solver failed. Falling back to BFGS.");
                let adaptive_scale = if step == 0 {
                    1.0
                } else {
                    let energy_current = state1.energy - state2.energy;
                    let energy_previous = opt_state.energy_history.back().unwrap_or(&energy_current);
                    optimizer::compute_adaptive_scale(energy_current, *energy_previous, grad.norm(), step)
                };
                optimizer::bfgs_step(&x_old, &grad, &hessian, &config, adaptive_scale)
            }
        } else {
            // Choose optimizer based on switch_step configuration
            let use_bfgs = if config.switch_step >= config.max_steps {
                // Always BFGS if switch_step >= max_steps (BFGS-only mode)
                true
            } else if config.switch_step == 0 {
                // Never BFGS if switch_step = 0 (DIIS-only mode)
                false
            } else {
                // Use BFGS until switch_step, then switch to DIIS
                step < config.switch_step
            };

            if use_bfgs || !opt_state.has_enough_history() {
                if config.switch_step >= config.max_steps {
                    println!("Using BFGS optimizer (BFGS-only mode)");
                } else {
                    println!(
                        "Using BFGS optimizer (step {} < switch point {})",
                        step + 1,
                        config.switch_step
                    );
                }
                let adaptive_scale = if step == 0 {
                    1.0
                } else {
                    let energy_current = state1.energy - state2.energy;
                    let energy_previous = opt_state.energy_history.back().unwrap_or(&energy_current);
                    optimizer::compute_adaptive_scale(energy_current, *energy_previous, grad.norm(), step)
                };
                optimizer::bfgs_step(&x_old, &grad, &hessian, &config, adaptive_scale)
            } else if config.use_gediis {
                println!(
                    "Using GEDIIS optimizer (step {} >= switch point {})",
                    step + 1,
                    config.switch_step
                );
                optimizer::gediis_step(&opt_state, &config)
            } else {
                println!(
                    "Using GDIIS optimizer (step {} >= switch point {})",
                    step + 1,
                    config.switch_step
                );
                optimizer::gdiis_step(&opt_state, &config)
            }
        };

        // Update geometry
        geometry.coords = x_new.clone();

        // Check convergence
        let conv = optimizer::check_convergence(
            state1.energy,
            state2.energy,
            &x_old,
            &x_new,
            &grad,
            &config,
        );

        // Compute current values for display
        let de = (state1.energy - state2.energy).abs();
        let disp_vec = &x_new - &x_old;
        let rms_disp = disp_vec.norm() / (disp_vec.len() as f64).sqrt();
        let max_disp = disp_vec.iter().map(|x| x.abs()).fold(0.0, f64::max);
        let rms_grad = grad.norm() / (grad.len() as f64).sqrt();
        let max_grad = grad.iter().map(|x| x.abs()).fold(0.0, f64::max);

        // Print energy and convergence status
        println!(
            "E1 = {:.8}, E2 = {:.8}, ΔE = {:.8}",
            state1.energy,
            state2.energy,
            de
        );
        print_convergence_status(&conv, de, rms_grad, max_grad, rms_disp, max_disp, &config);

        if conv.is_converged() {
            println!("\n****Congrats! MECP has converged****");
            println!("Final geometry saved to final.xyz");
            io::write_xyz(&geometry, Path::new("final.xyz"))?;
            return Ok(());
        }

        if step >= config.max_steps {
            break;
        }
    }

    Err("Maximum steps exceeded".into())
}

fn run_coordinate_driving(
    input_data: parser::InputData,
    qm: &dyn qm_interface::QMInterface,
    _job_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n****Running Coordinate Driving****");

    let config = &input_data.config;
    let geometry = &input_data.geometry;

    // Parse drive coordinate
    if config.drive_atoms.is_empty() || config.drive_type.is_empty() {
        return Err("Coordinate driving requires drive_type and drive_atoms parameters".into());
    }

    let coord_type = match config.drive_type.as_str() {
        "bond" => reaction_path::CoordinateType::Bond,
        "angle" => reaction_path::CoordinateType::Angle,
        "dihedral" => reaction_path::CoordinateType::Dihedral,
        _ => return Err(format!("Unknown drive_type: {}", config.drive_type).into()),
    };

    println!("Driving coordinate: {:?}", coord_type);

    let drive_coord = reaction_path::DriveCoordinate::new(
        coord_type,
        config.drive_atoms.clone(),
        config.drive_start,
    );
    println!(
        "Atoms: {:?}",
        config
            .drive_atoms
            .iter()
            .map(|&x| x + 1)
            .collect::<Vec<_>>()
    );
    println!(
        "From {:.3} to {:.3} in {} steps",
        config.drive_start, config.drive_end, config.drive_steps
    );

    // Generate reaction path
    let path = reaction_path::drive_coordinate(
        geometry,
        &drive_coord,
        config.drive_start,
        config.drive_end,
        config.drive_steps,
    );

    println!(
        "Generated {} geometries along reaction coordinate",
        path.len()
    );

    // Build headers
    let header_a = io::build_program_header(
        config,
        config.charge1,
        config.mult1,
        &config.td1,
        config.state1,
    );
    let header_b = io::build_program_header(
        config,
        config.charge2,
        config.mult2,
        &config.td2,
        config.state2,
    );

    // Run calculations along the path
    let mut energies_a = Vec::new();
    let mut energies_b = Vec::new();

    for (i, geom) in path.iter().enumerate() {
        let step = i + 1;
        println!("\n****Step {}/{}****", step, path.len());

        // Current coordinate value
        let current_value = drive_coord.current_value(geom);
        println!("Coordinate value: {:.3}", current_value);

        // Write input files
        let ext = get_input_file_extension(input_data.config.program);
        let drive_a_path = format!("job_dir/drive_{}_A.{}", step, ext);
        let drive_b_path = format!("job_dir/drive_{}_B.{}", step, ext);

        qm.write_input(geom, &header_a, &input_data.tail1, Path::new(&drive_a_path))?;
        qm.write_input(geom, &header_b, &input_data.tail2, Path::new(&drive_b_path))?;

        // Run calculations
        qm.run_calculation(Path::new(&drive_a_path))?;
        qm.run_calculation(Path::new(&drive_b_path))?;

        // Read results
        let output_ext = get_output_file_base(config.program);
        let state_a = qm.read_output(
            Path::new(&format!("job_dir/drive_{}_A.{}", step, output_ext)),
            config.state1,
        )?;
        let state_b = qm.read_output(
            Path::new(&format!("job_dir/drive_{}_B.{}", step, output_ext)),
            config.state2,
        )?;

        energies_a.push(state_a.energy);
        energies_b.push(state_b.energy);

        println!(
            "E1 = {:.8}, E2 = {:.8}, ΔE = {:.8}",
            state_a.energy,
            state_b.energy,
            state_a.energy - state_b.energy
        );
    }

    // Print energy profile
    println!("\n****Energy Profile Along Reaction Coordinate****");
    println!("Step | Coordinate |    E1    |    E2    |   ΔE   ");
    println!("-----|------------|----------|----------|--------");

    for (i, (&ea, &eb)) in energies_a.iter().zip(energies_b.iter()).enumerate() {
        let coord_value = config.drive_start
            + (config.drive_end - config.drive_start) * (i as f64)
                / ((config.drive_steps - 1) as f64);
        println!(
            "{:4} | {:>10.3} | {:>8.3} | {:>8.3} | {:>7.3}",
            i + 1,
            coord_value,
            ea,
            eb,
            ea - eb
        );
    }

    // Analyze path
    let stats = reaction_path::analyze_reaction_path(&path);
    println!("\n****Path Statistics****");
    println!("Total path length: {:.3} Å", stats.path_length);
    println!("Number of points: {}", stats.num_points);

    Ok(())
}

fn run_path_optimization(
    input_data: parser::InputData,
    qm: &dyn qm_interface::QMInterface,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n****Running Path Optimization****");

    let config = &input_data.config;
    let geometry = &input_data.geometry;

    // For path optimization, we need an initial path
    // This could come from coordinate driving or be provided as multiple geometries
    // For now, we'll create a simple initial path using coordinate driving

    if config.drive_atoms.is_empty() || config.drive_type.is_empty() {
        return Err("Path optimization requires drive_type and drive_atoms parameters to create initial path".into());
    }

    let coord_type = match config.drive_type.as_str() {
        "bond" => reaction_path::CoordinateType::Bond,
        "angle" => reaction_path::CoordinateType::Angle,
        "dihedral" => reaction_path::CoordinateType::Dihedral,
        _ => return Err(format!("Unknown drive_type: {}", config.drive_type).into()),
    };

    let drive_coord = reaction_path::DriveCoordinate::new(
        coord_type,
        config.drive_atoms.clone(),
        config.drive_start,
    );

    println!("Creating initial path via coordinate driving...");
    let initial_path = reaction_path::drive_coordinate(
        geometry,
        &drive_coord,
        config.drive_start,
        config.drive_end,
        config.drive_steps,
    );

    println!("Initial path created with {} points", initial_path.len());

    // Optimize the path using NEB
    println!("Optimizing path using Nudged Elastic Band (NEB) method...");
    let optimized_path =
        reaction_path::optimize_reaction_path(&initial_path, &input_data.constraints);

    println!(
        "Path optimization completed. Optimized path has {} points",
        optimized_path.len()
    );

    // Build headers for QM calculations
    let header_a = io::build_program_header(
        config,
        config.charge1,
        config.mult1,
        &config.td1,
        config.state1,
    );
    let header_b = io::build_program_header(
        config,
        config.charge2,
        config.mult2,
        &config.td2,
        config.state2,
    );

    // Run calculations along the optimized path
    let mut energies_a = Vec::new();
    let mut energies_b = Vec::new();

    for (i, geom) in optimized_path.iter().enumerate() {
        let step = i + 1;
        println!(
            "\n****Optimized Path Step {}/{}****",
            step,
            optimized_path.len()
        );

        // Write input files
        let ext = get_input_file_extension(input_data.config.program);
        let neb_a_path = format!("job_dir/neb_{}_A.{}", step, ext);
        let neb_b_path = format!("job_dir/neb_{}_B.{}", step, ext);

        qm.write_input(geom, &header_a, &input_data.tail1, Path::new(&neb_a_path))?;
        qm.write_input(geom, &header_b, &input_data.tail2, Path::new(&neb_b_path))?;

        // Run calculations
        qm.run_calculation(Path::new(&neb_a_path))?;
        qm.run_calculation(Path::new(&neb_b_path))?;

        // Read results
        let output_ext = get_output_file_base(config.program);
        let state_a = qm.read_output(
            Path::new(&format!("job_dir/neb_{}_A.{}", step, output_ext)),
            config.state1,
        )?;
        let state_b = qm.read_output(
            Path::new(&format!("job_dir/neb_{}_B.{}", step, output_ext)),
            config.state2,
        )?;

        energies_a.push(state_a.energy);
        energies_b.push(state_b.energy);

        println!(
            "E1 = {:.8}, E2 = {:.8}, ΔE = {:.8}",
            state_a.energy,
            state_b.energy,
            state_a.energy - state_b.energy
        );
    }

    // Print optimized energy profile
    println!("\n****Optimized Path Energy Profile****");
    println!("Step |    E1    |    E2    |   ΔE   ");
    println!("-----|----------|----------|--------");

    for (i, (&ea, &eb)) in energies_a.iter().zip(energies_b.iter()).enumerate() {
        println!("{:4} | {:>8.3} | {:>8.3} | {:>7.3}", i + 1, ea, eb, ea - eb);
    }

    // Analyze optimized path
    let stats = reaction_path::analyze_reaction_path(&optimized_path);
    println!("\n****Optimized Path Statistics****");
    println!("Total path length: {:.3} Å", stats.path_length);
    println!("Number of points: {}", stats.num_points);

    Ok(())
}

fn run_fix_de(
    _input_data: parser::InputData,
    _qm: &dyn qm_interface::QMInterface,
) -> Result<(), Box<dyn std::error::Error>> {
    Err("Fix-dE optimization is temporarily unavailable and needs to be reimplemented with the new constraint handling system.".into())
}
