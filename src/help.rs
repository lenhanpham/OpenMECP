//! Comprehensive help system for MECP
//!
//! This module provides detailed documentation for all keywords, methods,
//! and features of the MECP program.

use std::collections::HashMap;

/// Category for organizing keywords
/// Category for organizing keywords in the help system.
///
/// This enum helps categorize different configuration keywords for easier
/// navigation and understanding in the built-in help documentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeywordCategory {
    /// Keywords that are generally required for a calculation.
    Required,
    /// Keywords that are optional and have default values.
    Optional,
    /// Keywords related to convergence thresholds and criteria.
    Convergence,
    /// Keywords for specifying quantum chemistry program commands.
    Program,
    /// Keywords specific to Time-Dependent Density Functional Theory (TD-DFT).
    TdDft,
    /// Keywords for defining geometric constraints.
    Constraints,
    /// Keywords for advanced or specialized features.
    Advanced,
}

/// Keyword documentation entry
#[derive(Debug, Clone)]
/// Documentation entry for a single configuration keyword.
///
/// This struct holds all relevant information for a keyword, including its
/// name, category, description, default value, example usage, and whether it
/// is a required parameter.
pub struct Keyword {
    /// The name of the keyword (e.g., "nprocs", "method").
    pub name: &'static str,
    /// The category to which this keyword belongs.
    pub category: KeywordCategory,
    /// A brief description of what the keyword does.
    pub description: &'static str,
    /// The default value of the keyword, if applicable.
    pub default_value: Option<&'static str>,
    /// An example of how to use the keyword in an input file.
    pub example: Option<&'static str>,
    /// Indicates whether the keyword is required for a calculation.
    pub required: bool,
}

/// QM Program information
#[derive(Debug, Clone)]
/// Information about a supported quantum chemistry program.
///
/// This struct provides details for each QM program that OpenMECP can interface
/// with, including its name, a description, the executable command, and a list
/// of key features.
pub struct ProgramInfo {
    /// The name of the QM program (e.g., "Gaussian", "ORCA").
    pub name: &'static str,
    /// A brief description of the program.
    pub description: &'static str,
    /// The typical executable command for the program.
    pub executable: &'static str,
    /// A list of key features or capabilities of the program.
    pub features: &'static [&'static str],
}

/// Method information
#[derive(Debug, Clone)]
/// Information about a quantum chemistry method.
///
/// This struct provides details for each QM method, including its name,
/// category (e.g., "DFT", "Post-HF"), a description, the programs that
/// support it, and an example of its usage.
pub struct MethodInfo {
    /// The name of the method (e.g., "B3LYP", "MP2").
    pub name: &'static str,
    /// The category of the method (e.g., "DFT", "Post-HF", "Multireference").
    pub category: &'static str,
    /// A brief description of the method.
    pub description: &'static str,
    /// A list of QM programs that support this method.
    pub programs: &'static [&'static str],
    /// An example of how to specify this method in the input.
    pub example: Option<&'static str>,
}

/// Feature information
#[derive(Debug, Clone)]
/// Information about a specific feature of OpenMECP.
///
/// This struct provides details for each major feature, including its name,
/// a description, how to enable or use it, and an example.
pub struct FeatureInfo {
    /// The name of the feature (e.g., "Normal Mode", "Restart").
    pub name: &'static str,
    /// A brief description of the feature.
    pub description: &'static str,
    /// Instructions on how to use or enable the feature.
    pub usage: &'static str,
    /// An example demonstrating the feature's usage.
    pub example: Option<&'static str>,
}

/// All keyword documentation
pub const KEYWORDS: &[Keyword] = &[
    // REQUIRED PARAMETERS
    Keyword {
        name: "nprocs",
        category: KeywordCategory::Required,
        description: "Number of processors for parallel quantum chemistry calculation",
        default_value: Some("1"),
        example: Some("nprocs = 30"),
        required: false,
    },
    Keyword {
        name: "mem",
        category: KeywordCategory::Required,
        description: "Memory allocation for QM calculation (format depends on program)",
        default_value: Some("\"1GB\""),
        example: Some("mem = \"120GB\"  # Gaussian\nmem = \"8000\"   # ORCA (MB)"),
        required: false,
    },
    Keyword {
        name: "method",
        category: KeywordCategory::Required,
        description: "Quantum chemistry method and basis set (e.g., 'B3LYP/6-31G*')",
        default_value: None,
        example: Some("method = \"B3LYP/6-31G*\"\nmethod = \"n scf(maxcycle=500,xqc) uwb97xd/def2svpp\""),
        required: true,
    },
    Keyword {
        name: "charge",
        category: KeywordCategory::Required,
        description: "Molecular charge for both electronic states",
        default_value: Some("0"),
        example: Some("charge = 0\ncharge = 1"),
        required: false,
    },
    Keyword {
        name: "mult1",
        category: KeywordCategory::Required,
        description: "Spin multiplicity for the first electronic state (2S+1, where S is spin)",
        default_value: Some("1"),
        example: Some("mult1 = 1  # Singlet\nmult1 = 3  # Triplet"),
        required: false,
    },
    Keyword {
        name: "mult2",
        category: KeywordCategory::Required,
        description: "Spin multiplicity for the second electronic state",
        default_value: Some("1"),
        example: Some("mult2 = 1  # Singlet\nmult2 = 3  # Triplet"),
        required: false,
    },
    Keyword {
        name: "program",
        category: KeywordCategory::Required,
        description: "Quantum chemistry program to use for calculations",
        default_value: Some("\"gaussian\""),
        example: Some("program = gaussian\nprogram = orca"),
        required: false,
    },
    Keyword {
        name: "mode",
        category: KeywordCategory::Required,
        description: "Running mode for MECP optimization",
        default_value: Some("\"normal\""),
        example: Some("mode = normal\nmode = stable\nmode = read"),
        required: false,
    },

    // OPTIONAL PARAMETERS
    Keyword {
        name: "td1",
        category: KeywordCategory::Optional,
        description: "TD-DFT keywords for first state (Gaussian only, specified in *tail1 for ORCA)",
        default_value: Some("\".\""),
        example: Some("td1 = \"nstates=10\""),
        required: false,
    },
    Keyword {
        name: "td2",
        category: KeywordCategory::Optional,
        description: "TD-DFT keywords for second state (Gaussian only, specified in *tail2 for ORCA)",
        default_value: Some("\".\""),
        example: Some("td2 = \"nstates=10\""),
        required: false,
    },
    Keyword {
        name: "mp2",
        category: KeywordCategory::Optional,
        description: "Enable MP2 or doubly hybrid DFT calculation (Gaussian only)",
        default_value: Some("false"),
        example: Some("mp2 = true"),
        required: false,
    },
    Keyword {
        name: "bagel_model",
        category: KeywordCategory::Optional,
        description: "Path to BAGEL model input file (required when program=bagel)",
        default_value: None,
        example: Some("bagel_model = \"model.inp\""),
        required: false,
    },
    Keyword {
        name: "state1",
        category: KeywordCategory::Optional,
        description: "State index for multireference calculations in BAGEL (0-based)",
        default_value: Some("0"),
        example: Some("state1 = 0"),
        required: false,
    },
    Keyword {
        name: "state2",
        category: KeywordCategory::Optional,
        description: "State index for multireference calculations in BAGEL (0-based)",
        default_value: Some("1"),
        example: Some("state2 = 1"),
        required: false,
    },

    // PROGRAM COMMANDS
    Keyword {
        name: "gau_comm",
        category: KeywordCategory::Program,
        description: "Gaussian executable command",
        default_value: Some("\"g16\""),
        example: Some("gau_comm = \"g16\""),
        required: false,
    },
    Keyword {
        name: "orca_comm",
        category: KeywordCategory::Program,
        description: "ORCA executable command",
        default_value: Some("\"orca\""),
        example: Some("orca_comm = \"/opt/orca5/orca\""),
        required: false,
    },
    Keyword {
        name: "xtb_comm",
        category: KeywordCategory::Program,
        description: "xTB executable command",
        default_value: Some("\"xtb\""),
        example: Some("xtb_comm = \"xtb\""),
        required: false,
    },
    Keyword {
        name: "bagel_comm",
        category: KeywordCategory::Program,
        description: "BAGEL executable command (includes MPI options)",
        default_value: Some("\"bagel\""),
        example: Some("bagel_comm = \"mpirun -np 36 /opt/bagel/bin/BAGEL\""),
        required: false,
    },

    // CONVERGENCE THRESHOLDS
    Keyword {
        name: "de_thresh",
        category: KeywordCategory::Convergence,
        description: "Energy difference threshold (|E1-E2|) for convergence (in Hartree)",
        default_value: Some("0.000050"),
        example: Some("de_thresh = 0.000050"),
        required: false,
    },
    Keyword {
        name: "rms_thresh",
        category: KeywordCategory::Convergence,
        description: "RMS displacement threshold for convergence (in Angstrom)",
        default_value: Some("0.0025"),
        example: Some("rms_thresh = 0.0025"),
        required: false,
    },
    Keyword {
        name: "max_dis_thresh",
        category: KeywordCategory::Convergence,
        description: "Maximum atomic displacement threshold for convergence (in Angstrom)",
        default_value: Some("0.004"),
        example: Some("max_dis_thresh = 0.004"),
        required: false,
    },
    Keyword {
        name: "max_g_thresh",
        category: KeywordCategory::Convergence,
        description: "Maximum gradient component threshold for convergence (Hartree/Bohr)",
        default_value: Some("0.0007"),
        example: Some("max_g_thresh = 0.0007"),
        required: false,
    },
    Keyword {
        name: "rms_g_thresh",
        category: KeywordCategory::Convergence,
        description: "RMS gradient threshold for convergence (Hartree/Bohr)",
        default_value: Some("0.0005"),
        example: Some("rms_g_thresh = 0.0005"),
        required: false,
    },
    Keyword {
        name: "max_steps",
        category: KeywordCategory::Convergence,
        description: "Maximum number of optimization steps",
        default_value: Some("100"),
        example: Some("max_steps = 300"),
        required: false,
    },
    Keyword {
        name: "max_step_size",
        category: KeywordCategory::Convergence,
        description: "Maximum step size for geometry update (in Angstrom)",
        default_value: Some("0.1"),
        example: Some("max_step_size = 0.1"),
        required: false,
    },
    Keyword {
        name: "max_history",
        category: KeywordCategory::Convergence,
        description: "Maximum number of history entries for DIIS optimizers (GDIIS/GEDIIS). Controls how many previous iterations are retained for interpolation. Larger values can improve convergence but use more memory.",
        default_value: Some("5"),
        example: Some("max_history = 5"),
        required: false,
    },
    Keyword {
        name: "reduced_factor",
        category: KeywordCategory::Convergence,
        description: "Factor for reducing GDIIS step size when RMS gradient is near convergence",
        default_value: Some("0.5"),
        example: Some("reduced_factor = 0.5"),
        required: false,
    },
    Keyword {
        name: "use_gediis",
        category: KeywordCategory::Convergence,
        description: "Enable GEDIIS (energy-weighted DIIS) optimizer instead of GDIIS",
        default_value: Some("false"),
        example: Some("use_gediis = true"),
        required: false,
    },
    Keyword {
        name: "use_hybrid_gediis",
        category: KeywordCategory::Convergence,
        description: "Use hybrid GEDIIS (50% GDIIS + 50% GEDIIS) instead of pure GEDIIS. Matches Python MECP.py behavior.",
        default_value: Some("true"),
        example: Some("use_hybrid_gediis = true  # Hybrid mode\nuse_hybrid_gediis = false  # Pure GEDIIS"),
        required: false,
    },
    Keyword {
        name: "switch_step",
        category: KeywordCategory::Convergence,
        description: "Step number to switch from BFGS to DIIS optimizers (0=DIIS-only, >=max_steps=BFGS-only)",
        default_value: Some("3"),
        example: Some("switch_step = 10  # BFGS for steps 1-10, then DIIS"),
        required: false,
    },

    // ADVANCED OPTIONS
    Keyword {
        name: "checkpoint",
        category: KeywordCategory::Advanced,
        description: "Checkpoint file path for saving/restarting calculations",
        default_value: Some("\"mecp.chk\""),
        example: Some("checkpoint = \"restart.chk\""),
        required: false,
    },
    Keyword {
        name: "restart",
        category: KeywordCategory::Advanced,
        description: "Restart optimization from checkpoint file",
        default_value: Some("false"),
        example: Some("restart = true"),
        required: false,
    },
    Keyword {
        name: "print_checkpoint",
        category: KeywordCategory::Advanced,
        description: "Enable or disable checkpoint JSON file generation (supports true/false, yes/no, 1/0)",
        default_value: Some("true"),
        example: Some("print_checkpoint = false"),
        required: false,
    },
    Keyword {
        name: "fixedatoms",
        category: KeywordCategory::Constraints,
        description: "Comma-separated list or ranges of atom indices to freeze (1-based, converted to 0-based internally)",
        default_value: None,
        example: Some("fixedatoms = \"1,5,10\"      # atoms 1, 5, and 10\nfixedatoms = \"1-5,10-15\"  # ranges"),
        required: false,
    },
    Keyword {
        name: "fix_de",
        category: KeywordCategory::Advanced,
        description: "Fix energy difference to target value (in eV) for specialized optimizations",
        default_value: Some("0.0"),
        example: Some("fix_de = 2.5  # Target ΔE = 2.5 eV"),
        required: false,
    },

    // SCAN PARAMETERS
    Keyword {
        name: "drive_coordinate",
        category: KeywordCategory::Advanced,
        description: "Coordinate specification for reaction path following",
        default_value: None,
        example: Some("drive_coordinate = \"R 1 2\"  # Distance between atoms 1-2"),
        required: false,
    },
    Keyword {
        name: "drive_start",
        category: KeywordCategory::Advanced,
        description: "Starting value for coordinate driving (in Angstrom or degrees)",
        default_value: Some("0.0"),
        example: Some("drive_start = 1.0"),
        required: false,
    },
    Keyword {
        name: "drive_end",
        category: KeywordCategory::Advanced,
        description: "Ending value for coordinate driving (in Angstrom or degrees)",
        default_value: Some("0.0"),
        example: Some("drive_end = 2.0"),
        required: false,
    },
    Keyword {
        name: "drive_steps",
        category: KeywordCategory::Advanced,
        description: "Number of steps for coordinate driving",
        default_value: Some("10"),
        example: Some("drive_steps = 20"),
        required: false,
    },
    Keyword {
        name: "drive_type",
        category: KeywordCategory::Advanced,
        description: "Type of coordinate driving",
        default_value: None,
        example: Some("drive_type = \"bond\"      # Bond distance\ndrive_type = \"angle\"      # Bond angle"),
        required: false,
    },
    Keyword {
        name: "drive_atoms",
        category: KeywordCategory::Advanced,
        description: "Comma-separated atom indices for driving (1-based)",
        default_value: None,
        example: Some("drive_atoms = \"1,2\"       # For bond/angle\ndrive_atoms = \"1,2,3\"     # For angle"),
        required: false,
    },

    // ONIOM PARAMETERS
    Keyword {
        name: "isoniom",
        category: KeywordCategory::Advanced,
        description: "Enable ONIOM QM/MM embedding",
        default_value: Some("false"),
        example: Some("isoniom = true"),
        required: false,
    },
    Keyword {
        name: "chargeandmultforoniom1",
        category: KeywordCategory::Advanced,
        description: "Charge and multiplicity for ONIOM layer 1",
        default_value: None,
        example: Some("chargeandmultforoniom1 = \"0 1\""),
        required: false,
    },
    Keyword {
        name: "chargeandmultforoniom2",
        category: KeywordCategory::Advanced,
        description: "Charge and multiplicity for ONIOM layer 2",
        default_value: None,
        example: Some("chargeandmultforoniom2 = \"0 1\""),
        required: false,
    },

    // CUSTOM INTERFACE
    Keyword {
        name: "custom_interface_file",
        category: KeywordCategory::Advanced,
        description: "JSON configuration file for custom QM program interface",
        default_value: None,
        example: Some("custom_interface_file = \"custom.json\""),
        required: false,
    },
];

/// Program information
pub fn get_programs() -> &'static [ProgramInfo] {
    &[
        ProgramInfo {
            name: "Gaussian",
            description: "Most widely used quantum chemistry program with extensive functionality",
            executable: "g16 (or g09, g03)",
            features: &[
                "DFT, HF, post-HF methods",
                "TD-DFT excited states",
                "MP2 and higher",
                "Solvent models (PCM, SMD)",
                "ONIOM QM/MM",
                "Force calculations",
                "Frequency analysis",
            ],
        },
        ProgramInfo {
            name: "ORCA",
            description: "Modern quantum chemistry program with efficient algorithms",
            executable: "orca",
            features: &[
                "DFT, HF, post-HF methods",
                "TD-DFT excited states",
                "Strong correlation (CASSCF, CASPT2)",
                "Solvent models",
                "Relativistic corrections",
                "EPR/NMR properties",
            ],
        },
        ProgramInfo {
            name: "xTB",
            description: "Semi-empirical tight-binding program for large systems",
            executable: "xtb",
            features: &[
                "GFN2-xTB method",
                "Very fast for large systems",
                "Semi-empirical accuracy",
                "GPU acceleration",
                "Solvation models",
            ],
        },
        ProgramInfo {
            name: "BAGEL",
            description: "Multireference quantum chemistry program for strong correlation",
            executable: "mpirun -np N /path/to/BAGEL",
            features: &[
                "CASSCF/CASPT2",
                "MRCI methods",
                "Strongly correlated systems",
                "Excited states",
                "State-specific optimizations",
            ],
        },
        ProgramInfo {
            name: "Custom",
            description: "User-defined QM program via JSON configuration",
            executable: "User-specified",
            features: &[
                "Any program with text I/O",
                "Regex-based parsing",
                "Configurable via JSON",
                "Supports forces and energies",
            ],
        },
    ]
}

/// Method information
pub fn get_methods() -> &'static [MethodInfo] {
    &[
        MethodInfo {
            name: "B3LYP",
            category: "DFT",
            description: "Hybrid DFT functional, widely used for general chemistry",
            programs: &["Gaussian", "ORCA", "xTB"],
            example: Some("method = \"B3LYP/6-31G*\""),
        },
        MethodInfo {
            name: "wB97XD",
            category: "DFT",
            description: "Long-range corrected DFT functional with dispersion",
            programs: &["Gaussian"],
            example: Some("method = \"wB97XD/def2-SVP\""),
        },
        MethodInfo {
            name: "PBE0",
            category: "DFT",
            description: "Hybrid DFT functional with 25% exact exchange",
            programs: &["Gaussian", "ORCA"],
            example: Some("method = \"PBE0/def2-TZVP\""),
        },
        MethodInfo {
            name: "MP2",
            category: "Post-HF",
            description: "Second-order Møller-Plesset perturbation theory",
            programs: &["Gaussian", "ORCA"],
            example: Some("mp2 = true  # In Gaussian"),
        },
        MethodInfo {
            name: "CASSCF",
            category: "Multireference",
            description: "Complete Active Space Self-Consistent Field",
            programs: &["ORCA", "BAGEL"],
            example: Some("# In ORCA: %casscf nel 6 norb 6 end"),
        },
        MethodInfo {
            name: "GFN2-xTB",
            category: "Semi-empirical",
            description: "Semi-empirical tight-binding method for large systems",
            programs: &["xTB"],
            example: Some("# Automatically used with xTB program"),
        },
    ]
}

/// Feature information
pub const FEATURES: &[FeatureInfo] = &[
    FeatureInfo {
        name: "Normal Mode",
        description: "Standard MECP optimization between two electronic states",
        usage: "mode = normal",
        example: Some("mode = normal"),
    },
    FeatureInfo {
        name: "Stability Analysis",
        description: "Check wavefunction stability before optimization",
        usage: "mode = stable",
        example: Some("mode = stable  # Runs stability check first"),
    },
    FeatureInfo {
        name: "Restart",
        description: "Restart optimization from saved checkpoint",
        usage: "restart = true",
        example: Some("checkpoint = \"restart.chk\"\nrestart = true"),
    },
    FeatureInfo {
        name: "Configuration File (omecp_config.cfg)",
        description: "Hierarchical configuration system with local/user/system precedence",
        usage: "Create template: omecp ci omecp_config.cfg",
        example: Some("# See omecp_config.cfg template for all options"),
    },
    FeatureInfo {
        name: "Parameter Display",
        description: "Automatic display of all configuration parameters and settings source at startup",
        usage: "Enabled by default, shows which config file is loaded",
        example: Some("Displays: source file, all parameters, thresholds, settings"),
    },
    FeatureInfo {
        name: "Debug Logging",
        description: "Optional file-based debug logging with dynamic filenames",
        usage: "Enable in omecp_config.cfg: file_logging = true",
        example: Some("Creates: omecp_debug_<input_basename>.log"),
    },
    FeatureInfo {
        name: "BFGS Optimizer",
        description: "Quasi-Newton method used for first 3 steps",
        usage: "Used automatically for initial steps",
        example: None,
    },
    FeatureInfo {
        name: "GDIIS Optimizer",
        description: "Geometry Direct Inversion of Iterative Subspace for later steps",
        usage: "use_gediis = false (default)",
        example: Some("use_gediis = false"),
    },
    FeatureInfo {
        name: "GEDIIS Optimizer",
        description: "Energy-weighted DIIS for improved convergence",
        usage: "use_gediis = true",
        example: Some("use_gediis = true  # Enables GEDIIS"),
    },
    FeatureInfo {
        name: "Flexible Optimizer Switching",
        description: "Control when to switch from BFGS to DIIS optimizers",
        usage: "switch_step = N (default: 3)",
        example: Some("switch_step = 0   # DIIS-only\nswitch_step = 10  # BFGS for 10 steps\nswitch_step = 999 # BFGS-only"),
    },
    FeatureInfo {
        name: "Bond Constraints",
        description: "Fix bond distances during optimization",
        usage: "R i j value (in Angstrom)",
        example: Some("R 1 2 1.5  # Fix distance between atoms 1-2 to 1.5 Å"),
    },
    FeatureInfo {
        name: "Angle Constraints",
        description: "Fix bond angles during optimization",
        usage: "A i j k value (in degrees)",
        example: Some("A 1 2 3 109.5  # Fix angle 1-2-3 to 109.5°"),
    },
    FeatureInfo {
        name: "PES Scanning",
        description: "Scan potential energy surface along 1D or 2D coordinate",
        usage: "S R i j start points step  OR  S A i j k start points step",
        example: Some("S R 1 2 1.0 10 0.1  # Scan R(1,2) from 1.0 Å, 10 steps"),
    },
    FeatureInfo {
        name: "Coordinate Driving",
        description: "Drive along reaction coordinate",
        usage: "drive_type, drive_atoms, drive_start, drive_end, drive_steps",
        example: Some("drive_type = \"bond\"\ndrive_atoms = \"1,2\"\ndrive_start = 1.0\ndrive_end = 2.0"),
    },
    FeatureInfo {
        name: "LST Interpolation",
        description: "Linear/Quadratic Synchronous Transit interpolation",
        usage: "Provide *lst1 and *lst2 geometries in input file",
        example: Some("*lst1\n# Geometry 1\n*\n*lst2\n# Geometry 2\n*"),
    },
    FeatureInfo {
        name: "External Geometry",
        description: "Load geometry from external file (.xyz, .log, .gjf)",
        usage: "@filename in *geom section",
        example: Some("*geom\n@/path/to/molecule.xyz\n*"),
    },
    FeatureInfo {
        name: "Fixed Atoms",
        description: "Freeze specific atoms during optimization",
        usage: "fixedatoms = \"list\"",
        example: Some("fixedatoms = \"1,2,3\"  # Fix first 3 atoms"),
    },
    FeatureInfo {
        name: "ONIOM QM/MM",
        description: "Embed molecule in MM environment",
        usage: "isoniom = true with layer definitions",
        example: Some("isoniom = true\nchargeandmultforoniom1 = \"0 1\""),
    },
    FeatureInfo {
        name: "Custom Interface",
        description: "Define custom QM program via JSON configuration",
        usage: "custom_interface_file = \"config.json\"",
        example: Some("# See documentation for JSON format"),
    },
];

/// Print global help
pub fn print_global_help() {
    println!("MECP - Minimum Energy Crossing Point Optimization Tool");
    println!();
    println!("USAGE:");
    println!("    omecp [OPTIONS] <COMMAND>");
    println!();
    println!("COMMANDS:");
    println!("    ci <geometry_file> [output_file]");
    println!("                        Create a template input file from a geometry file");
    println!("                        Supported formats: .xyz, .log, .gjf");
    println!();
    println!("    ci omecp_config.cfg");
    println!("                        Create a configuration template file");
    println!("                        See 'Configuration File' section below");
    println!();
    println!("    <input_file>");
    println!("                        Run MECP optimization using the input file");
    println!();
    println!("OPTIONS:");
    println!("    -h, --help [topic]   Show help. Topics: keywords, methods, features, examples");
    println!();
    println!("CONFIGURATION FILE:");
    println!("    OpenMECP uses 'omecp_config.cfg' for program configuration.");
    println!("    Create template:     omecp ci omecp_config.cfg");
    println!("    Supported locations:");
    println!("      - ./omecp_config.cfg (local, highest priority)");
    println!("      - ~/.config/omecp/omecp_config.cfg (user)");
    println!("      - /etc/omecp/omecp_config.cfg (system)");
    println!("    Features: file extensions, logging, cleanup, debug output");
    println!("    Parameters are automatically displayed at startup");
    println!();
    println!("EXAMPLES:");
    println!("    Create template:     omecp ci molecule.xyz");
    println!("    Create with name:    omecp ci molecule.xyz custom.inp");
    println!("    Run MECP:            omecp input.inp > output.log");
    println!("    Create settings:     omecp ci omecp_config.cfg");
    println!("    View keywords:       omecp --help keywords");
    println!("    View methods:        omecp --help methods");
    println!("    View features:       omecp --help features");
    println!();
}

/// Print help for 'ci' command
pub fn print_ci_help() {
    println!("Create Input Template (ci) Command");
    println!("═════════════════════════════════════");
    println!();
    println!("USAGE:");
    println!("    omecp ci <geometry_file> [output_file]");
    println!();
    println!("DESCRIPTION:");
    println!("    Generates a template MECP input file from a geometry file.");
    println!("    The template includes all required and optional parameters with");
    println!("    sensible defaults, ready for customization.");
    println!();
    println!("ARGUMENTS:");
    println!("    <geometry_file>      Input geometry file (required)");
    println!("                        Supported formats:");
    println!("                        - .xyz: XYZ coordinate file");
    println!("                        - .gjf: Gaussian input file");
    println!("                        - .log: Gaussian output file");
    println!();
    println!("    [output_file]        Output template file (optional)");
    println!("                        Default: <geometry_stem>.inp");
    println!();
    println!("EXAMPLES:");
    println!("    omecp ci molecule.xyz");
    println!("                        Creates 'molecule.inp' with '@molecule.xyz' reference");
    println!();
    println!("    omecp ci calc.log custom.inp");
    println!("                        Creates 'custom.inp' with '@calc.log' reference");
    println!();
    println!("OUTPUT FORMAT:");
    println!("    Generated template includes:");
    println!("    - Required parameters (nprocs, mem, method, charge, mult, mode)");
    println!("    - Optional parameters with defaults");
    println!("    - Convergence thresholds");
    println!("    - Program settings");
    println!("    - *geom section with @reference");
    println!("    - Empty *tail1 and *tail2 sections");
    println!("    - Constraint examples");
    println!();
}

/// Print keyword reference
pub fn print_keyword_help() {
    println!("KEYWORD REFERENCE");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();

    // Group keywords by category
    let mut categories: HashMap<KeywordCategory, Vec<&Keyword>> = HashMap::new();
    for keyword in KEYWORDS {
        categories
            .entry(keyword.category)
            .or_default()
            .push(keyword);
    }

    // Print each category
    for (category, keywords) in categories {
        print_category_header(category);
        println!();

        for keyword in keywords {
            print_keyword(keyword);
            println!();
        }
        println!();
    }
}

/// Print method reference
pub fn print_method_help() {
    println!("METHOD REFERENCE");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();

    let methods = get_methods();

    // Group methods by category
    let mut categories: HashMap<&str, Vec<&MethodInfo>> = HashMap::new();
    for method in methods {
        categories.entry(method.category).or_default().push(method);
    }

    for (category, methods) in categories {
        println!("{}", category.to_uppercase());
        println!("{}", "─".repeat(76));
        println!();

        for method in methods {
            println!("{}", method.name);
            println!("    {}", method.description);
            println!("    Programs: {}", method.programs.join(", "));
            if let Some(example) = method.example {
                println!("    Example:  {}", example);
            }
            println!();
        }
        println!();
    }

    println!("SUPPORTED PROGRAMS");
    println!("{}", "─".repeat(76));
    println!();

    for program in get_programs() {
        println!("{}", program.name);
        println!("    {}", program.description);
        println!("    Executable: {}", program.executable);
        println!("    Features:");
        for feature in program.features {
            println!("      • {}", feature);
        }
        println!();
    }
}

/// Print feature reference
pub fn print_feature_help() {
    println!("FEATURE REFERENCE");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();

    for feature in FEATURES {
        println!("{}", feature.name);
        println!("    {}", feature.description);
        println!("    Usage: {}", feature.usage);
        if let Some(example) = feature.example {
            println!("    Example: {}", example);
        }
        println!();
    }

    println!("RUN MODES");
    println!("{}", "─".repeat(76));
    println!();
    println!("normal");
    println!("    Standard MECP optimization mode. Runs single-point calculations");
    println!("    and optimizes geometry to find minimum energy crossing point.");
    println!();
    println!("stable");
    println!("    Pre-point stability analysis. Checks wavefunction stability before");
    println!("    optimization. Useful for avoiding convergence to saddle points.");
    println!();
    println!("read");
    println!("    Restart mode with wavefunction reading. Uses saved checkpoint to");
    println!("    continue previous optimization or start from known wavefunction.");
    println!();
    println!("noread");
    println!("    Fresh start without reading wavefunction. Useful when initial");
    println!("    guess is poor or when switching between methods.");
    println!();
    println!("inter_read");
    println!("    Intermediate read mode. Copies wavefunction from state B to state A");
    println!("    with guess=mix for better convergence in Gaussian.");
    println!();
    println!("coordinate_drive");
    println!("    Drives the system along a specified reaction coordinate without");
    println!("    full optimization. Useful for exploring reaction paths.");
    println!();
    println!("fix_de");
    println!("    Fix-dE optimization. Constrains the energy difference between states");
    println!("    to a target value while optimizing geometry.");
    println!();

    println!("OPTIMIZATION ALGORITHMS");
    println!("{}", "─".repeat(76));
    println!();
    println!("BFGS");
    println!("    Broyden-Fletcher-Goldfarb-Shanno quasi-Newton method. Used for the");
    println!("    first 3 optimization steps when Hessian is not available.");
    println!();
    println!("GDIIS");
    println!("    Geometry Direct Inversion of Iterative Subspace. Used for later");
    println!("    steps when history is available. Improves convergence.");
    println!();
    println!("GEDIIS");
    println!("    Energy-weighted DIIS. Alternative to GDIIS that weights each");
    println!("    previous step by its energy. Enable with 'use_gediis = true'.");
    println!();
    println!("OPTIMIZER SWITCHING");
    println!("    Control when to switch from BFGS to DIIS with 'switch_step = N':");
    println!("    - switch_step = 0: Pure DIIS (no BFGS)");
    println!("    - switch_step = 3: Default hybrid (BFGS steps 1-3, then DIIS)");
    println!("    - switch_step = 10: Extended BFGS (steps 1-10, then DIIS)");
    println!("    - switch_step >= max_steps: Pure BFGS (no DIIS)");
    println!();

    println!("CONSTRAINT TYPES");
    println!("{}", "─".repeat(76));
    println!();
    println!("Bond Constraints (R)");
    println!("    Fix distance between two atoms:");
    println!("    R i j value    # value in Angstrom, atoms are 1-based");
    println!();
    println!("Angle Constraints (A)");
    println!("    Fix angle formed by three atoms:");
    println!("    A i j k value  # value in degrees, atoms are 1-based");
    println!();
    println!("Fixed Atoms");
    println!("    Freeze specific atoms during optimization:");
    println!("    fixedatoms = \"1,2,3\"");
    println!("    fixedatoms = \"1-5,10-15\"  # Can use ranges");
    println!();
    println!("PES Scans");
    println!("    Scan along 1D or 2D coordinate:");
    println!("    S R i j start num_points step_size");
    println!("    S A i j k start num_points step_size");
    println!();
}

/// Print example usages
pub fn print_examples() {
    println!("USAGE EXAMPLES");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();

    println!("BASIC USAGE");
    println!("{}", "─".repeat(76));
    println!();
    println!("1. Create template from XYZ file:");
    println!("   $ omecp ci water.xyz");
    println!("   → Creates water.inp with @water.xyz reference");
    println!();
    println!("2. Run MECP optimization:");
    println!("   $ omecp calculation.inp");
    println!();
    println!("3. View help:");
    println!("   $ omecp --help");
    println!("   $ omecp --help keywords");
    println!();

    println!("INPUT FILE EXAMPLE");
    println!("{}", "─".repeat(76));
    println!();
    println!("# Required parameters");
    println!("nprocs = 30");
    println!("mem = 120GB");
    println!("method = n scf(maxcycle=500,xqc) uwb97xd/def2svpp scrf=(smd,solvent=acetonitrile)");
    println!("charge = 1");
    println!("mult1 = 3");
    println!("mult2 = 1");
    println!("mode = normal");
    println!();
    println!("# Optional convergence thresholds");
    println!("de_thresh = 0.000050");
    println!("rms_thresh = 0.0025");
    println!("max_history = 4");
    println!();
    println!("# Program settings");
    println!("program = gaussian");
    println!("gau_comm = g16");
    println!();
    println!("# Geometry section");
    println!("*geom");
    println!("@molecule.xyz  # Or inline coordinates");
    println!("C    0.0    0.0    0.0");
    println!("H    0.0    0.0    1.0");
    println!("*");
    println!();
    println!("# Constraints (optional)");
    println!("*constr");
    println!("R 1 2 1.0  # Fix bond distance");
    println!("*");
    println!();

    println!("ADVANCED EXAMPLES");
    println!("{}", "─".repeat(76));
    println!();
    println!("1. Stability analysis mode:");
    println!("   mode = stable");
    println!();
    println!("2. Restart from checkpoint:");
    println!("   checkpoint = \"restart.chk\"");
    println!("   restart = true");
    println!();
    println!("3. Use custom QM program:");
    println!("   program = custom");
    println!("   custom_interface_file = \"my_qm.json\"");
    println!();
    println!("4. Coordinate driving:");
    println!("   mode = coordinate_drive");
    println!("   drive_type = \"bond\"");
    println!("   drive_atoms = \"1,2\"");
    println!("   drive_start = 1.0");
    println!("   drive_end = 2.0");
    println!("   drive_steps = 10");
    println!();
    println!("5. PES scan:");
    println!("   # Add to *constr section:");
    println!("   S R 1 2 1.0 10 0.1");
    println!();
    println!("6. Fix energy difference:");
    println!("   mode = fix_de");
    println!("   fix_de = 2.5  # Target ΔE in eV");
    println!();

    println!("TROUBLESHOOTING");
    println!("{}", "─".repeat(76));
    println!();
    println!("• Convergence problems:");
    println!("  - Increase max_steps");
    println!("  - Relax convergence thresholds");
    println!("  - Try stable mode for wavefunction check");
    println!("  - Use smaller max_step_size");
    println!();
    println!("• Wrong spin state:");
    println!("  - Check mult1 and mult2 values");
    println!("  - Use inter_read mode in Gaussian");
    println!();
    println!("• Large systems:");
    println!("  - Use xTB program for faster calculations");
    println!("  - Freeze non-reactive atoms with fixedatoms");
    println!("  - Use lower-cost methods for pre-optimization");
    println!();
}

/// Print category header
fn print_category_header(category: KeywordCategory) {
    match category {
        KeywordCategory::Required => {
            println!("REQUIRED PARAMETERS");
            println!("{}", "─".repeat(76));
        }
        KeywordCategory::Optional => {
            println!("OPTIONAL PARAMETERS");
            println!("{}", "─".repeat(76));
        }
        KeywordCategory::Convergence => {
            println!("CONVERGENCE THRESHOLDS");
            println!("{}", "─".repeat(76));
        }
        KeywordCategory::Program => {
            println!("PROGRAM COMMANDS");
            println!("{}", "─".repeat(76));
        }
        KeywordCategory::TdDft => {
            println!("TD-DFT PARAMETERS");
            println!("{}", "─".repeat(76));
        }
        KeywordCategory::Constraints => {
            println!("CONSTRAINT PARAMETERS");
            println!("{}", "─".repeat(76));
        }
        KeywordCategory::Advanced => {
            println!("ADVANCED PARAMETERS");
            println!("{}", "─".repeat(76));
        }
    }
}

/// Print single keyword
fn print_keyword(keyword: &Keyword) {
    let required_str = if keyword.required { " [REQUIRED]" } else { "" };

    println!("{}{}", keyword.name, required_str);
    println!("    {}", keyword.description);

    if let Some(default) = keyword.default_value {
        println!("    Default: {}", default);
    }

    if let Some(example) = keyword.example {
        println!("    Example: {}", example);
    }
}
