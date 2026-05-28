use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// Template generator for creating MECP input files from geometry files
///
/// Generate a template input file from a geometry file
/// Supports .xyz, .log, and .gjf formats
pub fn generate_template_from_file<P: AsRef<Path>>(
    geometry_file: P,
) -> Result<String, Box<dyn std::error::Error>> {
    let geometry_file = geometry_file.as_ref();

    if !geometry_file.exists() {
        return Err(format!("File not found: {}", geometry_file.display()).into());
    }

    let extension = geometry_file
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("");

    let (elements, coords) = match extension.to_lowercase().as_str() {
        "xyz" => extract_geometry_from_xyz(geometry_file)?,
        "log" => extract_geometry_from_log(geometry_file)?,
        "gjf" => extract_geometry_from_gjf(geometry_file)?,
        _ => return Err(format!("Unsupported file format: {}", extension).into()),
    };

    let geom_path = geometry_file.canonicalize()?;

    Ok(generate_template(elements, &coords, &geom_path))
}

/// Extract geometry from XYZ file
fn extract_geometry_from_xyz(
    path: &Path,
) -> Result<(Vec<String>, Vec<f64>), Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let lines: Vec<&str> = content.lines().collect();

    if lines.len() < 3 {
        return Err("Invalid XYZ file: not enough lines".into());
    }

    let num_atoms = lines[0]
        .trim()
        .parse::<usize>()
        .map_err(|_| "Invalid XYZ file: cannot parse number of atoms")?;

    let mut elements = Vec::new();
    let mut coords = Vec::new();

    for i in 2..2 + num_atoms {
        if i >= lines.len() {
            return Err("Invalid XYZ file: incomplete geometry".into());
        }

        let parts: Vec<&str> = lines[i].split_whitespace().collect();
        if parts.len() < 4 {
            return Err("Invalid XYZ file: malformed coordinate line".into());
        }

        elements.push(parts[0].to_string());
        coords.push(parts[1].parse::<f64>()?);
        coords.push(parts[2].parse::<f64>()?);
        coords.push(parts[3].parse::<f64>()?);
    }

    Ok((elements, coords))
}

/// Extract geometry from Gaussian .log file
fn extract_geometry_from_log(
    path: &Path,
) -> Result<(Vec<String>, Vec<f64>), Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;

    // Find "Input orientation" section
    let lines: Vec<&str> = content.lines().collect();
    let mut in_input_section = false;
    let mut elements = Vec::new();
    let mut coords = Vec::new();

    for line in lines {
        if line.contains("Input orientation") {
            in_input_section = true;
            continue;
        }

        if in_input_section {
            if line.contains("Distance matrix") || line.contains("Rotational constants") {
                break;
            }

            // Parse coordinate lines: atomic_number element x y z
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 5 && parts[0].parse::<usize>().is_ok() {
                let element = parts[1];
                let x = parts[3].parse::<f64>()?;
                let y = parts[4].parse::<f64>()?;
                let z = parts[5].parse::<f64>()?;

                elements.push(element.to_string());
                coords.extend_from_slice(&[x, y, z]);
            }
        }
    }

    if elements.is_empty() {
        return Err("Could not find geometry in log file".into());
    }

    Ok((elements, coords))
}

/// Extract geometry from Gaussian .gjf file
fn extract_geometry_from_gjf(
    path: &Path,
) -> Result<(Vec<String>, Vec<f64>), Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let lines: Vec<&str> = content.lines().collect();

    let mut elements = Vec::new();
    let mut coords = Vec::new();

    // Parse geometry section (after header, before empty line and tail)
    // State machine: 0=header, 1=title, 2=charge_mult, 3=geometry
    let mut state = 0;

    for line in lines {
        let trimmed = line.trim();

        match state {
            // Skip header lines (starting with % or #)
            0 => {
                if trimmed.is_empty() {
                    state = 1;
                } else if trimmed.starts_with('%') || trimmed.starts_with('#') {
                    continue;
                }
            }
            // Skip title line
            1 => {
                if trimmed.is_empty() {
                    state = 2;
                }
            }
            // Skip charge/mult line
            2 => {
                if !trimmed.is_empty() {
                    // This is the charge/mult line, advance to geometry state
                    state = 3;
                }
            }
            // Parse geometry
            3 => {
                // Empty line marks end of geometry section
                if trimmed.is_empty() {
                    break;
                }

                // Parse element and coordinates
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 4 {
                    elements.push(parts[0].to_string());
                    coords.push(parts[1].parse::<f64>()?);
                    coords.push(parts[2].parse::<f64>()?);
                    coords.push(parts[3].parse::<f64>()?);
                }
            }
            _ => {}
        }
    }

    if elements.is_empty() {
        return Err("Could not find geometry in gjf file".into());
    }

    Ok((elements, coords))
}

/// Generate the template input file content
fn generate_template(_elements: Vec<String>, _coords: &Vec<f64>, geometry_path: &Path) -> String {
    let geom_filename = geometry_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("geometry.xyz");

    format!(
        r#"#===============================================================================
# OpenMECP Input Template
#===============================================================================
# Generated from geometry: {geom_filename}
#===============================================================================

#===== Basic Settings (required) ===============================================
nprocs = 30                         # processors for QM program
mem = 120GB                         # memory (use maxcore value for ORCA)
method = n b3lyp/6-31g**            # QM method: n method/basis keywords ...
td_state_a =                        # TD-DFT keywords for state A (Gaussian). Short: td_a
td_state_b =                        # TD-DFT keywords for state B (Gaussian). Short: td_b
mp2 = false                         # true for MP2 or double-hybrid in Gaussian
charge = 1                          # molecular charge (same for both states unless charge_b)
mult_state_a = 3                    # multiplicity of state A. Short: mult_a
mult_state_b = 1                    # multiplicity of state B. Short: mult_b
mode = normal                       # normal | read | noread | stable | inter_read

#===== Convergence Thresholds ==================================================
delta_e = 0.000050                  # energy difference convergence (hartree)
rms_dis = 0.0025                    # RMS displacement convergence (angstrom)
max_dis = 0.004                     # max displacement convergence (angstrom)
max_grad = 0.001323                 # max gradient convergence (hartree/angstrom)
rms_grad = 0.000945                 # RMS gradient convergence (hartree/angstrom)

#===== Optimization Control ====================================================
max_steps = 100                     # maximum optimization steps
max_step_size = 0.1                 # maximum step size (angstrom)
max_history = 4                     # DIIS history length
reduced_factor = 0.5                # step reduction factor near convergence
step_reduction_multiplier = 10.0    # rms_g multiplier for step reduction threshold
steepest_descent_step = 0.01        # fallback step when DIIS/BFGS fails (angstrom)
bfgs_rho = 15                       # BFGS step amplification factor (rho)

#===== Optimizer Selection =====================================================
switch_step = 3                     # 0=DIIS-only, 3=BFGS→DIIS (default), >=max_steps=BFGS-only
hessian = direct_psb                # direct_psb (default) | inverse_bfgs | bofill | powell | bfgs_powell_mix
                                    # Note: blend mode requires a direct Hessian method.
use_gediis = false                  # options: false/none=GDIIS (default), true/sequential=GEDIIS,
                                    #           blend=GDIIS_blend with trust region
use_hybrid_gediis = false           # activated when use_gediis = true or blend
                                    # options: true/false
gediis_blend_mode = fixed_sequential # activated when use_gediis = blend AND use_hybrid_gediis = true
                                     # options: fixed, fixed_sequential (default), gradient, sequential
smart_history = false               # experimental: remove worst DIIS point instead of oldest

#===== Optimization Control for blend-mode trust radius only====================
trust_reduction_factor = 0.5        # trust radius contraction on energy increase
trust_increase_factor = 1.2         # trust radius expansion on energy decrease
trust_inc_threshold = 0.0001        # energy increase threshold for reduction (hartree)
trust_dec_threshold = 0.0001        # energy decrease threshold for expansion (hartree)
trust_min_radius = 0.01             # minimum trust radius (angstrom)
trust_max_radius = 1.0              # maximum trust radius (angstrom)

#===== Advanced Optimizer Settings =============================================
# ALL parameters below are commented out (= inactive). Their default values
# (shown after the = sign) are used unless you uncomment and change them.
#
#gediis_switch_rms = 0.005          # Phase 1→2: RMS gradient for GDIIS→GEDIIS switch
                                    # (active: use_gediis=true + use_hybrid_gediis=true)
#gediis_switch_step = 0.001         # Phase 2→3: RMS displacement for GEDIIS→GDIIS switch
                                    # (also used by blend's fixed_sequential mode)
#use_robust_diis = false            # Activate DIIS step validation (cosine & coefficient checks)
#gediis_variant = auto              # auto (default) | rfo | energy | simultaneous
                                    # (active: use_gediis=true + use_robust_diis=true)
#gdiis_cosine_check = standard      # none | zero | standard (default) | variable | strict
                                    # (active: use_robust_diis=true)
#gdiis_coeff_check = regular        # none | regular (default) | strict
                                    # (active: use_robust_diis=true)
#n_neg = 0                          # 0 = minimum search, 1 = transition state
#gediis_sim_switch = 0.0025         # (active: use_robust_diis=true)

#===== Program Settings ========================================================
program = gaussian                  # gaussian | orca | xtb | bagel
gau_comm = g16                      # Gaussian command
orca_comm = orca                    # ORCA command
xtb_comm = xtb                      # XTB command
bagel_comm = bagel                  # BAGEL command
bagel_model = model.inp             # BAGEL model file
#custom_interface_file = custom_qm.json  # custom QM program config

#===== Advanced Options ========================================================
#restart = false                    # restart from checkpoint file
#print_checkpoint = true            # save checkpoint JSON at each step
#fix_de = 0.0                       # eV: fix energy difference (FixDE mode)
#state_a = 0                        # state index for BAGEL multireference
#state_b = 1
#basis =                            # basis set (for programs separating method/basis)
#solvent =                          # solvent model
#dispersion =                       # dispersion correction
#charge_b =                         # separate charge for state B (default: same as charge)
#fixedatoms = 1,3,5-7               # comma/hyphen ranges of fixed atoms (1-based)

#===== ONIOM (QM/MM) ===========================================================
#isoniom = false
#chargeandmultforoniom1 = 0 1
#chargeandmultforoniom2 = 0 1

#===== Coordinate Driving ======================================================
#drive_type = bond                  # bond | angle | dihedral
#drive_atoms = 1,2                  # atom indices (1-based)
#drive_start = 1.0
#drive_end = 2.0
#drive_steps = 10

#===============================================================================
# Geometry section: Cartesian coordinates in Angstrom
#===============================================================================
*geom
@{geom_filename}
*

#===============================================================================
# Tail sections: extra keywords appended to QM input files
#===============================================================================
*tail_a
# Extra keywords for state A (e.g., SCF convergence, basis sets)
# For Gaussian: TD(NStates=5,Root=1)
# For ORCA: %tddft nroots 5 end
*

*tail_b
# Extra keywords for state B (e.g., SCF convergence, basis sets)
*

#===============================================================================
# Constraints section
# Syntax: R atom1 atom2 target      (bond constraint, Angstrom)
#         A a1 a2 a3 target         (angle constraint, degrees)
#         S R a1 a2 start num step  (1D scan)
#         S A a1 a2 a3 start num step (angle scan)
#         S R a1 a2 start1 n1 s1  R a3 a4 start2 n2 s2 (2D scan)
# Atom indices are 1-based.
#===============================================================================
*constr
#R 1 2 1.0                          # fix bond 1-2 at 1.0 Angstrom
#A 1 2 3 100.0                      # fix angle 1-2-3 at 100 degrees
#S R 1 2 1.0 10 0.1                 # scan bond 1-2 from 1.0, 10 steps of 0.1
*

#===== Output Control ==========================================================
print_level = 1                     # 0=quiet, 1=normal, 2=verbose (DIIS debug)

"#,
        geom_filename = geom_filename
    )
}

/// Write template to file
pub fn write_template_to_file<P: AsRef<Path>>(
    template: &str,
    output_path: P,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_path = output_path.as_ref();

    // Create parent directory if it doesn't exist
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    fs::write(output_path, template)?;
    Ok(())
}

/// Get default output filename based on input geometry file
///
/// This function generates a default `.inp` filename from the source geometry file.
/// To prevent overwriting QM output files when creating templates from them,
/// it adds a `_omecp` suffix when the source file has an extension that matches
/// common QM output formats (`.log`, `.out`, `.json`).
///
/// # Examples
///
/// - `abc.xyz` → `abc.inp` (no conflict)
/// - `abc.log` → `abc_omecp.inp` (prevents overwriting `abc.log` when run)
/// - `calc.out` → `calc_omecp.inp` (prevents overwriting `calc.out` when run)
/// - `result.json` → `result_omecp.inp` (prevents overwriting `result.json` when run)
pub fn get_default_output_path<P: AsRef<Path>>(geometry_file: P) -> PathBuf {
    let geometry_file = geometry_file.as_ref();
    let stem = geometry_file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("template");

    // Check if the source file has an extension that matches QM output formats
    let source_ext = geometry_file
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase());

    let is_qm_output = match source_ext.as_deref() {
        Some("log") | Some("out") | Some("json") => true,
        _ => false,
    };

    // If source is a QM output file, add suffix to prevent overwriting
    let output_stem = if is_qm_output {
        format!("{}_omecp", stem)
    } else {
        stem.to_string()
    };

    PathBuf::from(format!("{}.inp", output_stem))
}

/// Interactive prompt for user input
pub fn prompt_user(prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
    print!("{} ", prompt);
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(input.trim().to_string())
}

/// Validate file extension
pub fn is_supported_format(path: &Path) -> bool {
    match path.extension().and_then(|s| s.to_str()) {
        Some(ext) => matches!(ext.to_lowercase().as_str(), "xyz" | "log" | "gjf"),
        None => false,
    }
}
