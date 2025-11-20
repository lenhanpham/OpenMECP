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
        r#"#This subset is required. It controls your quantum chemistry tasks.
nprocs = 30 #processors
mem = 120GB # memory to be used. change this into the maxcore value if you want to use ORCA
method = n scf(maxcycle=500,xqc) uwb97xd/def2svpp scrf=(smd,solvent=acetonitrile) # your keywords line. It will be presented in the job files. Don't write guess=mix or stable=opt; they will be added automatically.
td_state_a =  # keywords for TD-DFT of state A (only for Gaussian; please write it in the tail part for ORCA). Can also use short form: td_a
td_state_b = # keywords for TD-DFT of state B (only for Gaussian; please write it in the tail part for ORCA). Can also use short form: td_b
mp2 = false #set true for MP2 or doubly hybrid calculation in Gaussian
charge = 1
mult_state_a = 3 # multiplicity of state A. Can also use short form: mult_a
mult_state_b = 1 # multiplicity of state B. Can also use short form: mult_b
mode = normal #normal; stable; read; inter_read; noread

#This subset is optional. It controls the convergence threshols, and the details of the GDIIS algorithm. Shown here are the default values.
dE_thresh = 0.000050
rms_thresh = 0.0025
max_dis_thresh = 0.004
max_g_thresh = 0.0007
rms_g_thresh = 0.0005
max_steps = 300
max_step_size = 0.1
max_history = 4


reduced_factor = 0.5 # the gdiis stepsize will be reduced by this factor when rms_gradient is close to converge

# Optimization settings
switch_step = 3
use_gediis = false 
use_hybrid_gediis = true  # only effective when use_gediis = true
smart_history = false # experimental and false by default; smart_history may speed up convergence in several cases

# This subset controls which program you are using, and how to call them
program = gaussian  #gaussian, orca, xtb, bagel
gau_comm = g16
orca_comm = /opt/orca5/orca
xtb_comm = xtb
bagel_comm = mpirun -np 36 /opt/bagel/bin/BAGEL
bagel_model = model.inp

#state_a = 0 #only set it for the multireference calculation using BAGEL
#state_b = 1 #only set it for the multireference calculation using BAGEL

#Between *geom and *, write the cartesian coordinate of your initial geometry (in angstrom)
*geom
@{geom_filename}
*

#If you have anything to be put at the end of the input file, write it here. This part is especially useful for ORCA: you can add anything related to your keywords here.
*tail1
#@/home/595/np9048/sources/BasisSetGaussian/def2tzvpd.gbs/N
#6-31G(d)
#****
*
*tail2
#@/home/595/np9048/sources/BasisSetGaussian/def2tzvpd.gbs/N
*

#This subset controls the constraints. R 1 2 1.0 means to fix distance between atom 1 and 2 (start from 1) to be 1.0 angstrom.
*constr
#R 1 2 1.0
#A 1 2 3 100.0 # to fix angle 1-2-3 to be 100 degree
#S R 1 2 1.0 10 0.1 # to run a scan of R(1,2) starting from 1.0 angstrom, with 10 steps of 0.1 angstrom
#S R 2 3 1.5 10 0.1 # you can at most set a 2D-scan
*

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
pub fn get_default_output_path<P: AsRef<Path>>(geometry_file: P) -> PathBuf {
    let geometry_file = geometry_file.as_ref();
    let stem = geometry_file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("template");

    PathBuf::from(format!("{}.inp", stem))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_supported_format() {
        assert!(is_supported_format(Path::new("test.xyz")));
        assert!(is_supported_format(Path::new("test.LOG")));
        assert!(is_supported_format(Path::new("test.gjf")));
        assert!(!is_supported_format(Path::new("test.txt")));
        assert!(!is_supported_format(Path::new("test")));
    }

    #[test]
    fn test_get_default_output_path() {
        let path = get_default_output_path(Path::new("molecule.xyz"));
        assert_eq!(path.to_str().unwrap(), "molecule.inp");

        let path = get_default_output_path(Path::new("/path/to/molecule.xyz"));
        assert_eq!(path.to_str().unwrap(), "molecule.inp");
    }
}
