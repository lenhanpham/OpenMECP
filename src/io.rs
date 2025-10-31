
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

/// Builds a Gaussian input file header string.
///
/// This function constructs the route section and title card for a Gaussian
/// input file based on the provided configuration and state-specific parameters.
/// It includes:
/// - Checkpoint file specification (`%chk`)
/// - Processor and memory allocation (`%nprocshared`, `%mem`)
/// - QM method and basis set
/// - `guess=read` for restarting calculations (except `NoRead` mode)
/// - Solvent model (SCRF)
/// - Dispersion correction
/// - Charge and multiplicity
///
/// # Arguments
///
/// * `config` - The global configuration for the MECP calculation
/// * `charge` - Molecular charge for the current state
/// * `mult` - Spin multiplicity for the current state
/// * `td` - TD-DFT keywords (e.g., "TD(NStates=5,Root=1)")
///
/// # Returns
///
/// Returns a `String` containing the formatted Gaussian input header.
///
/// # Examples
///
/// ```
/// use omecp::config::{Config, QMProgram, RunMode};
/// use omecp::io;
///
/// let mut config = Config::default();
/// config.method = "B3LYP".to_string();
/// config.basis_set = "6-31G*".to_string();
/// config.nprocs = 8;
/// config.mem = "8GB".to_string();
/// config.run_mode = RunMode::Normal;
///
/// let header = io::build_gaussian_header(&config, 0, 1, "");
/// println!("{}", header);
/// ```
pub fn build_gaussian_header(
    config: &crate::config::Config,
    charge: i32,
    mult: usize,
    td: &str,
) -> String {
    let mut method_str = format!("{} force", config.method);

    // Add basis set if specified
    if !config.basis_set.is_empty() {
        method_str = format!("{}/{}", method_str.trim_end_matches(" force"), config.basis_set);
    }

    // Add guess=read for all modes except noread
    if config.run_mode != crate::config::RunMode::NoRead {
        method_str.push_str(" guess=read");
    }

    // Add solvent model if specified
    if !config.solvent.is_empty() {
        method_str.push_str(&format!(" SCRF=(PCM,Solvent={})", config.solvent));
    }

    // Add dispersion correction if specified
    if !config.dispersion.is_empty() {
        method_str.push_str(&format!(" EmpiricalDispersion={}", config.dispersion));
    }

    format!(
        "%chk=calc.chk\n%nprocshared={}\n%mem={}\n# {} {} nosymm\n\n Title Card \n\n{} {}",
        config.nprocs, config.mem, method_str, td, charge, mult
    )
}
