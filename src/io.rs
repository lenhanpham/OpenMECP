
use crate::geometry::Geometry;
use std::fs;
use std::io::Result;
use std::path::Path;

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
