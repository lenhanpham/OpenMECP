use crate::geometry::Geometry;
use nalgebra::DMatrix;

/// Interpolation method for LST
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolationMethod {
    /// Linear Synchronous Transit (LST)
    Linear,
    /// Quadratic Synchronous Transit (QST) - requires third geometry
    Quadratic,
    /// Energy-weighted interpolation (requires energy values)
    EnergyWeighted,
}

/// Perform LST interpolation between two geometries with various methods
pub fn interpolate(geom1: &Geometry, geom2: &Geometry, num_points: usize, method: InterpolationMethod) -> Vec<Geometry> {
    match method {
        InterpolationMethod::Linear => interpolate_linear(geom1, geom2, num_points),
        InterpolationMethod::Quadratic => {
            // For QST, we need a third geometry (midpoint approximation)
            let midpoint = create_midpoint_geometry(geom1, geom2);
            interpolate_quadratic(geom1, &midpoint, geom2, num_points)
        },
        InterpolationMethod::EnergyWeighted => {
            // For now, fall back to linear (would need energy values)
            interpolate_linear(geom1, geom2, num_points)
        }
    }
}

/// Perform LST (Linear Synchronous Transit) interpolation between two geometries
/// Uses Kabsch algorithm for optimal alignment before interpolation
pub fn interpolate_linear(geom1: &Geometry, geom2: &Geometry, num_points: usize) -> Vec<Geometry> {
    if geom1.num_atoms != geom2.num_atoms {
        panic!("Geometries must have same number of atoms");
    }

    let n = geom1.num_atoms;

    // Convert to 3xN matrices
    let mut x = DMatrix::zeros(3, n);
    let mut y = DMatrix::zeros(3, n);

    for i in 0..n {
        let coords1 = geom1.get_atom_coords(i);
        let coords2 = geom2.get_atom_coords(i);
        x[(0, i)] = coords1[0];
        x[(1, i)] = coords1[1];
        x[(2, i)] = coords1[2];
        y[(0, i)] = coords2[0];
        y[(1, i)] = coords2[1];
        y[(2, i)] = coords2[2];
    }

    // Kabsch algorithm: find optimal rotation matrix
    let r = &y * x.transpose();
    let rt_r = r.transpose() * &r;
    let eigen = rt_r.symmetric_eigen();

    let mut mius = DMatrix::zeros(3, 3);
    for i in 0..3 {
        mius[(i, i)] = 1.0 / eigen.eigenvalues[i].sqrt();
    }

    let a = eigen.eigenvectors;
    let b = &mius * (&r * &a).transpose();
    let u = b.transpose() * &a;

    // Apply rotation to first geometry
    let x_aligned = &u * &x;

    // Linear interpolation
    let mut geometries = Vec::new();

    for i in 0..=num_points {
        let t = i as f64 / (num_points + 1) as f64;
        let interp = &x_aligned * (1.0 - t) + &y * t;

        let mut coords = Vec::new();
        for j in 0..n {
            coords.push(interp[(0, j)]);
            coords.push(interp[(1, j)]);
            coords.push(interp[(2, j)]);
        }

        geometries.push(Geometry::new(geom1.elements.clone(), coords));
    }

    geometries
}

/// Quadratic Synchronous Transit interpolation using three geometries
pub fn interpolate_quadratic(geom1: &Geometry, geom_mid: &Geometry, geom2: &Geometry, num_points: usize) -> Vec<Geometry> {
    if geom1.num_atoms != geom2.num_atoms || geom1.num_atoms != geom_mid.num_atoms {
        panic!("All geometries must have same number of atoms");
    }

    let _n = geom1.num_atoms;
    let mut geometries = Vec::new();

    // Convert all geometries to coordinate vectors
    let coords1 = geometry_to_coords(geom1);
    let coords_mid = geometry_to_coords(geom_mid);
    let coords2 = geometry_to_coords(geom2);

    for i in 0..=num_points {
        let t = i as f64 / (num_points + 1) as f64;

        // Quadratic interpolation: p(t) = (1-t)^2 * p1 + 2*(1-t)*t * p_mid + t^2 * p2
        let mut coords = Vec::new();
        for j in 0..coords1.len() {
            let val = (1.0 - t).powi(2) * coords1[j]
                    + 2.0 * (1.0 - t) * t * coords_mid[j]
                    + t.powi(2) * coords2[j];
            coords.push(val);
        }

        geometries.push(Geometry::new(geom1.elements.clone(), coords));
    }

    geometries
}

/// Create a simple midpoint geometry for QST when only two geometries are provided
fn create_midpoint_geometry(geom1: &Geometry, geom2: &Geometry) -> Geometry {
    let coords1 = geometry_to_coords(geom1);
    let coords2 = geometry_to_coords(geom2);

    let mut mid_coords = Vec::new();
    for i in 0..coords1.len() {
        mid_coords.push((coords1[i] + coords2[i]) / 2.0);
    }

    Geometry::new(geom1.elements.clone(), mid_coords)
}

/// Convert geometry to flat coordinate vector
fn geometry_to_coords(geom: &Geometry) -> Vec<f64> {
    let mut coords = Vec::new();
    for i in 0..geom.num_atoms {
        let atom_coords = geom.get_atom_coords(i);
        coords.push(atom_coords[0]);
        coords.push(atom_coords[1]);
        coords.push(atom_coords[2]);
    }
    coords
}

/// Validate that interpolated geometries are reasonable
pub fn validate_geometries(geometries: &[Geometry]) -> Result<(), String> {
    if geometries.is_empty() {
        return Err("No geometries to validate".to_string());
    }

    let num_atoms = geometries[0].num_atoms;
    let elements = &geometries[0].elements;

    for (i, geom) in geometries.iter().enumerate() {
        if geom.num_atoms != num_atoms {
            return Err(format!("Geometry {} has {} atoms, expected {}", i, geom.num_atoms, num_atoms));
        }

        if geom.elements != *elements {
            return Err(format!("Geometry {} has different elements than reference", i));
        }

        // Check for NaN or infinite coordinates
        for j in 0..geom.num_atoms {
            let coords = geom.get_atom_coords(j);
            for &coord in &coords {
                if !coord.is_finite() {
                    return Err(format!("Geometry {} atom {} has non-finite coordinate: {}", i, j, coord));
                }
            }
        }

        // Check for unreasonably large coordinates (> 1000 Å)
        for j in 0..geom.num_atoms {
            let coords = geom.get_atom_coords(j);
            for &coord in &coords {
                if coord.abs() > 1000.0 {
                    return Err(format!("Geometry {} atom {} has unreasonably large coordinate: {}", i, j, coord));
                }
            }
        }
    }

    Ok(())
}

/// Calculate path length along the interpolation
pub fn calculate_path_length(geometries: &[Geometry]) -> f64 {
    let mut total_length = 0.0;

    for i in 1..geometries.len() {
        let coords1 = geometry_to_coords(&geometries[i-1]);
        let coords2 = geometry_to_coords(&geometries[i]);

        let mut segment_length = 0.0;
        for j in 0..coords1.len() {
            let diff = coords1[j] - coords2[j];
            segment_length += diff * diff;
        }
        total_length += segment_length.sqrt();
    }

    total_length
}

/// Print geometry preview for interactive confirmation
pub fn print_geometry_preview(geometries: &[Geometry]) {
    println!("\n****Geometry Preview****");
    println!("Total geometries: {}", geometries.len());
    println!("Path length: {:.3} Å", calculate_path_length(geometries));

    // Show first, middle, and last geometries
    let indices = vec![0, geometries.len() / 2, geometries.len() - 1];

    for &idx in &indices {
        if idx < geometries.len() {
            println!("\n--- Geometry {} ---", idx + 1);
            let geom = &geometries[idx];
            for i in 0..geom.num_atoms.min(5) {  // Show first 5 atoms
                let coords = geom.get_atom_coords(i);
                println!("{:>2} {:>8.3} {:>8.3} {:>8.3}",
                        geom.elements[i], coords[0], coords[1], coords[2]);
            }
            if geom.num_atoms > 5 {
                println!("... ({} more atoms)", geom.num_atoms - 5);
            }
        }
    }
}
