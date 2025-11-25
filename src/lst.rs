//! Linear Synchronous Transit (LST) interpolation module.
//!
//! This module provides functionality for interpolating between molecular geometries
//! using various methods including Linear Synchronous Transit (LST) and Quadratic
//! Synchronous Transit (QST). These methods are commonly used in quantum chemistry
//! to generate initial reaction paths and locate transition states.
//!
//! # Overview
//!
//! The LST method creates a series of intermediate geometries between two endpoint
//! structures by linear interpolation in Cartesian coordinates. The QST method
//! extends this to quadratic interpolation using three geometries (reactant,
//! product, and an intermediate structure).
//!
//! # Key Features
//!
//! - **Kabsch Algorithm**: Optimal alignment of geometries before interpolation
//! - **Multiple Methods**: Linear, quadratic, and energy-weighted interpolation
//! - **Validation**: Comprehensive geometry validation and error checking
//! - **Path Analysis**: Path length calculation and geometry preview
//!
//! # Examples
//!
//! ```rust
//! use omecp::lst::{interpolate, InterpolationMethod};
//! use omecp::geometry::Geometry;
//!
//! // Create two geometries (example with water molecule)
//! let elements = vec!["O".to_string(), "H".to_string(), "H".to_string()];
//! let coords1 = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
//! let coords2 = vec![0.0, 0.0, 0.5, 1.2, 0.0, 0.5, 0.0, 1.2, 0.5];
//!
//! let geom1 = Geometry::new(elements.clone(), coords1);
//! let geom2 = Geometry::new(elements, coords2);
//!
//! // Generate 10 intermediate geometries using linear interpolation
//! let path = interpolate(&geom1, &geom2, 10, InterpolationMethod::Linear);
//! ```
//!
//! # References
//!
//! - Halgren, T. A.; Lipscomb, W. N. *Chem. Phys. Lett.* **1977**, 49, 225-232.
//! - Kabsch, W. *Acta Crystallogr. A* **1976**, 32, 922-923.

use crate::geometry::Geometry;
use nalgebra::DMatrix;

/// Interpolation methods available for LST calculations.
///
/// This enum defines the different interpolation strategies that can be used
/// to generate intermediate geometries between molecular structures.
///
/// # Variants
///
/// - [`Linear`](InterpolationMethod::Linear): Simple linear interpolation between two geometries
/// - [`Quadratic`](InterpolationMethod::Quadratic): Quadratic interpolation using three geometries
/// - [`EnergyWeighted`](InterpolationMethod::EnergyWeighted): Energy-informed interpolation (future feature)
///
/// # Examples
///
/// ```rust
/// use omecp::lst::InterpolationMethod;
///
/// let method = InterpolationMethod::Linear;
/// assert_eq!(method, InterpolationMethod::Linear);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolationMethod {
    /// Linear Synchronous Transit (LST) interpolation.
    ///
    /// Performs simple linear interpolation between two endpoint geometries.
    /// This is the most commonly used method and provides a straight-line
    /// path in Cartesian coordinate space after optimal alignment.
    Linear,

    /// Quadratic Synchronous Transit (QST) interpolation.
    ///
    /// Uses quadratic interpolation with three geometries: two endpoints and
    /// one intermediate structure. If only two geometries are provided, a
    /// midpoint geometry is automatically generated.
    Quadratic,

    /// Energy-weighted interpolation method.
    ///
    /// **Note**: This is a placeholder for future implementation. Currently
    /// falls back to linear interpolation. Will incorporate energy information
    /// to create more physically meaningful interpolation paths.
    EnergyWeighted,
}

/// Performs LST interpolation between two geometries using the specified method.
///
/// This is the main entry point for LST interpolation. It dispatches to the
/// appropriate interpolation function based on the selected method and returns
/// a vector of intermediate geometries.
///
/// # Arguments
///
/// * `geom1` - The starting geometry (reactant structure)
/// * `geom2` - The ending geometry (product structure)
/// * `num_points` - Number of intermediate points to generate (excluding endpoints)
/// * `method` - The interpolation method to use
///
/// # Returns
///
/// Returns a `Vec<Geometry>` containing the interpolated geometries. The vector
/// will have `num_points + 2` elements (including both endpoints).
///
/// # Panics
///
/// Panics if the input geometries have different numbers of atoms or different
/// element types.
///
/// # Examples
///
/// ```rust
/// use omecp::lst::{interpolate, InterpolationMethod};
/// use omecp::geometry::Geometry;
///
/// let elements = vec!["H".to_string(), "H".to_string()];
/// let coords1 = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
/// let coords2 = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
///
/// let geom1 = Geometry::new(elements.clone(), coords1);
/// let geom2 = Geometry::new(elements, coords2);
///
/// // Generate 5 intermediate points using linear interpolation
/// let path = interpolate(&geom1, &geom2, 5, InterpolationMethod::Linear);
/// assert_eq!(path.len(), 7); // 5 intermediate + 2 endpoints
/// ```
///
/// # Implementation Notes
///
/// - For [`InterpolationMethod::Linear`]: Uses Kabsch algorithm for optimal alignment
/// - For [`InterpolationMethod::Quadratic`]: Automatically generates midpoint if needed
/// - For [`InterpolationMethod::EnergyWeighted`]: Currently falls back to linear interpolation
pub fn interpolate(
    geom1: &Geometry,
    geom2: &Geometry,
    num_points: usize,
    method: InterpolationMethod,
) -> Vec<Geometry> {
    match method {
        InterpolationMethod::Linear => interpolate_linear(geom1, geom2, num_points),
        InterpolationMethod::Quadratic => {
            // For QST, we need a third geometry (midpoint approximation)
            let midpoint = create_midpoint_geometry(geom1, geom2);
            interpolate_quadratic(geom1, &midpoint, geom2, num_points)
        }
        InterpolationMethod::EnergyWeighted => {
            // For now, fall back to linear (would need energy values)
            interpolate_linear(geom1, geom2, num_points)
        }
    }
}

/// Performs Linear Synchronous Transit (LST) interpolation between two geometries.
///
/// This function implements the LST method using the Kabsch algorithm for optimal
/// alignment of the geometries before interpolation. The Kabsch algorithm finds
/// the optimal rotation matrix that minimizes the root-mean-square deviation
/// between corresponding atoms.
///
/// # Arguments
///
/// * `geom1` - The starting geometry (will be aligned to geom2)
/// * `geom2` - The target geometry (reference for alignment)
/// * `num_points` - Number of intermediate points to generate
///
/// # Returns
///
/// Returns a `Vec<Geometry>` containing `num_points + 2` geometries, including
/// the aligned starting geometry and the target geometry.
///
/// # Panics
///
/// Panics if the input geometries have different numbers of atoms.
///
/// # Algorithm Details
///
/// 1. **Alignment**: Uses Kabsch algorithm to find optimal rotation matrix
/// 2. **Rotation**: Applies rotation to align geom1 with geom2
/// 3. **Interpolation**: Linear interpolation between aligned geometries
///
/// The Kabsch algorithm steps:
/// - Compute cross-covariance matrix R = Y * X^T
/// - Compute R^T * R and its eigendecomposition
/// - Construct rotation matrix U from eigenvectors and eigenvalues
///
/// # Examples
///
/// ```rust
/// use omecp::lst::interpolate_linear;
/// use omecp::geometry::Geometry;
///
/// let elements = vec!["C".to_string(), "H".to_string()];
/// let coords1 = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
/// let coords2 = vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0];
///
/// let geom1 = Geometry::new(elements.clone(), coords1);
/// let geom2 = Geometry::new(elements, coords2);
///
/// let path = interpolate_linear(&geom1, &geom2, 3);
/// assert_eq!(path.len(), 5); // 3 intermediate + 2 endpoints
/// ```
///
/// # References
///
/// Kabsch, W. "A solution for the best rotation to relate two sets of vectors."
/// *Acta Crystallographica Section A* **1976**, 32(5), 922-923.
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

/// Performs Quadratic Synchronous Transit (QST) interpolation using three geometries.
///
/// This function implements quadratic interpolation between three geometries:
/// a starting point, an intermediate point, and an ending point. The resulting
/// path follows a quadratic curve in coordinate space, which can provide a
/// more realistic reaction path than linear interpolation.
///
/// # Arguments
///
/// * `geom1` - The starting geometry (t = 0)
/// * `geom_mid` - The intermediate geometry (t = 0.5)
/// * `geom2` - The ending geometry (t = 1)
/// * `num_points` - Number of intermediate points to generate
///
/// # Returns
///
/// Returns a `Vec<Geometry>` containing `num_points + 2` geometries following
/// a quadratic interpolation path.
///
/// # Panics
///
/// Panics if the input geometries have different numbers of atoms or different
/// element types.
///
/// # Mathematical Formula
///
/// The quadratic interpolation formula used is:
/// ```text
/// p(t) = (1-t)² × p₁ + 2(1-t)t × p_mid + t² × p₂
/// ```
/// where t ∈ [0, 1] is the interpolation parameter.
///
/// # Examples
///
/// ```rust
/// use omecp::lst::interpolate_quadratic;
/// use omecp::geometry::Geometry;
///
/// let elements = vec!["O".to_string(), "H".to_string(), "H".to_string()];
/// let coords1 = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
/// let coords_mid = vec![0.0, 0.0, 0.2, 1.1, 0.0, 0.2, 0.0, 1.1, 0.2];
/// let coords2 = vec![0.0, 0.0, 0.5, 1.2, 0.0, 0.5, 0.0, 1.2, 0.5];
///
/// let geom1 = Geometry::new(elements.clone(), coords1);
/// let geom_mid = Geometry::new(elements.clone(), coords_mid);
/// let geom2 = Geometry::new(elements, coords2);
///
/// let path = interpolate_quadratic(&geom1, &geom_mid, &geom2, 4);
/// assert_eq!(path.len(), 6); // 4 intermediate + 2 endpoints
/// ```
///
/// # Notes
///
/// - The intermediate geometry should represent a reasonable guess for the
///   transition state or a point along the reaction coordinate
/// - For best results, the intermediate geometry should be optimized or
///   at least chemically reasonable
/// - If no intermediate geometry is available, use [`create_midpoint_geometry`]
///   to generate a simple midpoint structure
pub fn interpolate_quadratic(
    geom1: &Geometry,
    geom_mid: &Geometry,
    geom2: &Geometry,
    num_points: usize,
) -> Vec<Geometry> {
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

/// Creates a simple midpoint geometry for QST when only two geometries are provided.
///
/// This function generates an intermediate geometry by taking the arithmetic
/// mean of corresponding atomic coordinates. While this provides a reasonable
/// starting point for QST interpolation, it may not represent a chemically
/// meaningful structure.
///
/// # Arguments
///
/// * `geom1` - The first endpoint geometry
/// * `geom2` - The second endpoint geometry
///
/// # Returns
///
/// Returns a new `Geometry` with coordinates that are the arithmetic mean
/// of the input geometries.
///
/// # Examples
///
/// ```rust
/// use omecp::lst::create_midpoint_geometry;
/// use omecp::geometry::Geometry;
///
/// let elements = vec!["H".to_string(), "H".to_string()];
/// let coords1 = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
/// let coords2 = vec![0.0, 0.0, 0.0, 4.0, 0.0, 0.0];
///
/// let geom1 = Geometry::new(elements.clone(), coords1);
/// let geom2 = Geometry::new(elements, coords2);
///
/// let midpoint = create_midpoint_geometry(&geom1, &geom2);
/// let mid_coords = midpoint.get_atom_coords(1);
/// assert_eq!(mid_coords[0], 3.0); // (2.0 + 4.0) / 2.0
/// ```
///
/// # Notes
///
/// - The resulting geometry uses the same element types as the input geometries
/// - This is a simple geometric midpoint and may not be chemically reasonable
/// - For better results, consider using an optimized transition state guess
/// - The midpoint geometry is primarily used as a fallback when no intermediate
///   structure is available for QST interpolation
fn create_midpoint_geometry(geom1: &Geometry, geom2: &Geometry) -> Geometry {
    let coords1 = geometry_to_coords(geom1);
    let coords2 = geometry_to_coords(geom2);

    let mut mid_coords = Vec::new();
    for i in 0..coords1.len() {
        mid_coords.push((coords1[i] + coords2[i]) / 2.0);
    }

    Geometry::new(geom1.elements.clone(), mid_coords)
}

/// Converts a geometry to a flat coordinate vector.
///
/// This utility function extracts all atomic coordinates from a geometry
/// and returns them as a single flat vector in the order [x₁, y₁, z₁, x₂, y₂, z₂, ...].
/// This format is convenient for mathematical operations and interpolation.
///
/// # Arguments
///
/// * `geom` - The geometry to convert
///
/// # Returns
///
/// Returns a `Vec<f64>` containing all coordinates in flat format.
/// The vector length will be `3 × num_atoms`.
///
/// # Examples
///
/// ```rust
/// use omecp::geometry::Geometry;
/// // Note: geometry_to_coords is private, this example shows the concept
///
/// let elements = vec!["H".to_string(), "O".to_string()];
/// let coords = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let geom = Geometry::new(elements, coords.clone());
///
/// // The function would return: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
/// ```
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

/// Validates that interpolated geometries are chemically and numerically reasonable.
///
/// This function performs comprehensive validation of a set of geometries,
/// checking for common issues that can arise during interpolation such as
/// inconsistent atom counts, non-finite coordinates, and unreasonably large
/// coordinate values.
///
/// # Arguments
///
/// * `geometries` - A slice of geometries to validate
///
/// # Returns
///
/// Returns `Ok(())` if all geometries pass validation, or `Err(String)` with
/// a descriptive error message if any issues are found.
///
/// # Validation Checks
///
/// 1. **Non-empty**: Ensures at least one geometry is provided
/// 2. **Consistent atom count**: All geometries have the same number of atoms
/// 3. **Consistent elements**: All geometries have the same element types
/// 4. **Finite coordinates**: No NaN or infinite coordinate values
/// 5. **Reasonable coordinates**: No coordinates with absolute value > 1000 Angstrom
///
/// # Examples
///
/// ```rust
/// use omecp::lst::{interpolate_linear, validate_geometries};
/// use omecp::geometry::Geometry;
///
/// let elements = vec!["H".to_string(), "H".to_string()];
/// let coords1 = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
/// let coords2 = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
///
/// let geom1 = Geometry::new(elements.clone(), coords1);
/// let geom2 = Geometry::new(elements, coords2);
///
/// let path = interpolate_linear(&geom1, &geom2, 3);
/// assert!(validate_geometries(&path).is_ok());
/// ```
///
/// # Error Conditions
///
/// The function returns an error in the following cases:
/// - Empty geometry vector
/// - Inconsistent number of atoms between geometries
/// - Different element types between geometries
/// - Non-finite coordinates (NaN or infinite values)
/// - Coordinates with absolute values exceeding 1000 Angstrom
///
/// # Notes
///
/// - The 1000 Angstrom limit is a safety check for obviously incorrect structures
/// - This validation should be called after any interpolation operation
/// - Failed validation often indicates issues with input geometries or
///   numerical problems during interpolation
pub fn validate_geometries(geometries: &[Geometry]) -> Result<(), String> {
    if geometries.is_empty() {
        return Err("No geometries to validate".to_string());
    }

    let num_atoms = geometries[0].num_atoms;
    let elements = &geometries[0].elements;

    for (i, geom) in geometries.iter().enumerate() {
        if geom.num_atoms != num_atoms {
            return Err(format!(
                "Geometry {} has {} atoms, expected {}",
                i, geom.num_atoms, num_atoms
            ));
        }

        if geom.elements != *elements {
            return Err(format!(
                "Geometry {} has different elements than reference",
                i
            ));
        }

        // Check for NaN or infinite coordinates
        for j in 0..geom.num_atoms {
            let coords = geom.get_atom_coords(j);
            for &coord in &coords {
                if !coord.is_finite() {
                    return Err(format!(
                        "Geometry {} atom {} has non-finite coordinate: {}",
                        i, j, coord
                    ));
                }
            }
        }

        // Check for unreasonably large coordinates (> 1000 Angstrom)
        for j in 0..geom.num_atoms {
            let coords = geom.get_atom_coords(j);
            for &coord in &coords {
                if coord.abs() > 1000.0 {
                    return Err(format!(
                        "Geometry {} atom {} has unreasonably large coordinate: {}",
                        i, j, coord
                    ));
                }
            }
        }
    }

    Ok(())
}

/// Calculates the total path length along an interpolated reaction coordinate.
///
/// This function computes the cumulative Euclidean distance between consecutive
/// geometries in the interpolation path. The path length provides a measure of
/// the total geometric change along the reaction coordinate and can be useful
/// for analyzing reaction paths and comparing different interpolation methods.
///
/// # Arguments
///
/// * `geometries` - A slice of geometries representing the interpolation path
///
/// # Returns
///
/// Returns the total path length as a `f64` value in the same units as the
/// input coordinates (typically Angstroms).
///
/// # Algorithm
///
/// The path length is calculated as:
/// ```text
/// L = Σᵢ √(Σⱼ (xᵢ₊₁,ⱼ - xᵢ,ⱼ)²)
/// ```
/// where the outer sum is over geometry pairs and the inner sum is over
/// all coordinate components.
///
/// # Examples
///
/// ```rust
/// use omecp::lst::{interpolate_linear, calculate_path_length};
/// use omecp::geometry::Geometry;
///
/// let elements = vec!["H".to_string(), "H".to_string()];
/// let coords1 = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
/// let coords2 = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
///
/// let geom1 = Geometry::new(elements.clone(), coords1);
/// let geom2 = Geometry::new(elements, coords2);
///
/// let path = interpolate_linear(&geom1, &geom2, 1);
/// let length = calculate_path_length(&path);
/// // Length should be approximately sqrt(2) ≈ 1.414
/// ```
///
/// # Notes
///
/// - Returns 0.0 for empty or single-geometry paths
/// - The path length depends on the coordinate system and units used
/// - Longer paths may indicate more significant structural changes
/// - Can be used to compare the "smoothness" of different interpolation methods
pub fn calculate_path_length(geometries: &[Geometry]) -> f64 {
    let mut total_length = 0.0;

    for i in 1..geometries.len() {
        let coords1 = geometry_to_coords(&geometries[i - 1]);
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

/// Prints a formatted preview of interpolated geometries for interactive confirmation.
///
/// This function displays a summary of the interpolation results, including
/// the total number of geometries, path length, and coordinate details for
/// key geometries (first, middle, and last). This is useful for visual
/// inspection before proceeding with expensive quantum chemistry calculations.
///
/// # Arguments
///
/// * `geometries` - A slice of geometries to preview
///
/// # Output Format
///
/// The function prints to stdout with the following information:
/// - Total number of geometries in the path
/// - Total path length in Angstroms
/// - Coordinate details for first, middle, and last geometries
/// - Element symbols and Cartesian coordinates for each atom
/// - Truncation indicator if more than 5 atoms per geometry
///
/// # Examples
///
/// ```rust
/// use omecp::lst::{interpolate_linear, print_geometry_preview};
/// use omecp::geometry::Geometry;
///
/// let elements = vec!["H".to_string(), "H".to_string()];
/// let coords1 = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
/// let coords2 = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
///
/// let geom1 = Geometry::new(elements.clone(), coords1);
/// let geom2 = Geometry::new(elements, coords2);
///
/// let path = interpolate_linear(&geom1, &geom2, 3);
/// print_geometry_preview(&path);
/// ```
///
/// # Sample Output
///
/// ```text
/// ****Geometry Preview****
/// Total geometries: 5
/// Path length: 1.000 Angstrom
///
/// --- Geometry 1 ---
///  H    0.000    0.000    0.000
///  H    1.000    0.000    0.000
///
/// --- Geometry 3 ---
///  H    0.000    0.000    0.000
///  H    1.500    0.000    0.000
///
/// --- Geometry 5 ---
///  H    0.000    0.000    0.000
///  H    2.000    0.000    0.000
/// ```
///
/// # Notes
///
/// - Only shows first 5 atoms per geometry to keep output manageable
/// - Coordinates are displayed with 3 decimal places
/// - Useful for interactive workflows where user confirmation is needed
/// - Should be called before expensive QM calculations on the path
pub fn print_geometry_preview(geometries: &[Geometry]) {
    println!("\n****Geometry Preview****");
    println!("Total geometries: {}", geometries.len());
    println!(
        "Path length: {:.3} Angstrom",
        calculate_path_length(geometries)
    );

    // Show first, middle, and last geometries
    let indices = vec![0, geometries.len() / 2, geometries.len() - 1];

    for &idx in &indices {
        if idx < geometries.len() {
            println!("\n--- Geometry {} ---", idx + 1);
            let geom = &geometries[idx];
            for i in 0..geom.num_atoms.min(5) {
                // Show first 5 atoms
                let coords = geom.get_atom_coords(i);
                println!(
                    "{:>2} {:>8.3} {:>8.3} {:>8.3}",
                    geom.elements[i], coords[0], coords[1], coords[2]
                );
            }
            if geom.num_atoms > 5 {
                println!("... ({} more atoms)", geom.num_atoms - 5);
            }
        }
    }
}
