//! Core geometry and state data structures for molecular representations.
//!
//! This module provides the fundamental data types for representing molecular
//! geometries and electronic states in MECP calculations. It includes:
//!
//! - [`Geometry`]: Molecular structure with element types and Cartesian coordinates
//! - [`State`]: Electronic state with energy, forces, and associated geometry
//!
//! All coordinates are in Bohrs and forces are in Hartree/Bohr.

use nalgebra::DVector;

/// Unit conversion constants for coordinate systems
const BOHR_TO_ANGSTROM: f64 = 0.529177210903;
const ANGSTROM_TO_BOHR: f64 = 1.0 / BOHR_TO_ANGSTROM;

/// Convert coordinates from Angstroms to Bohrs
pub fn angstrom_to_bohr(coords: &DVector<f64>) -> DVector<f64> {
    coords * ANGSTROM_TO_BOHR
}

/// Convert coordinates from Bohrs to Angstroms
pub fn bohr_to_angstrom(coords: &DVector<f64>) -> DVector<f64> {
    coords * BOHR_TO_ANGSTROM
}

/// Represents a molecular geometry with atomic elements and Cartesian coordinates.
///
/// The `Geometry` struct stores the fundamental information about a molecular
/// structure: the chemical elements of each atom and their 3D positions in
/// Cartesian coordinates. It uses a flat representation where coordinates are
/// stored as a single-dimensional vector in the order [x1, y1, z1, x2, y2, z2, ...].
///
/// # Coordinate System
///
/// - Units: Bohrs (a0)
/// - Coordinate frame: Cartesian (x, y, z)
/// - Origin: Arbitrary (typically centered for convenience)
///
/// # Storage Format
///
/// Coordinates are stored in a `DVector<f64>` for efficient linear algebra operations.
/// The flat representation enables direct use with nalgebra for matrix operations
/// commonly required in geometry optimization.
///
/// # Examples
///
/// ```
/// use omecp::geometry::Geometry;
///
/// // Create a water molecule geometry
/// let elements = vec![
///     "O".to_string(),
///     "H".to_string(),
///     "H".to_string(),
/// ];
/// let coords = vec![
///     0.0, 0.0, 0.0,        // O at origin
///     0.757, 0.586, 0.0,    // H1
///     -0.757, 0.586, 0.0,   // H2
/// ];
///
/// let geometry = Geometry::new(elements, coords);
/// assert_eq!(geometry.num_atoms, 3);
///
/// // Get coordinates of atom 0 (oxygen)
/// let oxygen_coords = geometry.get_atom_coords(0);
/// println!("Oxygen position: ({:.3}, {:.3}, {:.3})",
///          oxygen_coords[0], oxygen_coords[1], oxygen_coords[2]);
/// ```
#[derive(Debug, Clone)]
pub struct Geometry {
    /// Chemical element symbols for each atom in order
    pub elements: Vec<String>,
    /// Flattened Cartesian coordinates [x1, y1, z1, x2, y2, z2, ...] in Angstroms
    pub coords: DVector<f64>,
    /// Number of atoms in the molecule
    pub num_atoms: usize,
}

impl Geometry {
    /// Create a new `Geometry` from element list and coordinate vector.
    ///
    /// # Arguments
    ///
    /// * `elements` - Vector of element symbols (e.g., "C", "H", "O")
    /// * `coords` - Flattened coordinate vector of length 3 × num_atoms
    ///
    /// # Panics
    ///
    /// Panics if `coords.len() != elements.len() * 3`, ensuring data consistency.
    ///
    /// # Examples
    ///
    /// ```
    /// use omecp::geometry::Geometry;
    ///
    /// let elements = vec!["C".to_string(), "H".to_string()];
    /// let coords = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    /// let geometry = Geometry::new(elements, coords);
    /// ```
    pub fn new(elements: Vec<String>, coords: Vec<f64>) -> Self {
        let num_atoms = elements.len();
        assert_eq!(coords.len(), num_atoms * 3);
        Self {
            elements,
            coords: DVector::from_vec(coords),
            num_atoms,
        }
    }

    /// Get the Cartesian coordinates of a specific atom.
    ///
    /// # Arguments
    ///
    /// * `atom_idx` - Zero-based index of the atom (0 = first atom)
    ///
    /// # Returns
    ///
    /// Array of three coordinates [x, y, z] in Angstroms.
    ///
    /// # Examples
    ///
    /// ```
    /// use omecp::geometry::Geometry;
    ///
    /// let elements = vec!["O".to_string(), "H".to_string(), "H".to_string()];
    /// let coords = vec![0.0, 0.0, 0.0, 0.757, 0.586, 0.0, -0.757, 0.586, 0.0];
    /// let geometry = Geometry::new(elements, coords);
    ///
    /// // Get oxygen coordinates
    /// let o_coords = geometry.get_atom_coords(0);
    /// assert_eq!(o_coords, [0.0, 0.0, 0.0]);
    ///
    /// // Get first hydrogen coordinates
    /// let h1_coords = geometry.get_atom_coords(1);
    /// assert_eq!(h1_coords, [0.757, 0.586, 0.0]);
    /// ```
    pub fn get_atom_coords(&self, atom_idx: usize) -> [f64; 3] {
        let i = atom_idx * 3;
        [self.coords[i], self.coords[i + 1], self.coords[i + 2]]
    }
}

/// Represents an electronic state of a molecule with energy, forces, and geometry.
///
/// The `State` struct encapsulates all information about a single electronic state
/// needed for MECP optimization: the potential energy, the gradient (forces), and
/// the molecular geometry at which these were evaluated.
///
/// In the MECP algorithm, we track two states simultaneously (typically called
/// "state A" and "state B") and seek the geometry where their energies are equal
/// while minimizing the perpendicular gradient component.
///
/// # Energy Units
///
/// - Energy: Hartree (Eh)
/// - Forces: Hartree/Bohr (Eh/a0)
///
/// # Force Convention
///
/// Forces are the negative gradient of the energy with respect to nuclear positions:
/// ```text
/// F = -∇E
/// ```
/// This is the standard convention in quantum chemistry, where positive forces
/// point in the direction of decreasing energy.
///
#[derive(Debug, Clone)]
pub struct State {
    /// Potential energy of the state in Hartree (Eh)
    pub energy: f64,
    /// Gradient (negative of forces) in Hartree/Bohr
    /// Stored as a flat vector matching Geometry::coords format
    pub forces: DVector<f64>,
    /// Molecular geometry at which energy and forces were evaluated
    pub geometry: Geometry,
}
impl State {
    /// Validates that the State contains meaningful data and is not a zero-energy failure.
    ///
    /// This method checks for common failure modes where QM calculations appear to
    /// succeed but actually return invalid or default values. It prevents silent
    /// failures that could lead to incorrect MECP optimization results.
    ///
    /// # Validation Criteria
    ///
    /// The State is considered invalid if:
    /// - Energy is exactly zero (indicates parsing failure or uninitialized state)
    /// - All force components are zero (indicates gradient calculation failure)
    /// - Forces vector is empty (indicates missing gradient data)
    ///
    /// # Returns
    ///
    /// - `Ok(())` if the State contains valid, meaningful data
    /// - `Err(String)` with a descriptive error message if validation fails
    ///
    /// # Examples
    ///
    /// ```
    /// use omecp::geometry::{Geometry, State};
    /// use nalgebra::DVector;
    ///
    /// // Valid state with non-zero energy and forces
    /// let geometry = Geometry::new(vec!["H".to_string()], vec![0.0, 0.0, 0.0]);
    /// let valid_state = State {
    ///     energy: -0.5,
    ///     forces: DVector::from_vec(vec![0.1, -0.2, 0.0]),
    ///     geometry,
    /// };
    /// assert!(valid_state.validate().is_ok());
    ///
    /// // Invalid state with zero energy
    /// let geometry = Geometry::new(vec!["H".to_string()], vec![0.0, 0.0, 0.0]);
    /// let invalid_state = State {
    ///     energy: 0.0,
    ///     forces: DVector::from_vec(vec![0.1, -0.2, 0.0]),
    ///     geometry,
    /// };
    /// assert!(invalid_state.validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<(), String> {
        // Check for zero energy (common failure mode)
        if self.energy == 0.0 {
            return Err(
                "State contains zero energy, indicating parsing failure or uninitialized state"
                    .to_string(),
            );
        }

        // Check for empty forces vector
        if self.forces.is_empty() {
            return Err(
                "State contains empty forces vector, indicating missing gradient data".to_string(),
            );
        }

        // Check for all-zero forces (gradient calculation failure)
        if self.forces.iter().all(|&f| f == 0.0) {
            return Err(
                "State contains all-zero forces, indicating gradient calculation failure"
                    .to_string(),
            );
        }

        // Check force/geometry consistency
        let expected_force_components = self.geometry.num_atoms * 3;
        if self.forces.len() != expected_force_components {
            return Err(format!(
                "Force/geometry mismatch: expected {} force components for {} atoms, got {}",
                expected_force_components,
                self.geometry.num_atoms,
                self.forces.len()
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_validation_valid_state() {
        let geometry = Geometry::new(vec!["H".to_string()], vec![0.0, 0.0, 0.0]);
        let valid_state = State {
            energy: -0.5,
            forces: DVector::from_vec(vec![0.1, -0.2, 0.0]),
            geometry,
        };

        assert!(valid_state.validate().is_ok());
    }

    #[test]
    fn test_state_validation_zero_energy() {
        let geometry = Geometry::new(vec!["H".to_string()], vec![0.0, 0.0, 0.0]);
        let invalid_state = State {
            energy: 0.0,
            forces: DVector::from_vec(vec![0.1, -0.2, 0.0]),
            geometry,
        };

        let result = invalid_state.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("zero energy"));
    }

    #[test]
    fn test_state_validation_zero_forces() {
        let geometry = Geometry::new(vec!["H".to_string()], vec![0.0, 0.0, 0.0]);
        let invalid_state = State {
            energy: -0.5,
            forces: DVector::from_vec(vec![0.0, 0.0, 0.0]),
            geometry,
        };

        let result = invalid_state.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("all-zero forces"));
    }

    #[test]
    fn test_state_validation_empty_forces() {
        let geometry = Geometry::new(vec!["H".to_string()], vec![0.0, 0.0, 0.0]);
        let invalid_state = State {
            energy: -0.5,
            forces: DVector::from_vec(vec![]),
            geometry,
        };

        let result = invalid_state.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty forces"));
    }

    #[test]
    fn test_state_validation_force_geometry_mismatch() {
        let geometry = Geometry::new(
            vec!["H".to_string(), "H".to_string()],
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        );
        let invalid_state = State {
            energy: -0.5,
            forces: DVector::from_vec(vec![0.1, -0.2, 0.0]), // Only 3 components for 2 atoms
            geometry,
        };

        let result = invalid_state.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Force/geometry mismatch"));
    }
}
