use crate::geometry::Geometry;
use crate::constraints::{Constraint, evaluate_constraints};

/// Types of coordinates that can be driven
/// Types of coordinates that can be driven during reaction path following.
///
/// This enum defines the geometric parameters that can be systematically varied
/// to explore a reaction path or potential energy surface.
#[derive(Debug, Clone, PartialEq)]
pub enum CoordinateType {
    /// Represents a bond length between two atoms.
    Bond,
    /// Represents a bond angle between three atoms.
    Angle,
    /// Represents a dihedral angle between four atoms.
    Dihedral,
}

/// Represents a coordinate to drive during reaction path following
/// Represents a coordinate to drive during reaction path following.
///
/// This struct defines a specific geometric parameter (bond, angle, or dihedral)
/// that will be varied during a coordinate driving or path optimization procedure.
#[derive(Debug, Clone)]
pub struct DriveCoordinate {
    /// The type of geometric coordinate being driven.
    pub coord_type: CoordinateType,
    /// A vector of atom indices (0-based) defining the coordinate.
    /// - For `Bond`: `[atom1, atom2]`
    /// - For `Angle`: `[atom1, atom2, atom3]`
    /// - For `Dihedral`: `[atom1, atom2, atom3, atom4]`
    pub atoms: Vec<usize>,
    /// The target value for the coordinate (Angstroms for bonds, radians for angles/dihedrals).
    pub target_value: f64,
}

impl DriveCoordinate {
    /// Creates a new `DriveCoordinate` instance.
    ///
    /// # Arguments
    ///
    /// * `coord_type` - The type of geometric coordinate (Bond, Angle, Dihedral).
    /// * `atoms` - A vector of atom indices (0-based) defining the coordinate.
    /// * `target_value` - The target value for the coordinate.
    ///
    /// # Returns
    ///
    /// A new `DriveCoordinate` instance.
    pub fn new(coord_type: CoordinateType, atoms: Vec<usize>, target_value: f64) -> Self {
        Self {
            coord_type,
            atoms,
            target_value,
        }
    }

    /// Calculate the current value of this coordinate
    pub fn current_value(&self, geometry: &Geometry) -> f64 {
        let constraint = self.to_constraint();
        let violations = evaluate_constraints(geometry, &[constraint]);
        violations[0]
    }

    /// Create a constraint for this coordinate
    pub fn to_constraint(&self) -> Constraint {
        match self.coord_type {
            CoordinateType::Bond => Constraint::Bond { atoms: (self.atoms[0], self.atoms[1]), target: self.target_value },
            CoordinateType::Angle => Constraint::Angle { atoms: (self.atoms[0], self.atoms[1], self.atoms[2]), target: self.target_value },
            CoordinateType::Dihedral => Constraint::Dihedral { atoms: (self.atoms[0], self.atoms[1], self.atoms[2], self.atoms[3]), target: self.target_value },
        }
    }

    /// Create a constraint for this coordinate with a specific target value
    pub fn to_constraint_with_value(&self, target_value: f64) -> Constraint {
        match self.coord_type {
            CoordinateType::Bond => Constraint::Bond { atoms: (self.atoms[0], self.atoms[1]), target: target_value },
            CoordinateType::Angle => Constraint::Angle { atoms: (self.atoms[0], self.atoms[1], self.atoms[2]), target: target_value },
            CoordinateType::Dihedral => Constraint::Dihedral { atoms: (self.atoms[0], self.atoms[1], self.atoms[2], self.atoms[3]), target: target_value },
        }
    }
}

/// Generate a series of geometries by driving a coordinate from start to end
/// Uses constrained optimization to maintain the coordinate value while minimizing energy
pub fn drive_coordinate(
    initial_geom: &Geometry,
    drive_coord: &DriveCoordinate,
    start_value: f64,
    end_value: f64,
    num_steps: usize,
) -> Vec<Geometry> {
    let mut geometries = Vec::new();
    let step_size = (end_value - start_value) / (num_steps - 1) as f64;

    // Start with initial geometry
    let mut current_geom = initial_geom.clone();
    geometries.push(current_geom.clone());

    for i in 1..num_steps {
        let target_value = start_value + step_size * i as f64;
        println!("Driving coordinate to {:.3} (step {}/{})", target_value, i + 1, num_steps);

        // Create constraint for the target coordinate value
        let constraint = drive_coord.to_constraint_with_value(target_value);

        // Perform constrained optimization to reach the target coordinate
        current_geom = constrained_coordinate_driving(
            &current_geom,
            &[constraint],
            50, // max iterations for coordinate driving
            1e-4, // convergence threshold
        );

        geometries.push(current_geom.clone());
    }

    geometries
}

/// Perform constrained optimization to drive a coordinate to a target value
fn constrained_coordinate_driving(
    initial_geom: &Geometry,
    constraints: &[Constraint],
    max_iterations: usize,
    convergence_threshold: f64,
) -> Geometry {
    let mut geometry = initial_geom.clone();

    for iteration in 0..max_iterations {
        // Calculate constraint violations and forces
        let mut constraint_forces = vec![0.0; geometry.coords.len()];
        let mut total_violation = 0.0;

        for constraint in constraints {
            let violation = calculate_constraint_violation(&geometry, constraint);
            total_violation += violation.abs();

            // Calculate constraint force (simplified - would need proper gradients)
            let force = calculate_constraint_force(&geometry, constraint, violation);
            for i in 0..constraint_forces.len() {
                constraint_forces[i] += force[i];
            }
        }

        // Simple steepest descent step
        let step_size = 0.01;
        for i in 0..geometry.coords.len() {
            geometry.coords[i] += step_size * constraint_forces[i];
        }

        // Check convergence
        if total_violation < convergence_threshold {
            println!("Coordinate driving converged after {} iterations", iteration + 1);
            break;
        }

        if iteration % 10 == 0 {
            println!("Coordinate driving iteration {}: violation = {:.6}", iteration + 1, total_violation);
        }
    }

    geometry
}

/// Calculate constraint violation (difference from target value)
fn calculate_constraint_violation(geometry: &Geometry, constraint: &Constraint) -> f64 {
    let violations = evaluate_constraints(geometry, std::slice::from_ref(constraint));
    violations[0]
}

/// Calculate constraint force (simplified gradient)
fn calculate_constraint_force(geometry: &Geometry, constraint: &Constraint, violation: f64) -> Vec<f64> {
    let mut force = vec![0.0; geometry.coords.len()];
    let force_magnitude = -violation * 10.0; // Simple proportional control

    match constraint {
        Constraint::Bond { atoms: (a1, a2), .. } => {
            // Force along the bond vector
            let pos1 = geometry.get_atom_coords(*a1);
            let pos2 = geometry.get_atom_coords(*a2);
            let dist = calculate_distance(geometry, *a1, *a2);

            if dist > 0.0 {
                let dx = (pos2[0] - pos1[0]) / dist;
                let dy = (pos2[1] - pos1[1]) / dist;
                let dz = (pos2[2] - pos1[2]) / dist;

                // Apply force to move atom2
                let idx2 = a2 * 3;
                force[idx2] += force_magnitude * dx;
                force[idx2 + 1] += force_magnitude * dy;
                force[idx2 + 2] += force_magnitude * dz;

                // Apply opposite force to atom1
                let idx1 = a1 * 3;
                force[idx1] -= force_magnitude * dx;
                force[idx1 + 1] -= force_magnitude * dy;
                force[idx1 + 2] -= force_magnitude * dz;
            }
        }
        Constraint::Angle { .. } => {
            // For angles, use finite difference approximation
            // This is simplified - a real implementation would use analytical gradients
            let delta = 0.001;
            for i in 0..geometry.coords.len() {
                let mut geom_plus = geometry.clone();
                geom_plus.coords[i] += delta;
                let violation_plus = calculate_constraint_violation(&geom_plus, constraint);

                force[i] = -(violation_plus - violation) / delta;
            }
        }
        Constraint::Dihedral { .. } => {
            // For dihedrals, use finite difference approximation
            // This is simplified - a real implementation would use analytical gradients
            let delta = 0.001;
            for i in 0..geometry.coords.len() {
                let mut geom_plus = geometry.clone();
                geom_plus.coords[i] += delta;
                let violation_plus = calculate_constraint_violation(&geom_plus, constraint);

                force[i] = -(violation_plus - violation) / delta;
            }
        }
    }

    force
}

/// Optimize a reaction path using the Nudged Elastic Band (NEB) method
pub fn optimize_reaction_path(
    initial_path: &[Geometry],
    constraints: &[Constraint],
) -> Vec<Geometry> {
    if initial_path.len() < 3 {
        return initial_path.to_vec();
    }

    let mut path = initial_path.to_vec();
    let max_iterations = 100;
    let convergence_threshold = 1e-3;
    let spring_constant = 0.1; // Spring constant for NEB

    for iteration in 0..max_iterations {
        let mut max_force = 0.0;
        let mut new_path = Vec::new();

        for (i, geometry) in path.iter().enumerate() {
            if i == 0 || i == path.len() - 1 {
                // Endpoints are fixed
                new_path.push(geometry.clone());
                continue;
            }

            // Calculate forces for this image
            let force = calculate_neb_force(&path, i, spring_constant);

            // Update geometry (simple steepest descent for now)
            let step_size = 0.01;
            let mut new_coords = geometry.coords.clone();

            for j in 0..new_coords.len() {
                new_coords[j] += step_size * force[j];
            }

            let mut new_geometry = Geometry::new(geometry.elements.clone(), new_coords.data.into());

            // Apply constraints if any
            if !constraints.is_empty() {
                // This would need proper constrained optimization
                // For now, just apply the constraints directly
                for constraint in constraints {
                    if let Constraint::Bond { atoms: (a1, a2), target } = constraint {
                        // Simple bond constraint application
                        let current_dist = calculate_distance(&new_geometry, *a1, *a2);
                        if (current_dist - *target).abs() > 0.01 {
                            // Adjust bond length (simplified)
                            adjust_bond_length(&mut new_geometry, *a1, *a2, *target);
                        }
                    }
                }
            }

            new_path.push(new_geometry);

            // Track maximum force for convergence
            let force_magnitude = force.iter().map(|x| x * x).sum::<f64>().sqrt();
            if force_magnitude > max_force {
                max_force = force_magnitude;
            }
        }

        path = new_path;

        // Check convergence
        if max_force < convergence_threshold {
            println!("NEB converged after {} iterations", iteration + 1);
            break;
        }

        if iteration % 10 == 0 {
            println!("NEB iteration {}: max force = {:.6}", iteration + 1, max_force);
        }
    }

    path
}

/// Calculate the NEB force for a given image
fn calculate_neb_force(path: &[Geometry], image_index: usize, spring_constant: f64) -> Vec<f64> {
    let n_images = path.len();
    let current = &path[image_index];

    // Get neighboring images
    let prev = if image_index > 0 { Some(&path[image_index - 1]) } else { None };
    let next = if image_index < n_images - 1 { Some(&path[image_index + 1]) } else { None };

    // Calculate tangent vector
    let tangent = calculate_tangent(prev, current, next);

    // Calculate spring forces
    let mut spring_force = vec![0.0; current.coords.len()];

    if let Some(prev_geom) = prev {
        let dist_prev = calculate_rmsd(current, prev_geom);
        let spring_force_magnitude = spring_constant * (dist_prev - get_average_spacing(path));
        for i in 0..spring_force.len() {
            spring_force[i] += spring_force_magnitude * tangent[i];
        }
    }

    if let Some(next_geom) = next {
        let dist_next = calculate_rmsd(current, next_geom);
        let spring_force_magnitude = spring_constant * (dist_next - get_average_spacing(path));
        for i in 0..spring_force.len() {
            spring_force[i] -= spring_force_magnitude * tangent[i]; // Opposite direction
        }
    }

    // For now, return spring force only (true NEB would include potential energy gradients)
    // In a full implementation, this would be:
    // F = F_spring + (F_perp - F_perp·τ * τ) where F_perp is the perpendicular component of the energy gradient

    spring_force
}

/// Calculate tangent vector for NEB
fn calculate_tangent(prev: Option<&Geometry>, current: &Geometry, next: Option<&Geometry>) -> Vec<f64> {
    let mut tangent = vec![0.0; current.coords.len()];

    match (prev, next) {
        (Some(p), Some(n)) => {
            // Both neighbors exist - use central difference
            let v_plus = subtract_geometries(n, current);
            let v_minus = subtract_geometries(current, p);

            // Normalize vectors
            let norm_plus = v_plus.iter().map(|x| x * x).sum::<f64>().sqrt();
            let norm_minus = v_minus.iter().map(|x| x * x).sum::<f64>().sqrt();

            if norm_plus > 0.0 && norm_minus > 0.0 {
                let v_plus_norm: Vec<f64> = v_plus.iter().map(|x| x / norm_plus).collect();
                let v_minus_norm: Vec<f64> = v_minus.iter().map(|x| x / norm_minus).collect();

                for i in 0..tangent.len() {
                    tangent[i] = v_plus_norm[i] + v_minus_norm[i];
                }
            }
        }
        (Some(p), None) => {
            // Only previous neighbor - use forward difference
            let v = subtract_geometries(current, p);
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                for i in 0..tangent.len() {
                    tangent[i] = v[i] / norm;
                }
            }
        }
        (None, Some(n)) => {
            // Only next neighbor - use backward difference
            let v = subtract_geometries(n, current);
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                for i in 0..tangent.len() {
                    tangent[i] = v[i] / norm;
                }
            }
        }
        _ => {}
    }

    // Normalize tangent
    let norm = tangent.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for i in 0..tangent.len() {
            tangent[i] /= norm;
        }
    }

    tangent
}

/// Calculate RMSD between two geometries
fn calculate_rmsd(geom1: &Geometry, geom2: &Geometry) -> f64 {
    let mut sum_sq = 0.0;
    for i in 0..geom1.coords.len() {
        let diff = geom1.coords[i] - geom2.coords[i];
        sum_sq += diff * diff;
    }
    (sum_sq / geom1.coords.len() as f64).sqrt()
}

/// Subtract two geometries (coord-wise)
fn subtract_geometries(geom1: &Geometry, geom2: &Geometry) -> Vec<f64> {
    geom1.coords.iter().zip(geom2.coords.iter())
        .map(|(a, b)| a - b)
        .collect()
}

/// Get average spacing between images in the path
fn get_average_spacing(path: &[Geometry]) -> f64 {
    if path.len() < 2 {
        return 0.0;
    }

    let mut total_distance = 0.0;
    for i in 1..path.len() {
        total_distance += calculate_rmsd(&path[i], &path[i-1]);
    }

    total_distance / (path.len() - 1) as f64
}

/// Calculate distance between two atoms
fn calculate_distance(geometry: &Geometry, atom1: usize, atom2: usize) -> f64 {
    let pos1 = geometry.get_atom_coords(atom1);
    let pos2 = geometry.get_atom_coords(atom2);
    let dx = pos1[0] - pos2[0];
    let dy = pos1[1] - pos2[1];
    let dz = pos1[2] - pos2[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Adjust bond length between two atoms (simplified implementation)
fn adjust_bond_length(geometry: &mut Geometry, atom1: usize, atom2: usize, target_length: f64) {
    let current_length = calculate_distance(geometry, atom1, atom2);
    if current_length == 0.0 {
        return;
    }

    let scale = target_length / current_length;
    let pos1 = geometry.get_atom_coords(atom1);
    let pos2 = geometry.get_atom_coords(atom2);

    // Move atom2 towards/away from atom1
    let new_pos2 = [
        pos1[0] + (pos2[0] - pos1[0]) * scale,
        pos1[1] + (pos2[1] - pos1[1]) * scale,
        pos1[2] + (pos2[2] - pos1[2]) * scale,
    ];

    // Update coordinates
    let base_idx = atom2 * 3;
    geometry.coords[base_idx] = new_pos2[0];
    geometry.coords[base_idx + 1] = new_pos2[1];
    geometry.coords[base_idx + 2] = new_pos2[2];
}

/// Calculate reaction path statistics
pub fn analyze_reaction_path(geometries: &[Geometry]) -> PathStatistics {
    let energies = Vec::new(); // Would be filled from QM calculations
    let coordinates = Vec::new();

    // Calculate path length
    let mut path_length = 0.0;
    for i in 1..geometries.len() {
        let coords1 = geometry_to_coords(&geometries[i-1]);
        let coords2 = geometry_to_coords(&geometries[i]);

        let mut segment_length = 0.0;
        for j in 0..coords1.len() {
            let diff = coords1[j] - coords2[j];
            segment_length += diff * diff;
        }
        path_length += segment_length.sqrt();
    }

    PathStatistics {
        path_length,
        num_points: geometries.len(),
        energies,
        coordinates,
    }
}

/// Statistics for a reaction path
/// Statistics and data collected along a reaction path.
///
/// This struct holds various metrics and data points generated during a
/// reaction path optimization or coordinate driving procedure, such as
/// path length, number of points, and (potentially) energies and other
/// relevant coordinates.
#[derive(Debug, Clone)]
pub struct PathStatistics {
    /// The total length of the reaction path.
    pub path_length: f64,
    /// The number of discrete points (geometries) along the path.
    pub num_points: usize,
    /// A vector of energies corresponding to each point on the path.
    pub energies: Vec<f64>,
    /// A vector of reaction coordinate values corresponding to each point on the path.
    pub coordinates: Vec<f64>,
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