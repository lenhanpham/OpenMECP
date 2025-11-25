//! Reaction path optimization and coordinate driving methods.
//!
//! This module implements various methods for exploring reaction paths and potential
//! energy surfaces in quantum chemistry calculations. It provides tools for:
//!
//! - **Coordinate Driving**: Systematically vary geometric parameters along a reaction coordinate
//! - **Nudged Elastic Band (NEB)**: Optimize minimum energy paths between reactants and products
//! - **Path Analysis**: Calculate path statistics and analyze reaction mechanisms
//! - **Constraint Handling**: Apply geometric constraints during path optimization
//!
//! # Theoretical Background
//!
//! ## Reaction Coordinates
//!
//! A reaction coordinate is a geometric parameter that describes the progress of a
//! chemical reaction. Common reaction coordinates include:
//!
//! - **Bond lengths**: Distance between two atoms (r₁₂)
//! - **Bond angles**: Angle between three atoms (∠ABC)
//! - **Dihedral angles**: Torsion angle between four atoms (∠ABCD)
//!
//! ## Coordinate Driving
//!
//! Coordinate driving systematically varies a chosen reaction coordinate from an
//! initial value to a final value while optimizing all other degrees of freedom.
//! This generates a series of geometries that represent a possible reaction path.
//!
//! The method uses constrained optimization at each point:
//! ```text
//! minimize E(x) subject to g(x) = target_value
//! ```
//!
//! where E(x) is the energy, g(x) is the reaction coordinate, and target_value
//! varies from start to end.
//!
//! ## Nudged Elastic Band (NEB) Method
//!
//! The NEB method finds the minimum energy path (MEP) between two known structures
//! (reactant and product) by optimizing a chain of intermediate images connected
//! by springs. The method balances two forces:
//!
//! 1. **Spring forces**: Keep images evenly distributed along the path
//! 2. **True forces**: Drive images toward the minimum energy path
//!
//! The NEB force on image i is:
//! ```text
//! F_i = F_i^spring + F_i^⊥
//! ```
//!
//! where F_i^spring is the spring force along the path tangent and F_i^⊥ is the
//! component of the true force perpendicular to the path.
//!
//! # Applications
//!
//! - **Transition State Search**: Find saddle points along reaction paths
//! - **Reaction Mechanism Elucidation**: Understand how reactions proceed
//! - **Barrier Height Calculation**: Determine activation energies
//! - **Conformational Analysis**: Explore molecular conformational changes
//!
//! # References
//!
//! - Henkelman, G.; Jónsson, H. *J. Chem. Phys.* **2000**, 113, 9978-9985.
//! - Henkelman, G.; Uberuaga, B. P.; Jónsson, H. *J. Chem. Phys.* **2000**, 113, 9901-9904.
//! - Sheppard, D.; Terrell, R.; Henkelman, G. *J. Chem. Phys.* **2008**, 128, 134106.

use crate::constraints::{evaluate_constraints, Constraint};
use crate::geometry::Geometry;

/// Types of geometric coordinates that can be driven during reaction path exploration.
///
/// This enum defines the fundamental geometric parameters used in quantum chemistry
/// to describe molecular structure and reaction progress. Each coordinate type
/// corresponds to a specific mathematical relationship between atoms.
///
/// # Coordinate Definitions
///
/// ## Bond Length (r₁₂)
/// The distance between two atoms, calculated as:
/// ```text
/// r₁₂ = |r₂ - r₁| = √[(x₂-x₁)² + (y₂-y₁)² + (z₂-z₁)²]
/// ```
///
/// ## Bond Angle (θ₁₂₃)
/// The angle between three atoms, calculated using the dot product:
/// ```text
/// θ₁₂₃ = arccos[(r₁₂ · r₃₂) / (|r₁₂| × |r₃₂|)]
/// ```
/// where r₁₂ = r₁ - r₂ and r₃₂ = r₃ - r₂
///
/// ## Dihedral Angle (φ₁₂₃₄)
/// The torsion angle between four atoms, calculated using cross products:
/// ```text
/// φ₁₂₃₄ = arctan2[(r₂₃ · (n₁ × n₂)), (n₁ · n₂)]
/// ```
/// where n₁ = r₁₂ × r₂₃ and n₂ = r₂₃ × r₃₄
///
/// # Usage in Reaction Path Studies
///
/// - **Bond**: Ideal for bond formation/breaking reactions
/// - **Angle**: Useful for studying angular deformations and ring closures
/// - **Dihedral**: Essential for conformational changes and rotational barriers
///
/// # Examples
///
/// ```rust
/// use omecp::reaction_path::CoordinateType;
///
/// // For studying bond dissociation
/// let bond_coord = CoordinateType::Bond;
///
/// // For ring-opening reactions
/// let angle_coord = CoordinateType::Angle;
///
/// // For conformational isomerization
/// let dihedral_coord = CoordinateType::Dihedral;
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum CoordinateType {
    /// Bond length coordinate between two atoms.
    ///
    /// Measures the distance between atoms i and j. Units are typically
    /// in Angstroms (Angstrom) or Bohr radii. This is the most common reaction
    /// coordinate for bond formation and dissociation processes.
    Bond,

    /// Bond angle coordinate between three atoms.
    ///
    /// Measures the angle formed by atoms i-j-k, where j is the central atom.
    /// Units are in radians internally, but often displayed in degrees.
    /// Useful for studying angular deformations and ring strain.
    Angle,

    /// Dihedral (torsion) angle coordinate between four atoms.
    ///
    /// Measures the torsion angle around the j-k bond in the sequence i-j-k-l.
    /// Units are in radians internally. Essential for studying conformational
    /// changes, rotational barriers, and stereochemical transformations.
    Dihedral,
}

/// Represents a specific geometric coordinate to drive during reaction path exploration.
///
/// This struct encapsulates all information needed to define and manipulate a
/// reaction coordinate during path optimization or coordinate driving procedures.
/// It serves as the fundamental building block for reaction path methods.
///
/// # Coordinate Specification
///
/// The coordinate is fully specified by:
/// 1. **Type**: Bond, angle, or dihedral (determines the mathematical formula)
/// 2. **Atoms**: The specific atoms involved (defines which atoms to measure)
/// 3. **Target**: The desired value to drive the coordinate toward
///
/// # Atom Indexing Convention
///
/// All atom indices are **0-based** (first atom is index 0). The number of atoms
/// required depends on the coordinate type:
///
/// - **Bond**: 2 atoms `[i, j]` - distance between atoms i and j
/// - **Angle**: 3 atoms `[i, j, k]` - angle i-j-k with j as vertex
/// - **Dihedral**: 4 atoms `[i, j, k, l]` - torsion around j-k bond
///
/// # Units and Conventions
///
/// - **Bond lengths**: Angstroms (Angstrom) - typical range 0.5-5.0 Angstrom
/// - **Angles**: Radians internally - typical range 0 to π (0° to 180°)
/// - **Dihedrals**: Radians internally - range -π to π (-180° to 180°)
///
/// # Examples
///
/// ```rust
/// use omecp::reaction_path::{DriveCoordinate, CoordinateType};
///
/// // C-H bond dissociation: drive from 1.1 Angstrom to 3.0 Angstrom
/// let bond_breaking = DriveCoordinate::new(
///     CoordinateType::Bond,
///     vec![0, 1],  // Carbon (0) to Hydrogen (1)
///     3.0          // Target distance in Angstroms
/// );
///
/// // Ring opening: drive C-C-C angle from 60° to 120°
/// let ring_opening = DriveCoordinate::new(
///     CoordinateType::Angle,
///     vec![0, 1, 2],           // Atoms forming the angle
///     120.0_f64.to_radians()   // Target angle in radians
/// );
///
/// // Conformational change: rotate around C-C bond
/// let rotation = DriveCoordinate::new(
///     CoordinateType::Dihedral,
///     vec![0, 1, 2, 3],        // Four atoms defining dihedral
///     180.0_f64.to_radians()   // Target dihedral in radians
/// );
/// ```
#[derive(Debug, Clone)]
pub struct DriveCoordinate {
    /// The type of geometric coordinate being driven.
    ///
    /// Determines the mathematical formula used to calculate the coordinate
    /// value and its derivatives. This affects how the constraint forces
    /// are computed during optimization.
    pub coord_type: CoordinateType,

    /// Vector of atom indices (0-based) defining the coordinate.
    ///
    /// The length and interpretation depend on the coordinate type:
    /// - **Bond**: `[atom1, atom2]` - 2 atoms
    /// - **Angle**: `[atom1, atom2, atom3]` - 3 atoms (atom2 is vertex)
    /// - **Dihedral**: `[atom1, atom2, atom3, atom4]` - 4 atoms (rotation around bond 2-3)
    pub atoms: Vec<usize>,

    /// The target value for the coordinate.
    ///
    /// Units depend on coordinate type:
    /// - **Bond**: Angstroms (Angstrom)
    /// - **Angle**: Radians (use `.to_radians()` to convert from degrees)
    /// - **Dihedral**: Radians (range -π to π)
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

    /// Calculates the current value of this coordinate in the given geometry.
    ///
    /// This method evaluates the coordinate using the current atomic positions
    /// and returns the actual measured value. It's essential for monitoring
    /// the progress of coordinate driving and checking convergence.
    ///
    /// # Arguments
    ///
    /// * `geometry` - The molecular geometry to evaluate the coordinate in
    ///
    /// # Returns
    ///
    /// The current value of the coordinate in appropriate units:
    /// - Bond: distance in Angstroms
    /// - Angle: angle in radians
    /// - Dihedral: torsion angle in radians (-π to π)
    ///
    /// # Algorithm
    ///
    /// 1. Convert the coordinate specification to a constraint
    /// 2. Use the constraint evaluation system to calculate the current value
    /// 3. Return the measured value (constraint violation represents deviation from target)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use omecp::reaction_path::{DriveCoordinate, CoordinateType};
    ///
    /// let bond_coord = DriveCoordinate::new(
    ///     CoordinateType::Bond,
    ///     vec![0, 1],
    ///     1.5  // Target value (not used in current_value)
    /// );
    ///
    /// // let current_distance = bond_coord.current_value(&geometry);
    /// // println!("Current C-H distance: {:.3} Angstrom", current_distance);
    /// ```
    pub fn current_value(&self, geometry: &Geometry) -> f64 {
        let constraint = self.to_constraint();
        let violations = evaluate_constraints(geometry, &[constraint]);
        violations[0]
    }

    /// Creates a constraint object for this coordinate using the stored target value.
    ///
    /// This method converts the coordinate specification into a constraint that
    /// can be used by the optimization system. The constraint represents the
    /// mathematical relationship that should be satisfied.
    ///
    /// # Returns
    ///
    /// A `Constraint` enum variant appropriate for the coordinate type, with
    /// the target value set to `self.target_value`.
    ///
    /// # Implementation Details
    ///
    /// The method maps coordinate types to constraint types:
    /// - `CoordinateType::Bond` → `Constraint::Bond`
    /// - `CoordinateType::Angle` → `Constraint::Angle`
    /// - `CoordinateType::Dihedral` → `Constraint::Dihedral`
    ///
    /// Each constraint contains the atom indices and target value needed
    /// for constraint evaluation and gradient calculation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let coord = DriveCoordinate::new(
    ///     CoordinateType::Bond,
    ///     vec![0, 1],
    ///     1.8
    /// );
    ///
    /// let constraint = coord.to_constraint();
    /// // This constraint can now be used in optimization
    /// ```
    pub fn to_constraint(&self) -> Constraint {
        match self.coord_type {
            CoordinateType::Bond => Constraint::Bond {
                atoms: (self.atoms[0], self.atoms[1]),
                target: self.target_value,
            },
            CoordinateType::Angle => Constraint::Angle {
                atoms: (self.atoms[0], self.atoms[1], self.atoms[2]),
                target: self.target_value,
            },
            CoordinateType::Dihedral => Constraint::Dihedral {
                atoms: (self.atoms[0], self.atoms[1], self.atoms[2], self.atoms[3]),
                target: self.target_value,
            },
        }
    }

    /// Creates a constraint object for this coordinate with a custom target value.
    ///
    /// This method is similar to `to_constraint()` but allows specifying a different
    /// target value than the one stored in the coordinate. This is particularly
    /// useful during coordinate driving where the target value changes at each step.
    ///
    /// # Arguments
    ///
    /// * `target_value` - The desired target value for the constraint
    ///
    /// # Returns
    ///
    /// A `Constraint` enum variant with the specified target value.
    ///
    /// # Use Cases
    ///
    /// - **Coordinate Driving**: Generate constraints for intermediate target values
    /// - **Path Optimization**: Create constraints for specific path points
    /// - **Scanning**: Systematically vary the target value for PES exploration
    ///
    /// # Examples
    ///
    /// ```rust
    /// let coord = DriveCoordinate::new(
    ///     CoordinateType::Bond,
    ///     vec![0, 1],
    ///     2.0  // Original target
    /// );
    ///
    /// // Create constraint for intermediate value during driving
    /// let intermediate_constraint = coord.to_constraint_with_value(1.5);
    /// ```
    pub fn to_constraint_with_value(&self, target_value: f64) -> Constraint {
        match self.coord_type {
            CoordinateType::Bond => Constraint::Bond {
                atoms: (self.atoms[0], self.atoms[1]),
                target: target_value,
            },
            CoordinateType::Angle => Constraint::Angle {
                atoms: (self.atoms[0], self.atoms[1], self.atoms[2]),
                target: target_value,
            },
            CoordinateType::Dihedral => Constraint::Dihedral {
                atoms: (self.atoms[0], self.atoms[1], self.atoms[2], self.atoms[3]),
                target: target_value,
            },
        }
    }
}

/// Generates a series of geometries by systematically driving a coordinate from start to end value.
///
/// This function implements the coordinate driving method, which is fundamental in
/// computational chemistry for exploring reaction paths and potential energy surfaces.
/// It creates a series of molecular geometries where a chosen reaction coordinate
/// is systematically varied while all other degrees of freedom are optimized.
///
/// # Theoretical Background
///
/// Coordinate driving solves a series of constrained optimization problems:
/// ```text
/// For each target value t_i:
///   minimize E(x) subject to g(x) = t_i
/// ```
///
/// where:
/// - E(x) is the molecular energy
/// - g(x) is the reaction coordinate function
/// - t_i are intermediate target values from start_value to end_value
/// - x represents all atomic coordinates
///
/// # Algorithm Overview
///
/// 1. **Discretization**: Divide the coordinate range into `num_steps` equal intervals
/// 2. **Sequential Optimization**: For each target value:
///    - Create a constraint with the current target value
///    - Perform constrained optimization starting from the previous geometry
///    - Store the optimized geometry
/// 3. **Path Construction**: Return the series of optimized geometries
///
/// # Mathematical Details
///
/// The step size is calculated as:
/// ```text
/// Δt = (end_value - start_value) / (num_steps - 1)
/// ```
///
/// Target values are:
/// ```text
/// t_i = start_value + i × Δt  for i = 0, 1, ..., num_steps-1
/// ```
///
/// # Arguments
///
/// * `initial_geom` - Starting molecular geometry (reactant structure)
/// * `drive_coord` - Specification of the coordinate to drive (bond, angle, or dihedral)
/// * `start_value` - Initial value of the reaction coordinate
/// * `end_value` - Final value of the reaction coordinate  
/// * `num_steps` - Number of intermediate points to generate (including endpoints)
///
/// # Returns
///
/// Returns a `Vec<Geometry>` containing `num_steps` optimized geometries representing
/// the reaction path. The first geometry corresponds to `start_value` and the last
/// to `end_value`.
///
/// # Applications
///
/// - **Reaction Path Generation**: Create initial guess for transition state searches
/// - **Barrier Height Estimation**: Calculate approximate activation energies
/// - **Mechanism Elucidation**: Understand how molecular structure changes during reaction
/// - **Conformational Analysis**: Explore conformational transitions
///
/// # Examples
///
/// ```rust
/// use omecp::reaction_path::{drive_coordinate, DriveCoordinate, CoordinateType};
///
/// // Drive a C-H bond from 1.1 Angstrom to 3.0 Angstrom in 20 steps
/// let bond_coord = DriveCoordinate::new(
///     CoordinateType::Bond,
///     vec![0, 1],  // Carbon-Hydrogen bond
///     3.0          // Final target (not used in this function)
/// );
///
/// let path = drive_coordinate(
///     &initial_geometry,
///     &bond_coord,
///     1.1,    // Start: typical C-H bond length
///     3.0,    // End: dissociated state
///     20      // 20 intermediate geometries
/// );
///
/// println!("Generated {} geometries along the dissociation path", path.len());
/// ```
///
/// # Performance Considerations
///
/// - **Convergence**: Each constrained optimization may require 10-50 iterations
/// - **Step Size**: Smaller steps (more points) give smoother paths but cost more
/// - **Starting Guess**: Each optimization starts from the previous optimized geometry
/// - **Constraint Stiffness**: Tight constraints may require more iterations
///
/// # Limitations
///
/// - **Single Coordinate**: Only one coordinate can be driven at a time
/// - **Local Minima**: May get trapped in local minima along the path
/// - **Constraint Satisfaction**: Assumes constraints can be satisfied at all points
/// - **No Energy Information**: Does not consider energy barriers during driving
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
        println!(
            "Driving coordinate to {:.3} (step {}/{})",
            target_value,
            i + 1,
            num_steps
        );

        // Create constraint for the target coordinate value
        let constraint = drive_coord.to_constraint_with_value(target_value);

        // Perform constrained optimization to reach the target coordinate
        current_geom = constrained_coordinate_driving(
            &current_geom,
            &[constraint],
            50,   // max iterations for coordinate driving
            1e-4, // convergence threshold
        );

        geometries.push(current_geom.clone());
    }

    geometries
}

/// Performs constrained optimization to drive a coordinate to its target value.
///
/// This function implements a simplified constrained optimization algorithm that
/// adjusts atomic coordinates to satisfy geometric constraints while minimizing
/// constraint violations. It uses a steepest descent approach with constraint forces.
///
/// # Theoretical Background
///
/// The method solves the constrained optimization problem:
/// ```text
/// minimize ||g(x) - target||²
/// ```
/// where g(x) represents the constraint functions and target are the desired values.
///
/// # Algorithm Details
///
/// ## Force Calculation
/// For each constraint, the algorithm computes:
/// 1. **Violation**: Current deviation from target value
/// 2. **Constraint Force**: Gradient of the constraint with respect to coordinates
/// 3. **Force Direction**: Points toward satisfying the constraint
///
/// ## Update Step
/// The coordinates are updated using:
/// ```text
/// x_new = x_old + α × F_constraint
/// ```
/// where α is the step size and F_constraint is the total constraint force.
///
/// ## Convergence Criteria
/// The optimization converges when:
/// ```text
/// Σ|g_i(x) - target_i| < threshold
/// ```
///
/// # Arguments
///
/// * `initial_geom` - Starting geometry for the optimization
/// * `constraints` - Array of geometric constraints to satisfy
/// * `max_iterations` - Maximum number of optimization steps (typically 50-100)
/// * `convergence_threshold` - Tolerance for constraint satisfaction (typically 1e-4)
///
/// # Returns
///
/// Returns the optimized `Geometry` with constraints satisfied within the threshold.
///
/// # Implementation Notes
///
/// ## Simplified Approach
/// This implementation uses a basic steepest descent method rather than more
/// sophisticated techniques like:
/// - Lagrange multiplier methods
/// - Sequential quadratic programming (SQP)
/// - Augmented Lagrangian methods
///
/// ## Force Calculation Methods
/// - **Bond constraints**: Analytical gradient along bond vector
/// - **Angle/Dihedral constraints**: Finite difference approximation
///
/// ## Step Size Control
/// - Fixed step size (0.01) - could be improved with adaptive methods
/// - No line search or trust region methods
///
/// # Limitations
///
/// - **Convergence**: May be slow for difficult constraints
/// - **Stability**: No guarantee of convergence for all systems
/// - **Accuracy**: Finite difference gradients are approximate
/// - **Efficiency**: Not optimized for computational performance
///
/// # Future Improvements
///
/// - Implement analytical gradients for all constraint types
/// - Add adaptive step size control
/// - Include second-order optimization methods
/// - Add support for inequality constraints
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
        for (coord, &force) in geometry.coords.iter_mut().zip(&constraint_forces) {
            *coord += step_size * force;
        }

        // Check convergence
        if total_violation < convergence_threshold {
            println!(
                "Coordinate driving converged after {} iterations",
                iteration + 1
            );
            break;
        }

        if iteration % 10 == 0 {
            println!(
                "Coordinate driving iteration {}: violation = {:.6}",
                iteration + 1,
                total_violation
            );
        }
    }

    geometry
}

/// Calculates the constraint violation for a single geometric constraint.
///
/// This function evaluates how much the current geometry deviates from the
/// target constraint value. The violation is the difference between the
/// current coordinate value and the desired target value.
///
/// # Mathematical Definition
///
/// For a constraint g(x) = target, the violation is:
/// ```text
/// violation = g(x) - target
/// ```
///
/// # Arguments
///
/// * `geometry` - Current molecular geometry
/// * `constraint` - The geometric constraint to evaluate
///
/// # Returns
///
/// The constraint violation as a signed value:
/// - **Positive**: Current value exceeds target
/// - **Negative**: Current value is below target  
/// - **Zero**: Constraint is perfectly satisfied
///
/// # Examples
///
/// For a bond constraint with target 1.5 Angstrom:
/// - Current bond = 1.8 Angstrom → violation = +0.3 Angstrom (too long)
/// - Current bond = 1.2 Angstrom → violation = -0.3 Angstrom (too short)
/// - Current bond = 1.5 Angstrom → violation = 0.0 Angstrom (satisfied)
fn calculate_constraint_violation(geometry: &Geometry, constraint: &Constraint) -> f64 {
    let violations = evaluate_constraints(geometry, std::slice::from_ref(constraint));
    violations[0]
}

/// Calculates the constraint force (gradient) for a single geometric constraint.
///
/// This function computes the force that should be applied to atomic coordinates
/// to reduce the constraint violation. The force points in the direction that
/// will move the coordinate value toward its target.
///
/// # Theoretical Background
///
/// The constraint force is the negative gradient of the constraint violation:
/// ```text
/// F_constraint = -∇[g(x) - target] = -∇g(x)
/// ```
///
/// For constraint satisfaction, we want to minimize |g(x) - target|², so:
/// ```text
/// F = -2 × (g(x) - target) × ∇g(x)
/// ```
///
/// # Algorithm Implementation
///
/// ## Bond Constraints (Analytical)
/// For bond constraints, the gradient is computed analytically:
/// ```text
/// ∇r₁₂ = (r₂ - r₁) / |r₂ - r₁|  (unit vector along bond)
/// ```
/// Forces are applied equally and oppositely to the two atoms.
///
/// ## Angle/Dihedral Constraints (Finite Difference)
/// For complex constraints, finite difference approximation is used:
/// ```text
/// ∂g/∂xᵢ ≈ [g(x + δeᵢ) - g(x)] / δ
/// ```
/// where eᵢ is the unit vector in coordinate direction i.
///
/// # Arguments
///
/// * `geometry` - Current molecular geometry
/// * `constraint` - The geometric constraint to compute forces for
/// * `violation` - Current constraint violation (from `calculate_constraint_violation`)
///
/// # Returns
///
/// A vector of forces with length 3×N_atoms, where each group of 3 consecutive
/// elements represents the [Fx, Fy, Fz] force components for one atom.
///
/// # Force Scaling
///
/// The force magnitude is proportional to the violation:
/// ```text
/// |F| = k × |violation|
/// ```
/// where k = 10.0 is an empirical force constant.
///
/// # Implementation Details
///
/// ## Bond Force Calculation
/// 1. Calculate bond vector and distance
/// 2. Compute unit vector along bond
/// 3. Apply force proportional to violation
/// 4. Equal and opposite forces on the two atoms
///
/// ## Finite Difference Parameters
/// - **Step size (δ)**: 0.001 Angstrom (compromise between accuracy and numerical stability)
/// - **One-sided difference**: Uses forward difference for simplicity
/// - **Coordinate perturbation**: Each Cartesian coordinate perturbed independently
///
/// # Limitations
///
/// - **Finite difference errors**: Approximate gradients for angles/dihedrals
/// - **Fixed step size**: Not adaptive to constraint type or system size
/// - **No second derivatives**: Cannot account for constraint curvature
/// - **Proportional control**: Simple linear relationship between violation and force
fn calculate_constraint_force(
    geometry: &Geometry,
    constraint: &Constraint,
    violation: f64,
) -> Vec<f64> {
    let mut force = vec![0.0; geometry.coords.len()];
    let force_magnitude = -violation * 10.0; // Simple proportional control

    match constraint {
        Constraint::Bond {
            atoms: (a1, a2), ..
        } => {
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
            for (i, _) in geometry.coords.iter().enumerate() {
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
            for (i, _) in geometry.coords.iter().enumerate() {
                let mut geom_plus = geometry.clone();
                geom_plus.coords[i] += delta;
                let violation_plus = calculate_constraint_violation(&geom_plus, constraint);

                force[i] = -(violation_plus - violation) / delta;
            }
        }
    }

    force
}

/// Optimizes a reaction path using the Nudged Elastic Band (NEB) method.
///
/// The NEB method is a powerful technique for finding minimum energy paths (MEPs)
/// between known reactant and product structures. It optimizes a chain of intermediate
/// images (geometries) connected by springs to find the most probable reaction pathway.
///
/// # Theoretical Foundation
///
/// ## The NEB Method
/// NEB balances two competing forces on each intermediate image:
///
/// 1. **Spring Forces (F^spring)**: Keep images evenly distributed along the path
/// 2. **True Forces (F^true)**: Drive images toward lower energy regions
///
/// The total NEB force on image i is:
/// ```text
/// F_i^NEB = F_i^spring + F_i^⊥
/// ```
///
/// where F_i^⊥ is the component of the true force perpendicular to the path.
///
/// ## Spring Force Calculation
/// The spring force along the path tangent τ̂ is:
/// ```text
/// F_i^spring = k[(|R_{i+1} - R_i| - |R_i - R_{i-1}|)] τ̂_i
/// ```
///
/// where:
/// - k is the spring constant (typically 0.1-1.0 eV/Angstrom²)
/// - R_i represents the coordinates of image i
/// - τ̂_i is the normalized tangent vector at image i
///
/// ## True Force Projection
/// The perpendicular component of the true force is:
/// ```text
/// F_i^⊥ = F_i^true - (F_i^true · τ̂_i) τ̂_i
/// ```
///
/// This ensures forces don't interfere with spring-controlled spacing.
///
/// # Algorithm Overview
///
/// 1. **Initialization**: Start with initial path (linear interpolation or guess)
/// 2. **Force Calculation**: For each intermediate image:
///    - Calculate tangent vector from neighboring images
///    - Compute spring forces along the tangent
///    - Project true forces perpendicular to path
/// 3. **Geometry Update**: Move each image according to total NEB force
/// 4. **Convergence Check**: Continue until forces are below threshold
/// 5. **Constraint Application**: Apply any geometric constraints if specified
///
/// # Arguments
///
/// * `initial_path` - Starting path as a series of geometries (reactant → product)
/// * `constraints` - Optional geometric constraints to maintain during optimization
///
/// # Returns
///
/// Returns an optimized `Vec<Geometry>` representing the minimum energy path.
/// The first and last geometries (endpoints) remain fixed during optimization.
///
/// # Implementation Details
///
/// ## Current Implementation (Simplified)
/// This implementation uses a simplified NEB approach:
/// - **Spring forces only**: True energy gradients not included
/// - **Fixed endpoints**: Reactant and product geometries unchanged
/// - **Steepest descent**: Simple optimization without advanced methods
/// - **Basic constraints**: Limited constraint handling
///
/// ## Parameters
/// - **Spring constant**: k = 0.1 (relatively soft springs)
/// - **Step size**: α = 0.01 (conservative for stability)
/// - **Max iterations**: 100 (sufficient for most systems)
/// - **Convergence threshold**: 1e-3 (moderate precision)
///
/// # Applications
///
/// - **Transition State Location**: Find saddle points along reaction paths
/// - **Reaction Mechanism Studies**: Understand detailed reaction pathways
/// - **Activation Energy Calculation**: Determine energy barriers
/// - **Conformational Transitions**: Study large-scale molecular motions
///
/// # Examples
///
/// ```rust
/// use omecp::reaction_path::optimize_reaction_path;
///
/// // Create initial path by linear interpolation
/// let initial_path = vec![reactant_geom, intermediate_geom, product_geom];
///
/// // Optimize the path using NEB
/// let optimized_path = optimize_reaction_path(&initial_path, &[]);
///
/// println!("Optimized path has {} images", optimized_path.len());
/// ```
///
/// # Advanced NEB Variants (Not Implemented)
///
/// - **Climbing Image NEB (CI-NEB)**: Drives highest energy image to saddle point
/// - **Adaptive NEB**: Automatically adjusts number of images
/// - **String Method**: Alternative path optimization approach
/// - **Growing String Method**: Dynamically extends path length
///
/// # References
///
/// - Henkelman, G.; Jónsson, H. *J. Chem. Phys.* **2000**, 113, 9978-9985.
/// - Henkelman, G.; Uberuaga, B. P.; Jónsson, H. *J. Chem. Phys.* **2000**, 113, 9901-9904.
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
                    if let Constraint::Bond {
                        atoms: (a1, a2),
                        target,
                    } = constraint
                    {
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
            println!(
                "NEB iteration {}: max force = {:.6}",
                iteration + 1,
                max_force
            );
        }
    }

    path
}

/// Calculates the NEB force for a specific image in the reaction path.
///
/// This function computes the spring forces that maintain proper spacing between
/// images in the NEB method. It implements the core force calculation that drives
/// the path optimization toward the minimum energy pathway.
///
/// # Theoretical Background
///
/// The NEB force on image i consists of spring forces along the path tangent:
/// ```text
/// F_i^spring = k × [(d_{i+1} - d_avg) - (d_i - d_avg)] × τ̂_i
/// ```
///
/// where:
/// - k is the spring constant
/// - d_i is the distance from image i to image i-1
/// - d_avg is the average spacing between all images
/// - τ̂_i is the normalized tangent vector at image i
///
/// # Algorithm Steps
///
/// 1. **Neighbor Identification**: Get previous and next images
/// 2. **Tangent Calculation**: Compute path tangent at current image
/// 3. **Distance Measurement**: Calculate distances to neighboring images
/// 4. **Spring Force**: Apply Hooke's law along the tangent direction
/// 5. **Force Balancing**: Ensure forces maintain even spacing
///
/// # Arguments
///
/// * `path` - Complete reaction path as array of geometries
/// * `image_index` - Index of the current image (0 = first, n-1 = last)
/// * `spring_constant` - Spring stiffness parameter (typically 0.1-1.0)
///
/// # Returns
///
/// Vector of forces with length 3×N_atoms, representing [Fx, Fy, Fz] components
/// for each atom in the molecule.
///
/// # Force Direction Logic
///
/// ## Interior Images (i = 1, 2, ..., n-2)
/// - **Previous spring**: Pulls toward image i-1 if too far apart
/// - **Next spring**: Pulls toward image i+1 if too far apart
/// - **Net effect**: Maintains even spacing along path
///
/// ## Endpoint Images (i = 0, n-1)
/// - **Fixed positions**: No forces applied (endpoints don't move)
/// - **Boundary conditions**: Provide reference for neighboring images
///
/// # Spring Constant Effects
///
/// - **High k (stiff springs)**: Images stay evenly spaced but may resist optimization
/// - **Low k (soft springs)**: Images can cluster but may lose path resolution
/// - **Typical values**: k = 0.1-0.5 eV/Angstrom² for most chemical systems
///
/// # Implementation Notes
///
/// ## Current Limitations
/// - **Spring forces only**: True energy gradients not included
/// - **Simple spacing**: Uses RMSD rather than arc length
/// - **No adaptive springs**: Constant spring constant throughout
///
/// ## Distance Metrics
/// - **RMSD**: Root mean square deviation between geometries
/// - **Average spacing**: Mean distance between consecutive images
/// - **Force scaling**: Proportional to deviation from average spacing
fn calculate_neb_force(path: &[Geometry], image_index: usize, spring_constant: f64) -> Vec<f64> {
    let n_images = path.len();
    let current = &path[image_index];

    // Get neighboring images
    let prev = if image_index > 0 {
        Some(&path[image_index - 1])
    } else {
        None
    };
    let next = if image_index < n_images - 1 {
        Some(&path[image_index + 1])
    } else {
        None
    };

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

/// Calculates the tangent vector for NEB path optimization.
///
/// The tangent vector defines the local direction of the reaction path at each
/// image. It's crucial for the NEB method because it determines how spring forces
/// are applied and ensures that true forces don't interfere with path spacing.
///
/// # Theoretical Background
///
/// The tangent vector τ̂_i at image i represents the local path direction:
/// ```text
/// τ̂_i = (path direction at image i) / |path direction|
/// ```
///
/// Different algorithms exist for computing tangents:
/// - **Simple tangent**: τ = R_{i+1} - R_{i-1}
/// - **Bisector tangent**: τ = (R_{i+1} - R_i) + (R_i - R_{i-1})
/// - **Improved tangent**: Energy-weighted combination (not implemented here)
///
/// # Algorithm Implementation
///
/// ## Interior Images (both neighbors exist)
/// Uses the bisector method:
/// ```text
/// v_plus = (R_{i+1} - R_i) / |R_{i+1} - R_i|
/// v_minus = (R_i - R_{i-1}) / |R_i - R_{i-1}|
/// τ = v_plus + v_minus
/// τ̂ = τ / |τ|
/// ```
///
/// ## Boundary Images (one neighbor missing)
/// Uses simple difference:
/// ```text
/// τ = R_neighbor - R_current
/// τ̂ = τ / |τ|
/// ```
///
/// # Arguments
///
/// * `prev` - Previous geometry in path (None for first image)
/// * `current` - Current geometry for which to calculate tangent
/// * `next` - Next geometry in path (None for last image)
///
/// # Returns
///
/// Normalized tangent vector with length 3×N_atoms, representing the path
/// direction in Cartesian coordinate space.
///
/// # Tangent Vector Properties
///
/// ## Normalization
/// The tangent is always normalized: |τ̂| = 1
/// This ensures consistent force scaling regardless of path curvature.
///
/// ## Smoothness
/// The bisector method provides smoother tangents than simple differences,
/// reducing oscillations in the optimization.
///
/// ## Boundary Handling
/// Special treatment for endpoints prevents undefined tangents and
/// maintains proper force directions.
///
/// # Mathematical Details
///
/// ## Vector Operations
/// - **Subtraction**: Component-wise difference between geometries
/// - **Normalization**: Division by Euclidean norm
/// - **Addition**: Component-wise sum for bisector calculation
///
/// ## Coordinate Representation
/// The tangent vector contains [dx₁, dy₁, dz₁, dx₂, dy₂, dz₂, ...] where
/// (dxᵢ, dyᵢ, dzᵢ) represents the path direction for atom i.
///
/// # Applications in NEB
///
/// - **Spring force direction**: Springs act along τ̂
/// - **Force projection**: True forces projected perpendicular to τ̂
/// - **Path characterization**: Tangent describes local path geometry
fn calculate_tangent(
    prev: Option<&Geometry>,
    current: &Geometry,
    next: Option<&Geometry>,
) -> Vec<f64> {
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
        for val in &mut tangent {
            *val /= norm;
        }
    }

    tangent
}

/// Calculates the Root Mean Square Deviation (RMSD) between two molecular geometries.
///
/// RMSD is a standard metric for measuring the similarity between molecular structures.
/// It quantifies the average distance between corresponding atoms in two geometries,
/// providing a single number that represents structural similarity.
///
/// # Mathematical Definition
///
/// RMSD is calculated as:
/// ```text
/// RMSD = √[(1/N) × Σᵢ(rᵢ₁ - rᵢ₂)²]
/// ```
///
/// where:
/// - N is the total number of coordinate components (3 × number of atoms)
/// - rᵢ₁, rᵢ₂ are corresponding coordinate components in the two geometries
/// - The sum is over all x, y, z coordinates of all atoms
///
/// # Arguments
///
/// * `geom1` - First molecular geometry
/// * `geom2` - Second molecular geometry (must have same number of atoms)
///
/// # Returns
///
/// RMSD value in the same units as the input coordinates (typically Angstroms).
///
/// # Interpretation
///
/// - **RMSD ≈ 0**: Geometries are nearly identical
/// - **RMSD < 0.1 Angstrom**: Very similar structures (small conformational changes)
/// - **RMSD 0.1-0.5 Angstrom**: Moderate structural differences
/// - **RMSD > 0.5 Angstrom**: Significant structural changes
///
/// # Applications in Path Methods
///
/// - **Image spacing**: Measure distances between path images
/// - **Convergence criteria**: Check if optimization has converged
/// - **Path length calculation**: Sum of RMSD values along path
/// - **Quality assessment**: Evaluate path smoothness
fn calculate_rmsd(geom1: &Geometry, geom2: &Geometry) -> f64 {
    let mut sum_sq = 0.0;
    for i in 0..geom1.coords.len() {
        let diff = geom1.coords[i] - geom2.coords[i];
        sum_sq += diff * diff;
    }
    (sum_sq / geom1.coords.len() as f64).sqrt()
}

/// Subtracts two geometries coordinate-wise to compute displacement vectors.
///
/// This function performs element-wise subtraction of coordinate arrays to
/// compute the displacement vector between two molecular geometries. The
/// result represents the direction and magnitude of atomic movements.
///
/// # Mathematical Operation
///
/// For geometries with coordinates [x₁, y₁, z₁, x₂, y₂, z₂, ...]:
/// ```text
/// displacement = geom1 - geom2 = [x₁¹-x₁², y₁¹-y₁², z₁¹-z₁², ...]
/// ```
///
/// # Arguments
///
/// * `geom1` - First geometry (minuend)
/// * `geom2` - Second geometry (subtrahend)
///
/// # Returns
///
/// Vector of coordinate differences with length 3×N_atoms, where each group
/// of 3 elements represents [Δx, Δy, Δz] for one atom.
///
/// # Applications
///
/// - **Tangent vectors**: Compute path directions in NEB
/// - **Displacement analysis**: Understand atomic movements
/// - **Force calculations**: Determine direction of coordinate changes
/// - **Path characterization**: Analyze reaction coordinate changes
///
/// # Vector Interpretation
///
/// The resulting vector points from geom2 toward geom1:
/// - **Positive components**: Atom moved in positive coordinate direction
/// - **Negative components**: Atom moved in negative coordinate direction
/// - **Zero components**: No movement in that coordinate direction
fn subtract_geometries(geom1: &Geometry, geom2: &Geometry) -> Vec<f64> {
    geom1
        .coords
        .iter()
        .zip(geom2.coords.iter())
        .map(|(a, b)| a - b)
        .collect()
}

/// Calculates the average spacing between consecutive images in a reaction path.
///
/// This function computes the mean distance between neighboring geometries
/// along a reaction path. It's used in NEB to determine the reference spacing
/// for spring force calculations and to assess path quality.
///
/// # Mathematical Definition
///
/// Average spacing is calculated as:
/// ```text
/// d_avg = (1/(n-1)) × Σᵢ₌₁ⁿ⁻¹ RMSD(Rᵢ₊₁, Rᵢ)
/// ```
///
/// where:
/// - n is the number of images in the path
/// - RMSD(Rᵢ₊₁, Rᵢ) is the distance between consecutive images
/// - The sum is over all adjacent image pairs
///
/// # Arguments
///
/// * `path` - Array of geometries representing the reaction path
///
/// # Returns
///
/// Average spacing in the same units as coordinates (typically Angstroms).
/// Returns 0.0 for paths with fewer than 2 images.
///
/// # Applications in NEB
///
/// ## Spring Force Reference
/// The average spacing serves as the equilibrium length for NEB springs:
/// ```text
/// F_spring ∝ (d_actual - d_avg)
/// ```
///
/// ## Path Quality Assessment
/// - **Uniform spacing**: d_avg ≈ individual spacings (good path)
/// - **Non-uniform spacing**: Large variation in spacings (needs optimization)
///
/// ## Convergence Monitoring
/// - **Stable d_avg**: Path spacing has converged
/// - **Changing d_avg**: Path still optimizing
///
/// # Ideal Path Properties
///
/// - **Even spacing**: All consecutive distances ≈ d_avg
/// - **Smooth progression**: No large jumps between adjacent images
/// - **Appropriate resolution**: d_avg small enough to capture path features
fn get_average_spacing(path: &[Geometry]) -> f64 {
    if path.len() < 2 {
        return 0.0;
    }

    let mut total_distance = 0.0;
    for i in 1..path.len() {
        total_distance += calculate_rmsd(&path[i], &path[i - 1]);
    }

    total_distance / (path.len() - 1) as f64
}

/// Calculates the Euclidean distance between two atoms in a molecular geometry.
///
/// This function computes the bond length between two specified atoms using
/// the standard Euclidean distance formula in 3D Cartesian coordinates.
/// It's a fundamental operation in molecular geometry analysis.
///
/// # Mathematical Formula
///
/// The distance between atoms i and j is:
/// ```text
/// d_ij = √[(x_j - x_i)² + (y_j - y_i)² + (z_j - z_i)²]
/// ```
///
/// where (x_i, y_i, z_i) and (x_j, y_j, z_j) are the Cartesian coordinates
/// of atoms i and j, respectively.
///
/// # Arguments
///
/// * `geometry` - The molecular geometry containing atomic coordinates
/// * `atom1` - Index of the first atom (0-based)
/// * `atom2` - Index of the second atom (0-based)
///
/// # Returns
///
/// Distance between the two atoms in the same units as the input coordinates
/// (typically Angstroms).
///
/// # Applications
///
/// - **Bond length analysis**: Measure actual bond distances
/// - **Constraint evaluation**: Check if bond constraints are satisfied
/// - **Geometry validation**: Ensure reasonable molecular structure
/// - **Force calculations**: Compute bond-based constraint forces
///
/// # Typical Bond Lengths (for reference)
///
/// - **C-H**: ~1.1 Angstrom
/// - **C-C**: ~1.5 Angstrom (single), ~1.3 Angstrom (double), ~1.2 Angstrom (triple)
/// - **C-O**: ~1.4 Angstrom (single), ~1.2 Angstrom (double)
/// - **O-H**: ~1.0 Angstrom
/// - **N-H**: ~1.0 Angstrom
///
/// # Examples
///
/// ```rust
/// // Calculate C-H bond length
/// let ch_distance = calculate_distance(&geometry, 0, 1);
/// println!("C-H bond length: {:.3} Angstrom", ch_distance);
///
/// // Check if bond is within reasonable range
/// if ch_distance > 2.0 {
///     println!("Warning: Unusually long C-H bond!");
/// }
/// ```
fn calculate_distance(geometry: &Geometry, atom1: usize, atom2: usize) -> f64 {
    let pos1 = geometry.get_atom_coords(atom1);
    let pos2 = geometry.get_atom_coords(atom2);
    let dx = pos1[0] - pos2[0];
    let dy = pos1[1] - pos2[1];
    let dz = pos1[2] - pos2[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Adjusts the bond length between two atoms to a target value.
///
/// This function modifies the molecular geometry by moving one atom to achieve
/// a specific bond length. It's a simplified constraint satisfaction method
/// used in path optimization when geometric constraints must be maintained.
///
/// # Algorithm Description
///
/// The method uses a scaling approach:
/// 1. **Current distance**: Calculate existing bond length
/// 2. **Scaling factor**: Compute ratio of target to current length
/// 3. **Position adjustment**: Move atom2 along the bond vector
/// 4. **Coordinate update**: Apply the new position to the geometry
///
/// # Mathematical Implementation
///
/// The new position of atom2 is calculated as:
/// ```text
/// r₂_new = r₁ + (target_length / current_length) × (r₂ - r₁)
/// ```
///
/// where:
/// - r₁, r₂ are the current positions of atoms 1 and 2
/// - target_length is the desired bond length
/// - current_length is the existing bond length
///
/// # Arguments
///
/// * `geometry` - Mutable reference to the molecular geometry
/// * `atom1` - Index of the first atom (remains fixed)
/// * `atom2` - Index of the second atom (will be moved)
/// * `target_length` - Desired bond length in same units as coordinates
///
/// # Behavior Details
///
/// ## Fixed Atom Choice
/// - **Atom1**: Remains at its original position (anchor point)
/// - **Atom2**: Moved along the bond vector to achieve target distance
///
/// ## Direction Preservation
/// - The bond direction (unit vector) remains unchanged
/// - Only the magnitude (bond length) is modified
/// - No rotation or angular changes occur
///
/// ## Edge Cases
/// - **Zero distance**: Function returns early to avoid division by zero
/// - **Negative target**: Would invert bond direction (not physically meaningful)
///
/// # Applications
///
/// - **Constraint enforcement**: Maintain specific bond lengths during optimization
/// - **Geometry correction**: Fix unreasonable bond distances
/// - **Path optimization**: Apply bond constraints in NEB calculations
/// - **Structure preparation**: Set up initial geometries with desired bond lengths
///
/// # Limitations
///
/// ## Simplified Approach
/// - **Single bond focus**: Only considers the specified bond
/// - **No connectivity**: Ignores effects on other bonds and angles
/// - **No energy consideration**: Purely geometric adjustment
/// - **Fixed anchor**: Always keeps atom1 stationary
///
/// ## Potential Issues
/// - **Steric clashes**: May create unrealistic atomic overlaps
/// - **Angle distortion**: Can distort bond angles involving these atoms
/// - **Chain effects**: Changes may propagate through molecular structure
///
/// # Future Improvements
///
/// - **Mass-weighted adjustment**: Move both atoms based on their masses
/// - **Connectivity awareness**: Consider effects on neighboring bonds
/// - **Energy minimization**: Combine with local energy optimization
/// - **Multiple constraint handling**: Simultaneously satisfy several constraints
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

/// Analyzes a reaction path and computes comprehensive statistical information.
///
/// This function performs detailed analysis of a reaction path, calculating
/// various metrics that characterize the path quality, length, and properties.
/// It's essential for understanding reaction mechanisms and validating path
/// optimization results.
///
/// # Analysis Components
///
/// ## Path Length Calculation
/// Computes the total arc length along the reaction path:
/// ```text
/// L_total = Σᵢ₌₁ⁿ⁻¹ RMSD(Rᵢ₊₁, Rᵢ)
/// ```
///
/// This represents the total "distance" traveled in configuration space
/// from reactant to product.
///
/// ## Geometric Metrics
/// - **Number of images**: Total path resolution
/// - **Average spacing**: Mean distance between consecutive images
/// - **Path smoothness**: Variation in inter-image distances
///
/// ## Energy Analysis (Future Extension)
/// The framework supports energy analysis:
/// - **Barrier heights**: Maximum energy along path
/// - **Reaction energy**: Energy difference between endpoints
/// - **Energy profile**: Complete energy vs. reaction coordinate
///
/// # Arguments
///
/// * `geometries` - Array of molecular geometries representing the reaction path
///
/// # Returns
///
/// Returns a `PathStatistics` struct containing:
/// - `path_length`: Total path length in coordinate space
/// - `num_points`: Number of geometries in the path
/// - `energies`: Energy values (currently empty, for future use)
/// - `coordinates`: Reaction coordinate values (currently empty, for future use)
///
/// # Path Length Interpretation
///
/// ## Physical Meaning
/// The path length represents the "effort" required to transform the reactant
/// into the product through the specific pathway:
/// - **Short paths**: Direct, efficient transformations
/// - **Long paths**: Complex, multi-step mechanisms
/// - **Very long paths**: Potentially unphysical or poorly optimized
///
/// ## Typical Values
/// - **Simple reactions**: 1-5 Angstrom total path length
/// - **Complex rearrangements**: 5-20 Angstrom total path length
/// - **Conformational changes**: 2-10 Angstrom total path length
///
/// # Applications
///
/// ## Path Quality Assessment
/// - **Resolution check**: Ensure adequate number of images
/// - **Smoothness evaluation**: Identify poorly optimized regions
/// - **Convergence monitoring**: Track path changes during optimization
///
/// ## Mechanism Analysis
/// - **Pathway comparison**: Compare different reaction routes
/// - **Bottleneck identification**: Find regions requiring more images
/// - **Coordinate selection**: Choose appropriate reaction coordinates
///
/// ## Method Validation
/// - **Algorithm comparison**: Evaluate different path optimization methods
/// - **Parameter tuning**: Optimize NEB spring constants and convergence criteria
/// - **Benchmark studies**: Compare with experimental or high-level theoretical results
///
/// # Examples
///
/// ```rust
/// use omecp::reaction_path::analyze_reaction_path;
///
/// // Analyze an optimized reaction path
/// let stats = analyze_reaction_path(&optimized_path);
///
/// println!("Path analysis:");
/// println!("  Total length: {:.3} Angstrom", stats.path_length);
/// println!("  Number of images: {}", stats.num_points);
/// println!("  Average spacing: {:.3} Angstrom",
///          stats.path_length / (stats.num_points - 1) as f64);
/// ```
///
/// # Future Extensions
///
/// ## Energy Integration
/// When energy data becomes available:
/// ```rust
/// // Future capability
/// if !stats.energies.is_empty() {
///     let barrier_height = stats.energies.iter().fold(0.0, |a, &b| a.max(b));
///     println!("Activation barrier: {:.2} kcal/mol", barrier_height * 627.5);
/// }
/// ```
///
/// ## Advanced Metrics
/// - **Path curvature**: Measure of path deviation from straight line
/// - **Intrinsic reaction coordinate**: Arc length parameterization
/// - **Turning points**: Identify regions of high curvature
/// - **Bottleneck analysis**: Locate transition state regions
pub fn analyze_reaction_path(geometries: &[Geometry]) -> PathStatistics {
    let energies = Vec::new(); // Would be filled from QM calculations
    let coordinates = Vec::new();

    // Calculate path length
    let mut path_length = 0.0;
    for i in 1..geometries.len() {
        let coords1 = geometry_to_coords(&geometries[i - 1]);
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

/// Converts a molecular geometry to a flat coordinate vector.
///
/// This utility function extracts all atomic coordinates from a geometry object
/// and arranges them in a single flat vector. This representation is convenient
/// for mathematical operations, distance calculations, and vector arithmetic
/// used throughout reaction path methods.
///
/// # Data Layout
///
/// The output vector contains coordinates in the following order:
/// ```text
/// [x₁, y₁, z₁, x₂, y₂, z₂, ..., xₙ, yₙ, zₙ]
/// ```
///
/// where (xᵢ, yᵢ, zᵢ) are the Cartesian coordinates of atom i.
///
/// # Arguments
///
/// * `geom` - The molecular geometry to convert
///
/// # Returns
///
/// A `Vec<f64>` with length 3×N_atoms containing all coordinate components
/// in sequential order.
///
/// # Applications
///
/// ## Mathematical Operations
/// - **Vector arithmetic**: Addition, subtraction, scaling of entire geometries
/// - **Distance calculations**: RMSD and path length computations
/// - **Linear algebra**: Matrix operations on coordinate sets
///
/// ## Path Analysis
/// - **Interpolation**: Generate intermediate geometries along paths
/// - **Tangent vectors**: Compute path directions for NEB
/// - **Displacement analysis**: Study atomic movements during reactions
///
/// ## Optimization Algorithms
/// - **Gradient calculations**: Flat vectors for optimization routines
/// - **Constraint handling**: Vector operations for constraint satisfaction
/// - **Force applications**: Apply forces to all atoms simultaneously
///
/// # Memory Layout
///
/// The flat representation is memory-efficient and cache-friendly:
/// - **Contiguous storage**: All coordinates in a single array
/// - **Predictable access**: Sequential memory access patterns
/// - **SIMD compatibility**: Suitable for vectorized operations
///
/// # Examples
///
/// ```rust
/// // Convert geometry for mathematical operations
/// let coords = geometry_to_coords(&geometry);
/// println!("Total coordinates: {}", coords.len());
/// println!("Number of atoms: {}", coords.len() / 3);
///
/// // Access specific atom coordinates
/// let atom_index = 2;  // Third atom (0-based)
/// let x = coords[atom_index * 3];
/// let y = coords[atom_index * 3 + 1];
/// let z = coords[atom_index * 3 + 2];
/// println!("Atom {} position: ({:.3}, {:.3}, {:.3})", atom_index, x, y, z);
/// ```
///
/// # Coordinate System
///
/// - **Units**: Same as input geometry (typically Angstroms)
/// - **Origin**: Arbitrary (depends on input geometry)
/// - **Axes**: Standard Cartesian (x, y, z)
/// - **Handedness**: Right-handed coordinate system assumed
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
