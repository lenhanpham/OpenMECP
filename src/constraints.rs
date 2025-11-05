//! Geometric constraint system for molecular optimization.
//!
//! This module provides geometric constraints for MECP calculations.

use crate::geometry::Geometry;
use nalgebra::{DMatrix, DVector};
use std::error::Error;
use std::fmt;

/// Represents a geometric constraint that can be applied during optimization.
///
/// Constraints fix certain geometric parameters (bond lengths, angles, dihedrals)
/// to target values. They are enforced using Lagrange multipliers in the
/// optimization algorithm.
#[derive(Debug, Clone)]
pub enum Constraint {
    /// Fixes the bond length between two atoms.
    Bond {
        /// Tuple of (atom1_index, atom2_index) (0-based).
        atoms: (usize, usize),
        /// Target bond length in Angstroms.
        target: f64,
    },
    /// Fixes the bond angle between three atoms.
    Angle {
        /// Tuple of (atom1_index, atom2_index, atom3_index) (0-based).
        atoms: (usize, usize, usize),
        /// Target angle in radians.
        target: f64,
    },
    /// Fixes the dihedral angle between four atoms.
    Dihedral {
        /// Tuple of (atom1_index, atom2_index, atom3_index, atom4_index) (0-based).
        atoms: (usize, usize, usize, usize),
        /// Target dihedral angle in radians.
        target: f64,
    },
}

/// Calculates the current value of each constraint.
/// A positive value means the constraint is violated (e.g., bond is too long).
pub fn evaluate_constraints(geometry: &Geometry, constraints: &[Constraint]) -> DVector<f64> {
    let violations: Vec<f64> = constraints
        .iter()
        .map(|c| match c {
            Constraint::Bond { atoms, target } => {
                let p1 = geometry.get_atom_coords(atoms.0);
                let p2 = geometry.get_atom_coords(atoms.1);
                let dx = p1[0] - p2[0];
                let dy = p1[1] - p2[1];
                let dz = p1[2] - p2[2];
                (dx * dx + dy * dy + dz * dz).sqrt() - target
            }
            Constraint::Angle { atoms, target } => {
                let p1 = geometry.get_atom_coords(atoms.0);
                let p2 = geometry.get_atom_coords(atoms.1);
                let p3 = geometry.get_atom_coords(atoms.2);
                let v21 = [p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]];
                let v23 = [p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]];
                let dot = v21[0] * v23[0] + v21[1] * v23[1] + v21[2] * v23[2];
                let n21 = (v21[0].powi(2) + v21[1].powi(2) + v21[2].powi(2)).sqrt();
                let n23 = (v23[0].powi(2) + v23[1].powi(2) + v23[2].powi(2)).sqrt();
                (dot / (n21 * n23)).acos() - target
            }
            Constraint::Dihedral { atoms, target } => {
                let (a1, a2, a3, a4) = *atoms;
                let current_dihedral = calculate_dihedral(geometry, a1, a2, a3, a4);
                current_dihedral - target
            }
        })
        .collect();

    DVector::from_vec(violations)
}

/// Builds the constraint Jacobian matrix (C).
/// Each row is the gradient of a single constraint (∇gᵢ).
/// Dimensions: (num_constraints, 3 * num_atoms)
pub fn build_constraint_jacobian(geometry: &Geometry, constraints: &[Constraint]) -> DMatrix<f64> {
    let num_constraints = constraints.len();
    let num_dof = geometry.num_atoms * 3;
    let mut jacobian = DMatrix::zeros(num_constraints, num_dof);

    for (i, constraint) in constraints.iter().enumerate() {
        match constraint {
            Constraint::Bond {
                atoms: (a1, a2), ..
            } => {
                let grad = calculate_bond_gradient(geometry, *a1, *a2);
                // Place the gradient components into the correct row of the jacobian
                for j in 0..3 {
                    jacobian[(i, a1 * 3 + j)] = grad[j];
                    jacobian[(i, a2 * 3 + j)] = -grad[j];
                }
            }
            Constraint::Angle {
                atoms: (a1, a2, a3),
                ..
            } => {
                let (grad1, grad2, grad3) = calculate_angle_gradient(geometry, *a1, *a2, *a3);
                // Place gradients for all 3 atoms in the jacobian row
                for j in 0..3 {
                    jacobian[(i, a1 * 3 + j)] = grad1[j];
                    jacobian[(i, a2 * 3 + j)] = grad2[j];
                    jacobian[(i, a3 * 3 + j)] = grad3[j];
                }
            }
            Constraint::Dihedral { atoms, .. } => {
                let (a1, a2, a3, a4) = *atoms;
                let (grad1, grad2, grad3, grad4) =
                    calculate_dihedral_gradient(geometry, a1, a2, a3, a4);
                // Place gradients for all 4 atoms in the jacobian row
                for j in 0..3 {
                    jacobian[(i, a1 * 3 + j)] = grad1[j];
                    jacobian[(i, a2 * 3 + j)] = grad2[j];
                    jacobian[(i, a3 * 3 + j)] = grad3[j];
                    jacobian[(i, a4 * 3 + j)] = grad4[j];
                }
            }
        }
    }
    jacobian
}

// Helper function for bond gradient
fn calculate_bond_gradient(geometry: &Geometry, a1: usize, a2: usize) -> [f64; 3] {
    let pos1 = geometry.get_atom_coords(a1);
    let pos2 = geometry.get_atom_coords(a2);
    let vec = [pos1[0] - pos2[0], pos1[1] - pos2[1], pos1[2] - pos2[2]];
    let norm = (vec[0].powi(2) + vec[1].powi(2) + vec[2].powi(2)).sqrt();
    if norm == 0.0 {
        return [0.0, 0.0, 0.0];
    }
    [vec[0] / norm, vec[1] / norm, vec[2] / norm]
}

// calculate_angle_gradient
fn calculate_angle_gradient(
    geometry: &Geometry,
    i: usize,
    j: usize,
    k: usize,
) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let pi = geometry.get_atom_coords(i);
    let pj = geometry.get_atom_coords(j);
    let pk = geometry.get_atom_coords(k);

    let r_ji = [pi[0] - pj[0], pi[1] - pj[1], pi[2] - pj[2]];
    let r_jk = [pk[0] - pj[0], pk[1] - pj[1], pk[2] - pj[2]];

    let n_ji = (r_ji[0].powi(2) + r_ji[1].powi(2) + r_ji[2].powi(2)).sqrt();
    let n_jk = (r_jk[0].powi(2) + r_jk[1].powi(2) + r_jk[2].powi(2)).sqrt();

    let dot = r_ji[0] * r_jk[0] + r_ji[1] * r_jk[1] + r_ji[2] * r_jk[2];
    let cos_theta = dot / (n_ji * n_jk);

    // To avoid numerical instability with acos, clamp the value
    let cos_theta = cos_theta.clamp(-1.0, 1.0);
    let sin_theta = (1.0 - cos_theta.powi(2)).sqrt();

    if sin_theta.abs() < 1e-6 {
        return ([0.0; 3], [0.0; 3], [0.0; 3]);
    }

    let prefactor = -1.0 / (n_ji * n_jk * sin_theta);

    let term_i = [
        prefactor * (r_jk[0] / n_jk - cos_theta * r_ji[0] / n_ji),
        prefactor * (r_jk[1] / n_jk - cos_theta * r_ji[1] / n_ji),
        prefactor * (r_jk[2] / n_jk - cos_theta * r_ji[2] / n_ji),
    ];

    let term_k = [
        prefactor * (r_ji[0] / n_ji - cos_theta * r_jk[0] / n_jk),
        prefactor * (r_ji[1] / n_ji - cos_theta * r_jk[1] / n_jk),
        prefactor * (r_ji[2] / n_ji - cos_theta * r_jk[2] / n_jk),
    ];

    let term_j = [
        -term_i[0] - term_k[0],
        -term_i[1] - term_k[1],
        -term_i[2] - term_k[2],
    ];

    (term_i, term_j, term_k)
}

/// Helper function to compute cross product of two 3D vectors.
fn cross_product(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Helper function to compute dot product of two 3D vectors.
fn dot_product(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Calculate the dihedral angle between four atoms in radians.
///
/// The dihedral angle is computed using the atan2 method for proper sign.
/// The angle is in the range (-π, π].
///
/// # Arguments
///
/// * `geometry` - The molecular geometry containing atomic positions.
/// * `a1` - Index of the first atom (0-based).
/// * `a2` - Index of the second atom.
/// * `a3` - Index of the third atom.
/// * `a4` - Index of the fourth atom.
///
/// # Returns
///
/// The dihedral angle in radians. Returns 0.0 for degenerate cases.
pub fn calculate_dihedral(geometry: &Geometry, a1: usize, a2: usize, a3: usize, a4: usize) -> f64 {
    let p1 = geometry.get_atom_coords(a1);
    let p2 = geometry.get_atom_coords(a2);
    let p3 = geometry.get_atom_coords(a3);
    let p4 = geometry.get_atom_coords(a4);

    let v1 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
    let v2 = [p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]];
    let v3 = [p4[0] - p3[0], p4[1] - p3[1], p4[2] - p3[2]];

    let n1 = cross_product(&v1, &v2);
    let n2 = cross_product(&v2, &v3);

    let n1_norm = (n1[0].powi(2) + n1[1].powi(2) + n1[2].powi(2)).sqrt();
    let n2_norm = (n2[0].powi(2) + n2[1].powi(2) + n2[2].powi(2)).sqrt();

    if n1_norm < 1e-10 || n2_norm < 1e-10 {
        return 0.0;
    }

    let n1_unit = [n1[0] / n1_norm, n1[1] / n1_norm, n1[2] / n1_norm];
    let n2_unit = [n2[0] / n2_norm, n2[1] / n2_norm, n2[2] / n2_norm];

    // Calculate both cos and sin components for atan2
    let cos_phi = dot_product(&n1_unit, &n2_unit);

    // The sign is determined by (n1 × n2) · v2
    let cross_n1_n2 = cross_product(&n1_unit, &n2_unit);
    let sin_phi =
        dot_product(&cross_n1_n2, &v2) / (v2[0].powi(2) + v2[1].powi(2) + v2[2].powi(2)).sqrt();

    // Use atan2 for proper signed angle in (-π, π]
    sin_phi.atan2(cos_phi)
}

/// Calculate the analytical gradient of a dihedral angle with respect to atomic coordinates.
///
/// This function computes the partial derivatives of the dihedral angle φ with respect
/// to the coordinates of all four atoms involved. The gradients represent the force
/// vectors that would most effectively change the dihedral angle.
///
/// Uses the formula from Grigoryan lab reference.
///
/// # Arguments
///
/// * `geometry` - The molecular geometry
/// * `a1` - Index of first atom
/// * `a2` - Index of second atom
/// * `a3` - Index of third atom
/// * `a4` - Index of fourth atom
///
/// # Returns
///
/// A tuple of four 3D gradient vectors, one for each atom's coordinates.
/// Each gradient is in units of radians per unit distance.
pub fn calculate_dihedral_gradient(
    geometry: &Geometry,
    a1: usize,
    a2: usize,
    a3: usize,
    a4: usize,
) -> ([f64; 3], [f64; 3], [f64; 3], [f64; 3]) {
    let p1 = geometry.get_atom_coords(a1);
    let p2 = geometry.get_atom_coords(a2);
    let p3 = geometry.get_atom_coords(a3);
    let p4 = geometry.get_atom_coords(a4);

    let x1 = p1[0];
    let y1 = p1[1];
    let z1 = p1[2];
    let x2 = p2[0];
    let y2 = p2[1];
    let z2 = p2[2];
    let x3 = p3[0];
    let y3 = p3[1];
    let z3 = p3[2];
    let x4 = p4[0];
    let y4 = p4[1];
    let z4 = p4[2];

    // Compute N1 = (P2 - P1) × (P3 - P2)
    let n1x = (y2 - y1) * (z3 - z2) - (z2 - z1) * (y3 - y2);
    let n1y = (z2 - z1) * (x3 - x2) - (x2 - x1) * (z3 - z2);
    let n1z = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2);

    // Compute N2 = (P3 - P2) × (P4 - P3)
    let n2x = (y3 - y2) * (z4 - z3) - (z3 - z2) * (y4 - y3);
    let n2y = (z3 - z2) * (x4 - x3) - (x3 - x2) * (z4 - z3);
    let n2z = (x3 - x2) * (y4 - y3) - (y3 - y2) * (x4 - x3);

    let len_n1_sq = n1x.powi(2) + n1y.powi(2) + n1z.powi(2);
    let len_n2_sq = n2x.powi(2) + n2y.powi(2) + n2z.powi(2);

    if len_n1_sq < 1e-20 || len_n2_sq < 1e-20 {
        return ([0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3]);
    }

    let len_n1 = len_n1_sq.sqrt();
    let len_n2 = len_n2_sq.sqrt();

    let dot = n1x * n2x + n1y * n2y + n1z * n2z;
    let u = dot / (len_n1 * len_n2);

    let sin_a_sq = 1.0 - u.powi(2);
    if sin_a_sq < 1e-10 {
        return ([0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3]);
    }
    let sin_a = sin_a_sq.sqrt();

    // Compute ∂A/∂N1 components - CORRECTED: divide by sin_a, not multiply
    let da_dn1x = -(n2x - dot * n1x / len_n1_sq) / (len_n1 * len_n2 * sin_a);
    let da_dn1y = -(n2y - dot * n1y / len_n1_sq) / (len_n1 * len_n2 * sin_a);
    let da_dn1z = -(n2z - dot * n1z / len_n1_sq) / (len_n1 * len_n2 * sin_a);

    let da_dn2x = -(n1x - dot * n2x / len_n2_sq) / (len_n1 * len_n2 * sin_a);
    let da_dn2y = -(n1y - dot * n2y / len_n2_sq) / (len_n1 * len_n2 * sin_a);
    let da_dn2z = -(n1z - dot * n2z / len_n2_sq) / (len_n1 * len_n2 * sin_a);

    // Sigma = -sign(N1 · (P4 - P3)) - Keep the Grigoryan formula
    let sigma = -(n1x * (x4 - x3) + n1y * (y4 - y3) + n1z * (z4 - z3)).signum();

    // Calculate gradients
    let grad1_x = sigma * (da_dn1y * (z3 - z2) + da_dn1z * (y2 - y3));
    let grad1_y = sigma * (da_dn1z * (x3 - x2) + da_dn1x * (z2 - z3));
    let grad1_z = sigma * (da_dn1x * (y3 - y2) + da_dn1y * (x2 - x3));

    let grad2_x = sigma
        * (da_dn1y * (z1 - z3) + da_dn1z * (y3 - y1) + da_dn2y * (z3 - z4) + da_dn2z * (y4 - y3));
    let grad2_y = sigma
        * (da_dn1z * (x1 - x3) + da_dn1x * (z3 - z1) + da_dn2z * (x3 - x4) + da_dn2x * (z4 - z3));
    let grad2_z = sigma
        * (da_dn1x * (y3 - y1) + da_dn1y * (x1 - x3) + da_dn2x * (y4 - y3) + da_dn2y * (x3 - x4));

    let grad3_x = sigma
        * (da_dn1y * (z2 - z1) + da_dn1z * (y1 - y2) + da_dn2y * (z4 - z2) + da_dn2z * (y2 - y4));
    let grad3_y = sigma
        * (da_dn1z * (x2 - x1) + da_dn1x * (z1 - z2) + da_dn2z * (x4 - x2) + da_dn2x * (z2 - z4));
    let grad3_z = sigma
        * (da_dn1x * (y1 - y2) + da_dn1y * (x2 - x1) + da_dn2x * (y2 - y4) + da_dn2y * (x4 - x2));

    let grad4_x = sigma * (da_dn2y * (z2 - z3) + da_dn2z * (y3 - y2));
    let grad4_y = sigma * (da_dn2z * (x2 - x3) + da_dn2x * (z3 - z2));
    let grad4_z = sigma * (da_dn2x * (y2 - y3) + da_dn2y * (x3 - x2));

    (
        [grad1_x, grad1_y, grad1_z],
        [grad2_x, grad2_y, grad2_z],
        [grad3_x, grad3_y, grad3_z],
        [grad4_x, grad4_y, grad4_z],
    )
}

/// Error type for constraint-related operations.
#[derive(Debug)]
pub enum ConstraintError {
    /// Singular Jacobian matrix - constraints are linearly dependent
    SingularJacobian,
    /// Numerical instability in constraint solving
    NumericalInstability(String),
    /// Invalid constraint specification
    InvalidConstraint(String),
}

impl fmt::Display for ConstraintError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConstraintError::SingularJacobian => {
                write!(f, "Constraint Jacobian matrix is singular - constraints may be linearly dependent")
            }
            ConstraintError::NumericalInstability(msg) => {
                write!(f, "Numerical instability in constraint solving: {}", msg)
            }
            ConstraintError::InvalidConstraint(msg) => {
                write!(f, "Invalid constraint specification: {}", msg)
            }
        }
    }
}

impl Error for ConstraintError {}

/// Applies constraint forces using the Lagrange multiplier method.
///
/// This function implements the constrained optimization algorithm by:
/// 1. Evaluating constraint violations
/// 2. Building the constraint Jacobian matrix
/// 3. Solving for Lagrange multipliers using the constraint equation
/// 4. Applying constraint forces to the gradient
///
/// The constraint equation solved is:
/// C * C^T * λ = -g(x)
/// where C is the Jacobian, λ are the multipliers, and g(x) are violations.
///
/// The modified forces are: F_new = F_old + C^T * λ
///
/// # Arguments
///
/// * `geometry` - Current molecular geometry
/// * `forces` - Original forces from QM calculation (negative gradient)
/// * `constraints` - List of geometric constraints to enforce
/// * `lambdas` - Mutable reference to Lagrange multipliers (updated in-place)
///
/// # Returns
///
/// Modified forces with constraint contributions added.
///
/// # Errors
///
/// Returns `ConstraintError` if:
/// - The constraint Jacobian is singular (linearly dependent constraints)
/// - Numerical instability occurs during solving
/// - Constraints are improperly specified
///
/// # Examples
///
/// ```rust
/// use omecp::constraints::{add_constraint_lagrange, Constraint};
/// use omecp::geometry::Geometry;
/// use nalgebra::DVector;
///
/// let geometry = Geometry::new(/* ... */);
/// let forces = DVector::from_vec(vec![0.1, -0.2, 0.0, 0.3, -0.1, 0.2]);
/// let constraints = vec![
///     Constraint::Bond { atoms: (0, 1), target: 1.5 }
/// ];
/// let mut lambdas = vec![0.0];
///
/// let constrained_forces = add_constraint_lagrange(
///     &geometry, forces, &constraints, &mut lambdas
/// ).unwrap();
/// ```
pub fn add_constraint_lagrange(
    geometry: &Geometry,
    forces: DVector<f64>,
    constraints: &[Constraint],
    lambdas: &mut Vec<f64>,
) -> Result<DVector<f64>, ConstraintError> {
    // If no constraints, return original forces
    if constraints.is_empty() {
        lambdas.clear();
        return Ok(forces);
    }

    // Ensure lambdas vector has correct size
    lambdas.resize(constraints.len(), 0.0);

    // Evaluate constraint violations
    let violations = evaluate_constraints(geometry, constraints);
    
    // Build constraint Jacobian matrix C (num_constraints × 3*num_atoms)
    let jacobian = build_constraint_jacobian(geometry, constraints);
    
    // Check for zero violations - if all constraints are satisfied, no correction needed
    let max_violation = violations.iter().fold(0.0f64, |acc, &x| acc.max(x.abs()));
    if max_violation < 1e-12 {
        lambdas.fill(0.0);
        return Ok(forces);
    }

    // Solve the constraint equation: C * C^T * λ = -g(x)
    // where g(x) are the constraint violations
    let cct = &jacobian * jacobian.transpose();
    
    // Check if the matrix is singular
    let det = cct.determinant();
    if det.abs() < 1e-14 {
        return Err(ConstraintError::SingularJacobian);
    }

    // Solve for Lagrange multipliers
    let cct_inv = match cct.try_inverse() {
        Some(inv) => inv,
        None => return Err(ConstraintError::SingularJacobian),
    };
    
    let lambda_vec = cct_inv * (-violations);
    
    // Update lambdas vector
    for (i, &lambda) in lambda_vec.iter().enumerate() {
        if i < lambdas.len() {
            lambdas[i] = lambda;
        }
    }

    // Check for numerical instability
    for &lambda in &lambda_vec {
        if !lambda.is_finite() {
            return Err(ConstraintError::NumericalInstability(
                "Lagrange multipliers contain NaN or infinite values".to_string()
            ));
        }
    }

    // Apply constraint forces: F_new = F_old + C^T * λ
    let constraint_forces = jacobian.transpose() * lambda_vec;
    let modified_forces = forces + constraint_forces;

    Ok(modified_forces)
}

/// Validates constraint specifications for correctness.
///
/// Checks that:
/// - Atom indices are within valid range
/// - Target values are reasonable
/// - No duplicate constraints exist
///
/// # Arguments
///
/// * `constraints` - List of constraints to validate
/// * `num_atoms` - Total number of atoms in the system
///
/// # Returns
///
/// `Ok(())` if all constraints are valid, `Err(ConstraintError)` otherwise.
pub fn validate_constraints(
    constraints: &[Constraint],
    num_atoms: usize,
) -> Result<(), ConstraintError> {
    for (i, constraint) in constraints.iter().enumerate() {
        match constraint {
            Constraint::Bond { atoms: (a1, a2), target } => {
                if *a1 >= num_atoms || *a2 >= num_atoms {
                    return Err(ConstraintError::InvalidConstraint(
                        format!("Bond constraint {}: atom indices {} or {} exceed number of atoms {}", 
                               i, a1, a2, num_atoms)
                    ));
                }
                if a1 == a2 {
                    return Err(ConstraintError::InvalidConstraint(
                        format!("Bond constraint {}: cannot constrain atom {} to itself", i, a1)
                    ));
                }
                if *target <= 0.0 || *target > 10.0 {
                    return Err(ConstraintError::InvalidConstraint(
                        format!("Bond constraint {}: unreasonable target distance {:.3} Å", i, target)
                    ));
                }
            }
            Constraint::Angle { atoms: (a1, a2, a3), target } => {
                if *a1 >= num_atoms || *a2 >= num_atoms || *a3 >= num_atoms {
                    return Err(ConstraintError::InvalidConstraint(
                        format!("Angle constraint {}: atom indices {}, {}, or {} exceed number of atoms {}", 
                               i, a1, a2, a3, num_atoms)
                    ));
                }
                if a1 == a2 || a2 == a3 || a1 == a3 {
                    return Err(ConstraintError::InvalidConstraint(
                        format!("Angle constraint {}: duplicate atom indices {}, {}, {}", i, a1, a2, a3)
                    ));
                }
                if *target < 0.0 || *target > std::f64::consts::PI {
                    return Err(ConstraintError::InvalidConstraint(
                        format!("Angle constraint {}: target angle {:.3} rad is outside valid range [0, π]", 
                               i, target)
                    ));
                }
            }
            Constraint::Dihedral { atoms: (a1, a2, a3, a4), target } => {
                if *a1 >= num_atoms || *a2 >= num_atoms || *a3 >= num_atoms || *a4 >= num_atoms {
                    return Err(ConstraintError::InvalidConstraint(
                        format!("Dihedral constraint {}: atom indices {}, {}, {}, or {} exceed number of atoms {}", 
                               i, a1, a2, a3, a4, num_atoms)
                    ));
                }
                let atoms = [*a1, *a2, *a3, *a4];
                for j in 0..4 {
                    for k in (j+1)..4 {
                        if atoms[j] == atoms[k] {
                            return Err(ConstraintError::InvalidConstraint(
                                format!("Dihedral constraint {}: duplicate atom indices {}, {}, {}, {}", 
                                       i, a1, a2, a3, a4)
                            ));
                        }
                    }
                }
                if *target < -std::f64::consts::PI || *target > std::f64::consts::PI {
                    return Err(ConstraintError::InvalidConstraint(
                        format!("Dihedral constraint {}: target angle {:.3} rad is outside valid range [-π, π]", 
                               i, target)
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Reports the current status of all constraints during optimization.
///
/// This function evaluates all constraints and provides detailed status information
/// including current values, target values, violations, and convergence status.
///
/// # Arguments
///
/// * `geometry` - Current molecular geometry
/// * `constraints` - List of constraints to evaluate
/// * `lambdas` - Current Lagrange multipliers
/// * `step` - Current optimization step number
///
/// # Examples
///
/// ```rust
/// use omecp::constraints::{report_constraint_status, Constraint};
/// use omecp::geometry::Geometry;
///
/// let geometry = Geometry::new(/* ... */);
/// let constraints = vec![
///     Constraint::Bond { atoms: (0, 1), target: 1.5 }
/// ];
/// let lambdas = vec![0.1];
///
/// report_constraint_status(&geometry, &constraints, &lambdas, 5);
/// ```
pub fn report_constraint_status(
    geometry: &Geometry,
    constraints: &[Constraint],
    lambdas: &[f64],
    step: usize,
) {
    if constraints.is_empty() {
        return;
    }

    println!("\n--- Constraint Status (Step {}) ---", step);
    
    let violations = evaluate_constraints(geometry, constraints);
    let max_violation = violations.iter().fold(0.0f64, |acc, &x| acc.max(x.abs()));
    
    println!("Maximum constraint violation: {:.6}", max_violation);
    
    for (i, constraint) in constraints.iter().enumerate() {
        let violation = violations[i];
        let lambda = lambdas.get(i).copied().unwrap_or(0.0);
        
        match constraint {
            Constraint::Bond { atoms: (a, b), target } => {
                let current = calculate_bond_distance(geometry, *a, *b);
                println!(
                    "  Bond {}-{}: current={:.4} Å, target={:.4} Å, violation={:.6}, λ={:.6}",
                    a + 1, b + 1, current, target, violation, lambda
                );
            }
            Constraint::Angle { atoms: (a, b, c), target } => {
                let current = calculate_bond_angle(geometry, *a, *b, *c);
                println!(
                    "  Angle {}-{}-{}: current={:.2}°, target={:.2}°, violation={:.6}, λ={:.6}",
                    a + 1, b + 1, c + 1, 
                    current.to_degrees(), target.to_degrees(), violation, lambda
                );
            }
            Constraint::Dihedral { atoms: (a, b, c, d), target } => {
                let current = calculate_dihedral(geometry, *a, *b, *c, *d);
                println!(
                    "  Dihedral {}-{}-{}-{}: current={:.2}°, target={:.2}°, violation={:.6}, λ={:.6}",
                    a + 1, b + 1, c + 1, d + 1,
                    current.to_degrees(), target.to_degrees(), violation, lambda
                );
            }
        }
    }
    
    // Convergence status
    let converged = max_violation < 1e-6;
    println!("Constraint convergence: {}", if converged { "CONVERGED" } else { "NOT CONVERGED" });
    println!("--- End Constraint Status ---\n");
}

/// Calculates the current bond distance between two atoms.
fn calculate_bond_distance(geometry: &Geometry, a: usize, b: usize) -> f64 {
    let pos_a = geometry.get_atom_coords(a);
    let pos_b = geometry.get_atom_coords(b);
    let dx = pos_a[0] - pos_b[0];
    let dy = pos_a[1] - pos_b[1];
    let dz = pos_a[2] - pos_b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Calculates the current bond angle between three atoms.
fn calculate_bond_angle(geometry: &Geometry, a: usize, b: usize, c: usize) -> f64 {
    let pos_a = geometry.get_atom_coords(a);
    let pos_b = geometry.get_atom_coords(b);
    let pos_c = geometry.get_atom_coords(c);
    
    let v_ba = [pos_a[0] - pos_b[0], pos_a[1] - pos_b[1], pos_a[2] - pos_b[2]];
    let v_bc = [pos_c[0] - pos_b[0], pos_c[1] - pos_b[1], pos_c[2] - pos_b[2]];
    
    let dot = v_ba[0] * v_bc[0] + v_ba[1] * v_bc[1] + v_ba[2] * v_bc[2];
    let norm_ba = (v_ba[0].powi(2) + v_ba[1].powi(2) + v_ba[2].powi(2)).sqrt();
    let norm_bc = (v_bc[0].powi(2) + v_bc[1].powi(2) + v_bc[2].powi(2)).sqrt();
    
    if norm_ba < 1e-10 || norm_bc < 1e-10 {
        return 0.0;
    }
    
    let cos_angle = (dot / (norm_ba * norm_bc)).clamp(-1.0, 1.0);
    cos_angle.acos()
}
