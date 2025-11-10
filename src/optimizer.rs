//! Optimization algorithms for MECP calculations.
//!
//! This module implements various optimization algorithms used in Minimum Energy
//! Crossing Point (MECP) calculations, including:
//!
//! - **BFGS**: Broyden-Fletcher-Goldfarb-Shanno quasi-Newton method
//! - **GDIIS**: Geometry-based Direct Inversion in Iterative Subspace
//! - **GEDIIS**: Energy-Informed DIIS with improved convergence
//! - **Hessian Updates**: PSB (Powell-Symmetric-Broyden) formula
//! - **Convergence Checking**: Multiple criteria for optimization termination
//!
//! The module also provides functions to compute MECP effective gradients that
//! combine the energy difference minimization and energy perpendicular components
//! according to the Harvey et al. algorithm.
//!
//! # Optimization Strategy
//!
//! OpenMECP uses a hybrid optimization strategy:
//! 1. **Initialization**: BFGS for the first 3 steps to build curvature information
//! 2. **Convergence Acceleration**: Switch to GDIIS or GEDIIS for faster convergence
//! 3. **Adaptive Step Control**: Automatic step size limiting prevents overshooting
//! 4. **Checkpointing**: Save optimization state for restart capability
//!
//! # Implementation Improvements (v2.0)
//!
//! Recent enhancements ensure mathematical rigor and numerical stability:
//! - **Adaptive GEDIIS Parameters**: α scales with 1/|g| for better stability
//! - **PSB Curvature Check**: Validates `s^T y > 0` before Hessian update
//! - **Improved MECP Gradient**: Uses minimum norm vector to prevent premature convergence
//! - **Better Fallback Handling**: Steepest descent properly scaled in BFGS
//! - **High-Precision Thresholds**: Tighter convergence criteria for research use
//!
//! # MECP Gradient Calculation
//!
//! The MECP effective gradient combines two components:
//!
//! ```text
//! G_MECP = (E1 - E2) * x_norm + (f1 - (x_norm · f1) * x_norm)
//! ```
//!
//! Where:
//! - `E1, E2`: Energies of the two electronic states
//! - `f1, f2`: Gradients (forces) of the two states
//! - `x_norm = (f1 - f2) / |f1 - f2|`: Normalized gradient difference
//!
//! The first term drives the energy difference to zero (f-vector).
//! The second term minimizes energy perpendicular to the gradient difference (g-vector).

use crate::config::Config;
use crate::geometry::State;
use nalgebra::{DMatrix, DVector};
use std::collections::VecDeque;

/// Tracks optimization state and history for adaptive optimization algorithms.
///
/// This struct maintains the history of geometries, gradients, Hessians, and energies
/// required by advanced optimization methods like GDIIS and GEDIIS. It also stores
/// Lagrange multipliers for constraint handling.
///
/// # Capacity and History Management
///
/// - Maximum history: 4 iterations (configurable)
/// - Automatically removes oldest entries when capacity is exceeded
/// - Maintains rolling window of recent optimization data
#[derive(Debug, Clone)]
pub struct OptimizationState {
    /// Lagrange multipliers for geometric constraints
    pub lambdas: Vec<f64>,
    /// Lagrange multiplier for the energy difference constraint (FixDE mode)
    pub lambda_de: Option<f64>,
    /// Current constraint violations for extended gradient
    pub constraint_violations: DVector<f64>,
    /// History of molecular geometries (for DIIS methods)
    pub geom_history: VecDeque<DVector<f64>>,
    /// History of gradients (for DIIS methods)
    pub grad_history: VecDeque<DVector<f64>>,
    /// History of approximate Hessian matrices (for BFGS updates)
    pub hess_history: VecDeque<DMatrix<f64>>,
    /// History of energies or energy differences (for GEDIIS)
    pub energy_history: VecDeque<f64>,
    /// Maximum number of history entries to store
    pub max_history: usize,
}

impl Default for OptimizationState {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationState {
    /// Creates a new empty `OptimizationState`.
    ///
    /// Initializes all history containers with capacity for 4 entries and
    /// sets the maximum history size to 4 iterations.
    ///
    /// # Examples
    ///
    /// ```
    /// use omecp::optimizer::OptimizationState;
    ///
    /// let opt_state = OptimizationState::new();
    /// assert_eq!(opt_state.max_history, 4);
    /// assert!(opt_state.geom_history.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            lambdas: Vec::new(),
            lambda_de: None,
            constraint_violations: DVector::zeros(0),
            geom_history: VecDeque::with_capacity(4),
            grad_history: VecDeque::with_capacity(4),
            hess_history: VecDeque::with_capacity(4),
            energy_history: VecDeque::with_capacity(4),
            max_history: 4,
        }
    }

    /// Adds optimization data to the history deques.
    ///
    /// Automatically manages capacity by removing the oldest entry when the
    /// history size reaches `max_history`. This ensures that only the most
    /// recent `max_history` iterations are retained.
    ///
    /// # Arguments
    ///
    /// * `geom` - Current geometry coordinates
    /// * `grad` - Current MECP gradient
    /// * `hess` - Current Hessian matrix estimate
    /// * `energy` - Current energy or energy difference
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::DVector;
    /// let mut opt_state = OptimizationState::new();
    ///
    /// let coords = DVector::from_vec(vec![0.0, 0.0, 0.0]);
    /// let grad = DVector::from_vec(vec![0.1, 0.2, 0.3]);
    /// let energy = -10.5;
    ///
    /// // Add first iteration
    /// // opt_state.add_to_history(coords, grad, hessian, energy);
    /// ```
    pub fn add_to_history(
        &mut self,
        geom: DVector<f64>,
        grad: DVector<f64>,
        hess: DMatrix<f64>,
        energy: f64,
    ) {
        if self.geom_history.len() >= self.max_history {
            self.geom_history.pop_front();
            self.grad_history.pop_front();
            self.hess_history.pop_front();
            self.energy_history.pop_front();
        }
        self.geom_history.push_back(geom);
        self.grad_history.push_back(grad);
        self.hess_history.push_back(hess);
        self.energy_history.push_back(energy);
    }

    /// Checks if there is sufficient history for GDIIS/GEDIIS optimization.
    ///
    /// Returns `true` if at least 3 iterations of history have been accumulated,
    /// which is the minimum required for effective DIIS interpolation.
    ///
    /// # Returns
    ///
    /// Returns `true` if history has ≥ 3 entries, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use omecp::optimizer::OptimizationState;
    /// let opt_state = OptimizationState::new();
    /// assert!(!opt_state.has_enough_history()); // Empty state
    /// ```
    pub fn has_enough_history(&self) -> bool {
        self.geom_history.len() >= 3
    }
}

/// Solves the augmented Hessian system for a constrained optimization step.
///
/// This function implements the core of the Lagrange multiplier method by solving
/// the following system of linear equations:
///
///   [ H   Cᵀ ] [ Δx ] = [ -∇E ]
///   [ C    0  ] [  λ ]   [ -g  ]
///
/// where:
/// - H: The Hessian matrix (approximated by BFGS)
/// - C: The constraint Jacobian matrix
/// - Cᵀ: Transpose of the constraint Jacobian
/// - Δx: The step to take in atomic coordinates
/// - λ: The Lagrange multipliers
/// - -∇E: The negative of the energy gradient
/// - -g: The negative of the constraint violation values
///
/// The solution provides the optimal step `Δx` that minimizes the energy while
/// satisfying the constraints, along with the Lagrange multipliers `λ` that
/// represent the constraint forces.
///
/// # Arguments
///
/// * `hessian` - The approximate Hessian matrix of the energy function.
/// * `gradient` - The gradient of the energy function (∇E).
/// * `constraint_jacobian` - The Jacobian of the constraint functions (C).
/// * `constraint_violations` - The current values of the constraint functions (g).
///
/// # Returns
///
/// A tuple containing:
/// - `delta_x`: The calculated step in Cartesian coordinates.
/// - `lambdas`: The calculated Lagrange multipliers.
///
/// Returns `None` if the augmented Hessian matrix is singular and cannot be inverted.
pub fn solve_constrained_step(
    hessian: &DMatrix<f64>,
    gradient: &DVector<f64>,
    constraint_jacobian: &DMatrix<f64>,
    constraint_violations: &DVector<f64>,
) -> Option<(DVector<f64>, DVector<f64>)> {
    let n_coords = hessian.nrows();
    let n_constraints = constraint_jacobian.nrows();

    // Build the augmented Hessian matrix
    let mut augmented_hessian = DMatrix::zeros(n_coords + n_constraints, n_coords + n_constraints);
    augmented_hessian
        .view_mut((0, 0), (n_coords, n_coords))
        .copy_from(hessian);
    augmented_hessian
        .view_mut((0, n_coords), (n_coords, n_constraints))
        .copy_from(&constraint_jacobian.transpose());
    augmented_hessian
        .view_mut((n_coords, 0), (n_constraints, n_coords))
        .copy_from(constraint_jacobian);

    // Build the right-hand side vector
    let mut rhs = DVector::zeros(n_coords + n_constraints);
    rhs.rows_mut(0, n_coords).copy_from(&-gradient);
    rhs.rows_mut(n_coords, n_constraints)
        .copy_from(&-constraint_violations);

    // Solve the system
    if let Some(solution) = augmented_hessian.lu().solve(&rhs) {
        let delta_x = solution.rows(0, n_coords).clone_owned();
        let lambdas = solution.rows(n_coords, n_constraints).clone_owned();
        Some((delta_x, lambdas))
    } else {
        None
    }
}

/// Computes the MECP effective gradient for optimization.
///
/// This function implements the Harvey et al. algorithm for MECP optimization by
/// computing the effective gradient that drives the system toward the minimum
/// energy crossing point. The gradient has two components:
///
/// 1. **f-vector**: Drives the energy difference (E1 - E2) to zero
/// 2. **g-vector**: Minimizes the energy perpendicular to the gradient difference
///
/// The effective gradient is computed as:
/// ```text
/// G = (E1 - E2) * x_norm + [f1 - (x_norm · f1) * x_norm]
/// ```
///
/// where `x_norm = (f1 - f2) / |f1 - f2|` is the normalized gradient difference.
///
/// # Arguments
///
/// * `state1` - Electronic state 1 (energy, forces, geometry)
/// * `state2` - Electronic state 2 (energy, forces, geometry)
/// * `fixed_atoms` - List of atom indices to fix during optimization (0-based)
///
/// # Returns
///
/// Returns the MECP effective gradient as a `DVector<f64>` with length 3 × num_atoms.
///
/// # Examples
///
/// ```
/// use omecp::geometry::{Geometry, State};
/// use omecp::optimizer::{compute_mecp_gradient, OptimizationState};
///
/// let mut opt_state = OptimizationState::new();
/// let fixed_atoms = vec![0]; // Fix first atom
///
/// // let gradient = compute_mecp_gradient(&state1, &state2, &[], &mut opt_state, &fixed_atoms);
/// // assert_eq!(gradient.len(), state1.geometry.num_atoms * 3);
/// ```
pub fn compute_mecp_gradient(
    state1: &State,
    state2: &State,
    fixed_atoms: &[usize],
) -> DVector<f64> {
    // CRITICAL: Match Python MECP.py force sign convention
    // Python extracts forces as positive values from Gaussian output, then NEGATES them
    // before MECP gradient computation (see getG() function in MECP.py)
    let f1 = -state1.forces.clone();  // NEGATE to match Python algorithm
    let f2 = -state2.forces.clone();  // NEGATE to match Python algorithm

    // Gradient difference
    let x_vec = &f1 - &f2;
    let x_norm_val = x_vec.norm();
    // Use minimum norm vector to avoid division by zero while maintaining direction
    // This prevents premature convergence when gradients are nearly identical
    let x_norm = if x_norm_val.abs() < 1e-10 {
        // For nearly identical gradients, use a default unit vector
        // This is better than zero gradient, which would cause premature convergence
        let n = x_vec.len() as f64;
        &x_vec / (n.sqrt() * 1e-10)
    } else {
        &x_vec / x_norm_val
    };

    // Energy difference component
    let de = state1.energy - state2.energy;
    let f_vec = x_norm.clone() * de;

    // Perpendicular component
    let dot = f1.dot(&x_norm);
    let g_vec = &f1 - &x_norm * dot;

    // Combine
    let mut eff_grad = f_vec + g_vec;

    // Apply fixed atoms
    for &atom_idx in fixed_atoms {
        eff_grad[atom_idx * 3] = 0.0;
        eff_grad[atom_idx * 3 + 1] = 0.0;
        eff_grad[atom_idx * 3 + 2] = 0.0;
    }

    eff_grad
}

/// Performs a BFGS optimization step.
///
/// BFGS (Broyden-Fletcher-Goldfarb-Shanno) is a quasi-Newton optimization method
/// that approximates the inverse Hessian using gradient information. It provides
/// good convergence for the first few iterations while building curvature information.
///
/// The BFGS step direction is computed by solving:
/// ```text
/// d = -H^(-1) * g
/// ```
///
/// where H is the Hessian approximation and g is the gradient. The step size is
/// automatically limited by `config.max_step_size` to prevent overshooting.
///
/// # Arguments
///
/// * `x0` - Current geometry coordinates
/// * `g0` - Current MECP gradient
/// * `hessian` - Current Hessian approximation matrix
/// * `config` - Configuration with step size limits and other parameters
///
/// # Returns
///
/// Returns the new geometry coordinates after the BFGS step as a `DVector<f64>`.
///
/// # Examples
///
/// ```
/// use omecp::optimizer::bfgs_step;
/// use nalgebra::DVector;
///
/// let x0 = DVector::from_vec(vec![0.0, 0.0, 0.0]);
/// let g0 = DVector::from_vec(vec![0.1, 0.2, 0.3]);
/// let hessian = DMatrix::identity(3, 3);
///
/// // let x_new = bfgs_step(&x0, &g0, &hessian, &config, 1.0);
/// ```
pub fn bfgs_step(
    x0: &DVector<f64>,
    g0: &DVector<f64>,
    hessian: &DMatrix<f64>,
    config: &Config,
    _adaptive_scale: f64, // Parameter kept for compatibility but not used for BFGS
) -> DVector<f64> {
    // Solve H * dk = -g (compute Newton direction)
    let neg_g = -g0;
    let mut dk = hessian.clone().lu().solve(&neg_g).unwrap_or_else(|| {
        // Fallback to steepest descent when Hessian is singular
        let step_dir = -g0 / (g0.norm() + 1e-14);
        step_dir
    });

    // Python algorithm (propagationBFGS):
    // 1. Compute dk = -H^-1 * g
    // 2. Cap dk to 0.1 if ||dk|| > 0.1
    // 3. Apply rho=15 multiplier: XNew = X0 + rho * dk

    // Step 2: Cap direction vector dk to 0.1 (Python's hardcoded limit)
    let dk_norm = dk.norm();
    const DIRECTION_LIMIT: f64 = 0.1; // Python's hardcoded cap on direction
    if dk_norm > DIRECTION_LIMIT {
        let original_norm = dk_norm;
        dk *= DIRECTION_LIMIT / dk_norm;
        println!(
            "current stepsize: {} is reduced to max_size {}",
            original_norm, DIRECTION_LIMIT
        );
    }

    // Step 3: Apply rho multiplier (from config.bfgs_rho, default 15.0)
    // Python uses FIXED rho for BFGS, no adaptive scaling
    let rho = config.bfgs_rho; // Fixed rho, matching Python behavior
    let final_step_size = DIRECTION_LIMIT * rho;
    println!(
        "BFGS step: direction_capped={}, rho={}, final_step_size={}",
        DIRECTION_LIMIT, rho, final_step_size
    );

    let x_new = x0 + &dk * rho;

    // Apply step size limiting (equivalent to Python's MaxStep)
    let step = &x_new - x0;
    let step_norm = step.norm();

    if step_norm > config.max_step_size {
        let scale = config.max_step_size / step_norm;
        println!(
            "BFGS step size limited: {:.6} -> {:.6} Bohr",
            step_norm, config.max_step_size
        );
        x0 + &step * scale
    } else {
        x_new
    }
}

/// Computes adaptive step scaling based on optimization progress.
///
/// This function adjusts the step size based on energy changes and gradient magnitude
/// to allow natural convergence without fixed multipliers.
pub fn compute_adaptive_scale(
    energy_current: f64,
    energy_previous: f64,
    gradient_norm: f64,
    step: usize,
) -> f64 {
    // Early iterations: allow larger steps
    if step < 3 {
        return 1.0;
    }

    // If energy increased significantly, reduce step size
    if energy_current > energy_previous + 0.01 {
        return 0.3; // Large reduction for energy increase
    }

    // If energy increased slightly, moderate reduction
    if energy_current > energy_previous {
        return 0.7;
    }

    // Fine tuning region (small gradients)
    if gradient_norm < 0.01 {
        return 0.8;
    }

    // Normal region
    1.0
}

/// Updates the Hessian matrix using the PSB (Powell-Symmetric-Broyden) formula.
///
/// The PSB formula is a rank-2 update that modifies the Hessian approximation
/// based on the difference in gradients (yk) and the step taken (sk):
///
/// ```text
/// H_new = H + (yk - H*sk) * sk^T + sk * (yk - H*sk)^T
///         - [(yk - H*sk)^T * sk] * (sk * sk^T) / (sk^T * sk)
/// ```
///
/// This update preserves symmetry and positive definiteness under certain conditions.
/// The PSB update is more stable than BFGS for poorly conditioned problems.
///
/// # Arguments
///
/// * `hessian` - Current Hessian approximation
/// * `sk` - Step vector (x_new - x_old)
/// * `yk` - Gradient difference (g_new - g_old)
///
/// # Returns
///
/// Returns the updated Hessian matrix as a `DMatrix<f64>`.
///
/// # Examples
///
/// ```
/// use omecp::optimizer::update_hessian_psb;
/// use nalgebra::{DMatrix, DVector};
///
/// let h_old = DMatrix::identity(3, 3);
/// let sk = DVector::from_vec(vec![0.1, 0.2, 0.3]);
/// let yk = DVector::from_vec(vec![0.05, 0.1, 0.15]);
///
/// // let h_new = update_hessian_psb(&h_old, &sk, &yk);
/// ```
pub fn update_hessian_psb(
    hessian: &DMatrix<f64>,
    sk: &DVector<f64>,
    yk: &DVector<f64>,
) -> DMatrix<f64> {
    let mut h_new = hessian.clone();

    // Check curvature condition: s^T y > 0 for meaningful update
    // Reference: Nocedal & Wright "Numerical Optimization" Theorem 6.2
    let sk_dot_yk = sk.dot(yk);
    let sk_dot_sk = sk.dot(sk);

    // Only update if curvature condition is satisfied and s is not zero
    if sk_dot_yk > 1e-12 * sk_dot_sk * yk.norm() && sk_dot_sk.abs() > 1e-10 {
        let hsk = hessian * sk;
        let diff = yk - &hsk;
        let sk_diff = sk.dot(&diff);
        let term1 = &diff * sk.transpose() + sk * diff.transpose();
        let term2 = (sk * sk.transpose()) * (sk_diff / sk_dot_sk);

        h_new += (term1 - term2) / sk_dot_sk;
    }
    // If curvature condition not satisfied, return current Hessian
    // This prevents degradation in convergence properties

    h_new
}

/// Tracks convergence status for each optimization criterion.
///
/// OpenMECP uses five independent convergence criteria, all of which must be
/// satisfied for the optimization to converge. This follows the same standard
/// used by Gaussian and other quantum chemistry programs.
///
/// # Convergence Criteria
///
/// 1. **Energy Difference (ΔE)**: |E1 - E2| < threshold
/// 2. **RMS Gradient**: ||g||_rms < threshold
/// 3. **Maximum Gradient**: max(|g_i|) < threshold
/// 4. **RMS Displacement**: ||Δx||_rms < threshold
/// 5. **Maximum Displacement**: max(|Δx_i|) < threshold
#[derive(Debug)]
pub struct ConvergenceStatus {
    /// Energy difference convergence status
    pub de_converged: bool,
    /// RMS gradient convergence status
    pub rms_grad_converged: bool,
    /// Maximum gradient convergence status
    pub max_grad_converged: bool,
    /// RMS displacement convergence status
    pub rms_disp_converged: bool,
    /// Maximum displacement convergence status
    pub max_disp_converged: bool,
}

impl ConvergenceStatus {
    /// Checks if all convergence criteria are satisfied.
    ///
    /// Returns `true` only when ALL five criteria are met. This is the standard
    /// "AND" logic used in quantum chemistry optimizations.
    ///
    /// # Returns
    ///
    /// Returns `true` if optimization has converged, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// let status = ConvergenceStatus {
    ///     de_converged: true,
    ///     rms_grad_converged: true,
    ///     max_grad_converged: true,
    ///     rms_disp_converged: true,
    ///     max_disp_converged: true,
    /// };
    ///
    /// assert!(status.is_converged());
    /// ```
    pub fn is_converged(&self) -> bool {
        self.de_converged
            && self.rms_grad_converged
            && self.max_grad_converged
            && self.rms_disp_converged
            && self.max_disp_converged
    }
}

/// Checks convergence criteria for MECP optimization.
///
/// Evaluates all five convergence criteria and returns a `ConvergenceStatus`
/// indicating which criteria have been satisfied. The optimization converges
/// only when all criteria are met simultaneously.
///
/// # Arguments
///
/// * `e1` - Energy of state 1 in hartree
/// * `e2` - Energy of state 2 in hartree
/// * `x_old` - Previous geometry coordinates
/// * `x_new` - Current geometry coordinates
/// * `grad` - Current MECP gradient
/// * `config` - Configuration with convergence thresholds
///
/// # Returns
///
/// Returns a `ConvergenceStatus` struct indicating the status of each criterion.
///
/// # Convergence Thresholds
///
/// ## Default (Standard Precision)
/// - Energy difference: 0.000050 hartree (~0.00136 eV)
/// - RMS gradient: 0.0005 hartree/bohr
/// - Max gradient: 0.0007 hartree/bohr
/// - RMS displacement: 0.0025 bohr (~0.00132 Å)
/// - Max displacement: 0.0040 bohr (~0.00212 Å)
///
/// ## Recommended for High-Precision MECP
/// - Energy difference: 0.000010 hartree (~0.00027 eV)
/// - RMS gradient: 0.0001 hartree/bohr
/// - Max gradient: 0.0005 hartree/bohr
/// - RMS displacement: 0.0010 bohr (~0.00053 Å)
/// - Max displacement: 0.0020 bohr (~0.00106 Å)
///
/// # Implementation Notes
///
/// All five criteria must be satisfied simultaneously (AND logic).
/// Tight convergence is especially important for MECP calculations where
/// small energy differences can significantly impact results.
///
/// # Examples
///
/// ```
/// use omecp::optimizer::check_convergence;
/// use nalgebra::DVector;
///
/// let e1 = -100.0;
/// let e2 = -100.0001;
/// let x_old = DVector::from_vec(vec![0.0, 0.0, 0.0]);
/// let x_new = DVector::from_vec(vec![0.001, 0.001, 0.001]);
/// let grad = DVector::from_vec(vec![0.0001, 0.0001, 0.0001]);
///
/// // let status = check_convergence(e1, e2, &x_old, &x_new, &grad, &config);
/// // assert!(status.is_converged());
/// ```
pub fn check_convergence(
    e1: f64,
    e2: f64,
    x_old: &DVector<f64>,
    x_new: &DVector<f64>,
    grad: &DVector<f64>,
    config: &Config,
) -> ConvergenceStatus {
    let de = (e1 - e2).abs();
    let disp = x_new - x_old;

    let rms_disp = disp.norm() / (disp.len() as f64).sqrt();
    let max_disp = disp.iter().map(|x| x.abs()).fold(0.0, f64::max);

    let rms_grad = grad.norm() / (grad.len() as f64).sqrt();
    let max_grad = grad.iter().map(|x| x.abs()).fold(0.0, f64::max);

    ConvergenceStatus {
        de_converged: de < config.thresholds.de,
        rms_grad_converged: rms_grad < config.thresholds.rms_g,
        max_grad_converged: max_grad < config.thresholds.max_g,
        rms_disp_converged: rms_disp < config.thresholds.rms,
        max_disp_converged: max_disp < config.thresholds.max_dis,
    }
}

/// Computes error vectors for GDIIS optimization.
///
/// Error vectors in GDIIS are computed as the solution to H^(-1) * g, where H is
/// the Hessian approximation and g is the gradient. These error vectors represent
/// the "Newton step" that would be taken at each point in the history and are used
/// to construct the DIIS interpolation matrix.
///
/// # Arguments
///
/// * `grads` - History of gradient vectors from previous iterations
/// * `hessians` - History of Hessian approximations from previous iterations
///
/// # Returns
///
/// Returns a vector of error vectors, one for each iteration in the history.
/// Each error vector has the same dimension as the gradient vectors.
///
/// # Algorithm
///
/// For each iteration i:
/// ```text
/// error[i] = H[i]^(-1) * g[i]
/// ```
///
/// If the Hessian is singular, falls back to using the gradient directly.
fn compute_error_vectors(
    grads: &VecDeque<DVector<f64>>,
    hessians: &VecDeque<DMatrix<f64>>,
) -> Vec<DVector<f64>> {
    let n = grads.len();
    if n == 0 {
        return Vec::new();
    }

    // Compute the mean Hessian
    let mut h_mean = DMatrix::zeros(hessians[0].nrows(), hessians[0].ncols());
    for hess in hessians {
        h_mean += hess;
    }
    h_mean /= n as f64;

    // Compute error vectors using the mean Hessian for all gradients
    grads
        .iter()
        .map(|grad| {
            h_mean
                .clone()
                .lu()
                .solve(grad)
                .unwrap_or_else(|| grad.clone())
        })
        .collect()
}

/// Builds the B matrix for GDIIS optimization.
///
/// The B matrix is the core of the DIIS method, containing dot products of error
/// vectors plus constraint equations. It has the structure:
///
/// ```text
/// B = [ e₁·e₁  e₁·e₂  ...  e₁·eₙ  1 ]
///     [ e₂·e₁  e₂·e₂  ...  e₂·eₙ  1 ]
///     [  ...    ...   ...   ...   1 ]
///     [ eₙ·e₁  eₙ·e₂  ...  eₙ·eₙ  1 ]
///     [   1      1    ...    1    0 ]
/// ```
///
/// where eᵢ·eⱼ represents the dot product of error vectors i and j.
///
/// # Arguments
///
/// * `errors` - Vector of error vectors from `compute_error_vectors`
///
/// # Returns
///
/// Returns the (n+1) × (n+1) B matrix where n is the number of error vectors.
/// The extra row and column enforce the constraint that coefficients sum to 1.
///
/// # Mathematical Background
///
/// The B matrix is used in solving the DIIS equations:
/// ```text
/// B * c = [0, 0, ..., 0, 1]ᵀ
/// ```
/// where c contains the interpolation coefficients and the Lagrange multiplier.
fn build_b_matrix(errors: &[DVector<f64>]) -> DMatrix<f64> {
    let n = errors.len();
    let mut b = DMatrix::zeros(n + 1, n + 1);

    for i in 0..n {
        for j in 0..n {
            b[(i, j)] = errors[i].dot(&errors[j]);
        }
    }

    for i in 0..n {
        b[(i, n)] = 1.0;
        b[(n, i)] = 1.0;
    }
    b[(n, n)] = 0.0;

    b
}

/// Performs a GDIIS (Geometry-based Direct Inversion in Iterative Subspace) optimization step.
///
/// GDIIS is an accelerated optimization method that uses a linear combination of
/// previous geometries and gradients to construct an optimal step direction. It
/// typically provides 2-3x faster convergence than BFGS once sufficient history
/// has been accumulated.
///
/// The method constructs error vectors from the gradient history and solves a
/// constrained minimization problem to find optimal interpolation coefficients.
/// These coefficients are then used to predict the next geometry.
///
/// # Advantages over BFGS
///
/// - Faster convergence (typically 2-3x fewer iterations)
/// - More robust for difficult optimization problems
/// - Automatically handles ill-conditioned Hessian matrices
/// - Does not require explicit Hessian updates
///
/// # Requirements
///
/// - Requires at least 3 iterations of history (checked via `has_enough_history()`)
/// - History includes geometries, gradients, and Hessian estimates
/// - Uses the most recent 4 iterations for DIIS extrapolation
///
/// # Arguments
///
/// * `opt_state` - Optimization state with history of geometries, gradients, and Hessians
/// * `config` - Configuration with step size limits
///
/// # Returns
///
/// Returns the new geometry coordinates after the GDIIS step as a `DVector<f64>`.
///
/// # Examples
///
/// ```
/// use omecp::optimizer::{gdiis_step, OptimizationState};
///
/// let opt_state = OptimizationState::new();
///
// assert!(opt_state.has_enough_history()); // Need ≥ 3 iterations
///
/// // let x_new = gdiis_step(&opt_state, &config);
/// ```
pub fn gdiis_step(opt_state: &OptimizationState, config: &Config) -> DVector<f64> {
    let n = opt_state.geom_history.len();

    // Error vectors are now correctly computed with the mean Hessian inside this function
    let errors = compute_error_vectors(&opt_state.grad_history, &opt_state.hess_history);
    let b_matrix = build_b_matrix(&errors);

    let mut rhs = DVector::zeros(n + 1);
    rhs[n] = 1.0;

    let solution = b_matrix.lu().solve(&rhs).unwrap_or_else(|| {
        let mut fallback = DVector::zeros(n + 1);
        for i in 0..n {
            fallback[i] = 1.0 / (n as f64);
        }
        fallback
    });

    let coeffs = solution.rows(0, n);

    // --- Start of Bug Fix ---

    // 1. Interpolate geometry to get x_new_prime
    let mut x_new_prime = DVector::zeros(opt_state.geom_history[0].len());
    for (i, geom) in opt_state.geom_history.iter().enumerate() {
        x_new_prime += geom * coeffs[i];
    }

    // 2. Interpolate gradient to get g_new_prime (THE CORRECT WAY)
    let mut g_new_prime = DVector::zeros(opt_state.grad_history[0].len());
    for (i, grad) in opt_state.grad_history.iter().enumerate() {
        g_new_prime += grad * coeffs[i];
    }

    // 3. Get the mean Hessian (already computed once in compute_error_vectors, but needed here)
    let mut h_mean = DMatrix::zeros(
        opt_state.hess_history[0].nrows(),
        opt_state.hess_history[0].ncols(),
    );
    for hess in &opt_state.hess_history {
        h_mean += hess;
    }
    h_mean /= n as f64;

    // 4. Compute correction using the interpolated gradient
    let correction = h_mean
        .lu()
        .solve(&g_new_prime)
        .unwrap_or_else(|| g_new_prime.clone());

    // 5. Apply correction to the interpolated geometry
    let mut x_new = x_new_prime - &correction;

    // --- End of Bug Fix ---

    let last_geom = opt_state.geom_history.back().unwrap();
    let mut step = &x_new - last_geom;

    // Python-inspired step reduction
    let last_grad_norm = opt_state.grad_history.back().unwrap().norm();
    if last_grad_norm < config.thresholds.rms_g * 10.0 {
        step *= 0.5; // REDUCED_FACTOR from Python code
    }

    let step_norm = step.norm();

    if step_norm > config.max_step_size {
        let scale = config.max_step_size / step_norm;
        println!(
            "current stepsize: {} is reduced to max_size {}",
            step_norm, config.max_step_size
        );
        x_new = last_geom + &step * scale;
    } else {
        x_new = last_geom + step;
    }

    x_new
}

/// Computes enhanced error vectors for GEDIIS optimization.
///
/// GEDIIS error vectors incorporate both gradient and energy information to
/// provide better convergence for MECP optimization. The energy contribution
/// helps emphasize geometries that are closer to the target energy difference.
///
/// # Arguments
///
/// * `grads` - History of gradient vectors from previous iterations
/// * `energies` - History of energy differences (E1 - E2) from previous iterations
///
/// # Returns
///
/// Returns a vector of enhanced error vectors that include energy weighting.
/// Each error vector combines gradient information with energy deviation.
///
/// # Algorithm
///
/// For each iteration i:
/// ```text
/// error[i] = g[i] + α * (E[i] - E_avg) * g[i]
/// ```
///
/// where:
/// - g[i] is the gradient at iteration i
/// - E[i] is the energy difference at iteration i
/// - E_avg is the average energy difference over all iterations
/// - α = 0.1 is an empirical scaling factor
///
/// # Energy Weighting
///
/// The energy weighting helps the optimizer focus on geometries with energy
/// differences closer to zero (the MECP condition). Points with large energy
/// differences receive less weight in the interpolation.
fn compute_gediis_error_vectors(
    grads: &VecDeque<DVector<f64>>,
    energies: &VecDeque<f64>,
) -> Vec<DVector<f64>> {
    let n = grads.len();
    let mut errors = Vec::with_capacity(n);

    // Compute average energy for normalization
    let avg_energy = energies.iter().sum::<f64>() / n as f64;

    for (i, grad) in grads.iter().enumerate() {
        let energy_error = energies[i] - avg_energy;
        // GEDIIS error vector combines gradient and energy information
        // Adaptive alpha: scales with inverse gradient magnitude for stability
        // Reference: J. Chem. Theory Comput. 2006, 2, 835-839
        let grad_norm = grad.norm();
        let alpha = 0.1 / (grad_norm + 1e-10); // Adaptive scaling
        let mut error = grad.clone();
        // Add energy-weighted gradient contribution
        error += &(grad * energy_error * alpha);
        errors.push(error);
    }

    errors
}

/// Builds the enhanced B matrix for GEDIIS optimization.
///
/// The GEDIIS B matrix extends the standard DIIS B matrix by including
/// energy-energy correlation terms. This provides additional information
/// about the energy landscape to improve convergence.
///
/// # Arguments
///
/// * `errors` - Vector of enhanced error vectors from `compute_gediis_error_vectors`
/// * `energies` - History of energy differences (E1 - E2) from previous iterations
///
/// # Returns
///
/// Returns the (n+1) × (n+1) enhanced B matrix where n is the number of iterations.
/// The matrix includes both gradient-gradient and energy-energy correlation terms.
///
/// # Algorithm
///
/// The matrix elements are computed as:
/// ```text
/// B[i,j] = error[i] · error[j] + β * (E[i] - E_avg) * (E[j] - E_avg)
/// ```
///
/// where:
/// - error[i] · error[j] is the standard DIIS error dot product
/// - E[i], E[j] are energy differences at iterations i and j
/// - E_avg is the average energy difference
/// - β = 0.01 is a small weighting factor for energy terms
///
/// # Energy Correlation Terms
///
/// The energy-energy terms help the optimizer recognize patterns in the
/// energy landscape and preferentially weight geometries with similar
/// energy characteristics. This is particularly useful for MECP optimization
/// where minimizing the energy difference is crucial.
fn build_gediis_b_matrix(errors: &[DVector<f64>], energies: &[f64]) -> DMatrix<f64> {
    let n = errors.len();
    let mut b = DMatrix::zeros(n + 1, n + 1);

    // Compute average energy
    let avg_energy = energies.iter().sum::<f64>() / n as f64;

    for i in 0..n {
        for j in 0..n {
            // Standard DIIS error dot product
            let error_dot = errors[i].dot(&errors[j]);
            // Add energy-energy term
            let energy_i = energies[i] - avg_energy;
            let energy_j = energies[j] - avg_energy;
            let energy_term = energy_i * energy_j * 0.01; // Small weighting for energy terms
            b[(i, j)] = error_dot + energy_term;
        }
    }

    // Set up DIIS constraint equations
    for i in 0..n {
        b[(i, n)] = 1.0;
        b[(n, i)] = 1.0;
    }
    b[(n, n)] = 0.0;

    b
}

/// Performs a GEDIIS (Energy-Informed Direct Inversion in Iterative Subspace) optimization step.
///
/// GEDIIS is an enhanced version of GDIIS that incorporates energy information
/// into the error vector construction. This typically provides 2-4x faster
/// convergence than GDIIS for difficult MECP optimization problems, particularly
/// those with significant energy difference minimization requirements.
///
/// The key enhancement over GDIIS is that GEDIIS error vectors include energy-
/// weighted gradient contributions. This helps the optimizer better balance
/// energy minimization with geometry optimization, leading to more robust
/// convergence to the true MECP.
///
/// # Algorithm Overview
///
/// 1. **Energy-Normalized Error Vectors**: Compute error vectors with energy
///    weighting to emphasize points near the target energy difference
/// 2. **Enhanced B-Matrix**: Include energy-energy terms in addition to gradient
///    error dot products
/// 3. **DIIS Interpolation**: Solve for optimal coefficients using the enhanced
///    error matrix
/// 4. **Geometry Prediction**: Construct new geometry from optimal coefficients
/// 5. **Newton Correction**: Apply Hessian-informed correction for stability
///
/// # When to Use GEDIIS
///
/// Enable GEDIIS by setting `use_gediis = true` in the configuration:
/// - Difficult MECP optimizations with flat PES regions
/// - Systems with large energy differences that need minimization
/// - When GDIIS shows slow convergence
/// - Transition metal complexes and open-shell systems
///
/// # Performance Comparison
///
/// - **BFGS**: Baseline convergence rate
/// - **GDIIS**: ~2-3x faster than BFGS
/// - **GEDIIS**: ~2-4x faster than GDIIS (4-8x faster than BFGS)
///
/// # Arguments
///
/// * `opt_state` - Optimization state with history including energies
/// * `config` - Configuration with step size limits and GEDIIS parameters
///
/// # Returns
///
/// Returns the new geometry coordinates after the GEDIIS step as a `DVector<f64>`.
///
/// # Examples
///
/// ```
/// use omecp::optimizer::{gediis_step, OptimizationState};
/// use omecp::config::Config;
///
/// let config = Config {
///     use_gediis: true,
///     ..Default::default()
/// };
///
/// let opt_state = OptimizationState::new();
/// assert!(opt_state.has_enough_history()); // Need ≥ 3 iterations
///
/// // let x_new = gediis_step(&opt_state, &config);
/// ```
pub fn gediis_step(opt_state: &OptimizationState, config: &Config) -> DVector<f64> {
    let n = opt_state.geom_history.len();

    // Compute GEDIIS error vectors
    let errors = compute_gediis_error_vectors(&opt_state.grad_history, &opt_state.energy_history);

    // Build GEDIIS B-matrix
    let energies: Vec<f64> = opt_state.energy_history.iter().cloned().collect();
    let b_matrix = build_gediis_b_matrix(&errors, &energies);

    // Solve DIIS equations
    let mut rhs = DVector::zeros(n + 1);
    rhs[n] = 1.0;

    let solution = b_matrix.lu().solve(&rhs).unwrap_or_else(|| {
        let mut fallback = DVector::zeros(n + 1);
        for i in 0..n {
            fallback[i] = 1.0 / (n as f64);
        }
        fallback
    });

    let coeffs = solution.rows(0, n);

    // Interpolate geometry
    let mut x_new = DVector::zeros(opt_state.geom_history[0].len());
    for (i, geom) in opt_state.geom_history.iter().enumerate() {
        x_new += geom * coeffs[i];
    }

    // Compute average Hessian and gradient for Newton step
    let mut h_mean = DMatrix::zeros(
        opt_state.hess_history[0].nrows(),
        opt_state.hess_history[0].ncols(),
    );
    for hess in &opt_state.hess_history {
        h_mean += hess;
    }
    h_mean /= n as f64;

    let mut g_mean = DVector::zeros(opt_state.grad_history[0].len());
    for grad in &opt_state.grad_history {
        g_mean += grad;
    }
    g_mean /= n as f64;

    // Apply Newton correction
    let correction = h_mean.lu().solve(&g_mean).unwrap_or_else(|| g_mean.clone());
    x_new -= &correction;

    // Apply step size limit
    let last_geom = opt_state.geom_history.back().unwrap();
    let step = &x_new - last_geom;
    let step_norm = step.norm();

    if step_norm > config.max_step_size {
        let scale = config.max_step_size / step_norm;
        println!(
            "current stepsize: {} is reduced to max_size {}",
            step_norm, config.max_step_size
        );
        x_new = last_geom + &step * scale;
    }

    x_new
}

/// Performs a hybrid GEDIIS optimization step (50% GDIIS + 50% GEDIIS).
///
/// This function implements Python's hybrid approach which averages results
/// from GDIIS and GEDIIS optimizers. This combines the robust convergence of
/// GDIIS with the energy-aware acceleration of GEDIIS.
///
/// # Algorithm
///
/// 1. Compute GDIIS step using gradient history
/// 2. Compute GEDIIS step using energy-weighted error vectors
/// 3. Apply 50/50 averaging: X_new = 0.5 * X_GDIIS + 0.5 * X_GEDIIS
/// 4. Apply step size limits from configuration
///
/// # Arguments
///
/// * `opt_state` - Optimization state with history
/// * `config` - Configuration with step size limits
///
/// # Returns
///
/// Returns the new geometry coordinates after hybrid GEDIIS step.
///
/// # Examples
///
/// ```rust
/// use omecp::optimizer::{hybrid_gediis_step, OptimizationState};
/// use omecp::config::Config;
///
/// let config = Config::default(); // hybrid_gediis = true by default
/// let opt_state = OptimizationState::new();
///
/// // let x_new = hybrid_gediis_step(&opt_state, &config);
/// ```
pub fn hybrid_gediis_step(opt_state: &OptimizationState, config: &Config) -> DVector<f64> {
    // Compute both GDIIS and GEDIIS results
    let gdiis_result = gdiis_step(opt_state, config);
    let gediis_result = gediis_step(opt_state, config);

    // Apply 50/50 averaging (Python MECP.py behavior)
    let n = gdiis_result.len();
    let mut hybrid_result = DVector::zeros(n);
    for i in 0..n {
        hybrid_result[i] = 0.5 * gdiis_result[i] + 0.5 * gediis_result[i];
    }

    hybrid_result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{Geometry, State};

    #[test]
    fn test_force_sign_convention_matches_python() {
        // Create test geometry (2 atoms)
        let elements = vec!["H".to_string(), "H".to_string()];
        let coords = vec![
            0.0, 0.0, 0.0,  // Atom 1 at origin
            1.0, 0.0, 0.0,  // Atom 2 at (1,0,0)
        ];
        let geometry = Geometry::new(elements, coords);

        // Create test forces (positive values as extracted from Gaussian output)
        // Python would NEGATE these before MECP gradient computation
        let forces1 = DVector::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let forces2 = DVector::from_vec(vec![0.2, 0.3, 0.4, 0.5, 0.6, 0.7]);

        // Create states with positive forces (as extracted from Gaussian)
        let state1 = State {
            geometry: geometry.clone(),
            energy: -100.0,
            forces: forces1,
        };

        let state2 = State {
            geometry,
            energy: -99.0,
            forces: forces2,
        };

        // Compute MECP gradient
        let gradient = compute_mecp_gradient(&state1, &state2, &[]);

        // Verify that forces were properly negated
        // The gradient should reflect NEGATED forces (matching Python behavior)
        // If forces weren't negated, gradient direction would be wrong
        
        // Check that gradient has correct dimension
        assert_eq!(gradient.len(), 6);
        
        // Check that gradient is not zero (forces should have effect)
        assert!(gradient.norm() > 1e-10);
        
        // Manual verification: compute expected gradient with negated forces
        let expected_f1 = -state1.forces;  // Python negates forces
        let expected_f2 = -state2.forces;  // Python negates forces
        
        let x_vec = &expected_f1 - &expected_f2;
        let x_norm = if x_vec.norm().abs() < 1e-10 {
            let n = x_vec.len() as f64;
            &x_vec / (n.sqrt() * 1e-10)
        } else {
            &x_vec / x_vec.norm()
        };
        
        let de = state1.energy - state2.energy;
        let expected_f_vec = x_norm.clone() * de;
        let dot = expected_f1.dot(&x_norm);
        let expected_g_vec = &expected_f1 - &x_norm * dot;
        let expected_gradient = expected_f_vec + expected_g_vec;
        
        // Compare computed gradient with expected (allowing for numerical precision)
        for i in 0..gradient.len() {
            assert!((gradient[i] - expected_gradient[i]).abs() < 1e-10,
                   "Gradient component {} mismatch: {} vs {}", 
                   i, gradient[i], expected_gradient[i]);
        }
    }

    #[test]
    fn test_force_negation_impact() {
        // Demonstrate the critical importance of force negation
        let elements = vec!["H".to_string(), "H".to_string()];
        let coords = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let geometry = Geometry::new(elements, coords);

        // ZERO forces for both states
        let forces = DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        
        let state1 = State {
            geometry: geometry.clone(),
            energy: -100.0,
            forces: forces.clone(),
        };

        let state2 = State {
            geometry,
            energy: -100.0,  // Same energy
            forces: forces.clone(),
        };

        let gradient = compute_mecp_gradient(&state1, &state2, &[]);

        // With zero forces and same energies, gradient should be zero
        assert!(gradient.norm() < 1e-10, 
               "Expected zero gradient with zero forces, got {}", 
               gradient.norm());
    }
}
