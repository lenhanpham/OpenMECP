use crate::config::Config;
use crate::constraints::{build_jacobian, evaluate_constraints, Constraint};
use crate::geometry::State;
use nalgebra::{DMatrix, DVector};
use std::collections::VecDeque;

#[derive(Debug, Clone, Default)]
pub struct OptimizationState {
    pub lambdas: Vec<f64>,
    pub lambda_de: Option<f64>,
    pub geom_history: VecDeque<DVector<f64>>,
    pub grad_history: VecDeque<DVector<f64>>,
    pub hess_history: VecDeque<DMatrix<f64>>,
    pub energy_history: VecDeque<f64>,
    pub max_history: usize,
}

impl OptimizationState {
    pub fn new() -> Self {
        Self {
            lambdas: Vec::new(),
            lambda_de: None,
            geom_history: VecDeque::with_capacity(4),
            grad_history: VecDeque::with_capacity(4),
            hess_history: VecDeque::with_capacity(4),
            energy_history: VecDeque::with_capacity(4),
            max_history: 4,
        }
    }
    
    pub fn add_to_history(&mut self, geom: DVector<f64>, grad: DVector<f64>, hess: DMatrix<f64>, energy: f64) {
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
    
    pub fn has_enough_history(&self) -> bool {
        self.geom_history.len() >= 3
    }
}

/// Compute MECP effective gradient
pub fn compute_mecp_gradient(
    state1: &State,
    state2: &State,
    constraints: &[Constraint],
    opt_state: &mut OptimizationState,
    fixed_atoms: &[usize],
) -> DVector<f64> {
    let f1 = &state1.forces;
    let f2 = &state2.forces;
    
    // Gradient difference
    let x_vec = f1 - f2;
    let x_norm_val = x_vec.norm();
    let x_norm = &x_vec / x_norm_val;
    
    // Energy difference component
    let de = state1.energy - state2.energy;
    let f_vec = &x_norm * de;
    
    // Perpendicular component
    let dot = f1.dot(&x_norm);
    let g_vec = f1 - &x_norm * dot;
    
    // Combine
    let mut eff_grad = f_vec + g_vec;
    
    // Apply fixed atoms
    for &atom_idx in fixed_atoms {
        eff_grad[atom_idx * 3] = 0.0;
        eff_grad[atom_idx * 3 + 1] = 0.0;
        eff_grad[atom_idx * 3 + 2] = 0.0;
    }
    
    // Apply constraints (simplified - full Lagrange multiplier method would go here)
    if !constraints.is_empty() {
        apply_constraints(&mut eff_grad, &state1.geometry.coords, constraints, opt_state);
    }
    
    eff_grad
}

/// Compute MECP gradient with energy difference constraint (fix-dE optimization)
pub fn compute_mecp_gradient_with_de_constraint(
    state1: &State,
    state2: &State,
    constraints: &[Constraint],
    opt_state: &mut OptimizationState,
    fixed_atoms: &[usize],
    target_de: f64,
) -> DVector<f64> {
    // First compute the standard MECP gradient
    let mut eff_grad = compute_mecp_gradient(state1, state2, constraints, opt_state, fixed_atoms);

    // Add energy difference constraint
    // The constraint is: E1 - E2 = target_de
    // Gradient contribution: λ * (dE1/dx - dE2/dx)
    // where λ is determined to maintain the energy difference

    let current_de = state1.energy - state2.energy;
    let de_error = current_de - target_de;

    // Simple proportional control for the Lagrange multiplier
    let lambda_de = if let Some(prev_lambda) = opt_state.lambda_de {
        prev_lambda + 0.1 * de_error  // Adjust lambda based on energy difference error
    } else {
        0.1 * de_error
    };

    opt_state.lambda_de = Some(lambda_de);

    // Add the energy difference constraint gradient
    // d(E1 - E2)/dx = dE1/dx - dE2/dx
    for i in 0..eff_grad.len() {
        eff_grad[i] += lambda_de * (state1.forces[i] - state2.forces[i]);
    }

    // Apply geometric constraints if any
    if !constraints.is_empty() {
        apply_constraints(&mut eff_grad, &state1.geometry.coords, constraints, opt_state);
    }

    eff_grad
}

fn apply_constraints(
    grad: &mut DVector<f64>,
    coords: &DVector<f64>,
    constraints: &[Constraint],
    opt_state: &mut OptimizationState,
) {
    if constraints.is_empty() {
        return;
    }

    if opt_state.lambdas.is_empty() {
        opt_state.lambdas = vec![0.0; constraints.len()];
    }

    let jacobian = build_jacobian(coords, constraints);
    let violations = evaluate_constraints(coords, constraints);
    
    // Update multipliers: λ[i] += 0.5 * violation[i]
    for (i, &violation) in violations.iter().enumerate() {
        opt_state.lambdas[i] += 0.5 * violation;
    }
    
    // Apply constraint forces: g' = g + C^T * λ
    for (i, &lambda) in opt_state.lambdas.iter().enumerate() {
        for j in 0..grad.len() {
            grad[j] += lambda * jacobian[(i, j)];
        }
    }
}

/// BFGS step
pub fn bfgs_step(
    x0: &DVector<f64>,
    g0: &DVector<f64>,
    hessian: &DMatrix<f64>,
    config: &Config,
) -> DVector<f64> {
    // Solve H * dk = -g
    let neg_g = -g0;
    let dk = hessian.clone().lu().solve(&neg_g).unwrap_or_else(|| -g0.clone());
    
    // Apply step size limit
    let mut x_new = x0 + &dk;
    let step_norm = dk.norm();
    
    if step_norm > config.max_step_size {
        let scale = config.max_step_size / step_norm;
        x_new = x0 + &dk * scale;
    }
    
    x_new
}

/// Update Hessian using PSB formula
pub fn update_hessian_psb(
    hessian: &DMatrix<f64>,
    sk: &DVector<f64>,
    yk: &DVector<f64>,
) -> DMatrix<f64> {
    let mut h_new = hessian.clone();
    
    let hsk = hessian * sk;
    let diff = yk - &hsk;
    let sk_dot_sk = sk.dot(sk);
    
    if sk_dot_sk.abs() > 1e-10 {
        let sk_diff = sk.dot(&diff);
        let term1 = &diff * sk.transpose() + sk * diff.transpose();
        let term2 = (sk * sk.transpose()) * (sk_diff / sk_dot_sk);
        
        h_new += (term1 - term2) / sk_dot_sk;
    }
    
    h_new
}

#[derive(Debug)]
pub struct ConvergenceStatus {
    pub de_converged: bool,
    pub rms_grad_converged: bool,
    pub max_grad_converged: bool,
    pub rms_disp_converged: bool,
    pub max_disp_converged: bool,
}

impl ConvergenceStatus {
    pub fn is_converged(&self) -> bool {
        self.de_converged
            && self.rms_grad_converged
            && self.max_grad_converged
            && self.rms_disp_converged
            && self.max_disp_converged
    }
}

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

/// Compute error vectors for GDIIS
fn compute_error_vectors(
    grads: &VecDeque<DVector<f64>>,
    hessians: &VecDeque<DMatrix<f64>>,
) -> Vec<DVector<f64>> {
    grads.iter()
        .zip(hessians.iter())
        .map(|(grad, hess)| {
            hess.clone().lu().solve(grad).unwrap_or_else(|| grad.clone())
        })
        .collect()
}

/// Build B matrix for GDIIS
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

/// GDIIS optimization step
pub fn gdiis_step(opt_state: &OptimizationState, config: &Config) -> DVector<f64> {
    let n = opt_state.geom_history.len();
    
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
    
    let mut h_mean = DMatrix::zeros(
        opt_state.hess_history[0].nrows(),
        opt_state.hess_history[0].ncols(),
    );
    for hess in &opt_state.hess_history {
        h_mean += hess;
    }
    h_mean /= n as f64;
    
    let mut x_new = DVector::zeros(opt_state.geom_history[0].len());
    for (i, geom) in opt_state.geom_history.iter().enumerate() {
        x_new += geom * coeffs[i];
    }
    
    let mut g_mean = DVector::zeros(opt_state.grad_history[0].len());
    for grad in &opt_state.grad_history {
        g_mean += grad;
    }
    g_mean /= n as f64;
    
    let correction = h_mean.lu().solve(&g_mean).unwrap_or_else(|| g_mean.clone());
    x_new -= &correction;
    
    let last_geom = opt_state.geom_history.back().unwrap();
    let step = &x_new - last_geom;
    let step_norm = step.norm();
    
    if step_norm > config.max_step_size {
        let scale = config.max_step_size / step_norm;
        x_new = last_geom + &step * scale;
    }

    x_new
}

/// Compute GEDIIS error vectors (include energy information)
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
        // The energy contribution is scaled to be comparable to gradient magnitudes
        let energy_scale = 0.1; // Empirical scaling factor
        let mut error = grad.clone();
        // Add energy-weighted gradient contribution
        error += &(grad * energy_error * energy_scale);
        errors.push(error);
    }

    errors
}

/// Build B matrix for GEDIIS (includes energy-energy terms)
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

/// GEDIIS optimization step
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
        x_new = last_geom + &step * scale;
    }

    x_new
}
