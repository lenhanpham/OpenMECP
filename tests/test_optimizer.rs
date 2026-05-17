//! Unit tests for the optimizer module
//!
//! This file contains tests for the optimization algorithms used in OpenMECP,
//! including Hessian updates, convergence checking, and optimization steps.

use nalgebra::{DMatrix, DVector};
use omecp::optimizer::update_hessian;

/// Tests for the BFGS inverse Hessian update function.
///
/// The update_hessian function implements the BFGS formula for updating
/// the inverse Hessian matrix. It expects:
/// - h_inv: Current inverse Hessian in Bohr²/Ha
/// - sk: Step vector in Bohr
/// - yk: Gradient difference in Ha/Bohr

#[test]
fn test_update_hessian_basic() {
    // Test basic inverse Hessian update with simple vectors
    // Using inverse Hessian (not Hessian) - initialized to 0.7 diagonal
    let h_inv = DMatrix::from_diagonal(&DVector::from_vec(vec![0.7, 0.7, 0.7]));
    let sk = DVector::from_vec(vec![0.1, 0.2, 0.3]); // Step in Bohr
    let yk = DVector::from_vec(vec![0.05, 0.1, 0.15]); // Gradient diff in Ha/Bohr

    let h_new = update_hessian(&h_inv, &sk, &yk);

    // Check that the result is a valid matrix
    assert_eq!(h_new.nrows(), 3);
    assert_eq!(h_new.ncols(), 3);

    // Check that the update preserves symmetry (BFGS update should maintain symmetry)
    let h_transpose = h_new.transpose();
    assert!((h_new.clone() - h_transpose).norm() < 1e-10);
}

#[test]
fn test_update_hessian_zero_step() {
    // Test that zero step vector doesn't cause issues
    let h_inv = DMatrix::from_diagonal(&DVector::from_vec(vec![0.7, 0.7, 0.7]));
    let sk = DVector::from_vec(vec![0.0, 0.0, 0.0]);
    let yk = DVector::from_vec(vec![0.1, 0.2, 0.3]);

    let h_new = update_hessian(&h_inv, &sk, &yk);

    // Should return original inverse Hessian unchanged (denominators too small)
    assert_eq!(h_new, h_inv);
}

#[test]
fn test_update_hessian_negative_curvature() {
    // Test negative curvature condition
    // BFGS update may still proceed with negative curvature (unlike PSB)
    let h_inv = DMatrix::from_diagonal(&DVector::from_vec(vec![0.7, 0.7, 0.7]));
    let sk = DVector::from_vec(vec![0.1, 0.2, 0.3]);
    let yk = DVector::from_vec(vec![-0.1, -0.2, -0.3]); // Negative dot product with sk

    let h_new = update_hessian(&h_inv, &sk, &yk);

    // BFGS may skip update when fac_denom is negative (unstable)
    // Just verify it returns a valid matrix
    assert_eq!(h_new.nrows(), 3);
    assert_eq!(h_new.ncols(), 3);
}

#[test]
fn test_update_hessian_symmetry() {
    // Test that the update maintains matrix symmetry
    let h_inv = DMatrix::from_row_slice(3, 3, &[0.7, 0.1, 0.05, 0.1, 0.7, 0.08, 0.05, 0.08, 0.7]);
    let sk = DVector::from_vec(vec![0.1, -0.2, 0.15]);
    let yk = DVector::from_vec(vec![0.08, -0.12, 0.18]);

    let h_new = update_hessian(&h_inv, &sk, &yk);

    // Check symmetry preservation
    let h_transpose = h_new.transpose();
    assert!((h_new.clone() - h_transpose).norm() < 1e-12);

    // Check that update is reasonable (not too large)
    let update_norm = (&h_new - &h_inv).norm();
    assert!(update_norm < 10.0); // Reasonable update magnitude
}

#[test]
fn test_update_hessian_large_system() {
    // Test with a larger system (6x6 matrix) to ensure scalability
    let h_inv = DMatrix::from_diagonal(&DVector::from_vec(vec![0.7, 0.7, 0.7, 0.7, 0.7, 0.7]));
    let sk = DVector::from_vec(vec![0.1, 0.2, 0.3, 0.1, 0.2, 0.3]);
    let yk = DVector::from_vec(vec![0.05, 0.1, 0.15, 0.05, 0.1, 0.15]);

    let h_new = update_hessian(&h_inv, &sk, &yk);

    // Check that the result is a valid matrix
    assert_eq!(h_new.nrows(), 6);
    assert_eq!(h_new.ncols(), 6);

    // Check symmetry preservation
    let h_transpose = h_new.transpose();
    assert!((h_new.clone() - h_transpose).norm() < 1e-10);
}

#[test]
fn test_update_hessian_edge_case_small_values() {
    // Test with very small values to check numerical stability
    let h_inv = DMatrix::from_diagonal(&DVector::from_vec(vec![0.7, 0.7, 0.7]));
    let sk = DVector::from_vec(vec![1e-8, 2e-8, 3e-8]);
    let yk = DVector::from_vec(vec![5e-9, 1e-8, 1.5e-8]);

    let h_new = update_hessian(&h_inv, &sk, &yk);

    // Should handle small values gracefully (may skip update due to small denominators)
    assert_eq!(h_new.nrows(), 3);
    assert_eq!(h_new.ncols(), 3);

    // Check symmetry preservation even with small values
    let h_transpose = h_new.transpose();
    assert!((h_new.clone() - h_transpose).norm() < 1e-15);
}

#[test]
fn test_sequential_hybrid_gediis_step() {
    use omecp::config::Config;
    use omecp::optimizer::OptimizationState;
    use omecp::optimizer::sequential_hybrid_gediis_step;

    let mut config = Config::default();
    config.max_history = 5;
    config.max_step_size = 0.3;

    // Create minimal optimization state with 3-atom coordinates (9 coords)
    let ncoords = 9;
    let mut opt_state = OptimizationState::new(config.max_history);
    let geom = DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
    let grad = DVector::from_vec(vec![0.1; ncoords]);
    let f_vec = DVector::from_vec(vec![0.05; ncoords]);
    let hess = DMatrix::identity(ncoords, ncoords) * 0.7;
    for i in 0..3 {
        let offset = i as f64 * 0.1;
        let g = DVector::from_vec(vec![0.1 - offset; ncoords]);
        let fv = DVector::from_vec(vec![0.05; ncoords]);
        let h = DMatrix::identity(ncoords, ncoords) * (0.7 + offset);
        opt_state.add_to_history(
            DVector::from_vec(vec![offset; ncoords]),
            g.clone(),
            fv.clone(),
            h,
            0.0,
            vec![0.0; ncoords],
            Some(1.0),
            false,
        );
    }

    let result = sequential_hybrid_gediis_step(&mut opt_state, &config);
    assert_eq!(result.len(), ncoords);
    assert!(result.norm() > 0.0);
}

#[test]
fn test_hybrid_gediis_configuration() {
    use omecp::config::Config;

    let config = Config::default();
    assert!(!config.use_hybrid_gediis); // GDIIS is the default
}
