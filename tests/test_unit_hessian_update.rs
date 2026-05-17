use nalgebra::{DMatrix, DVector};
use omecp::hessian_update::{
    update_hessian_bfgs, update_hessian_bofill, update_hessian_powell,
    update_inverse_hessian_bfgs,
};

fn approx_eq(a: f64, b: f64, epsilon: f64) -> bool {
    (a - b).abs() < epsilon
}

#[test]
fn test_bfgs_update_basic() {
    let h = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1.0, 1.0]));
    let dx = DVector::from_vec(vec![0.1, 0.2, 0.3]);
    let dg = DVector::from_vec(vec![0.15, 0.25, 0.35]);

    let h_new = update_hessian_bfgs(&h, &dx, &dg);

    assert!(approx_eq((h_new.clone() - h_new.transpose()).norm(), 0.0, 1e-12));
    assert_eq!(h_new.nrows(), 3);
    assert_eq!(h_new.ncols(), 3);
}

#[test]
fn test_bfgs_negative_curvature_skipped() {
    let h = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1.0, 1.0]));
    let dx = DVector::from_vec(vec![0.1, 0.2, 0.3]);
    let dg = DVector::from_vec(vec![-0.1, -0.2, -0.3]); // Negative curvature

    let h_new = update_hessian_bfgs(&h, &dx, &dg);

    assert!(approx_eq((h_new - h).norm(), 0.0, 1e-12));
}

#[test]
fn test_powell_update_basic() {
    let h = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1.0, 1.0]));
    let dx = DVector::from_vec(vec![0.1, 0.2, 0.3]);
    let dg = DVector::from_vec(vec![0.15, 0.25, 0.35]);

    let h_new = update_hessian_powell(&h, &dx, &dg);

    assert!(approx_eq((h_new.clone() - h_new.transpose()).norm(), 0.0, 1e-12));
}

#[test]
fn test_bofill_update_basic() {
    let h = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1.0, 1.0]));
    let dx = DVector::from_vec(vec![0.1, 0.2, 0.3]);
    let dg = DVector::from_vec(vec![0.15, 0.25, 0.35]);

    let h_new = update_hessian_bofill(&h, &dx, &dg);

    assert!(approx_eq((h_new.clone() - h_new.transpose()).norm(), 0.0, 1e-12));
}

#[test]
fn test_inverse_hessian_bfgs() {
    let h_inv = DMatrix::from_diagonal(&DVector::from_vec(vec![0.7, 0.7, 0.7]));
    let dx = DVector::from_vec(vec![0.1, 0.2, 0.3]);
    let dg = DVector::from_vec(vec![0.05, 0.1, 0.15]);

    let h_inv_new = update_inverse_hessian_bfgs(&h_inv, &dx, &dg);

    assert!(approx_eq(
        (h_inv_new.clone() - h_inv_new.transpose()).norm(),
        0.0,
        1e-12
    ));
}

#[test]
fn test_zero_step_handled() {
    let h = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1.0, 1.0]));
    let dx = DVector::from_vec(vec![0.0, 0.0, 0.0]);
    let dg = DVector::from_vec(vec![0.1, 0.2, 0.3]);

    let h_bfgs = update_hessian_bfgs(&h, &dx, &dg);
    let h_powell = update_hessian_powell(&h, &dx, &dg);
    let h_bofill = update_hessian_bofill(&h, &dx, &dg);

    assert_eq!(h_bfgs, h);
    assert_eq!(h_powell, h);
    assert_eq!(h_bofill, h);
}
