//! Hessian update methods for optimization algorithms.
//!
//! This module implements various Hessian and inverse Hessian update formulas
//!
//! # Available Update Methods
//!
//! - **BFGS**: Broyden-Fletcher-Goldfarb-Shanno for minima
//! - **Bofill**: Weighted Powell/Murtagh-Sargent for saddle points
//! - **Powell**: Symmetric rank-one update
//! - **PSB**: Powell-Symmetric-Broyden (legacy)
//!
//! # References
//!
//! - Bofill, J. M. J. Comput. Chem. 1994, 15, 1-11.
//! - Powell, M. J. D. Math. Programming 1971, 1, 26-57.
//! - Murtagh, B. A.; Sargent, R. W. H. Comput. J. 1970, 13, 185-194.

use nalgebra::{DMatrix, DVector};

/// Hessian update method selection.
///
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HessianUpdateMethod {
    /// BFGS update for minima (MthUpd=3)
    #[default]
    Bfgs,
    /// Bofill weighted update for saddle points (MthUpd=4)
    Bofill,
    /// Pure BFGS without curvature check (MthUpd=5)
    BfgsPure,
    /// Powell symmetric rank-one update (MthUpd=6)
    Powell,
    /// BFGS/Powell mixture following Bofill (MthUpd=7)
    BfgsPowellMix,
}


/// Numerical thresholds for Hessian updates.
const SMALL: f64 = 1e-14;
const RMIN2: f64 = 1e-12;

/// Updates the Hessian matrix using the specified method.
///
/// This is the main entry point for Hessian updates, dispatching to the
/// appropriate algorithm based on the selected method.
///
/// # Arguments
///
/// * `hessian` - Current Hessian matrix (Ha/Bohr²)
/// * `delta_x` - Step vector (x_new - x_old) in Bohr
/// * `delta_g` - Gradient difference (g_new - g_old) in Ha/Bohr
/// * `method` - Update method to use
///
/// # Returns
///
/// Updated Hessian matrix in Ha/Bohr².
pub fn update_hessian_with_method(
    hessian: &DMatrix<f64>,
    delta_x: &DVector<f64>,
    delta_g: &DVector<f64>,
    method: HessianUpdateMethod,
) -> DMatrix<f64> {
    match method {
        HessianUpdateMethod::Bfgs => update_hessian_bfgs(hessian, delta_x, delta_g),
        HessianUpdateMethod::Bofill => update_hessian_bofill(hessian, delta_x, delta_g),
        HessianUpdateMethod::BfgsPure => update_hessian_bfgs_pure(hessian, delta_x, delta_g),
        HessianUpdateMethod::Powell => update_hessian_powell(hessian, delta_x, delta_g),
        HessianUpdateMethod::BfgsPowellMix => update_hessian_bfgs_powell_mix(hessian, delta_x, delta_g),
    }
}

/// BFGS Hessian update for minima (MthUpd=3).
///
/// Implements the standard BFGS formula from Old Code D2CorX:
/// ```text
/// H_new = H + (Δg·Δg^T)/(Δx·Δg) - (H·Δx·Δx^T·H)/(Δx^T·H·Δx)
/// ```
///
/// # Curvature Condition
///
/// The update is only applied if Δx·Δg > 0 (positive curvature).
/// This ensures the updated Hessian remains positive definite for minima.
pub fn update_hessian_bfgs(
    hessian: &DMatrix<f64>,
    delta_x: &DVector<f64>,
    delta_g: &DVector<f64>,
) -> DMatrix<f64> {
    let mut h_new = hessian.clone();
    
    // Check for valid inputs
    if !delta_x.iter().all(|v| v.is_finite()) || !delta_g.iter().all(|v| v.is_finite()) {
        return h_new;
    }
    
    let dx_dg = delta_x.dot(delta_g);  // Δx·Δg
    
    // Curvature condition: skip if not positive
    if dx_dg <= SMALL {
        return h_new;
    }
    
    // Compute H·Δx
    let h_dx = hessian * delta_x;
    let dx_h_dx = delta_x.dot(&h_dx);  // Δx^T·H·Δx
    
    if dx_h_dx.abs() <= SMALL {
        return h_new;
    }
    
    // BFGS update: H + (Δg·Δg^T)/(Δx·Δg) - (H·Δx)(H·Δx)^T/(Δx^T·H·Δx)
    let n = hessian.nrows();
    for i in 0..n {
        for j in 0..=i {
            let update = delta_g[i] * delta_g[j] / dx_dg 
                       - h_dx[i] * h_dx[j] / dx_h_dx;
            h_new[(i, j)] += update;
            if i != j {
                h_new[(j, i)] += update;
            }
        }
    }
    
    h_new
}

/// Pure BFGS update without curvature check (MthUpd=5).
///
/// Same as BFGS but proceeds even with negative curvature.
/// Use with caution - may produce indefinite Hessian.
pub fn update_hessian_bfgs_pure(
    hessian: &DMatrix<f64>,
    delta_x: &DVector<f64>,
    delta_g: &DVector<f64>,
) -> DMatrix<f64> {
    let mut h_new = hessian.clone();
    
    if !delta_x.iter().all(|v| v.is_finite()) || !delta_g.iter().all(|v| v.is_finite()) {
        return h_new;
    }
    
    let dx_dg = delta_x.dot(delta_g);
    
    if dx_dg.abs() <= SMALL {
        return h_new;
    }
    
    let h_dx = hessian * delta_x;
    let dx_h_dx = delta_x.dot(&h_dx);
    
    if dx_h_dx.abs() <= SMALL {
        return h_new;
    }
    
    let n = hessian.nrows();
    for i in 0..n {
        for j in 0..=i {
            let update = delta_g[i] * delta_g[j] / dx_dg 
                       - h_dx[i] * h_dx[j] / dx_h_dx;
            h_new[(i, j)] += update;
            if i != j {
                h_new[(j, i)] += update;
            }
        }
    }
    
    h_new
}

/// Powell symmetric rank-one update.
///
/// Implements the Powell/SR1 formula:
/// ```text
/// H_new = H + (Δg - H·Δx)(Δg - H·Δx)^T / [(Δg - H·Δx)·Δx]
/// ```
///
/// This update can handle negative curvature, making it suitable
/// for transition state searches.
pub fn update_hessian_powell(
    hessian: &DMatrix<f64>,
    delta_x: &DVector<f64>,
    delta_g: &DVector<f64>,
) -> DMatrix<f64> {
    let mut h_new = hessian.clone();
    
    if !delta_x.iter().all(|v| v.is_finite()) || !delta_g.iter().all(|v| v.is_finite()) {
        return h_new;
    }
    
    let dx_norm_sq = delta_x.norm_squared();
    if dx_norm_sq < RMIN2 {
        return h_new;
    }
    
    // Compute Δg - H·Δx
    let h_dx = hessian * delta_x;
    let diff = delta_g - &h_dx;
    
    // Denominator: (Δg - H·Δx)·Δx
    let denom = diff.dot(delta_x);
    
    if denom.abs() <= SMALL {
        return h_new;
    }
    
    // Powell/SR1 update
    let n = hessian.nrows();
    for i in 0..n {
        for j in 0..=i {
            let update = diff[i] * diff[j] / denom;
            h_new[(i, j)] += update;
            if i != j {
                h_new[(j, i)] += update;
            }
        }
    }
    
    h_new
}

/// Bofill weighted update for saddle points.
///
/// Implements Bofill's formula from J. Comput. Chem. 1994, 15, 1-11:
/// ```text
/// H_new = H + φ·Powell_term + (1-φ)·MS_term
/// ```
///
/// where:
/// - φ = 1 - (Δx·Δg - Δx·H·Δx)² / (|Δx|² · |Δg - H·Δx|²)
/// - Powell_term = symmetric rank-one update
/// - MS_term = Murtagh-Sargent update
///
/// This weighted combination provides good convergence for both
/// minima and saddle points.
pub fn update_hessian_bofill(
    hessian: &DMatrix<f64>,
    delta_x: &DVector<f64>,
    delta_g: &DVector<f64>,
) -> DMatrix<f64> {
    let mut h_new = hessian.clone();
    
    if !delta_x.iter().all(|v| v.is_finite()) || !delta_g.iter().all(|v| v.is_finite()) {
        return h_new;
    }
    
    let dx_norm_sq = delta_x.norm_squared();
    if dx_norm_sq < RMIN2 {
        return h_new;
    }
    
    // Compute intermediate quantities
    let h_dx = hessian * delta_x;
    let diff = delta_g - &h_dx;  // Δg - H·Δx
    let diff_norm_sq = diff.norm_squared();
    
    if diff_norm_sq < SMALL {
        return h_new;
    }
    
    let dx_dg = delta_x.dot(delta_g);
    let dx_h_dx = delta_x.dot(&h_dx);
    
    // Compute Bofill weight φ
    let r_num = dx_dg - dx_h_dx;
    let r_denom = dx_norm_sq * diff_norm_sq;
    
    let phi = if r_num.abs() > SMALL && r_denom.abs() > SMALL {
        1.0 - (r_num * r_num) / r_denom
    } else {
        1.0  // Default to pure Powell if denominators are small
    };
    
    // Apply Bofill update: φ·Powell + (1-φ)·MS
    let n = hessian.nrows();
    for i in 0..n {
        for j in 0..=i {
            // Powell term: (Δg - H·Δx)_i · Δx_j + Δx_i · (Δg - H·Δx)_j
            //            - (Δx·Δg - Δx·H·Δx) · Δx_i · Δx_j / |Δx|²
            let powell = (diff[i] * delta_x[j] + delta_x[i] * diff[j]) / dx_norm_sq
                       - r_num * delta_x[i] * delta_x[j] / (dx_norm_sq * dx_norm_sq);
            
            // Murtagh-Sargent term: (Δg - H·Δx)_i · (Δg - H·Δx)_j / (Δx·Δg - Δx·H·Δx)
            let ms = if r_num.abs() > SMALL {
                diff[i] * diff[j] / r_num
            } else {
                0.0
            };
            
            let update = (1.0 - phi) * ms + phi * powell;
            h_new[(i, j)] += update;
            if i != j {
                h_new[(j, i)] += update;
            }
        }
    }
    
    h_new
}

/// BFGS/Powell mixture following Bofill weighting.
///
/// Uses Bofill's φ parameter to blend BFGS and Powell updates.
/// Provides smooth transition between methods based on local curvature.
pub fn update_hessian_bfgs_powell_mix(
    hessian: &DMatrix<f64>,
    delta_x: &DVector<f64>,
    delta_g: &DVector<f64>,
) -> DMatrix<f64> {
    let mut h_new = hessian.clone();
    
    if !delta_x.iter().all(|v| v.is_finite()) || !delta_g.iter().all(|v| v.is_finite()) {
        return h_new;
    }
    
    let dx_norm_sq = delta_x.norm_squared();
    if dx_norm_sq < RMIN2 {
        return h_new;
    }
    
    let h_dx = hessian * delta_x;
    let diff = delta_g - &h_dx;
    let diff_norm_sq = diff.norm_squared();
    
    if diff_norm_sq < SMALL {
        return h_new;
    }
    
    let dx_dg = delta_x.dot(delta_g);
    let dx_h_dx = delta_x.dot(&h_dx);
    
    // Compute Bofill weight
    let r_num = dx_dg - dx_h_dx;
    let r_denom = dx_norm_sq * diff_norm_sq;
    
    let phi = if r_num.abs() > SMALL && r_denom.abs() > SMALL {
        (1.0 - (r_num * r_num) / r_denom).clamp(0.0, 1.0)
    } else {
        0.5
    };
    
    // Compute BFGS update terms
    let bfgs_valid = dx_dg.abs() > SMALL && dx_h_dx.abs() > SMALL;
    
    // Compute Powell update terms  
    let powell_denom = diff.dot(delta_x);
    let powell_valid = powell_denom.abs() > SMALL;
    
    let n = hessian.nrows();
    for i in 0..n {
        for j in 0..=i {
            let mut update = 0.0;
            
            // BFGS contribution (weighted by 1-φ)
            if bfgs_valid {
                let bfgs = delta_g[i] * delta_g[j] / dx_dg 
                         - h_dx[i] * h_dx[j] / dx_h_dx;
                update += (1.0 - phi) * bfgs;
            }
            
            // Powell contribution (weighted by φ)
            if powell_valid {
                let powell = diff[i] * diff[j] / powell_denom;
                update += phi * powell;
            }
            
            h_new[(i, j)] += update;
            if i != j {
                h_new[(j, i)] += update;
            }
        }
    }
    
    h_new
}

/// Updates the inverse Hessian using BFGS formula.
///
/// This is the standard BFGS inverse update used in the main optimizer:
/// ```text
/// H⁻¹_new = (I - ρ·s·y^T) · H⁻¹ · (I - ρ·y·s^T) + ρ·s·s^T
/// ```
/// where ρ = 1/(y^T·s), s = Δx, y = Δg.
///
/// Equivalent Old Code formula from UpdateX:
/// ```text
/// fac = 1 / (DelG · DelX)
/// fad = 1 / (DelG · H_inv · DelG)
/// w = fac * DelX - fad * H_inv · DelG
/// H_inv_new = H_inv + fac * DelX * DelX^T - fad * HDelG * HDelG^T + fae * w * w^T
/// ```
pub fn update_inverse_hessian_bfgs(
    h_inv: &DMatrix<f64>,
    delta_x: &DVector<f64>,
    delta_g: &DVector<f64>,
) -> DMatrix<f64> {
    if !delta_x.iter().all(|v| v.is_finite()) || !delta_g.iter().all(|v| v.is_finite()) {
        return h_inv.clone();
    }
    if !h_inv.iter().all(|v| v.is_finite()) {
        return h_inv.clone();
    }

    let mut h_inv_new = h_inv.clone();

    // Old Code BFGS update for inverse Hessian
    let h_del_g = h_inv * delta_g;
    
    let fac_denom = delta_g.dot(delta_x);  // DelG · DelX
    let fae = delta_g.dot(&h_del_g);       // DelG · H_inv · DelG
    
    if fac_denom.abs() < SMALL || fae.abs() < SMALL {
        return h_inv_new;
    }
    
    let fac = 1.0 / fac_denom;
    let fad = 1.0 / fae;
    
    // w = fac * DelX - fad * H_inv · DelG
    let w = delta_x * fac - &h_del_g * fad;
    
    // H_inv_new = H_inv + fac * DelX * DelX^T - fad * HDelG * HDelG^T + fae * w * w^T
    let term1 = (delta_x * delta_x.transpose()) * fac;
    let term2 = (&h_del_g * h_del_g.transpose()) * fad;
    let term3 = (&w * w.transpose()) * fae;
    
    h_inv_new += term1 - term2 + term3;

    // Symmetrize
    h_inv_new = 0.5 * (&h_inv_new + h_inv_new.transpose());
    
    // Clip non-finite entries
    for v in h_inv_new.iter_mut() {
        if !v.is_finite() {
            *v = 0.0;
        }
    }

    h_inv_new
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to check if two floats are approximately equal
    fn approx_eq(a: f64, b: f64, epsilon: f64) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_bfgs_update_basic() {
        let h = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1.0, 1.0]));
        let dx = DVector::from_vec(vec![0.1, 0.2, 0.3]);
        let dg = DVector::from_vec(vec![0.15, 0.25, 0.35]);
        
        let h_new = update_hessian_bfgs(&h, &dx, &dg);
        
        // Check symmetry
        assert!(approx_eq((h_new.clone() - h_new.transpose()).norm(), 0.0, 1e-12));
        
        // Check dimensions
        assert_eq!(h_new.nrows(), 3);
        assert_eq!(h_new.ncols(), 3);
    }

    #[test]
    fn test_bfgs_negative_curvature_skipped() {
        let h = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1.0, 1.0]));
        let dx = DVector::from_vec(vec![0.1, 0.2, 0.3]);
        let dg = DVector::from_vec(vec![-0.1, -0.2, -0.3]);  // Negative curvature
        
        let h_new = update_hessian_bfgs(&h, &dx, &dg);
        
        // Should return original (update skipped)
        assert!(approx_eq((h_new - h).norm(), 0.0, 1e-12));
    }

    #[test]
    fn test_powell_update_basic() {
        let h = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1.0, 1.0]));
        let dx = DVector::from_vec(vec![0.1, 0.2, 0.3]);
        let dg = DVector::from_vec(vec![0.15, 0.25, 0.35]);
        
        let h_new = update_hessian_powell(&h, &dx, &dg);
        
        // Check symmetry
        assert!(approx_eq((h_new.clone() - h_new.transpose()).norm(), 0.0, 1e-12));
    }

    #[test]
    fn test_bofill_update_basic() {
        let h = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1.0, 1.0]));
        let dx = DVector::from_vec(vec![0.1, 0.2, 0.3]);
        let dg = DVector::from_vec(vec![0.15, 0.25, 0.35]);
        
        let h_new = update_hessian_bofill(&h, &dx, &dg);
        
        // Check symmetry
        assert!(approx_eq((h_new.clone() - h_new.transpose()).norm(), 0.0, 1e-12));
    }

    #[test]
    fn test_inverse_hessian_bfgs() {
        let h_inv = DMatrix::from_diagonal(&DVector::from_vec(vec![0.7, 0.7, 0.7]));
        let dx = DVector::from_vec(vec![0.1, 0.2, 0.3]);
        let dg = DVector::from_vec(vec![0.05, 0.1, 0.15]);
        
        let h_inv_new = update_inverse_hessian_bfgs(&h_inv, &dx, &dg);
        
        // Check symmetry
        assert!(approx_eq((h_inv_new.clone() - h_inv_new.transpose()).norm(), 0.0, 1e-12));
    }

    #[test]
    fn test_zero_step_handled() {
        let h = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1.0, 1.0]));
        let dx = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let dg = DVector::from_vec(vec![0.1, 0.2, 0.3]);
        
        // All methods should handle zero step gracefully
        let h_bfgs = update_hessian_bfgs(&h, &dx, &dg);
        let h_powell = update_hessian_powell(&h, &dx, &dg);
        let h_bofill = update_hessian_bofill(&h, &dx, &dg);
        
        assert_eq!(h_bfgs, h);
        assert_eq!(h_powell, h);
        assert_eq!(h_bofill, h);
    }
}
