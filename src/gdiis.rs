//! GDIIS (Geometry Direct Inversion in Iterative Subspace) implementation.
//!
//! This module implements the experimental GDIIS algorithm
//!
//! # Algorithm Overview
//!
//! GDIIS accelerates geometry optimization by constructing an optimal linear
//! combination of previous geometries and error vectors. The method:
//!
//! 1. Builds an overlap matrix A from error vectors: A[i,j] = <e_i, e_j>
//! 2. Maintains A⁻¹ using SR1 (Symmetric Rank-One) updates
//! 3. Solves for coefficients c such that Σc_i = 1 and residual is minimized
//! 4. Validates solution using cosine and coefficient checks
//!

use nalgebra::{DMatrix, DVector};
use std::collections::VecDeque;

/// Cosine check mode for GDIIS validation.
///
/// Controls how the GDIIS step is validated against the last error vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CosineCheckMode {
    /// No cosine check (ICos=0)
    None,
    /// CosLim = 0.0 (ICos=1)
    Zero,
    /// CosLim = 0.71 (ICos=2)
    #[default]
    Standard,
    /// Variable CosLim based on number of vectors used (ICos=3)
    Variable,
    /// CosLim = √3/2 ≈ 0.866 (ICos≥4)
    Strict,
}

/// Coefficient check mode for GDIIS validation.
///
/// Controls validation of DIIS coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoeffCheckMode {
    /// No coefficient check (IChkC=0)
    None,
    /// Regular coefficient check (IChkC=1)
    #[default]
    Regular,
    /// Force last two vectors to have larger weight (IChkC=2)
    ForceRecent,
    /// Combined: Regular + ForceRecent (IChkC=3)
    Combined,
    /// Regular without cosine modification (IChkC=4)
    RegularNoCosine,
}

/// Error types for GDIIS operations.
#[derive(Debug, Clone)]
pub enum GdiisError {
    /// Matrix is singular or nearly singular
    SingularMatrix,
    /// Coefficient check failed
    CoefficientCheckFailed {
        /// Description of why the coefficient check failed
        reason: String,
    },
    /// Cosine check failed - GDIIS step direction differs too much from last error vector
    CosineCheckFailed {
        /// Computed cosine between GDIIS step and last error vector
        cos: f64,
        /// Required minimum cosine limit
        limit: f64,
    },
    /// Redundant vectors detected
    RedundantVectors,
    /// Error vector ratio too large
    RatioTooLarge,
    /// Insufficient history
    InsufficientHistory,
}


/// Lower triangular matrix storage for DIIS overlap matrix.
///
/// Stores elements in packed format: [A(1,1), A(2,1), A(2,2), A(3,1), ...]
#[derive(Debug, Clone)]
pub struct TriangularMatrix {
    /// Packed storage for lower triangular elements
    data: Vec<f64>,
    /// Matrix dimension
    size: usize,
}

impl TriangularMatrix {
    /// Creates a new triangular matrix of given size.
    pub fn new(size: usize) -> Self {
        let len = (size * (size + 1)) / 2;
        Self {
            data: vec![0.0; len],
            size,
        }
    }

    /// Returns the matrix dimension.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Converts (i, j) indices to packed storage index.
    #[inline]
    fn index(&self, i: usize, j: usize) -> usize {
        let (row, col) = if i >= j { (i, j) } else { (j, i) };
        (row * (row + 1)) / 2 + col
    }

    /// Gets element at (i, j).
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[self.index(i, j)]
    }

    /// Sets element at (i, j).
    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        let idx = self.index(i, j);
        self.data[idx] = value;
    }

    /// Multiplies matrix by vector: y = A * x
    ///
    /// # Panics
    ///
    /// Panics if vector length doesn't match matrix size.
    pub fn multiply_vector(&self, x: &[f64]) -> Vec<f64> {
        let n = self.size;
        // Use the minimum of matrix size and vector length to avoid out-of-bounds
        let len = n.min(x.len());
        let mut y = vec![0.0; len];
        
        for i in 0..len {
            for j in 0..len {
                y[i] += self.get(i, j) * x[j];
            }
        }
        y
    }

    /// Scales all elements by a factor.
    pub fn scale(&mut self, factor: f64) {
        for v in &mut self.data {
            *v *= factor;
        }
    }

    /// Copies elements from another triangular matrix.
    pub fn copy_from(&mut self, other: &TriangularMatrix) {
        self.data.copy_from_slice(&other.data);
    }

    /// Extends matrix by one row/column, copying existing elements.
    pub fn extend(&mut self) {
        self.size += 1;
        let new_len = (self.size * (self.size + 1)) / 2;
        self.data.resize(new_len, 0.0);
    }
}

/// DIIS coefficient solver using SR1 inverse updates.
///
/// Implements the DIISC algorithm.
/// Solves the DIIS equations iteratively while maintaining A⁻¹.
pub struct DiisCoeffSolver {
    /// Convergence tolerance
    convergence: f64,
    /// Maximum coefficient sum allowed
    max_coeff_sum: f64,
}

impl Default for DiisCoeffSolver {
    fn default() -> Self {
        Self {
            convergence: 1e-8,
            max_coeff_sum: 1e8,
        }
    }
}

impl DiisCoeffSolver {
    /// Creates a new solver with specified convergence tolerance.
    pub fn new(convergence: f64) -> Self {
        Self {
            convergence,
            max_coeff_sum: 1e8,
        }
    }

    /// Solves for DIIS coefficients using iterative SR1 updates.
    ///
    /// # Arguments
    ///
    /// * `a` - Lower triangular overlap matrix A[i,j] = <e_i, e_j>
    /// * `a_inv` - Inverse of A (maintained via SR1 updates)
    /// * `coeffs` - Initial coefficient guess (modified in place)
    /// * `n_start` - Starting number of vectors (with existing solution)
    /// * `n_total` - Total number of vectors to solve for
    ///
    /// # Returns
    ///
    /// Number of vectors actually used, or error if failed.
    pub fn solve(
        &self,
        a: &TriangularMatrix,
        a_inv: &mut TriangularMatrix,
        coeffs: &mut [f64],
        n_start: usize,
        n_total: usize,
    ) -> Result<usize, GdiisError> {
        let mut n_used = n_start;

        for n in (n_start + 1)..=n_total {
            // Save previous coefficients
            let coeffs_save: Vec<f64> = coeffs[..n].to_vec();

            // Compute residual: y = 1 - A*c
            let ac = a.multiply_vector(&coeffs[..n]);
            let mut y: Vec<f64> = vec![1.0; n];
            for i in 0..n {
                y[i] -= ac[i];
            }

            let err_max = y.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            let _y_len = (y.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();

            let mut iter = 0;
            let max_iter = 10 * n;

            while err_max > self.convergence && iter <= max_iter {
                // Compute dx = A_inv * y
                let dx = a_inv.multiply_vector(&y);
                
                // Compute dy = A * dx
                let dy = a.multiply_vector(&dx);

                // SR1 update for A_inv
                self.sr1_update(a_inv, &dy, &dx, n)?;

                // Compute step scaling
                let s11: f64 = dy.iter().map(|v| v * v).sum();
                let s22: f64 = y.iter().map(|v| v * v).sum();
                let s12: f64 = dy.iter().zip(y.iter()).map(|(a, b)| a * b).sum();

                let c12 = if s11 * s22 > 1e-30 {
                    s12 / (s11 * s22).sqrt()
                } else {
                    0.0
                };

                let ss = if c12.abs() > 1e-3 && s11 > 1e-30 {
                    c12 * (s22 / s11).sqrt()
                } else {
                    // Fallback: use y directly as search direction
                    let dy_new = a.multiply_vector(&y);
                    let s11_new: f64 = dy_new.iter().map(|v| v * v).sum();
                    let s12_new: f64 = dy_new.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
                    if s11_new > 1e-30 {
                        s12_new / s11_new * (s22 / s11_new).sqrt()
                    } else {
                        1.0
                    }
                };

                // Update coefficients
                for i in 0..n {
                    coeffs[i] += ss * dx[i];
                }

                // Check for divergence
                let coeff_sum: f64 = coeffs[..n].iter().sum();
                if coeff_sum.abs() > self.max_coeff_sum {
                    // Revert to previous solution
                    coeffs[..n].copy_from_slice(&coeffs_save);
                    return Ok(n.saturating_sub(1).max(n_start));
                }

                // Check redundancy
                let dx_len = (dx.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
                if dx_len <= self.convergence * self.convergence {
                    break;
                }

                // Recompute residual
                let ac = a.multiply_vector(&coeffs[..n]);
                for i in 0..n {
                    y[i] = 1.0 - ac[i];
                }

                iter += 1;
            }

            // Check if DIIS failed due to redundancy
            if (iter == 0 || iter > max_iter) && err_max > self.convergence {
                coeffs[..n].copy_from_slice(&coeffs_save);
                return Ok(n.saturating_sub(1).max(n_start));
            }

            n_used = n;
        }

        // Normalize coefficients
        // Sc = One/Abs(CSum); Call AScale(NN,SC,C,C)
        // This preserves the sign of coefficients while ensuring sum = ±1
        let coeff_sum: f64 = coeffs[..n_used].iter().sum();
        if coeff_sum.abs() <= 1e-4 {
            // Sum too small - reset to use only last point 
            for c in &mut coeffs[..n_used] {
                *c = 0.0;
            }
            coeffs[0] = 1.0;
        } else {
            // Scale by 1/|sum| to normalize
            let scale = 1.0 / coeff_sum.abs();
            for c in &mut coeffs[..n_used] {
                *c *= scale;
            }
        }

        Ok(n_used)
    }

    /// SR1 update for inverse matrix.
    fn sr1_update(
        &self,
        a_inv: &mut TriangularMatrix,
        dy: &[f64],
        dx: &[f64],
        n: usize,
    ) -> Result<(), GdiisError> {
        // Compute diff = dx - A_inv * dy
        let a_inv_dy = a_inv.multiply_vector(dy);
        let mut diff = vec![0.0; n];
        for i in 0..n {
            diff[i] = dx[i] - a_inv_dy[i];
        }

        // Compute denominator: dy · diff
        let denom: f64 = dy.iter().zip(diff.iter()).map(|(a, b)| a * b).sum();

        if denom.abs() < 1e-14 {
            return Ok(()); // Skip update if denominator too small
        }

        // SR1 update: A_inv += diff * diff^T / denom
        for i in 0..n {
            for j in 0..=i {
                let update = diff[i] * diff[j] / denom;
                let current = a_inv.get(i, j);
                a_inv.set(i, j, current + update);
            }
        }

        Ok(())
    }
}

/// Main GDIIS optimizer.
///
/// Implements the full GDIIS algorithm with validation checks.
pub struct GdiisOptimizer {
    /// Maximum number of vectors to store
    pub max_vectors: usize,
    /// Cosine check mode
    pub cosine_check: CosineCheckMode,
    /// Coefficient check mode
    pub coeff_check: CoeffCheckMode,
    /// Overlap matrix A
    a_matrix: TriangularMatrix,
    /// Inverse of A
    a_inv: TriangularMatrix,
    /// Current scale factor
    scale: f64,
    /// Minimum error norm squared
    el2_min: f64,
    /// DIIS coefficient solver
    solver: DiisCoeffSolver,
}

impl GdiisOptimizer {
    /// Creates a new GDIIS optimizer.
    pub fn new(max_vectors: usize) -> Self {
        Self {
            max_vectors,
            cosine_check: CosineCheckMode::Standard,
            coeff_check: CoeffCheckMode::Regular,
            a_matrix: TriangularMatrix::new(0),
            a_inv: TriangularMatrix::new(0),
            scale: 1.0,
            el2_min: 1.0,
            solver: DiisCoeffSolver::default(),
        }
    }

    /// Resets the optimizer state.
    pub fn reset(&mut self) {
        self.a_matrix = TriangularMatrix::new(0);
        self.a_inv = TriangularMatrix::new(0);
        self.scale = 1.0;
        self.el2_min = 1.0;
    }

    /// Computes a GDIIS step.
    ///
    /// # Arguments
    ///
    /// * `coords` - History of coordinate vectors
    /// * `errors` - History of error vectors (gradients or Newton steps)
    /// * `hessians` - History of Hessian matrices (for error vector computation)
    ///
    /// # Returns
    ///
    /// Tuple of (new_coords, coefficients, n_used) or error.
    pub fn compute_step(
        &mut self,
        coords: &VecDeque<DVector<f64>>,
        errors: &VecDeque<DVector<f64>>,
        _hessians: &VecDeque<DMatrix<f64>>,
    ) -> Result<(DVector<f64>, Vec<f64>, usize), GdiisError> {
        let n = coords.len();
        if n < 2 {
            return Err(GdiisError::InsufficientHistory);
        }

        let n_use = n.min(self.max_vectors);
        let dim = coords[0].len();

        // Initialize or extend matrices
        // Reset if matrix size doesn't match expected size (history may have changed)
        let current_size = self.a_matrix.size();
        if current_size == 0 || current_size > n_use || n_use > current_size + 1 {
            // Reinitialize matrices for consistency
            self.initialize_matrices(errors, n_use)?;
        } else if n_use > current_size {
            self.extend_matrices(errors, n_use)?;
        }
        // If n_use == current_size, matrices are already correct size

        // Solve for coefficients
        let mut coeffs = vec![0.0; n_use];
        coeffs[0] = 1.0;

        let n_used = self.solver.solve(
            &self.a_matrix,
            &mut self.a_inv,
            &mut coeffs,
            1,
            n_use,
        )?;

        // Validate coefficients
        self.validate_coefficients(&coeffs, n_used)?;

        // Compute interpolated geometry
        let mut x_interp = DVector::zeros(dim);
        for (i, coord) in coords.iter().take(n_used).enumerate() {
            x_interp += coord * coeffs[i];
        }

        // Compute interpolated error (residuum)
        let mut e_interp = DVector::zeros(dim);
        for (i, error) in errors.iter().take(n_used).enumerate() {
            e_interp += error * coeffs[i];
        }

        // Validate using cosine check
        if n_used >= 2 {
            self.validate_cosine(&e_interp, &errors[0], n_used)?;
        }

        // Compute GDIIS step: x_new = x_interp - e_interp
        let x_new = &x_interp - &e_interp;

        Ok((x_new, coeffs[..n_used].to_vec(), n_used))
    }

    /// Initializes overlap matrices from error vectors.
    fn initialize_matrices(
        &mut self,
        errors: &VecDeque<DVector<f64>>,
        n: usize,
    ) -> Result<(), GdiisError> {
        self.a_matrix = TriangularMatrix::new(n);
        self.a_inv = TriangularMatrix::new(n);

        // Build overlap matrix
        for i in 0..n {
            for j in 0..=i {
                let dot = errors[i].dot(&errors[j]);
                self.a_matrix.set(i, j, dot);
            }
        }

        // Find minimum diagonal for scaling
        self.el2_min = (0..n)
            .map(|i| self.a_matrix.get(i, i))
            .fold(f64::INFINITY, f64::min);

        if self.el2_min < 1e-30 {
            return Err(GdiisError::SingularMatrix);
        }

        self.scale = 1.0 / self.el2_min;

        // Scale matrix
        self.a_matrix.scale(self.scale);

        // Initialize inverse as identity (will be updated by solver)
        for i in 0..n {
            self.a_inv.set(i, i, 1.0);
        }

        Ok(())
    }

    /// Extends matrices with new error vector.
    fn extend_matrices(
        &mut self,
        errors: &VecDeque<DVector<f64>>,
        n: usize,
    ) -> Result<(), GdiisError> {
        let old_n = self.a_matrix.size();
        
        if n <= old_n {
            return Ok(());
        }

        // If size mismatch is too large, reinitialize instead of extending
        if n > old_n + 1 {
            // Matrix state is inconsistent, reinitialize
            return self.initialize_matrices(errors, n);
        }

        // Extend matrices by one
        self.a_matrix.extend();
        self.a_inv.extend();

        // Add new row/column to A
        let new_idx = n - 1;
        for i in 0..n {
            let dot = errors[i].dot(&errors[new_idx]);
            self.a_matrix.set(new_idx, i, dot * self.scale);
        }

        // Check if rescaling needed
        let el2_new = self.a_matrix.get(new_idx, new_idx) / self.scale;
        if el2_new < self.el2_min {
            let old_scale = self.scale;
            self.el2_min = el2_new;
            self.scale = 1.0 / self.el2_min;

            // Rescale matrices
            let ratio = old_scale / self.scale;
            for i in 0..old_n {
                for j in 0..=i {
                    let val = self.a_matrix.get(i, j);
                    self.a_matrix.set(i, j, val / ratio);
                    let inv_val = self.a_inv.get(i, j);
                    self.a_inv.set(i, j, inv_val * ratio);
                }
            }
        }

        // Initialize new row/column of inverse
        self.a_inv.set(new_idx, new_idx, 1.0);

        Ok(())
    }

    /// Validates DIIS coefficients.
    ///
    /// GDIIS coefficient check logic:
    /// - CMax = 15.0
    /// - CMin = 1 - CMax = -14 for IChkC != 1
    /// - CMin = 4 * (1 - CMax) = -56 for IChkC == 1 (Regular mode)
    fn validate_coefficients(&self, coeffs: &[f64], n_used: usize) -> Result<(), GdiisError> {
        if self.coeff_check == CoeffCheckMode::None {
            return Ok(());
        }

        let c_max = 15.0;
        // Old code: If(IChkC.eq.1) CMin = Two*Two*CMin
        // IChkC=1 corresponds to Regular mode
        let c_min = if self.coeff_check == CoeffCheckMode::Regular {
            4.0 * (1.0 - c_max) // -56.0
        } else {
            1.0 - c_max // -14.0
        };

        // Check for excessive extrapolation (sum of negative coefficients)
        let neg_sum: f64 = coeffs[..n_used]
            .iter()
            .filter(|&&c| c < 0.0)
            .sum();

        if neg_sum < c_min {
            return Err(GdiisError::CoefficientCheckFailed {
                reason: format!("Excessive extrapolation: neg_sum = {:.4} < {:.4}", neg_sum, c_min),
            });
        }

        // Check that last point has weight (Small = 1e-4)
        if coeffs[0].abs() < 1e-4 {
            return Err(GdiisError::CoefficientCheckFailed {
                reason: "Last point has no weight".to_string(),
            });
        }

        // For ForceRecent mode (IChkC > 1), check that recent points dominate
        if matches!(self.coeff_check, CoeffCheckMode::ForceRecent | CoeffCheckMode::Combined) {
            let max_idx = coeffs[..n_used]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            if max_idx != 0 {
                // Max coefficient not at last point
                if n_used > 2 && max_idx != 1 {
                    // Neither of the last two points has max weight
                    return Err(GdiisError::CoefficientCheckFailed {
                        reason: format!("Max coefficient at index {} (not recent)", max_idx),
                    });
                } else if n_used == 2 {
                    // For 2 points, Code swaps or fails based on coefficient values
                    let cof_max = coeffs[..n_used]
                        .iter()
                        .map(|c| c.abs())
                        .fold(0.0_f64, f64::max);
                    if cof_max > 1.0 && self.coeff_check == CoeffCheckMode::ForceRecent {
                        // Would swap, but we just fail here
                        return Err(GdiisError::CoefficientCheckFailed {
                            reason: "Two-point extrapolation with wrong direction".to_string(),
                        });
                    }
                }
            }
        }

        Ok(())
    }

    /// Validates GDIIS step using cosine check.
    fn validate_cosine(
        &self,
        residuum: &DVector<f64>,
        last_error: &DVector<f64>,
        n_used: usize,
    ) -> Result<(), GdiisError> {
        if self.cosine_check == CosineCheckMode::None {
            return Ok(());
        }

        let s11 = residuum.norm_squared();
        let s22 = last_error.norm_squared();
        let s12 = residuum.dot(last_error);

        if s11 * s22 < 1e-30 {
            return Ok(()); // Skip check if vectors too small
        }

        let cos = s12 / (s11 * s22).sqrt();

        let cos_limit = match self.cosine_check {
            CosineCheckMode::None => return Ok(()),
            CosineCheckMode::Zero => 0.0,
            CosineCheckMode::Standard => 0.71,
            CosineCheckMode::Variable => variable_cos_limit(n_used),
            CosineCheckMode::Strict => 0.866,
        };

        if cos < cos_limit {
            Err(GdiisError::CosineCheckFailed { cos, limit: cos_limit })
        } else {
            Ok(())
        }
    }
}

/// Returns variable cosine limit based on number of vectors used.
fn variable_cos_limit(n_used: usize) -> f64 {
    match n_used {
        2 => 0.97,
        3 => 0.84,
        4 => 0.71,
        5 => 0.67,
        6 => 0.62,
        7 => 0.56,
        8 => 0.49,
        9 => 0.41,
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangular_matrix() {
        let mut m = TriangularMatrix::new(3);
        m.set(0, 0, 1.0);
        m.set(1, 0, 2.0);
        m.set(1, 1, 3.0);
        m.set(2, 0, 4.0);
        m.set(2, 1, 5.0);
        m.set(2, 2, 6.0);

        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(1, 0), 2.0);
        assert_eq!(m.get(0, 1), 2.0); // Symmetric access
        assert_eq!(m.get(2, 2), 6.0);
    }

    #[test]
    fn test_triangular_matrix_multiply() {
        let mut m = TriangularMatrix::new(2);
        m.set(0, 0, 1.0);
        m.set(1, 0, 0.0);
        m.set(1, 1, 1.0);

        let x = vec![2.0, 3.0];
        let y = m.multiply_vector(&x);

        assert_eq!(y[0], 2.0);
        assert_eq!(y[1], 3.0);
    }

    #[test]
    fn test_variable_cos_limit() {
        assert_eq!(variable_cos_limit(2), 0.97);
        assert_eq!(variable_cos_limit(4), 0.71);
        assert_eq!(variable_cos_limit(10), 0.0);
    }

    #[test]
    fn test_gdiis_optimizer_creation() {
        let opt = GdiisOptimizer::new(5);
        assert_eq!(opt.max_vectors, 5);
        assert_eq!(opt.cosine_check, CosineCheckMode::Standard);
    }
}
