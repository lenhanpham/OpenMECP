//! GEDIIS (Geometry Energy Direct Inversion in Iterative Subspace) implementation.
//!
//! This module implements the experimental GEDIIS algorithm 
//!
//! # Algorithm Overview
//!
//! GEDIIS extends GDIIS by incorporating energy information into the DIIS matrix.
//! Three variants are available:
//!
//! - **RFO-DIIS**: Uses quadratic step overlaps A[i,j] = <q_i, q_j>
//! - **Energy-DIIS**: Uses energy-based metric A[i,j] = g_i·R_i + g_j·R_j - g_i·R_j - g_j·R_i
//! - **Simultaneous-DIIS**: Combines Energy-DIIS with quadratic terms
//!
//! # References
//!
//! - Li, X.; Frisch, M. J. J. Chem. Theory Comput. 2006, 2, 835-839.
//! - Kudin, K. N.; Scuseria, G. E.; Cancès, E. J. Chem. Phys. 2002, 116, 8255.

use nalgebra::{DMatrix, DVector};
use std::collections::VecDeque;

/// GEDIIS matrix variant selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GediisVariant {
    /// RFO-DIIS: A[i,j] = <q_i, q_j> where q = quadratic step
    #[default]
    RfoDiis,
    /// Energy-DIIS: A[i,j] = g_i·R_i + g_j·R_j - g_i·R_j - g_j·R_i
    EnergyDiis,
    /// Simultaneous: EnDIS + 0.5*(g_i·g_i/H + g_j·g_j/H)
    SimultaneousDiis,
}

/// Tracks energy rises during optimization.
#[derive(Debug, Clone, Default)]
pub struct EnergyRiseTracker {
    /// Number of consecutive energy rises
    pub n_rises: usize,
    /// Maximum allowed consecutive rises before interpolation
    pub max_rises: usize,
    /// Previous energy for comparison
    prev_energy: Option<f64>,
    /// Whether to force interpolation
    pub do_interpolate: bool,
}

impl EnergyRiseTracker {
    /// Creates a new tracker with specified max rises.
    pub fn new(max_rises: usize) -> Self {
        Self {
            n_rises: 0,
            max_rises,
            prev_energy: None,
            do_interpolate: false,
        }
    }


    /// Updates tracker with new energy value.
    ///
    /// # Arguments
    ///
    /// * `energy` - Current energy (or energy difference for MECP)
    /// * `tolerance` - Tolerance for energy rise detection
    pub fn update(&mut self, energy: f64, tolerance: f64) {
        if let Some(prev) = self.prev_energy {
            if energy > prev + tolerance {
                self.n_rises += 1;
            } else {
                self.n_rises = 0;
            }
        }
        self.prev_energy = Some(energy);
        self.do_interpolate = self.n_rises > self.max_rises;
    }

    /// Resets the tracker.
    pub fn reset(&mut self) {
        self.n_rises = 0;
        self.prev_energy = None;
        self.do_interpolate = false;
    }
}

/// Configuration for GEDIIS optimizer.
#[derive(Debug, Clone)]
pub struct GediisConfig {
    /// Maximum number of vectors to use
    pub max_vectors: usize,
    /// GEDIIS variant to use
    pub variant: GediisVariant,
    /// Switching threshold for RMS error (SimSw in Old Code)
    pub sim_switch: f64,
    /// Maximum consecutive energy rises before interpolation
    pub max_rises: usize,
    /// Whether to automatically switch variants
    pub auto_switch: bool,
    /// Scaling factor for TS search (SclDIS in Old Code)
    pub ts_scale: f64,
    /// Number of negative eigenvalues (0=min, 1=TS)
    pub n_neg: usize,
}

impl Default for GediisConfig {
    fn default() -> Self {
        Self {
            max_vectors: 5,
            variant: GediisVariant::RfoDiis,
            sim_switch: 0.005,
            max_rises: 1,
            auto_switch: true,
            ts_scale: 1.0,
            n_neg: 0,
        }
    }
}

/// Main GEDIIS optimizer.
pub struct GediisOptimizer {
    /// Configuration
    pub config: GediisConfig,
    /// Energy rise tracker
    pub rise_tracker: EnergyRiseTracker,
}

impl GediisOptimizer {
    /// Creates a new GEDIIS optimizer with default configuration.
    pub fn new() -> Self {
        Self {
            config: GediisConfig::default(),
            rise_tracker: EnergyRiseTracker::new(1),
        }
    }

    /// Creates a new GEDIIS optimizer with specified configuration.
    pub fn with_config(config: GediisConfig) -> Self {
        let rise_tracker = EnergyRiseTracker::new(config.max_rises);
        Self { config, rise_tracker }
    }

    /// Selects the appropriate GEDIIS variant based on optimization state.
    ///
    /// Implements the decision logic from Old Code GeoDIS:
    /// - OKEnD requires NGDIIS >= 4 for TS search, >= 2 for minimum
    /// - OKEnD also requires |DC| >= SmlDif where DC = E(1) - E(2)
    /// - EnDIS used when RMSErr > SimSw or energy rises
    pub fn select_variant(
        &self,
        rms_error: f64,
        energy_rises: bool,
        n_points: usize,
        energies: Option<&VecDeque<f64>>,
    ) -> GediisVariant {
        if !self.config.auto_switch {
            return self.config.variant;
        }

        // Check if Energy-DIIS is appropriate (OKEnD in Old Code)
        let mut ok_en_diis = if self.config.n_neg > 0 {
            n_points >= 4  // TS search needs more points
        } else {
            n_points >= 2  // Minimum search
        };

        // Additional check: energy difference must be significant
        // Old Code: OKEnD = OKEnD.and.(Abs(DC).ge.SmlDif.or.Rises)
        const SML_DIF: f64 = 1e-5;
        if let Some(e) = energies {
            if e.len() >= 2 {
                let dc = e[0] - e[1];
                ok_en_diis = ok_en_diis && (dc.abs() >= SML_DIF || energy_rises);
            }
        }

        let use_energy = rms_error > self.config.sim_switch || energy_rises;

        if use_energy && ok_en_diis {
            GediisVariant::EnergyDiis
        } else {
            GediisVariant::RfoDiis
        }
    }

    /// Computes a GEDIIS step.
    ///
    /// # Arguments
    ///
    /// * `coords` - History of coordinate vectors (Angstrom)
    /// * `grads` - History of gradient vectors (Ha/Å)
    /// * `energies` - History of energies (Ha)
    /// * `quad_steps` - History of quadratic steps (optional, for RFO-DIIS)
    ///
    /// # Returns
    ///
    /// Tuple of (new_coords, coefficients) or None if failed.
    pub fn compute_step(
        &mut self,
        coords: &VecDeque<DVector<f64>>,
        grads: &VecDeque<DVector<f64>>,
        energies: &VecDeque<f64>,
        quad_steps: Option<&VecDeque<DVector<f64>>>,
    ) -> Option<(DVector<f64>, Vec<f64>)> {
        let n = coords.len();
        if n < 2 {
            return None;
        }

        let n_use = n.min(self.config.max_vectors);
        let dim = coords[0].len();

        // Compute RMS error for variant selection
        let rms_error = if !grads.is_empty() {
            let g_norm_sq: f64 = grads[0].norm_squared();
            (g_norm_sq / dim as f64).sqrt()
        } else {
            0.0
        };

        // Check for energy rises
        if !energies.is_empty() {
            self.rise_tracker.update(energies[0], 1e-6);
        }

        // Select variant
        let variant = self.select_variant(
            rms_error,
            self.rise_tracker.do_interpolate,
            n_use,
            Some(energies),
        );

        // Build GEDIIS matrix based on variant
        let b_matrix = match variant {
            GediisVariant::RfoDiis => {
                if let Some(steps) = quad_steps {
                    self.build_rfo_diis_matrix(steps, n_use)
                } else {
                    // Fallback: use gradients as pseudo-steps
                    self.build_rfo_diis_matrix(grads, n_use)
                }
            }
            GediisVariant::EnergyDiis => {
                self.build_energy_diis_matrix(grads, coords, n_use)
            }
            GediisVariant::SimultaneousDiis => {
                self.build_sim_diis_matrix(grads, coords, energies, n_use, quad_steps)
            }
        };

        // Build RHS vector
        // For RFO-DIIS and Sim-DIIS: only constraint equation (matching Old Code)
        // For Energy-DIIS: energies are used in the EnCoef solver differently
        let mut rhs = DVector::zeros(n_use + 1);
        if variant == GediisVariant::EnergyDiis {
            // Energy-DIIS uses energies to bias toward lower energy points
            for i in 0..n_use {
                rhs[i] = -energies.get(i).copied().unwrap_or(0.0);
            }
        }
        // All variants: constraint that sum of coefficients = 1
        rhs[n_use] = 1.0;

        // Solve for coefficients
        let solution = b_matrix.lu().solve(&rhs)?;

        // Extract coefficients (exclude Lagrange multiplier)
        let coeffs: Vec<f64> = solution.rows(0, n_use).iter().copied().collect();

        // Validate coefficients
        if coeffs.iter().any(|c| !c.is_finite()) {
            return None;
        }

        // Interpolate geometry
        let mut x_interp = DVector::zeros(dim);
        for (i, coord) in coords.iter().take(n_use).enumerate() {
            x_interp += coord * coeffs[i];
        }

        // Interpolate gradient
        let mut g_interp = DVector::zeros(dim);
        for (i, grad) in grads.iter().take(n_use).enumerate() {
            g_interp += grad * coeffs[i];
        }

        // Compute new geometry: x_new = x_interp - g_interp
        let x_new = &x_interp - &g_interp;

        Some((x_new, coeffs))
    }

    /// Builds RFO-DIIS matrix from quadratic steps.
    ///
    /// A[i,j] = <q_i, q_j>
    /// For TS search, first coordinate is weighted by ts_scale.
    fn build_rfo_diis_matrix(
        &self,
        steps: &VecDeque<DVector<f64>>,
        n: usize,
    ) -> DMatrix<f64> {
        let mut b = DMatrix::zeros(n + 1, n + 1);

        for i in 0..n {
            for j in 0..n {
                let dot = if self.config.n_neg > 0 && steps[i].len() > 0 {
                    // TS search: weight first coordinate
                    self.config.ts_scale * steps[i][0] * steps[j][0]
                        + steps[i].rows(1, steps[i].len() - 1)
                            .dot(&steps[j].rows(1, steps[j].len() - 1))
                } else {
                    steps[i].dot(&steps[j])
                };
                b[(i, j)] = dot;
            }
        }

        // Add constraint row/column
        for i in 0..n {
            b[(i, n)] = 1.0;
            b[(n, i)] = 1.0;
        }
        b[(n, n)] = 0.0;

        b
    }

    /// Builds Energy-DIIS matrix.
    ///
    /// Trace: AMTrc[i,j] = -<g_i, x_j>
    /// A[i,j] = AMTrc[i,i] + AMTrc[j,j] - AMTrc[i,j] - AMTrc[j,i]
    ///
    /// For TS search (n_neg > 0), the first coordinate is weighted by ts_scale
    /// to emphasize the reaction coordinate, matching Old Code:
    /// `AMTrc(I+1,J+1) = -SclDIS*ANFF(1,I)*ANCC(1,J) - SProd(NVarT-1,ANFF(2,I),ANCC(2,J))`
    fn build_energy_diis_matrix(
        &self,
        grads: &VecDeque<DVector<f64>>,
        coords: &VecDeque<DVector<f64>>,
        n: usize,
    ) -> DMatrix<f64> {
        let mut b = DMatrix::zeros(n + 1, n + 1);

        // Build trace matrix: AMTrc[i,j] = -<g_i, x_j>
        // For TS search, weight first coordinate by ts_scale
        let mut trace = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                if self.config.n_neg > 0 && !grads[i].is_empty() && !coords[j].is_empty() {
                    // TS search: weight first coordinate
                    let first_term = -self.config.ts_scale * grads[i][0] * coords[j][0];
                    let rest_term = if grads[i].len() > 1 {
                        -grads[i].rows(1, grads[i].len() - 1)
                            .dot(&coords[j].rows(1, coords[j].len() - 1))
                    } else {
                        0.0
                    };
                    trace[(i, j)] = first_term + rest_term;
                } else {
                    trace[(i, j)] = -grads[i].dot(&coords[j]);
                }
            }
        }

        // Build Energy-DIIS matrix: A[i,j] = trace[i,i] + trace[j,j] - trace[i,j] - trace[j,i]
        for i in 0..n {
            for j in 0..n {
                b[(i, j)] = trace[(i, i)] + trace[(j, j)] - trace[(i, j)] - trace[(j, i)];
            }
        }

        // Add constraint row/column (sum of coefficients = 1)
        for i in 0..n {
            b[(i, n)] = 1.0;
            b[(n, i)] = 1.0;
        }
        b[(n, n)] = 0.0;

        b
    }

    /// Builds Simultaneous-DIIS matrix.
    ///
    /// Combines Energy-DIIS with quadratic energy approximation.
    /// Formula: A[i,j] = EnDIS[i,j] + 0.5*(q_i·f_i + q_j·f_j)
    /// where q = quadratic step and f = forces (negative gradient).
    ///
    /// This matches the Old Code GeoDIS implementation:
    /// `Half*(SProd(NVarT,AQuad(1,I),ANFF(1,I)) + SProd(NVarT,AQuad(1,J),ANFF(1,J)))`
    fn build_sim_diis_matrix(
        &self,
        grads: &VecDeque<DVector<f64>>,
        coords: &VecDeque<DVector<f64>>,
        _energies: &VecDeque<f64>,
        n: usize,
        quad_steps: Option<&VecDeque<DVector<f64>>>,
    ) -> DMatrix<f64> {
        // Start with Energy-DIIS matrix
        let mut b = self.build_energy_diis_matrix(grads, coords, n);

        // Add quadratic energy approximation: 0.5*(q_i·f_i + q_j·f_j)
        // where f = -grad (forces are negative gradients)
        // If quad_steps not available, use grads as approximation
        for i in 0..n {
            for j in 0..n {
                let quad_term = if let Some(steps) = quad_steps {
                    // Correct formula: 0.5*(q_i·(-g_i) + q_j·(-g_j))
                    // Since forces = -grads, this is -0.5*(q_i·g_i + q_j·g_j)
                    -0.5 * (steps[i].dot(&grads[i]) + steps[j].dot(&grads[j]))
                } else {
                    // Fallback: use gradient norm squared as approximation
                    // This is less accurate but provides a reasonable estimate
                    0.5 * (grads[i].norm_squared() + grads[j].norm_squared())
                };
                b[(i, j)] += quad_term;
            }
        }

        b
    }
}

impl Default for GediisOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Computes dynamic GEDIIS weight based on energy trend.
///
/// Returns weight in [0.0, 0.98] for blending GDIIS and GEDIIS.
pub fn compute_dynamic_gediis_weight(
    energies: &VecDeque<f64>,
    displacements: &VecDeque<f64>,
) -> f64 {
    let n = energies.len();
    if n < 5 {
        return 0.0;  // Not enough history
    }

    // Check for stuck optimizer
    let recent_displacements: Vec<f64> = displacements.iter().take(3).copied().collect();
    let avg_disp = recent_displacements.iter().sum::<f64>() / recent_displacements.len() as f64;
    if avg_disp < 1e-8 {
        return 0.0;  // Stuck, use pure GDIIS
    }

    // Count uphill steps
    let mut uphill_count = 0;
    for i in 0..(n - 1).min(5) {
        if energies[i] > energies[i + 1] + 1e-8 {
            uphill_count += 1;
        }
    }

    // If too many uphill steps, reduce GEDIIS weight
    if uphill_count >= 2 {
        return 0.0;
    }

    // Compute energy trend using linear regression
    let x_mean = (n - 1) as f64 / 2.0;
    let y_mean: f64 = energies.iter().sum::<f64>() / n as f64;

    let mut num = 0.0;
    let mut den = 0.0;
    for (i, &e) in energies.iter().enumerate() {
        let x = i as f64;
        num += (x - x_mean) * (e - y_mean);
        den += (x - x_mean) * (x - x_mean);
    }

    let slope = if den.abs() > 1e-14 { num / den } else { 0.0 };

    // Compute deviation from trend
    let mut max_dev: f64 = 0.0;
    for (i, &e) in energies.iter().enumerate() {
        let predicted = y_mean + slope * (i as f64 - x_mean);
        let dev = (e - predicted).abs();
        max_dev = max_dev.max(dev);
    }

    // Map deviation to weight
    let scale = y_mean.abs().max(1e-6);
    let rel_dev = max_dev / scale;

    let weight = if rel_dev < 0.01 {
        0.98  // Very smooth, use mostly GEDIIS
    } else if rel_dev < 0.05 {
        0.8
    } else if rel_dev < 0.1 {
        0.5
    } else if rel_dev < 0.2 {
        0.3
    } else {
        0.0  // Too noisy, use pure GDIIS
    };

    // Apply uphill penalty
    let penalty = 1.0 - 0.3 * uphill_count as f64;
    (weight * penalty).clamp(0.0, 0.98)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_rise_tracker() {
        let mut tracker = EnergyRiseTracker::new(2);
        
        tracker.update(-10.0, 1e-6);
        assert_eq!(tracker.n_rises, 0);
        
        tracker.update(-9.9, 1e-6);  // Energy rose
        assert_eq!(tracker.n_rises, 1);
        assert!(!tracker.do_interpolate);
        
        tracker.update(-9.8, 1e-6);  // Energy rose again
        assert_eq!(tracker.n_rises, 2);
        assert!(!tracker.do_interpolate);
        
        tracker.update(-9.7, 1e-6);  // Third rise
        assert_eq!(tracker.n_rises, 3);
        assert!(tracker.do_interpolate);  // Now should interpolate
    }

    #[test]
    fn test_gediis_config_default() {
        let config = GediisConfig::default();
        assert_eq!(config.max_vectors, 5);
        assert_eq!(config.variant, GediisVariant::RfoDiis);
        assert!(config.auto_switch);
    }

    #[test]
    fn test_variant_selection() {
        let opt = GediisOptimizer::new();
        
        // Create test energies with significant difference
        let mut energies = VecDeque::new();
        energies.push_back(-10.0);
        energies.push_back(-10.1);
        energies.push_back(-10.2);
        
        // Low error, no rises -> RFO-DIIS
        let variant = opt.select_variant(0.001, false, 3, Some(&energies));
        assert_eq!(variant, GediisVariant::RfoDiis);
        
        // High error -> Energy-DIIS (if enough points)
        let variant = opt.select_variant(0.01, false, 3, Some(&energies));
        assert_eq!(variant, GediisVariant::EnergyDiis);
    }

    #[test]
    fn test_dynamic_weight_empty() {
        let energies = VecDeque::new();
        let displacements = VecDeque::new();
        
        let weight = compute_dynamic_gediis_weight(&energies, &displacements);
        assert_eq!(weight, 0.0);
    }
}
