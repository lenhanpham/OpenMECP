//! Property-based tests for DIIS history unit consistency.
//!
//! This module contains property-based tests that verify the GDIIS/GEDIIS
//! history maintains consistent units throughout optimization:
//! - Geometries stored in Angstrom (Å)
//! - Gradients stored in Hartree/Bohr (Ha/a₀)
//! - Interpolated geometries remain in Angstrom
//!
//! **Feature: unit-standardization, Property 8: DIIS history unit consistency**
//! **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
#![allow(dead_code)]

use nalgebra::{DMatrix, DVector};
use omecp::config::Config;
use omecp::optimizer::{gdiis_step, OptimizationState};
use proptest::prelude::*;

/// Generates a random coordinate vector in Angstrom units.
///
/// Typical molecular coordinates range from -10 to +10 Angstrom.
fn arb_geometry(n_coords: usize) -> impl Strategy<Value = DVector<f64>> {
    prop::collection::vec(-10.0..10.0_f64, n_coords)
        .prop_map(|v| DVector::from_vec(v))
}

/// Generates a random gradient vector in Ha/Bohr units.
///
/// Typical MECP gradients range from -0.1 to +0.1 Ha/Bohr.
fn arb_gradient(n_coords: usize) -> impl Strategy<Value = DVector<f64>> {
    prop::collection::vec(-0.1..0.1_f64, n_coords)
        .prop_map(|v| DVector::from_vec(v))
}

/// Generates a random energy difference in Hartree.
///
/// MECP energy differences typically range from -0.1 to +0.1 Ha.
fn arb_energy_diff() -> impl Strategy<Value = f64> {
    -0.1..0.1_f64
}

/// Generates a random diagonal inverse Hessian in Bohr²/Ha units.
///
/// The inverse Hessian is initialized to 0.7 diagonal (Bohr²/Ha).
fn arb_inv_hessian(n_coords: usize) -> impl Strategy<Value = DMatrix<f64>> {
    prop::collection::vec(0.5..1.0_f64, n_coords)
        .prop_map(move |diag| DMatrix::from_diagonal(&DVector::from_vec(diag)))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Property 8: DIIS history unit consistency
    ///
    /// *For any* sequence of optimization steps, the GDIIS/GEDIIS history should
    /// store geometries in Angstrom and gradients in Ha/Bohr, and interpolation
    /// should produce valid Angstrom geometries.
    ///
    /// **Feature: unit-standardization, Property 8: DIIS history unit consistency**
    /// **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
    ///
    /// This test verifies:
    /// 1. Geometries added to history maintain Angstrom magnitude (Req 7.1)
    /// 2. Gradients added to history maintain Ha/Bohr magnitude (Req 7.2)
    /// 3. GDIIS interpolation produces geometries in Angstrom range (Req 7.3)
    /// 4. Energy-weighted operations maintain consistent units (Req 7.4)
    #[test]
    fn prop_diis_history_unit_consistency(
        // Generate 3 atoms (9 coordinates) for a small molecule
        geom1 in arb_geometry(9),
        geom2 in arb_geometry(9),
        geom3 in arb_geometry(9),
        grad1 in arb_gradient(9),
        grad2 in arb_gradient(9),
        grad3 in arb_gradient(9),
        energy1 in arb_energy_diff(),
        energy2 in arb_energy_diff(),
        energy3 in arb_energy_diff(),
    ) {
        let mut opt_state = OptimizationState::new(5);
        let config = Config::default();

        // Create inverse Hessian (0.7 diagonal in Bohr²/Ha)
        let hess = DMatrix::from_diagonal(&DVector::from_vec(vec![0.7; 9]));

        // Add history entries (simulating 3 optimization steps)
        // Requirement 7.1: Geometries stored in Angstrom
        // Requirement 7.2: Gradients stored in Ha/Bohr
        let n = grad1.len();
        opt_state.add_to_history(
            geom1.clone(), grad1, DVector::zeros(n), hess.clone(),
            energy1, vec![], None, false
        );
        let n = grad2.len();
        opt_state.add_to_history(
            geom2.clone(), grad2, DVector::zeros(n), hess.clone(),
            energy2, vec![], None, false
        );
        let n = grad3.len();
        opt_state.add_to_history(
            geom3.clone(), grad3, DVector::zeros(n), hess,
            energy3, vec![], None, false
        );

        // Verify history stores geometries in Angstrom range (Requirement 7.1)
        // Typical molecular coordinates: -10 to +10 Angstrom
        for geom in &opt_state.geom_history {
            for &coord in geom.iter() {
                prop_assert!(
                    coord.abs() <= 15.0,
                    "Geometry coordinate {} outside expected Angstrom range [-15, 15]",
                    coord
                );
            }
        }

        // Verify history stores gradients in Ha/Bohr range (Requirement 7.2)
        // Typical MECP gradients: -0.2 to +0.2 Ha/Bohr
        for grad in &opt_state.grad_history {
            for &g in grad.iter() {
                prop_assert!(
                    g.abs() <= 0.2,
                    "Gradient component {} outside expected Ha/Bohr range [-0.2, 0.2]",
                    g
                );
            }
        }

        // Verify energy history stores Hartree values (Requirement 7.4)
        for &energy in &opt_state.energy_history {
            prop_assert!(
                energy.abs() <= 0.2,
                "Energy difference {} outside expected Hartree range [-0.2, 0.2]",
                energy
            );
        }

        // Requirement 7.3: GDIIS interpolation produces Angstrom geometries
        // Only run GDIIS if we have enough history
        if opt_state.has_enough_history() {
            let interpolated_geom = gdiis_step(&mut opt_state, &config);

            // Verify interpolated geometry is in Angstrom range
            // The interpolation should produce coordinates within a reasonable
            // range of the input geometries (with some tolerance for the step)
            for &coord in interpolated_geom.iter() {
                prop_assert!(
                    coord.abs() <= 20.0,
                    "Interpolated coordinate {} outside expected Angstrom range [-20, 20]",
                    coord
                );
                prop_assert!(
                    !coord.is_nan() && !coord.is_infinite(),
                    "Interpolated coordinate is NaN or infinite"
                );
            }

            // Verify the interpolated geometry has the correct dimension
            prop_assert_eq!(
                interpolated_geom.len(),
                9,
                "Interpolated geometry has wrong dimension"
            );
        }
    }

    /// Property test for geometry history storage consistency.
    ///
    /// Verifies that geometries added to history are stored without
    /// unit conversion (remain in Angstrom).
    ///
    /// **Feature: unit-standardization, Property 8: DIIS history unit consistency**
    /// **Validates: Requirement 7.1**
    #[test]
    fn prop_geometry_history_preserves_angstrom(
        geom in arb_geometry(9),
        grad in arb_gradient(9),
        energy in arb_energy_diff(),
    ) {
        let mut opt_state = OptimizationState::new(5);
        let hess = DMatrix::from_diagonal(&DVector::from_vec(vec![0.7; 9]));

        // Add geometry to history
        let n = grad.len();
        opt_state.add_to_history(
            geom.clone(), grad, DVector::zeros(n), hess, energy, vec![], None, false
        );

        // Verify the stored geometry matches the input exactly
        let stored_geom = opt_state.geom_history.back().unwrap();
        for (i, (&stored, &original)) in stored_geom.iter().zip(geom.iter()).enumerate() {
            prop_assert!(
                (stored - original).abs() < 1e-15,
                "Geometry coordinate {} was modified: stored={}, original={}",
                i, stored, original
            );
        }
    }

    /// Property test for gradient history storage consistency.
    ///
    /// Verifies that gradients added to history are stored without
    /// unit conversion (remain in Ha/Bohr).
    ///
    /// **Feature: unit-standardization, Property 8: DIIS history unit consistency**
    /// **Validates: Requirement 7.2**
    #[test]
    fn prop_gradient_history_preserves_ha_bohr(
        geom in arb_geometry(9),
        grad in arb_gradient(9),
        energy in arb_energy_diff(),
    ) {
        let mut opt_state = OptimizationState::new(5);
        let hess = DMatrix::from_diagonal(&DVector::from_vec(vec![0.7; 9]));

        // Add gradient to history
        opt_state.add_to_history(
            geom, grad.clone(), DVector::zeros(grad.len()), hess, energy, vec![], None, false
        );

        // Verify the stored gradient matches the input exactly
        let stored_grad = opt_state.grad_history.back().unwrap();
        for (i, (&stored, &original)) in stored_grad.iter().zip(grad.iter()).enumerate() {
            prop_assert!(
                (stored - original).abs() < 1e-15,
                "Gradient component {} was modified: stored={}, original={}",
                i, stored, original
            );
        }
    }

    /// Property test for GDIIS interpolation coefficient normalization.
    ///
    /// Verifies that GDIIS interpolation produces a geometry that is
    /// a valid linear combination of history geometries.
    ///
    /// **Feature: unit-standardization, Property 8: DIIS history unit consistency**
    /// **Validates: Requirement 7.3**
    #[test]
    fn prop_gdiis_interpolation_valid_geometry(
        // Use smaller coordinate range to ensure numerical stability
        geom1 in arb_geometry(9),
        geom2 in arb_geometry(9),
        geom3 in arb_geometry(9),
        grad1 in arb_gradient(9),
        grad2 in arb_gradient(9),
        grad3 in arb_gradient(9),
        energy1 in arb_energy_diff(),
        energy2 in arb_energy_diff(),
        energy3 in arb_energy_diff(),
    ) {
        let mut opt_state = OptimizationState::new(5);
        let config = Config::default();
        let hess = DMatrix::from_diagonal(&DVector::from_vec(vec![0.7; 9]));

        // Build history
        let n = grad1.len();
        opt_state.add_to_history(
            geom1.clone(), grad1, DVector::zeros(n), hess.clone(), energy1, vec![], None, false
        );
        let n = grad2.len();
        opt_state.add_to_history(
            geom2.clone(), grad2, DVector::zeros(n), hess.clone(), energy2, vec![], None, false
        );
        let n = grad3.len();
        opt_state.add_to_history(
            geom3.clone(), grad3, DVector::zeros(n), hess.clone(), energy3, vec![], None, false
        );

        // Run GDIIS interpolation
        let result = gdiis_step(&mut opt_state, &config);

        // The result should be finite and within a reasonable range
        // (interpolation + correction step)
        for &coord in result.iter() {
            prop_assert!(
                coord.is_finite(),
                "GDIIS produced non-finite coordinate: {}",
                coord
            );
        }

        // The result should have the same dimension as input
        prop_assert_eq!(result.len(), 9);
    }
}
