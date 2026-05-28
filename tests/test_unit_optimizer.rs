use nalgebra::DVector;
use omecp::geometry::{Geometry, State};
use omecp::optimizer::compute_mecp_gradient;

#[test]
fn test_force_sign_convention() {
    let elements = vec!["H".to_string(), "H".to_string()];
    let coords = vec![
        0.0, 0.0, 0.0, // Atom 1 at origin
        1.0, 0.0, 0.0, // Atom 2 at (1,0,0)
    ];
    let geometry = Geometry::new(elements, coords);

    let forces1 = DVector::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    let forces2 = DVector::from_vec(vec![0.2, 0.3, 0.4, 0.5, 0.6, 0.7]);

    let state_a = State {
        geometry: geometry.clone(),
        energy: -100.0,
        forces: forces1,
    };

    let state_b = State {
        geometry,
        energy: -99.0,
        forces: forces2,
    };

    let gradient = compute_mecp_gradient(&state_a, &state_b, &[]);

    assert_eq!(gradient.combined.len(), 6);
    assert!(gradient.combined.norm() > 1e-10);

    let expected_f1 = -state_a.forces;
    let expected_f2 = -state_b.forces;

    let x_vec = &expected_f1 - &expected_f2;
    let x_norm = if x_vec.norm().abs() < 1e-10 {
        let n = x_vec.len() as f64;
        &x_vec / (n.sqrt() * 1e-10)
    } else {
        &x_vec / x_vec.norm()
    };

    let de = state_a.energy - state_b.energy;
    let expected_f_vec = x_norm.clone() * de;
    let dot = expected_f1.dot(&x_norm);
    let expected_g_vec = &expected_f1 - &x_norm * dot;
    let expected_combined = expected_f_vec + expected_g_vec;

    for i in 0..gradient.combined.len() {
        assert!(
            (gradient.combined[i] - expected_combined[i]).abs() < 1e-10,
            "Gradient component {} mismatch: {} vs {}",
            i,
            gradient.combined[i],
            expected_combined[i]
        );
    }
}

#[test]
fn test_force_negation_impact() {
    let elements = vec!["H".to_string(), "H".to_string()];
    let coords = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let geometry = Geometry::new(elements, coords);

    let forces = DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    let state_a = State {
        geometry: geometry.clone(),
        energy: -100.0,
        forces: forces.clone(),
    };

    let state_b = State {
        geometry,
        energy: -100.0,
        forces: forces.clone(),
    };

    let gradient = compute_mecp_gradient(&state_a, &state_b, &[]);

    assert!(
        gradient.combined.norm() < 1e-10,
        "Expected zero gradient with zero forces, got {}",
        gradient.combined.norm()
    );
}
