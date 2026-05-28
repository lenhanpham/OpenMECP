use nalgebra::DVector;
use omecp::constraints::{add_constraint_lagrange, Constraint};
use omecp::geometry::Geometry;
use omecp::geometry::State;
use omecp::optimizer::{compute_mecp_gradient, OptimizationState};

#[test]
fn test_constraint_integration_with_mecp_gradient() {
    // Create a simple three-atom system (H-H-H linear)
    let elements = vec!["H".to_string(), "H".to_string(), "H".to_string()];
    let coords = vec![
        0.0, 0.0, 0.0, // H1 at origin
        1.0, 0.0, 0.0, // H2 at (1,0,0)
        2.0, 0.0, 0.0, // H3 at (2,0,0)
    ];
    let geometry = Geometry::new(elements, coords);

    // Create mock states with different energies and forces
    let state_a = State {
        energy: -1.0,
        forces: DVector::from_vec(vec![
            0.1, 0.0, 0.0, // Force on H1
            0.0, 0.0, 0.0, // Force on H2
            -0.1, 0.0, 0.0, // Force on H3
        ]),
        geometry: geometry.clone(),
    };

    let state_b = State {
        energy: -0.9, // Slightly higher energy
        forces: DVector::from_vec(vec![
            -0.05, 0.0, 0.0, // Force on H1
            0.0, 0.0, 0.0, // Force on H2
            0.05, 0.0, 0.0, // Force on H3
        ]),
        geometry: geometry.clone(),
    };

    let fixed_atoms = vec![];

    // Compute MECP gradient without constraints (decomposed)
    let mecp_grad = compute_mecp_gradient(&state_a, &state_b, &fixed_atoms);

    // Create a bond constraint to keep H1-H2 distance at 1.0 Å
    let constraints = vec![Constraint::Bond {
        atoms: (0, 1),
        target: 1.0,
    }];

    let mut opt_state = OptimizationState::new(5);

    // Apply constraint forces to g_vec only (pure Ha/Å — consistent with
    // the new unit-standardized design where constraints act on gradients)
    let (constrained_grad, _) = add_constraint_lagrange(
        &geometry,
        mecp_grad.g_vec.clone(),
        &constraints,
        &mut opt_state.lambdas,
    )
    .expect("Constraint application should succeed");

    // Since the constraint is already satisfied (H1-H2 distance is 1.0 Å),
    // the gradient should be essentially unchanged
    let grad_diff = (&constrained_grad - &mecp_grad.g_vec).norm();
    assert!(
        grad_diff < 1e-10,
        "Gradient should be unchanged when constraint is satisfied"
    );

    // Lagrange multiplier should be near zero
    assert_eq!(opt_state.lambdas.len(), 1);
    assert!(
        opt_state.lambdas[0].abs() < 1e-10,
        "Lagrange multiplier should be near zero"
    );
}

#[test]
fn test_constraint_integration_with_violated_constraint() {
    // Use 3 atoms so the g-vector (perpendicular to gradient difference)
    // can be non-zero. Multiple force components in different directions
    // ensure g_vec ≠ 0.
    let elements = vec!["O".to_string(), "H".to_string(), "H".to_string()];
    let coords = vec![
        0.0, 0.0, 0.0,  // O at origin
        1.0, 0.0, 0.0,  // H1 along +x
        0.0, 1.0, 0.0,  // H2 along +y
    ];
    let geometry = Geometry::new(elements, coords);

    // Both states must have forces for g_vec ≠ 0.
    // State A: H1 feels +y force. State B: H1 feels +x force.
    // The two gradient directions differ, so the perpendicular
    // component (g_vec) is non-zero for H1.
    let state_a = State {
        energy: -1.0,
        forces: DVector::from_vec(vec![
            0.0, 0.0, 0.0,  // O: no forces
            1.0, 0.0, 0.0,  // H1: pushed +x
            0.0, 0.0, 0.0,  // H2: no forces
        ]),
        geometry: geometry.clone(),
    };

    let state_b = State {
        energy: -1.0,
        forces: DVector::from_vec(vec![
            0.0, 0.0, 0.0,  // O: no forces
            0.0, 1.0, 0.0,  // H1: pushed +y (different from state A)
            0.0, 0.0, 0.0,  // H2: no forces
        ]),
        geometry: geometry.clone(),
    };

    let fixed_atoms = vec![];

    // Compute MECP gradient (g_vec should be non-zero)
    let mecp_grad = compute_mecp_gradient(&state_a, &state_b, &fixed_atoms);

    // Create a bond constraint: keep O-H1 at 0.96 Å (currently 1.0 Å)
    let constraints = vec![Constraint::Bond {
        atoms: (0, 1),
        target: 0.96,
    }];

    let mut opt_state = OptimizationState::new(5);

    // Apply constraint forces to the gradient component (g_vec)
    let (constrained_grad, _) = add_constraint_lagrange(
        &geometry,
        mecp_grad.g_vec.clone(),
        &constraints,
        &mut opt_state.lambdas,
    )
    .expect("Constraint application should succeed");

    // The g_vec should have attractive forces to reduce bond length
    assert!(
        constrained_grad[0] > 0.0,
        "Force on H1 should be toward +x (attractive): got {}",
        constrained_grad[0]
    );
    assert!(
        constrained_grad[3] < 0.0,
        "Force on H2 should be toward -x (attractive): got {}",
        constrained_grad[3]
    );

    // Lagrange multiplier should be non-zero
    assert_eq!(opt_state.lambdas.len(), 1);
    assert!(
        opt_state.lambdas[0].abs() > 1e-10,
        "Lagrange multiplier should be non-zero"
    );
}

#[test]
fn test_optimization_state_lagrange_multipliers() {
    // Test that OptimizationState properly stores Lagrange multipliers
    let mut opt_state = OptimizationState::new(5);

    // Initially should be empty
    assert!(opt_state.lambdas.is_empty());

    // Create a simple constraint system
    let elements = vec!["H".to_string(), "H".to_string()];
    let coords = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let geometry = Geometry::new(elements, coords);

    let forces = DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let constraints = vec![Constraint::Bond {
        atoms: (0, 1),
        target: 1.0,
    }];

    // Apply constraints
    let _constrained_forces =
        add_constraint_lagrange(&geometry, forces, &constraints, &mut opt_state.lambdas)
            .expect("Should succeed");

    // Check that lambdas were updated
    assert_eq!(opt_state.lambdas.len(), 1);
}
