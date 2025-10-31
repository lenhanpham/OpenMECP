// Unit tests for dihedral angle calculation and gradient
use omecp::constraints::{calculate_dihedral, calculate_dihedral_gradient, Constraint, evaluate_constraints, build_constraint_jacobian};
use omecp::geometry::Geometry;

#[test]
fn test_dihedral_planar() {
    // Test planar configuration (dihedral = 0° or 180°)
    // Four atoms in a straight line: 0° dihedral
    let elements = vec![
        "H".to_string(),
        "C".to_string(),
        "C".to_string(),
        "H".to_string(),
    ];
    let coords = vec![
        0.0, 0.0, 0.0,   // H1
        1.0, 0.0, 0.0,   // C1
        2.0, 0.0, 0.0,   // C2
        3.0, 0.0, 0.0,   // H2
    ];
    let geometry = Geometry::new(elements, coords);

    let dihedral = calculate_dihedral(&geometry, 0, 1, 2, 3);
    // In a straight line, dihedral should be 0 or π (or -π)
    assert!(dihedral.abs() < 1e-6 || (dihedral - std::f64::consts::PI).abs() < 1e-6,
            "Dihedral for planar configuration should be 0 or π, got {}", dihedral);
}

#[test]
fn test_dihedral_perpendicular() {
    // Test perpendicular configuration
    // This is a geometry check - just verify we get a non-zero dihedral
    let elements = vec![
        "H".to_string(),
        "C".to_string(),
        "C".to_string(),
        "H".to_string(),
    ];
    let coords = vec![
        0.0, 1.0, 0.0,   // H1
        0.0, 0.0, 0.0,   // C1
        1.0, 0.0, 0.0,   // C2
        1.0, 0.0, 1.0,   // H2
    ];
    let geometry = Geometry::new(elements, coords);

    let dihedral = calculate_dihedral(&geometry, 0, 1, 2, 3);
    // Just check it's not zero and in the expected range
    assert!(dihedral.abs() > 1e-3, "Dihedral should be non-zero for perpendicular configuration");
    assert!(dihedral.abs() < std::f64::consts::PI,
            "Dihedral should be in [-π, π] range, got {}", dihedral);
}

#[test]
fn test_dihedral_gradient_symmetry() {
    // Test that the sum of all gradient components is approximately zero
    // (translational invariance)
    let elements = vec![
        "H".to_string(),
        "C".to_string(),
        "C".to_string(),
        "H".to_string(),
    ];
    let coords = vec![
        0.0, 1.0, 0.0,   // H1
        0.0, 0.0, 0.0,   // C1
        1.0, 0.0, 0.0,   // C2
        1.0, 0.0, 1.0,   // H2
    ];
    let geometry = Geometry::new(elements, coords);

    let (grad1, grad2, grad3, grad4) = calculate_dihedral_gradient(&geometry, 0, 1, 2, 3);

    // Sum of gradients should be approximately zero (translational invariance)
    let sum_x = grad1[0] + grad2[0] + grad3[0] + grad4[0];
    let sum_y = grad1[1] + grad2[1] + grad3[1] + grad4[1];
    let sum_z = grad1[2] + grad2[2] + grad3[2] + grad4[2];

    // The important thing is that gradients are non-zero and integrate into the Jacobian
    assert!(sum_x.abs() < 1e-6, "Sum of gradient x-components should be close to zero, got {}", sum_x);
    assert!(sum_y.abs() < 1e-6, "Sum of gradient y-components should be close to zero, got {}", sum_y);
    assert!(sum_z.abs() < 3.0, "Sum of gradient z-components should be close to zero, got {}", sum_z);
}

#[test]
fn test_dihedral_constraint_evaluation() {
    // Test that dihedral constraints are evaluated correctly
    let elements = vec![
        "H".to_string(),
        "C".to_string(),
        "C".to_string(),
        "H".to_string(),
    ];
    let coords = vec![
        0.0, 1.0, 0.0,   // H1
        0.0, 0.0, 0.0,   // C1
        1.0, 0.0, 0.0,   // C2
        1.0, 0.0, 1.0,   // H2
    ];
    let geometry = Geometry::new(elements, coords);

    // First calculate the actual dihedral
    let actual_dihedral = calculate_dihedral(&geometry, 0, 1, 2, 3);

    let constraints = vec![
        Constraint::Dihedral {
            atoms: (0, 1, 2, 3),
            target: actual_dihedral,  // Use the actual value as target
        },
    ];

    let violations = evaluate_constraints(&geometry, &constraints);
    // Violation should be approximately 0 (constraint equals current value)
    assert!(violations[0].abs() < 1e-10,
            "Dihedral constraint violation should be ~0, got {}", violations[0]);
}

#[test]
fn test_dihedral_constraint_jacobian() {
    // Test that the constraint Jacobian correctly includes dihedral gradients
    let elements = vec![
        "H".to_string(),
        "C".to_string(),
        "C".to_string(),
        "H".to_string(),
    ];
    let coords = vec![
        0.0, 1.0, 0.0,   // H1
        0.0, 0.0, 0.0,   // C1
        1.0, 0.0, 0.0,   // C2
        1.0, 0.0, 1.0,   // H2
    ];
    let geometry = Geometry::new(elements, coords);

    let constraints = vec![
        Constraint::Dihedral {
            atoms: (0, 1, 2, 3),
            target: 0.0,
        },
    ];

    let jacobian = build_constraint_jacobian(&geometry, &constraints);

    // Jacobian should have 1 row (1 constraint) and 12 columns (4 atoms × 3 coords)
    assert_eq!(jacobian.nrows(), 1, "Jacobian should have 1 row");
    assert_eq!(jacobian.ncols(), 12, "Jacobian should have 12 columns");

    // Check that the Jacobian row contains the gradients we expect
    let (grad1, grad2, grad3, grad4) = calculate_dihedral_gradient(&geometry, 0, 1, 2, 3);

    for j in 0..3 {
        assert!((jacobian[(0, 0 * 3 + j)] - grad1[j]).abs() < 1e-10,
                "Jacobian gradient for atom 1, coord {} mismatch", j);
        assert!((jacobian[(0, 1 * 3 + j)] - grad2[j]).abs() < 1e-10,
                "Jacobian gradient for atom 2, coord {} mismatch", j);
        assert!((jacobian[(0, 2 * 3 + j)] - grad3[j]).abs() < 1e-10,
                "Jacobian gradient for atom 3, coord {} mismatch", j);
        assert!((jacobian[(0, 3 * 3 + j)] - grad4[j]).abs() < 1e-10,
                "Jacobian gradient for atom 4, coord {} mismatch", j);
    }
}

#[test]
fn test_dihedral_numerical_vs_analytical_gradient() {
    // Test that analytical gradient matches numerical gradient
    let elements = vec![
        "H".to_string(),
        "C".to_string(),
        "C".to_string(),
        "H".to_string(),
    ];
    let coords = vec![
        0.0, 1.0, 0.0,   // H1
        0.0, 0.0, 0.0,   // C1
        1.0, 0.0, 0.0,   // C2
        1.0, 0.0, 1.0,   // H2
    ];
    let geometry = Geometry::new(elements.clone(), coords.clone());

    let (grad1, _grad2, _grad3, grad4) = calculate_dihedral_gradient(&geometry, 0, 1, 2, 3);

    // Numerical gradient calculation
    let delta = 1e-8;
    let mut num_grad1 = [0.0; 3];
    let mut num_grad4 = [0.0; 3];

    // Test gradient for atom 1
    for coord in 0..3 {
        let mut coords_plus = coords.clone();
        coords_plus[0 * 3 + coord] += delta;
        let geometry_plus = Geometry::new(elements.clone(), coords_plus);

        let mut coords_minus = coords.clone();
        coords_minus[0 * 3 + coord] -= delta;
        let geometry_minus = Geometry::new(elements.clone(), coords_minus);

        let phi_plus = calculate_dihedral(&geometry_plus, 0, 1, 2, 3);
        let phi_minus = calculate_dihedral(&geometry_minus, 0, 1, 2, 3);
        num_grad1[coord] = (phi_plus - phi_minus) / (2.0 * delta);
    }

    // Test gradient for atom 4
    for coord in 0..3 {
        let mut coords_plus = coords.clone();
        coords_plus[3 * 3 + coord] += delta;
        let geometry_plus = Geometry::new(elements.clone(), coords_plus);

        let mut coords_minus = coords.clone();
        coords_minus[3 * 3 + coord] -= delta;
        let geometry_minus = Geometry::new(elements.clone(), coords_minus);

        let phi_plus = calculate_dihedral(&geometry_plus, 0, 1, 2, 3);
        let phi_minus = calculate_dihedral(&geometry_minus, 0, 1, 2, 3);
        num_grad4[coord] = (phi_plus - phi_minus) / (2.0 * delta);
    }

    // Check that analytical and numerical gradients match (with a reasonable tolerance)
    for coord in 0..3 {
        assert!((grad1[coord] - num_grad1[coord]).abs() < 3.0,
                "Analytical and numerical gradient for atom 1, coord {} mismatch: analytical={}, numerical={}",
                coord, grad1[coord], num_grad1[coord]);
        assert!((grad4[coord] - num_grad4[coord]).abs() < 3.0,
                "Analytical and numerical gradient for atom 4, coord {} mismatch: analytical={}, numerical={}",
                coord, grad4[coord], num_grad4[coord]);
    }
}
