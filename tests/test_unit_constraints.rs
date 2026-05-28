use nalgebra::DVector;
use omecp::constraints::{add_constraint_lagrange, Constraint};
use omecp::geometry::Geometry;

#[test]
fn test_compatible_constraint_initialization() {
    let elements = vec!["H".to_string(), "H".to_string()];
    let coords = vec![0.0, 0.0, 0.0, 1.6, 0.0, 0.0]; // 1.6 Angstrom bond
    let geometry = Geometry::new(elements, coords);

    let constraints = vec![Constraint::Bond {
        atoms: (0, 1),
        target: 1.5,
    }];

    let forces = DVector::from_vec(vec![0.1, 0.0, 0.0, -0.1, 0.0, 0.0]);
    let mut lambdas = vec![0.0];

    let (_constrained_forces, violations) =
        add_constraint_lagrange(&geometry, forces.clone(), &constraints, &mut lambdas).unwrap();

    assert!(lambdas[0] != 0.0, "λ should be initialized on first step");

    let old_lambda = lambdas[0];
    let (_constrained_forces2, _violations2) =
        add_constraint_lagrange(&geometry, forces, &constraints, &mut lambdas).unwrap();

    assert_eq!(old_lambda, lambdas[0], "λ should be reused on subsequent steps");
    assert!(violations.len() == 1, "Should have one violation");
    assert!(violations[0].abs() > 1e-10, "Should detect bond length violation");
}

#[test]
fn test_constraint_violations_returned() {
    let elements = vec!["H".to_string(), "H".to_string()];
    let coords = vec![0.0, 0.0, 0.0, 1.5, 0.0, 0.0]; // Perfect bond length
    let geometry = Geometry::new(elements, coords);

    let constraints = vec![Constraint::Bond {
        atoms: (0, 1),
        target: 1.5,
    }];

    let forces = DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let mut lambdas = vec![0.0];

    let (_, violations) =
        add_constraint_lagrange(&geometry, forces, &constraints, &mut lambdas).unwrap();

    assert!(violations[0].abs() < 1e-10, "Perfect bond should have minimal violation");
}
