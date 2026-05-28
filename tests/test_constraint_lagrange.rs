use nalgebra::DVector;
use omecp::constraints::{add_constraint_lagrange, validate_constraints, Constraint};
use omecp::geometry::Geometry;

#[test]
fn test_bond_constraint_lagrange() {
    // Create a simple two-atom system (H2)
    let elements = vec!["H".to_string(), "H".to_string()];
    let coords = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 1.0 Å apart
    let geometry = Geometry::new(elements, coords);

    // Create forces that would pull atoms apart
    let forces = DVector::from_vec(vec![1.0, 0.0, 0.0, -1.0, 0.0, 0.0]);

    // Constrain bond to current length (1.0 Å)
    let constraints = vec![Constraint::Bond {
        atoms: (0, 1),
        target: 1.0,
    }];

    let mut lambdas = vec![0.0];

    // Apply constraint - should reduce forces since constraint is already satisfied
    let (constrained_forces, _) =
        add_constraint_lagrange(&geometry, forces.clone(), &constraints, &mut lambdas)
            .expect("Constraint application should succeed");

    // Since constraint is already satisfied, forces should be unchanged
    assert!((constrained_forces - forces).norm() < 1e-10);
    assert!(lambdas[0].abs() < 1e-10);
}

#[test]
fn test_violated_bond_constraint() {
    // Create a two-atom system with atoms too far apart
    let elements = vec!["H".to_string(), "H".to_string()];
    let coords = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0]; // 2.0 Å apart
    let geometry = Geometry::new(elements, coords);

    // Zero initial forces
    let forces = DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    // Constrain bond to 1.5 Å (shorter than current 2.0 Å)
    let constraints = vec![Constraint::Bond {
        atoms: (0, 1),
        target: 1.5,
    }];

    let mut lambdas = vec![0.0];

    // Apply constraint
    let (constrained_forces, _) = add_constraint_lagrange(&geometry, forces, &constraints, &mut lambdas)
        .expect("Constraint application should succeed");

    // Should have attractive forces to reduce bond length
    assert!(constrained_forces[0] > 0.0); // Force on atom 0 toward +x (attractive)
    assert!(constrained_forces[3] < 0.0); // Force on atom 1 toward -x (attractive)

    // The Lagrange multiplier should be non-zero (sign depends on convention)
    assert!(lambdas[0].abs() > 1e-10);
}

#[test]
fn test_no_constraints() {
    let elements = vec!["H".to_string(), "H".to_string()];
    let coords = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let geometry = Geometry::new(elements, coords);

    let forces = DVector::from_vec(vec![1.0, 0.0, 0.0, -1.0, 0.0, 0.0]);
    let constraints = vec![];
    let mut lambdas = vec![];

    let (constrained_forces, _) =
        add_constraint_lagrange(&geometry, forces.clone(), &constraints, &mut lambdas)
            .expect("No constraints should always succeed");

    // Forces should be unchanged
    assert_eq!(constrained_forces, forces);
    assert!(lambdas.is_empty());
}

#[test]
fn test_constraint_validation() {
    // Valid constraints
    let valid_constraints = vec![
        Constraint::Bond {
            atoms: (0, 1),
            target: 1.5,
        },
        Constraint::Angle {
            atoms: (0, 1, 2),
            target: 1.57,
        }, // ~90 degrees
        Constraint::Dihedral {
            atoms: (0, 1, 2, 3),
            target: 0.0,
        },
    ];

    assert!(validate_constraints(&valid_constraints, 4).is_ok());

    // Invalid bond - atom index out of range
    let invalid_bond = vec![Constraint::Bond {
        atoms: (0, 5),
        target: 1.5,
    }];
    assert!(validate_constraints(&invalid_bond, 4).is_err());

    // Invalid bond - negative target
    let negative_bond = vec![Constraint::Bond {
        atoms: (0, 1),
        target: -1.0,
    }];
    assert!(validate_constraints(&negative_bond, 4).is_err());

    // Invalid angle - same atoms
    let invalid_angle = vec![Constraint::Angle {
        atoms: (0, 0, 1),
        target: 1.57,
    }];
    assert!(validate_constraints(&invalid_angle, 4).is_err());
}
