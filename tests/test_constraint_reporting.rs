use omecp::constraints::{report_constraint_status, Constraint};
use omecp::geometry::Geometry;

#[test]
fn test_constraint_status_reporting() {
    // Create a simple three-atom system
    let elements = vec!["H".to_string(), "H".to_string(), "H".to_string()];
    let coords = vec![
        0.0, 0.0, 0.0, // H1 at origin
        1.0, 0.0, 0.0, // H2 at (1,0,0) - 1.0 Å from H1
        2.0, 0.0, 0.0, // H3 at (2,0,0) - 1.0 Å from H2, 2.0 Å from H1
    ];
    let geometry = Geometry::new(elements, coords);

    // Create constraints
    let constraints = vec![
        Constraint::Bond {
            atoms: (0, 1),
            target: 1.0,
        }, // Should be satisfied
        Constraint::Bond {
            atoms: (1, 2),
            target: 1.5,
        }, // Should be violated (current: 1.0, target: 1.5)
        Constraint::Angle {
            atoms: (0, 1, 2),
            target: std::f64::consts::PI,
        }, // 180 degrees - should be satisfied
    ];

    let lambdas = vec![0.0, -0.25, 0.0];

    // This should print constraint status without panicking
    report_constraint_status(&geometry, &constraints, &lambdas, 1);

    // Test passes if no panic occurs
    assert!(true);
}

#[test]
fn test_constraint_status_reporting_empty() {
    // Test with no constraints
    let elements = vec!["H".to_string(), "H".to_string()];
    let coords = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let geometry = Geometry::new(elements, coords);

    let constraints = vec![];
    let lambdas = vec![];

    // Should handle empty constraints gracefully
    report_constraint_status(&geometry, &constraints, &lambdas, 1);

    // Test passes if no panic occurs
    assert!(true);
}
