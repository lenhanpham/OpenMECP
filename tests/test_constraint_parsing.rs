use omecp::constraints::Constraint;
use omecp::parser::parse_input;
use std::fs::File;
use std::io::Write;
use std::path::Path;

#[test]
fn test_enhanced_bond_constraint_parsing() {
    let input = r#"
*GEOM
C 0.0 0.0 0.0
H 1.0 0.0 0.0
*
*CONSTR
r 1 2 1.5
*
"#;
    let path = Path::new("test_bond_parsing.inp");
    let mut file = File::create(&path).unwrap();
    write!(file, "{}", input).unwrap();

    let result = parse_input(&path);
    assert!(result.is_ok());
    let input_data = result.unwrap();
    assert_eq!(input_data.constraints.len(), 1);

    // Check constraint (r format)
    if let Constraint::Bond { atoms, target } = &input_data.constraints[0] {
        assert_eq!(*atoms, (0, 1));
        assert!((target - 1.5).abs() < 1e-6);
    } else {
        panic!("Expected a Bond constraint");
    }

    std::fs::remove_file(&path).unwrap();
}

#[test]
fn test_enhanced_angle_constraint_parsing() {
    let input = r#"
*GEOM
C 0.0 0.0 0.0
H 1.0 0.0 0.0
H 0.0 1.0 0.0
*
*CONSTR
a 1 2 3 120.0
*
"#;
    let path = Path::new("test_angle_parsing.inp");
    let mut file = File::create(&path).unwrap();
    write!(file, "{}", input).unwrap();

    let result = parse_input(&path);
    assert!(result.is_ok());
    let input_data = result.unwrap();
    assert_eq!(input_data.constraints.len(), 1);

    // Check constraint
    if let Constraint::Angle { atoms, target } = &input_data.constraints[0] {
        assert_eq!(*atoms, (0, 1, 2));
        assert!((target - (120.0 * std::f64::consts::PI / 180.0)).abs() < 1e-6);
    } else {
        panic!("Expected an Angle constraint");
    }

    std::fs::remove_file(&path).unwrap();
}

#[test]
fn test_constraint_parsing_errors() {
    // Test insufficient parameters
    let input = r#"
*GEOM
C 0.0 0.0 0.0
H 1.0 0.0 0.0
*
*CONSTR
r 1 2
*
"#;
    let path = Path::new("test_error_parsing.inp");
    let mut file = File::create(&path).unwrap();
    write!(file, "{}", input).unwrap();

    let result = parse_input(&path);
    assert!(result.is_err());
    if let Err(error) = result {
        assert!(error
            .to_string()
            .contains("Bond constraint requires 4 parameters"));
    }

    std::fs::remove_file(&path).unwrap();
}

#[test]
fn test_constraint_atom_index_validation() {
    // Test atom index out of bounds
    let input = r#"
*GEOM
C 0.0 0.0 0.0
H 1.0 0.0 0.0
*
*CONSTR
r 1 5 1.5
*
"#;
    let path = Path::new("test_index_validation.inp");
    let mut file = File::create(&path).unwrap();
    write!(file, "{}", input).unwrap();

    let result = parse_input(&path);
    assert!(result.is_err());
    if let Err(error) = result {
        assert!(error.to_string().contains("exceeds number of atoms"));
    }

    std::fs::remove_file(&path).unwrap();
}

#[test]
fn test_constraint_duplicate_atoms() {
    // Test same atom in bond constraint
    let input = r#"
*GEOM
C 0.0 0.0 0.0
H 1.0 0.0 0.0
*
*CONSTR
r 1 1 1.5
*
"#;
    let path = Path::new("test_duplicate_atoms.inp");
    let mut file = File::create(&path).unwrap();
    write!(file, "{}", input).unwrap();

    let result = parse_input(&path);
    assert!(result.is_err());
    if let Err(error) = result {
        assert!(error
            .to_string()
            .contains("cannot have the same atom twice"));
    }

    std::fs::remove_file(&path).unwrap();
}

#[test]
fn test_constraint_invalid_distance() {
    // Test negative distance
    let input = r#"
*GEOM
C 0.0 0.0 0.0
H 1.0 0.0 0.0
*
*CONSTR
r 1 2 -1.5
*
"#;
    let path = Path::new("test_invalid_distance.inp");
    let mut file = File::create(&path).unwrap();
    write!(file, "{}", input).unwrap();

    let result = parse_input(&path);
    assert!(result.is_err());
    if let Err(error) = result {
        assert!(error
            .to_string()
            .contains("Distance target must be positive"));
    }

    std::fs::remove_file(&path).unwrap();
}

#[test]
fn test_constraint_invalid_angle() {
    // Test angle out of range
    let input = r#"
*GEOM
C 0.0 0.0 0.0
H 1.0 0.0 0.0
H 0.0 1.0 0.0
*
*CONSTR
a 1 2 3 400.0
*
"#;
    let path = Path::new("test_invalid_angle.inp");
    let mut file = File::create(&path).unwrap();
    write!(file, "{}", input).unwrap();

    let result = parse_input(&path);
    assert!(result.is_err());
    if let Err(error) = result {
        assert!(error.to_string().contains("outside valid range [0, 360]"));
    }

    std::fs::remove_file(&path).unwrap();
}

#[test]
fn test_constraint_unknown_type() {
    // Test unknown constraint type
    let input = r#"
*GEOM
C 0.0 0.0 0.0
H 1.0 0.0 0.0
*
*CONSTR
x 1 2 1.5
*
"#;
    let path = Path::new("test_unknown_type.inp");
    let mut file = File::create(&path).unwrap();
    write!(file, "{}", input).unwrap();

    let result = parse_input(&path);
    assert!(result.is_err());
    if let Err(error) = result {
        assert!(error.to_string().contains("Unknown constraint type 'x'"));
    }

    std::fs::remove_file(&path).unwrap();
}

#[test]
fn test_constraint_duplicate_detection() {
    // Test duplicate constraint detection
    let input = r#"
*GEOM
C 0.0 0.0 0.0
H 1.0 0.0 0.0
*
*CONSTR
r 1 2 1.5
r 2 1 1.2
*
"#;
    let path = Path::new("test_duplicate_constraints.inp");
    let mut file = File::create(&path).unwrap();
    write!(file, "{}", input).unwrap();

    let result = parse_input(&path);
    assert!(result.is_err());
    if let Err(error) = result {
        assert!(error.to_string().contains("Duplicate constraints found"));
    }

    std::fs::remove_file(&path).unwrap();
}

#[test]
fn test_constraint_comments_and_empty_lines() {
    // Test that comments and empty lines are handled correctly
    let input = r#"
*GEOM
C 0.0 0.0 0.0
H 1.0 0.0 0.0
*
*CONSTR
# This is a comment
r 1 2 1.5  # Bond constraint

# Another comment
*
"#;
    let path = Path::new("test_comments.inp");
    let mut file = File::create(&path).unwrap();
    write!(file, "{}", input).unwrap();

    let result = parse_input(&path);
    assert!(result.is_ok());
    let input_data = result.unwrap();
    assert_eq!(input_data.constraints.len(), 1);

    if let Constraint::Bond { atoms, target } = &input_data.constraints[0] {
        assert_eq!(*atoms, (0, 1));
        assert!((target - 1.5).abs() < 1e-6);
    } else {
        panic!("Expected a Bond constraint");
    }

    std::fs::remove_file(&path).unwrap();
}

#[test]
fn test_zero_based_atom_indexing_error() {
    // Test that 0-based indexing is rejected
    let input = r#"
*GEOM
C 0.0 0.0 0.0
H 1.0 0.0 0.0
*
*CONSTR
r 0 1 1.5
*
"#;
    let path = Path::new("test_zero_indexing.inp");
    let mut file = File::create(&path).unwrap();
    write!(file, "{}", input).unwrap();

    let result = parse_input(&path);
    assert!(result.is_err());
    if let Err(error) = result {
        assert!(error.to_string().contains("must be 1-based"));
    }

    std::fs::remove_file(&path).unwrap();
}
