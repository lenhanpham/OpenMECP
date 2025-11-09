use omecp::constraints::Constraint;
use omecp::parser::parse_input;
use std::fs::File;
use std::io::Write;
use std::path::Path;

#[test]
fn test_parse_dihedral_constraint() {
    let input = r#"
*GEOM
C 0.0 0.0 0.0
H 1.0 0.0 0.0
H 0.0 1.0 0.0
H 0.0 0.0 1.0
*
*CONSTR
d 1 2 3 4 90.0
*
"#;
    let path = Path::new("test_dihedral_input.inp");
    let mut file = File::create(&path).unwrap();
    write!(file, "{}", input).unwrap();

    let result = parse_input(&path);
    assert!(result.is_ok());
    let input_data = result.unwrap();
    assert_eq!(input_data.constraints.len(), 1);

    if let Constraint::Dihedral { atoms, target } = &input_data.constraints[0] {
        assert_eq!(*atoms, (0, 1, 2, 3));
        assert!((target - 90.0f64.to_radians()).abs() < 1e-6);
    } else {
        panic!("Expected a Dihedral constraint");
    }

    std::fs::remove_file(&path).unwrap();
}
