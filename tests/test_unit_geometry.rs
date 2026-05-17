use nalgebra::DVector;
use omecp::geometry::{Geometry, State};

#[test]
fn test_state_validation_valid_state() {
    let geometry = Geometry::new(vec!["H".to_string()], vec![0.0, 0.0, 0.0]);
    let valid_state = State {
        energy: -0.5,
        forces: DVector::from_vec(vec![0.1, -0.2, 0.0]),
        geometry,
    };

    assert!(valid_state.validate().is_ok());
}

#[test]
fn test_state_validation_zero_energy() {
    let geometry = Geometry::new(vec!["H".to_string()], vec![0.0, 0.0, 0.0]);
    let invalid_state = State {
        energy: 0.0,
        forces: DVector::from_vec(vec![0.1, -0.2, 0.0]),
        geometry,
    };

    let result = invalid_state.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("zero energy"));
}

#[test]
fn test_state_validation_zero_forces() {
    let geometry = Geometry::new(vec!["H".to_string()], vec![0.0, 0.0, 0.0]);
    let invalid_state = State {
        energy: -0.5,
        forces: DVector::from_vec(vec![0.0, 0.0, 0.0]),
        geometry,
    };

    let result = invalid_state.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("all-zero forces"));
}

#[test]
fn test_state_validation_empty_forces() {
    let geometry = Geometry::new(vec!["H".to_string()], vec![0.0, 0.0, 0.0]);
    let invalid_state = State {
        energy: -0.5,
        forces: DVector::from_vec(vec![]),
        geometry,
    };

    let result = invalid_state.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("empty forces"));
}

#[test]
fn test_state_validation_force_geometry_mismatch() {
    let geometry = Geometry::new(
        vec!["H".to_string(), "H".to_string()],
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    );
    let invalid_state = State {
        energy: -0.5,
        forces: DVector::from_vec(vec![0.1, -0.2, 0.0]), // Only 3 components for 2 atoms
        geometry,
    };

    let result = invalid_state.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Force/geometry mismatch"));
}
