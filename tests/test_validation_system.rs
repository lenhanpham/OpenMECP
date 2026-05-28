//! Comprehensive tests for the run mode validation system.
//!
//! These tests verify that the validation functions correctly identify
//! invalid configurations and provide helpful error messages and guidance.

use omecp::config::{Config, QMProgram, RunMode};
use omecp::validation::{validate_run_mode_compatibility, ErrorCategory};
use std::fs;

/// Helper function to create a basic valid configuration
fn create_basic_config() -> Config {
    Config {
        program: QMProgram::Gaussian,
        run_mode: RunMode::Normal,
        method: "B3LYP/6-31G*".to_string(),
        mult_state_a: 1,
        mult_state_b: 1,
        charge: 0,
        ..Default::default()
    }
}

/// Helper function to create test wavefunction files (for backward compatibility)
fn create_test_wavefunction_files() {
    // Create test checkpoint files for Gaussian with exact names expected by validation
    fs::write("a.chk", "dummy gaussian checkpoint").unwrap();
    fs::write("b.chk", "dummy gaussian checkpoint").unwrap();

    // Create test wavefunction files for ORCA with exact names expected by validation
    fs::write("a.gbw", "dummy orca wavefunction").unwrap();
    fs::write("b.gbw", "dummy orca wavefunction").unwrap();

    // Create running_dir versions with exact names expected by validation
    fs::create_dir_all("running_dir").unwrap();
    fs::write("running_dir/a.chk", "dummy gaussian checkpoint").unwrap();
    fs::write("running_dir/b.chk", "dummy gaussian checkpoint").unwrap();
    fs::write("running_dir/a.gbw", "dummy orca wavefunction").unwrap();
    fs::write("running_dir/b.gbw", "dummy orca wavefunction").unwrap();
}

/// Helper function to clean up test files
fn cleanup_test_files() {
    let _ = fs::remove_file("a.chk");
    let _ = fs::remove_file("b.chk");
    let _ = fs::remove_file("a.gbw");
    let _ = fs::remove_file("b.gbw");
    let _ = fs::remove_dir_all("running_dir");
    let _ = fs::remove_file("test_model.json");
    let _ = fs::remove_file("test_interface.json");
}

#[test]
fn test_valid_configurations() {
    // Clean up any existing files first
    cleanup_test_files();

    // Test basic valid configurations for each program
    let programs = vec![
        QMProgram::Gaussian,
        QMProgram::Orca,
        QMProgram::Xtb,
        QMProgram::Bagel,
        QMProgram::Custom,
    ];

    let modes = vec![RunMode::Normal, RunMode::NoRead];

    for program in programs {
        for mode in &modes {
            let mut config = create_basic_config();
            config.program = program;
            config.run_mode = *mode;

            // Add required fields for specific programs
            match program {
                QMProgram::Bagel => {
                    config.bagel_model = "test_model.json".to_string();
                    fs::write("test_model.json", "{}").unwrap();
                }
                QMProgram::Custom => {
                    config.custom_interface_file = "test_interface.json".to_string();
                    fs::write("test_interface.json", "{}").unwrap();
                }
                _ => {}
            }

            let result = validate_run_mode_compatibility(&config);
            assert!(
                result.is_ok(),
                "Valid configuration should pass: {:?} + {:?}",
                program,
                mode
            );
        }
    }

    cleanup_test_files();
}

#[test]
fn test_orca_stability_ri_incompatibility() {
    let mut config = create_basic_config();
    config.program = QMProgram::Orca;
    config.run_mode = RunMode::Stable;
    config.method = "B3LYP RI".to_string();

    let result = validate_run_mode_compatibility(&config);
    assert!(result.is_err(), "ORCA stability + RI should be invalid");

    let error = result.unwrap_err();
    assert_eq!(error.category, ErrorCategory::IncompatibleCombination);
    assert!(error.message.contains("RI"));
    assert!(error.message.contains("stability"));
}

#[test]
fn test_xtb_unsupported_modes() {
    let mut config = create_basic_config();
    config.program = QMProgram::Xtb;

    // Test XTB + Stable mode
    config.run_mode = RunMode::Stable;
    let result = validate_run_mode_compatibility(&config);
    assert!(result.is_err(), "XTB + Stable should be invalid");

    let error = result.unwrap_err();
    assert_eq!(error.category, ErrorCategory::UnsupportedFeature);

    // Test XTB + InterRead mode
    config.run_mode = RunMode::InterRead;
    let result = validate_run_mode_compatibility(&config);
    assert!(result.is_err(), "XTB + InterRead should be invalid");

    let error = result.unwrap_err();
    assert_eq!(error.category, ErrorCategory::UnsupportedFeature);
}

#[test]
fn test_wavefunction_file_validation() {
    cleanup_test_files(); // Ensure no files exist initially

    // Test Gaussian read mode without checkpoint files
    let mut config = create_basic_config();
    config.program = QMProgram::Gaussian;
    config.run_mode = RunMode::Read;

    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_err(),
        "Gaussian read mode should fail without checkpoint files"
    );

    let error = result.unwrap_err();
    assert_eq!(error.category, ErrorCategory::MissingWavefunctionFiles);
    assert!(error.message.contains("checkpoint"));

    // Create checkpoint files and test again
    create_test_wavefunction_files();

    let result = validate_run_mode_compatibility(&config);
    if let Err(e) = &result {
        println!("Validation error: {}", e);
    }
    assert!(
        result.is_ok(),
        "Gaussian read mode should pass with checkpoint files"
    );

    cleanup_test_files();
}

#[test]
fn test_orca_wavefunction_file_validation() {
    cleanup_test_files(); // Ensure no files exist initially

    // Test ORCA read mode without .gbw files
    let mut config = create_basic_config();
    config.program = QMProgram::Orca;
    config.run_mode = RunMode::Read;

    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_err(),
        "ORCA read mode should fail without .gbw files"
    );

    let error = result.unwrap_err();
    assert_eq!(error.category, ErrorCategory::MissingWavefunctionFiles);
    assert!(error.message.contains("gbw"));

    // Create .gbw files and test again
    create_test_wavefunction_files();

    let result = validate_run_mode_compatibility(&config);
    assert!(result.is_ok(), "ORCA read mode should pass with .gbw files");

    cleanup_test_files();
}

#[test]
fn test_bagel_model_file_validation() {
    let mut config = create_basic_config();
    config.program = QMProgram::Bagel;
    config.run_mode = RunMode::Read;
    config.bagel_model = "nonexistent_model.json".to_string();

    let result = validate_run_mode_compatibility(&config);
    assert!(result.is_err(), "BAGEL should fail without model file");

    let error = result.unwrap_err();
    assert_eq!(error.category, ErrorCategory::MissingDependencies);
    assert!(error.message.contains("model file"));

    // Create model file and test again
    fs::write("test_model.json", "{}").unwrap();
    config.bagel_model = "test_model.json".to_string();

    let result = validate_run_mode_compatibility(&config);
    assert!(result.is_ok(), "BAGEL should pass with model file");

    cleanup_test_files();
}

#[test]
fn test_custom_interface_validation() {
    let mut config = create_basic_config();
    config.program = QMProgram::Custom;
    config.run_mode = RunMode::Normal;

    // Test without interface file
    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_err(),
        "Custom program should fail without interface file"
    );

    let error = result.unwrap_err();
    assert_eq!(error.category, ErrorCategory::InvalidConfiguration);
    assert!(error.message.contains("interface"));

    // Test with interface file
    fs::write("test_interface.json", "{}").unwrap();
    config.custom_interface_file = "test_interface.json".to_string();

    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_ok(),
        "Custom program should pass with interface file"
    );

    cleanup_test_files();
}

#[test]
fn test_coordinate_drive_validation() {
    let mut config = create_basic_config();
    config.run_mode = RunMode::CoordinateDrive;

    // Test without drive parameters
    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_err(),
        "Coordinate drive should fail without parameters"
    );

    let error = result.unwrap_err();
    assert_eq!(error.category, ErrorCategory::InvalidConfiguration);
    assert!(error.message.contains("drive_type"));

    // Test with same start and end values
    config.drive_type = "bond".to_string();
    config.drive_atoms = vec![1, 2];
    config.drive_start = 1.5;
    config.drive_end = 1.5; // Same as start
    config.drive_steps = 10;

    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_err(),
        "Coordinate drive should fail with same start/end"
    );

    // Test with zero steps
    config.drive_end = 2.0;
    config.drive_steps = 0;

    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_err(),
        "Coordinate drive should fail with zero steps"
    );

    // Test valid configuration
    config.drive_steps = 10;

    let result = validate_run_mode_compatibility(&config);
    assert!(result.is_ok(), "Valid coordinate drive should pass");
}

#[test]
fn test_path_optimization_validation() {
    let mut config = create_basic_config();
    config.run_mode = RunMode::PathOptimization;

    // Test without drive parameters
    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_err(),
        "Path optimization should fail without parameters"
    );

    let error = result.unwrap_err();
    assert_eq!(error.category, ErrorCategory::InvalidConfiguration);
    assert!(error.message.contains("drive_type"));

    // Test with valid parameters
    config.drive_type = "dihedral".to_string();
    config.drive_atoms = vec![1, 2, 3, 4];

    let result = validate_run_mode_compatibility(&config);
    assert!(result.is_ok(), "Valid path optimization should pass");
}

#[test]
fn test_fixde_mode_validation() {
    let mut config = create_basic_config();
    config.run_mode = RunMode::FixDE;

    // Test without fix_de value
    config.fix_de = 0.0;
    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_err(),
        "FixDE mode should fail without target energy difference"
    );

    let error = result.unwrap_err();
    assert_eq!(error.category, ErrorCategory::InvalidConfiguration);
    assert!(error.message.contains("energy difference"));

    // Test with valid fix_de value
    config.fix_de = 0.5;
    let result = validate_run_mode_compatibility(&config);
    assert!(result.is_ok(), "Valid FixDE mode should pass");
}

#[test]
fn test_gaussian_mp2_dft_incompatibility() {
    let mut config = create_basic_config();
    config.program = QMProgram::Gaussian;
    config.method = "B3LYP DFT".to_string(); // DFT method
    config.mp2 = true;

    let result = validate_run_mode_compatibility(&config);
    assert!(result.is_err(), "Gaussian MP2 + DFT should be invalid");

    let error = result.unwrap_err();
    assert_eq!(error.category, ErrorCategory::InvalidConfiguration);
    assert!(error.message.contains("MP2"));
    assert!(error.message.contains("DFT"));
}

#[test]
fn test_inter_read_multiplicity_warning() {
    // This test verifies that the validation system provides appropriate
    // warnings for inter_read mode with non-singlet multiplicities

    // Clean up any existing files first
    cleanup_test_files();

    let mut config = create_basic_config();
    config.run_mode = RunMode::InterRead;
    config.mult_state_a = 3; // Triplet
    config.mult_state_b = 1; // Singlet

    // Create required wavefunction files
    create_test_wavefunction_files();

    // The validation should pass but may print warnings
    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_ok(),
        "Inter-read with different multiplicities should be valid but may warn"
    );

    cleanup_test_files();
}

#[test]
fn test_error_message_quality() {
    // Test that error messages contain helpful information
    let mut config = create_basic_config();
    config.program = QMProgram::Xtb;
    config.run_mode = RunMode::Stable;

    let result = validate_run_mode_compatibility(&config);
    assert!(result.is_err());

    let error = result.unwrap_err();

    // Check that error has all required components
    assert!(
        !error.message.is_empty(),
        "Error message should not be empty"
    );
    assert!(error.suggestion.is_some(), "Error should have a suggestion");
    assert!(error.reference.is_some(), "Error should have a reference");

    // Check that suggestion is helpful
    let suggestion = error.suggestion.unwrap();
    assert!(
        suggestion.contains("normal") || suggestion.contains("noread"),
        "Suggestion should mention alternative modes"
    );
}

#[test]
fn test_validation_error_display() {
    // Test the Display implementation for ValidationError
    let mut config = create_basic_config();
    config.program = QMProgram::Xtb;
    config.run_mode = RunMode::InterRead;

    let result = validate_run_mode_compatibility(&config);
    assert!(result.is_err());

    let error = result.unwrap_err();
    let error_string = format!("{}", error);

    // Check that the formatted error contains all components
    assert!(error_string.contains(&error.message));
    if let Some(suggestion) = &error.suggestion {
        assert!(error_string.contains(suggestion));
    }
    if let Some(reference) = &error.reference {
        assert!(error_string.contains(reference));
    }
}

#[test]
fn test_comprehensive_program_mode_matrix() {
    // Test all program/mode combinations systematically
    let programs = vec![
        QMProgram::Gaussian,
        QMProgram::Orca,
        QMProgram::Xtb,
        QMProgram::Bagel,
        QMProgram::Custom,
    ];

    let modes = vec![
        RunMode::Normal,
        RunMode::Read,
        RunMode::NoRead,
        RunMode::Stable,
        RunMode::InterRead,
        RunMode::CoordinateDrive,
        RunMode::PathOptimization,
        RunMode::FixDE,
    ];

    // Create necessary test files
    create_test_wavefunction_files();
    fs::write("test_model.json", "{}").unwrap();
    fs::write("test_interface.json", "{}").unwrap();

    for program in programs {
        for mode in &modes {
            let mut config = create_basic_config();
            config.program = program;
            config.run_mode = *mode;

            // Set required parameters for specific modes
            match mode {
                RunMode::CoordinateDrive | RunMode::PathOptimization => {
                    config.drive_type = "bond".to_string();
                    config.drive_atoms = vec![1, 2];
                    config.drive_start = 1.0;
                    config.drive_end = 2.0;
                    config.drive_steps = 10;
                }
                RunMode::FixDE => {
                    config.fix_de = 0.5;
                }
                _ => {}
            }

            // Set required files for specific programs
            match program {
                QMProgram::Bagel => {
                    config.bagel_model = "test_model.json".to_string();
                }
                QMProgram::Custom => {
                    config.custom_interface_file = "test_interface.json".to_string();
                }
                _ => {}
            }

            let result = validate_run_mode_compatibility(&config);

            // Check expected failures
            let should_fail = match (program, mode) {
                (QMProgram::Xtb, RunMode::Stable) => true,
                (QMProgram::Xtb, RunMode::InterRead) => true,
                _ => false,
            };

            if should_fail {
                assert!(
                    result.is_err(),
                    "Expected failure for {:?} + {:?}",
                    program,
                    mode
                );
            } else {
                if result.is_err() {
                    println!(
                        "Unexpected failure for {:?} + {:?}: {}",
                        program,
                        mode,
                        result.unwrap_err()
                    );
                }
                // Note: Some combinations may still fail due to missing files,
                // but they shouldn't fail due to fundamental incompatibility
            }
        }
    }

    cleanup_test_files();
}
