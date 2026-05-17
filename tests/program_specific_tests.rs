//! Program-specific integration tests for different QM programs.
//!
//! This module tests program-specific functionality including file handling,
//! method modifications, and workflow integration.

use omecp::config::{Config, QMProgram, RunMode};
use omecp::validation::validate_run_mode_compatibility;
use std::fs;

/// Helper function to create a basic test configuration
fn create_test_config() -> Config {
    Config {
        program: QMProgram::Gaussian,
        run_mode: RunMode::Normal,
        method: "B3LYP/6-31G*".to_string(),
        mult_state_a: 1,
        mult_state_b: 1,
        charge: 0,
        nprocs: 1,
        mem: "1GB".to_string(),
        ..Default::default()
    }
}

/// Helper function to create test wavefunction files
fn create_test_wavefunction_files() {
    // Create checkpoint files for Gaussian (using correct names expected by validation)
    let _ = fs::write("state_A.chk", "dummy gaussian checkpoint");
    let _ = fs::write("state_B.chk", "dummy gaussian checkpoint");

    // Create wavefunction files for ORCA (using correct names expected by validation)
    let _ = fs::write("state_A.gbw", "dummy orca wavefunction");
    let _ = fs::write("state_B.gbw", "dummy orca wavefunction");

    // Create running_dir versions
    let _ = fs::create_dir_all("running_dir");
    let _ = fs::write("running_dir/state_A.chk", "dummy gaussian checkpoint");
    let _ = fs::write("running_dir/state_B.chk", "dummy gaussian checkpoint");
    let _ = fs::write("running_dir/state_A.gbw", "dummy orca wavefunction");
    let _ = fs::write("running_dir/state_B.gbw", "dummy orca wavefunction");
}

/// Helper function to clean up test files
fn cleanup_test_files() {
    let _ = fs::remove_file("state_A.chk");
    let _ = fs::remove_file("state_B.chk");
    let _ = fs::remove_file("state_A.gbw");
    let _ = fs::remove_file("state_B.gbw");
    let _ = fs::remove_dir_all("running_dir");
    let _ = fs::remove_file("test_model.json");
    let _ = fs::remove_file("test_interface.json");
    let _ = fs::remove_file("test_bagel_model.json");
}

/// Test Gaussian-specific run mode functionality
#[test]
fn test_gaussian_specific_features() {
    cleanup_test_files();
    create_test_wavefunction_files();

    // Test Gaussian checkpoint file handling
    let mut config = create_test_config();
    config.program = QMProgram::Gaussian;
    config.run_mode = RunMode::Read;

    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_ok(),
        "Gaussian read mode should work with checkpoint files"
    );

    // Test Gaussian MP2 + DFT incompatibility
    config.run_mode = RunMode::Normal;
    config.method = "B3LYP DFT/6-31G*".to_string();
    config.mp2 = true;

    let result = validate_run_mode_compatibility(&config);
    assert!(result.is_err(), "Gaussian MP2 + DFT should be incompatible");

    // Test Gaussian stability mode
    config.mp2 = false;
    config.run_mode = RunMode::Stable;
    config.method = "B3LYP/6-31G*".to_string();

    let result = validate_run_mode_compatibility(&config);
    assert!(result.is_ok(), "Gaussian stability mode should be valid");

    // Test Gaussian inter-read mode
    config.run_mode = RunMode::InterRead;

    let result = validate_run_mode_compatibility(&config);
    assert!(result.is_ok(), "Gaussian inter-read mode should be valid");

    cleanup_test_files();
}

/// Test ORCA-specific run mode functionality with .gbw handling
#[test]
fn test_orca_specific_features() {
    cleanup_test_files();
    create_test_wavefunction_files();

    // Test ORCA .gbw file handling
    let mut config = create_test_config();
    config.program = QMProgram::Orca;
    config.run_mode = RunMode::Read;

    let result = validate_run_mode_compatibility(&config);
    assert!(result.is_ok(), "ORCA read mode should work with .gbw files");

    // Test ORCA stability mode without RI
    config.run_mode = RunMode::Stable;
    config.method = "B3LYP def2-SVP".to_string();

    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_ok(),
        "ORCA stability mode without RI should be valid"
    );

    // Test ORCA stability mode with RI (should fail)
    config.method = "B3LYP def2-SVP RI".to_string();

    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_err(),
        "ORCA stability mode with RI should be invalid"
    );

    // Test ORCA inter-read mode
    config.method = "B3LYP def2-SVP".to_string();
    config.run_mode = RunMode::InterRead;

    let result = validate_run_mode_compatibility(&config);
    assert!(result.is_ok(), "ORCA inter-read mode should be valid");

    // Test ORCA normal mode
    config.run_mode = RunMode::Normal;

    let result = validate_run_mode_compatibility(&config);
    assert!(result.is_ok(), "ORCA normal mode should be valid");

    cleanup_test_files();
}

/// Test XTB basic workflow functionality
#[test]
fn test_xtb_basic_workflow() {
    cleanup_test_files();

    // Test XTB normal mode
    let mut config = create_test_config();
    config.program = QMProgram::Xtb;
    config.run_mode = RunMode::Normal;
    config.method = "GFN2-xTB".to_string();

    let result = validate_run_mode_compatibility(&config);
    assert!(result.is_ok(), "XTB normal mode should be valid");

    // Test XTB noread mode
    config.run_mode = RunMode::NoRead;

    let result = validate_run_mode_compatibility(&config);
    assert!(result.is_ok(), "XTB noread mode should be valid");

    // Test XTB with non-GFN method (should warn but be valid)
    config.method = "AM1".to_string();

    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_ok(),
        "XTB with non-GFN method should be valid but may warn"
    );

    // Test XTB unsupported modes
    let unsupported_modes = vec![RunMode::Stable, RunMode::InterRead];

    for mode in unsupported_modes {
        config.run_mode = mode;
        config.method = "GFN2-xTB".to_string();

        let result = validate_run_mode_compatibility(&config);
        assert!(result.is_err(), "XTB {:?} mode should be unsupported", mode);
    }

    cleanup_test_files();
}

/// Test BAGEL basic workflow functionality
#[test]
fn test_bagel_basic_workflow() {
    cleanup_test_files();

    // Create BAGEL model file
    let bagel_model_content = r#"{
        "bagel": [
            {
                "title": "molecule",
                "basis": "cc-pVDZ",
                "df_basis": "cc-pVDZ-jkfit",
                "angstrom": true,
                "geometry": []
            },
            {
                "title": "hf"
            }
        ]
    }"#;
    fs::write("test_bagel_model.json", bagel_model_content).unwrap();

    // Test BAGEL normal mode
    let mut config = create_test_config();
    config.program = QMProgram::Bagel;
    config.run_mode = RunMode::Normal;
    config.bagel_model = "test_bagel_model.json".to_string();

    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_ok(),
        "BAGEL normal mode should be valid with model file"
    );

    // Test BAGEL read mode
    config.run_mode = RunMode::Read;

    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_ok(),
        "BAGEL read mode should be valid with model file"
    );

    // Test BAGEL inter-read mode
    config.run_mode = RunMode::InterRead;

    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_ok(),
        "BAGEL inter-read mode should be valid with model file"
    );

    // Test BAGEL without model file (should fail for read modes)
    config.bagel_model = String::new();
    config.run_mode = RunMode::Read;

    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_err(),
        "BAGEL read mode should fail without model file"
    );

    // Test BAGEL with missing model file
    config.bagel_model = "nonexistent_model.json".to_string();

    let result = validate_run_mode_compatibility(&config);
    assert!(result.is_err(), "BAGEL should fail with missing model file");

    cleanup_test_files();
}

/// Test custom program interface functionality
#[test]
fn test_custom_program_interface() {
    cleanup_test_files();

    // Create custom interface file
    let interface_content = r#"{
        "program_name": "custom_qm",
        "executable": "custom_qm.exe",
        "input_format": "json",
        "output_format": "json",
        "supported_methods": ["DFT", "HF"],
        "file_extensions": {
            "input": ".inp",
            "output": ".out",
            "wavefunction": ".wfn"
        }
    }"#;
    fs::write("test_interface.json", interface_content).unwrap();

    // Test custom program normal mode
    let mut config = create_test_config();
    config.program = QMProgram::Custom;
    config.run_mode = RunMode::Normal;
    config.custom_interface_file = "test_interface.json".to_string();

    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_ok(),
        "Custom program normal mode should be valid with interface file"
    );

    // Test custom program read mode
    config.run_mode = RunMode::Read;

    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_ok(),
        "Custom program read mode should be valid with interface file"
    );

    // Test custom program without interface file
    config.custom_interface_file = String::new();

    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_err(),
        "Custom program should fail without interface file"
    );

    // Test custom program with missing interface file
    config.custom_interface_file = "nonexistent_interface.json".to_string();

    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_err(),
        "Custom program should fail with missing interface file"
    );

    cleanup_test_files();
}

/// Test constraint integration with optimization (conceptual test)
#[test]
fn test_constraint_integration() {
    cleanup_test_files();
    create_test_wavefunction_files();

    // Test that constraint-related run modes work with validation
    let constraint_modes = vec![
        RunMode::CoordinateDrive,
        RunMode::PathOptimization,
        RunMode::FixDE,
    ];

    for mode in constraint_modes {
        let mut config = create_test_config();
        config.program = QMProgram::Gaussian;
        config.run_mode = mode;

        // Add required parameters for constraint modes
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

        let result = validate_run_mode_compatibility(&config);
        assert!(result.is_ok(), "Constraint mode {:?} should be valid", mode);

        // Test with different programs
        for program in [QMProgram::Orca, QMProgram::Bagel] {
            config.program = program;

            // Add program-specific requirements
            match program {
                QMProgram::Bagel => {
                    fs::write("test_bagel_model.json", "{}").unwrap();
                    config.bagel_model = "test_bagel_model.json".to_string();
                }
                _ => {}
            }

            let result = validate_run_mode_compatibility(&config);
            assert!(
                result.is_ok(),
                "Constraint mode {:?} should work with {:?}",
                mode,
                program
            );
        }
    }

    cleanup_test_files();
}

/// Test program-specific error messages and guidance
#[test]
fn test_program_specific_error_messages() {
    cleanup_test_files();

    // Test ORCA RI + Stability error message
    let mut config = create_test_config();
    config.program = QMProgram::Orca;
    config.run_mode = RunMode::Stable;
    config.method = "B3LYP RI def2-SVP".to_string();

    let result = validate_run_mode_compatibility(&config);
    assert!(result.is_err(), "ORCA RI + Stability should fail");

    let error = result.unwrap_err();
    assert!(error.to_string().contains("RI"));
    assert!(error.to_string().contains("stability"));
    assert!(error.to_string().contains("Suggestion"));

    // Test XTB unsupported mode error message
    config.program = QMProgram::Xtb;
    config.run_mode = RunMode::InterRead;
    config.method = "GFN2-xTB".to_string();

    let result = validate_run_mode_compatibility(&config);
    assert!(result.is_err(), "XTB InterRead should fail");

    let error = result.unwrap_err();
    assert!(error.to_string().contains("XTB"));
    assert!(error.to_string().contains("inter_read"));
    assert!(error.to_string().contains("wavefunction files"));

    // Test BAGEL missing model file error message
    config.program = QMProgram::Bagel;
    config.run_mode = RunMode::Read;
    config.bagel_model = String::new();

    let result = validate_run_mode_compatibility(&config);
    assert!(result.is_err(), "BAGEL without model should fail");

    let error = result.unwrap_err();
    assert!(error.to_string().contains("BAGEL"));
    assert!(error.to_string().contains("model file"));

    cleanup_test_files();
}

/// Test wavefunction file format validation for different programs
#[test]
fn test_wavefunction_file_format_validation() {
    cleanup_test_files();

    // Test Gaussian checkpoint file validation (using correct names expected by validation)
    let _ = fs::write("state_A.chk", "dummy gaussian checkpoint");
    let _ = fs::write("state_B.chk", "dummy gaussian checkpoint");
    let _ = fs::create_dir_all("running_dir");
    let _ = fs::write("running_dir/state_A.chk", "dummy gaussian checkpoint");
    let _ = fs::write("running_dir/state_B.chk", "dummy gaussian checkpoint");

    let mut config = create_test_config();
    config.program = QMProgram::Gaussian;
    config.run_mode = RunMode::Read;

    let result = validate_run_mode_compatibility(&config);
    assert!(result.is_ok(), "Gaussian should accept .chk files");

    // Clean up Gaussian files and create ORCA files
    cleanup_test_files();
    let _ = fs::write("state_A.gbw", "dummy orca wavefunction");
    let _ = fs::write("state_B.gbw", "dummy orca wavefunction");
    let _ = fs::create_dir_all("running_dir");
    let _ = fs::write("running_dir/state_A.gbw", "dummy orca wavefunction");
    let _ = fs::write("running_dir/state_B.gbw", "dummy orca wavefunction");

    config.program = QMProgram::Orca;

    let result = validate_run_mode_compatibility(&config);
    assert!(result.is_ok(), "ORCA should accept .gbw files");

    cleanup_test_files();
}

/// Test program command customization (conceptual test)
#[test]
fn test_program_command_customization() {
    cleanup_test_files();

    // Test that custom program commands can be specified
    let mut config = create_test_config();
    config.program = QMProgram::Gaussian;
    config.run_mode = RunMode::Normal;

    // Add custom command mapping
    config
        .program_commands
        .insert("gaussian".to_string(), "g16".to_string());

    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_ok(),
        "Custom program commands should not affect validation"
    );

    // Test with ORCA
    config.program = QMProgram::Orca;
    config.program_commands.clear();
    config
        .program_commands
        .insert("orca".to_string(), "/opt/orca/orca".to_string());

    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_ok(),
        "Custom ORCA command should not affect validation"
    );

    cleanup_test_files();
}

/// Test multi-program compatibility in the same configuration
#[test]
fn test_multi_program_compatibility() {
    cleanup_test_files();
    create_test_wavefunction_files();

    // Create files for all programs
    fs::write("test_bagel_model.json", "{}").unwrap();
    fs::write("test_interface.json", "{}").unwrap();

    let programs = vec![
        QMProgram::Gaussian,
        QMProgram::Orca,
        QMProgram::Bagel,
        QMProgram::Custom,
    ];

    // Test that each program works with basic modes
    for program in programs {
        let mut config = create_test_config();
        config.program = program;
        config.run_mode = RunMode::Normal;

        // Set program-specific requirements
        match program {
            QMProgram::Bagel => {
                config.bagel_model = "test_bagel_model.json".to_string();
            }
            QMProgram::Custom => {
                config.custom_interface_file = "test_interface.json".to_string();
            }
            _ => {}
        }

        let result = validate_run_mode_compatibility(&config);
        assert!(
            result.is_ok(),
            "Program {:?} should work in normal mode",
            program
        );

        // Test read mode (where supported)
        if !matches!(program, QMProgram::Xtb) {
            config.run_mode = RunMode::Read;
            let result = validate_run_mode_compatibility(&config);
            // May fail due to missing files, but shouldn't fail due to program incompatibility
            if result.is_err() {
                let error_msg = result.unwrap_err().to_string();
                // Should be about missing files, not program incompatibility
                assert!(
                    error_msg.contains("file") || error_msg.contains("model"),
                    "Read mode failure should be about files, not program incompatibility: {}",
                    error_msg
                );
            }
        }
    }

    cleanup_test_files();
}
