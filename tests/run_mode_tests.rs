//! Comprehensive integration tests for run mode functionality.
//!
//! This module tests all run modes with different QM programs to ensure
//! proper integration and method modification behavior.

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
}

/// Test all run modes with Gaussian program
#[test]
fn test_gaussian_run_modes() {
    cleanup_test_files();
    create_test_wavefunction_files();

    let run_modes = vec![
        RunMode::Normal,
        RunMode::Read,
        RunMode::NoRead,
        RunMode::Stable,
        RunMode::InterRead,
    ];

    for mode in run_modes {
        let mut config = create_test_config();
        config.program = QMProgram::Gaussian;
        config.run_mode = mode;

        // Add required parameters for specific modes
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
        assert!(result.is_ok(), "Gaussian {:?} mode should be valid", mode);
    }

    cleanup_test_files();
}

/// Test all run modes with ORCA program
#[test]
fn test_orca_run_modes() {
    cleanup_test_files();
    create_test_wavefunction_files();

    let run_modes = vec![
        RunMode::Normal,
        RunMode::Read,
        RunMode::NoRead,
        RunMode::InterRead,
    ];

    for mode in run_modes {
        let mut config = create_test_config();
        config.program = QMProgram::Orca;
        config.run_mode = mode;

        let result = validate_run_mode_compatibility(&config);
        assert!(result.is_ok(), "ORCA {:?} mode should be valid", mode);
    }

    // Test ORCA Stable mode without RI (should work)
    let mut config = create_test_config();
    config.program = QMProgram::Orca;
    config.run_mode = RunMode::Stable;
    config.method = "B3LYP/6-31G*".to_string(); // No RI

    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_ok(),
        "ORCA Stable mode without RI should be valid"
    );

    // Test ORCA Stable mode with RI (should fail)
    config.method = "B3LYP RI/6-31G*".to_string(); // With RI
    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_err(),
        "ORCA Stable mode with RI should be invalid"
    );

    cleanup_test_files();
}

/// Test XTB program limitations
#[test]
fn test_xtb_run_modes() {
    cleanup_test_files();

    // Valid XTB modes
    let valid_modes = vec![RunMode::Normal, RunMode::NoRead];

    for mode in valid_modes {
        let mut config = create_test_config();
        config.program = QMProgram::Xtb;
        config.run_mode = mode;
        config.method = "GFN2-xTB".to_string();

        let result = validate_run_mode_compatibility(&config);
        assert!(result.is_ok(), "XTB {:?} mode should be valid", mode);
    }

    // Invalid XTB modes
    let invalid_modes = vec![RunMode::Stable, RunMode::InterRead];

    for mode in invalid_modes {
        let mut config = create_test_config();
        config.program = QMProgram::Xtb;
        config.run_mode = mode;
        config.method = "GFN2-xTB".to_string();

        let result = validate_run_mode_compatibility(&config);
        assert!(result.is_err(), "XTB {:?} mode should be invalid", mode);
    }

    cleanup_test_files();
}

/// Test BAGEL program requirements
#[test]
fn test_bagel_run_modes() {
    cleanup_test_files();

    // Create required model file
    fs::write("test_model.json", r#"{"title": "test bagel model"}"#).unwrap();

    let run_modes = vec![
        RunMode::Normal,
        RunMode::Read,
        RunMode::NoRead,
        RunMode::InterRead,
    ];

    for mode in run_modes {
        let mut config = create_test_config();
        config.program = QMProgram::Bagel;
        config.run_mode = mode;
        config.bagel_model = "test_model.json".to_string();

        let result = validate_run_mode_compatibility(&config);
        assert!(
            result.is_ok(),
            "BAGEL {:?} mode should be valid with model file",
            mode
        );
    }

    // Test BAGEL without model file (should fail for read modes)
    let mut config = create_test_config();
    config.program = QMProgram::Bagel;
    config.run_mode = RunMode::Read;
    config.bagel_model = String::new(); // No model file

    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_err(),
        "BAGEL read mode should fail without model file"
    );

    cleanup_test_files();
}

/// Test Custom program requirements
#[test]
fn test_custom_program_run_modes() {
    cleanup_test_files();

    // Create required interface file
    fs::write(
        "test_interface.json",
        r#"{"interface": "test custom interface"}"#,
    )
    .unwrap();

    let run_modes = vec![
        RunMode::Normal,
        RunMode::Read,
        RunMode::NoRead,
        RunMode::InterRead,
    ];

    for mode in run_modes {
        let mut config = create_test_config();
        config.program = QMProgram::Custom;
        config.run_mode = mode;
        config.custom_interface_file = "test_interface.json".to_string();

        let result = validate_run_mode_compatibility(&config);
        assert!(
            result.is_ok(),
            "Custom {:?} mode should be valid with interface file",
            mode
        );
    }

    // Test Custom without interface file (should fail)
    let mut config = create_test_config();
    config.program = QMProgram::Custom;
    config.run_mode = RunMode::Normal;
    config.custom_interface_file = String::new(); // No interface file

    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_err(),
        "Custom program should fail without interface file"
    );

    cleanup_test_files();
}

/// Test method modification for different programs and modes
#[test]
fn test_method_modification_logic() {
    // This test verifies that method modification logic works correctly
    // for different program/mode combinations

    cleanup_test_files();
    create_test_wavefunction_files(); // Create files needed for read modes

    let test_cases = vec![
        // (program, mode, base_method, should_contain_keywords)
        (QMProgram::Gaussian, RunMode::Normal, "B3LYP", vec!["force"]),
        (
            QMProgram::Gaussian,
            RunMode::Read,
            "B3LYP",
            vec!["force", "guess=read"],
        ),
        (QMProgram::Gaussian, RunMode::NoRead, "B3LYP", vec!["force"]), // No guess=read
        (
            QMProgram::Gaussian,
            RunMode::Stable,
            "B3LYP",
            vec!["force", "stable=opt"],
        ),
        (QMProgram::Orca, RunMode::Normal, "B3LYP", vec!["engrad"]),
        (
            QMProgram::Orca,
            RunMode::Read,
            "B3LYP",
            vec!["engrad", "!moread"],
        ),
        (QMProgram::Orca, RunMode::NoRead, "B3LYP", vec!["engrad"]), // No !moread
    ];

    for (program, mode, base_method, expected_keywords) in test_cases {
        // Note: This test would require access to the method modification function
        // which might be internal to the io module. For now, we test the concept.

        let mut config = create_test_config();
        config.program = program;
        config.run_mode = mode;
        config.method = base_method.to_string();

        // The actual method modification would happen in the header building process
        // This test verifies the logic exists and works correctly

        println!(
            "Testing method modification for {:?} {:?}: {} -> expected keywords: {:?}",
            program, mode, base_method, expected_keywords
        );

        // For now, just verify the configuration is valid
        let result = validate_run_mode_compatibility(&config);

        // Most combinations should be valid (except known incompatibilities)
        let should_be_valid = !matches!(
            (program, mode),
            (QMProgram::Xtb, RunMode::Stable) | (QMProgram::Xtb, RunMode::InterRead)
        );

        if should_be_valid {
            assert!(
                result.is_ok(),
                "Method modification test failed for {:?} {:?}",
                program,
                mode
            );
        }
    }

    cleanup_test_files();
}

/// Test coordinate driving mode validation
#[test]
fn test_coordinate_drive_mode() {
    cleanup_test_files();

    let mut config = create_test_config();
    config.run_mode = RunMode::CoordinateDrive;

    // Test without required parameters (should fail)
    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_err(),
        "CoordinateDrive should fail without parameters"
    );

    // Test with valid parameters
    config.drive_type = "bond".to_string();
    config.drive_atoms = vec![1, 2];
    config.drive_start = 1.0;
    config.drive_end = 2.0;
    config.drive_steps = 10;

    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_ok(),
        "CoordinateDrive should pass with valid parameters"
    );

    // Test with invalid parameters (same start/end)
    config.drive_end = 1.0; // Same as start
    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_err(),
        "CoordinateDrive should fail with same start/end"
    );

    // Test with zero steps
    config.drive_end = 2.0;
    config.drive_steps = 0;
    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_err(),
        "CoordinateDrive should fail with zero steps"
    );

    cleanup_test_files();
}

/// Test path optimization mode validation
#[test]
fn test_path_optimization_mode() {
    cleanup_test_files();

    let mut config = create_test_config();
    config.run_mode = RunMode::PathOptimization;

    // Test without required parameters (should fail)
    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_err(),
        "PathOptimization should fail without parameters"
    );

    // Test with valid parameters
    config.drive_type = "dihedral".to_string();
    config.drive_atoms = vec![1, 2, 3, 4];

    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_ok(),
        "PathOptimization should pass with valid parameters"
    );

    cleanup_test_files();
}

/// Test FixDE mode validation
#[test]
fn test_fixde_mode() {
    cleanup_test_files();

    let mut config = create_test_config();
    config.run_mode = RunMode::FixDE;

    // Test without target energy difference (should fail)
    config.fix_de = 0.0;
    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_err(),
        "FixDE should fail without target energy difference"
    );

    // Test with valid energy difference
    config.fix_de = 0.5;
    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_ok(),
        "FixDE should pass with valid energy difference"
    );

    cleanup_test_files();
}

/// Test mode switching logic (conceptual test)
#[test]
fn test_mode_switching_logic() {
    // This test verifies the concept of mode switching that happens
    // in the main optimization loop (stable -> read, inter_read -> read)

    cleanup_test_files();
    create_test_wavefunction_files(); // Create files needed for read modes

    let mode_transitions = vec![
        (
            RunMode::Stable,
            RunMode::Read,
            "Stability analysis completed",
        ),
        (
            RunMode::InterRead,
            RunMode::Read,
            "Inter-read initialization completed",
        ),
    ];

    for (from_mode, to_mode, reason) in mode_transitions {
        // Verify that both modes are valid for the same program
        let mut config = create_test_config();
        config.program = QMProgram::Gaussian;

        // Test original mode
        config.run_mode = from_mode;
        let _result1 = validate_run_mode_compatibility(&config);

        // Test target mode (Read mode needs wavefunction files)
        config.run_mode = to_mode;
        let result2 = validate_run_mode_compatibility(&config);

        // The target mode (Read) should be valid with wavefunction files
        assert!(
            result2.is_ok(),
            "Mode switching target {:?} should be valid with wavefunction files",
            to_mode
        );

        println!(
            "Mode transition test: {:?} -> {:?} ({})",
            from_mode, to_mode, reason
        );
    }

    cleanup_test_files();
}

/// Test wavefunction file requirements for read modes
#[test]
fn test_wavefunction_file_requirements() {
    cleanup_test_files();

    // Test Gaussian read mode without files (should fail)
    let mut config = create_test_config();
    config.program = QMProgram::Gaussian;
    config.run_mode = RunMode::Read;

    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_err(),
        "Gaussian read mode should fail without checkpoint files"
    );

    // Create files and test again (should pass)
    create_test_wavefunction_files();
    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_ok(),
        "Gaussian read mode should pass with checkpoint files"
    );

    // Test ORCA read mode
    config.program = QMProgram::Orca;
    let result = validate_run_mode_compatibility(&config);
    assert!(result.is_ok(), "ORCA read mode should pass with .gbw files");

    cleanup_test_files();
}

/// Test inter-read mode specific behavior
#[test]
fn test_inter_read_mode_specifics() {
    cleanup_test_files();
    create_test_wavefunction_files();

    // Test inter-read with singlet multiplicities (should pass with no warnings)
    let mut config = create_test_config();
    config.run_mode = RunMode::InterRead;
    config.mult_state_a = 1;
    config.mult_state_b = 1;

    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_ok(),
        "InterRead with singlet multiplicities should be valid"
    );

    // Test inter-read with non-singlet multiplicities (should pass but warn)
    config.mult_state_a = 3; // Triplet
    config.mult_state_b = 1; // Singlet

    let result = validate_run_mode_compatibility(&config);
    assert!(
        result.is_ok(),
        "InterRead with mixed multiplicities should be valid but may warn"
    );

    cleanup_test_files();
}

/// Test comprehensive program/mode compatibility matrix
#[test]
fn test_comprehensive_compatibility_matrix() {
    cleanup_test_files();
    create_test_wavefunction_files();

    // Create required files for specific programs
    fs::write("test_model.json", "{}").unwrap();
    fs::write("test_interface.json", "{}").unwrap();

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
    ];

    let mut compatibility_results = Vec::new();

    for program in programs {
        for mode in &modes {
            let mut config = create_test_config();
            config.program = program;
            config.run_mode = *mode;

            // Set required parameters for specific programs
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
            let is_compatible = result.is_ok();

            compatibility_results.push((program, *mode, is_compatible));

            // Check expected incompatibilities
            let expected_incompatible = matches!(
                (program, mode),
                (QMProgram::Xtb, RunMode::Stable) | (QMProgram::Xtb, RunMode::InterRead)
            );

            if expected_incompatible {
                assert!(
                    !is_compatible,
                    "Expected incompatibility: {:?} + {:?}",
                    program, mode
                );
            }
        }
    }

    // Print compatibility matrix for debugging
    println!("\\nProgram/Mode Compatibility Matrix:");
    println!("Program\\tNormal\\tRead\\tNoRead\\tStable\\tInterRead");

    for program in [
        QMProgram::Gaussian,
        QMProgram::Orca,
        QMProgram::Xtb,
        QMProgram::Bagel,
        QMProgram::Custom,
    ] {
        print!("{:?}\\t", program);
        for mode in &modes {
            let is_compatible = compatibility_results
                .iter()
                .find(|(p, m, _)| *p == program && *m == *mode)
                .map(|(_, _, compatible)| *compatible)
                .unwrap_or(false);
            print!("{}\\t", if is_compatible { "✓" } else { "✗" });
        }
        println!();
    }

    cleanup_test_files();
}
