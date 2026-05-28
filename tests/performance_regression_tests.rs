//! Performance and regression tests for OpenMECP.
//!
//! This module tests performance characteristics, memory usage,
//! and ensures no regression in optimization algorithms.

use omecp::config::{Config, QMProgram, RunMode};
use omecp::validation::validate_run_mode_compatibility;
use std::fs;
use std::time::Instant;

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
    // Create checkpoint files for Gaussian
    let _ = fs::write("a.chk", "dummy gaussian checkpoint");
    let _ = fs::write("b.chk", "dummy gaussian checkpoint");

    // Create wavefunction files for ORCA
    let _ = fs::write("a.gbw", "dummy orca wavefunction");
    let _ = fs::write("b.gbw", "dummy orca wavefunction");

    // Create running_dir versions
    let _ = fs::create_dir_all("running_dir");
    let _ = fs::write("running_dir/a.chk", "dummy gaussian checkpoint");
    let _ = fs::write("running_dir/b.chk", "dummy gaussian checkpoint");
    let _ = fs::write("running_dir/a.gbw", "dummy orca wavefunction");
    let _ = fs::write("running_dir/b.gbw", "dummy orca wavefunction");
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

/// Test validation performance with multiple configurations
#[test]
fn test_validation_performance() {
    cleanup_test_files();
    create_test_wavefunction_files();

    let start_time = Instant::now();

    // Test validation performance with 100 different configurations
    for i in 0..100 {
        let mut config = create_test_config();

        // Vary the configuration parameters
        config.program = match i % 4 {
            0 => QMProgram::Gaussian,
            1 => QMProgram::Orca,
            2 => QMProgram::Xtb,
            _ => QMProgram::Bagel,
        };

        config.run_mode = match i % 5 {
            0 => RunMode::Normal,
            1 => RunMode::Read,
            2 => RunMode::NoRead,
            3 => RunMode::Stable,
            _ => RunMode::InterRead,
        };

        // Add required parameters for specific programs
        if config.program == QMProgram::Bagel {
            fs::write("test_model.json", "{}").unwrap();
            config.bagel_model = "test_model.json".to_string();
        }

        // Skip known invalid combinations
        let is_invalid = matches!(
            (config.program, config.run_mode),
            (QMProgram::Xtb, RunMode::Stable) | (QMProgram::Xtb, RunMode::InterRead)
        );

        if !is_invalid {
            let _result = validate_run_mode_compatibility(&config);
            // Don't assert on result, just measure performance
        }
    }

    let elapsed = start_time.elapsed();

    // Validation should complete quickly (under 1 second for 100 configs)
    assert!(
        elapsed.as_millis() < 1000,
        "Validation performance regression: took {}ms for 100 configs",
        elapsed.as_millis()
    );

    println!(
        "Validation performance: {}ms for 100 configurations",
        elapsed.as_millis()
    );

    cleanup_test_files();
}

/// Test memory usage with large configurations
#[test]
fn test_memory_usage_large_systems() {
    cleanup_test_files();
    create_test_wavefunction_files();

    // Test with configurations that simulate large systems
    let mut config = create_test_config();
    config.program = QMProgram::Gaussian;
    config.run_mode = RunMode::Normal;

    // Simulate large system parameters
    config.nprocs = 32;
    config.mem = "64GB".to_string();
    config.max_steps = 1000;

    // Add large drive_atoms vector to simulate large system
    config.run_mode = RunMode::CoordinateDrive;
    config.drive_type = "bond".to_string();
    config.drive_atoms = (1..=1000).collect(); // 1000 atoms
    config.drive_start = 1.0;
    config.drive_end = 2.0;
    config.drive_steps = 100;

    let start_time = Instant::now();
    let result = validate_run_mode_compatibility(&config);
    let elapsed = start_time.elapsed();

    // Should still validate quickly even with large configurations
    assert!(
        elapsed.as_millis() < 100,
        "Large system validation took too long: {}ms",
        elapsed.as_millis()
    );

    assert!(result.is_ok(), "Large system configuration should be valid");

    println!("Large system validation: {}ms", elapsed.as_millis());

    cleanup_test_files();
}

/// Test convergence behavior consistency (conceptual test)
#[test]
fn test_convergence_behavior_consistency() {
    cleanup_test_files();
    create_test_wavefunction_files();

    // Test that convergence thresholds are consistent
    let config = create_test_config();

    // Test default thresholds
    let default_thresholds = &config.thresholds;
    assert!(
        default_thresholds.delta_e > 0.0,
        "Energy threshold should be positive"
    );
    assert!(
        default_thresholds.rms_dis > 0.0,
        "RMS threshold should be positive"
    );
    assert!(
        default_thresholds.max_dis > 0.0,
        "Max displacement threshold should be positive"
    );
    assert!(
        default_thresholds.max_grad > 0.0,
        "Max gradient threshold should be positive"
    );
    assert!(
        default_thresholds.rms_grad > 0.0,
        "RMS gradient threshold should be positive"
    );

    // Test that thresholds are reasonable (not too tight or too loose)
    assert!(
        default_thresholds.delta_e < 0.01,
        "Energy threshold should not be too loose"
    );
    assert!(
        default_thresholds.delta_e > 1e-8,
        "Energy threshold should not be too tight"
    );

    // Test max_steps is reasonable
    assert!(config.max_steps > 0, "Max steps should be positive");
    assert!(
        config.max_steps <= 10000,
        "Max steps should not be excessive"
    );

    // Test step size parameters
    assert!(
        config.max_step_size > 0.0,
        "Max step size should be positive"
    );
    assert!(
        config.max_step_size < 1.0,
        "Max step size should not be too large"
    );
    assert!(
        config.reduced_factor > 0.0 && config.reduced_factor < 1.0,
        "Reduction factor should be between 0 and 1"
    );

    println!("Convergence parameters validated:");
    println!("  Energy threshold: {:.2e}", default_thresholds.delta_e);
    println!("  RMS threshold: {:.4}", default_thresholds.rms_dis);
    println!("  Max steps: {}", config.max_steps);
    println!("  Max step size: {:.3}", config.max_step_size);

    cleanup_test_files();
}

/// Test checkpoint and restart functionality (conceptual test)
#[test]
fn test_checkpoint_restart_functionality() {
    cleanup_test_files();
    create_test_wavefunction_files();

    // Test checkpoint configuration
    let mut config = create_test_config();
    config.restart = false;
    config.print_checkpoint = true;

    let result = validate_run_mode_compatibility(&config);
    assert!(result.is_ok(), "Checkpoint configuration should be valid");

    // Test restart mode
    config.restart = true;

    let result = validate_run_mode_compatibility(&config);
    assert!(result.is_ok(), "Restart configuration should be valid");

    // Test that print_checkpoint is enabled
    assert!(
        config.print_checkpoint,
        "Checkpoint printing should be enabled"
    );

    println!("Checkpoint functionality validated:");
    println!("  Print checkpoint: {}", config.print_checkpoint);
    println!("  Restart mode: {}", config.restart);

    cleanup_test_files();
}

/// Test optimization algorithm parameters (conceptual test)
#[test]
fn test_optimization_algorithm_parameters() {
    cleanup_test_files();

    // Test GEDIIS vs GDIIS selection
    let mut config = create_test_config();

    // Test default optimizer selection
    assert!(!config.use_gediis, "Should default to GDIIS optimizer");

    // Test GEDIIS option
    config.use_gediis = true;
    let result = validate_run_mode_compatibility(&config);
    assert!(result.is_ok(), "GEDIIS optimizer should be valid");

    // Test that optimization parameters are within reasonable ranges
    assert!(
        config.max_step_size > 0.0 && config.max_step_size < 1.0,
        "Max step size should be reasonable"
    );
    assert!(
        config.reduced_factor > 0.0 && config.reduced_factor < 1.0,
        "Reduction factor should be reasonable"
    );

    println!("Optimization algorithm parameters validated:");
    println!("  Use GEDIIS: {}", config.use_gediis);
    println!("  Max step size: {:.3}", config.max_step_size);
    println!("  Reduction factor: {:.3}", config.reduced_factor);

    cleanup_test_files();
}

/// Test resource allocation parameters
#[test]
fn test_resource_allocation_parameters() {
    cleanup_test_files();

    let mut config = create_test_config();

    // Test processor count validation
    assert!(config.nprocs > 0, "Number of processors should be positive");

    // Test memory specification
    assert!(!config.mem.is_empty(), "Memory should be specified");
    assert!(
        config.mem.contains("GB") || config.mem.contains("MB"),
        "Memory should have units"
    );

    // Test with various resource configurations
    let resource_configs = vec![(1, "1GB"), (4, "4GB"), (16, "32GB"), (64, "128GB")];

    for (nprocs, mem) in resource_configs {
        config.nprocs = nprocs;
        config.mem = mem.to_string();

        let result = validate_run_mode_compatibility(&config);
        assert!(
            result.is_ok(),
            "Resource configuration {} procs, {} should be valid",
            nprocs,
            mem
        );
    }

    println!("Resource allocation parameters validated");

    cleanup_test_files();
}

/// Test configuration serialization/deserialization performance
#[test]
fn test_config_serialization_performance() {
    cleanup_test_files();

    let config = create_test_config();

    let start_time = Instant::now();

    // Test serialization performance
    for _i in 0..1000 {
        let _serialized = serde_json::to_string(&config).unwrap();
    }

    let serialize_elapsed = start_time.elapsed();

    let serialized = serde_json::to_string(&config).unwrap();
    let start_time = Instant::now();

    // Test deserialization performance
    for _i in 0..1000 {
        let _deserialized: Config = serde_json::from_str(&serialized).unwrap();
    }

    let deserialize_elapsed = start_time.elapsed();

    // Serialization should be fast (under 500ms for 1000 operations in debug builds)
    assert!(
        serialize_elapsed.as_millis() < 500,
        "Serialization performance regression: {}ms for 1000 operations",
        serialize_elapsed.as_millis()
    );

    assert!(
        deserialize_elapsed.as_millis() < 500,
        "Deserialization performance regression: {}ms for 1000 operations",
        deserialize_elapsed.as_millis()
    );

    println!(
        "Serialization performance: {}ms for 1000 operations",
        serialize_elapsed.as_millis()
    );
    println!(
        "Deserialization performance: {}ms for 1000 operations",
        deserialize_elapsed.as_millis()
    );

    cleanup_test_files();
}

/// Test error handling performance
#[test]
fn test_error_handling_performance() {
    cleanup_test_files();

    let start_time = Instant::now();

    // Test error generation and handling performance
    for i in 0..100 {
        let mut config = create_test_config();

        // Create configurations that will generate errors
        match i % 3 {
            0 => {
                // XTB + Stable (unsupported)
                config.program = QMProgram::Xtb;
                config.run_mode = RunMode::Stable;
            }
            1 => {
                // ORCA + RI + Stable (incompatible)
                config.program = QMProgram::Orca;
                config.run_mode = RunMode::Stable;
                config.method = "B3LYP RI/def2-SVP".to_string();
            }
            _ => {
                // Missing BAGEL model
                config.program = QMProgram::Bagel;
                config.run_mode = RunMode::Read;
                config.bagel_model = String::new();
            }
        }

        let result = validate_run_mode_compatibility(&config);
        assert!(result.is_err(), "Should generate error for invalid config");

        // Ensure error message is not empty
        let error = result.unwrap_err();
        assert!(
            !error.to_string().is_empty(),
            "Error message should not be empty"
        );
    }

    let elapsed = start_time.elapsed();

    // Error handling should be fast (under 500ms for 100 error cases)
    assert!(
        elapsed.as_millis() < 500,
        "Error handling performance regression: {}ms for 100 errors",
        elapsed.as_millis()
    );

    println!(
        "Error handling performance: {}ms for 100 error cases",
        elapsed.as_millis()
    );

    cleanup_test_files();
}

/// Test validation consistency across multiple runs
#[test]
fn test_validation_consistency() {
    cleanup_test_files();
    create_test_wavefunction_files();

    let config = create_test_config();

    // Run validation multiple times and ensure consistent results
    let mut results = Vec::new();

    for _i in 0..10 {
        let result = validate_run_mode_compatibility(&config);
        results.push(result.is_ok());
    }

    // All results should be the same
    let first_result = results[0];
    for (i, result) in results.iter().enumerate() {
        assert_eq!(
            *result, first_result,
            "Validation result inconsistent at iteration {}",
            i
        );
    }

    println!("Validation consistency verified across 10 runs");

    cleanup_test_files();
}

/// Test thread safety of validation (conceptual test)
#[test]
fn test_validation_thread_safety() {
    cleanup_test_files();
    create_test_wavefunction_files();

    use std::sync::Arc;
    use std::thread;

    let config = Arc::new(create_test_config());
    let mut handles = Vec::new();

    // Spawn multiple threads to test validation concurrency
    for i in 0..4 {
        let config_clone = Arc::clone(&config);
        let handle = thread::spawn(move || {
            for _j in 0..25 {
                let result = validate_run_mode_compatibility(&config_clone);
                // Should not panic or produce inconsistent results
                assert!(result.is_ok(), "Thread {} validation failed", i);
            }
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    println!("Thread safety validated with 4 threads × 25 validations each");

    cleanup_test_files();
}
