/// Tests for switch_step optimizer configuration functionality.
use omecp::config::Config;
use omecp::parser::parse_input;
use std::fs;
use tempfile::TempDir;

#[test]
fn test_default_switch_step() {
    // Test that default switch_step is 3
    let config = Config::default();
    assert_eq!(config.switch_step, 3);
}

#[test]
fn test_switch_step_parsing() {
    let temp_dir = TempDir::new().unwrap();
    let input_path = temp_dir.path().join("test.inp");

    // Create input file with custom switch_step
    let input_content = r#"
*GEOM
C 0.0 0.0 0.0
H 1.0 0.0 0.0
*

program = gaussian
method = B3LYP/6-31G*
charge = 0
mult_state_a = 1
mult_state_b = 3
switch_step = 10
use_gediis = true
max_steps = 50
"#;

    fs::write(&input_path, input_content).unwrap();

    // Parse and verify
    let input_data = parse_input(&input_path).unwrap();
    assert_eq!(input_data.config.switch_step, 10);
    assert_eq!(input_data.config.use_gediis, true);
    assert_eq!(input_data.config.max_steps, 50);
}

#[test]
fn test_bfgs_only_mode() {
    let temp_dir = TempDir::new().unwrap();
    let input_path = temp_dir.path().join("test.inp");

    // Create input file with BFGS-only mode
    let input_content = r#"
*GEOM
C 0.0 0.0 0.0
H 1.0 0.0 0.0
*

program = gaussian
method = B3LYP/6-31G*
charge = 0
mult_state_a = 1
mult_state_b = 3
max_steps = 20
switch_step = 20    # >= max_steps = BFGS-only
use_gediis = true   # Should be ignored in BFGS-only mode
"#;

    fs::write(&input_path, input_content).unwrap();

    // Parse and verify
    let input_data = parse_input(&input_path).unwrap();
    assert_eq!(input_data.config.switch_step, 20);
    assert_eq!(input_data.config.max_steps, 20);
    assert!(input_data.config.switch_step >= input_data.config.max_steps);
}

#[test]
fn test_diis_only_mode() {
    let temp_dir = TempDir::new().unwrap();
    let input_path = temp_dir.path().join("test.inp");

    // Create input file with DIIS-only mode
    let input_content = r#"
*GEOM
C 0.0 0.0 0.0
H 1.0 0.0 0.0
*

program = gaussian
method = B3LYP/6-31G*
charge = 0
mult_state_a = 1
mult_state_b = 3
switch_step = 0     # DIIS from step 1
use_gediis = true
"#;

    fs::write(&input_path, input_content).unwrap();

    // Parse and verify
    let input_data = parse_input(&input_path).unwrap();
    assert_eq!(input_data.config.switch_step, 0);
    assert_eq!(input_data.config.use_gediis, true);
}

#[test]
fn test_invalid_switch_step() {
    let temp_dir = TempDir::new().unwrap();
    let input_path = temp_dir.path().join("test.inp");

    // Create input file with invalid switch_step
    let input_content = r#"
*GEOM
C 0.0 0.0 0.0
H 1.0 0.0 0.0
*

program = gaussian
method = B3LYP/6-31G*
switch_step = invalid
"#;

    fs::write(&input_path, input_content).unwrap();

    // Parse should succeed with default value (3)
    let input_data = parse_input(&input_path).unwrap();
    assert_eq!(input_data.config.switch_step, 3); // Should fall back to default
}

#[test]
fn test_optimizer_logic_simulation() {
    // Test the optimizer selection logic for different scenarios

    // Scenario 1: Default hybrid (switch_step = 3, max_steps = 50)
    let config1 = Config {
        switch_step: 3,
        max_steps: 50,
        use_gediis: false,
        ..Config::default()
    };

    // Simulate optimizer selection for different steps
    for step in 0..10 {
        let use_bfgs = if config1.switch_step >= config1.max_steps {
            true
        } else if config1.switch_step == 0 {
            false
        } else {
            step < config1.switch_step
        };

        if step < 3 {
            assert!(use_bfgs, "Steps 0-2 should use BFGS");
        } else {
            assert!(!use_bfgs, "Steps 3+ should use DIIS");
        }
    }

    // Scenario 2: BFGS-only mode (switch_step >= max_steps)
    let config2 = Config {
        switch_step: 50,
        max_steps: 50,
        use_gediis: true, // Should be ignored
        ..Config::default()
    };

    for _step in 0..50 {
        let use_bfgs = config2.switch_step >= config2.max_steps;
        assert!(use_bfgs, "All steps should use BFGS in BFGS-only mode");
    }

    // Scenario 3: DIIS-only mode (switch_step = 0)
    let config3 = Config {
        switch_step: 0,
        max_steps: 50,
        use_gediis: true,
        ..Config::default()
    };

    for step in 0..10 {
        let use_bfgs = if config3.switch_step >= config3.max_steps {
            true
        } else if config3.switch_step == 0 {
            false
        } else {
            step < config3.switch_step
        };

        assert!(!use_bfgs, "All steps should use DIIS in DIIS-only mode");
    }
}

#[test]
fn test_switch_step_integration() {
    let temp_dir = TempDir::new().unwrap();
    let input_path = temp_dir.path().join("integration_test.inp");

    // Create input file with switch_step = 5
    let input_content = r#"
*GEOM
C 0.0 0.0 0.0
H 1.0 0.0 0.0
*

program = gaussian
method = B3LYP/6-31G*
charge = 0
mult_state_a = 1
mult_state_b = 3
switch_step = 5
use_gediis = true
max_steps = 20
"#;

    fs::write(&input_path, input_content).unwrap();

    // Parse the input file
    let input_data = parse_input(&input_path).unwrap();

    // Verify switch_step parameter was parsed correctly
    assert_eq!(input_data.config.switch_step, 5);
    assert_eq!(input_data.config.use_gediis, true);
    assert_eq!(input_data.config.max_steps, 20);

    // Verify optimizer logic works correctly
    let config = &input_data.config;

    // Test BFGS phase (steps 0-4)
    for step in 0..5 {
        let use_bfgs = if config.switch_step >= config.max_steps {
            true
        } else if config.switch_step == 0 {
            false
        } else {
            step < config.switch_step
        };

        assert!(
            use_bfgs,
            "Step {} should use BFGS (< switch_step {})",
            step, config.switch_step
        );
    }

    // Test DIIS phase (steps 5+)
    for step in 5..10 {
        let use_bfgs = if config.switch_step >= config.max_steps {
            true
        } else if config.switch_step == 0 {
            false
        } else {
            step < config.switch_step
        };

        assert!(
            !use_bfgs,
            "Step {} should use DIIS (>= switch_step {})",
            step, config.switch_step
        );
    }
}
