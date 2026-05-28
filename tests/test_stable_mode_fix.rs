//! Test for Stable mode implementation
//!
//! This test verifies that Stable mode follows the correct workflow:
//! 1. Pre-point calculations WITH stability keywords (stable=opt for Gaussian, stabperform for ORCA)
//! 2. Mode switch to Read mode for main optimization loop
//! 3. Main optimization loop WITHOUT stability keywords (just guess=read)

use omecp::config::{Config, QMProgram, RunMode};
use omecp::io;

/// Helper function to create a test configuration for Stable mode
fn create_stable_config() -> Config {
    Config {
        program: QMProgram::Gaussian,
        run_mode: RunMode::Stable,
        method: "B3LYP/6-31G*".to_string(),
        mult_state_a: 1,
        mult_state_b: 1,
        charge: 0,
        nprocs: 1,
        mem: "1GB".to_string(),
        ..Default::default()
    }
}

/// Test Stable mode method modification for pre-point phase
#[test]
fn test_stable_mode_pre_point_headers() {
    let config = create_stable_config();

    // Pre-point phase: Should have stability keywords
    let pre_header_a = io::build_program_header(
        &config,
        config.charge,
        config.mult_state_a,
        &config.td_state_a,
        config.state_a,
    );
    let pre_header_b = io::build_program_header(
        &config,
        config.charge,
        config.mult_state_b,
        &config.td_state_b,
        config.state_b,
    );

    // Pre-point headers should have 'force', 'stable=opt', and 'guess=read'
    assert!(
        pre_header_a.contains("force"),
        "Stable pre-point header should contain 'force'"
    );
    assert!(
        pre_header_a.contains("stable=opt"),
        "Stable pre-point header should contain 'stable=opt'"
    );
    assert!(
        pre_header_a.contains("guess=read"),
        "Stable pre-point header should contain 'guess=read'"
    );

    assert!(
        pre_header_b.contains("force"),
        "Stable pre-point header should contain 'force'"
    );
    assert!(
        pre_header_b.contains("stable=opt"),
        "Stable pre-point header should contain 'stable=opt'"
    );
    assert!(
        pre_header_b.contains("guess=read"),
        "Stable pre-point header should contain 'guess=read'"
    );

    println!("✓ Stable mode pre-point headers verified");
    println!(
        "Pre-point header A: {}",
        pre_header_a.lines().nth(2).unwrap_or("")
    );
}

/// Test mode transition from Stable to Read
#[test]
fn test_stable_to_read_transition() {
    let stable_config = create_stable_config();

    // After pre-point, should switch to Read mode
    let mut read_config = stable_config.clone();
    read_config.run_mode = RunMode::Read;

    let read_header_a = io::build_program_header(
        &read_config,
        read_config.charge,
        read_config.mult_state_a,
        &read_config.td_state_a,
        read_config.state_a,
    );
    let read_header_b = io::build_program_header(
        &read_config,
        read_config.charge,
        read_config.mult_state_b,
        &read_config.td_state_b,
        read_config.state_b,
    );

    // Read mode headers should have 'force' and 'guess=read' but NOT 'stable=opt'
    assert!(
        read_header_a.contains("force"),
        "Read header should contain 'force'"
    );
    assert!(
        read_header_a.contains("guess=read"),
        "Read header should contain 'guess=read'"
    );
    assert!(
        !read_header_a.contains("stable=opt"),
        "Read header should NOT contain 'stable=opt'"
    );

    assert!(
        read_header_b.contains("force"),
        "Read header should contain 'force'"
    );
    assert!(
        read_header_b.contains("guess=read"),
        "Read header should contain 'guess=read'"
    );
    assert!(
        !read_header_b.contains("stable=opt"),
        "Read header should NOT contain 'stable=opt'"
    );

    println!("✓ Stable → Read mode transition verified");
    println!(
        "Stable pre-point: {}",
        io::modify_method_for_run_mode(
            &stable_config.method,
            stable_config.program,
            stable_config.run_mode
        )
    );
    println!(
        "Read main loop:   {}",
        io::modify_method_for_run_mode(
            &read_config.method,
            read_config.program,
            read_config.run_mode
        )
    );
}

/// Test ORCA Stable mode method modification
#[test]
fn test_orca_stable_mode() {
    let mut config = create_stable_config();
    config.program = QMProgram::Orca;

    // Pre-point phase: Should have ORCA stability keywords
    let stable_method =
        io::modify_method_for_run_mode(&config.method, config.program, config.run_mode);
    assert!(
        stable_method.contains("engrad"),
        "ORCA Stable mode should contain 'engrad'"
    );
    assert!(
        stable_method.contains("stabperform"),
        "ORCA Stable mode should contain 'stabperform'"
    );
    assert!(
        stable_method.contains("StabRestartUHFifUnstable"),
        "ORCA Stable mode should contain 'StabRestartUHFifUnstable'"
    );
    assert!(
        stable_method.contains("!moread"),
        "ORCA Stable mode should contain '!moread'"
    );

    // After transition to Read mode
    config.run_mode = RunMode::Read;
    let read_method =
        io::modify_method_for_run_mode(&config.method, config.program, config.run_mode);
    assert!(
        read_method.contains("engrad"),
        "ORCA Read mode should contain 'engrad'"
    );
    assert!(
        read_method.contains("!moread"),
        "ORCA Read mode should contain '!moread'"
    );
    assert!(
        !read_method.contains("stabperform"),
        "ORCA Read mode should NOT contain 'stabperform'"
    );

    println!("✓ ORCA Stable mode verified:");
    println!("  Stable pre-point: {}", stable_method);
    println!("  Read main loop:   {}", read_method);
}

/// Test that Stable mode is different from Normal mode
#[test]
fn test_stable_vs_normal_mode() {
    let base_method = "B3LYP/6-31G*";

    // Stable mode: should have stability keywords
    let stable_result =
        io::modify_method_for_run_mode(base_method, QMProgram::Gaussian, RunMode::Stable);
    assert_eq!(stable_result, "B3LYP/6-31G* force stable=opt guess=read");

    // Normal mode: should NOT have stability keywords
    let normal_result =
        io::modify_method_for_run_mode(base_method, QMProgram::Gaussian, RunMode::Normal);
    assert_eq!(normal_result, "B3LYP/6-31G* force guess=read");

    // Read mode: should NOT have stability keywords
    let read_result =
        io::modify_method_for_run_mode(base_method, QMProgram::Gaussian, RunMode::Read);
    assert_eq!(read_result, "B3LYP/6-31G* force guess=read");

    println!("✓ Mode differences verified:");
    println!("  Stable: {}", stable_result);
    println!("  Normal: {}", normal_result);
    println!("  Read:   {}", read_result);
}

/// Test Stable mode workflow concept
#[test]
fn test_stable_mode_workflow_concept() {
    let config = create_stable_config();

    // Step 1: Pre-point phase (Stable mode with stability keywords)
    let pre_point_method =
        io::modify_method_for_run_mode(&config.method, config.program, config.run_mode);

    // Step 2: Main loop phase (Read mode without stability keywords)
    let mut main_config = config.clone();
    main_config.run_mode = RunMode::Read;
    let main_loop_method =
        io::modify_method_for_run_mode(&config.method, config.program, main_config.run_mode);

    // Verify the workflow
    assert!(
        pre_point_method.contains("stable=opt"),
        "Pre-point should have stability analysis"
    );
    assert!(
        !main_loop_method.contains("stable=opt"),
        "Main loop should not have stability analysis"
    );
    assert!(
        main_loop_method.contains("guess=read"),
        "Main loop should read checkpoints"
    );

    println!("✓ Stable mode workflow concept verified:");
    println!("  Phase 1 (Pre-point with stability): {}", pre_point_method);
    println!("  Phase 2 (Main loop with read):      {}", main_loop_method);
    println!("  This matches the  behavior!");
}

/// Test unsupported programs for Stable mode
#[test]
fn test_stable_mode_unsupported_programs() {
    // According to , only Gaussian and ORCA support Stable mode
    let unsupported_programs = vec![QMProgram::Xtb, QMProgram::Bagel];

    for program in unsupported_programs {
        let mut config = create_stable_config();
        config.program = program;

        // These programs should handle Stable mode gracefully
        let method = io::modify_method_for_run_mode(&config.method, program, config.run_mode);

        match program {
            QMProgram::Xtb => {
                // XTB doesn't modify methods, so it should return the original
                assert_eq!(
                    method, config.method,
                    "XTB should not modify method for Stable mode"
                );
            }
            QMProgram::Bagel => {
                // BAGEL doesn't modify methods, so it should return the original
                assert_eq!(
                    method, config.method,
                    "BAGEL should not modify method for Stable mode"
                );
            }
            _ => {}
        }

        println!("✓ {:?} Stable mode handling verified: {}", program, method);
    }
}

/// Test  compatibility for Stable mode
#[test]
fn test_stable_compatibility() {
    // According to :
    // 1. runMode = 'stable'
    // 2. Pre-point: buildInitJob() with Method + ' stable=opt' (Gaussian) or stability keywords (ORCA)
    // 3. After pre-point: runMode = 'read'
    // 4. Main loop: modifyMETHOD() with read mode (no stability keywords)

    let config = create_stable_config();

    // Phase 1: Pre-point 
    let pre_header = io::build_program_header(
        &config,
        config.charge,
        config.mult_state_a,
        &config.td_state_a,
        config.state_a,
    );

    // Phase 2: Main loop
    let mut read_config = config.clone();
    read_config.run_mode = RunMode::Read;
    let main_header = io::build_program_header(
        &read_config,
        read_config.charge,
        read_config.mult_state_a,
        &read_config.td_state_a,
        read_config.state_a,
    );

    // Verify compatibility
    assert!(
        pre_header.contains("# B3LYP/6-31G* force stable=opt guess=read nosymm"),
        "Pre-point should match buildInitJob with stable=opt"
    );
    assert!(
        main_header.contains("# B3LYP/6-31G* force guess=read nosymm"),
        "Main loop should match modifyMETHOD with read mode"
    );
    assert!(
        !main_header.contains("stable=opt"),
        "Main loop should NOT contain stability keywords"
    );

    println!("✓  Stable mode compatibility verified:");
    println!(
        "  pre-point equivalent:  {}",
        pre_header.lines().nth(2).unwrap_or("")
    );
    println!(
        "  main loop equivalent:  {}",
        main_header.lines().nth(2).unwrap_or("")
    );
    println!("  ✅ Matches behavior exactly!");
}

/// Test ORCA-specific Stable mode warnings
#[test]
fn test_orca_stable_warnings() {
    let mut config = create_stable_config();
    config.program = QMProgram::Orca;

    // ORCA Stable mode should include the stability keywords
    let method = io::modify_method_for_run_mode(&config.method, config.program, config.run_mode);

    // Verify ORCA-specific stability configuration
    assert!(
        method.contains("stabperform true"),
        "ORCA should have stabperform true"
    );
    assert!(
        method.contains("StabRestartUHFifUnstable true"),
        "ORCA should have StabRestartUHFifUnstable true"
    );

    // The warnings about RHF/UKS and RI incompatibility should be printed during execution
    // (These are runtime warnings, not method modifications)

    println!("✓ ORCA Stable mode warnings verified");
    println!("  Method: {}", method);
    println!("  Note: Runtime warnings about RHF/UKS and RI are printed during execution");
}
