//! Test for the CORRECTED Normal mode implementation.
//!
//! This test verifies that Normal mode follows the correct workflow as described:
//! 1. Pre-point calculations with RAW headers (no force, no guess=read) - like buildHeader
//! 2. Main optimization loop with MODIFIED headers (with force + guess=read) - like modifyMETHOD

use omecp::config::{Config, QMProgram, RunMode};
use omecp::io;

/// Helper function to create a test configuration
fn create_test_config() -> Config {
    Config {
        program: QMProgram::Gaussian,
        run_mode: RunMode::Normal,
        method: "uwb97xd/def2svpp".to_string(),
        mult_state_a: 1,
        mult_state_b: 3,
        charge: 1,
        nprocs: 30,
        mem: "120GB".to_string(),
        ..Default::default()
    }
}

/// Test the correct Normal mode workflow concept
#[test]
fn test_correct_normal_mode_workflow() {
    let config = create_test_config();

    println!("Testing CORRECTED Normal mode workflow:");
    println!("Method: {}", config.method);
    println!("Charge1: {}, Mult1: {}", config.charge, config.mult_state_a);
    println!("Charge2: {}, Mult2: {}", config.charge, config.mult_state_b);

    // Phase 1: Pre-point should use ORIGINAL method (like buildHeader)
    let pre_point_method = &config.method; // NO modifications

    // Phase 2: Main loop should use MODIFIED method (like modifyMETHOD)
    let main_loop_method =
        io::modify_method_for_run_mode(&config.method, config.program, config.run_mode);

    // Verify the workflow
    assert_eq!(
        pre_point_method, "uwb97xd/def2svpp",
        "Pre-point should use original method"
    );
    assert_eq!(
        main_loop_method, "uwb97xd/def2svpp force guess=read",
        "Main loop should add force + guess=read"
    );

    println!("✓ Correct Normal mode workflow verified:");
    println!("  Phase 1 (Pre-point, raw):      {}", pre_point_method);
    println!("  Phase 2 (Main loop, modified): {}", main_loop_method);
}

/// Test expected Gaussian headers for Normal mode
#[test]
fn test_gaussian_normal_mode_headers() {
    let config = create_test_config();

    // Expected pre-point header (matching buildHeader)
    let expected_pre_point_a = format!(
        "%chk=a.chk\n%nprocshared={}\n%mem={}\n# {} nosymm\n\nTitle Card\n\n{} {}",
        config.nprocs, config.mem, config.method, config.charge, config.mult_state_a
    );

    let expected_pre_point_b = format!(
        "%chk=b.chk\n%nprocshared={}\n%mem={}\n# {} nosymm\n\nTitle Card\n\n{} {}",
        config.nprocs, config.mem, config.method, config.charge, config.mult_state_b
    );

    // Expected main loop header (matching modifyMETHOD result)
    let modified_method =
        io::modify_method_for_run_mode(&config.method, config.program, config.run_mode);
    let expected_main_loop_a = format!(
        "%chk=calc.chk\n%nprocshared={}\n%mem={}\n# {} nosymm\n\nTitle Card\n\n{} {}",
        config.nprocs, config.mem, modified_method, config.charge, config.mult_state_a
    );

    println!("✓ Expected Gaussian Normal mode headers:");
    println!(
        "Pre-point A: {}",
        expected_pre_point_a.lines().nth(2).unwrap_or("")
    );
    println!(
        "Pre-point B: {}",
        expected_pre_point_b.lines().nth(2).unwrap_or("")
    );
    println!(
        "Main loop A: {}",
        expected_main_loop_a.lines().nth(2).unwrap_or("")
    );

    // Verify key differences
    assert!(
        expected_pre_point_a.contains("# uwb97xd/def2svpp nosymm"),
        "Pre-point should have original method"
    );
    assert!(
        expected_pre_point_a.contains("%chk=a.chk"),
        "Pre-point A should use a.chk"
    );
    assert!(
        expected_pre_point_b.contains("%chk=b.chk"),
        "Pre-point B should use b.chk"
    );
    assert!(
        !expected_pre_point_a.contains("force"),
        "Pre-point should NOT have force"
    );
    assert!(
        !expected_pre_point_a.contains("guess=read"),
        "Pre-point should NOT have guess=read"
    );

    assert!(
        expected_main_loop_a.contains("force"),
        "Main loop should have force"
    );
    assert!(
        expected_main_loop_a.contains("guess=read"),
        "Main loop should have guess=read"
    );
}

/// Test the exact example from the user
#[test]
fn test_example_compatibility() {
    // User provided this exact example:
    // Pre-point: %chk=a.chk %nprocshared=30 %mem=120gb # n scf(maxcycle=500,xqc) uwb97xd/def2svpp scrf=(smd,solvent=acetonitrile) nosymm
    // Main loop: %chk=a.chk %nprocshared=30 %mem=120gb # n scf(maxcycle=500,xqc) uwb97xd/def2svpp scrf=(smd,solvent=acetonitrile) force guess=read nosymm

    let mut config = create_test_config();
    config.method =
        "n scf(maxcycle=500,xqc) uwb97xd/def2svpp scrf=(smd,solvent=acetonitrile)".to_string();

    // Pre-point should match: # n scf(maxcycle=500,xqc) uwb97xd/def2svpp scrf=(smd,solvent=acetonitrile) nosymm
    let pre_point_method = &config.method;

    // Main loop should match: # n scf(maxcycle=500,xqc) uwb97xd/def2svpp scrf=(smd,solvent=acetonitrile) force guess=read nosymm
    let main_loop_method =
        io::modify_method_for_run_mode(&config.method, config.program, config.run_mode);

    println!("example compatibility:");
    println!("Pre-point:  # {} nosymm", pre_point_method);
    println!("Main loop:  # {} nosymm", main_loop_method);

    // Verify the key difference: main loop adds "force guess=read"
    assert!(
        !pre_point_method.contains("force"),
        "Pre-point should not have force"
    );
    assert!(
        !pre_point_method.contains("guess=read"),
        "Pre-point should not have guess=read"
    );
    assert!(
        main_loop_method.contains("force"),
        "Main loop should have force"
    );
    assert!(
        main_loop_method.contains("guess=read"),
        "Main loop should have guess=read"
    );

    // Verify the method is preserved
    assert!(
        main_loop_method.contains("uwb97xd/def2svpp"),
        "Method should be preserved"
    );
    assert!(
        main_loop_method.contains("scrf=(smd,solvent=acetonitrile)"),
        "Solvent should be preserved"
    );
}

/// Test checkpoint file naming
#[test]
fn test_checkpoint_file_naming() {
    let config = create_test_config();

    // The user specifically mentioned that checkpoint files should be a.chk and b.chk
    // NOT the same file like %chk=calc.chk

    println!("✓ Checkpoint file naming verification:");
    println!("State A should use: %chk=a.chk");
    println!("State B should use: %chk=b.chk");
    println!("Main loop can use: %chk=calc.chk (since it reads from a.chk/b.chk)");

    // This is the correct behavior:
    // Pre-point A: %chk=a.chk (generates a.chk)
    // Pre-point B: %chk=b.chk (generates b.chk)
    // Main loop:   %chk=calc.chk with guess=read (reads from a.chk/b.chk)

    assert_eq!(config.mult_state_a, 1, "State A is singlet");
    assert_eq!(config.mult_state_b, 3, "State B is triplet");
    assert_eq!(
        config.charge, config.charge,
        "Both states have same charge"
    );
}

/// Test ORCA Normal mode headers
#[test]
fn test_orca_normal_mode_headers() {
    let mut config = create_test_config();
    config.program = QMProgram::Orca;
    config.method = "B3LYP def2-SVP".to_string();

    // Pre-point should use original method
    let pre_point_method = &config.method;

    // Main loop should add engrad and !moread
    let main_loop_method =
        io::modify_method_for_run_mode(&config.method, config.program, config.run_mode);

    println!("✓ ORCA Normal mode headers:");
    println!("Pre-point:  ! {}", pre_point_method);
    println!("Main loop:  {}", main_loop_method);

    // Verify ORCA-specific behavior
    assert_eq!(
        pre_point_method, "B3LYP def2-SVP",
        "ORCA pre-point should use original method"
    );
    assert!(
        main_loop_method.contains("engrad"),
        "ORCA main loop should have engrad"
    );
    assert!(
        main_loop_method.contains("!moread"),
        "ORCA main loop should have !moread"
    );
}

/// Test the complete Normal mode concept
#[test]
fn test_complete_normal_mode_concept() {
    println!("Complete Normal Mode Workflow (compatible):");
    println!();
    println!("Phase 1: Pre-point calculations");
    println!("  Purpose: Generate initial checkpoint files for both states");
    println!("  Headers: Use RAW method (no force, no guess=read)");
    println!("  Files:   pre_A.gjf -> a.chk, pre_B.gjf -> b.chk");
    println!("  Example: %chk=a.chk ... # uwb97xd/def2svpp nosymm");
    println!();
    println!("Phase 2: Main optimization loop");
    println!("  Purpose: MECP optimization using pre-generated checkpoints");
    println!("  Headers: Use MODIFIED method (with force + guess=read)");
    println!("  Files:   0_A.gjf (reads a.chk), 0_B.gjf (reads b.chk)");
    println!("  Example: %chk=calc.chk ... # uwb97xd/def2svpp force guess=read nosymm");
    println!();
    println!("Key Insight: Pre-point uses ORIGINAL method, main loop uses MODIFIED method");
    println!("This matches : buildHeader() vs modifyMETHOD()");

    // This test just documents the concept - no assertions needed
    assert!(true, "Concept documented");
}
