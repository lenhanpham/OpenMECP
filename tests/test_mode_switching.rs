use omecp::*;

#[test]
fn test_mode_switching_headers() {
    // Test that headers are correctly rebuilt after mode switching

    // Create a test config with stable mode
    let mut config = config::Config::default();
    config.method = "B3LYP/6-31G*".to_string();
    config.program = config::QMProgram::Gaussian;
    config.run_mode = config::RunMode::Stable;
    config.nprocs = 4;
    config.mem = "4GB".to_string();
    config.charge = 0;
    config.mult_state_a = 1;
    config.charge = 0;
    config.mult_state_b = 3;

    // Build header with stable mode
    let header_stable = io::build_program_header(
        &config,
        config.charge,
        config.mult_state_a,
        &config.td_state_a,
        config.state_a,
    );

    // Should contain stable=opt keyword
    assert!(header_stable.contains("stable=opt"));
    assert!(header_stable.contains("force"));
    assert!(header_stable.contains("guess=read"));

    // Switch to read mode (simulating the mode switching logic)
    config.run_mode = config::RunMode::Read;

    // Build header with read mode
    let header_read = io::build_program_header(
        &config,
        config.charge,
        config.mult_state_a,
        &config.td_state_a,
        config.state_a,
    );

    // Should NOT contain stable=opt keyword anymore
    assert!(!header_read.contains("stable=opt"));
    // Should still contain force and guess=read
    assert!(header_read.contains("force"));
    assert!(header_read.contains("guess=read"));
}

#[test]
fn test_mode_switching_orca() {
    // Test mode switching with ORCA

    let mut config = config::Config::default();
    config.method = "B3LYP def2-SVP".to_string();
    config.program = config::QMProgram::Orca;
    config.run_mode = config::RunMode::Stable;
    config.nprocs = 4;
    config.mem = "4000".to_string();
    config.charge = 0;
    config.mult_state_a = 1;
    config.charge = 0;
    config.mult_state_b = 3;

    // Build header with stable mode
    let header_stable =
        io::build_program_header(&config, config.charge, config.mult_state_a, "", config.state_a);

    // Should contain ORCA stability keywords
    assert!(header_stable.contains("stabperform"));
    assert!(header_stable.contains("StabRestartUHFifUnstable"));
    assert!(header_stable.contains("engrad"));
    assert!(header_stable.contains("!moread"));

    // Switch to read mode
    config.run_mode = config::RunMode::Read;

    // Build header with read mode
    let header_read =
        io::build_program_header(&config, config.charge, config.mult_state_a, "", config.state_a);

    // Should NOT contain stability keywords anymore
    assert!(!header_read.contains("stabperform"));
    assert!(!header_read.contains("StabRestartUHFifUnstable"));
    // Should still contain engrad and moread
    assert!(header_read.contains("engrad"));
    assert!(header_read.contains("!moread"));
}

#[test]
fn test_inter_read_mode_switching() {
    // Test inter_read mode switching

    let mut config = config::Config::default();
    config.method = "B3LYP/6-31G*".to_string();
    config.program = config::QMProgram::Gaussian;
    config.run_mode = config::RunMode::InterRead;
    config.nprocs = 4;
    config.mem = "4GB".to_string();

    // Build header with inter_read mode
    let header_inter_read = io::build_program_header(
        &config,
        config.charge,
        config.mult_state_a,
        &config.td_state_a,
        config.state_a,
    );

    // Should contain normal keywords (no special inter_read keywords in header)
    assert!(header_inter_read.contains("force"));
    assert!(header_inter_read.contains("guess=read"));
    assert!(!header_inter_read.contains("stable=opt"));

    // Switch to read mode (simulating mode switching after pre-point)
    config.run_mode = config::RunMode::Read;

    // Build header with read mode
    let header_read = io::build_program_header(
        &config,
        config.charge,
        config.mult_state_a,
        &config.td_state_a,
        config.state_a,
    );

    // Should be identical to inter_read header (both use same keywords)
    assert!(header_read.contains("force"));
    assert!(header_read.contains("guess=read"));
    assert!(!header_read.contains("stable=opt"));
}
