use omecp::*;

#[test]
fn test_pre_point_program_dispatch() {
    // Test that pre-point calculations dispatch to correct program-specific functions
    // This is a unit test for the dispatch logic

    // Test config for each program
    let programs = vec![
        config::QMProgram::Gaussian,
        config::QMProgram::Orca,
        config::QMProgram::Xtb,
        config::QMProgram::Bagel,
        config::QMProgram::Custom,
    ];

    for program in programs {
        let mut config = config::Config::default();
        config.program = program;
        config.run_mode = config::RunMode::Normal;

        // Set appropriate method for each program
        match program {
            config::QMProgram::Gaussian | config::QMProgram::Custom => {
                config.method = "B3LYP/6-31G*".to_string();
            }
            config::QMProgram::Orca => {
                config.method = "B3LYP def2-SVP".to_string();
            }
            config::QMProgram::Xtb => {
                config.method = "GFN2-xTB".to_string();
            }
            config::QMProgram::Bagel => {
                config.method = "CASSCF".to_string();
            }
        }

        // Build headers to ensure they work with each program
        let header_a = io::build_program_header(
            &config,
            config.charge,
            config.mult_state_a,
            &config.td_state_a,
            config.state_a,
        );

        let _header_b = io::build_program_header(
            &config,
            config.charge,
            config.mult_state_b,
            &config.td_state_b,
            config.state_b,
        );

        // Verify headers are generated correctly for each program
        match program {
            config::QMProgram::Gaussian | config::QMProgram::Custom => {
                if !config.method.is_empty() {
                    assert!(header_a.contains("force"));
                    assert!(header_a.contains("guess=read"));
                }
            }
            config::QMProgram::Orca => {
                assert!(header_a.contains("engrad"));
                assert!(header_a.contains("!moread"));
            }
            config::QMProgram::Xtb => {
                assert!(header_a.contains("$chrg"));
            }
            config::QMProgram::Bagel => {
                assert!(header_a.contains("\"bagel\""));
            }
        }
    }
}

#[test]
fn test_stable_mode_method_modification() {
    // Test that stable mode correctly modifies method strings

    // Test Gaussian stable mode
    let mut config = config::Config::default();
    config.program = config::QMProgram::Gaussian;
    config.method = "B3LYP/6-31G*".to_string();
    config.run_mode = config::RunMode::Stable;

    let header = io::build_program_header(
        &config,
        config.charge,
        config.mult_state_a,
        &config.td_state_a,
        config.state_a,
    );

    // Should contain all stable mode keywords
    assert!(header.contains("force"));
    assert!(header.contains("stable=opt"));
    assert!(header.contains("guess=read"));

    // Test ORCA stable mode
    config.program = config::QMProgram::Orca;
    config.method = "B3LYP def2-SVP".to_string();

    let header = io::build_program_header(&config, config.charge, config.mult_state_a, "", config.state_a);

    // Should contain ORCA stability keywords
    assert!(header.contains("engrad"));
    assert!(header.contains("stabperform"));
    assert!(header.contains("StabRestartUHFifUnstable"));
    assert!(header.contains("!moread"));
}

#[test]
fn test_inter_read_mode_headers() {
    // Test that inter_read mode generates correct headers

    let mut config = config::Config::default();
    config.program = config::QMProgram::Gaussian;
    config.method = "B3LYP/6-31G*".to_string();
    config.run_mode = config::RunMode::InterRead;

    let header = io::build_program_header(
        &config,
        config.charge,
        config.mult_state_a,
        &config.td_state_a,
        config.state_a,
    );

    // Inter_read mode should have normal keywords (guess=(read,mix) is added in pre-point)
    assert!(header.contains("force"));
    assert!(header.contains("guess=read"));
    assert!(!header.contains("stable=opt"));

    // Test ORCA inter_read mode
    config.program = config::QMProgram::Orca;
    config.method = "B3LYP def2-SVP".to_string();

    let header = io::build_program_header(&config, config.charge, config.mult_state_a, "", config.state_a);

    // Should have normal ORCA keywords
    assert!(header.contains("engrad"));
    assert!(header.contains("!moread"));
    assert!(!header.contains("stabperform"));
}

#[test]
fn test_noread_mode_headers() {
    // Test that noread mode excludes wavefunction reading keywords

    let mut config = config::Config::default();
    config.program = config::QMProgram::Gaussian;
    config.method = "B3LYP/6-31G*".to_string();
    config.run_mode = config::RunMode::NoRead;

    let header = io::build_program_header(
        &config,
        config.charge,
        config.mult_state_a,
        &config.td_state_a,
        config.state_a,
    );

    // Should have force but NOT guess=read
    assert!(header.contains("force"));
    assert!(!header.contains("guess=read"));
    assert!(!header.contains("stable=opt"));

    // Test ORCA noread mode
    config.program = config::QMProgram::Orca;
    config.method = "B3LYP def2-SVP".to_string();

    let header = io::build_program_header(&config, config.charge, config.mult_state_a, "", config.state_a);

    // Should have engrad but NOT moread
    assert!(header.contains("engrad"));
    assert!(!header.contains("!moread"));
    assert!(!header.contains("stabperform"));
}

#[test]
fn test_read_mode_headers() {
    // Test that read mode has standard wavefunction reading keywords

    let mut config = config::Config::default();
    config.program = config::QMProgram::Gaussian;
    config.method = "B3LYP/6-31G*".to_string();
    config.run_mode = config::RunMode::Read;

    let header = io::build_program_header(
        &config,
        config.charge,
        config.mult_state_a,
        &config.td_state_a,
        config.state_a,
    );

    // Should have standard keywords
    assert!(header.contains("force"));
    assert!(header.contains("guess=read"));
    assert!(!header.contains("stable=opt"));

    // Test ORCA read mode
    config.program = config::QMProgram::Orca;
    config.method = "B3LYP def2-SVP".to_string();

    let header = io::build_program_header(&config, config.charge, config.mult_state_a, "", config.state_a);

    // Should have standard ORCA keywords
    assert!(header.contains("engrad"));
    assert!(header.contains("!moread"));
    assert!(!header.contains("stabperform"));
}

#[test]
fn test_orca_gbw_file_replacement() {
    // Test that ORCA headers properly replace *** with .gbw file paths

    let mut config = config::Config::default();
    config.program = config::QMProgram::Orca;
    config.method = "B3LYP def2-SVP".to_string();
    config.run_mode = config::RunMode::Normal;
    config.charge = 0;
    config.mult_state_a = 1;
    config.charge = 0;
    config.mult_state_b = 3;

    // Test state A (should use a.gbw)
    let header_a =
        io::build_program_header(&config, config.charge, config.mult_state_a, "", config.state_a);

    assert!(header_a.contains("running_dir/a.gbw"));
    assert!(!header_a.contains("***"));

    // Test state B (should use b.gbw)
    let header_b =
        io::build_program_header(&config, config.charge, config.mult_state_b, "", config.state_b);

    assert!(header_b.contains("running_dir/b.gbw"));
    assert!(!header_b.contains("***"));
}
