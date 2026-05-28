use omecp::*;

#[test]
fn test_orca_stability_keywords() {
    // Test that ORCA stability mode adds the correct keywords

    let mut config = config::Config::default();
    config.program = config::QMProgram::Orca;
    config.method = "B3LYP def2-SVP".to_string();
    config.run_mode = config::RunMode::Stable;
    config.nprocs = 4;
    config.mem = "4000".to_string();

    let header = io::build_program_header(&config, 0, 1, "", 0);

    // Should contain ORCA stability keywords
    assert!(header.contains("engrad"));
    assert!(header.contains("stabperform true"));
    assert!(header.contains("StabRestartUHFifUnstable true"));
    assert!(header.contains("%scf"));
    assert!(header.contains("end"));
    assert!(header.contains("!moread"));

    // Should not contain Gaussian-style keywords
    assert!(!header.contains("stable=opt"));
    assert!(!header.contains("force"));
}

#[test]
fn test_orca_normal_mode_keywords() {
    // Test that ORCA normal mode has correct keywords without stability

    let mut config = config::Config::default();
    config.program = config::QMProgram::Orca;
    config.method = "B3LYP def2-SVP".to_string();
    config.run_mode = config::RunMode::Normal;
    config.nprocs = 4;
    config.mem = "4000".to_string();

    let header = io::build_program_header(&config, 0, 1, "", 0);

    // Should contain normal ORCA keywords
    assert!(header.contains("engrad"));
    assert!(header.contains("!moread"));

    // Should NOT contain stability keywords
    assert!(!header.contains("stabperform"));
    assert!(!header.contains("StabRestartUHFifUnstable"));
    assert!(!header.contains("stable=opt"));
}

#[test]
fn test_orca_noread_mode_keywords() {
    // Test that ORCA noread mode excludes moread keywords

    let mut config = config::Config::default();
    config.program = config::QMProgram::Orca;
    config.method = "B3LYP def2-SVP".to_string();
    config.run_mode = config::RunMode::NoRead;
    config.nprocs = 4;
    config.mem = "4000".to_string();

    let header = io::build_program_header(&config, 0, 1, "", 0);

    // Should contain engrad but NOT moread
    assert!(header.contains("engrad"));
    assert!(!header.contains("!moread"));
    assert!(!header.contains("%moinp"));

    // Should not contain stability keywords
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

#[test]
fn test_orca_gbw_file_replacement_different_charges() {
    // Test .gbw file replacement with different charges/multiplicities

    let mut config = config::Config::default();
    config.program = config::QMProgram::Orca;
    config.method = "B3LYP def2-SVP".to_string();
    config.run_mode = config::RunMode::Normal;
    config.charge = -1;
    config.mult_state_a = 2;
    config.charge = 1;
    config.mult_state_b = 2;

    // Test state A (charge=-1, mult=2, should use a.gbw)
    let header_a =
        io::build_program_header(&config, config.charge, config.mult_state_a, "", config.state_a);

    assert!(header_a.contains("running_dir/a.gbw"));
    assert!(header_a.contains("*xyz -1 2"));

    // Test state B (charge=1, mult=2, should use b.gbw)
    let header_b =
        io::build_program_header(&config, config.charge, config.mult_state_b, "", config.state_b);

    assert!(header_b.contains("running_dir/b.gbw"));
    assert!(header_b.contains("*xyz 1 2"));
}

#[test]
fn test_orca_method_modification_consistency() {
    // Test that ORCA method modification is consistent across all run modes

    let programs = vec![config::QMProgram::Orca];
    let run_modes = vec![
        config::RunMode::Normal,
        config::RunMode::Read,
        config::RunMode::NoRead,
        config::RunMode::Stable,
        config::RunMode::InterRead,
    ];

    for program in programs {
        for run_mode in run_modes.iter() {
            let mut config = config::Config::default();
            config.program = program;
            config.method = "B3LYP def2-SVP".to_string();
            config.run_mode = *run_mode;

            let header = io::build_program_header(&config, 0, 1, "", 0);

            // All modes should have engrad
            assert!(
                header.contains("engrad"),
                "Missing engrad for {:?}",
                run_mode
            );

            // Check mode-specific keywords
            match run_mode {
                config::RunMode::NoRead => {
                    assert!(!header.contains("!moread"), "NoRead should not have moread");
                }
                config::RunMode::Stable => {
                    assert!(
                        header.contains("stabperform"),
                        "Stable should have stability keywords"
                    );
                    assert!(header.contains("!moread"), "Stable should have moread");
                }
                _ => {
                    assert!(
                        header.contains("!moread"),
                        "{:?} should have moread",
                        run_mode
                    );
                    assert!(
                        !header.contains("stabperform"),
                        "{:?} should not have stability keywords",
                        run_mode
                    );
                }
            }
        }
    }
}
