use omecp::config::{Config, QMProgram, RunMode};
use omecp::io::{
    build_bagel_header, build_gaussian_header, build_orca_header, build_program_header,
    build_program_header_with_basename, build_xtb_header, clean_gaussian_keywords, clean_keywords,
    modify_method_for_run_mode,
};

#[test]
fn test_clean_gaussian_keywords_empty_result() {
    let only_comments = "# This is a comment\n# Another comment";
    let result = clean_gaussian_keywords(only_comments);
    assert_eq!(result, "");
}

#[test]
fn test_clean_gaussian_keywords_mixed_content() {
    let mixed = "# Comment\nTD(NStates=5)\n# Another comment\nRoot=1";
    let result = clean_gaussian_keywords(mixed);
    assert_eq!(result, "TD(NStates=5) Root=1");
}

#[test]
fn test_clean_gaussian_keywords_inline_comments() {
    let inline = "TD(NStates=5) # This is an inline comment\nRoot=1 # Another inline comment";
    let result = clean_gaussian_keywords(inline);
    assert_eq!(result, "TD(NStates=5) Root=1");

    let only_inline = "# This entire line is a comment";
    let result = clean_gaussian_keywords(only_inline);
    assert_eq!(result, "");

    let mixed_inline = "TD(NStates=5) # inline comment\n# full line comment\nRoot=1";
    let result = clean_gaussian_keywords(mixed_inline);
    assert_eq!(result, "TD(NStates=5) Root=1");
}

#[test]
fn test_clean_gaussian_keywords_empty_input() {
    assert_eq!(clean_gaussian_keywords(""), "");
}

#[test]
fn test_clean_gaussian_keywords_whitespace_only() {
    assert_eq!(clean_gaussian_keywords("   \n\t\n   \n"), "");
}

#[test]
fn test_build_gaussian_header_empty_td() {
    let mut config = Config::default();
    config.method = "B3LYP".to_string();
    config.nprocs = 4;
    config.mem = "4GB".to_string();
    config.run_mode = RunMode::Normal;

    let header = build_gaussian_header(&config, 0, 1, "");

    assert!(header.contains("%nprocshared=4 "));
    assert!(header.contains("%mem=4GB "));
    assert!(header.contains("# B3LYP force guess=read nosymm"));
    assert!(header.contains("0 1"));
}

#[test]
fn test_build_gaussian_header_comment_only_td() {
    let mut config = Config::default();
    config.method = "B3LYP".to_string();
    config.nprocs = 4;
    config.mem = "4GB".to_string();
    config.run_mode = RunMode::Normal;

    let td_with_comments = "# This is a comment\n# Another comment";
    let header = build_gaussian_header(&config, 0, 1, td_with_comments);

    assert!(header.contains("%nprocshared=4 "));
    assert!(header.contains("%mem=4GB "));
    assert!(header.contains("# B3LYP force guess=read nosymm"));
    let lines: Vec<&str> = header.lines().collect();
    let route_line = lines.iter().find(|line| line.starts_with('#')).unwrap();
    assert!(!route_line.contains("This is a comment"));
    assert!(!route_line.contains("Another comment"));
}

#[test]
fn test_build_gaussian_header_mixed_td_content() {
    let mut config = Config::default();
    config.method = "B3LYP".to_string();
    config.nprocs = 4;
    config.mem = "4GB".to_string();
    config.run_mode = RunMode::Normal;

    let mixed_td = "# Comment\nTD(NStates=5)\n# Another comment\nRoot=1";
    let header = build_gaussian_header(&config, 0, 1, mixed_td);

    assert!(header.contains("%nprocshared=4 "));
    assert!(header.contains("%mem=4GB "));
    assert!(header.contains("# B3LYP force guess=read TD(NStates=5) Root=1 nosymm"));
    let lines: Vec<&str> = header.lines().collect();
    let route_line = lines.iter().find(|line| line.starts_with('#')).unwrap();
    assert!(!route_line.contains("Comment"));
    assert!(!route_line.contains("Another comment"));
}

#[test]
fn test_clean_keywords() {
    let mixed = "# Comment\nTD(NStates=5) # inline\n# Another comment\nRoot=1";
    let result = clean_keywords(mixed);
    assert_eq!(result, "TD(NStates=5) Root=1");

    let only_comments = "# Only comments\n# More comments";
    let result = clean_keywords(only_comments);
    assert_eq!(result, "");
}

#[test]
fn test_build_orca_header() {
    let mut config = Config::default();
    config.program = QMProgram::Orca;
    config.method = "B3LYP def2-SVP".to_string();
    config.nprocs = 8;
    config.mem = "8000".to_string();
    config.run_mode = RunMode::Normal;

    let header = build_orca_header(&config, 0, 1, "", "test_job");

    assert!(header.contains("%pal nprocs 8 end"));
    assert!(header.contains("%maxcore 8000"));
    assert!(header.contains("! B3LYP def2-SVP engrad"));
    assert!(header.contains("!moread"));
    assert!(header.contains("*xyz 0 1"));
}

#[test]
fn test_build_orca_header_with_tail() {
    let mut config = Config::default();
    config.program = QMProgram::Orca;
    config.method = "B3LYP def2-SVP".to_string();
    config.nprocs = 4;
    config.mem = "4000".to_string();
    config.run_mode = RunMode::Normal;

    let tail_with_comments = "# Comment\n%tddft\n  nroots 5\nend\n# Another comment";
    let header = build_orca_header(&config, -1, 2, tail_with_comments, "test_job");

    assert!(header.contains("B3LYP def2-SVP engrad"));
    assert!(header.contains("%tddft nroots 5 end"));
    assert!(header.contains("!moread"));
    assert!(header.contains("*xyz -1 2"));
    assert!(!header.contains("Comment"));
    assert!(!header.contains("Another comment"));
}

#[test]
fn test_build_orca_header_noread_mode() {
    let mut config = Config::default();
    config.program = QMProgram::Orca;
    config.method = "B3LYP def2-SVP".to_string();
    config.nprocs = 4;
    config.mem = "4000".to_string();
    config.run_mode = RunMode::NoRead;

    let header = build_orca_header(&config, 0, 1, "", "test_job");

    assert!(!header.contains("!moread"));
    assert!(header.contains("! B3LYP def2-SVP engrad"));
}

#[test]
fn test_build_xtb_header() {
    let config = Config::default();
    let header = build_xtb_header(&config, 1, 3, "");

    assert!(header.contains("$chrg 1"));
    assert!(header.contains("$uhf 2")); // mult=3 -> uhf=2
}

#[test]
fn test_build_bagel_header() {
    let mut config = Config::default();
    config.program = QMProgram::Bagel;
    config.basis_set = "cc-pVTZ".to_string();

    let header = build_bagel_header(&config, 0, 1, 2);

    assert!(header.contains("\"bagel\""));
    assert!(header.contains("\"charge\" : 0"));
    assert!(header.contains("\"nspin\" : 0")); // mult=1 -> nspin=0
    assert!(header.contains("\"target\" : 2"));
    assert!(header.contains("\"basis\" : \"cc-pVTZ\""));
}

#[test]
fn test_build_bagel_header_default_basis() {
    let config = Config::default();
    let header = build_bagel_header(&config, -1, 2, 0);

    assert!(header.contains("\"basis\" : \"cc-pVDZ\""));
    assert!(header.contains("\"charge\" : -1"));
    assert!(header.contains("\"nspin\" : 1")); // mult=2 -> nspin=1
}

#[test]
fn test_modify_method_for_run_mode_gaussian() {
    let result = modify_method_for_run_mode(
        "B3LYP/6-31G*",
        QMProgram::Gaussian,
        RunMode::Normal,
    );
    assert_eq!(result, "B3LYP/6-31G* force guess=read");

    let result = modify_method_for_run_mode(
        "B3LYP/6-31G*",
        QMProgram::Gaussian,
        RunMode::NoRead,
    );
    assert_eq!(result, "B3LYP/6-31G* force");

    let result = modify_method_for_run_mode(
        "B3LYP/6-31G*",
        QMProgram::Gaussian,
        RunMode::Stable,
    );
    assert_eq!(result, "B3LYP/6-31G* force stable=opt guess=read");

    let result = modify_method_for_run_mode(
        "B3LYP/6-31G*",
        QMProgram::Gaussian,
        RunMode::Read,
    );
    assert_eq!(result, "B3LYP/6-31G* force guess=read");

    let result = modify_method_for_run_mode(
        "B3LYP/6-31G*",
        QMProgram::Gaussian,
        RunMode::InterRead,
    );
    assert_eq!(result, "B3LYP/6-31G* force guess=read");
}

#[test]
fn test_modify_method_for_run_mode_orca() {
    let result = modify_method_for_run_mode(
        "B3LYP def2-SVP",
        QMProgram::Orca,
        RunMode::Normal,
    );
    assert!(result.contains("B3LYP def2-SVP engrad"));
    assert!(result.contains("!moread"));
    assert!(result.contains("%moinp \"***\""));

    let result = modify_method_for_run_mode(
        "B3LYP def2-SVP",
        QMProgram::Orca,
        RunMode::NoRead,
    );
    assert_eq!(result, "B3LYP def2-SVP engrad");

    let result = modify_method_for_run_mode(
        "B3LYP def2-SVP",
        QMProgram::Orca,
        RunMode::Stable,
    );
    assert!(result.contains("B3LYP def2-SVP engrad"));
    assert!(result.contains("stabperform true"));
    assert!(result.contains("StabRestartUHFifUnstable true"));
    assert!(result.contains("!moread"));
}

#[test]
fn test_modify_method_for_run_mode_orca_gaussian_syntax() {
    let result = modify_method_for_run_mode(
        "B3LYP/6-31G*",
        QMProgram::Orca,
        RunMode::Normal,
    );
    assert!(result.contains("B3LYP 6-31G*"));
    assert!(!result.contains("B3LYP/6-31G*"));
    assert!(result.contains("engrad"));
    assert!(result.contains("!moread"));
}

#[test]
fn test_modify_method_for_run_mode_xtb_bagel() {
    let result = modify_method_for_run_mode("GFN2-xTB", QMProgram::Xtb, RunMode::Normal);
    assert_eq!(result, "GFN2-xTB");

    let result = modify_method_for_run_mode("CASSCF", QMProgram::Bagel, RunMode::Stable);
    assert_eq!(result, "CASSCF");
}

#[test]
fn test_modify_method_for_run_mode_empty_method() {
    let result = modify_method_for_run_mode("", QMProgram::Gaussian, RunMode::Normal);
    assert_eq!(result, "");

    let result = modify_method_for_run_mode("", QMProgram::Orca, RunMode::Normal);
    assert_eq!(result, "");
}

#[test]
fn test_build_program_header_with_dynamic_modification() {
    let mut config = Config::default();
    config.program = QMProgram::Gaussian;
    config.method = "B3LYP/6-31G*".to_string();
    config.run_mode = RunMode::Stable;
    config.nprocs = 4;
    config.mem = "4GB".to_string();

    let header = build_program_header(&config, 0, 1, "", 0);
    assert!(header.contains("B3LYP/6-31G* force stable=opt guess=read"));

    config.program = QMProgram::Orca;
    config.method = "B3LYP def2-SVP".to_string();
    config.run_mode = RunMode::NoRead;

    let header = build_program_header_with_basename(&config, 0, 1, "", 0, "test_job");
    assert!(header.contains("B3LYP def2-SVP engrad"));
    assert!(!header.contains("!moread"));
}

#[test]
fn test_build_program_header_dispatch() {
    let mut config = Config::default();
    config.program = QMProgram::Gaussian;
    config.method = "B3LYP".to_string();
    config.charge = 0;
    config.mult_state_a = 1;
    let header = build_program_header(&config, 0, 1, "", 0);
    assert!(header.contains("%chk=state_A.chk"));
    assert!(header.contains("%nprocshared="));

    config.program = QMProgram::Orca;
    let header = build_program_header_with_basename(&config, 0, 1, "", 0, "test_job");
    assert!(header.contains("%pal nprocs"));
    assert!(header.contains("*xyz"));

    config.program = QMProgram::Xtb;
    let header = build_program_header(&config, 1, 2, "", 0);
    assert!(header.contains("$chrg 1"));
    assert!(header.contains("$uhf 1"));

    config.program = QMProgram::Bagel;
    let header = build_program_header(&config, 0, 1, "", 1);
    assert!(header.contains("\"bagel\""));
    assert!(header.contains("\"target\" : 1"));
}

#[test]
fn test_orca_header_gbw_replacement() {
    let mut config = Config::default();
    config.program = QMProgram::Orca;
    config.method = "B3LYP def2-SVP".to_string();
    config.run_mode = RunMode::Normal;
    config.charge = 0;
    config.mult_state_a = 1;
    config.mult_state_b = 3;

    let header = build_program_header_with_basename(&config, 0, 1, "", 0, "calc");
    assert!(header.contains("calc_state_A.gbw"));
    assert!(!header.contains("***"));

    let header = build_program_header_with_basename(&config, 0, 3, "", 0, "calc");
    assert!(header.contains("calc_state_B.gbw"));
    assert!(!header.contains("***"));

    let header = build_program_header_with_basename(&config, 0, 1, "", 0, "compound_x");
    assert!(header.contains("compound_x_state_A.gbw"));
    assert!(!header.contains("***"));
    assert!(!header.contains("calc"));

    let header = build_program_header_with_basename(&config, 0, 3, "", 0, "compound_x");
    assert!(header.contains("compound_x_state_B.gbw"));
    assert!(!header.contains("***"));
    assert!(!header.contains("calc"));
}
