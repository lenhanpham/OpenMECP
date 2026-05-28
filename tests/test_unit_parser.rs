use omecp::config::Config;
use omecp::parser::{
    filter_tail_content, parse_parameter, validate_tail_section,
    validate_tail_section_with_context, validate_tail_section_with_filtering_context,
};

#[test]
fn test_enhanced_error_messages_td_without_parentheses() {
    let result = validate_tail_section("TD NStates=5", "TAIL_a");
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("TD keyword found without required parentheses"));
    assert!(error_msg.contains("TD(NStates=N)"));
}

#[test]
fn test_enhanced_error_messages_unbalanced_parentheses() {
    let result = validate_tail_section("TD(NStates=5", "TAIL_a");
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Unbalanced parentheses"));
    assert!(error_msg.contains("Add 1 closing parenthesis"));
}

#[test]
fn test_enhanced_error_messages_comment_character() {
    let result = validate_tail_section("TD(NStates=5) # comment", "TAIL_a");
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Comment character '#' found"));
    assert!(error_msg.contains("Comments are not allowed in Gaussian route sections"));
}

#[test]
fn test_enhanced_error_messages_case_sensitivity() {
    let result = validate_tail_section("td(nstates=5)", "TAIL_a");
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("case-sensitive"));
}

#[test]
fn test_enhanced_error_messages_root_without_td() {
    let result = validate_tail_section("Root=1", "TAIL_b");
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("ROOT keyword found without TD keyword"));
    assert!(error_msg.contains("TD(NStates=N,Root=M)"));
}

#[test]
fn test_enhanced_error_messages_nstates_without_equals() {
    let result = validate_tail_section("TD(NStates)", "TAIL_a");
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("NStates keyword found without value assignment"));
    assert!(error_msg.contains("NStates=N"));
}

#[test]
fn test_enhanced_error_messages_valid_content() {
    let result = validate_tail_section("TD(NStates=5,Root=1)", "TAIL_a");
    assert!(result.is_ok());
}

#[test]
fn test_enhanced_error_messages_empty_content() {
    let result = validate_tail_section("", "TAIL_a");
    assert!(result.is_ok());
}

#[test]
fn test_graceful_empty_tail_section_handling() {
    assert!(validate_tail_section_with_context("", "TAIL_a").is_ok());
    assert!(validate_tail_section_with_context("", "TAIL_b").is_ok());
}

#[test]
fn test_graceful_empty_tail_section_with_filtering_context() {
    assert!(validate_tail_section_with_filtering_context("", "TAIL_a").is_ok());
    assert!(validate_tail_section_with_filtering_context("", "TAIL_b").is_ok());
}

#[test]
fn test_filter_tail_content_empty_result() {
    let only_comments = "# This is a comment\n# Another comment\n# More comments";
    assert_eq!(filter_tail_content(only_comments), "");

    let mixed_content = "# Comment\nTD(NStates=5)\n# Another comment\nRoot=1";
    assert_eq!(filter_tail_content(mixed_content), "TD(NStates=5) Root=1");

    assert_eq!(filter_tail_content(""), "");
    assert_eq!(filter_tail_content("   \n\t\n   "), "");

    let inline_comments =
        "TD(NStates=5) # This is an inline comment\nRoot=1 # Another inline comment";
    assert_eq!(filter_tail_content(inline_comments), "TD(NStates=5) Root=1");

    let only_inline = "   # This entire line is a comment";
    assert_eq!(filter_tail_content(only_inline), "");

    let mixed_inline = "TD(NStates=5) # inline comment\n# full line comment\nRoot=1";
    assert_eq!(filter_tail_content(mixed_inline), "TD(NStates=5) Root=1");
}

#[test]
fn test_enhanced_error_messages_multiple_spaces() {
    let result = validate_tail_section("TD(NStates=5)  Root=1", "TAIL_a");
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Multiple consecutive spaces"));
    assert!(error_msg.contains("single spaces"));
}

#[test]
fn test_user_friendly_suggestions() {
    let result = validate_tail_section("TD(NStates=5) # This is a comment", "TAIL_a");
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Suggestions for fixing TAIL_a section"));
    assert!(error_msg.contains("Example of valid TAIL_a content"));
}

#[test]
fn test_parameter_parsing_with_inline_comments() {
    let mut config = Config::default();
    let mut fixed_atoms = Vec::new();

    let result = parse_parameter("nprocs = 30 #processors", &mut config, &mut fixed_atoms);
    assert!(result.is_ok());
    assert_eq!(config.nprocs, 30);

    let result = parse_parameter(
        "mem = 120GB # memory to be used",
        &mut config,
        &mut fixed_atoms,
    );
    assert!(result.is_ok());
    assert_eq!(config.mem, "120GB");

    let result = parse_parameter(
        "method = B3LYP/6-31G* # method comment",
        &mut config,
        &mut fixed_atoms,
    );
    assert!(result.is_ok());
    assert_eq!(config.method, "B3LYP/6-31G*");

    let result = parse_parameter(
        "charge = 1 # molecular charge",
        &mut config,
        &mut fixed_atoms,
    );
    assert!(result.is_ok());
    assert_eq!(config.charge, 1);

    let result = parse_parameter(
        "mult_state_a = 3 # triplet state",
        &mut config,
        &mut fixed_atoms,
    );
    assert!(result.is_ok());
    assert_eq!(config.mult_state_a, 3);
}

#[test]
fn test_state_selection_parsing() {
    let mut config = Config::default();
    let mut fixed_atoms = Vec::new();

    let result = parse_parameter("state_a = 1", &mut config, &mut fixed_atoms);
    assert!(result.is_ok());
    assert_eq!(config.state_a, 1);

    let result = parse_parameter("state_b = 2", &mut config, &mut fixed_atoms);
    assert!(result.is_ok());
    assert_eq!(config.state_b, 2);
}
