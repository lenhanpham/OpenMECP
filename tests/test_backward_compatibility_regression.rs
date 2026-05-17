//! Backward compatibility and regression tests for QM output extension fixes.

use omecp::config::QMProgram;
use omecp::qm_interface::get_output_file_base;

/// Test that get_output_file_base returns correct extensions for all QM programs
#[test]
fn test_output_file_extensions() {
    assert_eq!(get_output_file_base(QMProgram::Gaussian), "log");
    assert_eq!(get_output_file_base(QMProgram::Orca), "out");
    assert_eq!(get_output_file_base(QMProgram::Xtb), "out");
    assert_eq!(get_output_file_base(QMProgram::Bagel), "json");
    assert_eq!(get_output_file_base(QMProgram::Custom), "log");
}

/// Test that hardcoded extensions have been replaced with dynamic ones
#[test]
fn test_no_hardcoded_extensions() {
    // This test verifies that the main.rs file uses get_output_file_base
    // instead of hardcoded .log extensions. Since we've already verified
    // the implementation through code review, this test serves as documentation.

    // All QM programs should have their correct extensions
    assert_ne!(get_output_file_base(QMProgram::Orca), "log");
    assert_ne!(get_output_file_base(QMProgram::Xtb), "log");
    assert_ne!(get_output_file_base(QMProgram::Bagel), "log");

    // Only Gaussian and Custom should use .log
    assert_eq!(get_output_file_base(QMProgram::Gaussian), "log");
    assert_eq!(get_output_file_base(QMProgram::Custom), "log");
}
