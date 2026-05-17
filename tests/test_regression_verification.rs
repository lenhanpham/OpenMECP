/// Regression verification tests for QM output extension fixes.
///
/// This test verifies that the fixes for hardcoded .log extensions
/// maintain backward compatibility and work correctly.
use omecp::config::QMProgram;
use omecp::qm_interface::get_output_file_base;

#[test]
fn test_extension_mapping() {
    // Verify that each QM program has the correct output extension
    assert_eq!(get_output_file_base(QMProgram::Gaussian), "log");
    assert_eq!(get_output_file_base(QMProgram::Orca), "out");
    assert_eq!(get_output_file_base(QMProgram::Xtb), "out");
    assert_eq!(get_output_file_base(QMProgram::Bagel), "json");
    assert_eq!(get_output_file_base(QMProgram::Custom), "log");
}

#[test]
fn test_backward_compatibility() {
    // Gaussian should still use .log (backward compatibility)
    assert_eq!(get_output_file_base(QMProgram::Gaussian), "log");

    // Custom interface should use .log (backward compatibility)
    assert_eq!(get_output_file_base(QMProgram::Custom), "log");

    // Other programs should NOT use .log (this was the bug)
    assert_ne!(get_output_file_base(QMProgram::Orca), "log");
    assert_ne!(get_output_file_base(QMProgram::Xtb), "log");
    assert_ne!(get_output_file_base(QMProgram::Bagel), "log");
}
