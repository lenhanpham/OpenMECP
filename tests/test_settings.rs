/// Tests for the configuration management system.
use omecp::config::QMProgram;
use omecp::settings::SettingsManager;
use std::fs;
use std::sync::Mutex;
use tempfile::TempDir;

// Use a mutex to ensure tests run sequentially to avoid directory conflicts
static TEST_MUTEX: Mutex<()> = Mutex::new(());

#[test]
fn test_default_settings() {
    let _guard = TEST_MUTEX.lock().unwrap();

    // Test that default settings work when no config file exists
    let settings = SettingsManager::load().unwrap();

    // Verify default extensions
    assert_eq!(settings.get_output_extension(QMProgram::Gaussian), "log");
    assert_eq!(settings.get_output_extension(QMProgram::Orca), "out");
    assert_eq!(settings.get_output_extension(QMProgram::Xtb), "out");
    assert_eq!(settings.get_output_extension(QMProgram::Bagel), "json");
    assert_eq!(settings.get_output_extension(QMProgram::Custom), "log");
}

#[test]
fn test_custom_extensions_config() {
    let _guard = TEST_MUTEX.lock().unwrap();

    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("settings.ini");

    // Create a custom configuration file
    let config_content = r#"
[extensions]
gaussian = custom_log
orca = custom_out
xtb = custom_xtb
bagel = custom_json
custom = custom_ext

[general]
max_memory = 8GB
default_nprocs = 8
temp_directory = /custom/temp

[logging]
level = debug
file = custom.log
"#;

    fs::write(&config_path, config_content).unwrap();

    // Change to the temp directory so the config file is found
    let original_dir = std::env::current_dir().unwrap();
    std::env::set_current_dir(temp_dir.path()).unwrap();

    // Load settings and verify custom values
    let settings = SettingsManager::load().unwrap();

    assert_eq!(
        settings.get_output_extension(QMProgram::Gaussian),
        "custom_log"
    );
    assert_eq!(settings.get_output_extension(QMProgram::Orca), "custom_out");
    assert_eq!(settings.get_output_extension(QMProgram::Xtb), "custom_xtb");
    assert_eq!(
        settings.get_output_extension(QMProgram::Bagel),
        "custom_json"
    );
    assert_eq!(
        settings.get_output_extension(QMProgram::Custom),
        "custom_ext"
    );

    assert_eq!(settings.general().max_memory, "8GB");
    assert_eq!(settings.general().default_nprocs, 8);
    assert_eq!(settings.general().print_level, 0);

    assert_eq!(settings.logging().level, "debug");
    assert_eq!(settings.logging().file_logging, false);

    // Restore original directory
    std::env::set_current_dir(original_dir).unwrap();
}

#[test]
fn test_partial_config() {
    let _guard = TEST_MUTEX.lock().unwrap();

    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("settings.ini");

    // Create a partial configuration file (only extensions)
    let config_content = r#"
[extensions]
orca = special_out
bagel = special_json
"#;

    fs::write(&config_path, config_content).unwrap();

    // Change to the temp directory
    let original_dir = std::env::current_dir().unwrap();

    // Ensure we're in a clean directory for this test
    std::env::set_current_dir(temp_dir.path()).unwrap();

    // Load settings and verify mixed default/custom values
    let settings = SettingsManager::load().unwrap();

    // Custom values
    assert_eq!(
        settings.get_output_extension(QMProgram::Orca),
        "special_out"
    );
    assert_eq!(
        settings.get_output_extension(QMProgram::Bagel),
        "special_json"
    );

    // Default values (not overridden)
    assert_eq!(settings.get_output_extension(QMProgram::Gaussian), "log");
    assert_eq!(settings.get_output_extension(QMProgram::Xtb), "out");
    assert_eq!(settings.get_output_extension(QMProgram::Custom), "log");

    // Default general settings
    assert_eq!(settings.general().max_memory, "4GB");
    assert_eq!(settings.general().default_nprocs, 4);

    // Restore original directory
    std::env::set_current_dir(original_dir).unwrap();
}

#[test]
fn test_settings_template_generation() {
    let _guard = TEST_MUTEX.lock().unwrap();

    let temp_dir = TempDir::new().unwrap();
    let settings_path = temp_dir.path().join("settings.ini");

    // Generate template
    omecp::settings::SettingsManager::create_template(&settings_path).unwrap();

    // Verify file was created
    assert!(settings_path.exists());

    // Verify content contains expected sections
    let content = fs::read_to_string(&settings_path).unwrap();
    assert!(content.contains("[extensions]"));
    assert!(content.contains("[general]"));
    assert!(content.contains("[logging]"));
    assert!(content.contains("gaussian = log"));
    assert!(content.contains("orca = out"));
    assert!(content.contains("max_memory = 4GB"));
    assert!(content.contains("level = info"));
}
