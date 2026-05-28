/// Tests for print_level configuration functionality.
use omecp::settings::SettingsManager;
use std::fs;
use std::sync::Mutex;
use tempfile::TempDir;

// Use a mutex to ensure tests run sequentially to avoid directory conflicts
static TEST_MUTEX: Mutex<()> = Mutex::new(());

#[test]
fn test_default_print_level() {
    let _guard = TEST_MUTEX
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    // Test that default print_level is 1 (normal)
    let settings = SettingsManager::load().unwrap();
    assert_eq!(settings.general().print_level, 1);
}

#[test]
fn test_custom_print_level() {
    let _guard = TEST_MUTEX
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("settings.ini");

    // Create a configuration file with custom print_level
    let config_content = r#"
[general]
print_level = 2
"#;

    fs::write(&config_path, config_content).unwrap();

    // Change to the temp directory so the config file is found
    let original_dir = std::env::current_dir().unwrap();
    std::env::set_current_dir(temp_dir.path()).unwrap();

    // Load settings and verify custom print_level
    let settings = SettingsManager::load().unwrap();
    assert_eq!(settings.general().print_level, 2);

    // Restore original directory
    std::env::set_current_dir(original_dir).unwrap();
}

#[test]
fn test_invalid_print_level() {
    let _guard = TEST_MUTEX
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("settings.ini");

    // Create a configuration file with invalid print_level
    let config_content = r#"
[general]
print_level = invalid
"#;

    fs::write(&config_path, config_content).unwrap();

    // Change to the temp directory
    let original_dir = std::env::current_dir().unwrap();
    std::env::set_current_dir(temp_dir.path()).unwrap();

    // Loading settings should either fail or fall back to defaults
    let result = SettingsManager::load();
    match result {
        Ok(settings) => {
            // If it doesn't fail, it should fall back to default print_level
            assert_eq!(
                settings.general().print_level,
                1,
                "Should fall back to default print_level"
            );
        }
        Err(_) => {
            // It's also acceptable for it to fail with invalid configuration
        }
    }

    // Restore original directory
    std::env::set_current_dir(original_dir).unwrap();
}

#[test]
fn test_print_level_in_template() {
    let _guard = TEST_MUTEX
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let temp_dir = TempDir::new().unwrap();
    let settings_path = temp_dir.path().join("settings.ini");

    // Generate template
    SettingsManager::create_template(&settings_path).unwrap();

    // Verify template contains print_level setting
    let content = fs::read_to_string(&settings_path).unwrap();
    assert!(content.contains("print_level = 1"));
    assert!(content.contains("# Print level for file operations"));
    assert!(content.contains("# 0 = quiet"));
    assert!(content.contains("# 1 = normal"));
    assert!(content.contains("# 2 = verbose"));
}
