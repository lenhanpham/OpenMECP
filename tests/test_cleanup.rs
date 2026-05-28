/// Unit tests for the cleanup module
///
/// Tests file cleanup functionality including:
/// - Whitelist preservation
/// - Step-based .engrad filtering
/// - User-specified extensions
/// - Cleanup enable/disable

use omecp::cleanup::{CleanupConfig, CleanupManager};
use omecp::config::QMProgram;
use std::fs::File;
use std::path::Path;
use tempfile::TempDir;

fn create_test_cleanup_config() -> CleanupConfig {
    // Create a test configuration directly without SettingsManager
    CleanupConfig {
        enabled: true,
        preserve_extensions: vec!["out".to_string(), "gbw".to_string()],
        verbose: 2,
        cleanup_frequency: 5,
        print_level: 2,
    }
}

#[test]
fn test_preserves_whitelist_files() {
    let config = create_test_cleanup_config();
    let manager = CleanupManager::new(config, QMProgram::Orca);

    // These should be preserved (in whitelist or always keep)
    assert!(manager.should_preserve_file("out", Path::new("test.out"), "test.out", 0));
    assert!(manager.should_preserve_file("log", Path::new("test.log"), "test.log", 0));
    assert!(manager.should_preserve_file("in", Path::new("test.in"), "test.in", 0));
    assert!(manager.should_preserve_file("inp", Path::new("test.inp"), "test.inp", 0));
    assert!(manager.should_preserve_file("gbw", Path::new("test.gbw"), "test.gbw", 0));
    assert!(manager.should_preserve_file("input.inp", Path::new("input.inp"), "input.inp", 0));
}

#[test]
fn test_deletes_non_whitelist_files() {
    let config = create_test_cleanup_config();
    let manager = CleanupManager::new(config, QMProgram::Orca);

    // These should be DELETED (not in whitelist)
    assert!(!manager.should_preserve_file("tmp", Path::new("test.tmp"), "test.tmp", 0));
    assert!(!manager.should_preserve_file("scf", Path::new("test.scf"), "test.scf", 0));
    assert!(!manager.should_preserve_file("bak", Path::new("test.bak"), "test.bak", 0));
    assert!(!manager.should_preserve_file("trash", Path::new("test.trash"), "test.trash", 0));
    assert!(!manager.should_preserve_file("lock", Path::new("test.lock"), "test.lock", 0));
}

#[test]
fn test_engrad_step_based_filtering() {
    let config = create_test_cleanup_config();
    let manager = CleanupManager::new(config, QMProgram::Orca);

    // With max_step = 60
    // These .engrad files should be PRESERVED (from latest step)
    assert!(manager.should_preserve_file(
        "engrad",
        Path::new("60_state_A.engrad"),
        "60_state_A.engrad",
        60
    ));
    assert!(manager.should_preserve_file(
        "engrad",
        Path::new("60_state_B.engrad"),
        "60_state_B.engrad",
        60
    ));

    // These .engrad files should be DELETED (from old steps)
    assert!(!manager.should_preserve_file(
        "engrad",
        Path::new("59_state_A.engrad"),
        "59_state_A.engrad",
        60
    ));
    assert!(!manager.should_preserve_file(
        "engrad",
        Path::new("59_state_B.engrad"),
        "59_state_B.engrad",
        60
    ));
    assert!(!manager.should_preserve_file(
        "engrad",
        Path::new("45_state_A.engrad"),
        "45_state_A.engrad",
        60
    ));
    assert!(!manager.should_preserve_file(
        "engrad",
        Path::new("10_state_B.engrad"),
        "10_state_B.engrad",
        60
    ));

    // .engrad files that don't match the pattern should be DELETED
    assert!(!manager.should_preserve_file(
        "engrad",
        Path::new("test.engrad"),
        "test.engrad",
        60
    ));
    assert!(!manager.should_preserve_file(
        "engrad",
        Path::new("random.engrad"),
        "random.engrad",
        60
    ));
}

#[test]
fn test_cleanup_directory_deletes_non_whitelist() {
    let temp_dir = TempDir::new().unwrap();
    let dir_path = temp_dir.path().to_path_buf();

    // Create test files
    let _ = File::create(dir_path.join("test.out")).unwrap(); // Should be preserved
    let _ = File::create(dir_path.join("test.gbw")).unwrap(); // Should be preserved
    let _ = File::create(dir_path.join("test.tmp")).unwrap(); // Should be deleted
    let _ = File::create(dir_path.join("test.scf")).unwrap(); // Should be deleted
    let _ = File::create(dir_path.join("test.trash")).unwrap(); // Should be deleted

    let config = create_test_cleanup_config();
    let manager = CleanupManager::new(config, QMProgram::Orca);

    // Run cleanup
    let result = manager.cleanup_directory(&dir_path);
    assert!(result.is_ok());

    // Verify cleanup - whitelist files preserved
    assert!(dir_path.join("test.out").exists());
    assert!(dir_path.join("test.gbw").exists());
    // Non-whitelist files deleted
    assert!(!dir_path.join("test.tmp").exists());
    assert!(!dir_path.join("test.scf").exists());
    assert!(!dir_path.join("test.trash").exists());
}

#[test]
fn test_cleanup_directory_step_based_engrad() {
    let temp_dir = TempDir::new().unwrap();
    let dir_path = temp_dir.path().to_path_buf();

    // Create .inp files to determine max step
    let _ = File::create(dir_path.join("45_state_A.inp")).unwrap();
    let _ = File::create(dir_path.join("45_state_B.inp")).unwrap();
    let _ = File::create(dir_path.join("60_state_A.inp")).unwrap();
    let _ = File::create(dir_path.join("60_state_B.inp")).unwrap();

    // Create .engrad files
    let _ = File::create(dir_path.join("45_state_A.engrad")).unwrap();
    let _ = File::create(dir_path.join("45_state_B.engrad")).unwrap();
    let _ = File::create(dir_path.join("60_state_A.engrad")).unwrap();
    let _ = File::create(dir_path.join("60_state_B.engrad")).unwrap();

    let config = create_test_cleanup_config();
    let manager = CleanupManager::new(config, QMProgram::Orca);

    // Run cleanup
    let result = manager.cleanup_directory(&dir_path);
    assert!(result.is_ok());

    // Verify - only 60_state_*.engrad files are preserved
    assert!(dir_path.join("60_state_A.engrad").exists());
    assert!(dir_path.join("60_state_B.engrad").exists());
    assert!(!dir_path.join("45_state_A.engrad").exists());
    assert!(!dir_path.join("45_state_B.engrad").exists());
}

#[test]
fn test_respects_user_output_extension() {
    use std::io::Write;

    // Create a temporary omecp_config.cfg file
    let temp_dir = TempDir::new().unwrap();
    let settings_path = temp_dir.path().join("omecp_config.cfg");

    // Write custom settings with ORCA extension set to "custom"
    let mut file = File::create(&settings_path).unwrap();
    writeln!(file, "[extensions]").unwrap();
    writeln!(file, "orca = custom").unwrap();

    // Change to temp directory and load settings
    let old_cwd = std::env::current_dir().unwrap();
    std::env::set_current_dir(temp_dir.path()).unwrap();

    // Load settings from the file we just created
    let settings_manager = omecp::settings::SettingsManager::load().unwrap();
    let config = CleanupConfig::from_settings_manager(&settings_manager, QMProgram::Orca);

    // Restore working directory
    std::env::set_current_dir(old_cwd).unwrap();

    // Should preserve "custom" (user-specified)
    assert!(config.preserve_extensions.contains(&"custom".to_string()));
    // Should also preserve program-specific files (gbw)
    assert!(config.preserve_extensions.contains(&"gbw".to_string()));
    // Should NOT preserve engrad by default (it's step-filtered)
    assert!(!config.preserve_extensions.contains(&"engrad".to_string()));
}

#[test]
fn test_cleanup_disabled() {
    let temp_dir = TempDir::new().unwrap();
    let dir_path = temp_dir.path().to_path_buf();

    // Create test files
    let _ = File::create(dir_path.join("test.tmp")).unwrap();

    let config = CleanupConfig {
        enabled: false,
        preserve_extensions: vec!["out".to_string(), "gbw".to_string()],
        verbose: 1,
        cleanup_frequency: 5,
        print_level: 1,
    };

    let manager = CleanupManager::new(config, QMProgram::Orca);

    // Run cleanup (should not delete anything)
    let result = manager.cleanup_directory(&dir_path);
    assert!(result.is_ok());

    // Verify nothing was deleted
    assert!(dir_path.join("test.tmp").exists());
}
