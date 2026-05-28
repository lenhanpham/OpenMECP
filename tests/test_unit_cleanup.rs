use omecp::cleanup::{CleanupConfig, CleanupManager};
use omecp::config::QMProgram;
use omecp::settings::SettingsManager;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use tempfile::TempDir;

fn create_test_cleanup_config() -> CleanupConfig {
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

    assert!(manager.should_preserve_file("out", Path::new("test.out"), "test.out", 0));
    assert!(manager.should_preserve_file("log", Path::new("test.log"), "test.log", 0));
    assert!(manager.should_preserve_file("in", Path::new("test.in"), "test.in", 0));
    assert!(manager.should_preserve_file("inp", Path::new("test.inp"), "test.inp", 0));
    assert!(manager.should_preserve_file("gbw", Path::new("test.gbw"), "test.gbw", 0));
    assert!(manager.should_preserve_file(
        "input.inp",
        Path::new("input.inp"),
        "input.inp",
        0
    ));
}

#[test]
fn test_deletes_non_whitelist_files() {
    let config = create_test_cleanup_config();
    let manager = CleanupManager::new(config, QMProgram::Orca);

    assert!(!manager.should_preserve_file("tmp", Path::new("test.tmp"), "test.tmp", 0));
    assert!(!manager.should_preserve_file("scf", Path::new("test.scf"), "test.scf", 0));
    assert!(!manager.should_preserve_file("bak", Path::new("test.bak"), "test.bak", 0));
    assert!(!manager.should_preserve_file(
        "trash",
        Path::new("test.trash"),
        "test.trash",
        0
    ));
    assert!(!manager.should_preserve_file("lock", Path::new("test.lock"), "test.lock", 0));
}

#[test]
fn test_engrad_step_based_filtering() {
    let config = create_test_cleanup_config();
    let manager = CleanupManager::new(config, QMProgram::Orca);

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

    let _ = File::create(dir_path.join("test.out")).unwrap(); // Preserved
    let _ = File::create(dir_path.join("test.gbw")).unwrap(); // Preserved
    let _ = File::create(dir_path.join("test.tmp")).unwrap(); // Deleted
    let _ = File::create(dir_path.join("test.scf")).unwrap(); // Deleted
    let _ = File::create(dir_path.join("test.trash")).unwrap(); // Deleted

    let config = create_test_cleanup_config();
    let manager = CleanupManager::new(config, QMProgram::Orca);

    let result = manager.cleanup_directory(&dir_path);
    assert!(result.is_ok());

    assert!(dir_path.join("test.out").exists());
    assert!(dir_path.join("test.gbw").exists());
    assert!(!dir_path.join("test.tmp").exists());
    assert!(!dir_path.join("test.scf").exists());
    assert!(!dir_path.join("test.trash").exists());
}

#[test]
fn test_cleanup_directory_step_based_engrad() {
    let temp_dir = TempDir::new().unwrap();
    let dir_path = temp_dir.path().to_path_buf();

    let _ = File::create(dir_path.join("45_state_A.inp")).unwrap();
    let _ = File::create(dir_path.join("45_state_B.inp")).unwrap();
    let _ = File::create(dir_path.join("60_state_A.inp")).unwrap();
    let _ = File::create(dir_path.join("60_state_B.inp")).unwrap();
    let _ = File::create(dir_path.join("45_state_A.engrad")).unwrap();
    let _ = File::create(dir_path.join("45_state_B.engrad")).unwrap();
    let _ = File::create(dir_path.join("60_state_A.engrad")).unwrap();
    let _ = File::create(dir_path.join("60_state_B.engrad")).unwrap();

    let config = create_test_cleanup_config();
    let manager = CleanupManager::new(config, QMProgram::Orca);

    let result = manager.cleanup_directory(&dir_path);
    assert!(result.is_ok());

    assert!(dir_path.join("60_state_A.engrad").exists());
    assert!(dir_path.join("60_state_B.engrad").exists());
    assert!(!dir_path.join("45_state_A.engrad").exists());
    assert!(!dir_path.join("45_state_B.engrad").exists());
}

#[test]
fn test_respects_user_output_extension() {
    let temp_dir = TempDir::new().unwrap();
    let settings_path = temp_dir.path().join("omecp_config.cfg");

    let mut file = File::create(&settings_path).unwrap();
    writeln!(file, "[extensions]").unwrap();
    writeln!(file, "orca = custom").unwrap();

    let old_cwd = std::env::current_dir().unwrap();
    std::env::set_current_dir(temp_dir.path()).unwrap();

    let settings_manager = SettingsManager::load().unwrap();
    let config = CleanupConfig::from_settings_manager(&settings_manager, QMProgram::Orca);

    std::env::set_current_dir(old_cwd).unwrap();

    assert!(config.preserve_extensions.contains(&"custom".to_string()));
    assert!(config.preserve_extensions.contains(&"gbw".to_string()));
    assert!(!config.preserve_extensions.contains(&"engrad".to_string()));
}

#[test]
fn test_cleanup_disabled() {
    let temp_dir = TempDir::new().unwrap();
    let dir_path = temp_dir.path().to_path_buf();

    let _ = File::create(dir_path.join("test.tmp")).unwrap();

    let config = CleanupConfig {
        enabled: false,
        preserve_extensions: vec!["out".to_string(), "gbw".to_string()],
        verbose: 1,
        cleanup_frequency: 5,
        print_level: 1,
    };

    let manager = CleanupManager::new(config, QMProgram::Orca);

    let result = manager.cleanup_directory(&dir_path);
    assert!(result.is_ok());

    assert!(dir_path.join("test.tmp").exists());
}
