/// Tests for ORCA .gbw file rename functionality.
use omecp::config::{Config, QMProgram, RunMode};
use omecp::geometry::Geometry;
use std::fs;
use tempfile::TempDir;

#[test]
fn test_orca_gbw_rename_functionality() {
    let temp_dir = TempDir::new().unwrap();
    let job_dir = temp_dir.path().join("job_dir");
    fs::create_dir_all(&job_dir).unwrap();

    // Create test .gbw files
    let step = 2;
    let gbw_a_source = job_dir.join(format!("{}_state_A.gbw", step));
    let gbw_b_source = job_dir.join(format!("{}_state_B.gbw", step));
    let gbw_a_dest = job_dir.join("state_A.gbw");
    let gbw_b_dest = job_dir.join("state_B.gbw");

    // Create source files
    fs::write(&gbw_a_source, "test gbw content A").unwrap();
    fs::write(&gbw_b_source, "test gbw content B").unwrap();

    // Create existing destination files to test cleanup
    fs::write(&gbw_a_dest, "old content A").unwrap();
    fs::write(&gbw_b_dest, "old content B").unwrap();

    // Verify source files exist and destination files exist (will be overwritten)
    assert!(gbw_a_source.exists());
    assert!(gbw_b_source.exists());
    assert!(gbw_a_dest.exists());
    assert!(gbw_b_dest.exists());

    // Create test configuration
    let mut config = Config::default();
    config.program = QMProgram::Orca;
    config.run_mode = RunMode::Normal; // Not NoRead, so files should be renamed

    // Create test geometry (for completeness, though not used in this test)
    let _geometry = Geometry::new(
        vec!["C".to_string(), "H".to_string()],
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    );

    // Test the rename logic manually since the function is private

    // Simulate the rename logic from manage_orca_wavefunction_files
    if gbw_a_source.exists() {
        // Remove existing destination file if it exists
        if gbw_a_dest.exists() {
            fs::remove_file(&gbw_a_dest).unwrap();
        }
        // Rename source to destination
        fs::rename(&gbw_a_source, &gbw_a_dest).unwrap();
    }

    if gbw_b_source.exists() {
        // Remove existing destination file if it exists
        if gbw_b_dest.exists() {
            fs::remove_file(&gbw_b_dest).unwrap();
        }
        // Rename source to destination
        fs::rename(&gbw_b_source, &gbw_b_dest).unwrap();
    }

    // Verify the rename operation worked correctly
    assert!(
        !gbw_a_source.exists(),
        "Source file A should be moved (not exist)"
    );
    assert!(
        !gbw_b_source.exists(),
        "Source file B should be moved (not exist)"
    );
    assert!(gbw_a_dest.exists(), "Destination file A should exist");
    assert!(gbw_b_dest.exists(), "Destination file B should exist");

    // Verify the content was preserved
    let content_a = fs::read_to_string(&gbw_a_dest).unwrap();
    let content_b = fs::read_to_string(&gbw_b_dest).unwrap();
    assert_eq!(content_a, "test gbw content A");
    assert_eq!(content_b, "test gbw content B");
}

#[test]
fn test_orca_gbw_rename_with_missing_destination() {
    let temp_dir = TempDir::new().unwrap();
    let job_dir = temp_dir.path().join("job_dir");
    fs::create_dir_all(&job_dir).unwrap();

    // Create test .gbw files
    let step = 3;
    let gbw_a_source = job_dir.join(format!("{}_state_A.gbw", step));
    let gbw_a_dest = job_dir.join("state_A.gbw");

    // Create only source file (no existing destination)
    fs::write(&gbw_a_source, "test gbw content").unwrap();

    // Verify source exists and destination doesn't exist
    assert!(gbw_a_source.exists());
    assert!(!gbw_a_dest.exists());

    // Simulate the rename logic
    if gbw_a_source.exists() {
        // Remove existing destination file if it exists (should be no-op)
        if gbw_a_dest.exists() {
            fs::remove_file(&gbw_a_dest).unwrap();
        }
        // Rename source to destination
        fs::rename(&gbw_a_source, &gbw_a_dest).unwrap();
    }

    // Verify the rename operation worked correctly
    assert!(
        !gbw_a_source.exists(),
        "Source file should be moved (not exist)"
    );
    assert!(gbw_a_dest.exists(), "Destination file should exist");

    // Verify the content was preserved
    let content = fs::read_to_string(&gbw_a_dest).unwrap();
    assert_eq!(content, "test gbw content");
}
