/// Unit tests for the naming module
///
/// Tests dynamic file naming functionality

use omecp::naming::FileNaming;
use std::path::Path;

#[test]
fn test_basename_extraction() {
    let naming = FileNaming::new(Path::new("compound_xyz_123.input"));
    assert_eq!(naming.basename(), "compound_xyz_123");

    let naming2 = FileNaming::new(Path::new("/path/to/test_file.inp"));
    assert_eq!(naming2.basename(), "test_file");

    let naming3 = FileNaming::new(Path::new("simple"));
    assert_eq!(naming3.basename(), "simple");
}

#[test]
fn test_checkpoint_file_names() {
    let naming = FileNaming::new(Path::new("compound_xyz_123.input"));

    assert_eq!(naming.state_a_chk(), "compound_xyz_123_state_A.chk");
    assert_eq!(naming.state_b_chk(), "compound_xyz_123_state_B.chk");
    assert_eq!(naming.a_chk(), "compound_xyz_123_a.chk");
    assert_eq!(naming.b_chk(), "compound_xyz_123_b.chk");
}

#[test]
fn test_checkpoint_file_paths() {
    let naming = FileNaming::new(Path::new("compound_xyz_123.input"));

    assert_eq!(
        naming.state_a_chk_path("job_dir"),
        "job_dir/compound_xyz_123_state_A.chk"
    );
    assert_eq!(
        naming.state_b_chk_path("job_dir"),
        "job_dir/compound_xyz_123_state_B.chk"
    );
    assert_eq!(naming.a_chk_path("job_dir"), "job_dir/compound_xyz_123_a.chk");
    assert_eq!(naming.b_chk_path("job_dir"), "job_dir/compound_xyz_123_b.chk");
}

#[test]
fn test_wavefunction_file_paths() {
    let naming = FileNaming::new(Path::new("compound_xyz_123.input"));

    assert_eq!(
        naming.state_a_gbw("job_dir"),
        "job_dir/compound_xyz_123_state_A.gbw"
    );
    assert_eq!(
        naming.state_b_gbw("job_dir"),
        "job_dir/compound_xyz_123_state_B.gbw"
    );
    assert_eq!(naming.a_gbw("job_dir"), "job_dir/compound_xyz_123_a.gbw");
    assert_eq!(naming.b_gbw("job_dir"), "job_dir/compound_xyz_123_b.gbw");
}

#[test]
fn test_pre_point_files() {
    let naming = FileNaming::new(Path::new("compound_xyz_123.input"));

    assert_eq!(
        naming.pre_a("job_dir", "inp"),
        "job_dir/compound_xyz_123_pre_A.inp"
    );
    assert_eq!(
        naming.pre_b("job_dir", "gjf"),
        "job_dir/compound_xyz_123_pre_B.gjf"
    );
    assert_eq!(
        naming.pre_a_chk("job_dir"),
        "job_dir/compound_xyz_123_pre_A.chk"
    );
    assert_eq!(
        naming.pre_b_chk("job_dir"),
        "job_dir/compound_xyz_123_pre_B.chk"
    );
    assert_eq!(
        naming.pre_a_gbw("job_dir"),
        "job_dir/compound_xyz_123_pre_A.gbw"
    );
    assert_eq!(
        naming.pre_b_gbw("job_dir"),
        "job_dir/compound_xyz_123_pre_B.gbw"
    );
}

#[test]
fn test_step_files() {
    let naming = FileNaming::new(Path::new("compound_xyz_123.input"));

    assert_eq!(
        naming.step_state_a("job_dir", 0, "gjf"),
        "job_dir/compound_xyz_123_0_state_A.gjf"
    );
    assert_eq!(
        naming.step_state_b("job_dir", 5, "inp"),
        "job_dir/compound_xyz_123_5_state_B.inp"
    );
    assert_eq!(
        naming.step_state_a_gbw("job_dir", 10),
        "job_dir/compound_xyz_123_10_state_A.gbw"
    );
    assert_eq!(
        naming.step_state_b_gbw("job_dir", 20),
        "job_dir/compound_xyz_123_20_state_B.gbw"
    );
    assert_eq!(
        naming.step_state_a_engrad("job_dir", 15),
        "job_dir/compound_xyz_123_15_state_A.engrad"
    );
    assert_eq!(
        naming.step_state_b_engrad("job_dir", 25),
        "job_dir/compound_xyz_123_25_state_B.engrad"
    );
}

#[test]
fn test_pes_analysis_files() {
    let naming = FileNaming::new(Path::new("compound_xyz_123.input"));

    assert_eq!(
        naming.step_a("job_dir", 0, "gjf"),
        "job_dir/compound_xyz_123_0_A.gjf"
    );
    assert_eq!(
        naming.step_b("job_dir", 1, "inp"),
        "job_dir/compound_xyz_123_1_B.inp"
    );
}

#[test]
fn test_special_mode_files() {
    let naming = FileNaming::new(Path::new("compound_xyz_123.input"));

    assert_eq!(
        naming.drive_file("job_dir", 5, "A", "gjf"),
        "job_dir/compound_xyz_123_drive_5_A.gjf"
    );
    assert_eq!(
        naming.neb_file("job_dir", 10, "B", "inp"),
        "job_dir/compound_xyz_123_neb_10_B.inp"
    );
}

#[test]
fn test_orca_basename() {
    let naming = FileNaming::new(Path::new("compound_xyz_123.input"));

    assert_eq!(naming.orca_basename("job_dir"), "job_dir/compound_xyz_123");
}

#[test]
fn test_fallback_basename() {
    // Test with a path that has no file stem
    let naming = FileNaming::new(Path::new(""));
    assert_eq!(naming.basename(), "mecp_job");
}

#[test]
fn test_different_extensions() {
    let naming1 = FileNaming::new(Path::new("test.input"));
    let naming2 = FileNaming::new(Path::new("test.inp"));
    let naming3 = FileNaming::new(Path::new("test.xyz"));

    // All should have the same basename
    assert_eq!(naming1.basename(), "test");
    assert_eq!(naming2.basename(), "test");
    assert_eq!(naming3.basename(), "test");

    // And generate the same file names
    assert_eq!(naming1.state_a_chk(), naming2.state_a_chk());
    assert_eq!(naming2.state_a_chk(), naming3.state_a_chk());
}

#[test]
fn test_complex_basename() {
    let naming = FileNaming::new(Path::new("my-compound_v2.3_test.input"));
    assert_eq!(naming.basename(), "my-compound_v2.3_test");
    assert_eq!(
        naming.state_a_chk(),
        "my-compound_v2.3_test_state_A.chk"
    );
}
