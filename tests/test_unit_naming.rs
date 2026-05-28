use omecp::naming::FileNaming;
use std::path::Path;

#[test]
fn test_final_mecp_xyz() {
    let naming = FileNaming::new(Path::new("compound_xyz_123.input"));
    assert_eq!(naming.final_mecp_xyz(), "compound_xyz_123_mecp.xyz");
}

#[test]
fn test_final_mecp_xyz_with_different_extensions() {
    let naming1 = FileNaming::new(Path::new("test.inp"));
    assert_eq!(naming1.final_mecp_xyz(), "test_mecp.xyz");

    let naming2 = FileNaming::new(Path::new("molecule.gjf"));
    assert_eq!(naming2.final_mecp_xyz(), "molecule_mecp.xyz");

    let naming3 = FileNaming::new(Path::new("calc_001.input"));
    assert_eq!(naming3.final_mecp_xyz(), "calc_001_mecp.xyz");
}

#[test]
fn test_final_mecp_xyz_with_path() {
    let naming = FileNaming::new(Path::new("/path/to/compound_xyz_123.input"));
    assert_eq!(naming.final_mecp_xyz(), "compound_xyz_123_mecp.xyz");
}

#[test]
fn test_basename_extraction() {
    let naming = FileNaming::new(Path::new("compound_xyz_123.input"));
    assert_eq!(naming.basename(), "compound_xyz_123");
}
