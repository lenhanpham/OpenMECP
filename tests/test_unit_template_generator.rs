use omecp::template_generator::{get_default_output_path, is_supported_format};
use std::path::Path;

#[test]
fn test_is_supported_format() {
    assert!(is_supported_format(Path::new("test.xyz")));
    assert!(is_supported_format(Path::new("test.LOG")));
    assert!(is_supported_format(Path::new("test.gjf")));
    assert!(!is_supported_format(Path::new("test.txt")));
    assert!(!is_supported_format(Path::new("test")));
}

#[test]
fn test_get_default_output_path() {
    // Regular geometry files - no suffix added
    let path = get_default_output_path(Path::new("molecule.xyz"));
    assert_eq!(path.to_str().unwrap(), "molecule.inp");

    let path = get_default_output_path(Path::new("/path/to/molecule.xyz"));
    assert_eq!(path.to_str().unwrap(), "molecule.inp");

    let path = get_default_output_path(Path::new("test.gjf"));
    assert_eq!(path.to_str().unwrap(), "test.inp");

    // QM output files - suffix added to prevent overwriting
    let path = get_default_output_path(Path::new("abc.log"));
    assert_eq!(path.to_str().unwrap(), "abc_omecp.inp");

    let path = get_default_output_path(Path::new("calc.out"));
    assert_eq!(path.to_str().unwrap(), "calc_omecp.inp");

    let path = get_default_output_path(Path::new("result.json"));
    assert_eq!(path.to_str().unwrap(), "result_omecp.inp");

    // Case insensitive extension check
    let path = get_default_output_path(Path::new("test.LOG"));
    assert_eq!(path.to_str().unwrap(), "test_omecp.inp");

    let path = get_default_output_path(Path::new("test.OUT"));
    assert_eq!(path.to_str().unwrap(), "test_omecp.inp");
}
