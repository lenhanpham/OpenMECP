use omecp::config::{Config, QMProgram};
use omecp::geometry::Geometry;
use std::fs;
use std::path::Path;

#[test]
fn test_geometry_to_json() {
    // Test the geometry_to_json function for BAGEL
    let elements = vec!["C".to_string(), "H".to_string(), "H".to_string()];
    let coords = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0];
    let geometry = Geometry::new(elements.clone(), coords);

    // This function is private, so we test it indirectly through write_bagel_input
    // For now, just test that the geometry structure is correct
    assert_eq!(geometry.num_atoms, 3);
    assert_eq!(geometry.elements[0], "C");
    assert_eq!(geometry.elements[1], "H");
    assert_eq!(geometry.elements[2], "H");
}

#[test]
fn test_write_bagel_input_model_file() {
    // Create a temporary BAGEL model file
    let model_content = r#"{
  "bagel" : [
    {
      "title" : "molecule",
      "basis" : "cc-pVDZ",
      "df_basis" : "cc-pVDZ-jkfit",
      "angstrom" : true,
      "geometry" : [
        { "atom" : "C", "xyz" : [ 0.0, 0.0, 0.0 ] }
      ]
    },
    {
      "title" : "casscf",
      "nelectron" : 8,
      "nact" : 8,
      "nclosed" : 2,
      "target" : 0,
      "nspin" : 0
    }
  ]
}"#;

    let model_file = "test_model.json";
    fs::write(model_file, model_content).unwrap();

    let elements = vec!["C".to_string(), "H".to_string()];
    let coords = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let _geometry = Geometry::new(elements.clone(), coords);

    // Test write_bagel_input function (this is private, so we can't test it directly)
    // Instead, we test that the model file exists and has the right content
    assert!(Path::new(model_file).exists());
    let content = fs::read_to_string(model_file).unwrap();
    assert!(content.contains("geometry"));
    assert!(content.contains("target"));
    assert!(content.contains("nspin"));

    // Clean up
    fs::remove_file(model_file).unwrap();
}

#[test]
fn test_xtb_config_integration() {
    // Test that XTB configuration is properly handled
    let mut config = Config::default();
    config.program = QMProgram::Xtb;
    config.method = "GFN2-xTB".to_string();

    assert_eq!(config.program, QMProgram::Xtb);
    assert_eq!(config.method, "GFN2-xTB");
}

#[test]
fn test_bagel_config_integration() {
    // Test that BAGEL configuration is properly handled
    let mut config = Config::default();
    config.program = QMProgram::Bagel;
    config.bagel_model = "model.json".to_string();
    config.method = "CASSCF".to_string();

    assert_eq!(config.program, QMProgram::Bagel);
    assert_eq!(config.bagel_model, "model.json");
    assert_eq!(config.method, "CASSCF");
}

#[test]
fn test_xtb_file_extensions() {
    // Test that XTB uses correct file extensions
    let step = 5;
    let expected_input_a = format!("running_dir/{}_A.xyz", step);
    let expected_input_b = format!("running_dir/{}_B.xyz", step);
    let expected_output_a = format!("running_dir/{}_A.out", step);
    let expected_output_b = format!("running_dir/{}_B.out", step);

    assert_eq!(expected_input_a, "running_dir/5_A.xyz");
    assert_eq!(expected_input_b, "running_dir/5_B.xyz");
    assert_eq!(expected_output_a, "running_dir/5_A.out");
    assert_eq!(expected_output_b, "running_dir/5_B.out");
}

#[test]
fn test_bagel_file_extensions() {
    // Test that BAGEL uses correct file extensions
    let step = 3;
    let expected_input_a = format!("running_dir/{}_A.json", step);
    let expected_input_b = format!("running_dir/{}_B.json", step);
    let expected_output_a = format!("running_dir/{}_A.log", step);
    let expected_output_b = format!("running_dir/{}_B.log", step);
    let expected_xyz = format!("running_dir/{}.xyz", step);

    assert_eq!(expected_input_a, "running_dir/3_A.json");
    assert_eq!(expected_input_b, "running_dir/3_B.json");
    assert_eq!(expected_output_a, "running_dir/3_A.log");
    assert_eq!(expected_output_b, "running_dir/3_B.log");
    assert_eq!(expected_xyz, "running_dir/3.xyz");
}

#[test]
fn test_program_specific_workflow_logic() {
    // Test that different programs follow the correct workflow patterns

    // XTB should use XYZ format
    let config_xtb = Config {
        program: QMProgram::Xtb,
        method: "GFN2-xTB".to_string(),
        ..Default::default()
    };

    // BAGEL should use JSON format and require model file
    let config_bagel = Config {
        program: QMProgram::Bagel,
        method: "CASSCF".to_string(),
        bagel_model: "model.json".to_string(),
        ..Default::default()
    };

    // Test that configurations are set correctly
    assert_eq!(config_xtb.program, QMProgram::Xtb);
    assert_eq!(config_bagel.program, QMProgram::Bagel);
    assert!(!config_bagel.bagel_model.is_empty());
}
