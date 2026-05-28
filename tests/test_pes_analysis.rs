use omecp::geometry::Geometry;
use omecp::pes_scan::{analyze_scan_results, ScanPointResult};
use std::fs;
use std::path::Path;

#[test]
fn test_scan_result_analysis() {
    // Create mock scan results
    let elements = vec!["C".to_string(), "H".to_string()];
    let coords = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let geometry = Geometry::new(elements, coords);

    let scan_results = vec![
        ScanPointResult {
            coord1: 1.0,
            coord2: 0.0,
            energy_a: -1.0,
            energy_b: -0.9,
            energy_diff: -0.1,
            converged: true,
            num_steps: 5,
            geometry: geometry.clone(),
        },
        ScanPointResult {
            coord1: 1.1,
            coord2: 0.0,
            energy_a: -0.95,
            energy_b: -0.85,
            energy_diff: -0.1,
            converged: true,
            num_steps: 7,
            geometry: geometry.clone(),
        },
        ScanPointResult {
            coord1: 1.2,
            coord2: 0.0,
            energy_a: -0.9,
            energy_b: -0.8,
            energy_diff: -0.1,
            converged: false,
            num_steps: 10,
            geometry: geometry.clone(),
        },
    ];

    // Test analysis function
    let output_file = "test_scan_analysis.txt";
    let result = analyze_scan_results(&scan_results, output_file);
    assert!(result.is_ok());

    // Check that output files were created
    assert!(Path::new(output_file).exists());
    assert!(Path::new(&format!("{}.dat", output_file)).exists());
    assert!(Path::new(&format!("{}_convergence.txt", output_file)).exists());

    // Read and verify summary content
    let summary_content = fs::read_to_string(output_file).unwrap();
    assert!(summary_content.contains("Total scan points: 3"));
    assert!(summary_content.contains("Converged points: 2"));
    assert!(summary_content.contains("Convergence rate: 66.7%"));

    // Clean up test files
    let _ = fs::remove_file(output_file);
    let _ = fs::remove_file(&format!("{}.dat", output_file));
    let _ = fs::remove_file(&format!("{}_convergence.txt", output_file));
}

#[test]
fn test_energy_surface_data_format() {
    // Test energy surface data generation format
    let elements = vec!["H".to_string(), "H".to_string()];
    let coords = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let geometry = Geometry::new(elements, coords);

    let scan_results = vec![
        ScanPointResult {
            coord1: 1.0,
            coord2: 2.0,
            energy_a: -1.123456,
            energy_b: -0.987654,
            energy_diff: -0.135802,
            converged: true,
            num_steps: 3,
            geometry: geometry.clone(),
        },
        ScanPointResult {
            coord1: 1.1,
            coord2: 2.0,
            energy_a: 0.0,
            energy_b: 0.0,
            energy_diff: 0.0,
            converged: false,
            num_steps: 0,
            geometry: geometry.clone(),
        },
    ];

    let result = analyze_scan_results(&scan_results, "test_analysis.txt");
    assert!(result.is_ok());

    // Verify file content
    let output_file = "test_analysis.txt.dat";
    let content = fs::read_to_string(output_file).unwrap();
    assert!(content.contains("1.000000 2.000000"));
    assert!(content.contains("-1.12345600"));
    assert!(content.contains("-0.98765400"));
    assert!(content.contains("NaN NaN NaN 0")); // Failed point

    // Clean up
    let _ = fs::remove_file("test_analysis.txt.dat");
    let _ = fs::remove_file("test_analysis.txt");
    let _ = fs::remove_file("test_analysis.txt_convergence.txt");
}
