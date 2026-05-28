use std::collections::VecDeque;
use omecp::gediis::{
    compute_dynamic_gediis_weight, EnergyRiseTracker, GediisConfig, GediisOptimizer, GediisVariant,
};

#[test]
fn test_energy_rise_tracker() {
    let mut tracker = EnergyRiseTracker::new(2);

    tracker.update(-10.0, 1e-6);
    assert_eq!(tracker.n_rises, 0);

    tracker.update(-9.9, 1e-6); // Energy rose
    assert_eq!(tracker.n_rises, 1);
    assert!(!tracker.do_interpolate);

    tracker.update(-9.8, 1e-6); // Energy rose again
    assert_eq!(tracker.n_rises, 2);
    assert!(!tracker.do_interpolate);

    tracker.update(-9.7, 1e-6); // Third rise
    assert_eq!(tracker.n_rises, 3);
    assert!(tracker.do_interpolate);
}

#[test]
fn test_gediis_config_default() {
    let config = GediisConfig::default();
    assert_eq!(config.max_vectors, 5);
    assert_eq!(config.variant, GediisVariant::RfoDiis);
    assert!(config.auto_switch);
}

#[test]
fn test_variant_selection() {
    let opt = GediisOptimizer::new();

    let mut energies = VecDeque::new();
    energies.push_back(-10.0);
    energies.push_back(-10.1);
    energies.push_back(-10.2);

    // Low error, no rises -> RFO-DIIS
    let variant = opt.select_variant(0.001, false, 3, Some(&energies));
    assert_eq!(variant, GediisVariant::RfoDiis);

    // High error -> Energy-DIIS (if enough points)
    let variant = opt.select_variant(0.01, false, 3, Some(&energies));
    assert_eq!(variant, GediisVariant::EnergyDiis);
}

#[test]
fn test_dynamic_weight_empty() {
    let energies = VecDeque::new();
    let displacements = VecDeque::new();

    let weight = compute_dynamic_gediis_weight(&energies, &displacements);
    assert_eq!(weight, 0.0);
}
