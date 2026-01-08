use algotrading::prelude::*;

#[test]
fn test_integration_smoke() {
    // Rolling stats example
    let mut stats = RollingStats::<f64, 3>::new();
    stats.update(1.0);
    stats.update(2.0);
    stats.update(3.0);

    assert!((stats.mean() - 2.0).abs() < 1e-12);

    // Markov regime switching example
    let mut regime = MarkovSwitching::spy_default();
    regime.update(0.002);
    regime.update(0.004);
    regime.update(-0.001);

    // Just make sure it runs and returns valid probabilities
    assert!(regime.state_probabilities().iter().all(|&p| p >= 0.0 && p <= 1.0));
}
