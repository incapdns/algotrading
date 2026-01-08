use algotrading::prelude::*;

#[test]
fn test_realistic_example() {
    // Simulated daily returns (% change)
    let returns = [0.001, 0.002, -0.001, 0.003, 0.004, -0.002, 0.005, -0.001];

    // === Rolling statistics ===
    let mut stats = RollingStats::<f64, 3>::new();

    println!("Rolling stats over last 3 days:");
    for (i, &r) in returns.iter().enumerate() {
        let (mean, std_dev) = stats.update(r);
        println!(
            "Day {:>2}: return={:+.4}, mean={:+.4}, std={:+.6}",
            i + 1,
            r,
            mean,
            std_dev
        );
    }

    // === Markov regime switching ===
    let mut regime = MarkovSwitching::spy_default();

    println!("\nMarkov regime probabilities:");
    for (i, &r) in returns.iter().enumerate() {
        regime.update(r);
        let probs = regime.state_probabilities();
        println!(
            "Day {:>2}: return={:+.4}, p(Bull)={:.3}, p(Bear)={:.3}",
            i + 1,
            r,
            probs[0],
            probs[1]
        );
    }

    // === Sanity checks ===
    assert!(regime.state_probabilities().iter().all(|&p| p >= 0.0 && p <= 1.0));
}
