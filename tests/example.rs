use algotrading::risk::*;
use algotrading::options::*;
use algotrading::matrix::*;

#[test]
fn test_portfolio_simulation_full() -> Result<(), Box<dyn std::error::Error>> {
    // --- Step 1: Portfolio setup ---
    let account_value = 100_000.0;
    let positions = [
        ("AAPL", 50.0, 160.0), // symbol, quantity, strike
        ("MSFT", 30.0, 210.0),
        ("GOOG", 10.0, 2800.0),
    ];

    let spot_prices = [155.0, 205.0, 2750.0];
    let vols = [0.25, 0.20, 0.30];

    // --- Step 2: Risk metrics ---
    let mut var = RollingVaR::<252>::new();
    let mut sharpe = RollingSharpe::<252>::new();
    let mut drawdown = MaxDrawdown::<1000>::new();

    // --- Step 3: Option pricing & greeks ---
    let rate = 0.05;
    let div_yield = 0.01;
    let time_to_expiry = 0.25;

    let mut deltas = [0.0; 3];
    for i in 0..positions.len() {
        let (_, _, strike) = positions[i];
        let price = black_scholes_call(spot_prices[i], strike, rate, div_yield, vols[i], time_to_expiry);
        let delta = delta_call(spot_prices[i], strike, rate, div_yield, vols[i], time_to_expiry);
        println!("{} Option price: {:.2}, Delta: {:.4}", positions[i].0, price, delta);
        deltas[i] = delta;
    }

    // --- Step 4: Portfolio variance ---
    let cov_matrix: [[f64; 3]; 3] = [
        [0.04, 0.01, 0.002],
        [0.01, 0.09, 0.003],
        [0.002, 0.003, 0.16],
    ];
    let chol = Cholesky::decompose(&cov_matrix)?;

    let mut weights = [0.0; 3];
    for i in 0..3 {
        weights[i] = deltas[i] * spot_prices[i] / account_value;
    }

    let port_var = chol.portfolio_variance(&weights);
    println!("Portfolio variance: {:.4}", port_var);
    println!("Portfolio volatility: {:.2}%", port_var.sqrt() * 100.0);

    // --- Step 5: Rolling risk metrics update ---
    let daily_returns = [0.01, -0.005, 0.002, -0.003];
    for r in daily_returns {
        let var_95 = var.update(r, 0.95);
        let sharpe_ratio = sharpe.update(r, 0.02);
        let (max_dd, _current_dd) = drawdown.update(account_value * (1.0 + r));
        println!("VaR95: {:.2}%, Sharpe: {:.2}, MaxDD: {:.2}%", var_95 * 100.0, sharpe_ratio, max_dd * 100.0);
    }

    // --- Step 6: Covariance analysis with EWMA ---
    let mut ewma = EWMACovarianceMatrix::<3>::from_halflife(20.0);
    let scenario_returns = [
        [0.01, 0.005, -0.002],
        [-0.003, 0.002, 0.004],
        [0.002, -0.001, 0.003],
    ];

    for obs in scenario_returns {
        ewma.update(&obs);
    }

    println!("EWMA Covariance Matrix:");
    let cov = ewma.covariance();
    for row in cov {
        println!("{:?}", row);
    }

    Ok(())
}
