use algotrading::risk::*;
use algotrading::options::*;
use algotrading::matrix::*;

#[test]
fn test_full_portfolio_strategy() -> Result<(), Box<dyn std::error::Error>> {
    // --- Step 1: Portfolio setup ---
    let account_value = 500_000.0;
    let positions = [
        ("AAPL", 100.0, 160.0),
        ("MSFT", 80.0, 220.0),
        ("GOOG", 50.0, 2800.0),
    ];

    let spot_prices: [f64; 3] = [155.0, 210.0, 2750.0];
    let vols: [f64; 3] = [0.25, 0.22, 0.30];

    // --- Step 2: Risk metrics ---
    let mut var = RollingVaR::<252>::new();
    let mut sharpe = RollingSharpe::<252>::new();
    let mut drawdown = MaxDrawdown::<1000>::new();

    // --- Step 3: Option pricing & deltas ---
    let rate = 0.05;
    let div_yield = 0.04;
    let time_to_expiry = 0.25;

    let mut deltas: [f64; 3] = [0.0; 3];
    let mut option_prices: [f64; 3] = [0.0; 3];

    for i in 0..3 {
        let (_, _, strike) = positions[i];
        let price = black_scholes_call(spot_prices[i], strike, rate, div_yield, vols[i], time_to_expiry);
        let delta = delta_call(spot_prices[i], strike, rate, div_yield, vols[i], time_to_expiry);
        option_prices[i] = price;
        deltas[i] = delta;
        println!("{} Option price: {:.2}, Delta: {:.4}", positions[i].0, price, delta);
    }

    // --- Step 4: Portfolio variance via covariance matrix ---
    let cov_matrix: [[f64; 3]; 3] = [
        [0.04, 0.01, 0.002],
        [0.01, 0.09, 0.004],
        [0.002, 0.004, 0.16],
    ];

    let chol = Cholesky::decompose(&cov_matrix)?;
    let mut weights: [f64; 3] = [0.0; 3];

    for i in 0..3 {
        weights[i] = deltas[i] * spot_prices[i] / account_value;
    }

    let portfolio_var = chol.portfolio_variance(&weights);
    println!("Portfolio variance: {:.4}", portfolio_var);
    println!("Portfolio volatility: {:.2}%", portfolio_var.sqrt() * 100.0);

    // --- Step 5: Risk metrics update with synthetic returns ---
    let daily_returns: [f64; 5] = [0.01, -0.005, 0.002, 0.004, -0.003];

    for daily_return in daily_returns {
        let var_95 = var.update(daily_return, 0.95);
        let sharpe_ratio = sharpe.update(daily_return, 0.02);
        let (max_dd, current_dd) = drawdown.update(account_value * (1.0 + daily_return));
        println!("VaR95: {:.2}%, Sharpe: {:.2}, MaxDD: {:.2}%", var_95*100.0, sharpe_ratio, max_dd*100.0);
    }

    // --- Step 6: Covariance estimation with Ledoit-Wolf ---
    let historical_returns = [
        [0.01, 0.02, -0.01],
        [-0.005, 0.015, 0.0],
        [0.002, -0.01, 0.01],
        [0.004, 0.005, -0.005],
    ];

    let lw = LedoitWolfEstimator::<3>::estimate(&historical_returns)?;
    let shrunk_cov = lw.shrunk_covariance();
    println!("Shrunk covariance matrix: {:?}", shrunk_cov);

    // --- Step 7: Hedging decision example ---
    // Compute hedge ratios using deltas
    let hedge_notional: [f64; 3] = [
        -deltas[0] * spot_prices[0] * positions[0].1,
        -deltas[1] * spot_prices[1] * positions[1].1,
        -deltas[2] * spot_prices[2] * positions[2].1,
    ];

    println!("Hedge notionals: {:?}", hedge_notional);

    // --- Step 8: Portfolio optimization example (simple) ---
    // Using inverse of shrunk covariance to compute minimum variance weights
    let inv_cov = invert_spd_cholesky(&shrunk_cov)?;
    let ones = [1.0; 3];

    let mut numerator = [0.0; 3];
    for i in 0..3 {
        for j in 0..3 {
            numerator[i] += inv_cov[i][j] * ones[j];
        }
    }

    let mut denominator = 0.0;
    for i in 0..3 {
        denominator += numerator[i];
    }

    let min_var_weights: [f64; 3] = [
        numerator[0] / denominator,
        numerator[1] / denominator,
        numerator[2] / denominator,
    ];

    println!("Minimum variance portfolio weights: {:?}", min_var_weights);

    Ok(())
}
