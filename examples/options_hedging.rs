//! Options Delta Hedging Example
//!
//! This example demonstrates a complete delta-hedging workflow for options market makers.
//! Covers:
//! - Calculating Greeks (Delta, Gamma, Vega, Theta)
//! - Dynamic delta hedging as underlying moves
//! - Gamma exposure monitoring
//! - P&L attribution (theta decay vs. gamma scalping)
//!
//! Run with: cargo run --example options_hedging

use algotrading::options::pricing::{
    black_scholes_call, delta_call, delta_put, gamma, vega, theta_call
};

fn main() {
    println!("=== Options Delta Hedging Example ===\n");

    // Market parameters
    let initial_spot = 100.0;
    let strike = 100.0;  // ATM call option
    let rate = 0.05;     // 5% risk-free rate
    let div_yield = 0.02; // 2% dividend yield
    let vol = 0.20;       // 20% implied volatility
    let time_to_expiry = 0.25;  // 3 months (0.25 years)

    // Portfolio
    let options_sold = 100.0;  // Sold 100 call contracts (short 10,000 options at 100 multiplier)

    println!("INITIAL POSITION");
    println!("{}", "=".repeat(60));
    println!("Short {} call contracts ({} options)", options_sold, options_sold * 100.0);
    println!("Strike: ${:.2}", strike);
    println!("Spot: ${:.2}", initial_spot);
    println!("Time to expiry: {:.1} days", time_to_expiry * 365.0);
    println!("Implied Vol: {:.1}%", vol * 100.0);
    println!("{}", "=".repeat(60));

    // Calculate initial Greeks
    let call_price = black_scholes_call(initial_spot, strike, rate, div_yield, vol, time_to_expiry);
    let delta = delta_call(initial_spot, strike, rate, div_yield, vol, time_to_expiry);
    let gamma_val = gamma(initial_spot, strike, rate, div_yield, vol, time_to_expiry);
    let vega_val = vega(initial_spot, strike, rate, div_yield, vol, time_to_expiry);
    let theta_val = theta_call(initial_spot, strike, rate, div_yield, vol, time_to_expiry);

    println!("\nINITIAL GREEKS (per option):");
    println!("Call Price:  ${:.4}", call_price);
    println!("Delta:       {:.4} ({:.0} shares per contract)", delta, delta * 100.0);
    println!("Gamma:       {:.4}", gamma_val);
    println!("Vega:        {:.4} ($ per 1% vol change)", vega_val);
    println!("Theta:       ${:.4} per day", theta_val);

    // Portfolio Greeks
    let total_options = options_sold * 100.0;  // 100 contracts = 10,000 options
    let portfolio_delta = -delta * total_options;  // Negative because we're short
    let portfolio_gamma = -gamma_val * total_options;
    let portfolio_vega = -vega_val * total_options;
    let portfolio_theta = -theta_val * total_options;

    println!("\nPORTFOLIO GREEKS:");
    println!("Net Delta:   {:.0} shares equivalent", portfolio_delta);
    println!("Net Gamma:   {:.2}", portfolio_gamma);
    println!("Net Vega:    ${:.2} per 1% vol", portfolio_vega);
    println!("Net Theta:   ${:.2} per day", portfolio_theta);

    // Initial hedge
    let shares_to_buy = -portfolio_delta;  // Buy shares to be delta-neutral
    let mut hedge_shares = shares_to_buy;
    let mut stock_pnl = 0.0;
    let mut option_pnl = 0.0;
    let mut total_pnl = 0.0;

    println!("\nINITIAL HEDGE:");
    println!("Buy {:.0} shares at ${:.2} for delta neutrality", shares_to_buy, initial_spot);
    println!("Cost: ${:.2}", shares_to_buy * initial_spot);

    // Simulate price moves over several days
    let price_path = vec![
        100.0, 101.5, 103.0, 102.0, 104.0,  // Up move
        103.0, 101.5, 100.0, 98.5, 97.0,     // Down move
        98.0, 99.0, 100.5, 102.0, 103.5,     // Back up
    ];

    println!("\n{}", "=".repeat(80));
    println!("DELTA HEDGING SIMULATION");
    println!("{}", "=".repeat(80));
    println!("Day | Spot    | Delta  | Hedge   | Stock P&L | Option P&L | Total P&L | Action");
    println!("{}", "-".repeat(80));

    let mut prev_spot = initial_spot;
    let mut prev_call_price = call_price;

    for (day, &spot) in price_path.iter().enumerate() {
        // Time decay (1 day = 1/365 years)
        let time_remaining = time_to_expiry - (day as f64 / 365.0);
        if time_remaining <= 0.0 {
            break;
        }

        // Recalculate Greeks at new spot
        let new_call_price = black_scholes_call(spot, strike, rate, div_yield, vol, time_remaining);
        let new_delta = delta_call(spot, strike, rate, div_yield, vol, time_remaining);
        let new_gamma = gamma(spot, strike, rate, div_yield, vol, time_remaining);

        // P&L calculation
        let spot_change = spot - prev_spot;
        let stock_pnl_day = hedge_shares * spot_change;
        let option_price_change = new_call_price - prev_call_price;
        let option_pnl_day = -total_options * option_price_change;  // Short position

        stock_pnl += stock_pnl_day;
        option_pnl += option_pnl_day;
        total_pnl = stock_pnl + option_pnl;

        // Current portfolio delta
        let current_portfolio_delta = -new_delta * total_options;
        let delta_imbalance = current_portfolio_delta + hedge_shares;

        // Rehedge decision (if delta imbalance > threshold)
        let rehedge_threshold = 500.0;  // 500 shares
        let mut action = "HOLD".to_string();
        if delta_imbalance.abs() > rehedge_threshold {
            let shares_to_trade = -delta_imbalance;
            hedge_shares += shares_to_trade;
            action = if shares_to_trade > 0.0 {
                format!("BUY {:.0}", shares_to_trade)
            } else {
                format!("SELL {:.0}", -shares_to_trade)
            };
        }

        println!(
            "{:3} | ${:6.2} | {:6.3} | {:7.0} | ${:8.2} | ${:9.2} | ${:8.2} | {}",
            day,
            spot,
            new_delta,
            hedge_shares,
            stock_pnl,
            option_pnl,
            total_pnl,
            action
        );

        // Gamma P&L attribution
        if day % 5 == 0 {
            let gamma_pnl = 0.5 * (-portfolio_gamma) * spot_change * spot_change;
            let theta_pnl = (-portfolio_theta);
            println!("    └─ Gamma P&L: ${:.2}, Theta P&L: ${:.2}", gamma_pnl, theta_pnl);
        }

        prev_spot = spot;
        prev_call_price = new_call_price;
    }

    // Final summary
    println!("\n{}", "=".repeat(80));
    println!("FINAL RESULTS");
    println!("{}", "=".repeat(80));
    println!("Total P&L:           ${:.2}", total_pnl);
    println!("  Stock hedge P&L:   ${:.2}", stock_pnl);
    println!("  Option position:   ${:.2}", option_pnl);
    println!("Final hedge:         {:.0} shares", hedge_shares);

    println!("\n{}", "=".repeat(80));
    println!("KEY INSIGHTS");
    println!("{}", "=".repeat(80));
    println!("• Delta hedging neutralizes directional risk");
    println!("• Gamma exposure means delta changes as price moves");
    println!("• Frequent rehedging is needed for large gamma positions");
    println!("• Theta decay (positive for short options) offsets trading costs");
    println!("• Market makers profit from theta and bid-ask spread");
    println!("• Gamma scalping: profit from volatility via rehedging");
    println!("\nRisk considerations:");
    println!("• Transaction costs can erode profits with frequent hedging");
    println!("• Large price jumps (gaps) violate continuous hedging assumption");
    println!("• Implied vol changes affect position value (vega risk)");
    println!("• Model risk: Black-Scholes assumptions may not hold");
}
