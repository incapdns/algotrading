//! Comprehensive end-to-end test demonstrating a complete trading strategy workflow
//!
//! This test showcases:
//! - Technical indicator creation and usage (MACD, RSI, Bollinger Bands)
//! - Regime detection with Markov Switching
//! - Position tracking and portfolio management
//! - Risk metrics calculation (VaR, Sharpe, Drawdown)
//! - Options pricing for hedging
//! - Strategy backtesting with realistic signals
//! - Performance reporting

use algotrading::prelude::*;

/// Position tracker for backtesting
struct Position {
    shares: f64,
    entry_price: f64,
    entry_time: usize,
}

/// Trade record for analysis
#[derive(Debug)]
#[allow(dead_code)]
struct Trade {
    entry_time: usize,
    exit_time: usize,
    entry_price: f64,
    exit_price: f64,
    shares: f64,
    pnl: f64,
    return_pct: f64,
}

/// Strategy state and metrics
struct StrategyState {
    // Technical indicators
    macd: MACD,
    rsi: RSI,
    bb: BollingerBands<f64, 20>,

    // Regime detection
    regime: MarkovSwitching,

    // Risk metrics
    rolling_stats: RollingStats<f64, 20>,
    var: RollingVaR<252>,
    sharpe: RollingSharpe<252>,
    drawdown: MaxDrawdown<1000>,

    // Portfolio tracking
    position: Option<Position>,
    cash: f64,
    portfolio_value: f64,

    // Trade history
    trades: Vec<Trade>,
    daily_returns: Vec<f64>,

    // Track last metric values for reporting
    last_var: f64,
    last_sharpe: f64,
}

impl StrategyState {
    fn new(initial_capital: f64) -> Self {
        Self {
            macd: MACD::standard(),
            rsi: RSI::standard(),
            bb: BollingerBands::standard(),
            regime: MarkovSwitching::spy_default(),
            rolling_stats: RollingStats::new(),
            var: RollingVaR::new(),
            sharpe: RollingSharpe::new(),
            drawdown: MaxDrawdown::new(),
            position: None,
            cash: initial_capital,
            portfolio_value: initial_capital,
            trades: Vec::new(),
            daily_returns: Vec::new(),
            last_var: 0.0,
            last_sharpe: 0.0,
        }
    }

    /// Update all indicators with new price
    fn update_indicators(&mut self, price: f64) -> SignalData {
        let (macd_line, signal_line, histogram) = self.macd.update(price);
        let rsi_value = self.rsi.update(price);
        let (upper, middle, lower, _percent_b) = self.bb.update(price);

        // Calculate log return for regime detection
        let prev_value = self.portfolio_value;
        let log_return = if prev_value > 0.0 {
            (price / (prev_value / (1.0 + self.position.as_ref().map_or(0.0, |p| p.shares)))).ln()
        } else {
            0.0
        };

        self.regime.update(log_return);
        let (_mean, std) = self.rolling_stats.update(log_return);

        SignalData {
            macd_line,
            signal_line,
            histogram,
            rsi: rsi_value,
            bb_middle: middle,
            bb_upper: upper,
            bb_lower: lower,
            is_volatile: self.regime.is_volatile(0.7),
            volatility: std,
        }
    }

    /// Generate trading signal based on indicators
    fn generate_signal(&self, signals: &SignalData, price: f64) -> Signal {
        // Don't trade in high volatility regimes
        if signals.is_volatile {
            if self.position.is_some() {
                // Exit positions during volatility
                return Signal::Exit;
            }
            return Signal::Hold;
        }

        // Long signal: Strong MACD + RSI oversold + price below middle BB
        let macd_bullish = signals.histogram > 0.5 && signals.macd_line > signals.signal_line;
        let rsi_ok_long = signals.rsi < 65.0 && signals.rsi > 35.0;
        let below_middle = price < signals.bb_middle;

        if macd_bullish && rsi_ok_long && below_middle && self.position.is_none() {
            return Signal::Long;
        }

        // Exit signal: MACD bearish OR RSI overbought OR significant profit/loss
        let macd_bearish = signals.histogram < -0.5 && signals.macd_line < signals.signal_line;
        let rsi_overbought = signals.rsi > 70.0;
        let rsi_oversold = signals.rsi < 30.0;

        // Stop loss / take profit
        if let Some(ref pos) = self.position {
            let unrealized_pnl_pct = (price - pos.entry_price) / pos.entry_price;
            if unrealized_pnl_pct > 0.05 || unrealized_pnl_pct < -0.03 {
                return Signal::Exit; // Take 5% profit or 3% stop loss
            }
        }

        if (macd_bearish || rsi_overbought || rsi_oversold) && self.position.is_some() {
            return Signal::Exit;
        }

        Signal::Hold
    }

    /// Execute trade based on signal
    fn execute_trade(&mut self, signal: Signal, price: f64, time: usize) {
        match signal {
            Signal::Long => {
                // Only trade if we have at least 10% of initial portfolio available
                let min_capital = self.portfolio_value * 0.10;
                if self.position.is_none() && self.cash >= min_capital {
                    // Use fixed 15% of portfolio value (more conservative)
                    // This ensures we don't overleverage even if Kelly suggests more
                    let position_fraction = 0.15; // Fixed 15% of portfolio

                    let capital_to_deploy = (self.portfolio_value * position_fraction).min(self.cash);
                    let shares = capital_to_deploy / price;

                    self.position = Some(Position {
                        shares,
                        entry_price: price,
                        entry_time: time,
                    });
                    self.cash -= capital_to_deploy;

                    println!("  [BUY]  Time: {}, Price: ${:.2}, Shares: {:.2}, Capital: ${:.2}",
                        time, price, shares, capital_to_deploy);
                } else if self.position.is_none() && self.cash < 1000.0 && self.cash > 0.0 {
                    // Log when we skip trades due to insufficient capital
                    if time % 100 == 0 {
                        println!("  [SKIP] Time: {}, Insufficient capital: ${:.2}", time, self.cash);
                    }
                }
            }
            Signal::Exit => {
                if let Some(pos) = self.position.take() {
                    let proceeds = pos.shares * price;
                    let pnl = proceeds - (pos.shares * pos.entry_price);
                    let return_pct = pnl / (pos.shares * pos.entry_price);

                    self.cash += proceeds;

                    println!("  [SELL] Time: {}, Price: ${:.2}, Shares: {:.2}, P&L: ${:.2} ({:.2}%)",
                        time, price, pos.shares, pnl, return_pct * 100.0);

                    self.trades.push(Trade {
                        entry_time: pos.entry_time,
                        exit_time: time,
                        entry_price: pos.entry_price,
                        exit_price: price,
                        shares: pos.shares,
                        pnl,
                        return_pct,
                    });
                }
            }
            Signal::Hold => {}
        }
    }

    /// Update portfolio value and risk metrics
    fn update_portfolio(&mut self, price: f64) {
        let prev_value = self.portfolio_value;

        // Calculate current portfolio value
        self.portfolio_value = self.cash;
        if let Some(ref pos) = self.position {
            self.portfolio_value += pos.shares * price;
        }

        // Calculate daily return
        let daily_return = if prev_value > 0.0 {
            (self.portfolio_value - prev_value) / prev_value
        } else {
            0.0
        };

        self.daily_returns.push(daily_return);

        // Update risk metrics and store for reporting
        self.last_var = self.var.update(daily_return, 0.95);
        self.last_sharpe = self.sharpe.update(daily_return, 0.02 / 252.0); // 2% risk-free rate annualized
        let (_max_dd, _) = self.drawdown.update(self.portfolio_value);
    }

    /// Generate performance report
    fn report(&mut self, initial_capital: f64) {
        let separator = "=".repeat(20);
        println!("\n{}=== PERFORMANCE REPORT ==={}",  separator, separator);

        // Overall metrics
        let total_return = (self.portfolio_value - initial_capital) / initial_capital;
        let total_return_pct = total_return * 100.0;

        println!("\nPortfolio Metrics:");
        println!("  Initial Capital:    ${:.2}", initial_capital);
        println!("  Final Value:        ${:.2}", self.portfolio_value);
        println!("  Total Return:       {:.2}%", total_return_pct);
        println!("  Cash:               ${:.2}", self.cash);

        if let Some(ref pos) = self.position {
            println!("  Open Position:      {:.2} shares @ ${:.2}", pos.shares, pos.entry_price);
        }

        // Trade statistics
        if !self.trades.is_empty() {
            let winning_trades: Vec<&Trade> = self.trades.iter().filter(|t| t.pnl > 0.0).collect();
            let losing_trades: Vec<&Trade> = self.trades.iter().filter(|t| t.pnl <= 0.0).collect();

            let total_pnl: f64 = self.trades.iter().map(|t| t.pnl).sum();
            let avg_win: f64 = if !winning_trades.is_empty() {
                winning_trades.iter().map(|t| t.pnl).sum::<f64>() / winning_trades.len() as f64
            } else {
                0.0
            };
            let avg_loss: f64 = if !losing_trades.is_empty() {
                losing_trades.iter().map(|t| t.pnl).sum::<f64>() / losing_trades.len() as f64
            } else {
                0.0
            };

            let win_rate = winning_trades.len() as f64 / self.trades.len() as f64;

            println!("\nTrade Statistics:");
            println!("  Total Trades:       {}", self.trades.len());
            println!("  Winning Trades:     {}", winning_trades.len());
            println!("  Losing Trades:      {}", losing_trades.len());
            println!("  Win Rate:           {:.2}%", win_rate * 100.0);
            println!("  Average Win:        ${:.2}", avg_win);
            println!("  Average Loss:       ${:.2}", avg_loss);
            println!("  Total P&L:          ${:.2}", total_pnl);

            if avg_loss < 0.0 {
                let profit_factor = -avg_win / avg_loss;
                println!("  Profit Factor:      {:.2}", profit_factor);
            }
        }

        // Risk metrics
        if self.daily_returns.len() > 1 {
            let (max_dd, _) = self.drawdown.update(self.portfolio_value); // Get current values

            println!("\nRisk Metrics:");
            println!("  Sharpe Ratio:       {:.3}", self.last_sharpe);
            println!("  Max Drawdown:       {:.2}%", max_dd * 100.0);
            println!("  VaR (95%):          {:.2}%", self.last_var * 100.0);
        }

        let separator = "=".repeat(20);
        println!("\n{}==============={}=", separator, separator);
    }
}

#[derive(Debug)]
#[allow(dead_code)]
struct SignalData {
    macd_line: f64,
    signal_line: f64,
    histogram: f64,
    rsi: f64,
    bb_middle: f64,
    bb_upper: f64,
    bb_lower: f64,
    is_volatile: bool,
    volatility: f64,
}

#[derive(Debug, PartialEq)]
enum Signal {
    Long,
    Exit,
    Hold,
}

/// Generate synthetic price data with realistic characteristics
fn generate_price_data(num_points: usize, initial_price: f64, volatility: f64) -> Vec<f64> {
    let mut prices = Vec::with_capacity(num_points);
    prices.push(initial_price);

    // Simple random walk with drift
    // Using a deterministic "random" generator for reproducibility
    let mut seed = 42u64;

    for _ in 1..num_points {
        // Linear congruential generator (simple PRNG)
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let random = (seed as f64 / u64::MAX as f64) * 2.0 - 1.0; // -1 to 1

        let drift = 0.0005; // Slight upward drift
        let return_val = drift + volatility * random;
        let new_price = prices.last().unwrap() * (1.0 + return_val);

        prices.push(new_price.max(1.0)); // Ensure price stays positive
    }

    prices
}

#[test]
fn test_complete_strategy_backtest() -> Result<(), Box<dyn std::error::Error>> {
    let separator = "=".repeat(15);
    println!("\n{}=== ALGORITHMIC TRADING STRATEGY BACKTEST ==={}=\n", separator, separator);

    // --- Configuration ---
    let initial_capital = 100_000.0;
    let num_days = 500;
    let initial_price = 400.0;
    let volatility = 0.015; // 1.5% daily volatility

    println!("Configuration:");
    println!("  Initial Capital: ${:.2}", initial_capital);
    println!("  Backtest Period: {} days", num_days);
    println!("  Initial Price:   ${:.2}", initial_price);
    println!("  Volatility:      {:.2}%", volatility * 100.0);

    // --- Generate Market Data ---
    println!("\n[1/6] Generating market data...");
    let prices = generate_price_data(num_days, initial_price, volatility);
    println!("  Generated {} price points", prices.len());
    println!("  Price range: ${:.2} - ${:.2}",
        prices.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );

    // --- Initialize Strategy ---
    println!("\n[2/6] Initializing strategy components...");
    let mut strategy = StrategyState::new(initial_capital);
    println!("  MACD(12, 26, 9): initialized");
    println!("  RSI(14): initialized");
    println!("  Bollinger Bands(20, 2.0): initialized");
    println!("  Markov Regime Detection: initialized");
    println!("  Risk metrics: VaR, Sharpe, Drawdown");

    // --- Backtest Loop ---
    println!("\n[3/6] Running backtest simulation...");
    println!("\nTrade Log:");

    for (day, &price) in prices.iter().enumerate() {
        // Update all indicators
        let signals = strategy.update_indicators(price);

        // Generate trading signal
        let signal = strategy.generate_signal(&signals, price);

        // Execute trade if signal generated
        strategy.execute_trade(signal, price, day);

        // Update portfolio value and risk metrics
        strategy.update_portfolio(price);

        // Periodic status update
        if day % 100 == 0 && day > 0 {
            println!("  [INFO] Day {}: Price=${:.2}, Portfolio=${:.2}, RSI={:.1}, MACD={:.4}",
                day, price, strategy.portfolio_value, signals.rsi, signals.histogram);
        }
    }

    // Close any open position at the end
    if strategy.position.is_some() {
        strategy.execute_trade(Signal::Exit, *prices.last().unwrap(), prices.len() - 1);
    }

    // --- Options Hedging Analysis ---
    println!("\n[4/6] Analyzing options hedging strategy...");
    let final_price = *prices.last().unwrap();
    let strike = final_price * 0.95; // 5% OTM put for protection
    let rate = 0.05;
    let div_yield = 0.02;
    let time_to_expiry = 30.0 / 365.0; // 30-day options
    let implied_vol = 0.25;

    let put_price = black_scholes_put(final_price, strike, rate, div_yield, implied_vol, time_to_expiry);
    let put_delta = delta_put(final_price, strike, rate, div_yield, implied_vol, time_to_expiry);
    let put_gamma = gamma(final_price, strike, rate, div_yield, implied_vol, time_to_expiry);
    let put_vega = vega(final_price, strike, rate, div_yield, implied_vol, time_to_expiry);

    println!("  Protective Put Analysis (95% strike):");
    println!("    Underlying Price:  ${:.2}", final_price);
    println!("    Strike Price:      ${:.2}", strike);
    println!("    Put Price:         ${:.2}", put_price);
    println!("    Delta:             {:.4}", put_delta);
    println!("    Gamma:             {:.4}", put_gamma);
    println!("    Vega:              {:.4}", put_vega);

    let portfolio_shares = strategy.cash / final_price; // Hypothetical shares with all cash
    let num_contracts = (portfolio_shares / 100.0).ceil(); // Options are per 100 shares
    let hedge_cost = num_contracts * put_price * 100.0;

    println!("    Contracts Needed:  {:.0}", num_contracts);
    println!("    Total Hedge Cost:  ${:.2}", hedge_cost);
    println!("    Cost as % of Portfolio: {:.2}%", (hedge_cost / strategy.portfolio_value) * 100.0);

    // --- Risk Analysis ---
    println!("\n[5/6] Performing risk analysis...");

    // Calculate correlation with synthetic market
    let market_returns: Vec<f64> = prices.windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();

    if !market_returns.is_empty() && strategy.daily_returns.len() > 20 {
        let mut correlation = RollingCorrelation::<f64, 50>::new();
        let mut final_corr = 0.0;

        let min_len = market_returns.len().min(strategy.daily_returns.len());
        // Need at least 20 samples for meaningful correlation
        if min_len >= 20 {
            for i in 0..min_len {
                final_corr = correlation.update(market_returns[i], strategy.daily_returns[i]);
            }
            println!("  Market Correlation:   {:.3}", final_corr);
        } else {
            println!("  Market Correlation:   N/A (insufficient data)");
        }
    }

    // Parametric VaR analysis
    if strategy.daily_returns.len() > 30 {
        let param_var = ParametricVaR::new(&strategy.daily_returns);
        let var_95 = param_var.var(0.95);
        let var_99 = param_var.var(0.99);

        println!("  Parametric VaR (95%): {:.2}%", var_95 * 100.0);
        println!("  Parametric VaR (99%): {:.2}%", var_99 * 100.0);
    }

    // --- Performance Report ---
    println!("\n[6/6] Generating final report...");
    strategy.report(initial_capital);

    // --- Validation Assertions ---
    println!("\n[VALIDATION] Running test assertions...");

    // Portfolio value should be positive
    assert!(strategy.portfolio_value > 0.0, "Portfolio value must be positive");
    println!("  ✓ Portfolio value is positive: ${:.2}", strategy.portfolio_value);

    // Should have made some trades
    assert!(strategy.trades.len() > 0, "Strategy should have executed at least one trade");
    println!("  ✓ Strategy executed {} trades", strategy.trades.len());

    // Risk metrics should be reasonable
    if let Some(last_return) = strategy.daily_returns.last() {
        assert!(last_return.abs() < 1.0, "Daily returns should be less than 100%");
        println!("  ✓ Daily returns are reasonable");
    }

    // Sharpe ratio should be calculable (not NaN)
    assert!(!strategy.last_sharpe.is_nan(), "Sharpe ratio should not be NaN");
    println!("  ✓ Sharpe ratio is valid: {:.3}", strategy.last_sharpe);

    // Max drawdown should be non-negative
    let (max_dd, _) = strategy.drawdown.update(strategy.portfolio_value);
    assert!(max_dd >= 0.0, "Max drawdown should be non-negative");
    println!("  ✓ Max drawdown is valid: {:.2}%", max_dd * 100.0);

    // Options greeks should be reasonable
    assert!(put_delta < 0.0 && put_delta > -1.0, "Put delta should be between -1 and 0");
    assert!(put_gamma >= 0.0, "Gamma should be non-negative");
    assert!(put_vega >= 0.0, "Vega should be non-negative");
    println!("  ✓ Options Greeks are valid");

    // Final value should be within reasonable bounds (not blown up or zero)
    let return_pct = (strategy.portfolio_value - initial_capital) / initial_capital;
    assert!(return_pct > -0.5 && return_pct < 5.0,
        "Total return should be between -50% and +500%");
    println!("  ✓ Total return is within reasonable bounds: {:.2}%", return_pct * 100.0);

    let separator = "=".repeat(20);
    println!("\n{}=== ALL TESTS PASSED ==={}=\n", separator, separator);

    Ok(())
}

#[test]
fn test_strategy_components_individually() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Testing Individual Strategy Components ===\n");

    // Test 1: Technical Indicators
    println!("[Test 1/5] Technical Indicators");
    let mut macd = MACD::standard();
    let mut rsi = RSI::standard();
    let mut bb = BollingerBands::<f64, 20>::standard();

    let test_prices = [100.0, 101.0, 102.5, 101.5, 103.0, 104.0, 103.5, 105.0];
    for &price in &test_prices {
        let (macd_line, signal, hist) = macd.update(price);
        let rsi_val = rsi.update(price);
        let (upper, mid, lower, _percent_b) = bb.update(price);

        assert!(!macd_line.is_nan() && !signal.is_nan() && !hist.is_nan());
        assert!(rsi_val >= 0.0 && rsi_val <= 100.0);
        assert!(lower <= mid && mid <= upper);
    }
    println!("  ✓ MACD, RSI, and Bollinger Bands working correctly");

    // Test 2: Regime Detection
    println!("\n[Test 2/5] Regime Detection");
    let mut regime = MarkovSwitching::spy_default();
    let returns = [0.01, -0.02, 0.015, -0.005, 0.02, -0.03, 0.01];

    for &ret in &returns {
        regime.update(ret);
    }

    let high_vol = regime.is_volatile(0.7);
    let state = regime.most_likely_state();
    println!("  ✓ Regime detection initialized, volatile: {}, state: {:?}", high_vol, state);

    // Test 3: Risk Metrics
    println!("\n[Test 3/5] Risk Metrics");
    let mut var = RollingVaR::<30>::new();
    let mut sharpe = RollingSharpe::<30>::new();
    let mut dd = MaxDrawdown::<100>::new();

    let mut portfolio_value = 100_000.0;
    for &ret in &returns {
        let var_95 = var.update(ret, 0.95);
        let sharpe_ratio = sharpe.update(ret, 0.0);
        portfolio_value *= 1.0 + ret;
        let (max_dd, curr_dd) = dd.update(portfolio_value);

        // VaR can be negative if returns are mostly positive (good scenario)
        assert!(!var_95.is_nan());
        assert!(!sharpe_ratio.is_nan());
        assert!(max_dd >= 0.0 && curr_dd >= 0.0);
    }
    println!("  ✓ VaR, Sharpe, and Drawdown calculations working");

    // Test 4: Options Pricing
    println!("\n[Test 4/5] Options Pricing");
    let spot = 100.0;
    let strike = 100.0;
    let rate = 0.05;
    let div = 0.02;
    let vol = 0.25;
    let time = 0.25;

    let call_price = black_scholes_call(spot, strike, rate, div, vol, time);
    let put_price = black_scholes_put(spot, strike, rate, div, vol, time);
    let delta_c = delta_call(spot, strike, rate, div, vol, time);
    let delta_p = delta_put(spot, strike, rate, div, vol, time);
    let gamma_val = gamma(spot, strike, rate, div, vol, time);
    let vega_val = vega(spot, strike, rate, div, vol, time);

    assert!(call_price > 0.0 && put_price > 0.0);
    assert!(delta_c > 0.0 && delta_c < 1.0);
    assert!(delta_p < 0.0 && delta_p > -1.0);
    assert!(gamma_val >= 0.0);
    assert!(vega_val >= 0.0);

    // Put-Call parity check: C - P = S*e^(-qT) - K*e^(-rT)
    let expected_diff = spot * (-div * time).exp() - strike * (-rate * time).exp();
    let actual_diff = call_price - put_price;
    let parity_diff = (actual_diff - expected_diff).abs();
    assert!(parity_diff < 1.0, "Put-call parity violated: diff = {:.4}, expected near zero", parity_diff);

    println!("  ✓ Options pricing and Greeks validated");
    println!("    Call: ${:.2}, Put: ${:.2}", call_price, put_price);
    println!("    Delta(C): {:.3}, Delta(P): {:.3}", delta_c, delta_p);

    // Test 5: Kelly Criterion
    println!("\n[Test 5/5] Position Sizing with Kelly");
    let win_prob = 0.55;
    let win_ratio = 1.5; // Win $1.50 for every $1.00 loss
    let kelly_fraction = Kelly::binary(win_prob, win_ratio);
    let half_kelly = Kelly::half_kelly(win_prob, win_ratio);

    assert!(kelly_fraction > 0.0 && kelly_fraction < 1.0);
    assert!(half_kelly == kelly_fraction * 0.5);

    println!("  ✓ Kelly criterion: {:.2}% (half-Kelly: {:.2}%)",
        kelly_fraction * 100.0, half_kelly * 100.0);

    println!("\n=== All Component Tests Passed ===\n");

    Ok(())
}
