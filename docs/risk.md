# Risk Metrics Module

Comprehensive risk measurement and position sizing for quantitative trading.

## Overview

The risk module provides:
- Value at Risk (VaR) - historical and parametric
- Expected Shortfall (CVaR)
- Performance metrics (Sharpe, Sortino, Calmar)
- Maximum drawdown tracking
- Kelly criterion position sizing

All implementations use fixed-size buffers and online algorithms for constant memory usage.

## Value at Risk (VaR)

### Rolling Historical VaR

Non-parametric VaR using historical returns.

```rust
use algotrading::risk::*;

let mut var = RollingVaR::<252>::new(); // 1 year of daily returns

for return_val in returns {
    let var_95 = var.update(return_val, 0.95);
    let var_99 = var.update(return_val, 0.99);

    println!("95% VaR: {:.4}%", var_95 * 100.0);
    println!("99% VaR: {:.4}%", var_99 * 100.0);
}
```

**Confidence Levels:**
- 0.95 = 95% VaR (1-in-20 event)
- 0.99 = 99% VaR (1-in-100 event)
- 0.999 = 99.9% VaR (1-in-1000 event)

**Interpretation:** "95% confident we won't lose more than X% tomorrow"

### Expected Shortfall (CVaR)

Average loss beyond VaR threshold.

```rust
let cvar = var.expected_shortfall(0.95);

println!("If VaR is breached, expected loss: {:.4}%", cvar * 100.0);
```

**Advantages over VaR:**
- Coherent risk measure
- Captures tail risk better
- Accounts for severity, not just frequency

### Parametric VaR

Uses Cornish-Fisher expansion (accounts for skewness and kurtosis).

```rust
let parametric = ParametricVaR::new(&returns);
let var = parametric.var(0.95);

println!("Parametric 95% VaR: {:.4}%", var * 100.0);
```

**When to use:**
- Returns are approximately normal (or adjust for skew/kurtosis)
- Need smooth VaR estimates
- Want to avoid historical gaps

**Cornish-Fisher Formula:**
```
z_adjusted = z + (z²-1)·S/6 + (z³-3z)·K/24 - (2z³-5z)·S²/36
VaR = -(μ + z_adjusted·σ)
```

## Performance Metrics

### Sharpe Ratio

Risk-adjusted return metric.

```rust
let mut sharpe = RollingSharpe::<252>::new(); // 252 trading days

for daily_return in returns {
    let ratio = sharpe.update(daily_return, risk_free_rate: 0.02);
    println!("Sharpe: {:.2}", ratio);
}
```

**Formula:** (Return - RiskFree) / Volatility

**Interpretation:**
- < 0: Underperforming risk-free
- 0-1: Subpar
- 1-2: Good
- 2-3: Very good
- \> 3: Excellent (or lucky!)

**Note:** Automatically annualized assuming daily returns

### Sortino Ratio

Only penalizes downside deviation.

```rust
let mut sortino = RollingSortino::<252>::new();

for daily_return in returns {
    let ratio = sortino.update(daily_return, target_return: 0.0);
    println!("Sortino: {:.2}", ratio);
}
```

**Formula:** (Return - Target) / DownsideDeviation

**Advantages over Sharpe:**
- Doesn't penalize upside volatility
- Better for strategies with asymmetric returns
- More realistic for most investors

### Maximum Drawdown

Track peak-to-trough decline.

```rust
let mut dd = MaxDrawdown::<1000>::new();

for equity_value in equity_curve {
    let (max_dd, current_dd) = dd.update(equity_value);

    println!("Max DD: {:.2}%", max_dd * 100.0);
    println!("Current DD: {:.2}%", current_dd * 100.0);

    if current_dd > 0.10 {
        println!("WARNING: In 10% drawdown!");
    }
}
```

**Interpretation:**
- Max DD = worst historical drawdown
- Current DD = current decline from peak
- Recovery = time from trough to new peak

### Calmar Ratio

Return per unit of maximum drawdown.

```rust
let calmar = dd.calmar_ratio(annualized_return: 0.15);

println!("Calmar: {:.2}", calmar);
```

**Formula:** AnnualizedReturn / MaxDrawdown

**Interpretation:**
- Higher is better
- \> 1.0 is good for hedge funds
- Complements Sharpe (captures tail risk)

## Position Sizing

### Kelly Criterion

Optimal bet sizing to maximize long-term growth.

```rust
// Binary outcomes (win/loss)
let win_prob = 0.55;
let win_loss_ratio = 1.5; // Win $1.5 per $1 risked

let kelly_fraction = Kelly::binary(win_prob, win_loss_ratio);
println!("Optimal position: {:.1}%", kelly_fraction * 100.0);

// Continuous returns (mean-variance)
let mean_return = 0.001; // 0.1% per trade
let variance = 0.0004;   // 2% volatility

let kelly_fraction = Kelly::continuous(mean_return, variance);
```

**Formula (binary):**
```
f* = (p(b+1) - 1) / b

where:
  p = win probability
  b = win/loss ratio
```

**Important:** Kelly can be aggressive! Consider:

```rust
// Half-Kelly (more conservative)
let conservative_fraction = Kelly::half_kelly(win_prob, win_loss_ratio);

// Quarter-Kelly (very conservative)
let very_conservative = kelly_fraction * 0.25;
```

### Position Sizing Strategies

```rust
struct PositionSizer {
    max_position: f64,
    max_var_limit: f64,
    kelly_fraction: f64,
}

impl PositionSizer {
    fn size_position(&self,
                     signal_strength: f64,
                     current_var: f64,
                     account_value: f64) -> f64 {
        // Base size from Kelly
        let base_size = self.kelly_fraction * account_value;

        // Scale by signal confidence
        let signal_size = base_size * signal_strength;

        // Risk limit
        let var_size = self.max_var_limit * account_value / current_var;

        // Take minimum
        signal_size.min(var_size).min(self.max_position * account_value)
    }
}
```

## Risk Monitoring

### Real-time Risk Dashboard

```rust
struct RiskMonitor {
    var: RollingVaR<252>,
    sharpe: RollingSharpe<252>,
    sortino: RollingSortino<252>,
    drawdown: MaxDrawdown<1000>,
}

impl RiskMonitor {
    fn update(&mut self,
              daily_return: f64,
              equity: f64,
              risk_free: f64) -> RiskReport {
        RiskReport {
            var_95: self.var.update(daily_return, 0.95),
            var_99: self.var.update(daily_return, 0.99),
            cvar_95: self.var.expected_shortfall(0.95),
            sharpe: self.sharpe.update(daily_return, risk_free),
            sortino: self.sortino.update(daily_return, 0.0),
            max_dd: self.drawdown.update(equity).0,
            current_dd: self.drawdown.update(equity).1,
        }
    }

    fn check_limits(&self, report: &RiskReport) -> Vec<String> {
        let mut alerts = Vec::new();

        if report.var_95 > 0.02 {
            alerts.push("VaR 95% exceeds 2%".to_string());
        }
        if report.current_dd > 0.15 {
            alerts.push("Drawdown exceeds 15%".to_string());
        }
        if report.sharpe < 1.0 {
            alerts.push("Sharpe below target".to_string());
        }

        alerts
    }
}
```

### Stop-Loss Rules

```rust
fn dynamic_stop_loss(
    entry_price: f64,
    current_volatility: f64,
    time_in_trade_days: usize,
) -> f64 {
    // ATR-based stop (2x volatility)
    let atr_stop = entry_price - 2.0 * current_volatility;

    // Trailing stop (5%)
    let trailing_stop = entry_price * 0.95;

    // Time-based decay (tighter stop as time passes)
    let time_multiplier = 1.0 - 0.1 * (time_in_trade_days as f64 / 20.0);

    atr_stop.max(trailing_stop) * time_multiplier
}
```

### Portfolio Heat

Total portfolio risk from all positions.

```rust
fn portfolio_heat(
    positions: &[(f64, f64)], // (size, stop_distance)
    account_value: f64,
) -> f64 {
    let total_risk: f64 = positions.iter()
        .map(|(size, stop)| size * stop)
        .sum();

    total_risk / account_value
}

// Example
let positions = vec![
    (10000.0, 0.05),  // $10k position, 5% stop
    (15000.0, 0.03),  // $15k position, 3% stop
];

let heat = portfolio_heat(&positions, account_value: 100000.0);
println!("Portfolio heat: {:.1}%", heat * 100.0);

if heat > 0.06 {
    println!("WARNING: Total risk exceeds 6% of account");
}
```

## Backtesting Metrics

### Comprehensive Performance

```rust
struct BacktestResults {
    total_return: f64,
    annual_return: f64,
    sharpe: f64,
    sortino: f64,
    max_dd: f64,
    calmar: f64,
    win_rate: f64,
    profit_factor: f64,
    num_trades: usize,
}

fn compute_backtest_metrics(
    equity_curve: &[f64],
    trades: &[Trade],
    risk_free: f64,
) -> BacktestResults {
    let returns: Vec<f64> = equity_curve.windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();

    let parametric_var = ParametricVaR::new(&returns);

    let wins: Vec<f64> = trades.iter()
        .filter(|t| t.pnl > 0.0)
        .map(|t| t.pnl)
        .collect();

    let losses: Vec<f64> = trades.iter()
        .filter(|t| t.pnl < 0.0)
        .map(|t| -t.pnl)
        .collect();

    BacktestResults {
        total_return: (equity_curve.last().unwrap() / equity_curve[0]) - 1.0,
        annual_return: /* annualize total return */,
        sharpe: /* compute from returns */,
        sortino: /* compute from downside */,
        max_dd: /* compute from equity curve */,
        calmar: annual_return / max_dd,
        win_rate: wins.len() as f64 / trades.len() as f64,
        profit_factor: wins.iter().sum::<f64>() / losses.iter().sum::<f64>(),
        num_trades: trades.len(),
    }
}
```

## Risk Budgeting

Allocate risk rather than capital.

```rust
fn risk_parity_weights(
    volatilities: &[f64],
) -> Vec<f64> {
    // Each asset contributes equally to portfolio risk
    let inv_vols: Vec<f64> = volatilities.iter()
        .map(|v| 1.0 / v)
        .collect();

    let sum_inv_vols: f64 = inv_vols.iter().sum();

    inv_vols.iter()
        .map(|iv| iv / sum_inv_vols)
        .collect()
}

// Example: Equal risk contribution
let vols = vec![0.15, 0.25, 0.30]; // SPY, QQQ, IWM
let weights = risk_parity_weights(&vols);

for (i, w) in weights.iter().enumerate() {
    println!("Asset {}: {:.1}%", i, w * 100.0);
}
```

## Performance

All risk metrics use:
- O(1) online updates (except sorting for VaR)
- Fixed-size buffers (no allocations)
- Cache-friendly access patterns
- 64-byte alignment

**Typical latencies:**
- Sharpe/Sortino update: ~5ns
- VaR update (with sort): ~200ns for N=252
- Drawdown update: ~10ns

## Best Practices

1. **Use multiple metrics** - No single metric tells the whole story
2. **Account for regime changes** - Recalibrate in different market conditions
3. **Size conservatively** - Half-Kelly or less for real money
4. **Monitor correlations** - VaR assumes independence
5. **Stress test** - Historical VaR doesn't capture unprecedented events

## Common Pitfalls

### 1. Overfitting VaR

```rust
// BAD: Using same period for backtest and VaR
let var = ParametricVaR::new(&backtest_returns);

// GOOD: Use out-of-sample data
let var = ParametricVaR::new(&live_returns);
```

### 2. Ignoring Fat Tails

```rust
// Parametric VaR assumes normality
// But financial returns have fat tails!

// Solution: Use historical VaR or Cornish-Fisher
let parametric = ParametricVaR::new(&returns); // Accounts for kurtosis
```

### 3. Full Kelly

```rust
// BAD: Full Kelly is too aggressive
let size = Kelly::binary(0.55, 1.5); // Might be 10%!

// GOOD: Use half-Kelly or quarter-Kelly
let size = Kelly::half_kelly(0.55, 1.5);
```

## Testing

```bash
cargo test risk
```

Tests cover:
- VaR quantile accuracy
- Kelly optimality
- Sharpe calculation
- Drawdown tracking
- Edge cases (all wins, all losses, etc.)

## References

- Jorion: "Value at Risk"
- Thorp: "The Kelly Criterion in Blackjack Sports Betting and the Stock Market"
- Lo: "The Statistics of Sharpe Ratios"
- Bacon: "Practical Portfolio Performance Measurement and Attribution"
