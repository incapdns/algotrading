# Options Pricing Module

Comprehensive options pricing, Greeks, and volatility analytics.

## Overview

The options module provides:
- Black-Scholes-Merton pricing with dividends
- Complete Greeks (first and second order)
- Implied volatility calculation
- Volatility surface modeling
- Exotic options pricing
- VIX-style indices

## Black-Scholes Pricing

### Basic Pricing

```rust
use algotrading::options::*;

let spot = 100.0;
let strike = 105.0;
let rate = 0.05;           // 5% risk-free rate
let dividend_yield = 0.02; // 2% dividend yield
let volatility = 0.25;     // 25% implied vol
let time = 1.0;            // 1 year to expiry

let call_price = black_scholes_call(
    spot, strike, rate, dividend_yield, volatility, time
);

let put_price = black_scholes_put(
    spot, strike, rate, dividend_yield, volatility, time
);

println!("Call: ${:.4}", call_price);
println!("Put: ${:.4}", put_price);
```

### Put-Call Parity

Verify: C - P = S*e^(-qT) - K*e^(-rT)

```rust
let lhs = call_price - put_price;
let rhs = spot * (-dividend_yield * time).exp()
        - strike * (-rate * time).exp();

assert!((lhs - rhs).abs() < 1e-10); // Parity holds
```

## The Greeks

### First-Order Greeks

#### Delta (∂V/∂S)

Sensitivity to spot price changes.

```rust
let delta = delta_call(spot, strike, rate, dividend_yield, volatility, time);
// Range: [0, 1] for calls, [-1, 0] for puts

// Hedge ratio: short delta shares per long call
let hedge_ratio = -delta;
```

#### Vega (∂V/∂σ)

Sensitivity to volatility changes.

```rust
let vega_val = vega(spot, strike, rate, dividend_yield, volatility, time);
// Same for calls and puts

// P&L from 1% vol move
let vol_pnl = vega_val * 0.01;
```

#### Theta (∂V/∂t)

Time decay (usually negative for long options).

```rust
let theta_val = theta_call(spot, strike, rate, dividend_yield, volatility, time);
// Per-year decay, divide by 365 for daily

let daily_decay = theta_val / 365.0;
```

#### Rho (∂V/∂r)

Interest rate sensitivity.

```rust
let rho_val = rho_call(spot, strike, rate, dividend_yield, volatility, time);

// P&L from 1bp rate change
let rate_pnl = rho_val * 0.0001;
```

### Second-Order Greeks

#### Gamma (∂²V/∂S²)

Convexity of delta (same for calls and puts).

```rust
let gamma_val = gamma(spot, strike, rate, dividend_yield, volatility, time);

// Delta change from $1 move
let delta_change = gamma_val * 1.0;

// P&L from realized vs implied variance
let gamma_pnl = 0.5 * gamma_val * (realized_move * realized_move
                                   - implied_move * implied_move);
```

#### Vanna (∂²V/∂S∂σ)

Cross-gamma for delta hedging.

```rust
let vanna_val = vanna(spot, strike, rate, dividend_yield, volatility, time);

// Delta change from 1% vol move
let delta_change = vanna_val * 0.01;
```

#### Volga/Vomma (∂²V/∂σ²)

Convexity of vega.

```rust
let volga_val = volga(spot, strike, rate, dividend_yield, volatility, time);

// Vega change from 1% vol move
let vega_change = volga_val * 0.01;
```

#### Charm (∂²V/∂S∂t)

Delta decay over time.

```rust
let charm_val = charm_call(spot, strike, rate, dividend_yield, volatility, time);

// Daily delta decay
let daily_delta_decay = charm_val / 365.0;
```

#### Veta (∂²V/∂σ∂t)

Vega decay over time.

```rust
let veta_val = veta(spot, strike, rate, dividend_yield, volatility, time);

// Daily vega decay
let daily_vega_decay = veta_val / 365.0;
```

## Implied Volatility

### Newton-Raphson Method

Fast IV calculation using vega.

```rust
let market_price = 5.50;
let implied_vol = implied_volatility_newton(
    market_price,
    spot,
    strike,
    rate,
    dividend_yield,
    time,
    true, // is_call
)?;

println!("Implied vol: {:.2}%", implied_vol * 100.0);
```

**Convergence:** Typically 3-5 iterations

### Bisection Method

More robust but slower.

```rust
let implied_vol = implied_volatility_call(
    market_price,
    spot,
    strike,
    rate,
    dividend_yield,
    time,
)?;
```

## Volatility Surface

### SVI Parameterization

Stochastic Volatility Inspired model for smile interpolation.

```rust
let svi = SVIParameters {
    a: 0.04,      // Overall level
    b: 0.1,       // Smile angle
    rho: -0.3,    // Skewness (negative for equity)
    m: 0.0,       // ATM position
    sigma: 0.2,   // Curvature
};

let log_moneyness = (strike / spot).ln();
let implied_vol = svi.implied_volatility(log_moneyness);
```

**SVI Formula:**
```
σ²(k) = a + b * (ρ(k-m) + √((k-m)² + σ²))
```

### Fitting SVI to Market Data

```rust
let strikes = vec![95.0, 100.0, 105.0, 110.0];
let implied_vols = vec![0.28, 0.25, 0.24, 0.26];

let svi = fit_svi_simple(&strikes, spot, &implied_vols);
```

### Local Volatility (Dupire)

Convert implied vol surface to local vol.

```rust
let local_vol = local_volatility_dupire(
    strike,
    time,
    spot,
    rate,
    dividend_yield,
    implied_vol,
    dvol_dtime,    // ∂σ/∂T
    dvol_dstrike,  // ∂σ/∂K
);
```

**Dupire Formula:**
```
σ_local² = [∂C/∂T + (r-q)K∂C/∂K + qC] / [0.5 K² ∂²C/∂K²]
```

### Forward Volatility

From term structure.

```rust
let vol_1y = 0.20;
let vol_2y = 0.22;

// What's the forward vol from year 1 to year 2?
let fwd_vol = forward_volatility(vol_1y, 1.0, vol_2y, 2.0);
```

**Formula:**
```
σ_fwd = √[(σ_T2² T2 - σ_T1² T1) / (T2 - T1)]
```

## Option Spreads

### Vertical Spreads

```rust
// Bull call spread
let long_call = black_scholes_call(spot, 100.0, rate, div_yield, vol, time);
let short_call = black_scholes_call(spot, 105.0, rate, div_yield, vol, time);
let spread_value = long_call - short_call;
```

### Calendar Spreads

```rust
let spread = calendar_spread_value(
    spot,
    strike: 100.0,
    rate,
    dividend_yield,
    near_vol: 0.25,
    near_expiry: 0.25, // 3 months
    far_vol: 0.28,
    far_expiry: 1.0,   // 1 year
);
```

### Butterfly

Pure convexity play.

```rust
let butterfly = butterfly_value(
    spot,
    strike_low: 95.0,
    strike_mid: 100.0,
    strike_high: 105.0,
    rate,
    dividend_yield,
    volatility,
    time,
);
```

**Payoff:** Long 1 @ 95, Short 2 @ 100, Long 1 @ 105

### Straddle

Long volatility position.

```rust
let straddle = straddle_value(
    spot,
    strike: 100.0, // ATM
    rate,
    dividend_yield,
    volatility,
    time,
);
```

**Breakeven:** S ± Straddle Cost

### Risk Reversal

Skew trade.

```rust
let risk_reversal = risk_reversal_value(
    spot,
    strike_put: 95.0,  // OTM put
    strike_call: 105.0, // OTM call
    rate,
    dividend_yield,
    volatility,
    time,
);
```

**Measures:** Skew = IV(put) - IV(call)

## VIX and Variance Swaps

### VIX Index Calculation

```rust
let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];
let otm_prices = vec![/* put prices K<F, call prices K≥F */];

let vix = vix_index_calculation(
    forward_price,
    rate,
    time,
    &strikes,
    &otm_prices,
);

println!("VIX: {:.2}", vix); // Already annualized & percentage
```

### Variance Swap

```rust
let var_strike = variance_swap_strike(
    &strikes,
    spot,
    rate,
    dividend_yield,
    time,
    &implied_vols,
);

// Payoff at maturity
let realized_variance = /* compute from price path */;
let payoff = realized_variance - var_strike;
```

### Corridor Variance Swap

Only variance between barriers.

```rust
let corridor_strike = corridor_variance_strike(
    spot,
    rate,
    dividend_yield,
    time,
    lower_barrier: 90.0,
    upper_barrier: 110.0,
    &strikes,
    &implied_vols,
);
```

### SKEW Index

Tail risk measure.

```rust
let skew = skew_index(
    spot,
    rate,
    dividend_yield,
    time,
    &strikes,
    &implied_vols,
);

// SKEW = 100 - 10 * normalized_tail_prob
// Normal market: ~100
// High tail risk: 130-150
```

## Exotic Options

### Digital/Binary Options

```rust
let digital = digital_call_value(
    spot,
    strike,
    rate,
    dividend_yield,
    volatility,
    time,
);

// Pays $1 if S > K at expiry, else $0
```

### Barrier Options

```rust
let prob = barrier_hit_probability(
    spot: 100.0,
    barrier: 110.0,     // Up-and-out
    drift: 0.05 - 0.5 * 0.25 * 0.25,
    volatility: 0.25,
    time: 1.0,
);
```

### Power Options

```rust
let power = power_call_value(
    spot,
    strike,
    power: 2.0,  // Payoff = (S/K)^2
    rate,
    dividend_yield,
    volatility,
    time,
);
```

### Cliquet Options

Locks in gains periodically.

```rust
let spot_path = vec![100.0, 105.0, 103.0, 110.0];
let reset_indices = vec![0, 1, 2, 3];

let payoff = cliquet_payoff(
    &spot_path,
    &reset_indices,
    local_cap: 0.10,      // Max 10% per period
    local_floor: -0.05,   // Min -5% per period
    global_cap: 0.50,     // Max 50% total
    global_floor: 0.0,    // Min 0% total
);
```

### Quanto Adjustment

For foreign underlying, domestic payout.

```rust
let adj = quanto_adjustment(
    correlation_fx_spot: -0.5,
    volatility_spot: 0.25,
    volatility_fx: 0.10,
);

// Adjusted drift = drift + adj
```

## Risk Measures

### Skewness

```rust
let skew = risk_neutral_skewness(
    spot,
    rate,
    dividend_yield,
    time,
    atm_vol: 0.25,
    otm_put_vol: 0.30,   // 25-delta put
    otm_call_vol: 0.22,  // 25-delta call
    delta: 0.25,
);

// Negative skew typical for equities
```

### Greeks Profile

```rust
struct GreeksProfile {
    delta: f64,
    gamma: f64,
    vega: f64,
    theta: f64,
}

fn compute_greeks(
    spot: f64,
    strike: f64,
    rate: f64,
    div_yield: f64,
    vol: f64,
    time: f64,
) -> GreeksProfile {
    GreeksProfile {
        delta: delta_call(spot, strike, rate, div_yield, vol, time),
        gamma: gamma(spot, strike, rate, div_yield, vol, time),
        vega: vega(spot, strike, rate, div_yield, vol, time),
        theta: theta_call(spot, strike, rate, div_yield, vol, time),
    }
}
```

## Performance

All functions use:
- Inline assembly hints
- Fast error function approximation
- Cached intermediate values
- Zero allocations

**Typical latencies (AVX2):**
- Black-Scholes: ~5ns
- Greeks: ~8ns
- Implied vol (Newton): ~40ns (3-4 iterations)

## Common Patterns

### Option Chain Analysis

```rust
let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];

for &strike in &strikes {
    let call = black_scholes_call(spot, strike, rate, div_yield, vol, time);
    let delta = delta_call(spot, strike, rate, div_yield, vol, time);
    let gamma_val = gamma(spot, strike, rate, div_yield, vol, time);

    println!("K={}: Price={:.4}, Δ={:.4}, Γ={:.6}",
             strike, call, delta, gamma_val);
}
```

### Volatility Smile

```rust
let spot = 100.0;
let strikes: Vec<f64> = (90..=110).map(|k| k as f64).collect();

for &strike in &strikes {
    let market_price = /* get from market */;
    let iv = implied_volatility_newton(
        market_price, spot, strike, rate, div_yield, time, true
    )?;

    let moneyness = strike / spot;
    println!("{:.2}: {:.2}%", moneyness, iv * 100.0);
}
```

## Testing

```bash
cargo test options
```

Tests verify:
- Put-call parity
- Delta bounds [0,1] for calls
- Gamma always positive
- Implied vol recovery
- ATM approximations

## References

- Hull: "Options, Futures, and Other Derivatives"
- Gatheral: "The Volatility Surface"
- Wilmott: "Paul Wilmott on Quantitative Finance"
- Carr & Madan (1998): "Towards a Theory of Volatility Trading"
