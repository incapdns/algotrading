/// Advanced Options Pricing Kernels for Index Trading
///
/// Black-Scholes-Merton formula with dividend yield
#[inline]
pub fn black_scholes_call(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    volatility: f64,
    time_to_expiry: f64,
) -> f64 {
    if time_to_expiry <= 0.0 {
        return (spot - strike).max(0.0);
    }
    
    let sqrt_t = time_to_expiry.sqrt();
    let d1 = ((spot / strike).ln() + (rate - dividend_yield + 0.5 * volatility * volatility) * time_to_expiry)
           / (volatility * sqrt_t);
    let d2 = d1 - volatility * sqrt_t;
    
    let nd1 = standard_normal_cdf(d1);
    let nd2 = standard_normal_cdf(d2);
    
    spot * (-dividend_yield * time_to_expiry).exp() * nd1
        - strike * (-rate * time_to_expiry).exp() * nd2
}

#[inline]
pub fn black_scholes_put(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    volatility: f64,
    time_to_expiry: f64,
) -> f64 {
    if time_to_expiry <= 0.0 {
        return (strike - spot).max(0.0);
    }
    
    let sqrt_t = time_to_expiry.sqrt();
    let d1 = ((spot / strike).ln() + (rate - dividend_yield + 0.5 * volatility * volatility) * time_to_expiry)
           / (volatility * sqrt_t);
    let d2 = d1 - volatility * sqrt_t;
    
    let nnd1 = standard_normal_cdf(-d1);
    let nnd2 = standard_normal_cdf(-d2);
    
    strike * (-rate * time_to_expiry).exp() * nnd2
        - spot * (-dividend_yield * time_to_expiry).exp() * nnd1
}

/// The Greeks - Options Sensitivities for Trading and Risk Management
///
/// The Greeks measure how option prices change with respect to market variables.
/// Essential for hedging, risk assessment, and position management.
///
/// # Delta (∂V/∂S): Directional Exposure
///
/// **What it means**: How much the option price changes per $1 move in the underlying.
///
/// **Call delta**: 0 to 1 (ITM calls near 1, OTM calls near 0)
/// **Put delta**: -1 to 0 (ITM puts near -1, OTM puts near 0)
///
/// **Trading interpretation**:
/// - Delta = 0.50 → Option behaves like owning 50 shares of stock
/// - Delta = 1.0 → Deep ITM call moves dollar-for-dollar with stock
/// - Delta = -0.30 → Put gains $0.30 per $1 stock decline
///
/// **Hedging usage**:
/// ```ignore
/// // Protect $100k stock portfolio with puts
/// let portfolio_value = 100_000.0;
/// let spot = 450.0;
/// let shares = portfolio_value / spot;
///
/// // Buy protective puts with delta = -0.30
/// let put_delta = delta_put(spot, 430.0, 0.05, 0.02, 0.20, 0.25);
/// let puts_needed = (shares * -1.0) / put_delta;
/// println!("Buy {} puts for protection", puts_needed.round());
/// ```
///
/// # Gamma (∂²V/∂S²): Delta Risk
///
/// **What it means**: How fast delta changes as stock moves.
///
/// **Trading interpretation**:
/// - High gamma (ATM options) → Delta changes rapidly
/// - Low gamma (ITM/OTM options) → Delta stays relatively stable
/// - Gamma is highest near expiration for ATM options
///
/// **Risk implications**:
/// - Long gamma = profit from large moves in either direction
/// - Short gamma = need frequent rebalancing, risk of losses on big moves
/// - Gamma scalping: profit from volatility by delta-hedging
///
/// **Position sizing**:
/// ```ignore
/// // Check gamma exposure before earnings
/// let gamma = gamma(spot, strike, rate, div, vol, time);
/// let gamma_dollars = gamma * spot * spot / 100.0;  // Dollar gamma
/// println!("Delta changes by ${:.2} per 1% move", gamma_dollars);
/// ```
///
/// # Vega (∂V/∂σ): Volatility Exposure
///
/// **What it means**: How much option value changes per 1% change in implied volatility.
///
/// **Trading interpretation**:
/// - Vega = 0.15 → Option gains $0.15 per 1 volatility point increase
/// - Long vega = bet on rising volatility (buy options)
/// - Short vega = bet on falling volatility (sell options)
///
/// **Common strategies**:
/// - **Volatility arbitrage**: Buy underpriced vol, sell overpriced vol
/// - **Pre-earnings**: Long vega into earnings (vol crush after)
/// - **VIX hedging**: Long vega protects in market crashes
///
/// ```ignore
/// // Calculate vega exposure for straddle
/// let vega_total = 2.0 * vega(spot, strike, rate, div, vol, time);
/// let vol_change = 5.0;  // Expecting 5-point vol increase
/// let pnl = vega_total * vol_change;
/// println!("P&L from vol rise: ${:.2}", pnl);
/// ```
///
/// # Theta (∂V/∂t): Time Decay
///
/// **What it means**: How much option value decays per day (always negative for long positions).
///
/// **Trading interpretation**:
/// - Theta = -0.05 → Option loses $0.05 per day due to time decay
/// - Accelerates as expiration approaches
/// - ATM options have highest theta
///
/// **Strategic implications**:
/// - **Theta decay sellers**: Sell options to collect premium (covered calls, cash-secured puts)
/// - **Theta buyers**: Must be right about direction/volatility quickly
/// - **Weekend decay**: Options lose ~3 days of theta over weekends
///
/// # Rho (∂V/∂r): Interest Rate Sensitivity
///
/// **What it means**: Change in value per 1% change in risk-free rate.
///
/// Usually the least important Greek for equity options (small effect).
/// More relevant for long-dated options and interest rate derivatives.

/// **Delta** for call options
///
/// Measures the rate of change of option value with respect to the underlying price.
///
/// # Trading Context
///
/// - **Hedging**: Delta tells you how many shares of stock to hedge with
/// - **Position equivalent**: Delta shows stock-equivalent position size
/// - **ATM calls**: Delta ≈ 0.50 (50% chance of expiring in-the-money)
/// - **Deep ITM calls**: Delta → 1.0 (moves like owning stock)
/// - **OTM calls**: Delta → 0 (little sensitivity to price moves)
///
/// # Returns
///
/// Delta value between 0 and 1 for calls. Multiply by 100 for shares equivalent.
#[inline]
pub fn delta_call(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    volatility: f64,
    time_to_expiry: f64,
) -> f64 {
    if time_to_expiry <= 0.0 {
        return if spot > strike { 1.0 } else { 0.0 };
    }
    
    let sqrt_t = time_to_expiry.sqrt();
    let d1 = ((spot / strike).ln() + (rate - dividend_yield + 0.5 * volatility * volatility) * time_to_expiry)
           / (volatility * sqrt_t);
    
    (-dividend_yield * time_to_expiry).exp() * standard_normal_cdf(d1)
}

#[inline]
pub fn delta_put(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    volatility: f64,
    time_to_expiry: f64,
) -> f64 {
    delta_call(spot, strike, rate, dividend_yield, volatility, time_to_expiry) - (-dividend_yield * time_to_expiry).exp()
}

/// Gamma: ∂²V/∂S² (same for calls and puts)
#[inline]
pub fn gamma(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    volatility: f64,
    time_to_expiry: f64,
) -> f64 {
    if time_to_expiry <= 0.0 {
        return 0.0;
    }
    
    let sqrt_t = time_to_expiry.sqrt();
    let d1 = ((spot / strike).ln() + (rate - dividend_yield + 0.5 * volatility * volatility) * time_to_expiry)
           / (volatility * sqrt_t);
    
    let nd1_prime = standard_normal_pdf(d1);
    (-dividend_yield * time_to_expiry).exp() * nd1_prime / (spot * volatility * sqrt_t)
}

/// Vega: ∂V/∂σ (same for calls and puts)
#[inline]
pub fn vega(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    volatility: f64,
    time_to_expiry: f64,
) -> f64 {
    if time_to_expiry <= 0.0 {
        return 0.0;
    }
    
    let sqrt_t = time_to_expiry.sqrt();
    let d1 = ((spot / strike).ln() + (rate - dividend_yield + 0.5 * volatility * volatility) * time_to_expiry)
           / (volatility * sqrt_t);
    
    let nd1_prime = standard_normal_pdf(d1);
    spot * (-dividend_yield * time_to_expiry).exp() * nd1_prime * sqrt_t
}

/// Theta: -∂V/∂t (time decay)
#[inline]
pub fn theta_call(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    volatility: f64,
    time_to_expiry: f64,
) -> f64 {
    if time_to_expiry <= 0.0 {
        return 0.0;
    }
    
    let sqrt_t = time_to_expiry.sqrt();
    let d1 = ((spot / strike).ln() + (rate - dividend_yield + 0.5 * volatility * volatility) * time_to_expiry)
           / (volatility * sqrt_t);
    let d2 = d1 - volatility * sqrt_t;
    
    let nd1 = standard_normal_cdf(d1);
    let nd2 = standard_normal_cdf(d2);
    let nd1_prime = standard_normal_pdf(d1);
    
    let term1 = -spot * nd1_prime * volatility * (-dividend_yield * time_to_expiry).exp() / (2.0 * sqrt_t);
    let term2 = dividend_yield * spot * nd1 * (-dividend_yield * time_to_expiry).exp();
    let term3 = -rate * strike * (-rate * time_to_expiry).exp() * nd2;
    
    term1 + term2 + term3
}

#[inline]
pub fn theta_put(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    volatility: f64,
    time_to_expiry: f64,
) -> f64 {
    if time_to_expiry <= 0.0 {
        return 0.0;
    }
    
    let sqrt_t = time_to_expiry.sqrt();
    let d1 = ((spot / strike).ln() + (rate - dividend_yield + 0.5 * volatility * volatility) * time_to_expiry)
           / (volatility * sqrt_t);
    let d2 = d1 - volatility * sqrt_t;
    
    let nnd1 = standard_normal_cdf(-d1);
    let nnd2 = standard_normal_cdf(-d2);
    let nd1_prime = standard_normal_pdf(d1);
    
    let term1 = -spot * nd1_prime * volatility * (-dividend_yield * time_to_expiry).exp() / (2.0 * sqrt_t);
    let term2 = -dividend_yield * spot * nnd1 * (-dividend_yield * time_to_expiry).exp();
    let term3 = rate * strike * (-rate * time_to_expiry).exp() * nnd2;
    
    term1 + term2 + term3
}

/// Rho: ∂V/∂r
#[inline]
pub fn rho_call(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    volatility: f64,
    time_to_expiry: f64,
) -> f64 {
    if time_to_expiry <= 0.0 {
        return 0.0;
    }
    
    let sqrt_t = time_to_expiry.sqrt();
    let d1 = ((spot / strike).ln() + (rate - dividend_yield + 0.5 * volatility * volatility) * time_to_expiry)
           / (volatility * sqrt_t);
    let d2 = d1 - volatility * sqrt_t;
    
    let nd2 = standard_normal_cdf(d2);
    strike * time_to_expiry * (-rate * time_to_expiry).exp() * nd2
}

#[inline]
pub fn rho_put(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    volatility: f64,
    time_to_expiry: f64,
) -> f64 {
    if time_to_expiry <= 0.0 {
        return 0.0;
    }
    
    let sqrt_t = time_to_expiry.sqrt();
    let d1 = ((spot / strike).ln() + (rate - dividend_yield + 0.5 * volatility * volatility) * time_to_expiry)
           / (volatility * sqrt_t);
    let d2 = d1 - volatility * sqrt_t;
    
    let nnd2 = standard_normal_cdf(-d2);
    -strike * time_to_expiry * (-rate * time_to_expiry).exp() * nnd2
}

/// Implied volatility using Brent's method
pub fn implied_volatility_call(
    market_price: f64,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    time_to_expiry: f64,
) -> Option<f64> {
    if time_to_expiry <= 0.0 {
        return None;
    }
    
    let intrinsic = (spot - strike).max(0.0);
    if market_price < intrinsic {
        return None; // Arbitrage
    }
    
    // Brent's method bounds
    let mut vol_low = 0.001;
    let mut vol_high = 5.0;
    const MAX_ITER: usize = 100;
    const TOLERANCE: f64 = 1e-6;
    
    for _ in 0..MAX_ITER {
        let vol_mid = (vol_low + vol_high) / 2.0;
        let price_mid = black_scholes_call(spot, strike, rate, dividend_yield, vol_mid, time_to_expiry);
        let diff = price_mid - market_price;
        
        if diff.abs() < TOLERANCE {
            return Some(vol_mid);
        }
        
        if diff > 0.0 {
            vol_high = vol_mid;
        } else {
            vol_low = vol_mid;
        }
        
        if vol_high - vol_low < TOLERANCE {
            return Some(vol_mid);
        }
    }
    
    Some((vol_low + vol_high) / 2.0)
}

/// Implied volatility using Newton-Raphson (faster with vega)
pub fn implied_volatility_newton(
    market_price: f64,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    time_to_expiry: f64,
    is_call: bool,
) -> Option<f64> {
    if time_to_expiry <= 0.0 {
        return None;
    }
    
    let mut vol = 0.3; // Initial guess
    const MAX_ITER: usize = 50;
    const TOLERANCE: f64 = 1e-6;
    
    for _ in 0..MAX_ITER {
        let price = if is_call {
            black_scholes_call(spot, strike, rate, dividend_yield, vol, time_to_expiry)
        } else {
            black_scholes_put(spot, strike, rate, dividend_yield, vol, time_to_expiry)
        };
        
        let diff = price - market_price;
        
        if diff.abs() < TOLERANCE {
            return Some(vol);
        }
        
        let v = vega(spot, strike, rate, dividend_yield, vol, time_to_expiry);
        
        if v < 1e-10 {
            return None; // Vega too small
        }
        
        vol = vol - diff / v;
        
        // Keep volatility positive and reasonable
        vol = vol.clamp(0.001, 5.0);
    }
    
    Some(vol)
}

/// Volatility smile/skew parameterization (SVI - Stochastic Volatility Inspired)
pub struct SVIParameters {
    pub a: f64,  // Overall level
    pub b: f64,  // Angle of smile
    pub rho: f64, // Skewness
    pub m: f64,  // ATM position
    pub sigma: f64, // Curvature
}

impl SVIParameters {
    /// Compute implied variance for given log-moneyness
    #[inline]
    pub fn implied_variance(&self, log_moneyness: f64) -> f64 {
        let k = log_moneyness;
        let diff = k - self.m;
        self.a + self.b * (self.rho * diff + (diff * diff + self.sigma * self.sigma).sqrt())
    }
    
    /// Compute implied volatility for given log-moneyness
    #[inline]
    pub fn implied_volatility(&self, log_moneyness: f64) -> f64 {
        self.implied_variance(log_moneyness).sqrt()
    }
}

/// Fit SVI to market data (simplified fitting)
pub fn fit_svi_simple(
    strikes: &[f64],
    spot: f64,
    implied_vols: &[f64],
) -> SVIParameters {
    assert_eq!(strikes.len(), implied_vols.len());
    
    // Compute log-moneyness
    let log_moneyness: Vec<f64> = strikes.iter()
        .map(|k| (k / spot).ln())
        .collect();
    
    // Simple initial guess based on ATM vol and extremes
    let atm_idx = log_moneyness.iter()
        .enumerate()
        .min_by(|(_, x), (_, y)| {
            // Use total_cmp for f64 to handle NaN correctly
            x.abs().total_cmp(&y.abs())
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    
    let atm_var = implied_vols[atm_idx] * implied_vols[atm_idx];
    
    SVIParameters {
        a: atm_var * 0.8,
        b: 0.1,
        rho: -0.3, // Typical equity skew
        m: log_moneyness[atm_idx],
        sigma: 0.2,
    }
}

/// Variance swap strike from smile
pub fn variance_swap_strike(
    strikes: &[f64],
    spot: f64,
    rate: f64,
    dividend_yield: f64,
    time_to_expiry: f64,
    implied_vols: &[f64],
) -> f64 {
    assert_eq!(strikes.len(), implied_vols.len());
    
    let forward = spot * ((rate - dividend_yield) * time_to_expiry).exp();
    let mut variance = 0.0;
    
    // Integrate using option prices
    for i in 0..strikes.len() {
        let k = strikes[i];
        let vol = implied_vols[i];
        
        let price = if k < forward {
            black_scholes_put(spot, k, rate, dividend_yield, vol, time_to_expiry)
        } else {
            black_scholes_call(spot, k, rate, dividend_yield, vol, time_to_expiry)
        };
        
        // Weight by 1/K^2
        let weight = if i < strikes.len() - 1 {
            strikes[i + 1] - strikes[i]
        } else if i > 0 {
            strikes[i] - strikes[i - 1]
        } else {
            1.0
        };
        
        variance += 2.0 * weight * price / (k * k);
    }
    
    variance *= (rate * time_to_expiry).exp() / time_to_expiry;
    variance.sqrt()
}

/// Risk-neutral skewness from option prices
pub fn risk_neutral_skewness(
    spot: f64,
    rate: f64,
    dividend_yield: f64,
    time_to_expiry: f64,
    atm_vol: f64,
    otm_put_vol: f64,
    otm_call_vol: f64,
    delta: f64, // Common delta for OTM options (e.g. 0.25)
) -> f64 {
    // Use Bakshi-Kapadia-Madan formula approximation
    let forward = spot * ((rate - dividend_yield) * time_to_expiry).exp();
    
    // Strikes corresponding to delta
    let put_strike = forward * (1.0 - delta);
    let call_strike = forward * (1.0 + delta);
    
    let put_price = black_scholes_put(spot, put_strike, rate, dividend_yield, otm_put_vol, time_to_expiry);
    let call_price = black_scholes_call(spot, call_strike, rate, dividend_yield, otm_call_vol, time_to_expiry);
    let atm_price = black_scholes_call(spot, forward, rate, dividend_yield, atm_vol, time_to_expiry);
    
    // Simplified skewness approximation
    (call_price - 2.0 * atm_price + put_price) / (delta * forward).powi(3)
}

/// Butterflies - pure convexity plays
#[inline]
pub fn butterfly_value(
    spot: f64,
    strike_low: f64,
    strike_mid: f64,
    strike_high: f64,
    rate: f64,
    dividend_yield: f64,
    volatility: f64,
    time_to_expiry: f64,
) -> f64 {
    let call_low = black_scholes_call(spot, strike_low, rate, dividend_yield, volatility, time_to_expiry);
    let call_mid = black_scholes_call(spot, strike_mid, rate, dividend_yield, volatility, time_to_expiry);
    let call_high = black_scholes_call(spot, strike_high, rate, dividend_yield, volatility, time_to_expiry);
    
    call_low - 2.0 * call_mid + call_high
}

/// Straddle (long vol position)
#[inline]
pub fn straddle_value(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    volatility: f64,
    time_to_expiry: f64,
) -> f64 {
    black_scholes_call(spot, strike, rate, dividend_yield, volatility, time_to_expiry)
        + black_scholes_put(spot, strike, rate, dividend_yield, volatility, time_to_expiry)
}

/// Strangle (cheaper long vol)
#[inline]
pub fn strangle_value(
    spot: f64,
    strike_put: f64,
    strike_call: f64,
    rate: f64,
    dividend_yield: f64,
    volatility: f64,
    time_to_expiry: f64,
) -> f64 {
    black_scholes_put(spot, strike_put, rate, dividend_yield, volatility, time_to_expiry)
        + black_scholes_call(spot, strike_call, rate, dividend_yield, volatility, time_to_expiry)
}

/// Risk reversal (skew trade)
#[inline]
pub fn risk_reversal_value(
    spot: f64,
    strike_put: f64,
    strike_call: f64,
    rate: f64,
    dividend_yield: f64,
    volatility: f64,
    time_to_expiry: f64,
) -> f64 {
    black_scholes_call(spot, strike_call, rate, dividend_yield, volatility, time_to_expiry)
        - black_scholes_put(spot, strike_put, rate, dividend_yield, volatility, time_to_expiry)
}

// Re-export consolidated probability utilities for convenience
pub use crate::probability::utils::{standard_normal_cdf, standard_normal_pdf, erf};

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_black_scholes_call() {
        let price = black_scholes_call(100.0, 100.0, 0.05, 0.02, 0.2, 1.0);
        assert!(price > 0.0 && price < 100.0);
    }
    
    #[test]
    fn test_put_call_parity() {
        let s = 100.0;
        let k = 100.0;
        let r = 0.05;
        let q = 0.02;
        let vol = 0.2;
        let t = 1.0;
        
        let call = black_scholes_call(s, k, r, q, vol, t);
        let put = black_scholes_put(s, k, r, q, vol, t);
        
        let lhs = call - put;
        let rhs = s * (-q * t).exp() - k * (-r * t).exp();
        
        assert!((lhs - rhs).abs() < 1e-10);
    }
    
    #[test]
    fn test_delta_bounds() {
        let delta = delta_call(100.0, 100.0, 0.05, 0.02, 0.2, 1.0);
        assert!(delta >= 0.0 && delta <= 1.0);
    }
    
    #[test]
    fn test_gamma_positive() {
        let gamma_val = gamma(100.0, 100.0, 0.05, 0.02, 0.2, 1.0);
        assert!(gamma_val > 0.0);
    }
    
    #[test]
    fn test_implied_vol_recovery() {
        let s = 100.0;
        let k = 100.0;
        let r = 0.05;
        let q = 0.02;
        let true_vol = 0.25;
        let t = 1.0;
        
        let market_price = black_scholes_call(s, k, r, q, true_vol, t);
        let implied = implied_volatility_newton(market_price, s, k, r, q, t, true).unwrap();
        
        assert!((implied - true_vol).abs() < 1e-4);
    }
}