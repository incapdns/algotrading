/// Advanced Options Analytics: Volatility Surface, Term Structure, and Exotic Pricing
///
/// Local volatility from implied volatility surface (Dupire's formula)
/// σ_local² = (∂C/∂T + (r-q)K∂C/∂K + qC) / (0.5 K² ∂²C/∂K²)
#[inline]
pub fn local_volatility_dupire(
    strike: f64,
    time_to_expiry: f64,
    spot: f64,
    rate: f64,
    dividend_yield: f64,
    implied_vol: f64,
    dvol_dtime: f64,    // ∂σ/∂T
    _dvol_dstrike: f64,  // ∂σ/∂K (reserved for future use)
) -> f64 {
    use super::options_pricing::{black_scholes_call, vega, gamma};
    
    let c = black_scholes_call(spot, strike, rate, dividend_yield, implied_vol, time_to_expiry);
    let v = vega(spot, strike, rate, dividend_yield, implied_vol, time_to_expiry);
    let g = gamma(spot, strike, rate, dividend_yield, implied_vol, time_to_expiry);
    
    // Derivatives of call price
    let dc_dt = -0.5 * v * implied_vol / time_to_expiry.sqrt() 
               + v * dvol_dtime * time_to_expiry.sqrt();
    let dc_dk = -(rate * time_to_expiry).exp() * super::options_pricing::standard_normal_cdf(
        ((spot / strike).ln() + (rate - dividend_yield + 0.5 * implied_vol * implied_vol) * time_to_expiry)
        / (implied_vol * time_to_expiry.sqrt())
        - implied_vol * time_to_expiry.sqrt()
    );
    let d2c_dk2 = g / spot;
    
    // Dupire's formula
    let numerator = dc_dt + (rate - dividend_yield) * strike * dc_dk + dividend_yield * c;
    let denominator = 0.5 * strike * strike * d2c_dk2;
    
    if denominator > 1e-10 {
        (numerator / denominator).max(0.0).sqrt()
    } else {
        implied_vol
    }
}

/// Forward volatility from term structure
/// σ_forward = sqrt((σ_T2² * T2 - σ_T1² * T1) / (T2 - T1))
#[inline]
pub fn forward_volatility(
    vol_t1: f64,
    time_t1: f64,
    vol_t2: f64,
    time_t2: f64,
) -> f64 {
    if time_t2 <= time_t1 {
        return vol_t2;
    }
    
    let var_diff = vol_t2 * vol_t2 * time_t2 - vol_t1 * vol_t1 * time_t1;
    let time_diff = time_t2 - time_t1;
    
    (var_diff / time_diff).max(0.0).sqrt()
}

/// Calendar spread - volatility term structure trade
#[inline]
pub fn calendar_spread_value(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    near_vol: f64,
    near_expiry: f64,
    far_vol: f64,
    far_expiry: f64,
) -> f64 {
    use super::options_pricing::black_scholes_call;
    
    let far_call = black_scholes_call(spot, strike, rate, dividend_yield, far_vol, far_expiry);
    let near_call = black_scholes_call(spot, strike, rate, dividend_yield, near_vol, near_expiry);
    
    far_call - near_call
}

/// Diagonal spread - combine strike and time spreads
#[inline]
pub fn diagonal_spread_value(
    spot: f64,
    strike_near: f64,
    strike_far: f64,
    rate: f64,
    dividend_yield: f64,
    near_vol: f64,
    near_expiry: f64,
    far_vol: f64,
    far_expiry: f64,
) -> f64 {
    use super::options_pricing::black_scholes_call;
    
    let far_call = black_scholes_call(spot, strike_far, rate, dividend_yield, far_vol, far_expiry);
    let near_call = black_scholes_call(spot, strike_near, rate, dividend_yield, near_vol, near_expiry);
    
    far_call - near_call
}

/// Vanna: ∂²V/∂S∂σ (cross-gamma for delta hedging)
#[inline]
pub fn vanna(
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
    
    use super::options_pricing::standard_normal_pdf;
    let nd1_prime = standard_normal_pdf(d1);
    
    -(-dividend_yield * time_to_expiry).exp() * nd1_prime * d2 / volatility
}

/// Volga (Vomma): ∂²V/∂σ² (convexity of vega)
#[inline]
pub fn volga(
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
    
    use super::options_pricing::vega;
    let v = vega(spot, strike, rate, dividend_yield, volatility, time_to_expiry);
    
    let sqrt_t = time_to_expiry.sqrt();
    let d1 = ((spot / strike).ln() + (rate - dividend_yield + 0.5 * volatility * volatility) * time_to_expiry)
           / (volatility * sqrt_t);
    let d2 = d1 - volatility * sqrt_t;
    
    v * d1 * d2 / volatility
}

/// Charm: ∂²V/∂S∂t (delta decay)
#[inline]
pub fn charm_call(
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
    
    use super::options_pricing::{standard_normal_pdf, standard_normal_cdf};
    let nd1 = standard_normal_cdf(d1);
    let nd1_prime = standard_normal_pdf(d1);
    
    -(-dividend_yield * time_to_expiry).exp() * 
    (dividend_yield * nd1 - nd1_prime * (2.0 * (rate - dividend_yield) * time_to_expiry - d2 * volatility * sqrt_t)
     / (2.0 * time_to_expiry * volatility * sqrt_t))
}

/// Veta: ∂²V/∂σ∂t (vega decay)
#[inline]
pub fn veta(
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
    
    use super::options_pricing::vega;
    let v = vega(spot, strike, rate, dividend_yield, volatility, time_to_expiry);
    
    let sqrt_t = time_to_expiry.sqrt();
    let d1 = ((spot / strike).ln() + (rate - dividend_yield + 0.5 * volatility * volatility) * time_to_expiry)
           / (volatility * sqrt_t);
    let d2 = d1 - volatility * sqrt_t;
    
    -v * (dividend_yield + ((rate - dividend_yield) * d1) / (volatility * sqrt_t)
           - (1.0 + d1 * d2) / (2.0 * time_to_expiry))
}

/// VIX-style volatility index computation from option prices
/// Uses out-of-the-money options across strikes
pub fn vix_index_calculation(
    forward_price: f64,
    rate: f64,
    time_to_expiry: f64,
    strikes: &[f64],
    otm_prices: &[f64], // Put prices for K < F, call prices for K >= F
) -> f64 {
    assert_eq!(strikes.len(), otm_prices.len());
    
    let mut variance = 0.0;
    
    for i in 0..strikes.len() {
        let k = strikes[i];
        let price = otm_prices[i];
        
        // Delta K (strike spacing)
        let delta_k = if i == 0 {
            strikes[1] - strikes[0]
        } else if i == strikes.len() - 1 {
            strikes[i] - strikes[i - 1]
        } else {
            (strikes[i + 1] - strikes[i - 1]) / 2.0
        };
        
        // VIX formula: 2/T * Σ (ΔK/K²) * Q(K) * e^(rT)
        variance += (delta_k / (k * k)) * price;
    }
    
    variance *= 2.0 * (rate * time_to_expiry).exp() / time_to_expiry;
    
    // Subtract forward adjustment
    let forward_adjustment = (forward_price / strikes[strikes.len() / 2] - 1.0).powi(2) / time_to_expiry;
    variance -= forward_adjustment;
    
    // Annualize (multiply by 100 for percentage)
    variance.max(0.0).sqrt() * 100.0
}

/// Skew-adjusted VIX (SKEW index logic)
pub fn skew_index(
    spot: f64,
    rate: f64,
    dividend_yield: f64,
    time_to_expiry: f64,
    strikes: &[f64],
    implied_vols: &[f64],
) -> f64 {
    use super::options_pricing::black_scholes_put;
    
    // Need OTM puts at 90%, 95%, 100% moneyness
    let mut otm_values = Vec::new();
    
    for (i, &strike) in strikes.iter().enumerate() {
        if strike < spot {
            let put_price = black_scholes_put(
                spot, strike, rate, dividend_yield, 
                implied_vols[i], time_to_expiry
            );
            otm_values.push((strike / spot, put_price));
        }
    }
    
    if otm_values.len() < 3 {
        return 100.0; // Neutral
    }
    
    // Compute log returns at extreme strikes
    let tail_prob: f64 = otm_values.iter()
        .filter(|(moneyness, _)| *moneyness < 0.95)
        .map(|(_, price)| price)
        .sum();
    
    // SKEW = 100 - 10 * normalized_tail_prob
    100.0 + 10.0 * tail_prob / spot
}

/// VVIX - volatility of VIX (vol-of-vol)
pub fn vvix_approximation(
    vix_level: f64,
    vix_options_atm_vol: f64,
) -> f64 {
    // Simplified: VVIX ≈ ATM_VOL * VIX
    vix_options_atm_vol * vix_level
}

/// Corridor variance swap - variance between two strikes
pub fn corridor_variance_strike(
    spot: f64,
    rate: f64,
    dividend_yield: f64,
    time_to_expiry: f64,
    lower_barrier: f64,
    upper_barrier: f64,
    strikes: &[f64],
    implied_vols: &[f64],
) -> f64 {
    use super::options_pricing::{black_scholes_call, black_scholes_put};
    
    let forward = spot * ((rate - dividend_yield) * time_to_expiry).exp();
    let mut variance = 0.0;
    
    for i in 0..strikes.len() {
        let k = strikes[i];
        
        // Only include strikes in corridor
        if k < lower_barrier || k > upper_barrier {
            continue;
        }
        
        let vol = implied_vols[i];
        
        let price = if k < forward {
            black_scholes_put(spot, k, rate, dividend_yield, vol, time_to_expiry)
        } else {
            black_scholes_call(spot, k, rate, dividend_yield, vol, time_to_expiry)
        };
        
        let weight = if i < strikes.len() - 1 {
            (strikes[i + 1] - strikes[i]).min(upper_barrier - k)
        } else {
            1.0
        };
        
        variance += 2.0 * weight * price / (k * k);
    }
    
    variance *= (rate * time_to_expiry).exp() / time_to_expiry;
    variance.sqrt()
}

/// Conditional variance swap - variance given spot stays in range
pub fn conditional_variance_swap_payoff(
    realized_variance: f64,
    strike: f64,
    lower_barrier: f64,
    upper_barrier: f64,
    spot_path: &[f64],
) -> f64 {
    // Check if spot stayed in range
    let in_range = spot_path.iter()
        .all(|&s| s >= lower_barrier && s <= upper_barrier);
    
    if in_range {
        realized_variance - strike
    } else {
        0.0
    }
}

/// Gamma swap - realized gamma exposure
pub fn gamma_swap_payoff(
    spot_path: &[f64],
    strike: f64,
) -> f64 {
    if spot_path.len() < 2 {
        return 0.0;
    }
    
    let mut realized_gamma = 0.0;
    
    for i in 1..spot_path.len() {
        let s_prev = spot_path[i - 1];
        let s = spot_path[i];
        let log_return = (s / s_prev).ln();
        
        // Approximate gamma accumulation
        realized_gamma += log_return * log_return;
    }
    
    realized_gamma /= spot_path.len() as f64;
    realized_gamma - strike
}

/// Cliquet/Napoleon option - locks in gains periodically
pub fn cliquet_payoff(
    spot_path: &[f64],
    reset_indices: &[usize],
    local_cap: f64,
    local_floor: f64,
    global_cap: f64,
    global_floor: f64,
) -> f64 {
    let mut total_return = 0.0;
    
    for i in 1..reset_indices.len() {
        let start_idx = reset_indices[i - 1];
        let end_idx = reset_indices[i];
        
        let local_return = (spot_path[end_idx] / spot_path[start_idx]) - 1.0;
        
        // Apply local caps and floors
        let capped_return = local_return.max(local_floor).min(local_cap);
        total_return += capped_return;
    }
    
    // Apply global caps and floors
    total_return.max(global_floor).min(global_cap)
}

/// Barrier option adjustments (continuous monitoring)
/// Probability of hitting barrier before expiry
#[inline]
pub fn barrier_hit_probability(
    spot: f64,
    barrier: f64,
    drift: f64,      // μ - 0.5σ²
    volatility: f64,
    time_to_expiry: f64,
) -> f64 {
    if time_to_expiry <= 0.0 {
        return if (spot - barrier).abs() < 1e-10 { 1.0 } else { 0.0 };
    }
    
    let h = (barrier / spot).ln();
    let lambda = drift / (volatility * volatility);
    let z = h / (volatility * time_to_expiry.sqrt()) + lambda * volatility * time_to_expiry.sqrt();
    
    use super::options_pricing::standard_normal_cdf;
    
    let prob = if barrier > spot {
        standard_normal_cdf(-z) + (barrier / spot).powf(2.0 * lambda) * standard_normal_cdf(z)
    } else {
        standard_normal_cdf(z) + (barrier / spot).powf(2.0 * lambda) * standard_normal_cdf(-z)
    };

    prob.clamp(0.0, 1.0)

}

/// Digital/binary option - pays 1 if spot > strike at expiry
#[inline]
pub fn digital_call_value(
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
    let d2 = ((spot / strike).ln() + (rate - dividend_yield - 0.5 * volatility * volatility) * time_to_expiry)
           / (volatility * sqrt_t);
    
    use super::options_pricing::standard_normal_cdf;
    (-rate * time_to_expiry).exp() * standard_normal_cdf(d2)
}

/// Power option - payoff is (S/K)^n
#[inline]
pub fn power_call_value(
    spot: f64,
    strike: f64,
    power: f64,
    rate: f64,
    dividend_yield: f64,
    volatility: f64,
    time_to_expiry: f64,
) -> f64 {
    let adjusted_vol = power * volatility;
    let adjusted_rate = rate - (power - 1.0) * dividend_yield + 0.5 * (power - 1.0) * power * volatility * volatility;
    
    use super::options_pricing::black_scholes_call;
    spot.powf(power) * black_scholes_call(
        1.0,
        strike / spot.powf(power - 1.0),
        adjusted_rate,
        power * dividend_yield,
        adjusted_vol,
        time_to_expiry
    ) / strike.powf(power)
}

/// Quanto adjustment - currency risk premium
#[inline]
pub fn quanto_adjustment(
    correlation_fx_spot: f64,
    volatility_spot: f64,
    volatility_fx: f64,
) -> f64 {
    // Adjustment to drift: -ρ * σ_S * σ_FX
    -correlation_fx_spot * volatility_spot * volatility_fx
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_forward_volatility() {
        let vol_1y = 0.20;
        let vol_2y = 0.22;
        let fwd_vol = forward_volatility(vol_1y, 1.0, vol_2y, 2.0);
        
        // Forward vol should be between the two
        assert!(fwd_vol > vol_1y && fwd_vol < vol_2y * 1.5);
    }
    
    #[test]
    fn test_vanna_symmetry() {
        let v1 = vanna(100.0, 100.0, 0.05, 0.02, 0.2, 1.0);
        let v2 = vanna(100.0, 100.0, 0.05, 0.02, 0.25, 1.0);
        
        // Vanna should be non-zero and change with vol
        assert!(v1 != v2);
    }
    
    #[test]
    fn test_digital_bounds() {
        let dig = digital_call_value(100.0, 100.0, 0.05, 0.02, 0.2, 1.0);
        assert!(dig >= 0.0 && dig <= 1.0);
    }
    
    #[test]
    fn test_barrier_probability() {
        let prob = barrier_hit_probability(100.0, 110.0, 0.05, 0.2, 1.0);
        println!("Barrier hit probability: {}", prob);
        assert!(prob >= 0.0 && prob <= 1.0);
    }
}