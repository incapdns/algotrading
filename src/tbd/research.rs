/// Realized volatility measures beyond simple variance

/// Realized Kernel (Barndorff-Nielsen et al.) - noise-robust volatility estimator
/// Uses Parzen kernel to handle market microstructure noise
#[inline]
pub fn realized_kernel(returns: &[f64], bandwidth: usize) -> f64 {
    let n = returns.len();
    if n < 2 {
        return 0.0;
    }
    
    let h = bandwidth.min(n / 2);
    
    // Parzen weight function
    let parzen_weight = |x: f64| -> f64 {
        let abs_x = x.abs();
        if abs_x <= 0.5 {
            1.0 - 6.0 * abs_x * abs_x + 6.0 * abs_x * abs_x * abs_x
        } else if abs_x <= 1.0 {
            2.0 * (1.0 - abs_x).powi(3)
        } else {
            0.0
        }
    };
    
    // Compute autocovariances
    let mut rv = 0.0;
    
    for lag in 0..h {
        let mut gamma = 0.0;
        for i in 0..(n - lag) {
            gamma += returns[i] * returns[i + lag];
        }
        
        let weight = parzen_weight(lag as f64 / (h as f64));
        if lag == 0 {
            rv += gamma * weight;
        } else {
            rv += 2.0 * gamma * weight;
        }
    }
    
    rv
}

/// Two-scale realized volatility (Zhang et al.) - separates noise from signal
#[inline]
pub fn two_scale_realized_volatility(returns: &[f64], subsampling_factor: usize) -> f64 {
    let n = returns.len();
    if n < subsampling_factor * 2 {
        return 0.0;
    }
    
    // Fast scale (all returns)
    let mut fast_rv = 0.0;
    for &r in returns {
        fast_rv += r * r;
    }
    
    // Slow scale (subsampled)
    let k = subsampling_factor;
    let mut slow_rv = 0.0;
    let mut count = 0;
    
    for i in (0..n).step_by(k) {
        if i + k < n {
            let subsampled_return: f64 = returns[i..i + k].iter().sum();
            slow_rv += subsampled_return * subsampled_return;
            count += 1;
        }
    }
    
    if count > 0 {
        slow_rv /= k as f64;
    }
    
    // Bias correction
    let n_bar = (n as f64) / (k as f64);
    fast_rv - (n_bar / n as f64) * slow_rv
}

/// Jump detection using bi-power variation (Barndorff-Nielsen & Shephard)
/// Returns (continuous_component, jump_component)
#[inline]
pub fn bipower_variation(returns: &[f64], mu1: f64) -> (f64, f64) {
    let n = returns.len();
    if n < 3 {
        return (0.0, 0.0);
    }
    
    // Realized variance
    let rv: f64 = returns.iter().map(|r| r * r).sum();
    
    // Bi-power variation: sum of |r_i| * |r_{i+1}|
    let mut bpv = 0.0;
    for i in 0..(n - 1) {
        bpv += returns[i].abs() * returns[i + 1].abs();
    }
    bpv *= (mu1 * mu1);
    
    let jump_component = (rv - bpv).max(0.0);
    let continuous_component = bpv;
    
    (continuous_component, jump_component)
}

/// Realized quarticity for inference on realized volatility
#[inline]
pub fn realized_quarticity(returns: &[f64]) -> f64 {
    let n = returns.len() as f64;
    returns.iter().map(|r| r.powi(4)).sum::<f64>() * n / 3.0
}

/// High-frequency covariance estimation with lead-lag adjustment
/// Hayashi-Yoshida estimator - handles asynchronous data
pub fn hayashi_yoshida_covariance(
    times_x: &[f64],
    values_x: &[f64],
    times_y: &[f64],
    values_y: &[f64],
) -> f64 {
    let mut cov = 0.0;
    
    let mut i = 0;
    let mut j = 0;
    
    while i < times_x.len() - 1 && j < times_y.len() - 1 {
        let t_i = times_x[i];
        let t_i_next = times_x[i + 1];
        let t_j = times_y[j];
        let t_j_next = times_y[j + 1];
        
        // Check if intervals overlap
        let overlap_start = t_i.max(t_j);
        let overlap_end = t_i_next.min(t_j_next);
        
        if overlap_start < overlap_end {
            let dx = values_x[i + 1] - values_x[i];
            let dy = values_y[j + 1] - values_y[j];
            cov += dx * dy;
        }
        
        // Advance the earlier-ending interval
        if t_i_next < t_j_next {
            i += 1;
        } else {
            j += 1;
        }
    }
    
    cov
}

/// Amihud illiquidity measure - price impact per dollar volume
#[inline]
pub fn amihud_illiquidity(returns: &[f64], volumes: &[f64]) -> f64 {
    assert_eq!(returns.len(), volumes.len());
    
    let mut illiquidity = 0.0;
    let mut count = 0;
    
    for i in 0..returns.len() {
        if volumes[i] > 1e-10 {
            illiquidity += returns[i].abs() / volumes[i];
            count += 1;
        }
    }
    
    if count > 0 {
        illiquidity / count as f64
    } else {
        0.0
    }
}

/// Roll's bid-ask spread estimator from serial covariance
#[inline]
pub fn roll_spread_estimator(returns: &[f64]) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    
    // Compute first-order autocovariance
    let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    
    let mut cov = 0.0;
    for i in 0..(returns.len() - 1) {
        cov += (returns[i] - mean) * (returns[i + 1] - mean);
    }
    cov /= (returns.len() - 1) as f64;
    
    // Roll's estimator: spread = 2 * sqrt(-cov)
    if cov < 0.0 {
        2.0 * (-cov).sqrt()
    } else {
        0.0 // No measurable spread
    }
}

/// Kyle's lambda (price impact coefficient)
/// Measures permanent price impact per unit volume
#[inline]
pub fn kyle_lambda(price_changes: &[f64], signed_volumes: &[f64]) -> f64 {
    assert_eq!(price_changes.len(), signed_volumes.len());
    
    let n = price_changes.len() as f64;
    
    let mean_dp: f64 = price_changes.iter().sum::<f64>() / n;
    let mean_vol: f64 = signed_volumes.iter().sum::<f64>() / n;
    
    let mut cov = 0.0;
    let mut var_vol = 0.0;
    
    for i in 0..price_changes.len() {
        let dp_dev = price_changes[i] - mean_dp;
        let vol_dev = signed_volumes[i] - mean_vol;
        cov += dp_dev * vol_dev;
        var_vol += vol_dev * vol_dev;
    }
    
    if var_vol > 1e-10 {
        cov / var_vol
    } else {
        0.0
    }
}

/// Hurst exponent via rescaled range (R/S) analysis
/// H > 0.5: trending, H < 0.5: mean-reverting, H = 0.5: random walk
pub fn hurst_exponent(series: &[f64], min_window: usize, max_window: usize) -> f64 {
    let n = series.len();
    if n < max_window {
        return 0.5;
    }
    
    let mut rs_values = Vec::new();
    let mut window_sizes = Vec::new();
    
    let mut window = min_window;
    while window <= max_window && window < n / 2 {
        let mut rs_sum = 0.0;
        let mut count = 0;
        
        for start in 0..=(n - window) {
            let chunk = &series[start..start + window];
            
            // Compute mean
            let mean: f64 = chunk.iter().sum::<f64>() / window as f64;
            
            // Compute cumulative deviations
            let mut cum_dev = vec![0.0; window];
            let mut sum = 0.0;
            for i in 0..window {
                sum += chunk[i] - mean;
                cum_dev[i] = sum;
            }
            
            // Range
            let range = cum_dev.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - cum_dev.iter().copied().fold(f64::INFINITY, f64::min);
            
            // Standard deviation
            let variance: f64 = chunk.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window as f64;
            let std_dev = variance.sqrt();
            
            if std_dev > 1e-10 {
                rs_sum += range / std_dev;
                count += 1;
            }
        }
        
        if count > 0 {
            rs_values.push((rs_sum / count as f64).ln());
            window_sizes.push((window as f64).ln());
        }
        
        window = (window as f64 * 1.5) as usize;
    }
    
    // Linear regression: log(R/S) = H * log(n) + c
    if rs_values.len() < 2 {
        return 0.5;
    }
    
    let n_points = rs_values.len() as f64;
    let mean_x: f64 = window_sizes.iter().sum::<f64>() / n_points;
    let mean_y: f64 = rs_values.iter().sum::<f64>() / n_points;
    
    let mut cov = 0.0;
    let mut var = 0.0;
    
    for i in 0..rs_values.len() {
        let dx = window_sizes[i] - mean_x;
        let dy = rs_values[i] - mean_y;
        cov += dx * dy;
        var += dx * dx;
    }
    
    if var > 1e-10 {
        (cov / var).max(0.0).min(1.0)
    } else {
        0.5
    }
}

/// Fractional differencing for making series stationary while preserving memory
/// Used in machine learning on time series
pub fn fractional_difference(series: &[f64], d: f64, threshold: f64) -> Vec<f64> {
    let n = series.len();
    
    // Compute weights
    let mut weights = vec![1.0];
    for k in 1..n {
        let weight = -weights[k - 1] * (d - k as f64 + 1.0) / k as f64;
        if weight.abs() < threshold {
            break;
        }
        weights.push(weight);
    }
    
    // Apply convolution
    let mut result = Vec::with_capacity(n);
    
    for i in 0..n {
        let mut sum = 0.0;
        for (j, &w) in weights.iter().enumerate() {
            if i >= j {
                sum += w * series[i - j];
            }
        }
        result.push(sum);
    }
    
    result
}

/// Triple barrier labeling for meta-labeling in ML
/// Returns: (outcome, time_to_exit) where outcome is 1 (up), -1 (down), or 0 (timeout)
pub fn triple_barrier_label(
    returns: &[f64],
    upper_barrier: f64,
    lower_barrier: f64,
    max_holding: usize,
) -> (i8, usize) {
    let mut cumulative = 0.0;
    
    for (i, &ret) in returns.iter().enumerate().take(max_holding) {
        cumulative += ret;
        
        if cumulative >= upper_barrier {
            return (1, i + 1);
        }
        if cumulative <= -lower_barrier {
            return (-1, i + 1);
        }
    }
    
    (0, max_holding) // Timeout
}

/// Entropy-based measures for complexity

/// Sample entropy - measures time series complexity
pub fn sample_entropy(series: &[f64], m: usize, r: f64) -> f64 {
    let n = series.len();
    if n < m + 1 {
        return 0.0;
    }
    
    let mut count_m = 0;
    let mut count_m1 = 0;
    
    // Count template matches
    for i in 0..(n - m) {
        for j in (i + 1)..(n - m) {
            // Check m-length templates
            let mut match_m = true;
            for k in 0..m {
                if (series[i + k] - series[j + k]).abs() > r {
                    match_m = false;
                    break;
                }
            }
            
            if match_m {
                count_m += 1;
                
                // Check m+1 length
                if (series[i + m] - series[j + m]).abs() <= r {
                    count_m1 += 1;
                }
            }
        }
    }
    
    if count_m > 0 && count_m1 > 0 {
        -(count_m1 as f64 / count_m as f64).ln()
    } else {
        0.0
    }
}

/// Approximate entropy
#[inline]
pub fn approximate_entropy(series: &[f64], m: usize, r: f64) -> f64 {
    let n = series.len();
    
    let phi = |m_val: usize| -> f64 {
        let mut patterns = vec![0; n - m_val + 1];
        
        for i in 0..=(n - m_val) {
            let mut count = 0;
            for j in 0..=(n - m_val) {
                let mut matches = true;
                for k in 0..m_val {
                    if (series[i + k] - series[j + k]).abs() > r {
                        matches = false;
                        break;
                    }
                }
                if matches {
                    count += 1;
                }
            }
            patterns[i] = count;
        }
        
        let sum: f64 = patterns.iter()
            .filter(|&&c| c > 0)
            .map(|&c| (c as f64 / (n - m_val + 1) as f64).ln())
            .sum();
        
        sum / (n - m_val + 1) as f64
    };
    
    phi(m) - phi(m + 1)
}

/// Recurrence quantification analysis (RQA) metrics
/// Measures deterministic structure in time series
pub struct RecurrenceMetrics {
    pub recurrence_rate: f64,
    pub determinism: f64,
    pub average_diagonal_line: f64,
    pub entropy: f64,
}

pub fn recurrence_quantification(series: &[f64], threshold: f64, min_line_length: usize) -> RecurrenceMetrics {
    let n = series.len();
    
    // Build recurrence matrix
    let mut recurrence_matrix = vec![vec![false; n]; n];
    let mut total_points = 0;
    
    for i in 0..n {
        for j in 0..n {
            if (series[i] - series[j]).abs() <= threshold {
                recurrence_matrix[i][j] = true;
                total_points += 1;
            }
        }
    }
    
    let recurrence_rate = total_points as f64 / (n * n) as f64;
    
    // Find diagonal lines
    let mut diagonal_lengths = Vec::new();
    
    for offset in -(n as i32 - 1)..(n as i32) {
        let mut length = 0;
        
        let start_i = if offset >= 0 { 0 } else { -offset as usize };
        let start_j = if offset >= 0 { offset as usize } else { 0 };
        
        for k in 0..n {
            let i = start_i + k;
            let j = start_j + k;
            
            if i >= n || j >= n {
                break;
            }
            
            if recurrence_matrix[i][j] {
                length += 1;
            } else {
                if length >= min_line_length {
                    diagonal_lengths.push(length);
                }
                length = 0;
            }
        }
        
        if length >= min_line_length {
            diagonal_lengths.push(length);
        }
    }
    
    let determinism = if !diagonal_lengths.is_empty() {
        let total_diagonal: usize = diagonal_lengths.iter().sum();
        total_diagonal as f64 / total_points as f64
    } else {
        0.0
    };
    
    let average_diagonal_line = if !diagonal_lengths.is_empty() {
        diagonal_lengths.iter().sum::<usize>() as f64 / diagonal_lengths.len() as f64
    } else {
        0.0
    };
    
    // Entropy of line length distribution
    let mut entropy = 0.0;
    if !diagonal_lengths.is_empty() {
        let total: usize = diagonal_lengths.iter().sum();
        let mut length_counts = std::collections::HashMap::new();
        
        for &len in &diagonal_lengths {
            *length_counts.entry(len).or_insert(0) += 1;
        }
        
        for count in length_counts.values() {
            let p = *count as f64 / diagonal_lengths.len() as f64;
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }
    }
    
    RecurrenceMetrics {
        recurrence_rate,
        determinism,
        average_diagonal_line,
        entropy,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_realized_kernel() {
        let returns = vec![0.001, -0.002, 0.003, -0.001, 0.002];
        let rk = realized_kernel(&returns, 2);
        assert!(rk > 0.0);
    }
    
    #[test]
    fn test_hurst_exponent() {
        // Trending series
        let trending: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let h = hurst_exponent(&trending, 10, 30);
        assert!(h > 0.5); // Should detect trend
    }
    
    #[test]
    fn test_sample_entropy() {
        let series = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        let se = sample_entropy(&series, 2, 0.5);
        assert!(se > 0.0);
    }
}