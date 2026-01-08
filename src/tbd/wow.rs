/// Copula-based dependence modeling

/// Empirical copula transformation - convert marginals to uniform [0,1]
#[inline]
pub fn empirical_copula_transform(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&i, &j| data[i].partial_cmp(&data[j]).unwrap());
    
    let mut ranks = vec![0.0; n];
    for (rank, &idx) in sorted_indices.iter().enumerate() {
        ranks[idx] = (rank + 1) as f64 / (n + 1) as f64;
    }
    
    ranks
}

/// Gaussian copula correlation from empirical data
pub fn gaussian_copula_correlation(data_x: &[f64], data_y: &[f64]) -> f64 {
    assert_eq!(data_x.len(), data_y.len());
    
    // Transform to uniform marginals
    let u_x = empirical_copula_transform(data_x);
    let u_y = empirical_copula_transform(data_y);
    
    // Transform to normal via inverse CDF (approximate)
    let probit = |u: f64| -> f64 {
        // Quick approximation of normal inverse CDF
        let x = u.clamp(0.0001, 0.9999);
        if x < 0.5 {
            let t = (-2.0 * x.ln()).sqrt();
            -(t - (2.30753 + 0.27061 * t) / (1.0 + 0.99229 * t + 0.04481 * t * t))
        } else {
            let t = (-2.0 * (1.0 - x).ln()).sqrt();
            t - (2.30753 + 0.27061 * t) / (1.0 + 0.99229 * t + 0.04481 * t * t)
        }
    };
    
    let z_x: Vec<f64> = u_x.iter().map(|&u| probit(u)).collect();
    let z_y: Vec<f64> = u_y.iter().map(|&u| probit(u)).collect();
    
    // Pearson correlation of transformed data
    let n = z_x.len() as f64;
    let mean_x: f64 = z_x.iter().sum::<f64>() / n;
    let mean_y: f64 = z_y.iter().sum::<f64>() / n;
    
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    
    for i in 0..z_x.len() {
        let dx = z_x[i] - mean_x;
        let dy = z_y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    
    let denom = (var_x * var_y).sqrt();
    if denom > 1e-10 {
        cov / denom
    } else {
        0.0
    }
}

/// Tail dependence coefficient (upper tail)
#[inline]
pub fn upper_tail_dependence(data_x: &[f64], data_y: &[f64], threshold: f64) -> f64 {
    assert_eq!(data_x.len(), data_y.len());
    
    let u_x = empirical_copula_transform(data_x);
    let u_y = empirical_copula_transform(data_y);
    
    let mut both_exceed = 0;
    let mut either_exceeds = 0;
    
    for i in 0..u_x.len() {
        if u_x[i] > threshold && u_y[i] > threshold {
            both_exceed += 1;
            either_exceeds += 1;
        } else if u_x[i] > threshold || u_y[i] > threshold {
            either_exceeds += 1;
        }
    }
    
    if either_exceeds > 0 {
        both_exceed as f64 / either_exceeds as f64
    } else {
        0.0
    }
}

/// Distance correlation - captures nonlinear dependence
pub fn distance_correlation(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    
    // Compute distance matrices
    let mut dist_x = vec![vec![0.0; n]; n];
    let mut dist_y = vec![vec![0.0; n]; n];
    
    for i in 0..n {
        for j in 0..n {
            dist_x[i][j] = (x[i] - x[j]).abs();
            dist_y[i][j] = (y[i] - y[j]).abs();
        }
    }
    
    // Double-center the distance matrices
    let center = |dist: &mut Vec<Vec<f64>>| {
        let n = dist.len();
        let mut row_means = vec![0.0; n];
        let mut col_means = vec![0.0; n];
        let mut grand_mean = 0.0;
        
        for i in 0..n {
            for j in 0..n {
                row_means[i] += dist[i][j];
                col_means[j] += dist[i][j];
                grand_mean += dist[i][j];
            }
        }
        
        for i in 0..n {
            row_means[i] /= n as f64;
            col_means[i] /= n as f64;
        }
        grand_mean /= (n * n) as f64;
        
        for i in 0..n {
            for j in 0..n {
                dist[i][j] = dist[i][j] - row_means[i] - col_means[j] + grand_mean;
            }
        }
    };
    
    center(&mut dist_x);
    center(&mut dist_y);
    
    // Compute distance covariance and variances
    let mut dcov = 0.0;
    let mut dvar_x = 0.0;
    let mut dvar_y = 0.0;
    
    for i in 0..n {
        for j in 0..n {
            dcov += dist_x[i][j] * dist_y[i][j];
            dvar_x += dist_x[i][j] * dist_x[i][j];
            dvar_y += dist_y[i][j] * dist_y[i][j];
        }
    }
    
    dcov /= (n * n) as f64;
    dvar_x /= (n * n) as f64;
    dvar_y /= (n * n) as f64;
    
    let denom = (dvar_x * dvar_y).sqrt();
    if denom > 1e-10 {
        dcov / denom
    } else {
        0.0
    }
}

/// Maximal information coefficient (MIC) - detects any functional relationship
/// Simplified version for speed
pub fn mutual_information_binned(x: &[f64], y: &[f64], n_bins: usize) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    
    // Find ranges
    let min_x = x.iter().copied().fold(f64::INFINITY, f64::min);
    let max_x = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min_y = y.iter().copied().fold(f64::INFINITY, f64::min);
    let max_y = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    
    let bin_width_x = (max_x - min_x) / n_bins as f64;
    let bin_width_y = (max_y - min_y) / n_bins as f64;
    
    if bin_width_x < 1e-10 || bin_width_y < 1e-10 {
        return 0.0;
    }
    
    // Build joint histogram
    let mut joint = vec![vec![0; n_bins]; n_bins];
    let mut marginal_x = vec![0; n_bins];
    let mut marginal_y = vec![0; n_bins];
    
    for i in 0..n {
        let bin_x = (((x[i] - min_x) / bin_width_x) as usize).min(n_bins - 1);
        let bin_y = (((y[i] - min_y) / bin_width_y) as usize).min(n_bins - 1);
        
        joint[bin_x][bin_y] += 1;
        marginal_x[bin_x] += 1;
        marginal_y[bin_y] += 1;
    }
    
    // Compute mutual information
    let mut mi = 0.0;
    let n_f64 = n as f64;
    
    for i in 0..n_bins {
        for j in 0..n_bins {
            if joint[i][j] > 0 {
                let p_xy = joint[i][j] as f64 / n_f64;
                let p_x = marginal_x[i] as f64 / n_f64;
                let p_y = marginal_y[j] as f64 / n_f64;
                
                mi += p_xy * (p_xy / (p_x * p_y)).ln();
            }
        }
    }
    
    mi
}

/// Generalized method of moments (GMM) estimation helpers

/// Compute optimal weighting matrix for GMM (inverse of moment covariance)
pub fn gmm_optimal_weight<const M: usize>(
    moment_conditions: &[[f64; M]],
) -> [[f64; M]; M] {
    let n = moment_conditions.len() as f64;
    
    // Compute sample covariance of moments
    let mut mean = [0.0; M];
    for mc in moment_conditions {
        for i in 0..M {
            mean[i] += mc[i];
        }
    }
    for i in 0..M {
        mean[i] /= n;
    }
    
    let mut cov = [[0.0; M]; M];
    for mc in moment_conditions {
        for i in 0..M {
            for j in 0..M {
                cov[i][j] += (mc[i] - mean[i]) * (mc[j] - mean[j]);
            }
        }
    }
    for i in 0..M {
        for j in 0..M {
            cov[i][j] /= n - 1.0;
        }
    }
    
    // Return inverse (would use proper matrix inversion in practice)
    cov // Placeholder - should invert
}

/// Newey-West HAC (heteroskedasticity and autocorrelation consistent) covariance
pub fn newey_west_covariance<const M: usize>(
    residuals: &[[f64; M]],
    max_lag: usize,
) -> [[f64; M]; M] {
    let n = residuals.len() as f64;
    
    // Bartlett kernel weight
    let bartlett = |lag: usize| -> f64 {
        1.0 - (lag as f64 / (max_lag + 1) as f64)
    };
    
    let mut cov = [[0.0; M]; M];
    
    // Lag 0 (variance)
    for r in residuals {
        for i in 0..M {
            for j in 0..M {
                cov[i][j] += r[i] * r[j];
            }
        }
    }
    
    // Autocovariances with kernel weighting
    for lag in 1..=max_lag {
        let weight = bartlett(lag);
        
        for t in lag..residuals.len() {
            for i in 0..M {
                for j in 0..M {
                    let contrib = residuals[t][i] * residuals[t - lag][j];
                    cov[i][j] += weight * contrib;
                    if i != j {
                        cov[j][i] += weight * residuals[t - lag][i] * residuals[t][j];
                    }
                }
            }
        }
    }
    
    // Scale by sample size
    for i in 0..M {
        for j in 0..M {
            cov[i][j] /= n;
        }
    }
    
    cov
}

/// Bootstrap for inference

/// Stationary bootstrap - preserves time dependence structure
pub fn stationary_bootstrap_sample(data: &[f64], avg_block_length: f64) -> Vec<f64> {
    let n = data.len();
    let p = 1.0 / avg_block_length; // Geometric distribution parameter
    
    let mut result = Vec::with_capacity(n);
    let mut rng_state = 12345u64; // Simple LCG for determinism
    
    let mut pos = (rng_state % n as u64) as usize;
    
    while result.len() < n {
        result.push(data[pos]);
        
        // Decide whether to continue block or jump
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let u = (rng_state as f64 / u64::MAX as f64);
        
        if u < p {
            // Start new block
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            pos = (rng_state % n as u64) as usize;
        } else {
            // Continue block
            pos = (pos + 1) % n;
        }
    }
    
    result
}

/// Circular block bootstrap
pub fn circular_block_bootstrap(data: &[f64], block_length: usize, n_samples: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = Vec::with_capacity(n_samples);
    let mut rng_state = 54321u64;
    
    while result.len() < n_samples {
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let start = (rng_state % n as u64) as usize;
        
        for i in 0..block_length.min(n_samples - result.len()) {
            result.push(data[(start + i) % n]);
        }
    }
    
    result
}

/// High-dimensional statistics

/// Sparse PCA (power iteration with soft thresholding)
pub fn sparse_pca_loadings<const N: usize>(
    covariance: &[[f64; N]; N],
    sparsity_param: f64,
    max_iter: usize,
) -> [f64; N] {
    // Initialize random vector
    let mut v = [1.0 / (N as f64).sqrt(); N];
    
    for _ in 0..max_iter {
        // Matrix-vector multiply
        let mut new_v = [0.0; N];
        for i in 0..N {
            for j in 0..N {
                new_v[i] += covariance[i][j] * v[j];
            }
        }
        
        // Soft thresholding
        for i in 0..N {
            if new_v[i].abs() < sparsity_param {
                new_v[i] = 0.0;
            } else {
                new_v[i] -= sparsity_param * new_v[i].signum();
            }
        }
        
        // Normalize
        let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for i in 0..N {
                new_v[i] /= norm;
            }
        }
        
        v = new_v;
    }
    
    v
}

/// LASSO coordinate descent for sparse regression
pub fn lasso_coordinate_descent<const N: usize>(
    x: &[[f64; N]], // Design matrix (each row is observation)
    y: &[f64],      // Response vector
    lambda: f64,    // Regularization parameter
    max_iter: usize,
) -> [f64; N] {
    let n = x.len();
    let mut beta = [0.0; N];
    
    // Precompute column norms
    let mut col_norms = [0.0; N];
    for j in 0..N {
        for i in 0..n {
            col_norms[j] += x[i][j] * x[i][j];
        }
    }
    
    for _ in 0..max_iter {
        for j in 0..N {
            if col_norms[j] < 1e-10 {
                continue;
            }
            
            // Compute partial residual
            let mut partial_resid = 0.0;
            for i in 0..n {
                let mut pred = 0.0;
                for k in 0..N {
                    if k != j {
                        pred += x[i][k] * beta[k];
                    }
                }
                partial_resid += x[i][j] * (y[i] - pred);
            }
            
            // Soft thresholding update
            let z = partial_resid / n as f64;
            beta[j] = if z.abs() <= lambda {
                0.0
            } else {
                (z - lambda * z.signum()) / (col_norms[j] / n as f64)
            };
        }
    }
    
    beta
}

/// Elastic net (LASSO + Ridge)
pub fn elastic_net_coordinate_descent<const N: usize>(
    x: &[[f64; N]],
    y: &[f64],
    lambda1: f64, // LASSO penalty
    lambda2: f64, // Ridge penalty
    max_iter: usize,
) -> [f64; N] {
    let n = x.len();
    let mut beta = [0.0; N];
    
    let mut col_norms = [0.0; N];
    for j in 0..N {
        for i in 0..n {
            col_norms[j] += x[i][j] * x[i][j];
        }
    }
    
    for _ in 0..max_iter {
        for j in 0..N {
            if col_norms[j] < 1e-10 {
                continue;
            }
            
            let mut partial_resid = 0.0;
            for i in 0..n {
                let mut pred = 0.0;
                for k in 0..N {
                    if k != j {
                        pred += x[i][k] * beta[k];
                    }
                }
                partial_resid += x[i][j] * (y[i] - pred);
            }
            
            let z = partial_resid / n as f64;
            let denom = col_norms[j] / n as f64 + lambda2;
            
            beta[j] = if z.abs() <= lambda1 {
                0.0
            } else {
                (z - lambda1 * z.signum()) / denom
            };
        }
    }
    
    beta
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_copula_transform() {
        let data = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let u = empirical_copula_transform(&data);
        
        // Should be uniform on (0, 1)
        for &val in &u {
            assert!(val > 0.0 && val < 1.0);
        }
    }
    
    #[test]
    fn test_distance_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect linear
        
        let dc = distance_correlation(&x, &y);
        assert!(dc > 0.9); // Should detect strong dependence
    }
    
    #[test]
    fn test_lasso() {
        let x = vec![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
        ];
        let y = vec![1.0, 2.0, 3.0];
        
        let beta = lasso_coordinate_descent(&x, &y, 0.1, 100);
        
        // Should find some solution
        assert!(beta.iter().any(|&b| b.abs() > 1e-6));
    }
}