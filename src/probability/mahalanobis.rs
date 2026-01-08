/// Stack-allocated Mahalanobis distance detector
/// 
/// Dimension N must be known at compile time for maximum performance.
#[repr(align(64))]
pub struct Mahalanobis<const N: usize> {
    mean: [f64; N],
    inv_cov: [[f64; N]; N],  // Flattened NxN matrix
    threshold_99: f64,       // Precomputed chi-squared(N) 99th percentile
}

impl<const N: usize> Mahalanobis<N> {
    /// Train from normal data
    pub fn train(data: &[[f64; N]]) -> Result<Self, &'static str> {
        if data.len() < N + 1 {
            return Err("Insufficient training data");
        }
        
        let n = data.len() as f64;
        
        // Compute mean
        let mut mean = [0.0; N];
        for row in data {
            for i in 0..N {
                mean[i] += row[i];
            }
        }
        for i in 0..N {
            mean[i] /= n;
        }
        
        // Compute covariance
        let mut cov = [[0.0; N]; N];
        for row in data {
            for i in 0..N {
                for j in 0..N {
                    let di = row[i] - mean[i];
                    let dj = row[j] - mean[j];
                    cov[i][j] += di * dj;
                }
            }
        }
        for i in 0..N {
            for j in 0..N {
                cov[i][j] /= n - 1.0;
            }
        }
        
        // Invert covariance using centralized matrix inversion
        let inv_cov = crate::matrix::kernels::invert_matrix(&cov)?;
        
        // Precompute chi-squared threshold
        let threshold_99 = chi_squared_quantile(N as f64, 0.99);
        
        Ok(Self {
            mean,
            inv_cov,
            threshold_99,
        })
    }
    
    /// Compute Mahalanobis distance squared
    #[inline(always)]
    pub fn distance_sq(&self, x: &[f64; N]) -> f64 {
        use crate::matrix::ops::quadratic_form;

        // diff = x - mean
        let mut diff = [0.0; N];
        for i in 0..N {
            diff[i] = x[i] - self.mean[i];
        }

        // result = diff^T * inv_cov * diff (this is a quadratic form!)
        quadratic_form(&self.inv_cov, &diff)
    }
    
    /// Fast anomaly check (returns true if > 99th percentile)
    #[inline(always)]
    pub fn is_anomaly(&self, x: &[f64; N]) -> bool {
        self.distance_sq(x) > self.threshold_99
    }
    
    /// Anomaly score (0 to 1, approximation)
    #[inline]
    pub fn score(&self, x: &[f64; N]) -> f64 {
        let d_sq = self.distance_sq(x);
        // Fast approximation: 1 - exp(-d_sq/2)
        1.0 - (-d_sq * 0.5).exp()
    }
}

// Matrix operations now use centralized functions from matrix:: module
// - invert_matrix: matrix::kernels::invert_matrix
// - quadratic_form: matrix::ops::quadratic_form
// - chi_squared_quantile: probability::utils::chi_squared_quantile
use crate::probability::utils::chi_squared_quantile;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mahalanobis() {
        let data: Vec<[f64; 2]> = vec![
            [0.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.6],
        ];
        
        let detector = Mahalanobis::train(&data).unwrap();
        
        let normal = [0.5, 0.5];
        assert!(!detector.is_anomaly(&normal));
        
        let extreme = [10.0, 10.0];
        assert!(detector.is_anomaly(&extreme));
    }
}