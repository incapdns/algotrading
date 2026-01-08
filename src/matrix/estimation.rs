/// Simple rolling covariance matrix estimator
/// Accumulates observations and computes sample covariance
pub struct RollingCovariance<const N: usize> {
    observations: Vec<[f64; N]>,
    mean: [f64; N],
    count: usize,
}

impl<const N: usize> RollingCovariance<N> {
    /// Create new rolling covariance estimator
    #[inline]
    pub fn new() -> Self {
        Self {
            observations: Vec::new(),
            mean: [0.0; N],
            count: 0,
        }
    }

    /// Update with new observation
    #[inline]
    pub fn update(&mut self, observation: &[f64; N]) {
        self.observations.push(*observation);
        self.count += 1;

        // Update running mean
        for i in 0..N {
            self.mean[i] += (observation[i] - self.mean[i]) / self.count as f64;
        }
    }

    /// Compute current covariance matrix
    pub fn covariance(&self) -> [[f64; N]; N] {
        if self.count < 2 {
            return [[0.0; N]; N];
        }

        let mut cov = [[0.0; N]; N];

        for obs in &self.observations {
            for i in 0..N {
                for j in 0..N {
                    let di = obs[i] - self.mean[i];
                    let dj = obs[j] - self.mean[j];
                    cov[i][j] += di * dj;
                }
            }
        }

        // Normalize by (n-1) for unbiased estimator
        let norm = (self.count - 1) as f64;
        for i in 0..N {
            for j in 0..N {
                cov[i][j] /= norm;
            }
        }

        cov
    }

    /// Get correlation matrix
    pub fn correlation(&self) -> [[f64; N]; N] {
        let cov = self.covariance();
        let mut corr = [[0.0; N]; N];

        for i in 0..N {
            for j in 0..N {
                let std_i = cov[i][i].sqrt();
                let std_j = cov[j][j].sqrt();

                if std_i > 1e-10 && std_j > 1e-10 {
                    corr[i][j] = cov[i][j] / (std_i * std_j);
                } else if i == j {
                    corr[i][j] = 1.0;
                }
            }
        }

        corr
    }

    /// Number of observations
    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }
}

impl<const N: usize> Default for RollingCovariance<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fast covariance matrix estimation with shrinkage
/// Ledoit-Wolf shrinkage for better conditioning with small samples
pub struct LedoitWolfEstimator<const N: usize> {
    sample_cov: [[f64; N]; N],
    shrinkage_intensity: f64,
}

impl<const N: usize> LedoitWolfEstimator<N> {
    /// Estimate covariance with automatic shrinkage
    pub fn estimate(data: &[[f64; N]]) -> Result<Self, &'static str> {
        if data.len() < 2 {
            return Err("Insufficient data");
        }
        
        let n_samples = data.len() as f64;
        
        // Compute sample mean
        let mut mean = [0.0; N];
        for row in data {
            for i in 0..N {
                mean[i] += row[i];
            }
        }
        for i in 0..N {
            mean[i] /= n_samples;
        }
        
        // Compute sample covariance
        let mut sample_cov = [[0.0; N]; N];
        for row in data {
            for i in 0..N {
                for j in 0..N {
                    let di = row[i] - mean[i];
                    let dj = row[j] - mean[j];
                    sample_cov[i][j] += di * dj;
                }
            }
        }
        for i in 0..N {
            for j in 0..N {
                sample_cov[i][j] /= n_samples - 1.0;
            }
        }
        
        // Compute shrinkage target (identity * average variance)
        // (computed inline in shrunk_covariance method)
        
        // Estimate optimal shrinkage intensity
        let shrinkage = Self::compute_shrinkage_intensity(&sample_cov, data, &mean);
        
        Ok(Self {
            sample_cov,
            shrinkage_intensity: shrinkage,
        })
    }
    
    /// Get shrunk covariance matrix
    pub fn shrunk_covariance(&self) -> [[f64; N]; N] {
        let mut result = [[0.0; N]; N];
        
        // Target: identity * average variance
        let mut avg_var = 0.0;
        for i in 0..N {
            avg_var += self.sample_cov[i][i];
        }
        avg_var /= N as f64;
        
        let delta = self.shrinkage_intensity;
        
        for i in 0..N {
            for j in 0..N {
                if i == j {
                    result[i][j] = delta * avg_var + (1.0 - delta) * self.sample_cov[i][j];
                } else {
                    result[i][j] = (1.0 - delta) * self.sample_cov[i][j];
                }
            }
        }
        
        result
    }
    
    fn compute_shrinkage_intensity(
        sample_cov: &[[f64; N]; N],
        data: &[[f64; N]],
        mean: &[f64; N],
    ) -> f64 {
        let n = data.len() as f64;
        
        // Compute average variance
        let mut avg_var = 0.0;
        for i in 0..N {
            avg_var += sample_cov[i][i];
        }
        avg_var /= N as f64;
        
        // Estimate π (asymptotic variance)
        let mut pi = 0.0;
        for row in data {
            for i in 0..N {
                for j in 0..N {
                    let di = row[i] - mean[i];
                    let dj = row[j] - mean[j];
                    let sample = di * dj;
                    let diff = sample - sample_cov[i][j];
                    pi += diff * diff;
                }
            }
        }
        pi /= n;
        
        // Estimate ρ (distance to target)
        let mut rho = 0.0;
        for i in 0..N {
            for j in 0..N {
                if i == j {
                    let diff = sample_cov[i][j] - avg_var;
                    rho += diff * diff;
                } else {
                    rho += sample_cov[i][j] * sample_cov[i][j];
                }
            }
        }
        
        // Optimal shrinkage
        let shrinkage = (pi / rho).min(1.0).max(0.0);
        shrinkage
    }
}

/// Exponentially weighted covariance matrix
/// Online estimation without storing full history
pub struct EWMACovarianceMatrix<const N: usize> {
    cov: [[f64; N]; N],
    mean: [f64; N],
    alpha: f64,
    initialized: bool,
}

impl<const N: usize> EWMACovarianceMatrix<N> {
    #[inline]
    pub fn new(alpha: f64) -> Self {
        Self {
            cov: [[0.0; N]; N],
            mean: [0.0; N],
            alpha,
            initialized: false,
        }
    }
    
    #[inline]
    pub fn from_halflife(halflife: f64) -> Self {
        let alpha = 1.0 - (-2.0_f64.ln() / halflife).exp();
        Self::new(alpha)
    }
    
    /// Update with new observation vector
    #[inline]
    pub fn update(&mut self, observation: &[f64; N]) {
        if !self.initialized {
            self.mean = *observation;
            self.initialized = true;
            return;
        }
        
        // Update mean
        for i in 0..N {
            self.mean[i] = self.alpha * observation[i] + (1.0 - self.alpha) * self.mean[i];
        }
        
        // Update covariance
        for i in 0..N {
            for j in 0..N {
                let di = observation[i] - self.mean[i];
                let dj = observation[j] - self.mean[j];
                self.cov[i][j] = (1.0 - self.alpha) * self.cov[i][j] + self.alpha * di * dj;
            }
        }
    }
    
    /// Get current covariance matrix
    #[inline]
    pub fn covariance(&self) -> [[f64; N]; N] {
        self.cov
    }
    
    /// Get correlation matrix
    pub fn correlation(&self) -> [[f64; N]; N] {
        let mut corr = [[0.0; N]; N];
        
        for i in 0..N {
            for j in 0..N {
                let std_i = self.cov[i][i].sqrt();
                let std_j = self.cov[j][j].sqrt();
                
                if std_i > 1e-10 && std_j > 1e-10 {
                    corr[i][j] = self.cov[i][j] / (std_i * std_j);
                } else if i == j {
                    corr[i][j] = 1.0;
                }
            }
        }
        
        corr
    }
}

/// Woodbury matrix identity for fast inverse updates
/// (A + UCV)^(-1) = A^(-1) - A^(-1)U(C^(-1) + VA^(-1)U)^(-1)VA^(-1)
/// Useful when updating covariance matrices with new factors
pub struct WoodburyUpdate<const N: usize> {
    a_inv: [[f64; N]; N],
}

impl<const N: usize> WoodburyUpdate<N> {
    pub fn new(a_inv: [[f64; N]; N]) -> Self {
        Self { a_inv }
    }
    
    /// Update inverse when adding rank-1 matrix uv^T
    #[inline]
    pub fn rank_one_update(&mut self, u: &[f64; N], v: &[f64; N]) {
        // Sherman-Morrison formula: (A + uv^T)^(-1) = A^(-1) - (A^(-1)uv^TA^(-1))/(1 + v^TA^(-1)u)
        
        let mut a_inv_u = [0.0; N];
        for i in 0..N {
            for j in 0..N {
                a_inv_u[i] += self.a_inv[i][j] * u[j];
            }
        }
        
        let mut v_t_a_inv = [0.0; N];
        for i in 0..N {
            for j in 0..N {
                v_t_a_inv[i] += v[j] * self.a_inv[j][i];
            }
        }
        
        let mut denom = 1.0;
        for i in 0..N {
            denom += v_t_a_inv[i] * u[i];
        }
        
        if denom.abs() < 1e-10 {
            return; // Singular update
        }
        
        // Update A^(-1)
        for i in 0..N {
            for j in 0..N {
                self.a_inv[i][j] -= (a_inv_u[i] * v_t_a_inv[j]) / denom;
            }
        }
    }
    
    #[inline]
    pub fn inverse(&self) -> &[[f64; N]; N] {
        &self.a_inv
    }
}

/// Compute matrix square root using Newton-Schulz iteration
///
/// Returns matrix B such that B * B = A (approximately)
///
/// # Arguments
///
/// * `matrix` - Symmetric positive definite matrix
/// * `max_iterations` - Maximum number of iterations (default: 20)
///
/// # Returns
///
/// Matrix square root, or error if convergence fails
///
/// # Algorithm
///
/// Uses Newton-Schulz iteration for the inverse square root X = A^{-1/2}:
/// X_{k+1} = 0.5 * X_k * (3*I - A * X_k^2)
///
/// Then computes B = A * X to get the square root.
///
/// # Performance
///
/// O(N^3) per iteration, typically converges in 5-10 iterations
pub fn matrix_sqrt_newton<const N: usize>(
    matrix: &[[f64; N]; N],
    max_iterations: usize,
) -> Result<[[f64; N]; N], &'static str> {
    use super::matrix_ops::{matmul, identity};

    const TOLERANCE: f64 = 1e-10;

    // Initialize X_0 = I / ||A||_F (Frobenius norm)
    let mut norm = 0.0;
    for i in 0..N {
        for j in 0..N {
            norm += matrix[i][j] * matrix[i][j];
        }
    }
    norm = norm.sqrt();

    if norm < 1e-14 {
        return Err("Matrix is too close to zero");
    }

    // X = I / norm
    let mut x = identity();
    for i in 0..N {
        x[i][i] /= norm;
    }

    // Newton-Schulz iteration: X_{k+1} = 0.5 * X_k * (3*I - A * X_k^2)
    for _ in 0..max_iterations {
        // Compute X^2
        let x_sq = matmul(&x, &x);

        // Compute A * X^2
        let ax_sq = matmul(matrix, &x_sq);

        // Compute 3*I - A*X^2
        let mut three_i_minus_ax_sq = identity();
        for i in 0..N {
            for j in 0..N {
                if i == j {
                    three_i_minus_ax_sq[i][j] = 3.0 - ax_sq[i][j];
                } else {
                    three_i_minus_ax_sq[i][j] = -ax_sq[i][j];
                }
            }
        }

        // X_{k+1} = 0.5 * X_k * (3*I - A*X^2)
        let x_new_temp = matmul(&x, &three_i_minus_ax_sq);
        let mut x_new = [[0.0; N]; N];
        for i in 0..N {
            for j in 0..N {
                x_new[i][j] = 0.5 * x_new_temp[i][j];
            }
        }

        // Check convergence
        let mut diff = 0.0;
        for i in 0..N {
            for j in 0..N {
                let d = x_new[i][j] - x[i][j];
                diff += d * d;
            }
        }

        x = x_new;

        if diff.sqrt() < TOLERANCE {
            // X is now A^{-1/2}, compute sqrt = A * X
            return Ok(matmul(matrix, &x));
        }
    }

    Err("Newton-Schulz iteration did not converge")
}

/// Batch matrix-vector operations (process multiple vectors at once)
/// Useful for portfolio optimization with multiple scenarios
#[inline]
pub fn batch_matvec<const N: usize, const K: usize>(
    matrix: &[[f64; N]; N],
    vectors: &[[f64; N]; K],
) -> [[f64; N]; K] {
    let mut results = [[0.0; N]; K];
    
    for k in 0..K {
        for i in 0..N {
            for j in 0..N {
                results[k][i] += matrix[i][j] * vectors[k][j];
            }
        }
    }
    
    results
}

/// Compute eigenvalue of largest magnitude (power method)
/// Fast for dominant eigenvalue without full decomposition
pub fn dominant_eigenvalue<const N: usize>(
    matrix: &[[f64; N]; N],
    max_iter: usize,
) -> (f64, [f64; N]) {
    const TOLERANCE: f64 = 1e-10;
    
    // Initial guess
    let mut v = [1.0 / (N as f64).sqrt(); N];
    let mut lambda = 0.0;
    
    for _ in 0..max_iter {
        // v_new = A * v
        let mut v_new = [0.0; N];
        for i in 0..N {
            for j in 0..N {
                v_new[i] += matrix[i][j] * v[j];
            }
        }
        
        // Compute eigenvalue estimate
        let lambda_new = {
            let mut num = 0.0;
            let mut denom = 0.0;
            for i in 0..N {
                num += v_new[i] * v[i];
                denom += v[i] * v[i];
            }
            num / denom
        };
        
        // Normalize
        let norm = {
            let mut sum = 0.0;
            for &val in &v_new {
                sum += val * val;
            }
            sum.sqrt()
        };
        
        for i in 0..N {
            v_new[i] /= norm;
        }
        
        // Check convergence
        if (lambda_new - lambda).abs() < TOLERANCE {
            return (lambda_new, v_new);
        }
        
        v = v_new;
        lambda = lambda_new;
    }
    
    (lambda, v)
}

/// Rayleigh quotient for quick eigenvalue approximation given eigenvector guess
#[inline]
pub fn rayleigh_quotient<const N: usize>(matrix: &[[f64; N]; N], vector: &[f64; N]) -> f64 {
    // λ ≈ (v^T A v) / (v^T v)
    
    let mut av = [0.0; N];
    for i in 0..N {
        for j in 0..N {
            av[i] += matrix[i][j] * vector[j];
        }
    }
    
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    
    for i in 0..N {
        numerator += vector[i] * av[i];
        denominator += vector[i] * vector[i];
    }
    
    numerator / denominator
}

/// Fast symmetric positive definite matrix inversion via Cholesky
/// Much faster than general Gaussian elimination for SPD matrices
#[inline]
pub fn invert_spd_cholesky<const N: usize>(
    matrix: &[[f64; N]; N],
) -> Result<[[f64; N]; N], &'static str> {
    use super::matrix_ops::Cholesky;
    
    // Decompose A = L L^T
    let chol = Cholesky::decompose(matrix)?;
    let l = chol.lower();
    
    // Solve L L^T X = I by:
    // 1. Solve LY = I for Y
    // 2. Solve L^T X = Y for X
    
    let mut inv = [[0.0; N]; N];
    
    for col in 0..N {
        // Forward substitution: L y = e_col
        let mut y = [0.0; N];
        for i in 0..N {
            let mut sum = if i == col { 1.0 } else { 0.0 };
            for j in 0..i {
                sum -= l[i][j] * y[j];
            }
            y[i] = sum / l[i][i];
        }
        
        // Back substitution: L^T x = y
        for i in (0..N).rev() {
            let mut sum = y[i];
            for j in (i + 1)..N {
                sum -= l[j][i] * inv[j][col];
            }
            inv[i][col] = sum / l[i][i];
        }
    }
    
    Ok(inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ledoit_wolf() {
        let data = vec![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
        ];
        
        let estimator = LedoitWolfEstimator::estimate(&data).unwrap();
        let shrunk = estimator.shrunk_covariance();
        
        // Shrunk covariance should be less extreme than sample covariance
        assert!(shrunk[0][0] > 0.0);
        assert!(shrunk[1][1] > 0.0);
    }
    
    #[test]
    fn test_ewma_covariance() {
        let mut cov = EWMACovarianceMatrix::<2>::new(0.1);
        
        for i in 0..10 {
            let obs = [i as f64, (i * 2) as f64];
            cov.update(&obs);
        }
        
        let c = cov.covariance();
        assert!(c[0][0] > 0.0);
        assert!(c[1][1] > 0.0);
    }
    
    #[test]
    fn test_dominant_eigenvalue() {
        let matrix = [
            [4.0, 1.0],
            [1.0, 3.0],
        ];
        
        let (lambda, _v) = dominant_eigenvalue(&matrix, 100);
        
        // Should converge to largest eigenvalue (~4.3)
        assert!(lambda > 4.0 && lambda < 5.0);
    }
}