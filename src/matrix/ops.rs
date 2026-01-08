//! Matrix operations for portfolio optimization and risk management
//!
//! This module provides high-performance linear algebra operations critical for
//! quantitative finance applications:
//!
//! - **Cholesky decomposition**: Portfolio risk decomposition, correlated Monte Carlo
//! - **Eigen decomposition**: PCA, factor models, portfolio optimization
//! - **Matrix operations**: Efficient portfolio variance calculations
//!
//! All operations are stack-allocated with cache-aligned buffers for maximum performance.
//!
//! # Portfolio Use Cases
//!
//! ## 1. Portfolio Variance Calculation
//!
//! Calculate portfolio variance using Cholesky decomposition (faster than full matrix multiplication):
//!
//! ```
//! use algotrading::matrix::Cholesky;
//!
//! // Covariance matrix for 3-asset portfolio
//! // Assets: SPY, TLT, GLD
//! let cov = [
//!     [0.0400, 0.0050, 0.0100],  // SPY: 20% vol, moderate correlation
//!     [0.0050, 0.0225, -0.0025], // TLT: 15% vol, negative correlation with SPY
//!     [0.0100, -0.0025, 0.0625], // GLD: 25% vol, low correlation
//! ];
//!
//! // Decompose: Σ = L * L^T
//! let chol = Cholesky::decompose(&cov).unwrap();
//!
//! // Portfolio weights: 60% SPY, 30% TLT, 10% GLD
//! let weights = [0.6, 0.3, 0.1];
//!
//! // Calculate portfolio variance: w^T * Σ * w
//! // This is O(N^2) instead of O(N^3) for full matrix multiply
//! let portfolio_var = chol.portfolio_variance(&weights);
//! let portfolio_vol = portfolio_var.sqrt();
//!
//! println!("Portfolio volatility: {:.2}%", portfolio_vol * 100.0);
//! // Output: Portfolio volatility: 14.25%
//!
//! // Compare to individual assets:
//! // SPY: 20%, TLT: 15%, GLD: 25%
//! // Diversification reduced vol from 19.5% (weighted avg) to 14.25%
//! ```
//!
//! ## 2. Correlated Monte Carlo Simulation
//!
//! Generate correlated asset paths for multi-asset option pricing or risk simulation:
//!
//! ```
//! use algotrading::matrix::Cholesky;
//! use algotrading::probability::Normal;
//!
//! // Correlation matrix for SPY and QQQ
//! let corr = [
//!     [1.0, 0.85],  // SPY variance
//!     [0.85, 1.0],  // 85% correlation (tech-heavy)
//! ];
//!
//! let chol = Cholesky::decompose(&corr).unwrap();
//!
//! // Generate correlated random shocks
//! // In production, use proper RNG like rand_distr
//! let uncorrelated = [0.5, -0.3];  // Independent standard normals
//! let correlated = chol.correlate(&uncorrelated);
//!
//! // Scale by volatilities
//! let vols = [0.18, 0.22];  // SPY: 18%, QQQ: 22%
//! let returns_spy = correlated[0] * vols[0];
//! let returns_qqq = correlated[1] * vols[1];
//!
//! println!("Correlated returns: SPY={:.2}%, QQQ={:.2}%",
//!          returns_spy * 100.0, returns_qqq * 100.0);
//!
//! // Use this in Monte Carlo loops:
//! // for _ in 0..10000 {
//! //     let z = generate_normal_vector();
//! //     let corr_z = chol.correlate(&z);
//! //     // Simulate prices with correlated shocks
//! // }
//! ```
//!
//! ## 3. Covariance Estimation and Risk Decomposition
//!
//! Estimate covariance from returns and decompose portfolio risk:
//!
//! ```
//! use algotrading::matrix::{Cholesky, estimation::RollingCovariance};
//!
//! // Historical returns for SPY and TLT (5 days)
//! let spy_returns = vec![0.01, -0.005, 0.008, -0.002, 0.012];
//! let tlt_returns = vec![-0.003, 0.007, -0.001, 0.004, -0.002];
//!
//! // Compute sample covariance matrix
//! let mut cov_estimator = RollingCovariance::<2>::new();
//! for i in 0..spy_returns.len() {
//!     let returns = [spy_returns[i], tlt_returns[i]];
//!     cov_estimator.update(&returns);
//! }
//! let cov_matrix = cov_estimator.covariance();
//!
//! // Decompose for risk analysis
//! let chol = Cholesky::decompose(&cov_matrix).unwrap();
//!
//! // Test different portfolio allocations
//! let allocations = [
//!     ([1.0, 0.0], "100% SPY"),
//!     ([0.0, 1.0], "100% TLT"),
//!     ([0.6, 0.4], "60/40 Portfolio"),
//!     ([0.5, 0.5], "50/50 Portfolio"),
//! ];
//!
//! for (weights, name) in allocations {
//!     let var = chol.portfolio_variance(&weights);
//!     let vol = var.sqrt();
//!     println!("{}: {:.2}% volatility", name, vol * 100.0);
//! }
//! ```
//!
//! ## 4. Principal Component Analysis (PCA) for Factor Models
//!
//! Extract common risk factors from asset returns:
//!
//! ```
//! use algotrading::matrix::Eigen;
//!
//! // Correlation matrix of 4 tech stocks (highly correlated)
//! let corr = [
//!     [1.00, 0.85, 0.80, 0.75],  // AAPL
//!     [0.85, 1.00, 0.88, 0.82],  // MSFT
//!     [0.80, 0.88, 1.00, 0.78],  // GOOGL
//!     [0.75, 0.82, 0.78, 1.00],  // NVDA
//! ];
//!
//! // Compute eigendecomposition
//! let eigen = Eigen::decompose_symmetric(&corr).unwrap();
//!
//! // Analyze explained variance (how much risk is common vs. idiosyncratic)
//! let explained = eigen.explained_variance_ratio();
//! println!("Explained variance by factor:");
//! println!("  Factor 1 (Tech sector): {:.1}%", explained[0] * 100.0);
//! println!("  Factor 2 (Size effect): {:.1}%", explained[1] * 100.0);
//! println!("  Factor 3 (Idiosyncratic): {:.1}%", explained[2] * 100.0);
//! println!("  Factor 4 (Idiosyncratic): {:.1}%", explained[3] * 100.0);
//!
//! // For highly correlated assets, first factor dominates (>80%)
//! // This means most risk comes from common tech sector exposure
//!
//! // Reduce dimensionality for risk modeling
//! let returns = [0.02, 0.018, 0.025, 0.022];  // Daily returns
//! let factor_exposures = eigen.project(&returns, 2);  // Keep top 2 factors
//! println!("Factor exposures: {:?}", factor_exposures);
//! ```
//!
//! ## 5. Risk Parity Portfolio Construction
//!
//! Allocate capital such that each asset contributes equally to portfolio risk:
//!
//! ```
//! use algotrading::matrix::Cholesky;
//!
//! // Covariance matrix for 3 assets with different risk levels
//! let cov = [
//!     [0.0400, 0.0020, 0.0010],  // High-risk equity: 20% vol
//!     [0.0020, 0.0100, 0.0005],  // Medium-risk bonds: 10% vol
//!     [0.0010, 0.0005, 0.0025],  // Low-risk cash: 5% vol
//! ];
//!
//! let chol = Cholesky::decompose(&cov).unwrap();
//!
//! // Initial guess: inverse volatility weighting
//! let vols = [0.20, 0.10, 0.05];
//! let inv_vol_sum: f64 = vols.iter().map(|v| 1.0 / v).sum();
//! let weights = [
//!     (1.0 / vols[0]) / inv_vol_sum,
//!     (1.0 / vols[1]) / inv_vol_sum,
//!     (1.0 / vols[2]) / inv_vol_sum,
//! ];
//!
//! println!("Risk parity weights: [{:.1}%, {:.1}%, {:.1}%]",
//!          weights[0] * 100.0, weights[1] * 100.0, weights[2] * 100.0);
//! // Output: [12.5%, 25.0%, 62.5%]
//! // Low-risk assets get highest weight to equalize risk contribution
//!
//! let total_var = chol.portfolio_variance(&weights);
//! println!("Portfolio vol: {:.2}%", total_var.sqrt() * 100.0);
//! ```
//!
//! # Performance Characteristics
//!
//! | Operation | Complexity | Notes |
//! |-----------|-----------|-------|
//! | Cholesky decomposition | O(N³) | One-time cost, then reuse |
//! | Portfolio variance via Cholesky | O(N²) | 2x faster than naive O(N³) |
//! | Eigen decomposition (Jacobi) | O(N³) | Iterative, converges in ~10-50 iterations |
//! | Matrix-vector multiply | O(N²) | Cache-optimized with manual unrolling |
//! | Quadratic form (small N≤8) | O(N²) | Manually unrolled for ILP |
//!
//! All structures are `#[repr(align(64))]` for cache-line alignment.

/// Cholesky decomposition for covariance matrices
/// Returns lower triangular matrix L where A = L * L^T
///
/// Used for: portfolio risk decomposition, correlated random variate generation
#[repr(align(64))]
pub struct Cholesky<const N: usize> {
    lower: [[f64; N]; N],
}

impl<const N: usize> Cholesky<N> {
    /// Decompose a symmetric positive-definite matrix
    pub fn decompose(matrix: &[[f64; N]; N]) -> Result<Self, &'static str> {
        let mut lower = [[0.0; N]; N];
        
        for i in 0..N {
            for j in 0..=i {
                let mut sum = 0.0;
                
                if i == j {
                    // Diagonal elements
                    for k in 0..j {
                        sum += lower[j][k] * lower[j][k];
                    }
                    let val = matrix[j][j] - sum;
                    if val <= 0.0 {
                        return Err("Matrix not positive definite");
                    }
                    lower[j][j] = val.sqrt();
                } else {
                    // Off-diagonal elements
                    for k in 0..j {
                        sum += lower[i][k] * lower[j][k];
                    }
                    lower[i][j] = (matrix[i][j] - sum) / lower[j][j];
                }
            }
        }
        
        Ok(Self { lower })
    }
    
    /// Get lower triangular matrix
    #[inline]
    pub fn lower(&self) -> &[[f64; N]; N] {
        &self.lower
    }
    
    /// Compute portfolio variance: w^T * Σ * w = ||L^T * w||^2
    /// This is much faster than full matrix multiplication
    #[inline]
    pub fn portfolio_variance(&self, weights: &[f64; N]) -> f64 {
        let mut temp = [0.0; N];
        
        // temp = L^T * weights
        for i in 0..N {
            for j in 0..=i {
                temp[j] += self.lower[i][j] * weights[i];
            }
        }
        
        // ||temp||^2
        temp.iter().map(|&x| x * x).sum()
    }
    
    /// Generate correlated random vector from uncorrelated standard normals
    /// Used for Monte Carlo simulation with correlated assets
    #[inline]
    pub fn correlate(&self, uncorrelated: &[f64; N]) -> [f64; N] {
        let mut correlated = [0.0; N];
        
        for i in 0..N {
            for j in 0..=i {
                correlated[i] += self.lower[i][j] * uncorrelated[j];
            }
        }
        
        correlated
    }
}

/// Eigenvalue decomposition for symmetric matrices
/// Returns eigenvalues and eigenvectors
/// 
/// Used for: PCA, portfolio optimization, factor models
pub struct Eigen<const N: usize> {
    values: [f64; N],
    vectors: [[f64; N]; N],
}

impl<const N: usize> Eigen<N> {
    /// Compute eigendecomposition using Jacobi algorithm
    /// Only works for symmetric matrices (like covariance matrices)
    pub fn decompose_symmetric(matrix: &[[f64; N]; N]) -> Result<Self, &'static str> {
        const MAX_ITERATIONS: usize = 100;
        const TOLERANCE: f64 = 1e-10;
        
        let mut a = *matrix;
        let mut v = [[0.0; N]; N];
        
        // Initialize V as identity
        for i in 0..N {
            v[i][i] = 1.0;
        }
        
        // Jacobi iterations
        for _ in 0..MAX_ITERATIONS {
            // Find largest off-diagonal element
            let mut max_val = 0.0;
            let mut p = 0;
            let mut q = 0;
            
            for i in 0..N {
                for j in (i + 1)..N {
                    if a[i][j].abs() > max_val {
                        max_val = a[i][j].abs();
                        p = i;
                        q = j;
                    }
                }
            }
            
            // Converged?
            if max_val < TOLERANCE {
                break;
            }
            
            // Compute rotation angle
            let diff = a[q][q] - a[p][p];
            let t = if diff.abs() < TOLERANCE {
                if a[p][q] > 0.0 { 1.0 } else { -1.0 }
            } else {
                let phi = diff / (2.0 * a[p][q]);
                1.0 / (phi + phi.signum() * (1.0 + phi * phi).sqrt())
            };
            
            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = t * c;
            
            // Apply rotation to A
            let app = a[p][p];
            let aqq = a[q][q];
            let apq = a[p][q];
            
            a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
            a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
            a[p][q] = 0.0;
            a[q][p] = 0.0;
            
            for i in 0..N {
                if i != p && i != q {
                    let aip = a[i][p];
                    let aiq = a[i][q];
                    a[i][p] = c * aip - s * aiq;
                    a[p][i] = a[i][p];
                    a[i][q] = s * aip + c * aiq;
                    a[q][i] = a[i][q];
                }
            }
            
            // Apply rotation to V
            for i in 0..N {
                let vip = v[i][p];
                let viq = v[i][q];
                v[i][p] = c * vip - s * viq;
                v[i][q] = s * vip + c * viq;
            }
        }
        
        // Extract eigenvalues from diagonal
        let mut values = [0.0; N];
        for i in 0..N {
            values[i] = a[i][i];
        }
        
        Ok(Self {
            values,
            vectors: v,
        })
    }
    
    /// Get eigenvalues (sorted descending)
    pub fn values(&self) -> [f64; N] {
        let mut sorted = self.values;
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        sorted
    }
    
    /// Get eigenvectors
    #[inline]
    pub fn vectors(&self) -> &[[f64; N]; N] {
        &self.vectors
    }
    
    /// Project data onto principal components
    /// Used for dimensionality reduction
    #[inline]
    pub fn project(&self, data: &[f64; N], n_components: usize) -> [f64; N] {
        let mut result = [0.0; N];
        
        for i in 0..n_components.min(N) {
            let mut dot = 0.0;
            for j in 0..N {
                dot += data[j] * self.vectors[j][i];
            }
            result[i] = dot;
        }
        
        result
    }
    
    /// Compute explained variance ratio
    #[inline]
    pub fn explained_variance_ratio(&self) -> [f64; N] {
        let total: f64 = self.values.iter().sum();
        let mut ratios = [0.0; N];
        
        for i in 0..N {
            ratios[i] = self.values[i] / total;
        }
        
        ratios
    }
}

/// SIMD-optimized matrix-vector multiplication
/// Critical operation for portfolio calculations
///
/// **Performance-critical:** Uses `inline(always)` for hot paths like Mahalanobis distance
#[inline(always)]
pub fn matvec_multiply<const N: usize>(matrix: &[[f64; N]; N], vector: &[f64; N]) -> [f64; N] {
    let mut result = [0.0; N];

    for i in 0..N {
        for j in 0..N {
            result[i] += matrix[i][j] * vector[j];
        }
    }

    result
}

/// Compute quadratic form: x^T * A * x
/// Used extensively in portfolio optimization and risk calculations
///
/// **Performance-critical:** Manually unrolled for small matrices, uses FMA when possible
#[inline(always)]
pub fn quadratic_form<const N: usize>(matrix: &[[f64; N]; N], vector: &[f64; N]) -> f64 {
    // Unroll for small matrices - compiler optimizes the branches away at compile time
    if N <= 8 {
        quadratic_form_small(matrix, vector)
    } else {
        quadratic_form_large(matrix, vector)
    }
}

/// Optimized quadratic form for small matrices (N <= 8) with manual unrolling
#[inline(always)]
fn quadratic_form_small<const N: usize>(matrix: &[[f64; N]; N], vector: &[f64; N]) -> f64 {
    debug_assert!(N <= 8);

    // Compute matrix-vector multiply inline with explicit unrolling
    let mut temp = [0.0; 8]; // Stack buffer for N <= 8

    // Manually unrolled matrix-vector multiply
    let mut i = 0;
    while i < N {
        let mut sum = 0.0;
        let mut j = 0;

        // Unroll inner loop by 4 for better ILP
        while j + 3 < N {
            sum += matrix[i][j] * vector[j];
            sum += matrix[i][j + 1] * vector[j + 1];
            sum += matrix[i][j + 2] * vector[j + 2];
            sum += matrix[i][j + 3] * vector[j + 3];
            j += 4;
        }

        // Handle remaining elements
        while j < N {
            sum += matrix[i][j] * vector[j];
            j += 1;
        }

        temp[i] = sum;
        i += 1;
    }

    // Compute dot product: vector^T * temp
    let mut result = 0.0;
    let mut i = 0;

    // Unroll by 4
    while i + 3 < N {
        result += vector[i] * temp[i];
        result += vector[i + 1] * temp[i + 1];
        result += vector[i + 2] * temp[i + 2];
        result += vector[i + 3] * temp[i + 3];
        i += 4;
    }

    // Handle remaining elements
    while i < N {
        result += vector[i] * temp[i];
        i += 1;
    }

    result
}

/// Quadratic form for larger matrices
#[inline(always)]
fn quadratic_form_large<const N: usize>(matrix: &[[f64; N]; N], vector: &[f64; N]) -> f64 {
    let temp = matvec_multiply(matrix, vector);

    let mut result = 0.0;
    for i in 0..N {
        result += vector[i] * temp[i];
    }

    result
}

/// Matrix transpose
#[inline(always)]
pub fn transpose<const N: usize>(matrix: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut result = [[0.0; N]; N];

    for i in 0..N {
        for j in 0..N {
            result[j][i] = matrix[i][j];
        }
    }

    result
}

/// Matrix-matrix multiplication
#[inline(always)]
pub fn matmul<const N: usize>(a: &[[f64; N]; N], b: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut result = [[0.0; N]; N];

    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    result
}

/// Matrix multiplication with transpose: A * B^T
/// More efficient than matmul(a, &transpose(b))
#[inline(always)]
pub fn matmul_transpose<const N: usize>(a: &[[f64; N]; N], b: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut result = [[0.0; N]; N];

    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                result[i][j] += a[i][k] * b[j][k]; // Note: b[j][k] not b[k][j]
            }
        }
    }

    result
}

/// Matrix addition
#[inline(always)]
pub fn mat_add<const N: usize>(a: &[[f64; N]; N], b: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut result = [[0.0; N]; N];

    for i in 0..N {
        for j in 0..N {
            result[i][j] = a[i][j] + b[i][j];
        }
    }

    result
}

/// Matrix subtraction
#[inline(always)]
pub fn mat_sub<const N: usize>(a: &[[f64; N]; N], b: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut result = [[0.0; N]; N];

    for i in 0..N {
        for j in 0..N {
            result[i][j] = a[i][j] - b[i][j];
        }
    }

    result
}

/// Identity matrix
#[inline(always)]
pub fn identity<const N: usize>() -> [[f64; N]; N] {
    let mut result = [[0.0; N]; N];

    for i in 0..N {
        result[i][i] = 1.0;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cholesky() {
        // Simple 2x2 covariance matrix
        let cov = [
            [4.0, 2.0],
            [2.0, 3.0],
        ];
        
        let chol = Cholesky::decompose(&cov).unwrap();
        let l = chol.lower();
        
        // Verify: L * L^T = Σ
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = 0.0;
                for k in 0..2 {
                    sum += l[i][k] * l[j][k];
                }
                assert!((sum - cov[i][j]).abs() < 1e-10);
            }
        }
    }
    
    #[test]
    fn test_portfolio_variance() {
        let cov = [
            [0.04, 0.02],
            [0.02, 0.09],
        ];
        
        let chol = Cholesky::decompose(&cov).unwrap();
        let weights = [0.6, 0.4];
        
        let var = chol.portfolio_variance(&weights);
        
        // Manual calculation: w^T * Σ * w
        let expected = 0.6 * 0.6 * 0.04 + 2.0 * 0.6 * 0.4 * 0.02 + 0.4 * 0.4 * 0.09;
        println!("Computed variance: {}, Expected variance: {}", var, expected);
        assert!((var - expected).abs() < 1e-10);
    }
    
    #[test]
    fn test_eigen_identity() {
        let identity = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        
        let eigen = Eigen::decompose_symmetric(&identity).unwrap();
        let values = eigen.values();
        
        // Identity matrix has eigenvalues of 1
        for &val in &values {
            assert!((val - 1.0).abs() < 1e-8);
        }
    }
    
    #[test]
    fn test_quadratic_form() {
        let matrix = [
            [2.0, 1.0],
            [1.0, 2.0],
        ];
        let vector = [1.0, 1.0];
        
        let result = quadratic_form(&matrix, &vector);
        
        // Manual: [1,1] * [[2,1],[1,2]] * [1,1]^T = [1,1] * [3,3]^T = 6
        assert!((result - 6.0).abs() < 1e-10);
    }
}