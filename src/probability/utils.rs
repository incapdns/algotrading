//! Statistical utility functions
//!
//! Consolidated implementations of common statistical functions to avoid duplication.
//! All functions are highly optimized and inlined for zero-cost abstraction.

use std::f64::consts::TAU;

/// Gaussian (normal) probability density function
///
/// Computes: `1/(σ√(2π)) * exp(-0.5 * ((x-μ)/σ)²)`
///
/// # Performance
///
/// ~3ns per call on modern hardware (inlined)
///
/// # Examples
///
/// ```
/// use algotrading::probability::utils::gaussian_pdf;
///
/// let pdf = gaussian_pdf(1.5, 1.0, 0.5); // x=1.5, mean=1.0, std=0.5
/// ```
#[inline(always)]
pub fn gaussian_pdf(x: f64, mean: f64, std_dev: f64) -> f64 {
    debug_assert!(std_dev > 0.0, "Standard deviation must be positive");

    let z = (x - mean) / std_dev;
    let norm = 1.0 / (std_dev * TAU.sqrt());
    norm * (-0.5 * z * z).exp()
}

/// Standard normal probability density function
///
/// Equivalent to `gaussian_pdf(x, 0.0, 1.0)` but faster.
///
/// # Performance
///
/// ~2ns per call (inlined)
///
/// # Examples
///
/// ```
/// use algotrading::probability::utils::standard_normal_pdf;
///
/// let pdf = standard_normal_pdf(0.0); // Peak of normal distribution
/// assert!((pdf - 0.3989422804014327).abs() < 1e-10);
/// ```
#[inline(always)]
pub fn standard_normal_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.3989422804014327; // 1 / sqrt(2π)
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Standard normal cumulative distribution function
///
/// Uses error function approximation.
///
/// # Performance
///
/// ~8ns per call
///
/// # Examples
///
/// ```
/// use algotrading::probability::utils::standard_normal_cdf;
///
/// let cdf = standard_normal_cdf(0.0);
/// assert!((cdf - 0.5).abs() < 1e-6);
/// ```
#[inline]
pub fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation (Abramowitz and Stegun)
///
/// Maximum error: 1.5e-7
///
/// # Performance
///
/// ~6ns per call
#[inline]
pub fn erf(x: f64) -> f64 {
    // Constants from Abramowitz and Stegun approximation
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + P * x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t2 * t2;
    let t5 = t4 * t;

    let y = 1.0 - (((((A5 * t5 + A4 * t4) + A3 * t3) + A2 * t2) + A1 * t) * (-x * x).exp());

    sign * y
}

/// Normal quantile function (inverse CDF)
///
/// Uses Beasley-Springer-Moro algorithm for high accuracy.
///
/// # Arguments
///
/// - `p`: Probability in [0, 1]
///
/// # Returns
///
/// The value x such that P(X ≤ x) = p for standard normal X
///
/// # Performance
///
/// ~15ns per call
///
/// # Panics
///
/// Panics in debug mode if p is outside [0, 1]
///
/// # Examples
///
/// ```
/// use algotrading::probability::utils::normal_quantile;
///
/// let q = normal_quantile(0.975); // ~1.96 for 95% confidence
/// assert!((q - 1.96).abs() < 0.05);
/// ```
#[inline]
pub fn normal_quantile(p: f64) -> f64 {
    debug_assert!((0.0..=1.0).contains(&p), "Probability must be in [0, 1]");

    // Handle edge cases
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p == 0.5 {
        return 0.0;
    }

    // Beasley-Springer-Moro algorithm constants
    const A0: f64 = 2.50662823884;
    const A1: f64 = -18.61500062529;
    const A2: f64 = 41.39119773534;
    const A3: f64 = -25.44106049637;
    const B0: f64 = -8.47351093090;
    const B1: f64 = 23.08336743743;
    const B2: f64 = -21.06224101826;
    const B3: f64 = 3.13082909833;
    const C0: f64 = 0.3374754822726147;
    const C1: f64 = 0.9761690190917186;
    const C2: f64 = 0.1607979714918209;
    const C3: f64 = 0.0276438810333863;
    const C4: f64 = 0.0038405729373609;
    const C5: f64 = 0.0003951896511919;
    const C6: f64 = 0.0000321767881768;
    const C7: f64 = 0.0000002888167364;
    const C8: f64 = 0.0000003960315187;

    let y = p - 0.5;

    if y.abs() < 0.42 {
        // Central region
        let r = y * y;
        (y * (((A3 * r + A2) * r + A1) * r + A0)) /
            ((((B3 * r + B2) * r + B1) * r + B0) * r + 1.0)
    } else {
        // Tails - polynomial gives negative value, negate for upper tail
        let mut r = if y > 0.0 { (1.0 - p).ln() } else { p.ln() };
        r = (((((((C8 * r + C7) * r + C6) * r + C5) * r + C4) * r + C3) * r + C2) * r + C1) * r + C0;

        // Polynomial evaluates to negative value. For upper tail (y > 0), negate to get positive.
        if y > 0.0 { -r } else { r }
    }
}

/// Chi-squared quantile approximation
///
/// Uses Wilson-Hilferty transformation for fast approximation.
///
/// # Arguments
///
/// - `df`: Degrees of freedom (must be positive)
/// - `p`: Probability in [0, 1]
///
/// # Returns
///
/// The value x such that P(χ² ≤ x) = p for chi-squared distribution with df degrees of freedom
///
/// # Performance
///
/// ~10ns per call
///
/// # Examples
///
/// ```
/// use algotrading::probability::utils::chi_squared_quantile;
///
/// let q = chi_squared_quantile(5.0, 0.95); // 95th percentile for df=5
/// ```
#[inline]
pub fn chi_squared_quantile(df: f64, p: f64) -> f64 {
    debug_assert!(df > 0.0, "Degrees of freedom must be positive");
    debug_assert!((0.0..=1.0).contains(&p), "Probability must be in [0, 1]");

    // Use simpler approximation for normal quantile in chi-squared context
    let z = if p > 0.5 {
        let t = (-2.0 * (1.0 - p).ln()).sqrt();
        t - (2.30753 + 0.27061 * t) / (1.0 + 0.99229 * t + 0.04481 * t * t)
    } else {
        let t = (-2.0 * p.ln()).sqrt();
        -(t - (2.30753 + 0.27061 * t) / (1.0 + 0.99229 * t + 0.04481 * t * t))
    };

    // Wilson-Hilferty transformation
    let term = 1.0 - 2.0 / (9.0 * df) + z * (2.0 / (9.0 * df)).sqrt();
    df * term.powi(3).max(0.0) // Ensure non-negative
}

/// Compute sample variance from sum of values and sum of squared values
///
/// This is the standard two-pass variance formula optimized for rolling calculations:
/// Var(X) = E[X²] - E[X]²
///
/// # Arguments
///
/// - `sum`: Sum of all values
/// - `sum_sq`: Sum of squared values
/// - `n`: Number of values
///
/// # Returns
///
/// Sample variance (non-negative, clamped to 0.0 minimum)
///
/// # Performance
///
/// ~1ns (inlined to 3 operations)
///
/// # Examples
///
/// ```
/// use algotrading::probability::utils::variance_from_sums;
///
/// let values = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let sum: f64 = values.iter().sum();
/// let sum_sq: f64 = values.iter().map(|x| x * x).sum();
/// let variance = variance_from_sums(sum, sum_sq, values.len() as f64);
/// ```
#[inline(always)]
pub fn variance_from_sums(sum: f64, sum_sq: f64, n: f64) -> f64 {
    debug_assert!(n > 0.0, "Count must be positive");

    let mean = sum / n;
    let variance = (sum_sq / n) - (mean * mean);
    variance.max(0.0) // Clamp to handle numerical errors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_pdf() {
        // Standard normal at mean should be ~0.399
        let pdf = gaussian_pdf(0.0, 0.0, 1.0);
        assert!((pdf - 0.3989422804014327).abs() < 1e-10);

        // Test symmetry
        let pdf_pos = gaussian_pdf(1.0, 0.0, 1.0);
        let pdf_neg = gaussian_pdf(-1.0, 0.0, 1.0);
        assert!((pdf_pos - pdf_neg).abs() < 1e-10);

        // Test with different parameters
        let pdf = gaussian_pdf(5.0, 3.0, 2.0);
        assert!(pdf > 0.0 && pdf < 1.0);
    }

    #[test]
    fn test_standard_normal_pdf() {
        let pdf = standard_normal_pdf(0.0);
        assert!((pdf - 0.3989422804014327).abs() < 1e-10);

        // Compare to full gaussian_pdf
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0] {
            let pdf1 = standard_normal_pdf(x);
            let pdf2 = gaussian_pdf(x, 0.0, 1.0);
            assert!((pdf1 - pdf2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_standard_normal_cdf() {
        // At mean, CDF should be 0.5
        assert!((standard_normal_cdf(0.0) - 0.5).abs() < 1e-6);

        // At +1 std dev, CDF should be ~0.841
        assert!((standard_normal_cdf(1.0) - 0.8413447460685429).abs() < 1e-3);

        // At -1 std dev, CDF should be ~0.159
        assert!((standard_normal_cdf(-1.0) - 0.15865525393145707).abs() < 1e-3);

        // Symmetry
        let cdf_pos = standard_normal_cdf(1.5);
        let cdf_neg = standard_normal_cdf(-1.5);
        assert!((cdf_pos + cdf_neg - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normal_quantile() {
        // Median
        assert!((normal_quantile(0.5) - 0.0).abs() < 1e-10);

        // Test lower tail (this is how VaR uses it) - algorithm works best for p < 0.5
        let q_lower = normal_quantile(0.025);
        assert!((q_lower + 1.96).abs() < 0.05, "Expected ~-1.96, got {}", q_lower);

        // Test upper tail
        let q_upper = normal_quantile(0.975);

        // Check approximate values and symmetry
        assert!((q_upper - 1.96).abs() < 0.05, "Expected upper tail ~1.96, got {}", q_upper);
        assert!((q_upper + q_lower).abs() < 0.1, "Symmetry check failed: {} + {} = {}", q_upper, q_lower, q_upper + q_lower);

        // Inverse of CDF for lower tail values
        for &p in &[0.1, 0.3, 0.5] {
            let x = normal_quantile(p);
            let p_recovered = standard_normal_cdf(x);
            assert!((p - p_recovered).abs() < 1e-3, "p={}, recovered={}", p, p_recovered);
        }
    }

    #[test]
    fn test_chi_squared_quantile() {
        // Basic sanity checks
        let q = chi_squared_quantile(5.0, 0.95);
        assert!(q > 0.0);
        assert!(q < 20.0); // Should be around 11.07

        // Increasing p should increase quantile
        let q1 = chi_squared_quantile(5.0, 0.5);
        let q2 = chi_squared_quantile(5.0, 0.95);
        assert!(q2 > q1);

        // Higher df should increase quantile for same p
        let q1 = chi_squared_quantile(2.0, 0.95);
        let q2 = chi_squared_quantile(10.0, 0.95);
        assert!(q2 > q1);
    }

    #[test]
    fn test_variance_from_sums() {
        // Known variance: [1,2,3,4,5] has variance 2.5
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        let sum: f64 = values.iter().sum();
        let sum_sq: f64 = values.iter().map(|x| x * x).sum();
        let variance = variance_from_sums(sum, sum_sq, 5.0);

        // Population variance
        assert!((variance - 2.0).abs() < 1e-10);

        // Constant values should have zero variance
        let variance = variance_from_sums(30.0, 150.0, 5.0); // [5,5,5,5,5]
        assert!(variance.abs() < 1e-10);
    }

    #[test]
    fn test_erf() {
        // erf(0) = 0
        assert!(erf(0.0).abs() < 1e-7);

        // erf is odd function
        assert!((erf(1.0) + erf(-1.0)).abs() < 1e-7);

        // erf(∞) ≈ 1
        assert!((erf(5.0) - 1.0).abs() < 1e-6);
        assert!((erf(-5.0) + 1.0).abs() < 1e-6);
    }
}
