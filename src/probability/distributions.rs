//! Probability distributions for quantitative finance
//!
//! This module provides efficient implementations of common probability distributions
//! used in algorithmic trading and risk management.
//!
//! # Distributions
//!
//! - **Normal (Gaussian)**: Asset returns, parametric VaR, Black-Scholes
//! - **Student's T**: Heavy-tailed returns, robust statistics
//! - **Log-Normal**: Asset prices, option pricing
//!
//! # Trading Applications
//!
//! ## 1. Risk Assessment
//!
//! Model return distributions for Value-at-Risk (VaR) calculations:
//!
//! ```
//! use algotrading::probability::{Normal, Distribution};
//!
//! // Fit normal distribution to historical returns
//! let returns = vec![-0.02, 0.01, -0.005, 0.015, 0.008];
//! let dist = Normal::from_sample::<f64>(&returns);
//!
//! // Calculate 5% VaR (95% confidence)
//! let var_95 = dist.mean + dist.std * (-1.645);  // 1.645 = z-score for 5%
//! println!("95% VaR: {:.2}%", var_95 * 100.0);
//! ```
//!
//! ## 2. Option Pricing
//!
//! Use log-normal distribution for modeling asset prices:
//!
//! ```
//! use algotrading::probability::{LogNormal, Distribution};
//!
//! // Model stock price distribution
//! let current_price = 100.0_f64;
//! let expected_return = 0.10_f64;  // 10% annual
//! let volatility = 0.20_f64;        // 20% annual
//! let time = 1.0_f64;               // 1 year
//!
//! let mu = (expected_return - 0.5 * volatility.powi(2)) * time;
//! let sigma = volatility * time.sqrt();
//! let price_dist = LogNormal::new(mu, sigma);
//!
//! // Probability of price > $120
//! let prob_above_120 = 1.0 - price_dist.cdf(120.0);
//! println!("P(S > $120) = {:.2}%", prob_above_120 * 100.0);
//! ```
//!
//! ## 3. Tail Risk Analysis
//!
//! Use Student's t-distribution for heavy-tailed returns:
//!
//! ```
//! use algotrading::probability::{StudentT, Distribution};
//!
//! // Model returns with fat tails (degrees of freedom = 5)
//! let t_dist = StudentT::new(5.0);
//!
//! // Probability density at 3 standard deviations
//! let pdf_3std = t_dist.pdf(3.0);
//! println!("PDF at 3σ: {:.6}", pdf_3std);
//!
//! // Note: t-distribution has heavier tails than normal
//! // Better for modeling crashes and extreme moves
//! ```
//!
//! # Distribution Trait
//!
//! All distributions implement the `Distribution` trait with two key methods:
//! - `pdf(x)`: Probability density function - how likely is this exact value?
//! - `cdf(x)`: Cumulative distribution function - probability of value ≤ x

use std::f64::consts::PI;
use crate::stats::core::{mean, variance};
use crate::numeric::Numeric;

/// Trait for probability distributions
///
/// # Methods
///
/// - `pdf`: Probability density function (likelihood of exact value)
/// - `cdf`: Cumulative distribution function (probability ≤ x)
pub trait Distribution {
    /// Probability density function
    ///
    /// Returns the probability density at point `x`. For continuous distributions,
    /// this is the relative likelihood of observing a value near `x`.
    fn pdf(&self, x: f64) -> f64;

    /// Cumulative distribution function
    ///
    /// Returns P(X ≤ x), the probability that a random variable from this
    /// distribution is less than or equal to `x`.
    fn cdf(&self, x: f64) -> f64;
}

// ===========================================================
// Normal Distribution
// ===========================================================

/// Normal (Gaussian) distribution
///
/// The normal distribution is the most common model for asset returns in finance.
/// It's the foundation of many quantitative models including:
/// - Black-Scholes option pricing
/// - Parametric Value-at-Risk
/// - Mean-variance portfolio optimization
///
/// # Examples
///
/// ```
/// use algotrading::probability::{Normal, Distribution};
///
/// // Create normal distribution with μ=0, σ=1
/// let std_normal = Normal::new(0.0, 1.0);
///
/// // Probability of return < -2σ
/// let prob_crash = std_normal.cdf(-2.0);
/// println!("P(return < -2σ) = {:.2}%", prob_crash * 100.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Normal {
    /// Mean (μ) - center of the distribution
    pub mean: f64,
    /// Standard deviation (σ) - measure of spread/risk
    pub std: f64,
}

impl Normal {
    pub fn new(mean: f64, std: f64) -> Self {
        Self { mean, std }
    }

    pub fn from_sample<T: Numeric>(data: &[f64]) -> Self {
        let m = mean::<T>(data);
        let s = variance::<T>(data).sqrt();
        Self { mean: m, std: s }
    }

    #[inline]
    pub fn z(&self, x: f64) -> f64 {
        (x - self.mean) / self.std
    }
}

impl Distribution for Normal {
    fn pdf(&self, x: f64) -> f64 {
        let z = self.z(x);
        (1.0 / (self.std * (2.0 * PI).sqrt())) * (-0.5 * z.powi(2)).exp()
    }

    fn cdf(&self, x: f64) -> f64 {
        0.5 * (1.0 + erf(self.z(x) / (2.0f64).sqrt()))
    }
}

// ===========================================================
// Student's T Distribution
// ===========================================================

#[derive(Debug, Clone, Copy)]
pub struct StudentT {
    pub df: f64, // degrees of freedom
}

impl StudentT {
    pub fn new(df: f64) -> Self {
        Self { df }
    }

    #[inline]
    fn gamma(x: f64) -> f64 {
        // Stirling’s approximation (good enough for most df)
        (2.0 * PI / x).sqrt() * (x / std::f64::consts::E).powf(x)
    }
}

impl Distribution for StudentT {
    fn pdf(&self, x: f64) -> f64 {
        let df = self.df;
        let c = (Self::gamma((df + 1.0) / 2.0))
            / ((df * PI).sqrt() * Self::gamma(df / 2.0));
        c * (1.0 + (x.powi(2) / df)).powf(-(df + 1.0) / 2.0)
    }

    fn cdf(&self, x: f64) -> f64 {
        // Approximation using Hill's algorithm for Student's t CDF
        // This provides reasonable accuracy for most practical purposes
        // For high precision requirements, use a specialized statistics library

        let df = self.df;

        // Handle special cases
        if x.is_nan() {
            return f64::NAN;
        }
        if x.is_infinite() {
            return if x > 0.0 { 1.0 } else { 0.0 };
        }

        // Use symmetry for negative values
        if x < 0.0 {
            return 1.0 - self.cdf(-x);
        }

        // For x = 0, use symmetry
        if x.abs() < 1e-10 {
            return 0.5;
        }

        // Hill's approximation via incomplete beta function relationship
        // CDF(t) = 1 - 0.5 * I_x(df/2, 1/2) where x = df/(df + t^2)
        let a = df / 2.0;
        let x_beta = df / (df + x * x);

        // Approximate incomplete beta using series expansion (good for df > 2)
        let beta_approx = incomplete_beta_approx(a, 0.5, x_beta);

        0.5 + 0.5 * (1.0 - beta_approx) * x.signum()
    }
}

// ===========================================================
// Log-Normal Distribution
// ===========================================================

#[derive(Debug, Clone, Copy)]
pub struct LogNormal {
    pub mu: f64,
    pub sigma: f64,
}

impl LogNormal {
    pub fn new(mu: f64, sigma: f64) -> Self {
        Self { mu, sigma }
    }

    pub fn from_sample<T: Numeric>(data: &[f64]) -> Self {
        let logs: Vec<f64> = data.iter().filter(|&&x| x > 0.0).map(|&x| x.ln()).collect();
        let m = mean::<T>(&logs);
        let s = variance::<T>(&logs).sqrt();
        Self { mu: m, sigma: s }
    }
}

impl Distribution for LogNormal {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let z = (x.ln() - self.mu) / self.sigma;
        (1.0 / (x * self.sigma * (2.0 * PI).sqrt())) * (-0.5 * z.powi(2)).exp()
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        0.5 * (1.0 + erf((x.ln() - self.mu) / (self.sigma * (2.0f64).sqrt())))
    }
}

// ===========================================================
// Common Utilities
// ===========================================================

pub fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

/// Approximate incomplete beta function I_x(a, b)
/// Uses continued fraction representation for better accuracy
fn incomplete_beta_approx(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use symmetry relation if needed for better convergence
    let (a, b, x, reflect) = if x > (a + 1.0) / (a + b + 2.0) {
        (b, a, 1.0 - x, true)
    } else {
        (a, b, x, false)
    };

    // Compute beta(a, b) using log gamma
    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);

    // Compute the coefficient
    let coef = (a * x.ln() + b * (1.0 - x).ln() - ln_beta).exp() / a;

    // Continued fraction approximation (Lentz's algorithm)
    let mut result = continued_fraction_beta(a, b, x);
    result *= coef;

    if reflect {
        1.0 - result
    } else {
        result
    }
}

/// Log gamma function using Lanczos approximation
fn ln_gamma(x: f64) -> f64 {
    let coefficients = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.001208650973866179,
        -0.000005395239384953,
    ];

    let mut y = x;
    let mut tmp = x + 5.5;
    tmp -= (x + 0.5) * tmp.ln();
    let mut ser = 1.000000000190015;
    for (_i, &coef) in coefficients.iter().enumerate() {
        y += 1.0;
        ser += coef / y;
    }

    -tmp + (2.5066282746310005 * ser / (x + 0.5)).ln()
}

/// Continued fraction for incomplete beta function
fn continued_fraction_beta(a: f64, b: f64, x: f64) -> f64 {
    const MAX_ITER: usize = 100;
    const EPSILON: f64 = 1e-10;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;

    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAX_ITER {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;

        // Even step
        let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        h *= d * c;

        // Odd step
        let aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < EPSILON {
            break;
        }
    }

    h
}
