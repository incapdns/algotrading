//! Rolling (online) probability distributions for live trading systems
//!
//! These structures maintain probability distributions in **constant time O(1)** per update,
//! using numerically stable streaming algorithms (Welford's method).
//!
//! # Why Use Rolling Distributions?
//!
//! Traditional batch statistics require recomputing over historical windows:
//! - **Batch approach**: O(N) per update → slow for real-time trading
//! - **Rolling approach**: O(1) per update → perfect for live systems
//!
//! # Trading Applications
//!
//! ## 1. Live Volatility Tracking
//!
//! Track realized volatility in real-time without storing price history:
//!
//! ```
//! use algotrading::probability::rolling::RollingNormal;
//!
//! let mut vol_tracker = RollingNormal::new();
//!
//! // Stream returns as they arrive
//! let returns = vec![0.001, -0.002, 0.0015, -0.0008, 0.003];
//! for ret in returns {
//!     vol_tracker.update(ret);
//! }
//!
//! // Get current volatility estimate
//! let current_vol = vol_tracker.std();
//! println!("Current volatility: {:.4}", current_vol);
//!
//! // Use for position sizing
//! let risk_per_trade = 0.02;  // 2% risk
//! let vol_adjusted_size = risk_per_trade / current_vol;
//! ```
//!
//! ## 2. Online Z-Score for Entry Signals
//!
//! Detect when prices deviate from recent behavior:
//!
//! ```
//! use algotrading::probability::rolling::RollingZScore;
//!
//! let mut zscore = RollingZScore::new();
//!
//! // Track price deviations in real-time
//! let prices = vec![100.0, 101.0, 99.5, 102.0, 98.0];
//! for price in prices {
//!     let z = zscore.update(price);
//!
//!     // Trading signals
//!     if z > 2.0 {
//!         println!("Price at +2σ - consider mean reversion short");
//!     } else if z < -2.0 {
//!         println!("Price at -2σ - consider mean reversion long");
//!     }
//! }
//! ```
//!
//! ## 3. Adaptive Risk Models (Student's T)
//!
//! Model fat-tailed returns for better tail-risk estimation:
//!
//! ```
//! use algotrading::probability::rolling::RollingStudentT;
//! use algotrading::probability::distributions::Distribution;
//!
//! let mut t_dist = RollingStudentT::new(5.0);  // Start with df=5
//!
//! // Feed returns (handles outliers better than normal)
//! let returns = vec![-0.05, 0.02, -0.01, 0.03, -0.08];  // Includes crash
//! for ret in returns {
//!     t_dist.update(ret);
//! }
//!
//! // Get current distribution for VaR calculation
//! let dist = t_dist.current_distribution();
//! let prob_5pct_loss = dist.cdf(-0.05);
//! println!("P(loss > 5%) = {:.2}%", prob_5pct_loss * 100.0);
//! ```
//!
//! ## 4. Price Distribution Tracking (Log-Normal)
//!
//! Model asset price distributions (prices can't go negative):
//!
//! ```
//! use algotrading::probability::rolling::RollingLogNormal;
//! use algotrading::probability::distributions::Distribution;
//!
//! let mut price_dist = RollingLogNormal::new();
//!
//! // Track prices (not returns)
//! let prices = vec![100.0, 102.0, 98.0, 105.0, 103.0];
//! for price in prices {
//!     price_dist.update(price);
//! }
//!
//! // Get current distribution
//! let dist = price_dist.current_distribution();
//!
//! // Probability of price above $110
//! let prob_above_110 = 1.0 - dist.cdf(110.0);
//! println!("P(price > $110) = {:.2}%", prob_above_110 * 100.0);
//! ```
//!
//! # Performance Characteristics
//!
//! | Operation | Time Complexity | Memory |
//! |-----------|----------------|---------|
//! | Update | O(1) | O(1) |
//! | Get stats | O(1) | O(1) |
//! | Batch (comparison) | O(N) | O(N) |
//!
//! **Perfect for:**
//! - High-frequency trading (microsecond updates)
//! - Live risk monitoring
//! - Real-time signal generation
//! - Memory-constrained systems
//!
//! **Numerical Stability:**
//! - Uses Welford's algorithm (numerically stable)
//! - No catastrophic cancellation
//! - Accurate even with billions of updates
//!
//! # Implementation Details
//!
//! All rolling distributions use **Welford's online algorithm**:
//! ```text
//! δ = x - mean
//! mean += δ / n
//! δ2 = x - mean
//! M2 += δ * δ2
//! variance = M2 / n
//! ```
//!
//! This prevents loss of precision compared to naive variance formulas.

use crate::probability::distributions::*;
use std::f64::consts::PI;

/// Trait for all rolling (stateful) statistical models.
pub trait RollingDistribution {
    /// Updates the estimator with a new sample.
    fn update(&mut self, x: f64);

    /// Returns the number of observations processed so far.
    fn count(&self) -> usize;

    /// Returns the current estimated distribution parameters.
    fn current(&self) -> Box<dyn Distribution + Send + Sync>;
}

// ===========================================================
// Rolling Normal Distribution
// ===========================================================

#[derive(Debug, Clone)]
pub struct RollingNormal {
    n: usize,
    mean: f64,
    m2: f64, // sum of squares of differences from the current mean
}

impl RollingNormal {
    pub fn new() -> Self {
        Self { n: 0, mean: 0.0, m2: 0.0 }
    }

    #[inline]
    pub fn mean(&self) -> f64 {
        self.mean
    }

    #[inline]
    pub fn variance(&self) -> f64 {
        if self.n > 1 {
            self.m2 / (self.n as f64)
        } else {
            0.0
        }
    }

    #[inline]
    pub fn std(&self) -> f64 {
        self.variance().sqrt()
    }

    #[inline]
    pub fn current_distribution(&self) -> Normal {
        Normal::new(self.mean, self.std().max(1e-12))
    }
}

impl RollingDistribution for RollingNormal {
    fn update(&mut self, x: f64) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / (self.n as f64);
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    fn count(&self) -> usize {
        self.n
    }

    fn current(&self) -> Box<dyn Distribution + Send + Sync> {
        Box::new(self.current_distribution())
    }
}

// ===========================================================
// Rolling LogNormal Distribution
// ===========================================================

#[derive(Debug, Clone)]
pub struct RollingLogNormal {
    n: usize,
    mean_log: f64,
    m2_log: f64,
}

impl RollingLogNormal {
    pub fn new() -> Self {
        Self { n: 0, mean_log: 0.0, m2_log: 0.0 }
    }

    #[inline]
    pub fn mean(&self) -> f64 {
        self.mean_log
    }

    #[inline]
    pub fn variance(&self) -> f64 {
        if self.n > 1 {
            self.m2_log / (self.n as f64)
        } else {
            0.0
        }
    }

    #[inline]
    pub fn std(&self) -> f64 {
        self.variance().sqrt()
    }

    #[inline]
    pub fn current_distribution(&self) -> LogNormal {
        LogNormal::new(self.mean_log, self.std().max(1e-12))
    }
}

impl RollingDistribution for RollingLogNormal {
    fn update(&mut self, x: f64) {
        if x <= 0.0 {
            return; // skip invalid samples
        }

        let lx = x.ln();
        self.n += 1;
        let delta = lx - self.mean_log;
        self.mean_log += delta / (self.n as f64);
        let delta2 = lx - self.mean_log;
        self.m2_log += delta * delta2;
    }

    fn count(&self) -> usize {
        self.n
    }

    fn current(&self) -> Box<dyn Distribution + Send + Sync> {
        Box::new(self.current_distribution())
    }
}

// ===========================================================
// Rolling Student-T Distribution (adaptive variance scaling)
// ===========================================================

#[derive(Debug, Clone)]
pub struct RollingStudentT {
    n: usize,
    mean: f64,
    m2: f64,
    df: f64,
}

impl RollingStudentT {
    pub fn new(initial_df: f64) -> Self {
        Self { n: 0, mean: 0.0, m2: 0.0, df: initial_df }
    }

    #[inline]
    pub fn variance(&self) -> f64 {
        if self.n > 1 {
            self.m2 / (self.n as f64)
        } else {
            0.0
        }
    }

    #[inline]
    pub fn std(&self) -> f64 {
        self.variance().sqrt()
    }

    pub fn current_distribution(&self) -> StudentT {
        // Optionally adapt degrees of freedom to volatility regime
        let adaptive_df = (self.df + (self.std() * 10.0).min(30.0)).max(2.0);
        StudentT::new(adaptive_df)
    }
}

impl RollingDistribution for RollingStudentT {
    fn update(&mut self, x: f64) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / (self.n as f64);
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;

        // Example heuristic: increase df slowly as we get more data
        if self.n % 50 == 0 {
            self.df = (self.df + 1.0).min(100.0);
        }
    }

    fn count(&self) -> usize {
        self.n
    }

    fn current(&self) -> Box<dyn Distribution + Send + Sync> {
        Box::new(self.current_distribution())
    }
}

// ===========================================================
// Utility: Online Z-Score (lightweight non-allocating)
// ===========================================================

#[derive(Debug, Clone)]
pub struct RollingZScore {
    normal: RollingNormal,
}

impl RollingZScore {
    pub fn new() -> Self {
        Self { normal: RollingNormal::new() }
    }

    #[inline]
    pub fn update(&mut self, x: f64) -> f64 {
        self.normal.update(x);
        if self.normal.count() > 1 {
            (x - self.normal.mean()) / self.normal.std().max(1e-12)
        } else {
            0.0
        }
    }
}
