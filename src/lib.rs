//! # Algotrading Library
//!
//! High-performance quantitative trading primitives optimized for AWS EC2.
//!
//! ## Design Philosophy
//!
//! - **f64 only**: Financial precision is non-negotiable
//! - **Stack allocated**: Const generics, zero heap allocations
//! - **SIMD optimized**: AVX2 baseline (all modern EC2 instances)
//! - **Cache-friendly**: 64-byte alignment, structure-of-arrays layout
//!
//! ## Modules
//!
//! - `stats`: Rolling statistics, PDF estimation
//! - `probability`: Mahalanobis distance, kernel density, empirical CDF
//! - `filters`: Kalman filters for state estimation
//! - `regime`: Markov switching, HMM
//! - `data`: Data quality checks and validation
//! - `matrix`: Linear algebra operations (Cholesky, Eigen, SVD, etc.)
//! - `options`: Black-Scholes pricing, Greeks, implied volatility
//! - `risk`: VaR, Sharpe ratio, drawdown, Kelly criterion
//! - `ta`: Technical indicators (MACD, RSI, Bollinger Bands, etc.)
//!
//! ## Performance Targets (c6i.xlarge)
//!
//! - Rolling stats: ~2.5ns per update
//! - KDE density eval: ~50ns
//! - Mahalanobis distance: ~20ns
//! - Markov update: ~15ns
//! - Batch (4-wide SIMD): ~8ns per series
//!
//! ## Example
//!
//! ```rust
//! use algotrading::prelude::*;
//!
//! // Stack-allocated, 300-sample window
//! let mut stats = RollingStats::<f64, 300>::new();
//! let (mean, std) = stats.update(0.001);
//!
//! // Regime detection
//! let mut regime = MarkovSwitching::spy_default();
//! regime.update(mean);
//! 
//! if regime.is_volatile(0.8) {
//!     println!("Market is volatile!");
//! }
//! ```

#![cfg_attr(feature = "simd", feature(portable_simd))]

pub mod numeric;
pub mod core;
pub mod stats;
pub mod probability;
pub mod filters;
pub mod regime;
pub mod data;
pub mod matrix;
pub mod options;
pub mod risk;
pub mod ta;


/// Common imports
pub mod prelude {
    // Numeric traits
    pub use crate::numeric::Numeric;

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    pub use crate::numeric::{f64x4, f64x8};

    // Core primitives
    pub use crate::core::RollingBuffer;

    // Stats - clean API with default type parameters
    pub use crate::stats::{
        RollingStats,
        RollingVolatility,
    };

    // Probability
    pub use crate::probability::{
        Mahalanobis,
        KernelDensity,
        Histogram,
        EmpiricalCDF,
    };

    // Regime Detection
    pub use crate::regime::{
        MarkovSwitching,
        RegimeState,
        HiddenMarkov,
        BullBearHMM,
    };

    // Filters
    pub use crate::filters::{
        KalmanFilter1D,
        KalmanFilterND,
    };

    // Data Quality
    pub use crate::data::{
        FeedDiscrepancy,
        StalenessDetector,
        JumpDetector,
        SequenceMonitor,
        QuoteQuality,
    };

    // Matrix Operations
    pub use crate::matrix::{
        Cholesky,
        Eigen,
        QRDecomposition,
        SVD,
        LUDecomposition,
        LedoitWolfEstimator,
        EWMACovarianceMatrix,
    };

    // Options Pricing
    pub use crate::options::{
        black_scholes_call,
        black_scholes_put,
        delta_call,
        delta_put,
        gamma,
        vega,
        theta_call,
        theta_put,
        implied_volatility_newton,
        SVIParameters,
    };

    // Risk Metrics
    pub use crate::risk::{
        RollingVaR,
        ParametricVaR,
        Kelly,
        RollingSharpe,
        RollingSortino,
        MaxDrawdown,
    };

    // Technical Analysis
    pub use crate::ta::{
        EWMA,
        MACD,
        RSI,
        BollingerBands,
        ATR,
        RollingCorrelation,
        Stochastic,
    };

    #[cfg(target_arch = "x86_64")]
    pub use crate::stats::RollingStatsBatch4;
}
