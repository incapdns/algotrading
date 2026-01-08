//! Statistical computations with optional SIMD acceleration
//!
//! This module provides rolling statistics that work with both scalar (f64)
//! and SIMD types (f64x4, f64x8) using clean default type parameters.
//!
//! # Examples
//!
//! ```
//! use algotrading::stats::RollingStats;
//!
//! // Scalar (default) - clean API
//! let mut stats = RollingStats::<f64, 100>::new();
//! let (mean, std) = stats.update(42.0);
//!
//! // SIMD (explicit when needed)
//! #[cfg(all(feature = "simd", target_arch = "x86_64"))]
//! {
//!     use algotrading::numeric::f64x4;
//!     let mut stats = RollingStats::<f64x4, 100>::new();
//!     let values = f64x4::from_array([1.0, 2.0, 3.0, 4.0]);
//!     let (mean, std) = stats.update(values);
//! }
//! ```

mod rolling;
pub mod core;

// Re-export the clean implementations
pub use rolling::{RollingStats, RollingVolatility};

// Re-export core statistical functions
pub use core::{
    mean, variance, stddev, covariance, correlation,
    zscore, skewness, kurtosis, percentile, minmax, minmax_scale
};

// Keep generic module for backward compatibility but mark as deprecated
#[deprecated(since = "0.2.0", note = "Use RollingStats with default type parameters instead")]
pub mod generic;

#[cfg(target_arch = "x86_64")]
mod batch;

#[cfg(target_arch = "x86_64")]
pub use batch::RollingStatsBatch4;