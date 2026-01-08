//! Technical indicators with optional SIMD acceleration
//!
//! This module provides technical analysis indicators that work with both
//! scalar (f64) and SIMD types (f64x4, f64x8) using clean default type parameters.
//!
//! # Examples
//!
//! ```
//! use algotrading::ta::{EWMA, MACD, RSI};
//!
//! // Scalar (default) - clean API
//! let mut ema = EWMA::from_period(20);
//! let value = ema.update(100.0);
//!
//! let mut macd = MACD::standard();
//! let (line, signal, hist) = macd.update(100.0);
//!
//! // SIMD (explicit when needed)
//! #[cfg(all(feature = "simd", target_arch = "x86_64"))]
//! {
//!     use algotrading::numeric::f64x4;
//!     let mut ema = EWMA::<f64x4>::from_period(20);
//!     let values = f64x4::from_array([100.0, 101.0, 99.0, 102.0]);
//!     let result = ema.update(values);
//! }
//! ```

pub mod indicators;
pub mod clean;

// Re-export the clean implementations
pub use indicators::*;

