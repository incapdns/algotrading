//! Stochastic processes for modeling asset dynamics
//!
//! This module provides implementations of common stochastic differential equations (SDEs)
//! used in quantitative finance for modeling asset prices, interest rates, and volatility.
//!
//! # Processes
//!
//! - **GBM (Geometric Brownian Motion)**: Stock prices, FX rates
//! - **Ornstein-Uhlenbeck**: Mean-reverting processes, interest rates, pairs trading
//!
//! # Trading Applications
//!
//! ## 1. Stock Price Simulation (GBM)
//!
//! Geometric Brownian Motion is the standard model for stock prices:
//!
//! ```
//! use algotrading::probability::stochastic::GBM;
//! use algotrading::probability::Process;
//!
//! // Model SPY with 10% annual return, 20% volatility
//! let gbm = GBM {
//!     mu: 0.10,               // 10% drift
//!     sigma: 0.20,            // 20% volatility
//!     dt: 1.0 / 252.0,        // Daily time step
//! };
//!
//! // Simulate one day forward with random shock
//! let current_price = 450.0;
//! let z = 0.5;  // Standard normal random variable
//! let next_price = gbm.step(current_price, z);
//! println!("Price tomorrow: ${:.2}", next_price);
//! ```
//!
//! ## 2. Mean Reversion (Ornstein-Uhlenbeck)
//!
//! Model mean-reverting processes like interest rates or pairs spreads:
//!
//! ```
//! use algotrading::probability::stochastic::OrnsteinUhlenbeck;
//! use algotrading::probability::Process;
//!
//! // Model spread between two cointegrated stocks
//! let ou = OrnsteinUhlenbeck {
//!     theta: 2.0,      // Speed of mean reversion (fast = 2.0)
//!     mu: 0.0,         // Long-term mean (spread equilibrium)
//!     sigma: 0.15,     // Volatility of spread
//!     dt: 1.0 / 252.0, // Daily
//! };
//!
//! // Current spread is 0.5 standard deviations above mean
//! let current_spread = 0.5;
//! let z = -0.3;  // Random shock
//! let next_spread = ou.step(current_spread, z);
//! println!("Spread tomorrow: {:.3}", next_spread);
//! // Expected to revert toward 0
//! ```
//!
//! ## 3. Monte Carlo with Processes
//!
//! Combine with Monte Carlo engine for path simulation:
//!
//! ```ignore
//! use algotrading::probability::montecarlo::MonteCarloEngine;
//! use algotrading::probability::stochastic::GBM;
//! use algotrading::probability::{Process, MonteCarlo};
//!
//! let gbm = GBM {
//!     mu: 0.08,
//!     sigma: 0.25,
//!     dt: 1.0 / 252.0,
//! };
//!
//! let mc = MonteCarloEngine::new(Some(42));
//! let paths = mc.simulate(&gbm, 100.0, 252, 1000);
//!
//! // Analyze paths for option pricing, risk metrics, etc.
//! ```
//!
//! # Mathematical Background
//!
//! ## Geometric Brownian Motion (GBM)
//!
//! dS = μS dt + σS dW
//!
//! Where:
//! - S = asset price
//! - μ = drift (expected return)
//! - σ = volatility (standard deviation of returns)
//! - dW = Wiener process (Brownian motion)
//!
//! ## Ornstein-Uhlenbeck Process
//!
//! dx = θ(μ - x) dt + σ dW
//!
//! Where:
//! - x = current value
//! - θ = mean reversion speed
//! - μ = long-term mean
//! - σ = volatility

use crate::numeric::Numeric;
use crate::probability::Process;

/// Geometric Brownian Motion (GBM)
///
/// Standard model for stock prices and other assets that exhibit exponential growth
/// with random fluctuations. Prices cannot go negative, which matches real markets.
///
/// # Fields
///
/// - `mu`: Drift (expected return per unit time)
/// - `sigma`: Volatility (standard deviation of returns)
/// - `dt`: Time step size
///
/// # When to Use
///
/// - Stock price simulation
/// - FX rate modeling
/// - Commodity prices
/// - Index levels
///
/// **Not suitable for:**
/// - Interest rates (can't go negative, no mean reversion)
/// - Spreads in pairs trading (need mean reversion)
#[derive(Clone, Copy)]
pub struct GBM<T: Numeric> {
    /// Drift parameter (μ) - expected return
    pub mu: T,
    /// Volatility parameter (σ) - standard deviation
    pub sigma: T,
    /// Time step (dt)
    pub dt: T,
}

impl<T: Numeric> Process<T> for GBM<T> {
    fn step(&self, state: T, z: T) -> T {
        let drift = (self.mu - T::from_f64(0.5) * self.sigma * self.sigma) * self.dt;
        let shock = self.sigma * z * self.dt.sqrt();
        state * (drift + shock).exp()
    }
}

/// Ornstein-Uhlenbeck Process
///
/// Mean-reverting process that pulls toward a long-term mean. Essential for modeling
/// quantities that can't drift away indefinitely (unlike GBM).
///
/// # Fields
///
/// - `theta`: Mean reversion speed (higher = faster reversion)
/// - `mu`: Long-term mean (equilibrium level)
/// - `sigma`: Volatility around the mean
/// - `dt`: Time step size
///
/// # When to Use
///
/// - **Pairs trading**: Model spread between cointegrated stocks
/// - **Interest rates**: Short-term rates mean-revert to central bank targets
/// - **Volatility**: VIX and realized vol mean-revert
/// - **Credit spreads**: Corporate spreads revert to historical averages
///
/// # Mean Reversion Speed (theta)
///
/// - `theta = 0.5`: Slow reversion (half-life = 1.4 time units)
/// - `theta = 2.0`: Fast reversion (half-life = 0.35 time units)
/// - `theta = 10.0`: Very fast reversion (half-life = 0.07 time units)
///
/// Half-life formula: ln(2) / theta
#[derive(Clone, Copy)]
pub struct OrnsteinUhlenbeck<T: Numeric> {
    /// Mean reversion speed (θ)
    pub theta: T,
    /// Long-term mean (μ)
    pub mu: T,
    /// Volatility (σ)
    pub sigma: T,
    /// Time step (dt)
    pub dt: T,
}

impl<T: Numeric> Process<T> for OrnsteinUhlenbeck<T> {
    fn step(&self, x: T, z: T) -> T {
        let dx = self.theta * (self.mu - x) * self.dt + self.sigma * z * self.dt.sqrt();
        x + dx
    }
}
