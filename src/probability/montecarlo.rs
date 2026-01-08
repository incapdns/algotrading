//! Monte Carlo simulation for option pricing and risk analysis
//!
//! Monte Carlo methods are essential for pricing complex derivatives and
//! estimating risk metrics when analytical solutions don't exist.
//!
//! # Applications
//!
//! - **Path-dependent options**: Asian options, lookback options, barrier options
//! - **Portfolio VaR**: Simulate correlated asset returns
//! - **Scenario analysis**: Stress testing under different market conditions
//! - **Greeks estimation**: Numerical differentiation of simulated prices
//!
//! # Example: Simulating GBM for Stock Prices
//!
//! ```
//! use algotrading::probability::montecarlo::MonteCarloEngine;
//! use algotrading::probability::stochastic::GBM;
//! use algotrading::probability::{Process, MonteCarlo};
//!
//! // Geometric Brownian Motion parameters
//! let mu = 0.10;      // 10% drift
//! let sigma = 0.20;   // 20% volatility
//! let dt = 1.0 / 252.0;  // Daily steps
//!
//! let gbm = GBM {
//!     mu: mu,
//!     sigma: sigma,
//!     dt: dt,
//! };
//!
//! // Monte Carlo engine with fixed seed for reproducibility
//! let mc = MonteCarloEngine::new(Some(42));
//!
//! // Simulate 1000 paths over 252 steps (1 year of daily data)
//! let start_price = 100.0;
//! let paths = mc.simulate(&gbm, start_price, 252, 1000);
//!
//! // Analyze terminal prices
//! let terminal_prices: Vec<f64> = paths.iter()
//!     .map(|path| path.last().unwrap())
//!     .map(|&p| p)
//!     .collect();
//!
//! let avg_terminal = terminal_prices.iter().sum::<f64>() / terminal_prices.len() as f64;
//! println!("Average terminal price: ${:.2}", avg_terminal);
//! ```
//!
//! # Example: Asian Option Pricing
//!
//! Price an average-price Asian call option:
//!
//! ```ignore
//! use algotrading::probability::montecarlo::MonteCarloEngine;
//! use algotrading::probability::stochastic::GBM;
//! use algotrading::probability::{Process, MonteCarlo};
//!
//! let gbm = GBM {
//!     mu: 0.05,       // Risk-free rate
//!     sigma: 0.30,    // 30% vol
//!     dt: 1.0 / 12.0, // Monthly
//! };
//!
//! let mc = MonteCarloEngine::new(None);
//! let s0 = 100.0;
//! let strike = 105.0;
//! let paths = mc.simulate(&gbm, s0, 12, 10000);
//!
//! // Asian option payoff: max(avg_price - strike, 0)
//! let payoffs: Vec<f64> = paths.iter().map(|path| {
//!     let avg_price = path.iter().sum::<f64>() / path.len() as f64;
//!     (avg_price - strike).max(0.0)
//! }).collect();
//!
//! let discount = (-0.05_f64).exp();  // Discount at risk-free rate
//! let option_price = discount * payoffs.iter().sum::<f64>() / payoffs.len() as f64;
//! println!("Asian call option value: ${:.2}", option_price);
//! ```
//!
//! # Performance Notes
//!
//! - Uses deterministic seeding for reproducible results in backtesting
//! - Supports both scalar and SIMD types via the `Numeric` trait
//! - For SIMD: process multiple independent paths simultaneously

use crate::numeric::Numeric;
use crate::probability::Process;
use rand::prelude::*;

/// Monte Carlo simulation engine
///
/// Generates random paths for stochastic processes using pseudo-random number generation.
///
/// # Fields
///
/// - `seed`: Optional seed for reproducible simulations (useful for testing/backtesting)
pub struct MonteCarloEngine {
    pub seed: Option<u64>,
}

impl MonteCarloEngine {
    pub fn new(seed: Option<u64>) -> Self {
        Self { seed }
    }
}

impl<T: Numeric> super::MonteCarlo<T> for MonteCarloEngine {
    fn simulate<P: Process<T>>(
        &self,
        process: &P,
        start: T,
        steps: usize,
        paths: usize,
    ) -> Vec<Vec<T>> {
        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let mut all_paths = Vec::with_capacity(paths);
        for _ in 0..paths {
            let mut path = Vec::with_capacity(steps);
            let mut state = start;
            path.push(state);

            for _ in 1..steps {
                // draw standard normal z
                let z_scalar: f64 = rng.sample(rand_distr::StandardNormal);
                let z = T::splat(z_scalar);
                state = process.step(state, z);
                path.push(state);
            }
            all_paths.push(path);
        }
        all_paths
    }
}
