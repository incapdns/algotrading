//! Rolling simulation for live path generation
//!
//! This module combines stochastic processes with online (rolling) simulation,
//! allowing you to generate price paths one step at a time in real-time systems.
//!
//! # Use Cases
//!
//! ## 1. Live Option Pricing
//!
//! Update option prices as market data arrives without re-simulating entire paths:
//!
//! ```ignore
//! use algotrading::simulation::rolling::RollingSim;
//! use algotrading::probability::stochastic::GBM;
//!
//! // Initialize GBM simulator
//! let gbm = GBM {
//!     mu: 0.10,
//!     sigma: 0.20,
//!     dt: 1.0 / 252.0,  // Daily
//! };
//!
//! let mut sim = RollingSim {
//!     process: gbm,
//!     state: 100.0,  // Initial price
//! };
//!
//! // As market ticks arrive, update simulation
//! let random_shock = 0.5;  // From RNG or market data
//! let next_price = sim.update(random_shock);
//! println!("Simulated price: ${:.2}", next_price);
//! ```
//!
//! ## 2. Real-Time Risk Scenario Generation
//!
//! Generate forward scenarios for stress testing:
//!
//! ```ignore
//! use algotrading::simulation::rolling::RollingSim;
//! use algotrading::probability::stochastic::OrnsteinUhlenbeck;
//!
//! // Mean-reverting spread simulation
//! let ou = OrnsteinUhlenbeck {
//!     theta: 2.0,      // Fast mean reversion
//!     mu: 0.0,         // Equilibrium at 0
//!     sigma: 0.15,
//!     dt: 1.0 / 252.0,
//! };
//!
//! let mut spread_sim = RollingSim {
//!     process: ou,
//!     state: 0.5,  // Current spread
//! };
//!
//! // Simulate forward for risk check
//! for _ in 0..10 {
//!     let z = rand::random::<f64>() - 0.5;  // Random shock
//!     let future_spread = spread_sim.update(z);
//!
//!     if future_spread.abs() > 2.0 {
//!         println!("Risk warning: Spread could hit {:.2}", future_spread);
//!     }
//! }
//! ```
//!
//! ## 3. Adaptive Monte Carlo
//!
//! Combine with rolling statistics to adapt simulations to changing market conditions:
//!
//! ```ignore
//! use algotrading::simulation::rolling::RollingSim;
//! use algotrading::probability::stochastic::GBM;
//! use algotrading::stats::RollingStats;
//!
//! // Volatility adapts based on recent realized vol
//! let mut realized_vol = RollingStats::<f64, 20>::new();
//!
//! let mut gbm = GBM {
//!     mu: 0.08,
//!     sigma: 0.20,  // Initial guess
//!     dt: 1.0 / 252.0,
//! };
//!
//! let mut sim = RollingSim {
//!     process: gbm,
//!     state: 100.0,
//! };
//!
//! // Update volatility adaptively
//! for price in market_prices {
//!     let (_, std) = realized_vol.update(price);
//!     sim.process.sigma = std;  // Adaptive volatility
//!
//!     let z = generate_random_shock();
//!     let simulated = sim.update(z);
//! }
//! ```
//!
//! # Benefits Over Batch Simulation
//!
//! | Feature | Batch Monte Carlo | Rolling Sim |
//! |---------|------------------|-------------|
//! | Memory | O(paths × steps) | O(1) |
//! | Latency | Generate all paths | One step at a time |
//! | Adaptivity | Fixed parameters | Can update parameters |
//! | Use case | Offline pricing | Live trading |
//!
//! # Performance
//!
//! - **Constant memory**: Only stores current state
//! - **Low latency**: Single update per tick
//! - **SIMD support**: Can simulate multiple assets in parallel using `Numeric` trait

use crate::numeric::Numeric;
use crate::probability::Process;
use crate::rolling::RollingStats;

/// Rolling simulator for stochastic processes
///
/// Maintains a single state and evolves it forward one step at a time,
/// perfect for live trading systems and real-time option pricing.
///
/// # Type Parameters
///
/// - `P`: Process type (GBM, Ornstein-Uhlenbeck, etc.)
/// - `T`: Numeric type (f64 for scalar, f64x4/f64x8 for SIMD)
///
/// # Fields
///
/// - `process`: The stochastic process model
/// - `state`: Current state (price, spread, etc.)
pub struct RollingSim<P, T>
where
    P: Process<T>,
    T: Numeric,
{
    pub process: P,
    pub state: T,
}

impl<P, T> RollingSim<P, T>
where
    P: Process<T>,
    T: Numeric,
{
    pub fn update(&mut self, z: T) -> T {
        self.state = self.process.step(self.state, z);
        self.state
    }
}

impl<P, T> RollingStats<T> for RollingSim<P, T>
where
    P: Process<T>,
    T: Numeric,
{
    fn update(&mut self, input: T) -> T {
        self.update(input)
    }

    fn value(&self) -> T {
        self.state
    }
}
