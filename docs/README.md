# Algotrading Library Documentation

High-performance quantitative trading primitives optimized for AWS EC2.

## Table of Contents

- [Overview](#overview)
- [Design Philosophy](#design-philosophy)
- [Modules](#modules)
  - [Stats](#stats)
  - [Probability](#probability)
  - [Filters](#filters)
  - [Regime](#regime)
  - [Data Quality](#data-quality)
  - [Matrix Operations](#matrix-operations)
  - [Options Pricing](#options-pricing)
  - [Risk Metrics](#risk-metrics)
  - [Technical Analysis](#technical-analysis)
- [Performance](#performance)
- [Examples](#examples)

## Overview

This library provides a comprehensive suite of quantitative trading tools including:

- Statistical analysis and probability distributions
- Kalman filtering and regime detection
- Data quality validation
- Linear algebra and matrix operations
- Options pricing and Greeks calculation
- Risk metrics (VaR, Sharpe, etc.)
- Technical indicators

## Design Philosophy

- **f64 only**: Financial precision is non-negotiable
- **Stack allocated**: Const generics, zero heap allocations where possible
- **SIMD optimized**: AVX2 baseline (all modern EC2 instances)
- **Cache-friendly**: 64-byte alignment, structure-of-arrays layout

## Modules

### Stats

High-performance statistical computations with optional SIMD acceleration.

**Core Functions (SIMD-enabled):**
- `mean`, `variance`, `stddev` - Basic statistics with 4-8x SIMD speedup
- `covariance`, `correlation` - Pairwise statistics
- `skewness`, `kurtosis` - Higher-order moments
- `minmax`, `percentile` - Extrema and quantiles

**Rolling Statistics:**
- `RollingStats<T, N>` - Rolling mean, variance, skewness, kurtosis
- `RollingVolatility` - Specialized volatility estimator
- Generic over scalar (f64) or SIMD types (f64x4, f64x8)

**Performance:**
- Core functions: ~10-50ns (scalar), ~3-15ns (SIMD f64x8)
- Rolling stats update: ~2.5ns per update

### Probability

Probability distributions, density estimation, and statistical modeling.

**Parametric Distributions:**
- `Normal` - Gaussian distribution with PDF/CDF
- `StudentT` - Student's t-distribution for heavy tails
- `LogNormal` - Log-normal distribution for asset prices
- All support `from_sample` with SIMD acceleration

**Nonparametric Methods:**
- `KernelDensity` - Adaptive KDE with Scott's rule
- `Histogram` - Fast binned distribution
- `EmpiricalCDF` - Quantile estimation
- `Mahalanobis` - Multivariate distance metric

**Performance:**
- Distribution fitting: ~1-5μs (1000 samples)
- PDF/CDF evaluation: ~10-50ns
- KDE density eval: ~50ns

### Filters

State-space filtering for signal estimation.

**Key Types:**
- `KalmanFilter1D` - Single-dimensional Kalman filter
- `KalmanFilterND<N>` - N-dimensional Kalman filter

### Regime

Market regime detection using hidden Markov models.

**Key Types:**
- `MarkovSwitching` - 2-state Markov regime switcher
- `HiddenMarkov<N>` - N-state HMM with Viterbi decoding
- `BullBearHMM` - Pre-configured bull/bear detector

**Performance:** Markov update ~15ns

### Data Quality

Real-time data validation and anomaly detection.

**Key Types:**
- `FeedDiscrepancy` - Cross-feed price validation
- `StalenessDetector` - Quote staleness monitoring
- `JumpDetector` - Price jump detection
- `SequenceMonitor` - Sequence gap detection
- `QuoteQuality` - Multi-factor quality scoring

**Use Cases:**
- Detecting crossed/locked markets
- Monitoring NBBO consistency
- Identifying wash trades
- Validating tick size compliance

See [data_quality.md](data_quality.md) for details.

### Matrix Operations

High-performance linear algebra for portfolio optimization.

**Key Types:**
- `Cholesky<N>` - Cholesky decomposition for SPD matrices
- `Eigen<N>` - Eigenvalue decomposition (Jacobi method)
- `QRDecomposition<N>` - QR factorization (Householder)
- `SVD<N>` - Singular value decomposition
- `LUDecomposition<N>` - LU factorization with pivoting
- `LedoitWolfEstimator<N>` - Shrinkage covariance estimation
- `EWMACovarianceMatrix<N>` - Online covariance tracking

**Operations:**
- Matrix-vector multiply (SIMD optimized)
- Quadratic forms
- Portfolio variance calculation
- Eigenvalue/eigenvector computation

See [matrix.md](matrix.md) for details.

### Options Pricing

Black-Scholes-Merton pricing with Greeks and volatility analytics.

**Pricing Functions:**
- `black_scholes_call` / `black_scholes_put`
- `implied_volatility_newton` - Fast IV calculation using Newton-Raphson
- `variance_swap_strike` - Fair strike from smile
- `vix_index_calculation` - VIX-style index computation

**Greeks:**
- First order: Delta, Vega, Theta, Rho
- Second order: Gamma, Vanna, Volga, Charm, Veta

**Volatility Analytics:**
- `SVIParameters` - Volatility smile parameterization
- `local_volatility_dupire` - Dupire's local vol
- `forward_volatility` - Forward vol from term structure
- `skew_index` - SKEW index calculation

**Exotic Options:**
- Digital options
- Power options
- Barrier options
- Cliquet options
- Corridor variance swaps

See [options.md](options.md) for details.

### Risk Metrics

Comprehensive risk measurement and position sizing.

**Value at Risk:**
- `RollingVaR<N>` - Historical VaR
- `ParametricVaR` - Cornish-Fisher VaR (accounts for skew/kurtosis)

**Performance Metrics:**
- `RollingSharpe<N>` - Annualized Sharpe ratio
- `RollingSortino<N>` - Downside deviation only
- `MaxDrawdown<N>` - Peak-to-trough tracking

**Position Sizing:**
- `Kelly::binary` - Kelly criterion for binary outcomes
- `Kelly::continuous` - Mean-variance Kelly
- `Kelly::half_kelly` - Conservative sizing

See [risk.md](risk.md) for details.

### Technical Analysis

Classical and modern technical indicators.

**Moving Averages:**
- `EWMA` - Exponential weighted moving average
- `MACD` - Moving average convergence/divergence

**Momentum:**
- `RSI` - Relative strength index
- `Stochastic<N>` - Stochastic oscillator

**Volatility:**
- `BollingerBands<N>` - Bollinger bands with %B
- `ATR` - Average true range

**Statistical:**
- `RollingCorrelation<N>` - Correlation between two series
- `EWMACovariance` - Online covariance estimation

See [technical_analysis.md](technical_analysis.md) for details.

## Performance

The library targets nanosecond to microsecond latencies for all core operations.

**Quick Reference:**

| Operation | Latency (Scalar) | Latency (SIMD) | Speedup |
|-----------|------------------|----------------|---------|
| Rolling stats update | ~10ns | ~10ns | N/A (O(1)) |
| Mean (10k samples) | ~40μs | ~8μs | 5x |
| Correlation (1k samples) | ~20μs | ~4μs | 5x |
| Distribution fitting (1k samples) | ~10μs | ~2μs | 5x |
| Cholesky decomposition 4×4 | ~300ns | ~300ns | N/A |
| KDE density eval | ~50ns | ~50ns | N/A |
| Mahalanobis distance | ~20ns | ~20ns | N/A |

For comprehensive benchmarks, optimization guidelines, and hardware recommendations, see [PERFORMANCE.md](PERFORMANCE.md).

## Examples

### Basic Usage

```rust
use algotrading::prelude::*;

// Stack-allocated, 300-sample rolling stats
let mut stats = RollingStats::<300>::new();
let (mean, std) = stats.update(0.001);

// Regime detection
let mut regime = MarkovSwitching::spy_default();
regime.update(mean);

if regime.is_volatile(0.8) {
    println!("Market is volatile!");
}
```

### Options Pricing

```rust
use algotrading::prelude::*;

// Price a call option
let spot = 100.0;
let strike = 105.0;
let rate = 0.05;
let div_yield = 0.02;
let vol = 0.25;
let time = 1.0; // 1 year

let price = black_scholes_call(spot, strike, rate, div_yield, vol, time);
let delta = delta_call(spot, strike, rate, div_yield, vol, time);
let gamma_val = gamma(spot, strike, rate, div_yield, vol, time);

println!("Call price: {:.4}", price);
println!("Delta: {:.4}", delta);
println!("Gamma: {:.6}", gamma_val);
```

### Risk Management

```rust
use algotrading::prelude::*;

// Track VaR
let mut var = RollingVaR::<252>::new();
let var_95 = var.update(daily_return, 0.95);

// Position sizing with Kelly
let win_prob = 0.55;
let win_loss_ratio = 1.5;
let kelly_fraction = Kelly::binary(win_prob, win_loss_ratio);

// Conservative: use half-Kelly
let position_size = Kelly::half_kelly(win_prob, win_loss_ratio);
```

### Technical Analysis

```rust
use algotrading::prelude::*;

// MACD
let mut macd = MACD::standard();
let (macd_line, signal, histogram) = macd.update(price);

// RSI
let mut rsi = RSI::standard();
let rsi_value = rsi.update(price);

// Bollinger Bands
let mut bb = BollingerBands::<20>::standard();
let (upper, middle, lower, percent_b) = bb.update(price);
```

### Data Quality

```rust
use algotrading::prelude::*;

// Monitor feed quality
let mut staleness = StalenessDetector::new(100.0); // 100ms threshold
let is_stale = staleness.check_update(timestamp, price);

// Detect price jumps
let mut jump_detector = JumpDetector::new(3.0); // 3-sigma threshold
let is_jump = jump_detector.check_price(price, current_vol);

// Validate quote quality
let quality = compute_quote_quality(
    bid, ask, bid_size, ask_size,
    time_since_update, volatility,
    reference_spread, reference_depth
);
```

## Module Documentation

For detailed documentation on each module, see:

- [Performance Guide](PERFORMANCE.md) - Comprehensive benchmarks and optimization tips
- [Data Quality](data_quality.md)
- [Matrix Operations](matrix.md)
- [Options Pricing](options.md)
- [Risk Metrics](risk.md)

## Testing

Run tests with:

```bash
cargo test
```

Build documentation:

```bash
cargo doc --no-deps --open
```

## License

See LICENSE file for details.
