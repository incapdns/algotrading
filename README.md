# Algotrading

A high-performance algorithmic trading library for real-time market systems.

Algotrading provides low-latency statistical and econometric primitives optimized for quantitative trading strategies. It is:

* **Fast**: SIMD-accelerated operations deliver nanosecond latencies with 4-8x speedups on modern hardware.

* **No Dependencies**: Pure Rust implementation with zero external dependencies for easy integration into trading systems.

* **Zero-cost**: Stack-allocated, const-generic data structures with no heap allocations in hot paths.

* **Flexible**: Generic over scalar and SIMD types (f64, f64x4, f64x8) with a clean, unified API.

[![Crates.io][crates-badge]][crates-url]
[![MIT licensed][mit-badge]][mit-url]
[![Documentation][docs-badge]][docs-url]
[![Build Status][build-badge]][build-url]

[crates-badge]: https://img.shields.io/crates/v/algotrading.svg
[crates-url]: https://crates.io/crates/algotrading
[mit-badge]: https://img.shields.io/badge/license-MIT-blue.svg
[mit-url]: https://github.com/ndavidson19/algotrading/blob/main/LICENSE
[docs-badge]: https://docs.rs/algotrading/badge.svg
[docs-url]: https://docs.rs/algotrading
[build-badge]: https://github.com/ndavidson19/algotrading/actions/workflows/rust.yml/badge.svg
[build-url]: https://github.com/ndavidson19/algotrading/actions/workflows/rust.yml


[Documentation](https://docs.rs/algotrading) | [API Reference](https://docs.rs/algotrading/latest/algotrading) | [Examples](examples/) | [Performance Guide](docs/PERFORMANCE.md)

## Overview

Algotrading enables you to build high-frequency and medium-frequency trading strategies with:

- **Statistical functions** — Mean, variance, correlation, skewness, kurtosis with SIMD
- **Probability distributions** — Normal, Student-T, Log-Normal with fast parametric fitting
- **Rolling statistics** — Constant-time O(1) updates with compile-time window sizes
- **Technical indicators** — MACD, RSI, Bollinger Bands, EWMA, ATR
- **Options pricing** — Black-Scholes, Greeks (Delta, Gamma, Vega, Theta), implied volatility
- **Risk metrics** — VaR, Sharpe ratio, max drawdown, Kelly criterion
- **Matrix operations** — Cholesky, eigenvalue decomposition, portfolio optimization
- **Regime detection** — Markov switching models, Hidden Markov Models
- **Data quality** — Real-time anomaly detection, staleness monitoring

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
algotrading = "0.1.0-alpha.2"
```

### Example: Building a Trading Signal

```rust
use algotrading::prelude::*;

fn main() {
    // Rolling statistics with 100-sample window
    let mut stats = RollingStats::<f64, 100>::new();

    // Technical indicators
    let mut rsi = RSI::standard();
    let mut macd = MACD::standard();

    // Regime detection
    let mut regime = MarkovSwitching::spy_default();

    for price in price_stream {
        // Update indicators
        let (mean, std) = stats.update(price);
        let rsi_val = rsi.update(price);
        let (macd_line, signal, _) = macd.update(price);

        // Calculate return and update regime
        let ret = (price - prev_price) / prev_price;
        regime.update(ret);

        // Generate signal
        if macd_line > signal && rsi_val < 70.0 && regime.current_state() == 1 {
            println!("BUY signal at {}", price);
        }
    }
}
```

### SIMD Acceleration

Process multiple assets in parallel with 4-8x speedup:

```rust
use algotrading::prelude::*;
use algotrading::numeric::f64x8;

// Process 8 assets simultaneously
let mut stats = RollingStats::<f64x8, 100>::new();
let prices = f64x8::from_array([100.0, 101.0, 99.0, 102.0, 98.0, 103.0, 97.0, 104.0]);
let (means, stds) = stats.update(prices);
```

See [examples/](examples/) for complete trading strategy implementations.

## Features

Enable SIMD acceleration for 4-8x speedup on modern hardware:

```toml
[dependencies]
algotrading = { version = "0.1", features = ["simd"] }
```

## Performance

Target latencies for trading operations:

| Operation | Latency | Notes |
|-----------|---------|-------|
| Rolling stats update | ~10ns | O(1) constant time |
| Mean (10K samples, SIMD) | ~8μs | 5x faster than scalar |
| Distribution fitting (1K samples) | ~2μs | With SIMD acceleration |
| Cholesky decomposition 4×4 | ~300ns | Portfolio variance |
| Options Greeks | ~50ns | Delta, Gamma, Vega |

See the [Performance Guide](docs/PERFORMANCE.md) for comprehensive benchmarks and optimization tips.

## Documentation

- **[API Documentation](https://docs.rs/algotrading)** — Complete API reference
- **[Performance Guide](docs/PERFORMANCE.md)** — Benchmarks and optimization
- **[Examples](examples/)** — Real-world trading strategies
- **[Module Documentation](docs/)** — Detailed module guides

## Core Modules

| Module | Description |
|--------|-------------|
| **`stats`** | Core statistics (mean, variance, correlation) + rolling statistics with SIMD |
| **`probability`** | Distributions (Normal, Student-T, Log-Normal), KDE, Monte Carlo |
| **`ta`** | Technical indicators (MACD, RSI, Bollinger Bands, EWMA, ATR) |
| **`options`** | Black-Scholes pricing, Greeks, implied volatility, local volatility |
| **`risk`** | VaR, Sharpe ratio, max drawdown, Kelly criterion |
| **`matrix`** | Cholesky, eigenvalue decomposition, portfolio optimization |
| **`regime`** | Markov switching, Hidden Markov Models |
| **`filters`** | Kalman filtering for state estimation |
| **`data`** | Real-time data quality monitoring and anomaly detection |

See the [Module Documentation](docs/) for detailed guides.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Before submitting a PR:
- Run `cargo test` to ensure all tests pass
- Run `cargo clippy` to check for common mistakes
- Add benchmarks for performance-critical code
- Update documentation as needed

## Benchmarks

Run the benchmark suite:

```bash
cargo bench
```

Benchmark results are saved in `target/criterion/` with detailed HTML reports.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

Built by [Nicholas Davidson](https://github.com/ndavidson19) for the quantitative trading community.

Special thanks to all [contributors](https://github.com/ndavidson19/algotrading/graphs/contributors).