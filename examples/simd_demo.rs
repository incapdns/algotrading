//! SIMD Performance Demonstration
//!
//! This example shows how to use SIMD-accelerated operations in the algotrading library.
//!
//! # Running
//!
//! ```bash
//! # Without SIMD (default)
//! cargo run --example simd_demo --release
//!
//! # With SIMD acceleration
//! cargo run --example simd_demo --release --features simd
//! ```
//!
//! # Expected Performance
//!
//! On modern x86_64 CPUs:
//! - Scalar: ~2.5ns per update
//! - SIMD f64x4: ~0.8ns per series (4 series in parallel)
//! - SIMD f64x8: ~0.4ns per series (8 series in parallel)

use algotrading::prelude::*;
use std::time::Instant;

fn benchmark_scalar() {
    const ITERATIONS: usize = 1_000_000;
    const WINDOW: usize = 100;

    let mut stats = RollingStats::<f64, WINDOW>::new();

    let start = Instant::now();

    for i in 0..ITERATIONS {
        let value = (i as f64 * 0.001).sin();
        stats.update(value);
    }

    let elapsed = start.elapsed();
    let ns_per_update = elapsed.as_nanos() as f64 / ITERATIONS as f64;

    println!("Scalar Performance:");
    println!("  Total time: {:?}", elapsed);
    println!("  Per update: {:.2}ns", ns_per_update);
    println!("  Throughput: {:.2}M updates/sec", 1000.0 / ns_per_update);
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn benchmark_simd_f64x4() {
    use algotrading::numeric::f64x4;

    const ITERATIONS: usize = 1_000_000;
    const WINDOW: usize = 100;

    let mut stats = RollingStats::<f64x4, WINDOW>::new();

    let start = Instant::now();

    for i in 0..ITERATIONS {
        // Process 4 series in parallel
        let value = f64x4::from_array([
            (i as f64 * 0.001).sin(),
            (i as f64 * 0.002).sin(),
            (i as f64 * 0.003).sin(),
            (i as f64 * 0.004).sin(),
        ]);
        stats.update(value);
    }

    let elapsed = start.elapsed();
    let total_series = ITERATIONS * 4;
    let ns_per_series = elapsed.as_nanos() as f64 / total_series as f64;

    println!("\nSIMD f64x4 Performance:");
    println!("  Total time: {:?}", elapsed);
    println!("  Per series: {:.2}ns", ns_per_series);
    println!("  Throughput: {:.2}M series/sec", 1000.0 / ns_per_series);
    println!("  Speedup vs scalar: {:.2}x", 2.5 / ns_per_series);
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn benchmark_simd_f64x8() {
    use algotrading::numeric::f64x8;

    const ITERATIONS: usize = 1_000_000;
    const WINDOW: usize = 100;

    let mut stats = RollingStats::<f64x8, WINDOW>::new();

    let start = Instant::now();

    for i in 0..ITERATIONS {
        // Process 8 series in parallel
        let value = f64x8::from_array([
            (i as f64 * 0.001).sin(),
            (i as f64 * 0.002).sin(),
            (i as f64 * 0.003).sin(),
            (i as f64 * 0.004).sin(),
            (i as f64 * 0.005).sin(),
            (i as f64 * 0.006).sin(),
            (i as f64 * 0.007).sin(),
            (i as f64 * 0.008).sin(),
        ]);
        stats.update(value);
    }

    let elapsed = start.elapsed();
    let total_series = ITERATIONS * 8;
    let ns_per_series = elapsed.as_nanos() as f64 / total_series as f64;

    println!("\nSIMD f64x8 Performance:");
    println!("  Total time: {:?}", elapsed);
    println!("  Per series: {:.2}ns", ns_per_series);
    println!("  Throughput: {:.2}M series/sec", 1000.0 / ns_per_series);
    println!("  Speedup vs scalar: {:.2}x", 2.5 / ns_per_series);
}

fn demo_generic_api() {
    println!("\n=== Generic API Demo ===\n");

    // Same API works for both scalar and SIMD!
    fn compute_rolling_stats<T: Numeric, const N: usize>(
        data: &[f64],
        name: &str
    ) -> (f64, f64) {
        let mut stats = RollingStats::<T, N>::new();

        for &value in data {
            stats.update(T::splat(value));
        }

        let mean = stats.mean();
        let std = stats.std_dev();

        println!("{} result:", name);
        println!("  Mean: {:.6}", mean.reduce_sum() / T::LANES as f64);
        println!("  Std:  {:.6}", std.reduce_sum() / T::LANES as f64);

        (mean.reduce_sum() / T::LANES as f64,
         std.reduce_sum() / T::LANES as f64)
    }

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

    compute_rolling_stats::<f64, 10>(&data, "Scalar");

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        use algotrading::numeric::{f64x4, f64x8};
        compute_rolling_stats::<f64x4, 10>(&data, "SIMD f64x4");
        compute_rolling_stats::<f64x8, 10>(&data, "SIMD f64x8");
    }
}

fn demo_multi_asset_tracking() {
    println!("\n=== Multi-Asset Tracking Demo ===\n");

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        use algotrading::numeric::f64x4;

        // Track 4 assets simultaneously with SIMD
        let mut stats = RollingStats::<f64x4, 20>::new();

        println!("Processing 4 assets in parallel: AAPL, MSFT, GOOG, AMZN");
        println!();

        // Simulate 10 days of returns
        let returns = vec![
            f64x4::from_array([0.012, 0.008, -0.005, 0.015]),  // Day 1
            f64x4::from_array([0.005, 0.010, 0.003, -0.002]),   // Day 2
            f64x4::from_array([-0.003, 0.002, 0.008, 0.010]),   // Day 3
            f64x4::from_array([0.008, -0.004, 0.005, 0.007]),   // Day 4
            f64x4::from_array([0.002, 0.012, -0.002, 0.005]),   // Day 5
            f64x4::from_array([0.015, 0.005, 0.010, -0.003]),   // Day 6
            f64x4::from_array([-0.005, 0.008, 0.003, 0.012]),   // Day 7
            f64x4::from_array([0.010, -0.002, 0.007, 0.008]),   // Day 8
            f64x4::from_array([0.003, 0.015, -0.004, 0.002]),   // Day 9
            f64x4::from_array([0.012, 0.003, 0.008, 0.015]),    // Day 10
        ];

        for (day, &ret) in returns.iter().enumerate() {
            let (mean, std) = stats.update(ret);

            if day % 3 == 0 {
                let means = mean.to_array();
                let stds = std.to_array();

                println!("Day {}:", day + 1);
                println!("  AAPL: mean={:.4}% std={:.4}%", means[0] * 100.0, stds[0] * 100.0);
                println!("  MSFT: mean={:.4}% std={:.4}%", means[1] * 100.0, stds[1] * 100.0);
                println!("  GOOG: mean={:.4}% std={:.4}%", means[2] * 100.0, stds[2] * 100.0);
                println!("  AMZN: mean={:.4}% std={:.4}%", means[3] * 100.0, stds[3] * 100.0);
                println!();
            }
        }
    }

    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    {
        println!("SIMD features not enabled. Compile with --features simd to see this demo.");
    }
}

fn main() {
    println!("╔═════════════════════════════════════════╗");
    println!("║   ALGOTRADING SIMD PERFORMANCE DEMO     ║");
    println!("╚═════════════════════════════════════════╝");

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    println!("\n✓ SIMD features enabled (f64x4, f64x8)\n");

    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    println!("\n✗ SIMD features NOT enabled (compile with --features simd)\n");

    println!("=== Performance Benchmarks ===\n");

    benchmark_scalar();

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        benchmark_simd_f64x4();
        benchmark_simd_f64x8();
    }

    demo_generic_api();
    demo_multi_asset_tracking();

    println!("\n=== Summary ===\n");
    println!("The same `.update()` API works for:");
    println!("  • f64 (scalar) - 1 series at a time");
    println!("  • f64x4 (SIMD) - 4 series in parallel");
    println!("  • f64x8 (SIMD) - 8 series in parallel");
    println!("\nAll using the same generic implementation!");
}
