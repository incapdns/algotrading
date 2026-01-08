use criterion::{black_box, criterion_group, criterion_main, Criterion};
use algotrading::stats::*;

// Helper to generate sample data
fn generate_sample_data(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64 * 0.01).sin()).collect()
}

fn bench_stats_scalar_vs_simd(c: &mut Criterion) {
    let sizes = [100, 1000, 10000];

    for &size in &sizes {
        let data = generate_sample_data(size);

        // Mean benchmarks
        let mut group = c.benchmark_group(format!("mean_{}", size));

        group.bench_function("scalar", |b| {
            b.iter(|| {
                black_box(mean::<f64>(black_box(&data)))
            })
        });

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            use algotrading::numeric::f64x4;
            group.bench_function("simd_f64x4", |b| {
                b.iter(|| {
                    black_box(mean::<f64x4>(black_box(&data)))
                })
            });

            use algotrading::numeric::f64x8;
            group.bench_function("simd_f64x8", |b| {
                b.iter(|| {
                    black_box(mean::<f64x8>(black_box(&data)))
                })
            });
        }

        group.finish();
    }
}

fn bench_variance_scalar_vs_simd(c: &mut Criterion) {
    let sizes = [100, 1000, 10000];

    for &size in &sizes {
        let data = generate_sample_data(size);

        let mut group = c.benchmark_group(format!("variance_{}", size));

        group.bench_function("scalar", |b| {
            b.iter(|| {
                black_box(variance::<f64>(black_box(&data)))
            })
        });

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            use algotrading::numeric::f64x4;
            group.bench_function("simd_f64x4", |b| {
                b.iter(|| {
                    black_box(variance::<f64x4>(black_box(&data)))
                })
            });

            use algotrading::numeric::f64x8;
            group.bench_function("simd_f64x8", |b| {
                b.iter(|| {
                    black_box(variance::<f64x8>(black_box(&data)))
                })
            });
        }

        group.finish();
    }
}

fn bench_stddev_scalar_vs_simd(c: &mut Criterion) {
    let sizes = [100, 1000, 10000];

    for &size in &sizes {
        let data = generate_sample_data(size);

        let mut group = c.benchmark_group(format!("stddev_{}", size));

        group.bench_function("scalar", |b| {
            b.iter(|| {
                black_box(stddev::<f64>(black_box(&data)))
            })
        });

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            use algotrading::numeric::f64x4;
            group.bench_function("simd_f64x4", |b| {
                b.iter(|| {
                    black_box(stddev::<f64x4>(black_box(&data)))
                })
            });

            use algotrading::numeric::f64x8;
            group.bench_function("simd_f64x8", |b| {
                b.iter(|| {
                    black_box(stddev::<f64x8>(black_box(&data)))
                })
            });
        }

        group.finish();
    }
}

fn bench_skewness_kurtosis(c: &mut Criterion) {
    let data = generate_sample_data(1000);

    c.bench_function("skewness_1000", |b| {
        b.iter(|| {
            black_box(skewness::<f64>(black_box(&data)))
        })
    });

    c.bench_function("kurtosis_1000", |b| {
        b.iter(|| {
            black_box(kurtosis::<f64>(black_box(&data)))
        })
    });

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        use algotrading::numeric::f64x8;

        c.bench_function("skewness_1000_simd", |b| {
            b.iter(|| {
                black_box(skewness::<f64x8>(black_box(&data)))
            })
        });

        c.bench_function("kurtosis_1000_simd", |b| {
            b.iter(|| {
                black_box(kurtosis::<f64x8>(black_box(&data)))
            })
        });
    }
}

fn bench_correlation_covariance(c: &mut Criterion) {
    let data_x = generate_sample_data(1000);
    let data_y: Vec<f64> = data_x.iter().map(|&x| x * 0.8 + 0.1).collect();

    c.bench_function("covariance_1000", |b| {
        b.iter(|| {
            black_box(covariance::<f64>(black_box(&data_x), black_box(&data_y)))
        })
    });

    c.bench_function("correlation_1000", |b| {
        b.iter(|| {
            black_box(correlation::<f64>(black_box(&data_x), black_box(&data_y)))
        })
    });

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        use algotrading::numeric::f64x8;

        c.bench_function("covariance_1000_simd", |b| {
            b.iter(|| {
                black_box(covariance::<f64x8>(black_box(&data_x), black_box(&data_y)))
            })
        });

        c.bench_function("correlation_1000_simd", |b| {
            b.iter(|| {
                black_box(correlation::<f64x8>(black_box(&data_x), black_box(&data_y)))
            })
        });
    }
}

fn bench_minmax(c: &mut Criterion) {
    let data = generate_sample_data(1000);

    c.bench_function("minmax_1000", |b| {
        b.iter(|| {
            black_box(minmax::<f64>(black_box(&data)))
        })
    });

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        use algotrading::numeric::f64x8;

        c.bench_function("minmax_1000_simd", |b| {
            b.iter(|| {
                black_box(minmax::<f64x8>(black_box(&data)))
            })
        });
    }
}

criterion_group!(
    benches,
    bench_stats_scalar_vs_simd,
    bench_variance_scalar_vs_simd,
    bench_stddev_scalar_vs_simd,
    bench_skewness_kurtosis,
    bench_correlation_covariance,
    bench_minmax
);
criterion_main!(benches);
