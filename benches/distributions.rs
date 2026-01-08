use criterion::{black_box, criterion_group, criterion_main, Criterion};
use algotrading::probability::{Distribution, Normal, StudentT, LogNormal};

fn generate_returns(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64 * 0.0173).sin() * 0.02).collect()
}

fn bench_normal_distribution(c: &mut Criterion) {
    let dist = Normal::new(0.0, 1.0);

    c.bench_function("normal_pdf", |b| {
        b.iter(|| {
            black_box(dist.pdf(black_box(0.5)))
        })
    });

    c.bench_function("normal_cdf", |b| {
        b.iter(|| {
            black_box(dist.cdf(black_box(0.5)))
        })
    });

    // Test from_sample with SIMD
    let returns = generate_returns(1000);

    c.bench_function("normal_from_sample_scalar", |b| {
        b.iter(|| {
            black_box(Normal::from_sample::<f64>(black_box(&returns)))
        })
    });

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        use algotrading::numeric::f64x8;

        c.bench_function("normal_from_sample_simd", |b| {
            b.iter(|| {
                black_box(Normal::from_sample::<f64x8>(black_box(&returns)))
            })
        });
    }
}

fn bench_student_t_distribution(c: &mut Criterion) {
    let dist = StudentT::new(5.0);

    c.bench_function("student_t_pdf", |b| {
        b.iter(|| {
            black_box(dist.pdf(black_box(0.5)))
        })
    });

    c.bench_function("student_t_cdf", |b| {
        b.iter(|| {
            black_box(dist.cdf(black_box(0.5)))
        })
    });
}

fn bench_lognormal_distribution(c: &mut Criterion) {
    let dist = LogNormal::new(0.0, 0.25);

    c.bench_function("lognormal_pdf", |b| {
        b.iter(|| {
            black_box(dist.pdf(black_box(1.1)))
        })
    });

    c.bench_function("lognormal_cdf", |b| {
        b.iter(|| {
            black_box(dist.cdf(black_box(1.1)))
        })
    });

    // Test from_sample with SIMD
    let prices: Vec<f64> = (0..1000).map(|i| 100.0 * (1.0 + 0.0001 * i as f64)).collect();

    c.bench_function("lognormal_from_sample_scalar", |b| {
        b.iter(|| {
            black_box(LogNormal::from_sample::<f64>(black_box(&prices)))
        })
    });

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        use algotrading::numeric::f64x8;

        c.bench_function("lognormal_from_sample_simd", |b| {
            b.iter(|| {
                black_box(LogNormal::from_sample::<f64x8>(black_box(&prices)))
            })
        });
    }
}

fn bench_distribution_fitting(c: &mut Criterion) {
    let returns = generate_returns(10000);

    let mut group = c.benchmark_group("distribution_fitting");

    group.bench_function("normal_10k_samples_scalar", |b| {
        b.iter(|| {
            black_box(Normal::from_sample::<f64>(black_box(&returns)))
        })
    });

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        use algotrading::numeric::{f64x4, f64x8};

        group.bench_function("normal_10k_samples_f64x4", |b| {
            b.iter(|| {
                black_box(Normal::from_sample::<f64x4>(black_box(&returns)))
            })
        });

        group.bench_function("normal_10k_samples_f64x8", |b| {
            b.iter(|| {
                black_box(Normal::from_sample::<f64x8>(black_box(&returns)))
            })
        });
    }

    group.finish();
}

fn bench_var_calculation(c: &mut Criterion) {
    let returns = generate_returns(1000);
    let dist = Normal::from_sample::<f64>(&returns);

    c.bench_function("var_95_normal", |b| {
        b.iter(|| {
            // 95% VaR = mean + 1.645 * std
            black_box(dist.mean + dist.std * (-1.645))
        })
    });

    c.bench_function("var_99_normal", |b| {
        b.iter(|| {
            // 99% VaR = mean + 2.326 * std
            black_box(dist.mean + dist.std * (-2.326))
        })
    });
}

criterion_group!(
    benches,
    bench_normal_distribution,
    bench_student_t_distribution,
    bench_lognormal_distribution,
    bench_distribution_fitting,
    bench_var_calculation
);
criterion_main!(benches);
