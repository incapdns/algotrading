use criterion::{black_box, criterion_group, criterion_main, Criterion};
use algotrading::prelude::*;

fn bench_rolling_stats(c: &mut Criterion) {
    let mut stats = RollingStats::<f64, 300>::new();
    c.bench_function("rolling_stats_300", |b| {
        b.iter(|| black_box(stats.update(black_box(0.001))))
    });
}

fn bench_kde(c: &mut Criterion) {
    let samples: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
    let kde = KernelDensity::<100>::new(&samples);
    
    c.bench_function("kde_density", |b| {
        b.iter(|| black_box(kde.density(black_box(0.5))))
    });
}


fn bench_markov(c: &mut Criterion) {
    let mut regime = MarkovSwitching::spy_default();
    
    c.bench_function("markov_update", |b| {
        b.iter(|| black_box(regime.update(black_box(0.001))))
    });
}

criterion_group!(
    benches,
    bench_rolling_stats,
    bench_kde,
    bench_markov
);
criterion_main!(benches);
