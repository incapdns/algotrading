use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use algotrading::prelude::*;

fn bench_rolling_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("rolling_stats");
    
    for window in [100, 300, 1000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(window),
            &window,
            |b, &window| {
                match window {
                    100 => {
                        let mut stats = RollingStats::<f64, 100>::new();
                        b.iter(|| {
                            black_box(stats.update(black_box(0.001)))
                        });
                    }
                    300 => {
                        let mut stats = RollingStats::<f64, 300>::new();
                        b.iter(|| {
                            black_box(stats.update(black_box(0.001)))
                        });
                    }
                    1000 => {
                        let mut stats = RollingStats::<f64, 1000>::new();
                        b.iter(|| {
                            black_box(stats.update(black_box(0.001)))
                        });
                    }
                    _ => unreachable!(),
                }
            },
        );
    }
    
    group.finish();
}

fn bench_mahalanobis(c: &mut Criterion) {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    // Generate data with actual variance
    let data: Vec<[f64; 4]> = (0..1000)
        .map(|_| [
            rng.gen_range(-0.01..0.01),
            rng.gen_range(-0.01..0.01),
            rng.gen_range(-0.01..0.01),
            rng.gen_range(-0.01..0.01),
        ])
        .collect();
    
    let detector = Mahalanobis::<4>::train(&data).unwrap();
    let observation = [0.002, 0.1, 0.05, 0.015];
    
    c.bench_function("mahalanobis_distance", |b| {
        b.iter(|| {
            black_box(detector.distance_sq(black_box(&observation)))
        })
    });
}

criterion_group!(benches, bench_rolling_stats, bench_mahalanobis);
criterion_main!(benches);