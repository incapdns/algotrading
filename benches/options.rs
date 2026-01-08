use criterion::{criterion_group, criterion_main, Criterion, black_box};
use rand::prelude::*;
use algotrading::prelude::*; // Replace with your actual crate name

fn bench_black_scholes(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);
    let n = 100_000;

    let spots: Vec<f64> = (0..n).map(|_| rng.gen_range(80.0..120.0)).collect();
    let strikes: Vec<f64> = (0..n).map(|_| rng.gen_range(90.0..110.0)).collect();
    let vols: Vec<f64> = (0..n).map(|_| rng.gen_range(0.1..0.4)).collect();
    let rates: Vec<f64> = (0..n).map(|_| rng.gen_range(0.01..0.05)).collect();
    let times: Vec<f64> = (0..n).map(|_| rng.gen_range(0.05..1.0)).collect();

    c.bench_function("black_scholes_call_100k", |b| {
        b.iter(|| {
            for i in 0..n {
                black_box(black_scholes_call(
                    spots[i],
                    strikes[i],
                    rates[i],
                    0.02,
                    vols[i],
                    times[i],
                ));
            }
        });
    });
}

fn bench_implied_vol(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(1337);
    let n = 10_000;

    let spots: Vec<f64> = (0..n).map(|_| rng.gen_range(90.0..110.0)).collect();
    let strikes: Vec<f64> = (0..n).map(|_| rng.gen_range(90.0..110.0)).collect();
    let vols: Vec<f64> = (0..n).map(|_| rng.gen_range(0.15..0.35)).collect();

    let market_prices: Vec<f64> = spots
        .iter()
        .zip(&strikes)
        .zip(&vols)
        .map(|((&s, &k), &v)| black_scholes_call(s, k, 0.05, 0.02, v, 0.5))
        .collect();

    c.bench_function("implied_vol_newton_10k", |b| {
        b.iter(|| {
            for i in 0..n {
                black_box(implied_volatility_newton(
                    market_prices[i],
                    spots[i],
                    strikes[i],
                    0.05,
                    0.02,
                    0.5,
                    true,
                ));
            }
        });
    });
}

fn bench_order_book_analysis(c: &mut Criterion) {
    let n = 1_000;
    let mut rng = StdRng::seed_from_u64(1234);

    // Simulate rolling VaR for real-time returns
    let mut var = RollingVaR::<256>::new();
    let mut returns: Vec<f64> = (0..n).map(|_| rng.gen_range(-0.02..0.02)).collect();

    c.bench_function("rolling_var_1k", |b| {
        b.iter(|| {
            for &r in &returns {
                black_box(var.update(r, 0.99));
            }
        });
    });

    // Rolling Sharpe ratio over live data
    let mut sharpe = RollingSharpe::<256>::new();
    returns = (0..n).map(|_| rng.gen_range(-0.01..0.01)).collect();

    c.bench_function("rolling_sharpe_1k", |b| {
        b.iter(|| {
            for &r in &returns {
                black_box(sharpe.update(r, 0.01));
            }
        });
    });
}

criterion_group!(
    benches,
    bench_black_scholes,
    bench_implied_vol,
    bench_order_book_analysis
);
criterion_main!(benches);
