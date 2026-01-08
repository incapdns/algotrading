use criterion::{black_box, criterion_group, criterion_main, Criterion};
use algotrading::matrix::{Cholesky, Eigen, quadratic_form};

fn bench_cholesky_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("cholesky");

    // 2x2 covariance matrix
    let cov_2x2 = [
        [1.0, 0.5],
        [0.5, 1.0],
    ];

    group.bench_function("2x2", |b| {
        b.iter(|| {
            black_box(Cholesky::decompose(black_box(&cov_2x2)).unwrap())
        })
    });

    // 4x4 covariance matrix
    let cov_4x4 = [
        [1.0, 0.5, 0.3, 0.2],
        [0.5, 1.0, 0.4, 0.3],
        [0.3, 0.4, 1.0, 0.5],
        [0.2, 0.3, 0.5, 1.0],
    ];

    group.bench_function("4x4", |b| {
        b.iter(|| {
            black_box(Cholesky::decompose(black_box(&cov_4x4)).unwrap())
        })
    });

    // 8x8 covariance matrix (larger portfolio)
    let mut cov_8x8 = [[0.0; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            if i == j {
                cov_8x8[i][j] = 1.0;
            } else {
                cov_8x8[i][j] = 0.3 / (1.0 + (i as f64 - j as f64).abs());
            }
        }
    }

    group.bench_function("8x8", |b| {
        b.iter(|| {
            black_box(Cholesky::decompose(black_box(&cov_8x8)).unwrap())
        })
    });

    group.finish();
}

fn bench_portfolio_variance(c: &mut Criterion) {
    let cov_4x4 = [
        [0.04, 0.01, 0.008, 0.006],
        [0.01, 0.09, 0.012, 0.009],
        [0.008, 0.012, 0.16, 0.015],
        [0.006, 0.009, 0.015, 0.25],
    ];

    let weights = [0.25, 0.25, 0.25, 0.25];

    c.bench_function("portfolio_variance_4x4", |b| {
        b.iter(|| {
            let chol = Cholesky::decompose(&cov_4x4).unwrap();
            black_box(chol.portfolio_variance(black_box(&weights)))
        })
    });
}

fn bench_correlated_sampling(c: &mut Criterion) {
    let corr_4x4 = [
        [1.0, 0.8, 0.6, 0.4],
        [0.8, 1.0, 0.7, 0.5],
        [0.6, 0.7, 1.0, 0.6],
        [0.4, 0.5, 0.6, 1.0],
    ];

    let chol = Cholesky::decompose(&corr_4x4).unwrap();
    let uncorrelated = [0.5, -0.3, 0.8, -0.2];

    c.bench_function("correlate_samples_4x4", |b| {
        b.iter(|| {
            black_box(chol.correlate(black_box(&uncorrelated)))
        })
    });
}

fn bench_eigenvalue_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("eigen");

    // 2x2 matrix
    let mat_2x2 = [
        [2.0, 1.0],
        [1.0, 2.0],
    ];

    group.bench_function("2x2", |b| {
        b.iter(|| {
            black_box(Eigen::decompose_symmetric(black_box(&mat_2x2)).unwrap())
        })
    });

    // 4x4 matrix
    let mat_4x4 = [
        [4.0, 1.0, 0.5, 0.3],
        [1.0, 3.0, 0.4, 0.2],
        [0.5, 0.4, 2.0, 0.3],
        [0.3, 0.2, 0.3, 1.0],
    ];

    group.bench_function("4x4", |b| {
        b.iter(|| {
            black_box(Eigen::decompose_symmetric(black_box(&mat_4x4)).unwrap())
        })
    });

    group.finish();
}

fn bench_quadratic_form(c: &mut Criterion) {
    let mat_4x4 = [
        [1.0, 0.2, 0.1, 0.05],
        [0.2, 1.0, 0.15, 0.1],
        [0.1, 0.15, 1.0, 0.2],
        [0.05, 0.1, 0.2, 1.0],
    ];

    let vec = [1.0, 2.0, 3.0, 4.0];

    c.bench_function("quadratic_form_4x4", |b| {
        b.iter(|| {
            black_box(quadratic_form(black_box(&mat_4x4), black_box(&vec)))
        })
    });
}

fn bench_covariance_estimation(c: &mut Criterion) {
    use algotrading::matrix::estimation::EWMACovarianceMatrix;

    let mut ewma_2x2 = EWMACovarianceMatrix::<2>::new(0.94);
    let returns = [0.01, -0.02];

    c.bench_function("ewma_covariance_2x2_update", |b| {
        b.iter(|| {
            black_box(ewma_2x2.update(black_box(&returns)))
        })
    });

    let mut ewma_4x4 = EWMACovarianceMatrix::<4>::new(0.94);
    let returns_4 = [0.01, -0.02, 0.015, -0.008];

    c.bench_function("ewma_covariance_4x4_update", |b| {
        b.iter(|| {
            black_box(ewma_4x4.update(black_box(&returns_4)))
        })
    });
}

criterion_group!(
    benches,
    bench_cholesky_decomposition,
    bench_portfolio_variance,
    bench_correlated_sampling,
    bench_eigenvalue_decomposition,
    bench_quadratic_form,
    bench_covariance_estimation
);
criterion_main!(benches);
