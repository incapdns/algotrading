# Matrix Operations Module

High-performance linear algebra for portfolio optimization and quantitative finance.

## Overview

The matrix module provides numerical linear algebra operations optimized for financial applications:
- Portfolio variance calculation
- Risk decomposition
- Factor models
- Covariance estimation

All operations use stack-allocated const-generic arrays for zero-allocation performance.

## Core Decompositions

### Cholesky Decomposition

Factorize symmetric positive-definite matrices: A = L L^T

```rust
use algotrading::matrix::*;

let cov = [
    [4.0, 2.0],
    [2.0, 3.0],
];

let chol = Cholesky::decompose(&cov)?;
let l = chol.lower();

// Portfolio variance calculation (much faster than full matrix multiply)
let weights = [0.6, 0.4];
let portfolio_var = chol.portfolio_variance(&weights);
```

**Use Cases:**
- Fast portfolio variance: ||L^T w||^2 instead of w^T Σ w
- Generate correlated random variates for Monte Carlo
- Solve linear systems

**Performance:** O(N^3) decomposition, O(N^2) variance calculation

### Eigenvalue Decomposition

Compute eigenvalues and eigenvectors of symmetric matrices.

```rust
let matrix = [
    [4.0, 1.0],
    [1.0, 3.0],
];

let eigen = Eigen::decompose_symmetric(&matrix)?;
let values = eigen.values(); // Sorted descending
let vectors = eigen.vectors();

// PCA projection
let data = [1.0, 2.0];
let projected = eigen.project(&data, 1); // Keep 1 component

// Explained variance
let ratios = eigen.explained_variance_ratio();
```

**Algorithm:** Jacobi method (good numerical stability)

**Use Cases:**
- Principal Component Analysis (PCA)
- Factor models
- Portfolio optimization
- Risk attribution

### QR Decomposition

Factorize any matrix: A = QR (Q orthogonal, R upper triangular)

```rust
let matrix = [
    [12.0, -51.0, 4.0],
    [6.0, 167.0, -68.0],
    [-4.0, 24.0, -41.0],
];

let qr = QRDecomposition::decompose(&matrix)?;

// Solve least squares: minimize ||Ax - b||²
let b = [1.0, 2.0, 3.0];
let x = qr.solve_least_squares(&b);
```

**Algorithm:** Householder reflections

**Use Cases:**
- Linear regression
- Least squares optimization
- Orthogonalization

### SVD (Singular Value Decomposition)

Factorize any matrix: A = U Σ V^T

```rust
let matrix = [
    [1.0, 2.0],
    [3.0, 4.0],
];

let svd = SVD::decompose(&matrix)?;
let singular_values = svd.singular_values();

// Compute pseudoinverse
let pinv = svd.pseudoinverse();

// Condition number (numerical stability measure)
let cond = svd.condition_number();
```

**Use Cases:**
- Matrix pseudoinverse
- Rank determination
- Stability analysis
- Latent semantic indexing

### LU Decomposition

Factorize with partial pivoting: PA = LU

```rust
let matrix = [
    [2.0, 1.0, 1.0],
    [4.0, 3.0, 3.0],
    [8.0, 7.0, 9.0],
];

let lu = LUDecomposition::decompose(&matrix)?;

// Solve Ax = b
let b = [1.0, 2.0, 3.0];
let x = lu.solve(&b);

// Compute determinant
let det = lu.determinant();
```

**Use Cases:**
- Solving linear systems
- Matrix inversion
- Determinant calculation

## Covariance Estimation

### Ledoit-Wolf Shrinkage

Improves covariance estimation with small samples.

```rust
let data = vec![
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.0],
    [4.0, 5.0],
];

let estimator = LedoitWolfEstimator::estimate(&data)?;
let shrunk_cov = estimator.shrunk_covariance();
```

**Theory:**
- Shrinks sample covariance toward structured target
- Optimal shrinkage intensity (minimize MSE)
- Better conditioned than sample covariance

**Formula:** Σ_shrunk = δ * Target + (1-δ) * Sample

### EWMA Covariance Matrix

Online covariance estimation without storing history.

```rust
let mut cov = EWMACovarianceMatrix::<3>::new(0.94);
// or
let mut cov = EWMACovarianceMatrix::<3>::from_halflife(20.0);

// Update with new returns
cov.update(&[0.01, -0.005, 0.002]);

// Get current estimates
let covariance = cov.covariance();
let correlation = cov.correlation();
```

**Use Cases:**
- Real-time risk monitoring
- High-frequency portfolio rebalancing
- Adaptive volatility targeting

### Woodbury Matrix Update

Fast inverse updates when adding low-rank matrices.

```rust
let a_inv = [[1.0, 0.0], [0.0, 1.0]];
let mut woodbury = WoodburyUpdate::new(a_inv);

// Update inverse when adding rank-1 matrix uv^T
let u = [1.0, 2.0];
let v = [3.0, 4.0];
woodbury.rank_one_update(&u, &v);

let updated_inv = woodbury.inverse();
```

**Formula:** (A + uv^T)^(-1) = A^(-1) - (A^(-1) u v^T A^(-1)) / (1 + v^T A^(-1) u)

**Use Cases:**
- Sequential covariance updates
- Kalman filter gain updates
- Online factor model updates

## Matrix Operations

### Basic Operations

```rust
use algotrading::matrix::*;

// Matrix-vector multiply
let result = matvec_multiply(&matrix, &vector);

// Quadratic form: x^T A x (portfolio variance)
let quad = quadratic_form(&matrix, &vector);

// Transpose
let transposed = transpose(&matrix);

// Matrix-matrix multiply
let product = matmul(&a, &b);

// Trace (sum of diagonal)
let tr = trace(&matrix);

// Frobenius norm
let norm = frobenius_norm(&matrix);
```

### SIMD-Optimized Operations

```rust
// Auto-detects AVX2 and uses fast path if available
let result = matvec_simd(&matrix, &vector);

// Blocked matrix multiply (cache-friendly)
let product = matmul_blocked(&a, &b);
```

### Power Iteration

Find dominant eigenvalue without full decomposition.

```rust
let (lambda, eigenvector) = dominant_eigenvalue(&matrix, max_iter: 100);

// Quick approximation given eigenvector guess
let lambda_approx = rayleigh_quotient(&matrix, &guess);
```

**Use Cases:**
- PageRank computation
- Power method for spectral analysis
- Quick condition number estimation

### Batch Operations

Process multiple vectors at once.

```rust
let vectors = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
let results = batch_matvec(&matrix, &vectors);
```

## Advanced Features

### Fast SPD Matrix Inversion

Symmetric positive-definite matrices via Cholesky.

```rust
let inv = invert_spd_cholesky(&cov_matrix)?;
```

**Performance:** ~2x faster than general Gaussian elimination

### Matrix Power

For symmetric matrices via eigendecomposition.

```rust
let matrix_sqrt = matrix_power_symmetric(&matrix, 0.5)?; // Square root
let matrix_squared = matrix_power_symmetric(&matrix, 2.0)?;
```

**Formula:** A^p = V Λ^p V^T

## Portfolio Applications

### Portfolio Variance

```rust
let returns_cov = [
    [0.04, 0.02, 0.01],
    [0.02, 0.09, 0.03],
    [0.01, 0.03, 0.16],
];

let weights = [0.5, 0.3, 0.2];

// Method 1: Direct (slower)
let variance = quadratic_form(&returns_cov, &weights);

// Method 2: Cholesky (faster)
let chol = Cholesky::decompose(&returns_cov)?;
let variance = chol.portfolio_variance(&weights);

let volatility = variance.sqrt();
```

### Risk Decomposition

```rust
// Decompose portfolio variance into independent factors
let chol = Cholesky::decompose(&cov)?;
let l = chol.lower();

// Each row of L represents a risk factor
// Factor contribution: (L * weights)_i^2
```

### Correlated Random Variates

For Monte Carlo simulation.

```rust
let chol = Cholesky::decompose(&target_covariance)?;

// Generate uncorrelated standard normals
let uncorrelated = [random_normal(), random_normal(), random_normal()];

// Transform to have target covariance
let correlated = chol.correlate(&uncorrelated);
```

### Minimum Variance Portfolio

```rust
// Find weights w that minimize w^T Σ w subject to w^T 1 = 1

let n = 3;
let cov = /* covariance matrix */;

// Using Lagrange multipliers:
// w = (Σ^(-1) * 1) / (1^T * Σ^(-1) * 1)

let inv_cov = invert_spd_cholesky(&cov)?;
let ones = [1.0; 3];
let numerator = matvec_multiply(&inv_cov, &ones);
let denominator = quadratic_form(&inv_cov, &ones);

let weights: Vec<f64> = numerator.iter()
    .map(|&x| x / denominator)
    .collect();
```

## Performance Tips

1. **Use Cholesky for SPD matrices** - Much faster than LU or general inversion
2. **Batch operations** - Process multiple vectors together when possible
3. **Const generics** - Sizes known at compile time = zero-cost abstractions
4. **SIMD** - Automatically used on x86_64 with AVX2
5. **Cache blocking** - Use `matmul_blocked` for large matrices

## Numerical Stability

### Condition Numbers

```rust
let svd = SVD::decompose(&matrix)?;
let cond = svd.condition_number();

if cond > 1e10 {
    println!("Warning: Matrix is ill-conditioned");
    // Consider regularization or shrinkage
}
```

### Shrinkage for Small Samples

```rust
// When T (observations) < N (assets):
// Sample covariance is singular!

if observations < assets {
    // Use Ledoit-Wolf shrinkage
    let estimator = LedoitWolfEstimator::estimate(&data)?;
    let stable_cov = estimator.shrunk_covariance();
}
```

## Testing

```bash
cargo test matrix
```

Comprehensive tests cover:
- Decomposition accuracy (A = reconstructed)
- Orthogonality (Q^T Q = I)
- Numerical stability
- Edge cases (singular matrices, near-singular)

## References

- Golub & Van Loan: "Matrix Computations"
- Ledoit & Wolf (2004): "Honey, I Shrunk the Sample Covariance Matrix"
- Nocedal & Wright: "Numerical Optimization"
