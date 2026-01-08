#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD-optimized matrix multiplication kernels
/// Uses AVX2 for 4x speedup on compatible hardware
///
/// Blocked matrix multiplication for cache efficiency
/// Block size tuned for L1 cache (64 bytes = 8 doubles)
pub fn matmul_blocked<const N: usize>(a: &[[f64; N]; N], b: &[[f64; N]; N]) -> [[f64; N]; N] {
    const BLOCK_SIZE: usize = 8;
    let mut result = [[0.0; N]; N];
    
    for i_block in (0..N).step_by(BLOCK_SIZE) {
        for j_block in (0..N).step_by(BLOCK_SIZE) {
            for k_block in (0..N).step_by(BLOCK_SIZE) {
                // Micro-kernel: multiply blocks
                let i_max = (i_block + BLOCK_SIZE).min(N);
                let j_max = (j_block + BLOCK_SIZE).min(N);
                let k_max = (k_block + BLOCK_SIZE).min(N);
                
                for i in i_block..i_max {
                    for k in k_block..k_max {
                        let a_ik = a[i][k];
                        for j in j_block..j_max {
                            result[i][j] += a_ik * b[k][j];
                        }
                    }
                }
            }
        }
    }
    
    result
}

/// AVX2-accelerated matrix-vector multiplication (processes 4 f64 values at once)
///
/// # Safety
///
/// This function is marked unsafe because it uses AVX2 and FMA intrinsics. The caller must ensure:
///
/// 1. **CPU Feature Detection**: This function should only be called after verifying
///    `is_x86_feature_detected!("avx2")` returns true. The `#[target_feature]` attribute
///    enables AVX2 code generation, but runtime detection is required for safety.
///
/// 2. **Memory Alignment**: Uses `_mm256_loadu_pd` (unaligned load) for flexibility:
///    - `&matrix[i][j]` points to a valid subslice of the matrix row (compile-time bounds)
///    - `&vector[j]` points to a valid subslice of the vector (compile-time bounds)
///    - Unaligned loads are safe but slightly slower than aligned loads
///
/// 3. **Memory Validity**: All pointer operations are derived from valid Rust references:
///    - Matrix is a `&[[f64; N]; N]` - fully initialized, bounds-checked at compile time
///    - Vector is a `&[f64; N]` - fully initialized, bounds-checked at compile time
///    - Loop condition `j + 4 <= N` ensures we never read beyond array boundaries
///
/// 4. **Remainder Handling**: Scalar fallback loop handles `N % 4` remaining elements,
///    preventing out-of-bounds access when N is not a multiple of 4.
///
/// # Invariants
///
/// - `matrix[i][j..j+4]` is always within bounds when accessed (ensured by `j + 4 <= N`)
/// - `vector[j..j+4]` is always within bounds (ensured by `j + 4 <= N`)
/// - All f64 values in matrix and vector are initialized (guaranteed by Rust references)
///
#[target_feature(enable = "avx2", enable = "fma")]
#[cfg(target_arch = "x86_64")]
unsafe fn matvec_avx2<const N: usize>(matrix: &[[f64; N]; N], vector: &[f64; N]) -> [f64; N] {
    let mut result = [0.0; N];

    for i in 0..N {
        let mut sum_vec = _mm256_setzero_pd();

        // Process 4 elements at a time using SIMD
        let mut j = 0;
        while j + 4 <= N {
            // Safe: j + 4 <= N ensures these slices are within bounds
            let mat_vec = _mm256_loadu_pd(&matrix[i][j]);
            let vec_vec = _mm256_loadu_pd(&vector[j]);
            sum_vec = _mm256_fmadd_pd(mat_vec, vec_vec, sum_vec);
            j += 4;
        }
        
        // Horizontal sum of the 4 values in sum_vec
        let mut sum_array = [0.0; 4];
        _mm256_storeu_pd(sum_array.as_mut_ptr(), sum_vec);
        let mut sum = sum_array.iter().sum::<f64>();
        
        // Handle remaining elements
        while j < N {
            sum += matrix[i][j] * vector[j];
            j += 1;
        }
        
        result[i] = sum;
    }
    
    result
}

/// Public wrapper with runtime detection
#[inline]
pub fn matvec_simd<const N: usize>(matrix: &[[f64; N]; N], vector: &[f64; N]) -> [f64; N] {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { return matvec_avx2(matrix, vector); }
        }
    }
    
    // Fallback
    let mut result = [0.0; N];
    for i in 0..N {
        for j in 0..N {
            result[i] += matrix[i][j] * vector[j];
        }
    }
    result
}

/// QR Decomposition using Householder reflections
/// Critical for least squares regression and portfolio optimization
pub struct QRDecomposition<const N: usize> {
    q: [[f64; N]; N],
    r: [[f64; N]; N],
}

impl<const N: usize> QRDecomposition<N> {
    /// Decompose A = QR where Q is orthogonal and R is upper triangular
    pub fn decompose(matrix: &[[f64; N]; N]) -> Result<Self, &'static str> {
        let mut a = *matrix;
        let mut q = [[0.0; N]; N];
        
        // Initialize Q as identity
        for i in 0..N {
            q[i][i] = 1.0;
        }
        
        // Householder transformations
        for k in 0..N {
            // Compute Householder vector for column k
            let mut norm = 0.0;
            for i in k..N {
                norm += a[i][k] * a[i][k];
            }
            norm = norm.sqrt();
            
            if norm < 1e-10 {
                continue;
            }
            
            let mut v = [0.0; N];
            v[k] = a[k][k] + norm.copysign(a[k][k]);
            for i in (k + 1)..N {
                v[i] = a[i][k];
            }
            
            // Normalize v
            let mut v_norm = 0.0;
            for i in k..N {
                v_norm += v[i] * v[i];
            }
            v_norm = v_norm.sqrt();
            
            if v_norm < 1e-10 {
                continue;
            }
            
            for i in k..N {
                v[i] /= v_norm;
            }
            
            // Apply Householder transformation to A
            for j in k..N {
                let mut dot = 0.0;
                for i in k..N {
                    dot += v[i] * a[i][j];
                }
                dot *= 2.0;
                
                for i in k..N {
                    a[i][j] -= dot * v[i];
                }
            }
            
            // Apply to Q
            for j in 0..N {
                let mut dot = 0.0;
                for i in k..N {
                    dot += v[i] * q[i][j];
                }
                dot *= 2.0;
                
                for i in k..N {
                    q[i][j] -= dot * v[i];
                }
            }
        }
        
        Ok(Self { q, r: a })
    }
    
    /// Get Q matrix
    #[inline]
    pub fn q(&self) -> &[[f64; N]; N] {
        &self.q
    }
    
    /// Get R matrix
    #[inline]
    pub fn r(&self) -> &[[f64; N]; N] {
        &self.r
    }
    
    /// Solve least squares: minimize ||Ax - b||²
    #[inline]
    pub fn solve_least_squares(&self, b: &[f64; N]) -> [f64; N] {
        // x = R^(-1) * Q^T * b
        
        // Step 1: c = Q^T * b
        let mut c = [0.0; N];
        for i in 0..N {
            for j in 0..N {
                c[i] += self.q[j][i] * b[j];
            }
        }
        
        // Step 2: Solve Rx = c by back substitution
        let mut x = [0.0; N];
        for i in (0..N).rev() {
            let mut sum = c[i];
            for j in (i + 1)..N {
                sum -= self.r[i][j] * x[j];
            }
            x[i] = sum / self.r[i][i];
        }
        
        x
    }
}

/// SVD (Singular Value Decomposition) via Jacobi algorithm
/// A = U * Σ * V^T
/// Essential for PCA, factor models, matrix pseudoinverse
pub struct SVD<const N: usize> {
    u: [[f64; N]; N],
    singular_values: [f64; N],
    vt: [[f64; N]; N],
}

impl<const N: usize> SVD<N> {
    /// Decompose using one-sided Jacobi algorithm
    pub fn decompose(matrix: &[[f64; N]; N]) -> Result<Self, &'static str> {
        const MAX_ITER: usize = 100;
        const TOLERANCE: f64 = 1e-10;
        
        let mut u = *matrix;
        let mut v = [[0.0; N]; N];
        
        // Initialize V as identity
        for i in 0..N {
            v[i][i] = 1.0;
        }
        
        // Jacobi rotations
        for _ in 0..MAX_ITER {
            let mut converged = true;
            
            for p in 0..N {
                for q in (p + 1)..N {
                    // Compute dot products
                    let mut ap_ap = 0.0;
                    let mut aq_aq = 0.0;
                    let mut ap_aq = 0.0;
                    
                    for i in 0..N {
                        ap_ap += u[i][p] * u[i][p];
                        aq_aq += u[i][q] * u[i][q];
                        ap_aq += u[i][p] * u[i][q];
                    }
                    
                    if ap_aq.abs() < TOLERANCE {
                        continue;
                    }
                    
                    converged = false;
                    
                    // Compute rotation angle
                    let tau = (aq_aq - ap_ap) / (2.0 * ap_aq);
                    let t = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
                    let c = 1.0 / (1.0 + t * t).sqrt();
                    let s = t * c;
                    
                    // Apply rotation to U
                    for i in 0..N {
                        let up = u[i][p];
                        let uq = u[i][q];
                        u[i][p] = c * up - s * uq;
                        u[i][q] = s * up + c * uq;
                    }
                    
                    // Apply rotation to V
                    for i in 0..N {
                        let vp = v[i][p];
                        let vq = v[i][q];
                        v[i][p] = c * vp - s * vq;
                        v[i][q] = s * vp + c * vq;
                    }
                }
            }
            
            if converged {
                break;
            }
        }
        
        // Extract singular values (column norms of U)
        let mut singular_values = [0.0; N];
        for j in 0..N {
            let mut norm = 0.0;
            for i in 0..N {
                norm += u[i][j] * u[i][j];
            }
            singular_values[j] = norm.sqrt();
            
            // Normalize column
            if singular_values[j] > 1e-10 {
                for i in 0..N {
                    u[i][j] /= singular_values[j];
                }
            }
        }
        
        // Transpose V to get V^T
        let vt = transpose(&v);
        
        Ok(Self { u, singular_values, vt })
    }
    
    /// Get U matrix
    #[inline]
    pub fn u(&self) -> &[[f64; N]; N] {
        &self.u
    }
    
    /// Get singular values (diagonal of Σ)
    #[inline]
    pub fn singular_values(&self) -> &[f64; N] {
        &self.singular_values
    }
    
    /// Get V^T matrix
    #[inline]
    pub fn vt(&self) -> &[[f64; N]; N] {
        &self.vt
    }
    
    /// Compute pseudoinverse: A^+ = V * Σ^(-1) * U^T
    pub fn pseudoinverse(&self) -> [[f64; N]; N] {
        let mut result = [[0.0; N]; N];
        
        for i in 0..N {
            for j in 0..N {
                let mut sum = 0.0;
                for k in 0..N {
                    if self.singular_values[k] > 1e-10 {
                        sum += self.vt[k][i] * self.u[j][k] / self.singular_values[k];
                    }
                }
                result[i][j] = sum;
            }
        }
        
        result
    }
    
    /// Compute condition number (max singular value / min singular value)
    #[inline]
    pub fn condition_number(&self) -> f64 {
        let mut max_sv = 0.0;
        let mut min_sv = f64::INFINITY;
        
        for &sv in &self.singular_values {
            if sv > max_sv {
                max_sv = sv;
            }
            if sv > 1e-10 && sv < min_sv {
                min_sv = sv;
            }
        }
        
        if min_sv < f64::INFINITY {
            max_sv / min_sv
        } else {
            f64::INFINITY
        }
    }
}

/// LU Decomposition with partial pivoting
/// Faster than Gaussian elimination for repeated solves
pub struct LUDecomposition<const N: usize> {
    lu: [[f64; N]; N],
    permutation: [usize; N],
}

impl<const N: usize> LUDecomposition<N> {
    /// Decompose PA = LU where P is a permutation matrix
    pub fn decompose(matrix: &[[f64; N]; N]) -> Result<Self, &'static str> {
        let mut lu = *matrix;
        let mut perm = [0; N];
        for i in 0..N {
            perm[i] = i;
        }
        
        for k in 0..N {
            // Find pivot
            let mut max_val = 0.0;
            let mut max_row = k;
            for i in k..N {
                if lu[i][k].abs() > max_val {
                    max_val = lu[i][k].abs();
                    max_row = i;
                }
            }
            
            if max_val < 1e-10 {
                return Err("Matrix is singular");
            }
            
            // Swap rows
            if max_row != k {
                for j in 0..N {
                    let temp = lu[k][j];
                    lu[k][j] = lu[max_row][j];
                    lu[max_row][j] = temp;
                }
                perm.swap(k, max_row);
            }
            
            // Eliminate
            for i in (k + 1)..N {
                lu[i][k] /= lu[k][k];
                for j in (k + 1)..N {
                    lu[i][j] -= lu[i][k] * lu[k][j];
                }
            }
        }
        
        Ok(Self { lu, permutation: perm })
    }
    
    /// Solve Ax = b
    pub fn solve(&self, b: &[f64; N]) -> [f64; N] {
        let mut x = [0.0; N];
        let mut y = [0.0; N];
        
        // Forward substitution: Ly = Pb
        for i in 0..N {
            let mut sum = b[self.permutation[i]];
            for j in 0..i {
                sum -= self.lu[i][j] * y[j];
            }
            y[i] = sum;
        }
        
        // Back substitution: Ux = y
        for i in (0..N).rev() {
            let mut sum = y[i];
            for j in (i + 1)..N {
                sum -= self.lu[i][j] * x[j];
            }
            x[i] = sum / self.lu[i][i];
        }
        
        x
    }
    
    /// Compute determinant
    #[inline]
    pub fn determinant(&self) -> f64 {
        let mut det = 1.0;
        for i in 0..N {
            det *= self.lu[i][i];
        }
        
        // Account for row swaps
        let mut swaps = 0;
        for i in 0..N {
            if self.permutation[i] != i {
                swaps += 1;
            }
        }
        
        if swaps % 2 == 1 {
            det = -det;
        }
        
        det
    }
}

/// Helper function already defined earlier
fn transpose<const N: usize>(matrix: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut result = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            result[j][i] = matrix[i][j];
        }
    }
    result
}

// Note: Kronecker product and vectorize functions removed due to const generic limitations
// in stable Rust. These can be added back when const_evaluatable_checked is stabilized.

/// Vectorization (stack columns) - manual version for specific sizes
/// Converts matrix to vector - useful for matrix calculus
/// Example for 2x2: vectorize_2x2, for 3x3: vectorize_3x3, etc.
pub fn vectorize_manual(matrix: &[&[f64]]) -> Vec<f64> {
    let n = matrix.len();
    let mut result = Vec::with_capacity(n * n);

    for j in 0..n {
        for i in 0..n {
            result.push(matrix[i][j]);
        }
    }

    result
}

/// Matrix power (efficient via eigendecomposition for symmetric matrices)
#[inline]
pub fn matrix_power_symmetric<const N: usize>(
    matrix: &[[f64; N]; N],
    power: f64,
) -> Result<[[f64; N]; N], &'static str> {
    // For symmetric matrices: A^p = V * Λ^p * V^T
    use super::ops::Eigen;
    
    let eigen = Eigen::decompose_symmetric(matrix)?;
    let values = eigen.values();
    let vectors = eigen.vectors();
    
    // Compute Λ^p
    let mut lambda_p = [0.0; N];
    for i in 0..N {
        lambda_p[i] = values[i].powf(power);
    }
    
    // Reconstruct: V * Λ^p * V^T
    let mut result = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            let mut sum = 0.0;
            for k in 0..N {
                sum += vectors[i][k] * lambda_p[k] * vectors[j][k];
            }
            result[i][j] = sum;
        }
    }
    
    Ok(result)
}

/// Trace (sum of diagonal elements)
#[inline]
pub fn trace<const N: usize>(matrix: &[[f64; N]; N]) -> f64 {
    let mut sum = 0.0;
    for i in 0..N {
        sum += matrix[i][i];
    }
    sum
}

/// Frobenius norm: sqrt(sum of squared elements)
#[inline]
pub fn frobenius_norm<const N: usize>(matrix: &[[f64; N]; N]) -> f64 {
    let mut sum = 0.0;
    for i in 0..N {
        for j in 0..N {
            sum += matrix[i][j] * matrix[i][j];
        }
    }
    sum.sqrt()
}

/// General matrix inversion via Gaussian elimination with pivoting
///
/// Uses stack allocation for maximum performance (no heap allocations).
/// Works for general square matrices (not just SPD matrices).
///
/// # Arguments
/// * `matrix` - Square matrix to invert
///
/// # Returns
/// * `Ok(inv)` - The inverted matrix
/// * `Err(msg)` - If the matrix is singular or nearly singular
///
/// # Performance
/// - O(N^3) complexity
/// - For SPD matrices, use `invert_spd_cholesky` instead (faster)
/// - Stack-allocated, no heap allocations
///
/// # Examples
/// ```
/// use algotrading::matrix::kernels::invert_matrix;
///
/// let matrix = [
///     [2.0, 1.0],
///     [1.0, 2.0],
/// ];
/// let inv = invert_matrix(&matrix).unwrap();
/// ```
pub fn invert_matrix<const N: usize>(matrix: &[[f64; N]; N]) -> Result<[[f64; N]; N], &'static str> {
    const MAX_DIM: usize = 32;

    if N > MAX_DIM {
        return Err("Matrix dimension exceeds MAX_DIM=32");
    }

    let mut aug = [[0.0; MAX_DIM * 2]; MAX_DIM];

    // Create augmented matrix [A | I]
    for i in 0..N {
        for j in 0..N {
            aug[i][j] = matrix[i][j];
            aug[i][j + N] = if i == j { 1.0 } else { 0.0 };
        }
    }

    // Forward elimination with partial pivoting
    for i in 0..N {
        // Find pivot (row with largest absolute value in column i)
        let mut max_row = i;
        for k in (i + 1)..N {
            if aug[k][i].abs() > aug[max_row][i].abs() {
                max_row = k;
            }
        }

        // Swap rows if needed
        if max_row != i {
            for j in 0..(N * 2) {
                let temp = aug[i][j];
                aug[i][j] = aug[max_row][j];
                aug[max_row][j] = temp;
            }
        }

        // Check for singular matrix
        if aug[i][i].abs() < 1e-10 {
            return Err("Matrix is singular or nearly singular");
        }

        // Scale pivot row
        let pivot = aug[i][i];
        for j in 0..(N * 2) {
            aug[i][j] /= pivot;
        }

        // Eliminate column below and above pivot
        for k in 0..N {
            if k != i {
                let factor = aug[k][i];
                for j in 0..(N * 2) {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }
    }

    // Extract inverse from right half of augmented matrix
    let mut inv = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            inv[i][j] = aug[i][j + N];
        }
    }

    Ok(inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_qr_decomposition() {
        let matrix = [
            [12.0, -51.0, 4.0],
            [6.0, 167.0, -68.0],
            [-4.0, 24.0, -41.0],
        ];
        
        let qr = QRDecomposition::decompose(&matrix).unwrap();
        
        // Verify Q is orthogonal: Q^T * Q = I
        let q = qr.q();
        for i in 0..3 {
            for j in 0..3 {
                let mut dot = 0.0;
                for k in 0..3 {
                    dot += q[k][i] * q[k][j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-8);
            }
        }
    }
    
    #[test]
    fn test_lu_decomposition() {
        let matrix = [
            [2.0, 1.0, 1.0],
            [4.0, 3.0, 3.0],
            [8.0, 7.0, 9.0],
        ];
        
        let lu = LUDecomposition::decompose(&matrix).unwrap();
        let det = lu.determinant();
        println!("Determinant: {}", det);
        assert!((det.abs() - 4.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_svd_identity() {
        let identity = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        
        let svd = SVD::decompose(&identity).unwrap();
        let sv = svd.singular_values();
        
        // Identity has singular values of 1
        for &val in sv {
            assert!((val - 1.0).abs() < 1e-8);
        }
    }
    
    #[test]
    fn test_trace() {
        let matrix = [
            [1.0, 2.0],
            [3.0, 4.0],
        ];
        
        assert_eq!(trace(&matrix), 5.0);
    }
}