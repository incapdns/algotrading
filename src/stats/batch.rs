use std::arch::x86_64::*;

/// Process 4 independent rolling stats in parallel using AVX2
#[repr(align(64))]
pub struct RollingStatsBatch4<const N: usize> {
    // AoS → SoA layout for SIMD
    values: [[f64; 4]; N],  // Transposed: values[time][series]
    head: usize,
    count: usize,
    sums: [f64; 4],
    sums_sq: [f64; 4],
}

impl<const N: usize> RollingStatsBatch4<N> {
    #[inline]
    pub const fn new() -> Self {
        Self {
            values: [[0.0; 4]; N],
            head: 0,
            count: 0,
            sums: [0.0; 4],
            sums_sq: [0.0; 4],
        }
    }
    
    /// Update all 4 series at once (SIMD)
    #[inline]
    pub fn update(&mut self, new_values: [f64; 4]) -> [(f64, f64); 4] {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    return self.update_avx2(new_values);
                }
            }
        }
        
        // Fallback: scalar
        self.update_scalar(new_values)
    }
    
    /// SIMD-accelerated update using AVX2 instructions
    ///
    /// # Safety
    ///
    /// This function is marked unsafe because it uses AVX2 intrinsics. The caller must ensure:
    ///
    /// 1. **CPU Feature Detection**: This function should only be called after verifying
    ///    `is_x86_feature_detected!("avx2")` returns true. The `#[target_feature]` attribute
    ///    ensures the compiler generates AVX2 instructions, but runtime detection is required.
    ///
    /// 2. **Alignment Requirements**: While `_mm256_loadu_pd` (unaligned load) is used,
    ///    the arrays must still be valid memory locations:
    ///    - `new_values` is a [f64; 4] on the stack (naturally aligned to 8 bytes minimum)
    ///    - `self.values[self.head]` is a [f64; 4] within the struct array
    ///    - `self.sums` and `self.sums_sq` are [f64; 4] arrays in the struct
    ///
    /// 3. **Memory Validity**: All pointer operations use `as_ptr()` and `as_mut_ptr()` on
    ///    valid Rust arrays, ensuring memory is initialized and within bounds.
    ///
    /// 4. **Data Race Freedom**: This method takes `&mut self`, ensuring exclusive access
    ///    to all mutated data (values, sums, sums_sq, head, count).
    ///
    /// # Invariants
    ///
    /// - `self.head` is always < N (maintained by modulo operation)
    /// - `self.count` is always <= N
    /// - All f64 values in arrays are initialized (zeros at construction)
    ///
    #[target_feature(enable = "avx2", enable = "fma")]
    #[cfg(target_arch = "x86_64")]
    unsafe fn update_avx2(&mut self, new_values: [f64; 4]) -> [(f64, f64); 4] {
        // Load new values into AVX2 register (unaligned load is safe for stack arrays)
        let new_vec = _mm256_loadu_pd(new_values.as_ptr());

        // Remove old values if at capacity
        if self.count >= N {
            let old_vec = _mm256_loadu_pd(self.values[self.head].as_ptr());
            let sums_vec = _mm256_loadu_pd(self.sums.as_ptr());
            let sums_sq_vec = _mm256_loadu_pd(self.sums_sq.as_ptr());

            // sums -= old_values
            let new_sums = _mm256_sub_pd(sums_vec, old_vec);
            _mm256_storeu_pd(self.sums.as_mut_ptr(), new_sums);

            // sums_sq -= old_values^2
            let old_sq = _mm256_mul_pd(old_vec, old_vec);
            let new_sums_sq = _mm256_sub_pd(sums_sq_vec, old_sq);
            _mm256_storeu_pd(self.sums_sq.as_mut_ptr(), new_sums_sq);
        } else {
            self.count += 1;
        }

        // Store new values (transposed AoS->SoA layout already expected by values[head])
        _mm256_storeu_pd(self.values[self.head].as_mut_ptr(), new_vec);

        // Add new values to sums
        let sums_vec = _mm256_loadu_pd(self.sums.as_ptr());
        let new_sums = _mm256_add_pd(sums_vec, new_vec);
        _mm256_storeu_pd(self.sums.as_mut_ptr(), new_sums);

        // Add new_values^2 to sums_sq (FMA)
        let sums_sq_vec = _mm256_loadu_pd(self.sums_sq.as_ptr());
        let new_sums_sq = _mm256_fmadd_pd(new_vec, new_vec, sums_sq_vec);
        _mm256_storeu_pd(self.sums_sq.as_mut_ptr(), new_sums_sq);

        // Advance head
        self.head = (self.head + 1) % N;

        // Calculate stats (vectorized)
        let n = _mm256_set1_pd(self.count as f64);
        let means = _mm256_div_pd(new_sums, n);

        // variance = sums_sq/n - means^2
        let means_sq = _mm256_mul_pd(means, means);
        let variances = _mm256_sub_pd(_mm256_div_pd(new_sums_sq, n), means_sq);

        // clamp variance >= 0 before sqrt to avoid NaN
        let zero = _mm256_set1_pd(0.0);
        let variances_clamped = _mm256_max_pd(variances, zero);
        let std_devs = _mm256_sqrt_pd(variances_clamped);

        // Store results via temporaries (SoA) and then interleave into results (AoS)
        let mut means_arr: [f64; 4] = [0.0; 4];
        let mut stds_arr: [f64; 4] = [0.0; 4];
        _mm256_storeu_pd(means_arr.as_mut_ptr(), means);
        _mm256_storeu_pd(stds_arr.as_mut_ptr(), std_devs);

        let mut results = [(0.0f64, 0.0f64); 4];
        for i in 0..4 {
            results[i] = (means_arr[i], stds_arr[i]);
        }

        results
    }

    
    fn update_scalar(&mut self, new_values: [f64; 4]) -> [(f64, f64); 4] {
        // Remove old
        if self.count >= N {
            for i in 0..4 {
                let old = self.values[self.head][i];
                self.sums[i] -= old;
                self.sums_sq[i] -= old * old;
            }
        } else {
            self.count += 1;
        }
        
        // Add new
        for i in 0..4 {
            self.values[self.head][i] = new_values[i];
            self.sums[i] += new_values[i];
            self.sums_sq[i] += new_values[i] * new_values[i];
        }
        
        self.head = (self.head + 1) % N;
        
        // Calculate
        let n = self.count as f64;
        let mut results = [(0.0, 0.0); 4];
        for i in 0..4 {
            let mean = self.sums[i] / n;
            let variance = (self.sums_sq[i] / n) - (mean * mean);
            results[i] = (mean, variance.max(0.0).sqrt());
        }
        
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_batch_simd() {
        let mut batch = RollingStatsBatch4::<100>::new();
        
        let values = [1.0, 2.0, 3.0, 4.0];
        let results = batch.update(values);
        
        for i in 0..4 {
            assert!((results[i].0 - values[i]).abs() < 1e-10);
        }
    }
}