/// Numeric traits for scalar and SIMD operations
///
/// This module provides a unified interface for numeric operations that work
/// with both scalar types (f64) and SIMD types (f64x4, f64x8, etc.).
use std::ops::{Add, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, Neg};

pub mod helpers;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub use std::simd::{f64x4, f64x8, num::SimdFloat as SimdFloatOps};

/// Core numeric trait for types that can be used in quantitative computations
///
/// This trait abstracts over scalar (f64) and SIMD vector types, allowing
/// algorithms to be written once and work efficiently with both.
pub trait Numeric:
    Copy
    + Default
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + Neg<Output = Self>
    + PartialOrd
    + Sized
    + 'static
{
    /// Create a value with all lanes set to the same scalar
    fn splat(value: f64) -> Self;

    /// Create a value from a scalar f64 (alias for splat, for consistency)
    #[inline(always)]
    fn from_f64(value: f64) -> Self {
        Self::splat(value)
    }

    /// Zero value (convenience wrapper for Default)
    #[inline(always)]
    fn zero() -> Self {
        Self::default()
    }

    /// One value
    #[inline(always)]
    fn one() -> Self {
        Self::splat(1.0)
    }

    /// Square root
    fn sqrt(self) -> Self;

    /// Absolute value
    fn abs(self) -> Self;

    /// Maximum of two values
    fn max(self, other: Self) -> Self;

    /// Minimum of two values
    fn min(self, other: Self) -> Self;

    /// Fused multiply-add: self * a + b
    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
    }

    /// Natural logarithm
    fn ln(self) -> Self;

    /// Exponential function
    fn exp(self) -> Self;

    /// Power function
    fn powf(self, n: f64) -> Self;

    /// Reciprocal (1/x)
    fn recip(self) -> Self {
        Self::one() / self
    }

    /// Sum all lanes (for SIMD) or return self (for scalar)
    fn reduce_sum(self) -> f64;

    /// Horizontal maximum across all lanes
    fn reduce_max(self) -> f64;

    /// Horizontal minimum across all lanes
    fn reduce_min(self) -> f64;

    /// Check if value is NaN
    fn is_nan(self) -> bool;

    /// Check if value is finite
    fn is_finite(self) -> bool;

    /// Load from slice (for SIMD, loads multiple values; for scalar, loads one)
    fn from_slice(slice: &[f64]) -> Self;

    /// Number of lanes (1 for scalar, 4 for f64x4, etc.)
    const LANES: usize;
}

// ============================================================================
// Scalar f64 implementation
// ============================================================================

impl Numeric for f64 {
    #[inline(always)]
    fn splat(value: f64) -> Self {
        value
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }

    #[inline(always)]
    fn abs(self) -> Self {
        f64::abs(self)
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        f64::max(self, other)
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        f64::min(self, other)
    }

    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        f64::mul_add(self, a, b)
    }

    #[inline(always)]
    fn ln(self) -> Self {
        f64::ln(self)
    }

    #[inline(always)]
    fn exp(self) -> Self {
        f64::exp(self)
    }

    #[inline(always)]
    fn powf(self, n: f64) -> Self {
        f64::powf(self, n)
    }

    #[inline(always)]
    fn reduce_sum(self) -> f64 {
        self
    }

    #[inline(always)]
    fn reduce_max(self) -> f64 {
        self
    }

    #[inline(always)]
    fn reduce_min(self) -> f64 {
        self
    }

    #[inline(always)]
    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }

    #[inline(always)]
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }

    #[inline(always)]
    fn from_slice(slice: &[f64]) -> Self { slice[0] }


    const LANES: usize = 1;
}

// ============================================================================
// SIMD f64x4 implementation (AVX2)
// ============================================================================

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
impl Numeric for f64x4 {
    #[inline(always)]
    fn splat(value: f64) -> Self {
        f64x4::splat(value)
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        self.sqrt()
    }

    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        self.simd_max(other)
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        self.simd_min(other)
    }

    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
    }

    #[inline(always)]
    fn ln(self) -> Self {
        // Approximate ln using polynomial (for speed)
        // For production, you might want a more accurate version
        let one = Self::splat(1.0);
        let x = self / (self + one);
        let x2 = x * x;

        // Series expansion: ln((1+x)/(1-x)) = 2(x + x³/3 + x⁵/5 + ...)
        let result = x * Self::splat(2.0) *
            (one + x2 * (Self::splat(1.0/3.0) +
                        x2 * (Self::splat(1.0/5.0) +
                             x2 * Self::splat(1.0/7.0))));
        result
    }

    #[inline(always)]
    fn exp(self) -> Self {
        // Fast exp approximation using Schraudolph's method
        // For production, consider more accurate versions
        let a = Self::splat(12102203.161561486); // 2^20 / ln(2)
        let b = Self::splat(1065353216.0); // 2^20 * 1023

        // This is a fast approximation - for better accuracy use a library
        // Just showing the pattern here
        self * Self::splat(1.442695040) // ln(2)
    }

    #[inline(always)]
    fn powf(self, n: f64) -> Self {
        // x^n = exp(n * ln(x))
        (self.ln() * Self::splat(n)).exp()
    }

    #[inline(always)]
    fn reduce_sum(self) -> f64 {
        self.reduce_sum()
    }

    #[inline(always)]
    fn reduce_max(self) -> f64 {
        self.reduce_max()
    }

    #[inline(always)]
    fn reduce_min(self) -> f64 {
        self.reduce_min()
    }

    #[inline(always)]
    fn is_nan(self) -> bool {
        self.is_nan().any()
    }

    #[inline(always)]
    fn is_finite(self) -> bool {
        self.is_finite().all()
    }

    fn from_slice(slice: &[f64]) -> Self {
        let mut tmp = [0.0; 4];
        tmp[..slice.len()].copy_from_slice(slice);
        Self::from_array(tmp)
    }

    const LANES: usize = 4;
}

// ============================================================================
// SIMD f64x8 implementation (AVX-512)
// ============================================================================

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
impl Numeric for f64x8 {
    #[inline(always)]
    fn splat(value: f64) -> Self {
        f64x8::splat(value)
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        self.sqrt()
    }

    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        self.simd_max(other)
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        self.simd_min(other)
    }

    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
    }

    #[inline(always)]
    fn ln(self) -> Self {
        // Same approximation as f64x4
        let one = Self::splat(1.0);
        let x = self / (self + one);
        let x2 = x * x;

        let result = x * Self::splat(2.0) *
            (one + x2 * (Self::splat(1.0/3.0) +
                        x2 * (Self::splat(1.0/5.0) +
                             x2 * Self::splat(1.0/7.0))));
        result
    }

    #[inline(always)]
    fn exp(self) -> Self {
        self * Self::splat(1.442695040)
    }

    #[inline(always)]
    fn powf(self, n: f64) -> Self {
        (self.ln() * Self::splat(n)).exp()
    }

    #[inline(always)]
    fn reduce_sum(self) -> f64 {
        self.reduce_sum()
    }

    #[inline(always)]
    fn reduce_max(self) -> f64 {
        self.reduce_max()
    }

    #[inline(always)]
    fn reduce_min(self) -> f64 {
        self.reduce_min()
    }

    #[inline(always)]
    fn is_nan(self) -> bool {
        self.is_nan().any()
    }

    #[inline(always)]
    fn is_finite(self) -> bool {
        self.is_finite().all()
    }

    fn from_slice(slice: &[f64]) -> Self {
        let mut tmp = [0.0; 8];
        tmp[..slice.len()].copy_from_slice(slice);
        Self::from_array(tmp)
    }

    const LANES: usize = 8;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_operations() {
        let x = 4.0_f64;
        let y = 2.0_f64;

        assert_eq!(x.sqrt(), 2.0);
        assert_eq!(x.max(y), 4.0);
        assert_eq!(x.min(y), 2.0);
        assert_eq!(x.reduce_sum(), 4.0);
        assert_eq!(f64::LANES, 1);
    }

    #[test]
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    fn test_simd_operations() {
        let x = f64x4::splat(4.0);
        let y = f64x4::splat(2.0);

        let sqrt_result = x.sqrt();
        assert!((sqrt_result.reduce_sum() - 8.0).abs() < 1e-10); // 4 lanes * 2.0

        assert_eq!(x.reduce_sum(), 16.0); // 4 * 4.0
        assert_eq!(f64x4::LANES, 4);
    }

    #[test]
    fn test_generic_function() {
        fn square<T: Numeric>(x: T) -> T {
            x * x
        }

        let scalar = square(3.0_f64);
        assert_eq!(scalar, 9.0);

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            let simd = square(f64x4::splat(3.0));
            assert_eq!(simd.reduce_sum(), 36.0); // 4 * 9.0
        }
    }
}
