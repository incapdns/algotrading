//! Clean SIMD-enabled technical indicators using default type parameters
//!
//! This module shows the preferred pattern: structs are generic with `T: Numeric = f64`
//! so they default to scalar but support SIMD when explicitly requested.

use crate::numeric::Numeric;

/// Exponential Moving Average
///
/// Defaults to `f64` but supports SIMD types like `f64x4` and `f64x8`.
///
/// # Examples
///
/// ```
/// use algotrading::ta::EWMA;
///
/// // Scalar (default)
/// let mut ema = EWMA::from_period(20);
/// let value = ema.update(100.0);
///
/// // SIMD (explicit)
/// #[cfg(feature = "simd")]
/// {
///     use algotrading::numeric::f64x4;
///     let mut ema_simd = EWMA::<f64x4>::from_period(20);
///     let values = f64x4::from_array([100.0, 101.0, 99.0, 102.0]);
///     let result = ema_simd.update(values);
/// }
/// ```
#[repr(align(64))]
pub struct EWMA<T: Numeric = f64> {
    value: T,
    alpha: f64,
    initialized: bool,
}

impl<T: Numeric> EWMA<T> {
    /// Create with decay factor alpha
    ///
    /// alpha = 2 / (period + 1), e.g. alpha=0.1 ≈ 19-period EMA
    #[inline]
    pub fn new(alpha: f64) -> Self {
        Self {
            value: T::default(),
            alpha,
            initialized: false,
        }
    }

    /// Create from period (more intuitive)
    #[inline]
    pub fn from_period(period: usize) -> Self {
        Self::new(2.0 / (period as f64 + 1.0))
    }

    /// Update with new value
    #[inline(always)]
    pub fn update(&mut self, value: T) -> T {
        if !self.initialized {
            self.value = value;
            self.initialized = true;
        } else {
            self.value = value * T::splat(self.alpha) +
                        self.value * T::splat(1.0 - self.alpha);
        }
        self.value
    }

    /// Get current value without updating
    #[inline(always)]
    pub fn value(&self) -> T {
        self.value
    }

    /// Reset state
    #[inline]
    pub fn reset(&mut self) {
        self.value = T::default();
        self.initialized = false;
    }
}

impl<T: Numeric> Default for EWMA<T> {
    fn default() -> Self {
        Self::from_period(20)
    }
}

/// MACD (Moving Average Convergence Divergence)
///
/// Defaults to `f64` but supports SIMD types.
#[repr(align(64))]
pub struct MACD<T: Numeric = f64> {
    fast: EWMA<T>,
    slow: EWMA<T>,
    signal: EWMA<T>,
}

impl<T: Numeric> MACD<T> {
    /// Standard MACD(12, 26, 9)
    #[inline]
    pub fn standard() -> Self {
        Self::new(12, 26, 9)
    }

    #[inline]
    pub fn new(fast_period: usize, slow_period: usize, signal_period: usize) -> Self {
        Self {
            fast: EWMA::from_period(fast_period),
            slow: EWMA::from_period(slow_period),
            signal: EWMA::from_period(signal_period),
        }
    }

    /// Update with new price, returns (macd_line, signal_line, histogram)
    #[inline]
    pub fn update(&mut self, price: T) -> (T, T, T) {
        let fast_val = self.fast.update(price);
        let slow_val = self.slow.update(price);
        let macd_line = fast_val - slow_val;
        let signal_line = self.signal.update(macd_line);
        let histogram = macd_line - signal_line;

        (macd_line, signal_line, histogram)
    }
}

impl<T: Numeric> Default for MACD<T> {
    fn default() -> Self {
        Self::standard()
    }
}

/// RSI (Relative Strength Index)
///
/// Defaults to `f64` but supports SIMD types.
#[repr(align(64))]
pub struct RSI<T: Numeric = f64> {
    gain_ema: EWMA<T>,
    loss_ema: EWMA<T>,
    prev_price: T,
    initialized: bool,
}

impl<T: Numeric> RSI<T> {
    /// Standard RSI(14)
    #[inline]
    pub fn standard() -> Self {
        Self::new(14)
    }

    #[inline]
    pub fn new(period: usize) -> Self {
        let alpha = 1.0 / period as f64; // Wilder's smoothing
        Self {
            gain_ema: EWMA::new(alpha),
            loss_ema: EWMA::new(alpha),
            prev_price: T::default(),
            initialized: false,
        }
    }

    /// Update with new price, returns RSI (0-100)
    #[inline]
    pub fn update(&mut self, price: T) -> T {
        if !self.initialized {
            self.prev_price = price;
            self.initialized = true;
            return T::splat(50.0); // Neutral
        }

        let change = price - self.prev_price;
        self.prev_price = price;

        // For SIMD, we use max to simulate conditional assignment
        let zero = T::default();
        let gain = change.max(zero);
        let loss = (-change).max(zero);

        let avg_gain = self.gain_ema.update(gain);
        let avg_loss = self.loss_ema.update(loss);

        // RSI = 100 - (100 / (1 + RS))
        let rs = avg_gain / (avg_loss + T::splat(1e-10));
        T::splat(100.0) - (T::splat(100.0) / (T::splat(1.0) + rs))
    }
}

impl<T: Numeric> Default for RSI<T> {
    fn default() -> Self {
        Self::standard()
    }
}

/// Bollinger Bands
///
/// Defaults to `f64` but supports SIMD types.
#[repr(align(64))]
pub struct BollingerBands<T: Numeric = f64, const N: usize = 20> {
    values: [T; N],
    head: usize,
    count: usize,
    sum: T,
    sum_sq: T,
    num_std: f64,
}

impl<T: Numeric, const N: usize> BollingerBands<T, N> {
    /// Create with standard deviation multiplier (typically 2.0)
    #[inline]
    pub fn new(num_std: f64) -> Self {
        Self {
            values: [T::default(); N],
            head: 0,
            count: 0,
            sum: T::default(),
            sum_sq: T::default(),
            num_std,
        }
    }

    /// Standard Bollinger(20, 2.0)
    #[inline]
    pub fn standard() -> Self {
        Self::new(2.0)
    }

    /// Update with new price, returns (upper, middle, lower, %b)
    #[inline]
    pub fn update(&mut self, price: T) -> (T, T, T, T) {
        // Remove old
        if self.count >= N {
            let old = self.values[self.head];
            self.sum -= old;
            self.sum_sq -= old * old;
        } else {
            self.count += 1;
        }

        // Add new
        self.values[self.head] = price;
        self.sum += price;
        self.sum_sq += price * price;
        self.head = (self.head + 1) % N;

        // Calculate bands
        let n = T::splat(self.count as f64);
        let middle = self.sum / n;
        let variance = (self.sum_sq / n) - (middle * middle);
        let std_dev = variance.max(T::default()).sqrt();

        let upper = middle + std_dev * T::splat(self.num_std);
        let lower = middle - std_dev * T::splat(self.num_std);

        // %b indicator
        let bandwidth = upper - lower;
        let percent_b = (price - lower) / (bandwidth + T::splat(1e-10));

        (upper, middle, lower, percent_b)
    }
}

impl<T: Numeric, const N: usize> Default for BollingerBands<T, N> {
    fn default() -> Self {
        Self::standard()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ewma_defaults_to_f64() {
        // No type annotation needed - defaults to f64
        let mut ema = EWMA::from_period(10);
        let result = ema.update(100.0);
        assert_eq!(result, 100.0);
    }

    #[test]
    fn test_ewma_explicit_f64() {
        let mut ema = EWMA::<f64>::from_period(10);
        let result = ema.update(100.0);
        assert_eq!(result, 100.0);
    }

    #[test]
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    fn test_ewma_simd() {
        use crate::numeric::f64x4;

        let mut ema = EWMA::<f64x4>::from_period(10);
        let values = f64x4::from_array([100.0, 101.0, 99.0, 102.0]);
        let result = ema.update(values);

        let arr = result.to_array();
        assert_eq!(arr, [100.0, 101.0, 99.0, 102.0]);
    }

    #[test]
    fn test_macd_defaults() {
        let mut macd = MACD::standard();
        let (line, signal, hist) = macd.update(100.0);
        assert!(!line.is_nan());
        assert!(!signal.is_nan());
        assert!(!hist.is_nan());
    }

    #[test]
    fn test_bollinger_defaults() {
        let mut bb = BollingerBands::<f64, 20>::standard();
        let (upper, middle, lower, _percent_b) = bb.update(100.0);
        assert!(upper >= middle);
        assert!(middle >= lower);
    }

    #[test]
    fn test_generic_function() {
        fn compute_rsi<T: Numeric>(prices: &[f64]) -> f64 {
            let mut rsi = RSI::<T>::standard();
            let mut result = T::default();

            for &price in prices {
                result = rsi.update(T::splat(price));
            }

            result.reduce_sum() / T::LANES as f64
        }

        let prices = [100.0, 102.0, 104.0, 103.0, 105.0];

        let rsi_scalar = compute_rsi::<f64>(&prices);

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            use crate::numeric::f64x4;
            let rsi_simd = compute_rsi::<f64x4>(&prices);
            assert!((rsi_scalar - rsi_simd).abs() < 1e-10);
        }

        assert!(rsi_scalar > 0.0 && rsi_scalar <= 100.0);
    }
}
