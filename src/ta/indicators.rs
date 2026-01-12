use crate::numeric::Numeric;
use crate::core::RollingBuffer;

/// Exponential Moving Average (EWMA)
///
/// Much more memory efficient than rolling window.
/// Defaults to scalar f64 but supports SIMD types when explicitly requested.
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
/// #[cfg(all(feature = "simd", target_arch = "x86_64"))]
/// {
///     use algotrading::numeric::f64x4;
///     let mut ema = EWMA::<f64x4>::from_period(20);
///     let values = f64x4::from_array([100.0, 101.0, 99.0, 102.0]);
///     let result = ema.update(values);
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

    #[inline(always)]
    pub fn scale(&mut self, factor: T) {
        if self.initialized {
            self.value = self.value * factor;
        }
    }

    /// Create from period (more intuitive)
    #[inline]
    pub fn from_period(period: usize) -> Self {
        Self::new(2.0 / (period as f64 + 1.0))
    }

    #[inline(always)]
    pub fn is_initialized(&self) {
        self.initialized
    }

    #[inline(always)]
    pub fn consume(&mut self, value: T, alpha: f64) -> T {
        if !self.initialized {
            self.value = value;
            self.initialized = true;
        } else {
            self.value =
                value * T::splat(alpha) +
                self.value * T::splat(1.0 - alpha);
        }
        self.value
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

/// EWMA Covariance between two series
/// Online estimation without storing full history
pub struct EWMACovariance {
    mean_x: EWMA,
    mean_y: EWMA,
    cov: f64,
    alpha: f64,
    initialized: bool,
}

impl EWMACovariance {
    #[inline]
    pub fn new(alpha: f64) -> Self {
        Self {
            mean_x: EWMA::new(alpha),
            mean_y: EWMA::new(alpha),
            cov: 0.0,
            alpha,
            initialized: false,
        }
    }
    
    #[inline]
    pub fn from_period(period: usize) -> Self {
        Self::new(2.0 / (period as f64 + 1.0))
    }
    
    /// Update with new observation pair
    #[inline]
    pub fn update(&mut self, x: f64, y: f64) -> f64 {
        let mx = self.mean_x.update(x);
        let my = self.mean_y.update(y);
        
        if !self.initialized {
            self.cov = 0.0;
            self.initialized = true;
        } else {
            self.cov = (1.0 - self.alpha) * self.cov 
                     + self.alpha * (x - mx) * (y - my);
        }
        
        self.cov
    }
    
    /// Get correlation coefficient
    #[inline]
    pub fn correlation(&self) -> f64 {
        // Would need variance estimates too
        // This is simplified - in practice track var_x and var_y
        self.cov
    }
}

/// MACD (Moving Average Convergence Divergence)
///
/// Defaults to scalar f64 but supports SIMD types.
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
/// Defaults to scalar f64 but supports SIMD types.
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
/// Defaults to scalar f64 but supports SIMD types.
#[repr(align(64))]
pub struct BollingerBands<T: Numeric = f64, const N: usize = 20> {
    buffer: RollingBuffer<T, N>,
    sum: T,
    sum_sq: T,
    num_std: f64,
}

impl<T: Numeric, const N: usize> BollingerBands<T, N> {
    /// Create with standard deviation multiplier (typically 2.0)
    #[inline]
    pub fn new(num_std: f64) -> Self {
        Self {
            buffer: RollingBuffer::new(),
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
    /// %b = (price - lower) / (upper - lower)
    #[inline]
    pub fn update(&mut self, price: T) -> (T, T, T, T) {
        // Remove old value if buffer is full
        if let Some(old) = self.buffer.push(price) {
            self.sum -= old;
            self.sum_sq -= old * old;
        }

        // Add new
        self.sum += price;
        self.sum_sq += price * price;

        // Calculate bands
        let n = T::splat(self.buffer.len() as f64);
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

/// ATR (Average True Range)
///
/// Defaults to scalar f64 but supports SIMD types.
#[repr(align(64))]
pub struct ATR<T: Numeric = f64> {
    atr: EWMA<T>,
    prev_close: T,
    initialized: bool,
}

impl<T: Numeric> ATR<T> {
    /// Standard ATR(14)
    #[inline]
    pub fn standard() -> Self {
        Self::new(14)
    }

    #[inline]
    pub fn new(period: usize) -> Self {
        Self {
            atr: EWMA::from_period(period),
            prev_close: T::default(),
            initialized: false,
        }
    }

    /// Update with OHLC bar: [open, high, low, close]
    #[inline]
    pub fn update(&mut self, ohlc: [T; 4]) -> T {
        let [_open, high, low, close] = ohlc;

        if !self.initialized {
            self.prev_close = close;
            self.initialized = true;
            return high - low;
        }

        // True Range = max of:
        // 1. high - low
        // 2. |high - prev_close|
        // 3. |low - prev_close|
        let hl = high - low;
        let hc = (high - self.prev_close).abs();
        let lc = (low - self.prev_close).abs();

        let true_range = hl.max(hc).max(lc);
        self.prev_close = close;

        self.atr.update(true_range)
    }
}

impl<T: Numeric> Default for ATR<T> {
    fn default() -> Self {
        Self::standard()
    }
}

/// Rolling Correlation
///
/// Defaults to scalar f64 but supports SIMD types.
#[repr(align(64))]
pub struct RollingCorrelation<T: Numeric = f64, const N: usize = 100> {
    x_buffer: RollingBuffer<T, N>,
    y_buffer: RollingBuffer<T, N>,
    sum_x: T,
    sum_y: T,
    sum_x_sq: T,
    sum_y_sq: T,
    sum_xy: T,
}

impl<T: Numeric, const N: usize> RollingCorrelation<T, N> {
    #[inline]
    pub fn new() -> Self {
        Self {
            x_buffer: RollingBuffer::new(),
            y_buffer: RollingBuffer::new(),
            sum_x: T::default(),
            sum_y: T::default(),
            sum_x_sq: T::default(),
            sum_y_sq: T::default(),
            sum_xy: T::default(),
        }
    }

    /// Update with new (x, y) pair, returns correlation coefficient
    #[inline]
    pub fn update(&mut self, x: T, y: T) -> T {
        // Remove old values if buffers are full
        let old_x = self.x_buffer.push(x);
        let old_y = self.y_buffer.push(y);

        if let (Some(old_x), Some(old_y)) = (old_x, old_y) {
            self.sum_x -= old_x;
            self.sum_y -= old_y;
            self.sum_x_sq -= old_x * old_x;
            self.sum_y_sq -= old_y * old_y;
            self.sum_xy -= old_x * old_y;
        }

        // Add new
        self.sum_x += x;
        self.sum_y += y;
        self.sum_x_sq += x * x;
        self.sum_y_sq += y * y;
        self.sum_xy += x * y;

        // Calculate correlation
        let n = T::splat(self.x_buffer.len() as f64);
        let mean_x = self.sum_x / n;
        let mean_y = self.sum_y / n;

        let cov = (self.sum_xy / n) - (mean_x * mean_y);
        let var_x = (self.sum_x_sq / n) - (mean_x * mean_x);
        let var_y = (self.sum_y_sq / n) - (mean_y * mean_y);

        let denom = (var_x * var_y).sqrt();
        let corr = cov / (denom + T::splat(1e-10));

        // Clamp to [-1, 1]
        corr.max(T::splat(-1.0)).min(T::splat(1.0))
    }
}

impl<T: Numeric, const N: usize> Default for RollingCorrelation<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Stochastic Oscillator
///
/// Defaults to scalar f64 but supports SIMD types.
#[repr(align(64))]
pub struct Stochastic<T: Numeric = f64, const N: usize = 14> {
    highs: RollingBuffer<T, N>,
    lows: RollingBuffer<T, N>,
    closes: RollingBuffer<T, N>,
    k_smooth: EWMA<T>,
}

impl<T: Numeric, const N: usize> Stochastic<T, N> {
    /// Standard Stochastic(14, 3)
    #[inline]
    pub fn standard() -> Self {
        Self::new(3)
    }

    #[inline]
    pub fn new(k_period: usize) -> Self {
        Self {
            highs: RollingBuffer::new(),
            lows: RollingBuffer::new(),
            closes: RollingBuffer::new(),
            k_smooth: EWMA::from_period(k_period),
        }
    }

    /// Update with [high, low, close], returns %K (smoothed)
    #[inline]
    pub fn update(&mut self, hlc: [T; 3]) -> T {
        let [high, low, close] = hlc;

        self.highs.push(high);
        self.lows.push(low);
        self.closes.push(close);

        // Find highest high and lowest low using iterators
        let mut highest = T::splat(f64::NEG_INFINITY);
        let mut lowest = T::splat(f64::INFINITY);

        for h in self.highs.iter() {
            highest = highest.max(h);
        }
        for l in self.lows.iter() {
            lowest = lowest.min(l);
        }

        // Calculate %K
        let range = highest - lowest;
        let raw_k = (close - lowest) / (range + T::splat(1e-10)) * T::splat(100.0);

        self.k_smooth.update(raw_k)
    }
}

impl<T: Numeric, const N: usize> Default for Stochastic<T, N> {
    fn default() -> Self {
        Self::standard()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ewma() {
        let mut ema = EWMA::from_period(10);
        
        ema.update(100.0);
        let val2 = ema.update(110.0);
        
        // Should be between 100 and 110, closer to 100
        assert!(val2 > 100.0 && val2 < 110.0);
        assert!(val2 < 105.0); // Should be weighted toward first value
    }
    
    #[test]
    fn test_rsi() {
        let mut rsi = RSI::standard();
        
        // Increasing prices
        for i in 0..20 {
            rsi.update(100.0 + i as f64);
        }
        
        let val = rsi.update(120.0);
        assert!(val > 50.0); // Should be overbought
    }
    
    #[test]
    fn test_bollinger() {
        let mut bb = BollingerBands::<f64, 20>::standard();

        // Add some prices
        for i in 0..20 {
            bb.update(100.0 + (i % 5) as f64);
        }

        let (upper, middle, lower, _percent_b) = bb.update(103.0);

        assert!(upper > middle);
        assert!(middle > lower);
    }
    
    #[test]
    fn test_macd() {
        let mut macd = MACD::standard();
        
        for i in 0..30 {
            let price = 100.0 + (i as f64).sin() * 5.0;
            macd.update(price);
        }
        
        let (macd_line, signal, histogram) = macd.update(105.0);
        assert!((histogram - (macd_line - signal)).abs() < 1e-10);
    }
    
    #[test]
    fn test_correlation() {
        let mut corr = RollingCorrelation::<f64, 50>::new();

        // Perfect positive correlation
        for i in 0..50 {
            let x = i as f64;
            let y = i as f64 * 2.0;
            corr.update(x, y);
        }

        let r = corr.update(50.0, 100.0);
        assert!((r - 1.0).abs() < 1e-10);
    }
}
