/// Rolling Value at Risk (VaR) calculator
/// Historical method using sorted returns
#[repr(align(64))]
pub struct RollingVaR<const N: usize> {
    returns: [f64; N],
    sorted: [f64; N],
    head: usize,
    count: usize,
}

impl<const N: usize> RollingVaR<N> {
    #[inline]
    pub const fn new() -> Self {
        Self {
            returns: [0.0; N],
            sorted: [0.0; N],
            head: 0,
            count: 0,
        }
    }
    
    /// Update with new return, get VaR at confidence level
    /// confidence: 0.95 = 95% VaR, 0.99 = 99% VaR
    #[inline]
    pub fn update(&mut self, return_val: f64, confidence: f64) -> f64 {
        // Add new return
        if self.count < N {
            self.count += 1;
        }
        
        self.returns[self.head] = return_val;
        self.head = (self.head + 1) % N;
        
        // Copy and sort active returns
        self.sorted[..self.count].copy_from_slice(&self.returns[..self.count]);
        self.sorted[..self.count].sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Get quantile (negative returns are losses)
        let index = ((1.0 - confidence) * self.count as f64) as usize;
        -self.sorted[index.min(self.count - 1)]
    }
    
    /// Get Expected Shortfall (CVaR) - average loss beyond VaR
    #[inline]
    pub fn expected_shortfall(&mut self, confidence: f64) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        
        // Sort returns
        self.sorted[..self.count].copy_from_slice(&self.returns[..self.count]);
        self.sorted[..self.count].sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = ((1.0 - confidence) * self.count as f64) as usize;
        
        // Average of tail losses
        let mut sum = 0.0;
        for i in 0..=index.min(self.count - 1) {
            sum += self.sorted[i];
        }
        
        -sum / (index + 1) as f64
    }
}

/// Parametric VaR using Cornish-Fisher expansion
/// Accounts for skewness and kurtosis
pub struct ParametricVaR {
    mean: f64,
    std_dev: f64,
    skewness: f64,
    excess_kurtosis: f64,
}

impl ParametricVaR {
    /// Calculate from return statistics
    pub fn new(returns: &[f64]) -> Self {
        let n = returns.len() as f64;
        let mean = returns.iter().sum::<f64>() / n;
        
        let mut m2 = 0.0;
        let mut m3 = 0.0;
        let mut m4 = 0.0;
        
        for &r in returns {
            let diff = r - mean;
            let diff2 = diff * diff;
            m2 += diff2;
            m3 += diff * diff2;
            m4 += diff2 * diff2;
        }
        
        let variance = m2 / (n - 1.0);
        let std_dev = variance.sqrt();
        let skewness = (m3 / n) / variance.powf(1.5);
        let excess_kurtosis = (m4 / n) / (variance * variance) - 3.0;
        
        Self {
            mean,
            std_dev,
            skewness,
            excess_kurtosis,
        }
    }
    
    /// Compute VaR using Cornish-Fisher expansion
    #[inline]
    pub fn var(&self, confidence: f64) -> f64 {
        let z = normal_quantile(1.0 - confidence);
        let z2 = z * z;
        let z3 = z2 * z;
        
        // Cornish-Fisher expansion
        let adjusted_z = z 
            + (z2 - 1.0) * self.skewness / 6.0
            + (z3 - 3.0 * z) * self.excess_kurtosis / 24.0
            - (2.0 * z3 - 5.0 * z) * self.skewness * self.skewness / 36.0;
        
        -(self.mean + adjusted_z * self.std_dev)
    }
}

/// Kelly Criterion for optimal position sizing
pub struct Kelly;

impl Kelly {
    /// Calculate optimal fraction for binary outcomes
    /// win_prob: probability of winning
    /// win_ratio: average win / average loss
    #[inline]
    pub fn binary(win_prob: f64, win_ratio: f64) -> f64 {
        (win_prob * (win_ratio + 1.0) - 1.0) / win_ratio
    }
    
    /// Calculate optimal fraction for continuous returns
    /// Uses mean-variance approximation
    #[inline]
    pub fn continuous(mean_return: f64, variance: f64) -> f64 {
        if variance <= 0.0 {
            return 0.0;
        }
        mean_return / variance
    }
    
    /// Half-Kelly (more conservative)
    #[inline]
    pub fn half_kelly(win_prob: f64, win_ratio: f64) -> f64 {
        Self::binary(win_prob, win_ratio) * 0.5
    }
}

/// Rolling Sharpe Ratio
#[repr(align(64))]
pub struct RollingSharpe<const N: usize> {
    returns: [f64; N],
    head: usize,
    count: usize,
    sum: f64,
    sum_sq: f64,
}

impl<const N: usize> RollingSharpe<N> {
    #[inline]
    pub const fn new() -> Self {
        Self {
            returns: [0.0; N],
            head: 0,
            count: 0,
            sum: 0.0,
            sum_sq: 0.0,
        }
    }
    
    /// Update with new return, get annualized Sharpe ratio
    /// risk_free_rate: annualized risk-free rate
    #[inline]
    pub fn update(&mut self, return_val: f64, risk_free_rate: f64) -> f64 {
        // Remove old if buffer full
        if self.count >= N {
            let old = self.returns[self.head];
            self.sum -= old;
            self.sum_sq -= old * old;
        } else {
            self.count += 1;
        }

        // Add new
        self.returns[self.head] = return_val;
        self.sum += return_val;
        self.sum_sq += return_val * return_val;

        self.head = (self.head + 1) % N;

        // Require at least 2 returns for std_dev
        if self.count < 2 {
            return 0.0;
        }

        let n = self.count as f64;
        let mean = self.sum / n;
        let variance = (self.sum_sq / n) - mean * mean;
        let std_dev = variance.max(0.0).sqrt().max(1e-16); // prevent div-by-zero

        // Excess return (daily)
        let excess_return = mean - (risk_free_rate / 252.0);

        // Annualized Sharpe assuming 252 trading days
        (excess_return / std_dev) * 252.0_f64.sqrt()
    }
}

/// Rolling Sortino Ratio (only penalizes downside deviation)
#[repr(align(64))]
pub struct RollingSortino<const N: usize> {
    returns: [f64; N],
    head: usize,
    count: usize,
    sum: f64,
    downside_sum_sq: f64,
}

impl<const N: usize> RollingSortino<N> {
    #[inline]
    pub const fn new() -> Self {
        Self {
            returns: [0.0; N],
            head: 0,
            count: 0,
            sum: 0.0,
            downside_sum_sq: 0.0,
        }
    }
    
    /// Update with new return, get annualized Sortino ratio
    #[inline]
    pub fn update(&mut self, return_val: f64, target_return: f64) -> f64 {
        // Remove old
        if self.count >= N {
            let old = self.returns[self.head];
            self.sum -= old;
            if old < target_return {
                let diff = old - target_return;
                self.downside_sum_sq -= diff * diff;
            }
        } else {
            self.count += 1;
        }
        
        // Add new
        self.returns[self.head] = return_val;
        self.sum += return_val;
        if return_val < target_return {
            let diff = return_val - target_return;
            self.downside_sum_sq += diff * diff;
        }
        self.head = (self.head + 1) % N;
        
        // Calculate
        let n = self.count as f64;
        let mean = self.sum / n;
        let downside_dev = (self.downside_sum_sq / n).sqrt();
        
        if downside_dev < 1e-10 {
            return 0.0;
        }
        
        // Annualize
        let excess_return = mean - (target_return / 252.0);
        (excess_return / downside_dev) * (252.0_f64).sqrt()
    }
}

/// Maximum Drawdown tracker
#[repr(align(64))]
pub struct MaxDrawdown<const N: usize> {
    equity_curve: [f64; N],
    head: usize,
    count: usize,
}

impl<const N: usize> MaxDrawdown<N> {
    #[inline]
    pub const fn new() -> Self {
        Self {
            equity_curve: [0.0; N],
            head: 0,
            count: 0,
        }
    }
    
    /// Update with new equity value, returns (max_dd, current_dd)
    #[inline]
    pub fn update(&mut self, equity: f64) -> (f64, f64) {
        if self.count < N {
            self.count += 1;
        }
        
        self.equity_curve[self.head] = equity;
        self.head = (self.head + 1) % N;
        
        let mut peak = f64::NEG_INFINITY;
        let mut max_dd = 0.0;
        
        // Find running maximum and maximum drawdown
        for i in 0..self.count {
            let val = self.equity_curve[i];
            if val > peak {
                peak = val;
            }
            let dd = (peak - val) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }
        
        // Current drawdown from most recent peak
        let current_dd = (peak - equity) / peak;
        
        (max_dd, current_dd)
    }
    
    /// Calmar ratio: annualized return / max drawdown
    #[inline]
    pub fn calmar_ratio(&self, annualized_return: f64) -> f64 {
        let (max_dd, _) = self.max_drawdown();
        if max_dd < 1e-10 {
            return 0.0;
        }
        annualized_return / max_dd
    }
    
    fn max_drawdown(&self) -> (f64, f64) {
        let mut peak = f64::NEG_INFINITY;
        let mut max_dd = 0.0;
        
        for i in 0..self.count {
            let val = self.equity_curve[i];
            if val > peak {
                peak = val;
            }
            let dd = (peak - val) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }
        
        let last_idx = if self.head == 0 { self.count - 1 } else { self.head - 1 };
        let current_dd = (peak - self.equity_curve[last_idx]) / peak;
        
        (max_dd, current_dd)
    }
}

// Use consolidated normal_quantile from probability::utils
use crate::probability::utils::normal_quantile;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rolling_var() {
        let mut var = RollingVaR::<100>::new();
        
        // Add some returns
        for i in 0..50 {
            let ret = (i as f64 - 25.0) / 100.0; // -0.25 to 0.24
            var.update(ret, 0.95);
        }
        
        let var_95 = var.update(0.1, 0.95);
        assert!(var_95 > 0.0);
    }
    
    #[test]
    fn test_kelly() {
        // 60% win rate, 2:1 win/loss ratio
        let f = Kelly::binary(0.6, 2.0);
        assert!(f > 0.0 && f <= 1.0);
        
        // Should be: (0.6 * 3 - 1) / 2 = 0.4
        assert!((f - 0.4).abs() < 1e-10);
    }
    
    #[test]
    fn test_sharpe() {
        let mut sharpe = RollingSharpe::<100>::new();
        
        // Positive returns should give positive Sharpe
        for _ in 0..50 {
            sharpe.update(0.01, 0.02);
        }
        
        let ratio = sharpe.update(0.01, 0.02);
        println!("Sharpe Ratio: {}", ratio);
        assert!(ratio > 0.0);
    }
    
    #[test]
    fn test_max_drawdown() {
        let mut dd = MaxDrawdown::<100>::new();
        
        dd.update(100.0);
        dd.update(110.0);
        dd.update(90.0);  // 18% drawdown from peak
        
        let (max_dd, current_dd) = dd.update(95.0);
        
        assert!((max_dd - 0.1818).abs() < 0.01);
        assert!((current_dd - 0.1363).abs() < 0.01);
    }
}