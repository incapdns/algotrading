/// Cross-feed price discrepancy detection
pub struct FeedDiscrepancy {
    pub absolute_diff: f64,
    pub relative_diff: f64,
    pub z_score: f64,
    pub is_anomaly: bool,
}

/// Compare prices from two feeds at the same timestamp
#[inline]
pub fn compare_feed_prices(
    price_a: f64,
    price_b: f64,
    historical_std: f64,
    threshold_sigmas: f64,
) -> FeedDiscrepancy {
    let mid = (price_a + price_b) / 2.0;
    let absolute_diff = (price_a - price_b).abs();
    let relative_diff = if mid > 1e-10 {
        absolute_diff / mid
    } else {
        0.0
    };
    
    let z_score = if historical_std > 1e-10 {
        absolute_diff / historical_std
    } else {
        0.0
    };
    
    let is_anomaly = z_score > threshold_sigmas;
    
    FeedDiscrepancy {
        absolute_diff,
        relative_diff,
        z_score,
        is_anomaly,
    }
}

/// Detect stale quotes (no updates for too long)
#[repr(align(64))]
pub struct StalenessDetector {
    last_update_time: f64,
    last_price: f64,
    max_staleness_ms: f64,
    staleness_events: usize,
}

impl StalenessDetector {
    #[inline]
    pub fn new(max_staleness_ms: f64) -> Self {
        Self {
            last_update_time: 0.0,
            last_price: 0.0,
            max_staleness_ms,
            staleness_events: 0,
        }
    }
    
    /// Check if data is stale at current time
    #[inline]
    pub fn check_update(&mut self, current_time: f64, price: f64) -> bool {
        let is_stale = if self.last_update_time > 0.0 {
            (current_time - self.last_update_time) > self.max_staleness_ms
        } else {
            false
        };
        
        if is_stale {
            self.staleness_events += 1;
        }
        
        self.last_update_time = current_time;
        self.last_price = price;
        
        is_stale
    }
    
    #[inline]
    pub fn staleness_count(&self) -> usize {
        self.staleness_events
    }
}

/// Detect quote reversals (bid > ask)
#[inline]
pub fn detect_crossed_market(bid: f64, ask: f64) -> bool {
    bid > ask && (bid - ask).abs() > 1e-10
}

/// Detect locked market (bid == ask)
#[inline]
pub fn detect_locked_market(bid: f64, ask: f64, tolerance: f64) -> bool {
    (bid - ask).abs() < tolerance
}

/// Quote quality score (0 to 1)
pub struct QuoteQuality {
    pub spread_score: f64,      // Tighter is better
    pub depth_score: f64,        // More size is better
    pub staleness_score: f64,    // Fresher is better
    pub consistency_score: f64,  // Less volatile is better
    pub overall_score: f64,
}

#[inline]
pub fn compute_quote_quality(
    bid: f64,
    ask: f64,
    bid_size: f64,
    ask_size: f64,
    time_since_update_ms: f64,
    price_volatility: f64,
    reference_spread: f64,
    reference_depth: f64,
) -> QuoteQuality {
    let mid = (bid + ask) / 2.0;
    let spread = ask - bid;
    
    // Spread score (0 = very wide, 1 = tight)
    let spread_score = if reference_spread > 1e-10 {
        (1.0 - (spread / reference_spread).min(1.0)).max(0.0)
    } else {
        0.5
    };
    
    // Depth score (0 = thin, 1 = deep)
    let total_depth = bid_size + ask_size;
    let depth_score = if reference_depth > 1e-10 {
        (total_depth / reference_depth).min(1.0)
    } else {
        0.5
    };
    
    // Staleness score (exponential decay)
    let staleness_score = (-time_since_update_ms / 1000.0).exp();
    
    // Consistency score (inverse of volatility)
    let consistency_score = if price_volatility > 1e-10 {
        1.0 / (1.0 + price_volatility / mid)
    } else {
        1.0
    };
    
    // Weighted average
    let overall_score = 0.3 * spread_score 
                      + 0.2 * depth_score 
                      + 0.3 * staleness_score
                      + 0.2 * consistency_score;
    
    QuoteQuality {
        spread_score,
        depth_score,
        staleness_score,
        consistency_score,
        overall_score,
    }
}

/// Detect price jumps (beyond expected volatility)
#[repr(align(64))]
pub struct JumpDetector {
    last_price: f64,
    volatility: f64,
    jump_threshold_sigmas: f64,
    jumps_detected: usize,
}

impl JumpDetector {
    #[inline]
    pub fn new(jump_threshold_sigmas: f64) -> Self {
        Self {
            last_price: 0.0,
            volatility: 0.0,
            jump_threshold_sigmas,
            jumps_detected: 0,
        }
    }
    
    #[inline]
    pub fn check_price(&mut self, price: f64, current_volatility: f64) -> bool {
        self.volatility = current_volatility;
        
        if self.last_price > 0.0 && self.volatility > 1e-10 {
            let log_return = (price / self.last_price).ln();
            let expected_move = self.volatility;
            
            if log_return.abs() > self.jump_threshold_sigmas * expected_move {
                self.jumps_detected += 1;
                self.last_price = price;
                return true;
            }
        }
        
        self.last_price = price;
        false
    }
    
    #[inline]
    pub fn jump_count(&self) -> usize {
        self.jumps_detected
    }
}

/// Sequence number gap detection (for exchange feeds)
#[repr(align(64))]
pub struct SequenceMonitor {
    last_sequence: u64,
    gaps_detected: usize,
    total_messages: usize,
}

impl SequenceMonitor {
    #[inline]
    pub const fn new() -> Self {
        Self {
            last_sequence: 0,
            gaps_detected: 0,
            total_messages: 0,
        }
    }
    
    #[inline]
    pub fn check_sequence(&mut self, sequence: u64) -> Option<u64> {
        self.total_messages += 1;
        
        if self.last_sequence > 0 {
            let expected = self.last_sequence + 1;
            if sequence > expected {
                self.gaps_detected += 1;
                let gap_size = sequence - expected;
                self.last_sequence = sequence;
                return Some(gap_size);
            } else if sequence < expected {
                // Out of order or duplicate
                return Some(0); // Signal irregularity
            }
        }
        
        self.last_sequence = sequence;
        None
    }
    
    #[inline]
    pub fn gap_rate(&self) -> f64 {
        if self.total_messages > 0 {
            self.gaps_detected as f64 / self.total_messages as f64
        } else {
            0.0
        }
    }
}

impl Default for SequenceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Time synchronization check between feeds
pub struct TimeSyncChecker {
    reference_times: Vec<f64>,
    feed_times: Vec<f64>,
}

impl TimeSyncChecker {
    pub fn new() -> Self {
        Self {
            reference_times: Vec::new(),
            feed_times: Vec::new(),
        }
    }
    
    pub fn add_observation(&mut self, reference_time: f64, feed_time: f64) {
        self.reference_times.push(reference_time);
        self.feed_times.push(feed_time);
    }
    
    /// Compute clock skew and drift
    pub fn analyze_skew(&self) -> (f64, f64) {
        let n = self.reference_times.len();
        if n < 2 {
            return (0.0, 0.0);
        }
        
        // Linear regression: feed_time = offset + drift * reference_time
        let n_f = n as f64;
        let mean_ref: f64 = self.reference_times.iter().sum::<f64>() / n_f;
        let mean_feed: f64 = self.feed_times.iter().sum::<f64>() / n_f;
        
        let mut cov = 0.0;
        let mut var = 0.0;
        
        for i in 0..n {
            let dx = self.reference_times[i] - mean_ref;
            let dy = self.feed_times[i] - mean_feed;
            cov += dx * dy;
            var += dx * dx;
        }
        
        let drift = if var > 1e-10 { cov / var } else { 1.0 };
        let offset = mean_feed - drift * mean_ref;
        
        (offset, drift)
    }
}

impl Default for TimeSyncChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// NBBO (National Best Bid Offer) consistency checker
pub struct NBBOValidator {}

impl NBBOValidator {
    pub fn new() -> Self {
        Self {}
    }
    
    /// Check if NBBO is consistent across feeds
    pub fn validate_nbbo(
        &self,
        feed_quotes: &[(f64, f64)], // (bid, ask) pairs
    ) -> (bool, Option<usize>) {
        if feed_quotes.len() < 2 {
            return (true, None);
        }
        
        // Compute true NBBO
        let best_bid = feed_quotes.iter()
            .map(|(bid, _)| bid)
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        
        let best_ask = feed_quotes.iter()
            .map(|(_, ask)| ask)
            .copied()
            .fold(f64::INFINITY, f64::min);
        
        // Check each feed
        for (i, &(bid, ask)) in feed_quotes.iter().enumerate() {
            // Feed bid should not exceed best bid
            if bid > best_bid + 1e-10 {
                return (false, Some(i));
            }
            // Feed ask should not be below best ask
            if ask < best_ask - 1e-10 {
                return (false, Some(i));
            }
        }
        
        (true, None)
    }
}

impl Default for NBBOValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Tick size compliance checker
#[inline]
pub fn check_tick_size_compliance(price: f64, tick_size: f64) -> bool {
    let remainder = price % tick_size;
    remainder.abs() < 1e-10 || (tick_size - remainder).abs() < 1e-10
}

/// Outlier detection via median absolute deviation (MAD)
/// More robust than z-score for heavy-tailed distributions
#[inline]
pub fn mad_outlier_score(value: f64, values: &[f64]) -> f64 {
    if values.len() < 3 {
        return 0.0;
    }
    
    // Compute median using total_cmp to handle NaN correctly
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let median = sorted[sorted.len() / 2];

    // Compute MAD
    let mut abs_devs: Vec<f64> = values.iter()
        .map(|x| (x - median).abs())
        .collect();
    abs_devs.sort_by(|a, b| a.total_cmp(b));
    let mad = abs_devs[abs_devs.len() / 2];
    
    if mad > 1e-10 {
        (value - median).abs() / (1.4826 * mad) // Scale factor for normal distribution
    } else {
        0.0
    }
}

/// Trade-quote latency analyzer
#[repr(align(64))]
pub struct TradeQuoteLatency {
    last_quote_time: f64,
    latencies: Vec<f64>,
    max_history: usize,
}

impl TradeQuoteLatency {
    pub fn new(max_history: usize) -> Self {
        Self {
            last_quote_time: 0.0,
            latencies: Vec::with_capacity(max_history),
            max_history,
        }
    }
    
    pub fn update_quote(&mut self, time: f64) {
        self.last_quote_time = time;
    }
    
    pub fn record_trade(&mut self, trade_time: f64) {
        if self.last_quote_time > 0.0 {
            let latency = trade_time - self.last_quote_time;
            if latency >= 0.0 {
                self.latencies.push(latency);
                if self.latencies.len() > self.max_history {
                    self.latencies.remove(0);
                }
            }
        }
    }
    
    pub fn average_latency(&self) -> f64 {
        if self.latencies.is_empty() {
            return 0.0;
        }
        self.latencies.iter().sum::<f64>() / self.latencies.len() as f64
    }
    
    pub fn percentile_latency(&self, p: f64) -> f64 {
        if self.latencies.is_empty() {
            return 0.0;
        }
        let mut sorted = self.latencies.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let idx = ((p * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1);
        sorted[idx]
    }
}

/// Volume consistency checker (detect wash trades)
pub struct VolumeAnomalyDetector {
    typical_volume_mean: f64,
    typical_volume_std: f64,
}

impl VolumeAnomalyDetector {
    pub fn new(typical_volume_mean: f64, typical_volume_std: f64) -> Self {
        Self {
            typical_volume_mean,
            typical_volume_std,
        }
    }
    
    #[inline]
    pub fn is_anomalous(&self, volume: f64, threshold_sigmas: f64) -> bool {
        if self.typical_volume_std < 1e-10 {
            return false;
        }
        
        let z_score = (volume - self.typical_volume_mean).abs() / self.typical_volume_std;
        z_score > threshold_sigmas
    }
    
    /// Detect suspiciously round volumes (often fake)
    #[inline]
    pub fn is_suspiciously_round(volume: f64) -> bool {
        let log_vol = volume.log10();
        let frac = log_vol.fract().abs();
        frac < 0.05 || frac > 0.95 // Very close to powers of 10
    }
}

/// Price improvement analysis (vs NBBO)
#[inline]
pub fn price_improvement(
    trade_price: f64,
    side: i8, // 1 for buy, -1 for sell
    nbbo_bid: f64,
    nbbo_ask: f64,
) -> f64 {
    match side {
        1 => {
            // Buy: improvement if paid less than NBBO ask
            (nbbo_ask - trade_price).max(0.0)
        }
        -1 => {
            // Sell: improvement if received more than NBBO bid
            (trade_price - nbbo_bid).max(0.0)
        }
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_feed_comparison() {
        let disc = compare_feed_prices(100.0, 100.5, 0.1, 3.0);
        assert!(disc.absolute_diff > 0.0);
        assert!(disc.z_score > 3.0); // Should flag as anomaly
    }
    
    #[test]
    fn test_staleness_detector() {
        let mut detector = StalenessDetector::new(100.0);
        
        detector.check_update(0.0, 100.0);
        assert!(!detector.check_update(50.0, 100.0)); // Not stale
        assert!(detector.check_update(200.0, 100.0)); // Stale
    }
    
    #[test]
    fn test_crossed_market() {
        assert!(detect_crossed_market(100.5, 100.0)); // Crossed
        assert!(!detect_crossed_market(100.0, 100.5)); // Normal
    }
    
    #[test]
    fn test_mad_outlier() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let score = mad_outlier_score(10.0, &values);
        assert!(score > 3.0); // Should be outlier
    }
}