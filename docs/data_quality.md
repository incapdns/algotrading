# Data Quality Module

Real-time data validation and anomaly detection for market data feeds.

## Overview

The data quality module provides tools for monitoring and validating market data in real-time. Critical for HFT and automated trading systems where data quality directly impacts P&L.

## Components

### Feed Discrepancy Detection

Compare prices from multiple feeds to detect anomalies.

```rust
use algotrading::data::*;

let discrepancy = compare_feed_prices(
    price_a: 100.50,
    price_b: 100.55,
    historical_std: 0.02,
    threshold_sigmas: 3.0,
);

if discrepancy.is_anomaly {
    println!("Alert! Z-score: {}", discrepancy.z_score);
}
```

**Fields:**
- `absolute_diff` - Absolute price difference
- `relative_diff` - Relative difference (percentage)
- `z_score` - Standard deviations from historical mean
- `is_anomaly` - Boolean flag if exceeds threshold

### Staleness Detection

Monitor if quotes are updating frequently enough.

```rust
let mut detector = StalenessDetector::new(100.0); // 100ms max staleness

if detector.check_update(current_time_ms, price) {
    println!("WARNING: Stale data detected");
}

let count = detector.staleness_count();
```

**Use Cases:**
- Detect feed interruptions
- Monitor quote update frequency
- Alert on connectivity issues

### Market Quality Checks

#### Crossed Market Detection

```rust
let is_crossed = detect_crossed_market(bid, ask);
// Returns true if bid > ask (invalid market)
```

#### Locked Market Detection

```rust
let is_locked = detect_locked_market(bid, ask, tolerance);
// Returns true if bid ≈ ask (within tolerance)
```

#### Quote Quality Score

Multi-factor scoring of quote quality (0 to 1).

```rust
let quality = compute_quote_quality(
    bid: 100.00,
    ask: 100.10,
    bid_size: 1000.0,
    ask_size: 1500.0,
    time_since_update_ms: 50.0,
    price_volatility: 0.02,
    reference_spread: 0.10,
    reference_depth: 2000.0,
);

println!("Overall quality: {:.2}%", quality.overall_score * 100.0);
```

**Components:**
- `spread_score` (30%) - Tighter spreads = better
- `depth_score` (20%) - More size = better
- `staleness_score` (30%) - Fresher = better
- `consistency_score` (20%) - Less volatile = better

### Price Jump Detection

Detect abnormal price movements.

```rust
let mut detector = JumpDetector::new(3.0); // 3-sigma threshold

if detector.check_price(new_price, current_volatility) {
    println!("Jump detected! Count: {}", detector.jump_count());
}
```

**Algorithm:**
- Compares log returns to expected volatility
- Flags moves beyond N standard deviations
- Adapts to current market volatility

### Sequence Gap Monitoring

For exchange feeds with sequence numbers.

```rust
let mut monitor = SequenceMonitor::new();

if let Some(gap_size) = monitor.check_sequence(sequence_num) {
    if gap_size > 0 {
        println!("Gap detected: {} messages missed", gap_size);
    }
}

let gap_rate = monitor.gap_rate();
```

**Features:**
- Detects missing messages
- Identifies out-of-order delivery
- Tracks gap statistics

### NBBO Validation

Validate National Best Bid/Offer consistency.

```rust
let validator = NBBOValidator::new();

let feed_quotes = vec![
    (100.00, 100.10), // Feed 1: (bid, ask)
    (99.95, 100.05),  // Feed 2
    (100.02, 100.12), // Feed 3
];

let (is_valid, violating_feed) = validator.validate_nbbo(&feed_quotes);

if !is_valid {
    println!("NBBO violation from feed {}", violating_feed.unwrap());
}
```

### Tick Size Compliance

Check if prices conform to exchange tick sizes.

```rust
let compliant = check_tick_size_compliance(
    price: 100.05,
    tick_size: 0.01,
);
```

### Outlier Detection (MAD)

Median Absolute Deviation - robust to heavy tails.

```rust
let values = vec![1.0, 2.0, 3.0, 4.0, 100.0]; // 100 is outlier
let score = mad_outlier_score(100.0, &values);

if score > 3.0 {
    println!("Outlier detected with MAD score: {}", score);
}
```

**Advantages over Z-score:**
- More robust to outliers
- Better for non-normal distributions
- Uses median instead of mean

### Trade-Quote Latency Analysis

Measure latency between quote updates and trades.

```rust
let mut latency = TradeQuoteLatency::new(1000); // Keep 1000 samples

latency.update_quote(quote_time);
latency.record_trade(trade_time);

let avg_latency = latency.average_latency();
let p99_latency = latency.percentile_latency(0.99);
```

### Volume Anomaly Detection

Detect unusual trading volumes (potential wash trades).

```rust
let detector = VolumeAnomalyDetector::new(
    typical_volume_mean: 10000.0,
    typical_volume_std: 2000.0,
);

if detector.is_anomalous(volume: 50000.0, threshold_sigmas: 3.0) {
    println!("Abnormal volume detected");
}

// Detect suspiciously round volumes
if VolumeAnomalyDetector::is_suspiciously_round(10000.0) {
    println!("Volume is suspiciously round (likely fake)");
}
```

### Time Synchronization

Check clock skew between feeds.

```rust
let mut checker = TimeSyncChecker::new();

// Add observations
for (ref_time, feed_time) in observations {
    checker.add_observation(ref_time, feed_time);
}

let (offset, drift) = checker.analyze_skew();
println!("Clock offset: {:.2}ms, drift: {:.6}", offset, drift);
```

### Price Improvement Analysis

Measure execution quality vs NBBO.

```rust
let improvement = price_improvement(
    trade_price: 100.08,
    side: 1, // 1 = buy, -1 = sell
    nbbo_bid: 100.00,
    nbbo_ask: 100.10,
);

println!("Price improvement: ${:.4}", improvement);
```

## Best Practices

### Real-time Monitoring

```rust
struct FeedMonitor {
    staleness: StalenessDetector,
    jump_detector: JumpDetector,
    sequence_monitor: SequenceMonitor,
}

impl FeedMonitor {
    fn check_quote(&mut self, quote: &Quote) -> bool {
        let mut is_valid = true;

        // Check staleness
        if self.staleness.check_update(quote.timestamp, quote.mid_price()) {
            warn!("Stale quote detected");
            is_valid = false;
        }

        // Check for price jumps
        if self.jump_detector.check_price(quote.mid_price(), quote.volatility) {
            warn!("Price jump detected");
        }

        // Check sequence
        if let Some(gap) = self.sequence_monitor.check_sequence(quote.sequence) {
            if gap > 0 {
                error!("Sequence gap: {}", gap);
                is_valid = false;
            }
        }

        // Check market quality
        if detect_crossed_market(quote.bid, quote.ask) {
            error!("Crossed market!");
            is_valid = false;
        }

        is_valid
    }
}
```

### Quality Dashboard

```rust
struct QualityMetrics {
    staleness_rate: f64,
    jump_rate: f64,
    gap_rate: f64,
    average_quality_score: f64,
}

fn compute_quality_metrics(monitor: &FeedMonitor) -> QualityMetrics {
    QualityMetrics {
        staleness_rate: monitor.staleness.staleness_count() as f64 / total_quotes,
        jump_rate: monitor.jump_detector.jump_count() as f64 / total_quotes,
        gap_rate: monitor.sequence_monitor.gap_rate(),
        average_quality_score: /* rolling average of quality scores */,
    }
}
```

## Performance Considerations

- All detectors use fixed-size buffers or O(1) updates
- No heap allocations in hot path
- Cache-friendly 64-byte alignment
- Suitable for microsecond-level latency requirements

## Common Patterns

### Multi-Feed Validation

```rust
let feeds = vec!["Feed A", "Feed B", "Feed C"];
let prices = vec![100.00, 100.05, 99.95];

// Find best price
let best_bid = prices.iter().copied().fold(f64::NEG_INFINITY, f64::max);

// Check for outliers
for (i, &price) in prices.iter().enumerate() {
    let score = mad_outlier_score(price, &prices);
    if score > 3.0 {
        println!("{} has outlier price", feeds[i]);
    }
}
```

### Alert Thresholds

```rust
const STALENESS_THRESHOLD_MS: f64 = 100.0;
const JUMP_THRESHOLD_SIGMA: f64 = 3.0;
const QUALITY_THRESHOLD: f64 = 0.7;

if quality.overall_score < QUALITY_THRESHOLD {
    // Pause trading or switch to backup feed
}
```

## Testing

The module includes comprehensive tests:

```bash
cargo test data::quality
```

Key test scenarios:
- Feed price discrepancies
- Staleness detection
- Crossed/locked markets
- MAD outlier detection
