// src/stats/core.rs

use crate::numeric::Numeric;

/// Helper function to process SIMD chunks and handle remainder
/// Eliminates code duplication across statistical functions
#[inline(always)]
fn process_chunked_simd<T: Numeric>(
    data: &[f64],
    mut process: impl FnMut(T),
) {
    let lanes = T::LANES;

    // Process full SIMD chunks
    for chunk in data.chunks_exact(lanes) {
        process(T::from_slice(chunk));
    }

    // Handle remainder elements (if data.len() % lanes != 0)
    let remainder_start = (data.len() / lanes) * lanes;
    let remainder = &data[remainder_start..];
    if !remainder.is_empty() {
        // Zero-padded temporary buffer for partial SIMD load
        let mut tmp = [0.0f64; 8]; // Support up to f64x8
        for (i, &val) in remainder.iter().enumerate() {
            tmp[i] = val;
        }
        process(T::from_slice(&tmp[..lanes]));
    }
}

#[inline]
pub fn mean<T: Numeric>(data: &[f64]) -> f64 {
    let mut sum = T::zero();

    process_chunked_simd::<T>(
        data,
        |v| sum += v,
    );

    sum.reduce_sum() / (data.len() as f64)
}

#[inline]
pub fn variance<T: Numeric>(data: &[f64]) -> f64 {
    let m = mean::<T>(data);
    let mut sum_sq = T::zero();
    let m_vec = T::splat(m);

    process_chunked_simd::<T>(
        data,
        |v| {
            let diff = v - m_vec;
            sum_sq += diff * diff;
        },
    );

    sum_sq.reduce_sum() / (data.len() as f64)
}

#[inline]
pub fn stddev<T: Numeric>(data: &[f64]) -> f64 {
    variance::<T>(data).sqrt()
}

#[inline]
pub fn covariance<T: Numeric>(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let mx = mean::<T>(x);
    let my = mean::<T>(y);
    let lanes = T::LANES;
    let mut sum = T::zero();
    let mx_vec = T::splat(mx);
    let my_vec = T::splat(my);

    // Process full SIMD chunks
    for (cx, cy) in x.chunks_exact(lanes).zip(y.chunks_exact(lanes)) {
        let vx = T::from_slice(cx);
        let vy = T::from_slice(cy);
        sum += (vx - mx_vec) * (vy - my_vec);
    }

    // Handle remainder
    let remainder_start = (x.len() / lanes) * lanes;
    let rem_x = &x[remainder_start..];
    let rem_y = &y[remainder_start..];
    if !rem_x.is_empty() {
        let mut tmpx = [0.0f64; 8];
        let mut tmpy = [0.0f64; 8];
        for (i, (&xi, &yi)) in rem_x.iter().zip(rem_y.iter()).enumerate() {
            tmpx[i] = xi;
            tmpy[i] = yi;
        }
        let vx = T::from_slice(&tmpx[..lanes]);
        let vy = T::from_slice(&tmpy[..lanes]);
        sum += (vx - mx_vec) * (vy - my_vec);
    }

    sum.reduce_sum() / (x.len() as f64)
}

#[inline]
pub fn correlation<T: Numeric>(x: &[f64], y: &[f64]) -> f64 {
    covariance::<T>(x, y) / (stddev::<T>(x) * stddev::<T>(y))
}

#[inline]
pub fn zscore<T: Numeric>(data: &[f64]) -> Vec<f64> {
    let m = mean::<T>(data);
    let s = stddev::<T>(data);
    data.iter().map(|&v| (v - m) / s).collect()
}

#[inline]
pub fn skewness<T: Numeric>(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    let m = mean::<T>(data);
    let s = stddev::<T>(data);
    let mut sum = T::zero();
    let m_vec = T::splat(m);
    let s_vec = T::splat(s);

    process_chunked_simd::<T>(
        data,
        |v| {
            let z = (v - m_vec) / s_vec;
            sum += z * z * z;
        },
    );

    sum.reduce_sum() / n
}

#[inline]
pub fn kurtosis<T: Numeric>(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    let m = mean::<T>(data);
    let s = stddev::<T>(data);
    let mut sum = T::zero();
    let m_vec = T::splat(m);
    let s_vec = T::splat(s);

    process_chunked_simd::<T>(
        data,
        |v| {
            let z = (v - m_vec) / s_vec;
            sum += z * z * z * z;
        },
    );

    sum.reduce_sum() / n - 3.0 // excess kurtosis
}

/// Percentile using linear interpolation
pub fn percentile<T: Numeric>(data: &[f64], p: f64) -> f64 {
    assert!(!data.is_empty() && p >= 0.0 && p <= 1.0);
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let idx = p * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let w = idx - lo as f64;
        sorted[lo] * (1.0 - w) + sorted[hi] * w
    }
}

/// Returns min and max in one pass (SIMD-friendly)
#[inline]
pub fn minmax<T: Numeric>(data: &[f64]) -> (f64, f64) {
    let mut min_v = T::splat(f64::INFINITY);
    let mut max_v = T::splat(f64::NEG_INFINITY);

    process_chunked_simd::<T>(
        data,
        |v| {
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        },
    );

    (min_v.reduce_min(), max_v.reduce_max())
}

/// Normalized data to [0, 1]
#[inline]
pub fn minmax_scale<T: Numeric>(data: &[f64]) -> Vec<f64> {
    let (min, max) = minmax::<T>(data);
    let range = max - min;
    data.iter().map(|&x| (x - min) / range).collect()
}
