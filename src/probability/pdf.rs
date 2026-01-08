/// Kernel Density Estimation (KDE) with Gaussian kernel
/// 
/// Stack-allocated for performance. Window size N is compile-time constant.
#[repr(align(64))]
pub struct KernelDensity<const N: usize> {
    samples: [f64; N],
    count: usize,
    bandwidth: f64,
}

impl<const N: usize> KernelDensity<N> {
    /// Create new KDE with automatic bandwidth selection (Silverman's rule)
    #[inline]
    pub fn new(samples: &[f64]) -> Self {
        assert!(samples.len() <= N, "Too many samples for window size");
        
        let mut kde_samples = [0.0; N];
        kde_samples[..samples.len()].copy_from_slice(samples);
        
        // Silverman's rule of thumb: h = 1.06 * σ * n^(-1/5)
        let bandwidth = Self::silverman_bandwidth(samples);
        
        Self {
            samples: kde_samples,
            count: samples.len(),
            bandwidth,
        }
    }
    
    /// Create with explicit bandwidth
    #[inline]
    pub fn with_bandwidth(samples: &[f64], bandwidth: f64) -> Self {
        assert!(samples.len() <= N, "Too many samples for window size");
        
        let mut kde_samples = [0.0; N];
        kde_samples[..samples.len()].copy_from_slice(samples);
        
        Self {
            samples: kde_samples,
            count: samples.len(),
            bandwidth,
        }
    }
    
    /// Estimate PDF at point x
    #[inline]
    pub fn density(&self, x: f64) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        
        let mut sum = 0.0;
        let h = self.bandwidth;
        let norm = 1.0 / (self.count as f64 * h * (2.0 * std::f64::consts::PI).sqrt());
        
        for i in 0..self.count {
            let u = (x - self.samples[i]) / h;
            sum += (-0.5 * u * u).exp();
        }
        
        norm * sum
    }
    
    /// Log-likelihood of observation under this distribution
    #[inline]
    pub fn log_likelihood(&self, x: f64) -> f64 {
        let density = self.density(x);
        if density > 1e-300 {
            density.ln()
        } else {
            -690.0 // Approximate ln(1e-300)
        }
    }
    
    /// Silverman's bandwidth selection
    fn silverman_bandwidth(samples: &[f64]) -> f64 {
        if samples.len() < 2 {
            return 0.1;
        }
        
        let n = samples.len() as f64;
        
        // Compute sample standard deviation
        let mean = samples.iter().sum::<f64>() / n;
        let variance = samples.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();
        
        // Silverman's rule
        1.06 * std_dev * n.powf(-0.2)
    }
}

/// Histogram-based PDF estimation (faster but less smooth)
#[repr(align(64))]
pub struct Histogram<const BINS: usize> {
    counts: [u32; BINS],
    min: f64,
    max: f64,
    bin_width: f64,
    total_count: u32,
}

impl<const BINS: usize> Histogram<BINS> {
    /// Create histogram from data
    pub fn new(data: &[f64]) -> Self {
        assert!(!data.is_empty(), "Cannot create histogram from empty data");
        
        let min = data.iter().copied().fold(f64::INFINITY, f64::min);
        let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let bin_width = (max - min) / BINS as f64;
        
        let mut counts = [0u32; BINS];
        
        for &value in data {
            let bin = ((value - min) / bin_width).floor() as usize;
            let bin = bin.min(BINS - 1); // Clamp to last bin
            counts[bin] += 1;
        }
        
        Self {
            counts,
            min,
            max,
            bin_width,
            total_count: data.len() as u32,
        }
    }
    
    /// Estimate density at point x (piecewise constant)
    #[inline]
    pub fn density(&self, x: f64) -> f64 {
        if x < self.min || x >= self.max {
            return 0.0;
        }
        
        let bin = ((x - self.min) / self.bin_width).floor() as usize;
        let bin = bin.min(BINS - 1);
        
        let count = self.counts[bin] as f64;
        count / (self.total_count as f64 * self.bin_width)
    }
    
    /// Get bin index for value
    #[inline]
    pub fn bin_index(&self, x: f64) -> Option<usize> {
        if x < self.min || x >= self.max {
            return None;
        }
        
        let bin = ((x - self.min) / self.bin_width).floor() as usize;
        Some(bin.min(BINS - 1))
    }
    
    /// Get count in bin
    #[inline]
    pub fn bin_count(&self, bin: usize) -> u32 {
        if bin < BINS {
            self.counts[bin]
        } else {
            0
        }
    }
}

/// Empirical CDF (for quantile calculations)
#[repr(align(64))]
pub struct EmpiricalCDF<const N: usize> {
    sorted_values: [f64; N],
    count: usize,
}

impl<const N: usize> EmpiricalCDF<N> {
    /// Create from data (will be sorted internally)
    pub fn new(data: &[f64]) -> Self {
        assert!(data.len() <= N, "Too many samples");
        assert!(!data.is_empty(), "Cannot create CDF from empty data");
        
        let mut sorted_values = [0.0; N];
        sorted_values[..data.len()].copy_from_slice(data);
        
        // Sort the values using total_cmp to handle NaN correctly
        sorted_values[..data.len()].sort_by(|a, b| a.total_cmp(b));
        
        Self {
            sorted_values,
            count: data.len(),
        }
    }
    
    /// Evaluate CDF at point x
    #[inline]
    pub fn cdf(&self, x: f64) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        
        // Binary search for position using total_cmp to handle NaN correctly
        let pos = self.sorted_values[..self.count]
            .binary_search_by(|probe| probe.total_cmp(&x))
            .unwrap_or_else(|e| e);
        
        pos as f64 / self.count as f64
    }
    
    /// Get quantile (inverse CDF)
    #[inline]
    pub fn quantile(&self, p: f64) -> f64 {
        assert!((0.0..=1.0).contains(&p), "Quantile must be in [0, 1]");
        
        if self.count == 0 {
            return 0.0;
        }
        
        let index = (p * (self.count - 1) as f64).round() as usize;
        self.sorted_values[index.min(self.count - 1)]
    }
    
    /// Get median (50th percentile)
    #[inline]
    pub fn median(&self) -> f64 {
        self.quantile(0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_kde() {
        let samples = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let kde = KernelDensity::<100>::new(&samples);
        
        // Density should be higher near sample points
        let density_at_sample = kde.density(2.0);
        let density_far = kde.density(10.0);
        
        assert!(density_at_sample > density_far);
        assert!(density_at_sample > 0.0);
    }
    
    #[test]
    fn test_histogram() {
        let data = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let hist = Histogram::<10>::new(&data);
        
        assert!(hist.density(0.5) > 0.0);
        assert_eq!(hist.density(10.0), 0.0); // Outside range
    }
    
    #[test]
    fn test_empirical_cdf() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cdf = EmpiricalCDF::<100>::new(&data);
        
        assert_eq!(cdf.cdf(0.0), 0.0);
        assert_eq!(cdf.cdf(3.0), 0.4); // 2/5
        assert_eq!(cdf.cdf(10.0), 1.0);
        
        assert_eq!(cdf.median(), 3.0);
        assert_eq!(cdf.quantile(0.0), 1.0);
        assert_eq!(cdf.quantile(1.0), 5.0);
    }
}