/// Discrete Bayesian filter for N states
/// 
/// Maintains posterior probability distribution over states
/// and updates via Bayes rule with incoming observations.
#[repr(align(64))]
pub struct BayesianFilter<const N: usize> {
    /// Current posterior probabilities (prior for next update)
    posterior: [f64; N],
}

impl<const N: usize> BayesianFilter<N> {
    /// Create with uniform prior
    #[inline]
    pub fn new() -> Self {
        Self {
            posterior: [1.0 / N as f64; N],
        }
    }
    
    /// Create with custom prior
    #[inline]
    pub fn with_prior(prior: [f64; N]) -> Self {
        // Validate sum to 1
        let sum: f64 = prior.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Prior must sum to 1");
        
        Self { posterior: prior }
    }
    
    /// Update with observation
    /// 
    /// Bayes rule: P(H|D) ∝ P(D|H) × P(H)
    /// 
    /// # Arguments
    /// * `likelihoods` - P(observation | state_i) for each state
    #[inline]
    pub fn update(&mut self, likelihoods: [f64; N]) -> [f64; N] {
        // Multiply prior by likelihood
        for i in 0..N {
            self.posterior[i] *= likelihoods[i];
        }
        
        // Normalize to get posterior
        let sum: f64 = self.posterior.iter().sum();
        
        if sum > 1e-300 {
            for i in 0..N {
                self.posterior[i] /= sum;
            }
        }
        
        self.posterior
    }
    
    /// Get current posterior
    #[inline(always)]
    pub fn posterior(&self) -> [f64; N] {
        self.posterior
    }
    
    /// Get probability of specific state
    #[inline(always)]
    pub fn prob(&self, state: usize) -> f64 {
        if state < N {
            self.posterior[state]
        } else {
            0.0
        }
    }
    
    /// Get most likely state
    #[inline]
    pub fn most_likely(&self) -> usize {
        let mut max_prob = self.posterior[0];
        let mut max_state = 0;
        
        for i in 1..N {
            if self.posterior[i] > max_prob {
                max_prob = self.posterior[i];
                max_state = i;
            }
        }
        
        max_state
    }
    
    /// Shannon entropy (measure of uncertainty)
    /// 
    /// H = -Σ p(x) log p(x)
    /// 
    /// Returns normalized entropy in [0, 1]
    #[inline]
    pub fn entropy(&self) -> f64 {
        let mut h = 0.0;
        for &p in &self.posterior {
            if p > 1e-300 {
                h -= p * p.ln();
            }
        }
        
        // Normalize by maximum entropy (uniform distribution)
        h / (N as f64).ln()
    }
    
    /// Reset to uniform prior
    #[inline]
    pub fn reset(&mut self) {
        self.posterior = [1.0 / N as f64; N];
    }
    
    /// Reset to custom prior
    #[inline]
    pub fn reset_to(&mut self, prior: [f64; N]) {
        self.posterior = prior;
    }
}

impl<const N: usize> Default for BayesianFilter<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bayesian_filter() {
        // 3-state filter: bull, neutral, bear
        let mut filter = BayesianFilter::<3>::new();
        
        // Initially uniform
        assert_eq!(filter.posterior(), [1.0/3.0, 1.0/3.0, 1.0/3.0]);
        
        // Observe strong positive return
        // Likelihood: high for bull, low for bear
        let likelihoods = [0.8, 0.5, 0.1];
        filter.update(likelihoods);
        
        // Should now favor bull state
        assert!(filter.prob(0) > filter.prob(1));
        assert!(filter.prob(0) > filter.prob(2));
        assert_eq!(filter.most_likely(), 0);
    }
    
    #[test]
    fn test_entropy() {
        let mut filter = BayesianFilter::<3>::new();
        
        // Uniform: maximum entropy
        let h_uniform = filter.entropy();
        assert!((h_uniform - 1.0).abs() < 1e-10);
        
        // Strong belief in one state: low entropy
        filter.reset_to([0.9, 0.05, 0.05]);
        let h_certain = filter.entropy();
        assert!(h_certain < 0.5);
    }
}