use crate::probability::utils::gaussian_pdf;

/// Two-state Markov regime switching model
/// 
/// Models transitions between "normal" and "volatile" market regimes.
/// Uses forward algorithm for online Bayesian filtering.
#[repr(align(64))]
pub struct MarkovSwitching {
    // State probabilities: [normal, volatile]
    state_probs: [f64; 2],
    
    // Transition matrix: P[i][j] = P(state_t = j | state_{t-1} = i)
    transition: [[f64; 2]; 2],
    
    // Emission parameters for each state: (mean, std_dev)
    emission_params: [(f64, f64); 2],
    
    // Log-likelihood accumulator (for diagnostics)
    log_likelihood: f64,
}

impl MarkovSwitching {
    /// Create with custom parameters
    #[inline]
    pub fn new(
        normal_std: f64,
        volatile_std: f64,
        prob_stay_normal: f64,
        prob_stay_volatile: f64,
    ) -> Self {
        assert!(normal_std > 0.0, "Standard deviation must be positive");
        assert!(volatile_std > 0.0, "Standard deviation must be positive");
        assert!((0.0..=1.0).contains(&prob_stay_normal), "Probability must be in [0,1]");
        assert!((0.0..=1.0).contains(&prob_stay_volatile), "Probability must be in [0,1]");
        
        Self {
            state_probs: [1.0, 0.0], // Start in normal state
            transition: [
                [prob_stay_normal, 1.0 - prob_stay_normal],
                [1.0 - prob_stay_volatile, prob_stay_volatile],
            ],
            emission_params: [
                (0.0, normal_std),
                (0.0, volatile_std),
            ],
            log_likelihood: 0.0,
        }
    }
    
    /// Create with SPY-calibrated defaults
    /// 
    /// Normal: ~0.1% per second volatility
    /// Volatile: ~0.5% per second volatility
    /// Stay normal: 99% (rare transitions)
    /// Stay volatile: 90% (mean revert faster)
    #[inline]
    pub fn spy_default() -> Self {
        Self::new(
            0.001,  // normal_std
            0.005,  // volatile_std
            0.99,   // stay_normal
            0.90,   // stay_volatile
        )
    }
    
    /// Update with new observation (forward algorithm)
    #[inline]
    pub fn update(&mut self, observation: f64) {
        let mut new_probs = [0.0; 2];
        
        // For each state j
        for j in 0..2 {
            // P(obs | state_j)
            let emission = gaussian_pdf(
                observation,
                self.emission_params[j].0,
                self.emission_params[j].1,
            );
            
            // Sum over all previous states i
            new_probs[j] = emission * (
                self.transition[0][j] * self.state_probs[0] +
                self.transition[1][j] * self.state_probs[1]
            );
        }
        
        // Normalize to get posterior probabilities
        let sum = new_probs[0] + new_probs[1];
        
        if sum > 1e-300 {
            self.state_probs[0] = new_probs[0] / sum;
            self.state_probs[1] = new_probs[1] / sum;
            
            // Update log-likelihood
            self.log_likelihood += sum.ln();
        }
        // If sum is too small, keep previous probabilities
    }
    
    /// Get probability of normal regime
    #[inline(always)]
    pub fn normal_prob(&self) -> f64 {
        self.state_probs[0]
    }
    
    /// Get probability of volatile regime
    #[inline(always)]
    pub fn volatile_prob(&self) -> f64 {
        self.state_probs[1]
    }
    
    #[inline(always)]
    pub fn state_probabilities(&self) -> [f64; 2] {
        self.state_probs
    }

    /// Check if currently in volatile regime (threshold-based)
    #[inline(always)]
    pub fn is_volatile(&self, threshold: f64) -> bool {
        self.state_probs[1] > threshold
    }
    
    /// Get most likely current state
    #[inline(always)]
    pub fn most_likely_state(&self) -> RegimeState {
        if self.state_probs[0] > self.state_probs[1] {
            RegimeState::Normal
        } else {
            RegimeState::Volatile
        }
    }
    
    /// Get accumulated log-likelihood
    #[inline(always)]
    pub fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }
    
    /// Reset to initial state
    #[inline]
    pub fn reset(&mut self) {
        self.state_probs = [1.0, 0.0];
        self.log_likelihood = 0.0;
    }
    
    /// Get emission standard deviation for a state
    #[inline]
    pub fn emission_std(&self, state: RegimeState) -> f64 {
        match state {
            RegimeState::Normal => self.emission_params[0].1,
            RegimeState::Volatile => self.emission_params[1].1,
        }
    }
}

/// Regime state enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegimeState {
    Normal,
    Volatile,
}

// gaussian_pdf is imported at the top

/// Hidden Markov Model (generalized to N states)
/// 
/// For more complex regime modeling (e.g., bull/neutral/bear)
#[repr(align(64))]
pub struct HiddenMarkov<const N: usize> {
    state_probs: [f64; N],
    transition: [[f64; N]; N],
    emission_params: [(f64, f64); N],
    log_likelihood: f64,
}

impl<const N: usize> HiddenMarkov<N> {
    /// Create new HMM with given parameters
    pub fn new(
        initial_probs: [f64; N],
        transition: [[f64; N]; N],
        emission_params: [(f64, f64); N],
    ) -> Self {
        // Validate initial probabilities sum to 1
        let sum: f64 = initial_probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Initial probabilities must sum to 1");
        
        // Validate transition matrix rows sum to 1
        for i in 0..N {
            let row_sum: f64 = transition[i].iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-6, "Transition matrix rows must sum to 1");
        }
        
        Self {
            state_probs: initial_probs,
            transition,
            emission_params,
            log_likelihood: 0.0,
        }
    }
    
    /// Update with observation
    #[inline]
    pub fn update(&mut self, observation: f64) {
        let mut new_probs = [0.0; N];
        
        for j in 0..N {
            let emission = gaussian_pdf(
                observation,
                self.emission_params[j].0,
                self.emission_params[j].1,
            );
            
            let mut transition_sum = 0.0;
            for i in 0..N {
                transition_sum += self.transition[i][j] * self.state_probs[i];
            }
            
            new_probs[j] = emission * transition_sum;
        }
        
        // Normalize
        let sum: f64 = new_probs.iter().sum();
        
        if sum > 1e-300 {
            for i in 0..N {
                self.state_probs[i] = new_probs[i] / sum;
            }
            self.log_likelihood += sum.ln();
        }
    }
    
    /// Get probability of state i
    #[inline(always)]
    pub fn state_prob(&self, state: usize) -> f64 {
        if state < N {
            self.state_probs[state]
        } else {
            0.0
        }
    }
    
    /// Get most likely state
    #[inline]
    pub fn most_likely_state(&self) -> usize {
        let mut max_prob = self.state_probs[0];
        let mut max_state = 0;
        
        for i in 1..N {
            if self.state_probs[i] > max_prob {
                max_prob = self.state_probs[i];
                max_state = i;
            }
        }
        
        max_state
    }
    
    /// Reset
    #[inline]
    pub fn reset(&mut self, initial_probs: [f64; N]) {
        self.state_probs = initial_probs;
        self.log_likelihood = 0.0;
    }
}

/// 3-state HMM for bull/neutral/bear markets
pub type BullBearHMM = HiddenMarkov<3>;

impl BullBearHMM {
    /// Create with typical bull/neutral/bear parameters
    pub fn bull_bear_default() -> Self {
        // Initial: assume neutral
        let initial = [0.0, 1.0, 0.0];
        
        // Transition matrix (favor staying in current state)
        let transition = [
            [0.95, 0.04, 0.01],  // From bull
            [0.02, 0.96, 0.02],  // From neutral
            [0.01, 0.04, 0.95],  // From bear
        ];
        
        // Emission parameters: (mean_return, std_dev)
        let emissions = [
            (0.0003, 0.001),   // Bull: +0.03% per second, low vol
            (0.0, 0.0015),     // Neutral: 0% drift, medium vol
            (-0.0003, 0.002),  // Bear: -0.03% per second, high vol
        ];
        
        Self::new(initial, transition, emissions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_markov_switching() {
        let mut regime = MarkovSwitching::spy_default();
        
        // Feed normal returns
        for _ in 0..100 {
            regime.update(0.0001);
        }
        
        assert!(regime.normal_prob() > 0.9, "Should be in normal regime");
        assert_eq!(regime.most_likely_state(), RegimeState::Normal);
        
        // Feed volatile returns
        for _ in 0..20 {
            regime.update(0.01); // 1% moves
        }
        
        assert!(regime.volatile_prob() > 0.8, "Should transition to volatile");
        assert_eq!(regime.most_likely_state(), RegimeState::Volatile);
    }
    
    #[test]
    fn test_hmm_3state() {
        let mut hmm = BullBearHMM::bull_bear_default();
        
        // Should start neutral
        assert_eq!(hmm.most_likely_state(), 1);
        
        // Feed bullish returns
        for _ in 0..50 {
            hmm.update(0.0005); // Positive returns
        }
        
        // Should move to bull state
        assert_eq!(hmm.most_likely_state(), 0);
    }
    
    #[test]
    fn test_reset() {
        let mut regime = MarkovSwitching::spy_default();
        
        regime.update(0.01);
        regime.update(0.01);
        
        let ll_before = regime.log_likelihood();
        
        regime.reset();
        
        assert_eq!(regime.normal_prob(), 1.0);
        assert_eq!(regime.log_likelihood(), 0.0);
        assert_ne!(ll_before, 0.0);
    }
}