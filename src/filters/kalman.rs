/// 1-dimensional Kalman filter (optimized for speed)
#[repr(align(64))]
#[derive(Debug, Clone, Copy)]
pub struct KalmanFilter1D {
    /// State estimate
    x: f64,
    
    /// Estimate uncertainty (variance)
    p: f64,
    
    /// Process noise variance
    q: f64,
    
    /// Measurement noise variance
    r: f64,
    
    /// State transition (default: 1.0 for random walk)
    f: f64,
}

impl KalmanFilter1D {
    /// Create new Kalman filter
    /// 
    /// # Arguments
    /// * `initial_state` - Initial state estimate
    /// * `initial_uncertainty` - Initial estimate uncertainty
    /// * `process_noise` - Process noise variance (how much state changes)
    /// * `measurement_noise` - Measurement noise variance (sensor accuracy)
    #[inline]
    pub fn new(
        initial_state: f64,
        initial_uncertainty: f64,
        process_noise: f64,
        measurement_noise: f64,
    ) -> Self {
        Self {
            x: initial_state,
            p: initial_uncertainty,
            q: process_noise,
            r: measurement_noise,
            f: 1.0,
        }
    }
    
    /// Create with state transition factor
    #[inline]
    pub fn with_transition(
        initial_state: f64,
        initial_uncertainty: f64,
        process_noise: f64,
        measurement_noise: f64,
        transition: f64,
    ) -> Self {
        Self {
            x: initial_state,
            p: initial_uncertainty,
            q: process_noise,
            r: measurement_noise,
            f: transition,
        }
    }
    
    /// Predict step (time update)
    #[inline(always)]
    pub fn predict(&mut self) {
        // State prediction: x_pred = f * x
        self.x *= self.f;
        
        // Uncertainty prediction: p_pred = f^2 * p + q
        self.p = self.f * self.f * self.p + self.q;
    }

    /// Update process noise
    #[inline(always)]
    pub fn set_process_noise(&mut self, q: f64) { 
        self.q = q; 
    }

    /// Update measurement noise
    #[inline(always)]
    pub fn set_measurement_noise(&mut self, r: f64) { 
        self.r = r; 
    }
    
    /// Update step with measurement (measurement update)
    #[inline(always)]
    pub fn update(&mut self, measurement: f64) -> f64 {
        // Innovation: y = z - x
        let innovation = measurement - self.x;
        
        // Innovation covariance: s = p + r
        let s = self.p + self.r;
        
        // Kalman gain: k = p / s
        let k = self.p / s;
        
        // Update state estimate: x = x + k * y
        self.x += k * innovation;
        
        // Update uncertainty: p = (1 - k) * p
        self.p *= 1.0 - k;
        
        self.x
    }
    
    /// Combined predict + update (most common usage)
    #[inline]
    pub fn filter(&mut self, measurement: f64) -> f64 {
        self.predict();
        self.update(measurement)
    }
    
    /// Get current state estimate
    #[inline(always)]
    pub fn state(&self) -> f64 {
        self.x
    }
    
    /// Get current uncertainty
    #[inline(always)]
    pub fn uncertainty(&self) -> f64 {
        self.p
    }
    
    /// Reset to initial conditions
    #[inline]
    pub fn reset(&mut self, state: f64, uncertainty: f64) {
        self.x = state;
        self.p = uncertainty;
    }
}

/// N-dimensional Kalman filter (for multivariate state)
#[repr(align(64))]
pub struct KalmanFilterND<const N: usize> {
    /// State vector
    x: [f64; N],
    
    /// Covariance matrix (flattened)
    p: [[f64; N]; N],
    
    /// State transition matrix
    f: [[f64; N]; N],
    
    /// Observation matrix
    h: [[f64; N]; N],
    
    /// Process noise covariance
    q: [[f64; N]; N],
    
    /// Measurement noise covariance
    r: [[f64; N]; N],
}

impl<const N: usize> KalmanFilterND<N> {
    /// Create new ND Kalman filter
    pub fn new(
        initial_state: [f64; N],
        initial_covariance: [[f64; N]; N],
        transition: [[f64; N]; N],
        observation: [[f64; N]; N],
        process_noise: [[f64; N]; N],
        measurement_noise: [[f64; N]; N],
    ) -> Self {
        Self {
            x: initial_state,
            p: initial_covariance,
            f: transition,
            h: observation,
            q: process_noise,
            r: measurement_noise,
        }
    }
    
    /// Predict step
    pub fn predict(&mut self) {
        use crate::matrix::ops::{matvec_multiply, matmul, matmul_transpose, mat_add};

        // x_pred = F * x
        let x_pred = matvec_multiply(&self.f, &self.x);
        self.x = x_pred;

        // P_pred = F * P * F^T + Q
        let fp = matmul(&self.f, &self.p);
        let fpft = matmul_transpose(&fp, &self.f);
        self.p = mat_add(&fpft, &self.q);
    }

    /// Update step
    pub fn update(&mut self, measurement: [f64; N]) {
        use crate::matrix::ops::{matvec_multiply, matmul, matmul_transpose, mat_add, mat_sub, transpose, identity};
        use crate::matrix::kernels::invert_matrix;

        // Innovation: y = z - H * x
        let hx = matvec_multiply(&self.h, &self.x);
        let innovation = vec_sub(&measurement, &hx);

        // Innovation covariance: S = H * P * H^T + R
        let hp = matmul(&self.h, &self.p);
        let hpht = matmul_transpose(&hp, &self.h);
        let s = mat_add(&hpht, &self.r);

        // Kalman gain: K = P * H^T * S^-1
        let s_inv = invert_matrix(&s).expect("Innovation covariance not invertible");
        let ht = transpose(&self.h);
        let pht = matmul(&self.p, &ht);
        let k = matmul(&pht, &s_inv);

        // Update state: x = x + K * y
        let ky = matvec_multiply(&k, &innovation);
        self.x = vec_add(&self.x, &ky);

        // Update covariance: P = (I - K * H) * P
        let kh = matmul(&k, &self.h);
        let i_kh = mat_sub(&identity(), &kh);
        self.p = matmul(&i_kh, &self.p);
    }
    
    /// Get state
    #[inline]
    pub fn state(&self) -> [f64; N] {
        self.x
    }
}

// Simple vector operations (kept local as they're not matrix operations)
#[inline(always)]
fn vec_add<const N: usize>(a: &[f64; N], b: &[f64; N]) -> [f64; N] {
    let mut result = [0.0; N];
    for i in 0..N {
        result[i] = a[i] + b[i];
    }
    result
}

#[inline(always)]
fn vec_sub<const N: usize>(a: &[f64; N], b: &[f64; N]) -> [f64; N] {
    let mut result = [0.0; N];
    for i in 0..N {
        result[i] = a[i] - b[i];
    }
    result
}

// Note: All matrix operations now use centralized functions from:
// - matrix::ops: matvec_multiply, matmul, matmul_transpose, transpose, mat_add, mat_sub, identity
// - matrix::kernels: invert_matrix

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_kalman_1d() {
        let mut kf = KalmanFilter1D::new(0.0, 1.0, 0.01, 0.1);
        
        // Noisy measurements around 5.0
        let measurements = [5.1, 4.9, 5.2, 4.8, 5.0];
        
        let mut estimates = Vec::new();
        for &z in &measurements {
            let estimate = kf.filter(z);
            estimates.push(estimate);
        }
        
        // Should converge near 5.0
        assert!((estimates.last().unwrap() - 5.0).abs() < 0.3);
    }
}
