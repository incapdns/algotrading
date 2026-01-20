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

// ============================================================
// KALMAN UPDATE RESULT
// ============================================================

#[derive(Debug, Clone, Copy)]
pub struct KalmanUpdate {
    pub state: f64,           // x̂ (estimativa da média)
    pub innovation: f64,      // y = z - x̂ 
    pub innovation_var: f64,  // S = P + R
    pub innovation_std: f64,  // sqrt(S)
    pub kalman_gain: f64,     // K
    pub state_var: f64,       // P (pós-update)
    pub state_std: f64,       // sqrt(P)
}

impl KalmanFilter1D {
    #[inline(always)]
    pub fn update_full(&mut self, measurement: f64) -> KalmanUpdate {
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
        
        KalmanUpdate {
            state: self.x,
            innovation,
            innovation_var: s,
            innovation_std: s.sqrt(),
            kalman_gain: k,
            state_var: self.p,
            state_std: self.p.sqrt(),
        }
    }
    
    /// Backward compatible
    #[inline(always)]
    #[deprecated(note = "Use update_full()")]
    pub fn update(&mut self, measurement: f64) -> f64 {
        self.update_full(measurement).state
    }
}

// ============================================
// ADICIONAR APÓS KalmanUpdate existente
// ============================================

/// Result of N-dimensional Kalman update step
/// Contains all values needed for TickEngine V6 signal generation
#[derive(Debug, Clone)]
pub struct KalmanUpdateND<const N: usize> {
    /// Updated state vector [β, α] for N=2
    pub state: [f64; N],
    
    /// Innovation (measurement residual): y = z - H*x_pred
    pub innovation: f64,
    
    /// Innovation variance: S = H*P*Hᵀ + R
    pub innovation_var: f64,
    
    /// Innovation standard deviation: sqrt(S)
    pub innovation_std: f64,
    
    /// Kalman gain vector
    pub kalman_gain: [f64; N],
    
    /// Updated covariance matrix
    pub covariance: [[f64; N]; N],
}

impl<const N: usize> KalmanUpdateND<N> {
    /// Compute Z-score: innovation / innovation_std
    /// This is the primary signal for TickEngine V6
    #[inline]
    pub fn z_score(&self) -> f64 {
        if self.innovation_std > 1e-10 {
            self.innovation / self.innovation_std
        } else {
            0.0
        }
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

// ============================================
// Kalman Filter 2D otimizado para TickEngine V6
// State: [β (hedge_ratio), α (intercept)]
// ============================================

/// 2D Kalman Filter for spread statistics (TickEngine V6)
/// 
/// Estimates:
/// - β (hedge ratio): optimal relationship between spot and futures
/// - α (intercept): spread mean
/// 
/// Features:
/// - Joseph form covariance update (numerical stability)
/// - Dynamic observation matrix H
/// - Adaptive Q/R support
#[repr(align(64))]
#[derive(Debug, Clone)]
pub struct KalmanFilter2D {
    /// State vector: [β, α]
    x: [f64; 2],
    
    /// Covariance matrix 2x2
    p: [[f64; 2]; 2],
    
    /// State transition matrix (usually identity for random walk)
    f: [[f64; 2]; 2],
    
    /// Process noise covariance (adaptive)
    q: [[f64; 2]; 2],
    
    /// Measurement noise variance (adaptive, scalar for 1D measurement)
    r: f64,
}

impl KalmanFilter2D {
    /// Create new 2D Kalman filter for spread estimation
    ///
    /// # Arguments
    /// * `initial_beta` - Initial hedge ratio estimate (typically 1.0)
    /// * `initial_alpha` - Initial intercept estimate (typically 0.0)
    /// * `p_beta` - Initial uncertainty for β (typically 1.0) [1]
    /// * `p_alpha` - Initial uncertainty for α (typically 0.01) [1]
    /// * `q_beta` - Process noise for β (typically 1e-6) [1]
    /// * `q_alpha` - Process noise for α (typically 1e-8) [1]
    /// * `r` - Measurement noise variance (typically 1e-4) [1]
    pub fn new(
        initial_beta: f64,
        initial_alpha: f64,
        p_beta: f64,    // ← Separado
        p_alpha: f64,   // ← Separado
        q_beta: f64,
        q_alpha: f64,
        r: f64,
    ) -> Self {
        Self {
            x: [initial_beta, initial_alpha],
            p: [
                [p_beta, 0.0],     // ← P_beta = 1.0
                [0.0, p_alpha],    // ← P_alpha = 0.01
            ],
            f: [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            q: [
                [q_beta, 0.0],
                [0.0, q_alpha],
            ],
            r,
        }
    }

    /// Create with TickEngine V6 default parameters
    /// From documentation: Q diagonal, R scalar
    pub fn default_tick_engine() -> Self {
        Self::new(
            1.0,    // initial_beta = 1.0 (hedge ratio starts at 1:1)
            0.0,    // initial_alpha = 0.0 (no intercept initially)
            1.0,    // p_beta = 1.0 (alta incerteza inicial para β)
            0.01,   // p_alpha = 0.01 (intercepto mais estável)
            1e-6,   // q_beta = 1e-6 (β muda lentamente)
            1e-8,   // q_alpha = 1e-8 (α ainda mais lento)
            1e-4,   // r = 1e-4 (~1 bps² de measurement noise)
        )
    }

    /// Predict step
    /// 
    /// x̂[t|t-1] = F × x̂[t-1|t-1]
    /// P[t|t-1] = F × P[t-1|t-1] × Fᵀ + Q
    #[inline]
    pub fn predict(&mut self) {
        // For identity F, x stays the same
        // x_pred = F * x (identity, so no change)
        
        // P_pred = F * P * Fᵀ + Q
        // For identity F: P_pred = P + Q
        self.p[0][0] += self.q[0][0];
        self.p[0][1] += self.q[0][1];
        self.p[1][0] += self.q[1][0];
        self.p[1][1] += self.q[1][1];
    }

    /// Update step with dynamic H matrix
    /// 
    /// For TickEngine V6:
    /// - measurement = vamp_spot
    /// - h = [vamp_futures, 1.0]
    /// 
    /// Uses Joseph form for numerical stability:
    /// P = (I - K*H) * P * (I - K*H)ᵀ + K * R * Kᵀ
    #[inline]
    pub fn update(&mut self, measurement: f64, h: [f64; 2]) -> KalmanUpdateND<2> {
        // ===== Innovation =====
        // y = z - H * x_pred
        let h_x = h[0] * self.x[0] + h[1] * self.x[1];
        let innovation = measurement - h_x;

        // ===== Innovation covariance =====
        // S = H * P * Hᵀ + R
        // For H = [h0, h1], S is scalar
        let ph0 = self.p[0][0] * h[0] + self.p[0][1] * h[1];
        let ph1 = self.p[1][0] * h[0] + self.p[1][1] * h[1];
        let s = h[0] * ph0 + h[1] * ph1 + self.r;

        // ===== Kalman Gain =====
        // K = P * Hᵀ * S⁻¹
        let s_inv = 1.0 / s.max(1e-10);
        let k = [ph0 * s_inv, ph1 * s_inv];

        // ===== State update =====
        // x = x + K * y
        self.x[0] += k[0] * innovation;
        self.x[1] += k[1] * innovation;

        // ===== Covariance update (Joseph form) =====
        // I_KH = I - K * H
        // P = I_KH * P * I_KHᵀ + K * R * Kᵀ
        let i_kh = [
            [1.0 - k[0] * h[0], -k[0] * h[1]],
            [-k[1] * h[0], 1.0 - k[1] * h[1]],
        ];

        // temp = I_KH * P
        let temp = [
            [
                i_kh[0][0] * self.p[0][0] + i_kh[0][1] * self.p[1][0],
                i_kh[0][0] * self.p[0][1] + i_kh[0][1] * self.p[1][1],
            ],
            [
                i_kh[1][0] * self.p[0][0] + i_kh[1][1] * self.p[1][0],
                i_kh[1][0] * self.p[0][1] + i_kh[1][1] * self.p[1][1],
            ],
        ];

        // P_joseph = temp * I_KHᵀ
        let p_joseph = [
            [
                temp[0][0] * i_kh[0][0] + temp[0][1] * i_kh[0][1],
                temp[0][0] * i_kh[1][0] + temp[0][1] * i_kh[1][1],
            ],
            [
                temp[1][0] * i_kh[0][0] + temp[1][1] * i_kh[0][1],
                temp[1][0] * i_kh[1][0] + temp[1][1] * i_kh[1][1],
            ],
        ];

        // K * R * Kᵀ (R is scalar)
        let kr_kt = [
            [k[0] * self.r * k[0], k[0] * self.r * k[1]],
            [k[1] * self.r * k[0], k[1] * self.r * k[1]],
        ];

        // P = P_joseph + K * R * Kᵀ
        self.p = [
            [p_joseph[0][0] + kr_kt[0][0], p_joseph[0][1] + kr_kt[0][1]],
            [p_joseph[1][0] + kr_kt[1][0], p_joseph[1][1] + kr_kt[1][1]],
        ];

        let innovation_std = s.sqrt();

        KalmanUpdateND {
            state: self.x,
            innovation,
            innovation_var: s,
            innovation_std,
            kalman_gain: k,
            covariance: self.p,
        }
    }

    /// Combined predict + update (convenience method)
    #[inline]
    pub fn step(&mut self, measurement: f64, h: [f64; 2]) -> KalmanUpdateND<2> {
        self.predict();
        self.update(measurement, h)
    }

    // ===== Adaptive Q/R methods =====

    /// Update process noise covariance
    #[inline]
    pub fn set_q(&mut self, q_beta: f64, q_alpha: f64) {
        self.q[0][0] = q_beta;
        self.q[1][1] = q_alpha;
    }

    /// Update measurement noise variance
    #[inline]
    pub fn set_r(&mut self, r: f64) {
        self.r = r;
    }

    /// Adapt Q based on innovation (TickEngine V6 adaptive)
    /// Increases Q when innovations are large
    #[inline]
    pub fn adapt_q(&mut self, innovation_var: f64, base_q: f64, sensitivity: f64) {
        let adaptive_q = base_q * (1.0 + sensitivity * innovation_var);
        self.q[0][0] = adaptive_q;
        self.q[1][1] = adaptive_q;
    }

    /// Adapt R based on recent measurement variance
    #[inline]
    pub fn adapt_r(&mut self, measurement_var: f64, min_r: f64) {
        self.r = measurement_var.max(min_r);
    }

    // ===== Accessors =====

    /// Get current hedge ratio (β)
    #[inline]
    pub fn beta(&self) -> f64 {
        self.x[0]
    }

    /// Get current intercept (α)
    #[inline]
    pub fn alpha(&self) -> f64 {
        self.x[1]
    }

    /// Get current state [β, α]
    #[inline]
    pub fn state(&self) -> [f64; 2] {
        self.x
    }

    /// Get current covariance matrix
    #[inline]
    pub fn covariance(&self) -> [[f64; 2]; 2] {
        self.p
    }

    /// Get β uncertainty (std dev)
    #[inline]
    pub fn beta_std(&self) -> f64 {
        self.p[0][0].sqrt()
    }

    /// Get α uncertainty (std dev)
    #[inline]
    pub fn alpha_std(&self) -> f64 {
        self.p[1][1].sqrt()
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
