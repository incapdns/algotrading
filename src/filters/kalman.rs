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

/// 2D Kalman Filter for spread statistics (TickEngine V6)
/// 
/// Estimates:
/// - β (hedge ratio): optimal relationship between spot and futures
/// - α (intercept): spread mean
/// 
/// Features:
/// - Joseph form covariance update (numerical stability)
/// - Dynamic observation matrix H
/// - Adaptive Q/R via Robbins-Monro conforme §3.6 [10]
#[repr(align(64))]
#[derive(Debug, Clone)]
pub struct KalmanFilter2D {
    /// State vector: [β, α]
    x: [f64; 2],
    
    /// Covariance matrix 2x2
    p: [[f64; 2]; 2],
    
    /// Covariance matrix predita P[t|t-1] - guardada APÓS predict() para uso em adapt_r_robbins_monro()
    /// Conforme §3.6.1 e §15.1.4 [10]
    p_pred: [[f64; 2]; 2],
    
    /// State transition matrix (usually identity for random walk)
    f: [[f64; 2]; 2],
    
    /// Process noise covariance (adaptive via Robbins-Monro)
    q: [[f64; 2]; 2],
    
    /// Measurement noise variance (adaptive via Robbins-Monro, scalar for 1D measurement)
    r: f64,

    /// Alpha para adaptação de Q conforme §3.6 [10] (default: 0.01)
    alpha_q: f64,

    /// Alpha para adaptação de R conforme §3.6 [10] (default: 0.05)
    alpha_r: f64,

    /// Floor mínimo para Q conforme §15.1 [10] (default: 1e-8)
    q_min: f64,

    /// Floor mínimo para R conforme §15.1 [10] (default: 1e-6)
    r_min: f64,

    /// Último H usado (para cálculo de R adaptativo)
    last_h: [f64; 2],

    /// Última innovation (para cálculo de Q adaptativo)
    last_innovation: f64,

    /// Último Kalman gain (para cálculo de Q adaptativo)
    last_k: [f64; 2],
}

/// Informações sobre saúde da matriz de covariância
#[derive(Debug, Clone, Copy)]
pub struct CovarianceHealth {
    pub determinant: f64,
    pub trace: f64,
    pub asymmetry: f64,
    pub is_positive_definite: bool,
    pub is_symmetric: bool,
}

impl KalmanFilter2D {
    /// Create new 2D Kalman filter for spread estimation
    ///
    /// # Arguments
    /// * `initial_beta` - Initial hedge ratio estimate (typically 1.0)
    /// * `initial_alpha` - Initial intercept estimate (typically 0.0)
    /// * `p_beta` - Initial uncertainty for β (typically 1.0) [10] §15.1
    /// * `p_alpha` - Initial uncertainty for α (typically 0.01) [10] §15.1
    /// * `q_beta` - Process noise for β (typically 1e-6) [10] §15.1
    /// * `q_alpha` - Process noise for α (typically 1e-8) [10] §15.1
    /// * `r` - Measurement noise variance (typically 1e-4) [10] §15.1
    pub fn new(
        initial_beta: f64,
        initial_alpha: f64,
        p_beta: f64,
        p_alpha: f64,
        q_beta: f64,
        q_alpha: f64,
        r: f64,
    ) -> Self {
        Self {
            x: [initial_beta, initial_alpha],
            p: [
                [p_beta, 0.0],
                [0.0, p_alpha],
            ],
            // CORREÇÃO: Inicializar p_pred com os mesmos valores de p
            p_pred: [
                [p_beta, 0.0],
                [0.0, p_alpha],
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
            // Parâmetros adaptativos conforme §3.6 e §15.1 [10]
            alpha_q: 0.01,
            alpha_r: 0.05,
            q_min: 1e-8,
            r_min: 1e-6,
            last_h: [1.0, 1.0],
            last_innovation: 0.0,
            last_k: [0.0, 0.0],
        }
    }

    /// Create with TickEngine V6 default parameters
    /// From documentation §15.1 [10]: Q diagonal, R scalar
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

    #[inline]
    pub fn symmetrize_covariance(&mut self) -> bool {
        // Para matriz 2x2, apenas P[0][1] e P[1][0] precisam ser simétricos
        let asymmetry = (self.p[0][1] - self.p[1][0]).abs();
        
        if asymmetry > 1e-15 {
            // P = (P + Pᵀ) / 2
            let avg = (self.p[0][1] + self.p[1][0]) / 2.0;
            self.p[0][1] = avg;
            self.p[1][0] = avg;
            
            // Também aplicar em p_pred para consistência
            let avg_pred = (self.p_pred[0][1] + self.p_pred[1][0]) / 2.0;
            self.p_pred[0][1] = avg_pred;
            self.p_pred[1][0] = avg_pred;
            
            return true;
        }
        false
    }

    #[inline]
    pub fn check_covariance_health(&self) -> (bool, CovarianceHealth) {
        let trace = self.p[0][0] + self.p[1][1];
        let det = self.p[0][0] * self.p[1][1] - self.p[0][1] * self.p[1][0];
        let asymmetry = (self.p[0][1] - self.p[1][0]).abs();
        
        // P deve ser positiva definida: det > 0 e trace > 0
        let is_positive_definite = det > 1e-12 && trace > 0.0;
        let is_symmetric = asymmetry < 1e-10;
        
        let health = CovarianceHealth {
            determinant: det,
            trace,
            asymmetry,
            is_positive_definite,
            is_symmetric,
        };
        
        (is_positive_definite && is_symmetric, health)
    }

    /// Create with custom adaptive parameters
    /// Conforme §3.6 [10]
    pub fn with_adaptive_params(
        initial_beta: f64,
        initial_alpha: f64,
        p_beta: f64,
        p_alpha: f64,
        q_beta: f64,
        q_alpha: f64,
        r: f64,
        alpha_q: f64,
        alpha_r: f64,
        q_min: f64,
        r_min: f64,
    ) -> Self {
        let mut filter = Self::new(initial_beta, initial_alpha, p_beta, p_alpha, q_beta, q_alpha, r);
        filter.alpha_q = alpha_q;
        filter.alpha_r = alpha_r;
        filter.q_min = q_min;
        filter.r_min = r_min;
        filter
    }

    /// Predict step
    /// 
    /// x̂[t|t-1] = F × x̂[t-1|t-1]
    /// P[t|t-1] = F × P[t-1|t-1] × Fᵀ + Q
    /// 
    /// Conforme §3.6.1 [10]: guarda P_pred APÓS adicionar Q para uso em adapt_r_robbins_monro()
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
        
        // Guarda P[t|t-1] APÓS adicionar Q para adapt_r_robbins_monro()
        // Conforme §3.6.1 [10]
        self.p_pred = self.p;
    }

    /// Update step with dynamic H matrix
    /// 
    /// For TickEngine V6 conforme §3.4 [10]:
    /// - measurement = vamp_spot
    /// - h = [vamp_futures, 1.0]
    /// 
    /// Uses Joseph form for numerical stability conforme §3.4 [10]:
    /// P = (I - K*H) * P * (I - K*H)ᵀ + K * R * Kᵀ
    #[inline]
    pub fn update(&mut self, measurement: f64, h: [f64; 2]) -> KalmanUpdateND<2> {
        // Guardar H para adaptação de R
        self.last_h = h;

        // ===== Innovation =====
        // y = z - H * x_pred conforme §3.4 [10]
        let h_x = h[0] * self.x[0] + h[1] * self.x[1];
        let innovation = measurement - h_x;
        self.last_innovation = innovation;

        // ===== Innovation covariance =====
        // S = H * P * Hᵀ + R conforme §3.4 [10]
        let ph0 = self.p[0][0] * h[0] + self.p[0][1] * h[1];
        let ph1 = self.p[1][0] * h[0] + self.p[1][1] * h[1];
        let s = h[0] * ph0 + h[1] * ph1 + self.r;

        // ===== Kalman Gain =====
        // K = P * Hᵀ * S⁻¹ conforme §3.4 [10]
        let s_inv = 1.0 / s.max(1e-10);
        let k = [ph0 * s_inv, ph1 * s_inv];
        self.last_k = k;

        // ===== State update =====
        // x = x + K * y conforme §3.4 [10]
        self.x[0] += k[0] * innovation;
        self.x[1] += k[1] * innovation;

        // ===== Covariance update (Joseph form) =====
        // Conforme §3.4 [10]: P = (I - K*H) * P * (I - K*H)ᵀ + K * R * Kᵀ
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

    /// Combined predict + update with adaptive Q/R (Robbins-Monro)
    /// Conforme §3.6 [10]
    #[inline]
    pub fn step_adaptive(&mut self, measurement: f64, h: [f64; 2]) -> KalmanUpdateND<2> {
        self.predict();
        let result = self.update(measurement, h);
        
        // Adaptar Q e R após o update
        self.adapt_r_robbins_monro();
        self.adapt_q_robbins_monro();
        
        result
    }

    // ===== Adaptive Q/R methods (Robbins-Monro) conforme §3.6 [10] =====

    /// Adapt R (Measurement Noise) via Robbins-Monro
    /// Conforme §3.6.1 [10]:
    /// 
    /// ```text
    /// R[t] = (1 - α_r) × R[t-1] + α_r × (y² - H × P[t|t-1] × Hᵀ)
    /// R[t] = max(R[t], R_min)
    /// ```
    /// 
    /// Nota: Usa p_pred (P[t|t-1]) que foi guardado no predict(), não P pós-update.
    #[inline]
    pub fn adapt_r_robbins_monro(&mut self) {
        let h = self.last_h;
        let innovation = self.last_innovation;
        
        // innovation_sq = y²
        let innovation_sq = innovation * innovation;
        
        // H × P_pred × Hᵀ (usando P[t|t-1] correto, conforme §3.6.1 [10])
        let ph0 = self.p_pred[0][0] * h[0] + self.p_pred[0][1] * h[1];
        let ph1 = self.p_pred[1][0] * h[0] + self.p_pred[1][1] * h[1];
        let h_p_ht = h[0] * ph0 + h[1] * ph1;
        
        // R[t] = (1 - α_r) × R[t-1] + α_r × (y² - H × P_pred × Hᵀ)
        let r_update = innovation_sq - h_p_ht;
        self.r = (1.0 - self.alpha_r) * self.r + self.alpha_r * r_update;
        
        // Floor para estabilidade conforme §3.6 [10]
        self.r = self.r.max(self.r_min);
    }

    /// Adapt Q (Process Noise) via Robbins-Monro
    /// Conforme §3.6 [10]:
    /// 
    /// ```text
    /// state_correction = K[t] × y[t]
    /// Q[t] = (1 - α_q) × Q[t-1] + α_q × (K[t] × y[t] × y[t]ᵀ × K[t]ᵀ)
    /// Q[t] = max(Q[t], Q_min)
    /// ```
    #[inline]
    pub fn adapt_q_robbins_monro(&mut self) {
        let k = self.last_k;
        let y = self.last_innovation;
        
        // K × y × yᵀ × Kᵀ = (K × y) × (K × y)ᵀ
        // Para state 2D: outer product de (K × y)
        let ky = [k[0] * y, k[1] * y];
        
        // Q_update = ky × kyᵀ (outer product)
        let q_update = [
            [ky[0] * ky[0], ky[0] * ky[1]],
            [ky[1] * ky[0], ky[1] * ky[1]],
        ];
        
        // Q[t] = (1 - α_q) × Q[t-1] + α_q × Q_update
        self.q[0][0] = (1.0 - self.alpha_q) * self.q[0][0] + self.alpha_q * q_update[0][0];
        self.q[0][1] = (1.0 - self.alpha_q) * self.q[0][1] + self.alpha_q * q_update[0][1];
        self.q[1][0] = (1.0 - self.alpha_q) * self.q[1][0] + self.alpha_q * q_update[1][0];
        self.q[1][1] = (1.0 - self.alpha_q) * self.q[1][1] + self.alpha_q * q_update[1][1];
        
        // Floor para estabilidade conforme §3.6 [10]
        self.q[0][0] = self.q[0][0].max(self.q_min);
        self.q[1][1] = self.q[1][1].max(self.q_min);
    }

    /// Update process noise covariance manually
    #[inline]
    pub fn set_q(&mut self, q_beta: f64, q_alpha: f64) {
        self.q[0][0] = q_beta.max(self.q_min);
        self.q[1][1] = q_alpha.max(self.q_min);
    }

    /// Update measurement noise variance manually
    #[inline]
    pub fn set_r(&mut self, r: f64) {
        self.r = r.max(self.r_min);
    }

    /// Set adaptive parameters
    /// Conforme §15.1 [10]: alpha_q=0.01, alpha_r=0.05
    #[inline]
    pub fn set_adaptive_params(&mut self, alpha_q: f64, alpha_r: f64, q_min: f64, r_min: f64) {
        self.alpha_q = alpha_q;
        self.alpha_r = alpha_r;
        self.q_min = q_min;
        self.r_min = r_min;
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

    /// Get predicted covariance matrix P[t|t-1]
    #[inline]
    pub fn covariance_pred(&self) -> [[f64; 2]; 2] {
        self.p_pred
    }

    /// Get current Q matrix
    #[inline]
    pub fn q(&self) -> [[f64; 2]; 2] {
        self.q
    }

    /// Get current R value
    #[inline]
    pub fn r(&self) -> f64 {
        self.r
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

    /// Reset filter to initial state
    pub fn reset(&mut self) {
        self.x = [1.0, 0.0];
        self.p = [[1.0, 0.0], [0.0, 0.01]];
        self.p_pred = [[1.0, 0.0], [0.0, 0.01]];
        self.q = [[1e-6, 0.0], [0.0, 1e-8]];
        self.r = 1e-4;
        self.last_h = [1.0, 1.0];
        self.last_innovation = 0.0;
        self.last_k = [0.0, 0.0];
    }
}

// ============================================================
// POSITION-VELOCITY KALMAN FILTER (Constant Velocity Model)
// ============================================================

/// Configuração do Position-Velocity Kalman
/// Análogo ao KalmanFilter2D mas para tracking de velocity
#[derive(Debug, Clone, Copy)]
pub struct PVKalmanConfig {
    /// Incerteza inicial da position
    pub initial_position_var: f64,
    /// Incerteza inicial da velocity
    pub initial_velocity_var: f64,
    /// Process noise da position (default: 1e-5)
    pub q_position: f64,
    /// Process noise da velocity (default: 1e-6)
    pub q_velocity: f64,
    /// Measurement noise (default: 1e-4)
    pub r: f64,
    /// Alpha para adaptação de Q (default: 0.01)
    pub alpha_q: f64,
    /// Alpha para adaptação de R (default: 0.05)
    pub alpha_r: f64,
    /// Floor mínimo para Q
    pub q_min: f64,
    /// Floor mínimo para R
    pub r_min: f64,
    /// Mínimo de ticks para warmup
    pub min_ticks: u64,
}

impl Default for PVKalmanConfig {
    fn default() -> Self {
        Self {
            initial_position_var: 1.0,
            initial_velocity_var: 0.1,
            q_position: 1e-5,
            q_velocity: 1e-6,
            r: 1e-4,
            alpha_q: 0.01,
            alpha_r: 0.05,
            q_min: 1e-8,
            r_min: 1e-6,
            min_ticks: 10,
        }
    }
}

/// Resultado do Position-Velocity Kalman update
#[derive(Debug, Clone, Copy)]
pub struct PVKalmanResult {
    /// Position filtrada
    pub position: f64,
    /// Velocity filtrada (taxa de mudança)
    pub velocity: f64,
    /// Innovation (residual)
    pub innovation: f64,
    /// Innovation variance
    pub innovation_var: f64,
    /// Desvio padrão da velocity
    pub velocity_std: f64,
    /// Kalman gain [K_position, K_velocity]
    pub kalman_gain: [f64; 2],
    /// Se velocity é confiável (passou warmup)
    pub is_valid: bool,
}

impl PVKalmanResult {
    /// Z-score da innovation
    #[inline]
    pub fn z_score(&self) -> f64 {
        let std = self.innovation_var.sqrt();
        if std > 1e-10 {
            self.innovation / std
        } else {
            0.0
        }
    }
}

/// Position-Velocity Kalman Filter
/// 
/// Modelo Newtoniano (Constant Velocity):
/// - position[t+1] = position[t] + velocity[t]
/// - velocity[t+1] = velocity[t] (random walk)
/// 
/// Usado para detectar reversão de tendência via velocity < 0
/// 
/// Features (análogo ao KalmanFilter2D):
/// - Q/R adaptativos via Robbins-Monro [1] §3.6
/// - Joseph form para estabilidade numérica [1] §3.4
/// - Simetrização periódica de covariância [1] §13.2
/// - Health check da matriz de covariância
#[repr(align(64))]
#[derive(Debug, Clone)]
pub struct PositionVelocityKalman {
    /// Estado: [position, velocity]
    x: [f64; 2],
    
    /// Covariância 2x2 P[t|t]
    p: [[f64; 2]; 2],
    
    /// Covariância predita P[t|t-1] - guardada após predict()
    /// para uso em adapt_r_robbins_monro() [1] §3.6.1
    p_pred: [[f64; 2]; 2],
    
    /// Process noise covariance Q (adaptativo)
    q: [[f64; 2]; 2],
    
    /// Measurement noise variance R (adaptativo, escalar)
    r: f64,
    
    /// Alpha para adaptação de Q
    alpha_q: f64,
    
    /// Alpha para adaptação de R
    alpha_r: f64,
    
    /// Floor mínimo para Q
    q_min: f64,
    
    /// Floor mínimo para R
    r_min: f64,
    
    /// Última innovation (para Q adaptativo)
    last_innovation: f64,
    
    /// Último Kalman gain (para Q adaptativo)
    last_k: [f64; 2],
    
    /// Contador de ticks para warmup
    tick_count: u64,
    
    /// Mínimo de ticks para considerar válido
    min_ticks: u64,
}

impl PositionVelocityKalman {
    /// Do PREDICT
    /// 
    /// Reference: Bar-Shalom (2001)
    pub fn predict_only(&mut self) -> PVKalmanResult {
        // State prediction (Constant Velocity Model)
        // position_pred = position + velocity
        self.x[0] = self.x[0] + self.x[1];
        // velocity permanece (random walk)
        
        // Covariance prediction: P = F × P × F' + Q
        let fp00 = self.p[0][0] + self.p[1][0];
        let fp01 = self.p[0][1] + self.p[1][1];
        let fp10 = self.p[1][0];
        let fp11 = self.p[1][1];
        
        self.p[0][0] = fp00 + fp01 + self.q[0][0];
        self.p[0][1] = fp01;
        self.p[1][0] = fp10 + fp11;
        self.p[1][1] = fp11 + self.q[1][1];
        
        self.p_pred = self.p;
        
        PVKalmanResult {
            position: self.x[0],
            velocity: self.x[1],
            innovation: 0.0,
            innovation_var: self.p[0][0] + self.r,
            velocity_std: self.p[1][1].sqrt(),
            kalman_gain: [0.0, 0.0],
            is_valid: self.tick_count >= self.min_ticks,
        }
    }
}

impl PositionVelocityKalman {
    /// Cria novo filtro com configuração padrão
    pub fn new() -> Self {
        Self::with_config(PVKalmanConfig::default())
    }
    
    /// Cria com configuração customizada
    pub fn with_config(config: PVKalmanConfig) -> Self {
        Self {
            x: [0.0, 0.0],
            p: [
                [config.initial_position_var, 0.0],
                [0.0, config.initial_velocity_var],
            ],
            p_pred: [
                [config.initial_position_var, 0.0],
                [0.0, config.initial_velocity_var],
            ],
            q: [
                [config.q_position, 0.0],
                [0.0, config.q_velocity],
            ],
            r: config.r,
            alpha_q: config.alpha_q,
            alpha_r: config.alpha_r,
            q_min: config.q_min,
            r_min: config.r_min,
            last_innovation: 0.0,
            last_k: [0.0, 0.0],
            tick_count: 0,
            min_ticks: config.min_ticks,
        }
    }
    
    /// Predict step
    /// 
    /// State transition (Constant Velocity Model):
    /// ```text
    /// F = [1, 1]  (position += velocity * dt, dt=1)
    ///     [0, 1]  (velocity = random walk)
    /// ```
    /// 
    /// x_pred = F × x
    /// P_pred = F × P × Fᵀ + Q
    /// 
    /// Guarda P_pred para adapt_r_robbins_monro() [1] §3.6.1
    #[inline]
    pub fn predict(&mut self) {
        // x_pred = F × x
        // F = [[1,1], [0,1]]
        // position_pred = position + velocity
        // velocity_pred = velocity
        self.x[0] = self.x[0] + self.x[1];
        // self.x[1] permanece (random walk)
        
        // P_pred = F × P × Fᵀ + Q
        // Expandido para F = [[1,1], [0,1]]:
        //
        // F × P = [[P00+P10, P01+P11],
        //          [P10,     P11    ]]
        //
        // (F × P) × Fᵀ = [[P00+P10+P01+P11, P01+P11],
        //                 [P10+P11,         P11    ]]
        
        let fp00 = self.p[0][0] + self.p[1][0];
        let fp01 = self.p[0][1] + self.p[1][1];
        let fp10 = self.p[1][0];
        let fp11 = self.p[1][1];
        
        // (F × P) × Fᵀ
        self.p[0][0] = fp00 + fp01 + self.q[0][0];
        self.p[0][1] = fp01;
        self.p[1][0] = fp10 + fp11;
        self.p[1][1] = fp11 + self.q[1][1];
        
        // Guardar P_pred APÓS adicionar Q [1] §3.6.1
        self.p_pred = self.p;
    }
    
    /// Update step
    /// 
    /// H = [1, 0] (observamos apenas position)
    /// 
    /// Uses Joseph form for numerical stability [1] §3.4
    #[inline]
    pub fn update(&mut self, observation: f64) -> PVKalmanResult {
        // H = [1, 0]
        
        // ═══════════════════════════════════════════════════════
        // Innovation: y = z - H × x_pred = observation - position
        // ═══════════════════════════════════════════════════════
        let innovation = observation - self.x[0];
        self.last_innovation = innovation;
        
        // ═══════════════════════════════════════════════════════
        // Innovation covariance: S = H × P × Hᵀ + R
        // Com H = [1, 0]: S = P[0][0] + R
        // ═══════════════════════════════════════════════════════
        let s = self.p[0][0] + self.r;
        let s_inv = 1.0 / s.max(1e-12);
        
        // ═══════════════════════════════════════════════════════
        // Kalman Gain: K = P × Hᵀ × S⁻¹
        // Com H = [1, 0]: K = [P[0][0], P[1][0]]ᵀ × S⁻¹
        // ═══════════════════════════════════════════════════════
        let k = [self.p[0][0] * s_inv, self.p[1][0] * s_inv];
        self.last_k = k;
        
        // ═══════════════════════════════════════════════════════
        // State update: x = x_pred + K × y
        // ═══════════════════════════════════════════════════════
        self.x[0] += k[0] * innovation;
        self.x[1] += k[1] * innovation;
        
        // ═══════════════════════════════════════════════════════
        // Covariance update (Joseph form) [1] §3.4
        // P = (I - K×H) × P × (I - K×H)ᵀ + K × R × Kᵀ
        // ═══════════════════════════════════════════════════════
        
        // I - K×H com H = [1, 0]
        // = [[1-K0, 0],
        //    [-K1,  1]]
        let i_kh = [
            [1.0 - k[0], 0.0],
            [-k[1], 1.0],
        ];
        
        // temp = (I - K×H) × P
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
        
        // P_joseph = temp × (I - K×H)ᵀ
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
        
        // K × R × Kᵀ
        let kr_kt = [
            [k[0] * self.r * k[0], k[0] * self.r * k[1]],
            [k[1] * self.r * k[0], k[1] * self.r * k[1]],
        ];
        
        // P = P_joseph + K × R × Kᵀ
        self.p = [
            [p_joseph[0][0] + kr_kt[0][0], p_joseph[0][1] + kr_kt[0][1]],
            [p_joseph[1][0] + kr_kt[1][0], p_joseph[1][1] + kr_kt[1][1]],
        ];
        
        self.tick_count += 1;
        
        PVKalmanResult {
            position: self.x[0],
            velocity: self.x[1],
            innovation,
            innovation_var: s,
            velocity_std: self.p[1][1].sqrt(),
            kalman_gain: k,
            is_valid: self.tick_count >= self.min_ticks,
        }
    }
    
    /// Combined predict + update
    #[inline]
    pub fn step(&mut self, observation: f64) -> PVKalmanResult {
        self.predict();
        self.update(observation)
    }
    
    /// Combined predict + update with adaptive Q/R (Robbins-Monro)
    /// Conforme [1] §3.6
    #[inline]
    pub fn step_adaptive(&mut self, observation: f64) -> PVKalmanResult {
        self.predict();
        let result = self.update(observation);
        self.adapt_r_robbins_monro();
        self.adapt_q_robbins_monro();
        result
    }
    
    // ═══════════════════════════════════════════════════════════════
    // Adaptive Q/R (Robbins-Monro) [1] §3.6
    // ═══════════════════════════════════════════════════════════════
    
    /// Adapt R (Measurement Noise) via Robbins-Monro
    /// Conforme [1] §3.6.1
    /// 
    /// R[t] = (1 - α_r) × R[t-1] + α_r × (y² - H × P_pred × Hᵀ)
    #[inline]
    pub fn adapt_r_robbins_monro(&mut self) {
        let innovation_sq = self.last_innovation * self.last_innovation;
        
        // H × P_pred × Hᵀ com H = [1, 0] = P_pred[0][0]
        let h_p_ht = self.p_pred[0][0];
        
        let r_update = innovation_sq - h_p_ht;
        self.r = (1.0 - self.alpha_r) * self.r + self.alpha_r * r_update;
        self.r = self.r.max(self.r_min);
    }
    
    /// Adapt Q (Process Noise) via Robbins-Monro
    /// Conforme [1] §3.6
    /// 
    /// Q[t] = (1 - α_q) × Q[t-1] + α_q × (K × y × yᵀ × Kᵀ)
    #[inline]
    pub fn adapt_q_robbins_monro(&mut self) {
        let k = self.last_k;
        let y = self.last_innovation;
        
        // K × y
        let ky = [k[0] * y, k[1] * y];
        
        // Q_update = (K × y) × (K × y)ᵀ (outer product)
        let q_update = [
            [ky[0] * ky[0], ky[0] * ky[1]],
            [ky[1] * ky[0], ky[1] * ky[1]],
        ];
        
        // Q[t] = (1 - α_q) × Q[t-1] + α_q × Q_update
        self.q[0][0] = ((1.0 - self.alpha_q) * self.q[0][0] + self.alpha_q * q_update[0][0]).max(self.q_min);
        self.q[0][1] = (1.0 - self.alpha_q) * self.q[0][1] + self.alpha_q * q_update[0][1];
        self.q[1][0] = (1.0 - self.alpha_q) * self.q[1][0] + self.alpha_q * q_update[1][0];
        self.q[1][1] = ((1.0 - self.alpha_q) * self.q[1][1] + self.alpha_q * q_update[1][1]).max(self.q_min);
    }
    
    // ═══════════════════════════════════════════════════════════════
    // Numerical Stability [1] §13.2
    // ═══════════════════════════════════════════════════════════════
    
    /// Simetriza matriz de covariância
    /// P = (P + Pᵀ) / 2
    /// Conforme [1] §13.2
    #[inline]
    pub fn symmetrize_covariance(&mut self) -> bool {
        let asymmetry = (self.p[0][1] - self.p[1][0]).abs();
        
        if asymmetry > 1e-15 {
            let avg = (self.p[0][1] + self.p[1][0]) / 2.0;
            self.p[0][1] = avg;
            self.p[1][0] = avg;
            
            // Também p_pred
            let avg_pred = (self.p_pred[0][1] + self.p_pred[1][0]) / 2.0;
            self.p_pred[0][1] = avg_pred;
            self.p_pred[1][0] = avg_pred;
            
            return true;
        }
        false
    }
    
    /// Verifica saúde da matriz de covariância
    #[inline]
    pub fn check_covariance_health(&self) -> (bool, CovarianceHealth) {
        let trace = self.p[0][0] + self.p[1][1];
        let det = self.p[0][0] * self.p[1][1] - self.p[0][1] * self.p[1][0];
        let asymmetry = (self.p[0][1] - self.p[1][0]).abs();
        
        let is_positive_definite = det > 1e-12 && trace > 0.0;
        let is_symmetric = asymmetry < 1e-10;
        
        let health = CovarianceHealth {
            determinant: det,
            trace,
            asymmetry,
            is_positive_definite,
            is_symmetric,
        };
        
        (is_positive_definite && is_symmetric, health)
    }
    
    // ═══════════════════════════════════════════════════════════════
    // Accessors
    // ═══════════════════════════════════════════════════════════════
    
    /// Retorna velocity atual (taxa de mudança filtrada)
    #[inline]
    pub fn velocity(&self) -> f64 {
        self.x[1]
    }
    
    /// Retorna position atual
    #[inline]
    pub fn position(&self) -> f64 {
        self.x[0]
    }
    
    /// Retorna estado completo [position, velocity]
    #[inline]
    pub fn state(&self) -> [f64; 2] {
        self.x
    }
    
    /// Retorna covariância
    #[inline]
    pub fn covariance(&self) -> [[f64; 2]; 2] {
        self.p
    }
    
    /// Retorna Q atual
    #[inline]
    pub fn q(&self) -> [[f64; 2]; 2] {
        self.q
    }
    
    /// Retorna R atual
    #[inline]
    pub fn r(&self) -> f64 {
        self.r
    }
    
    /// Retorna desvio padrão da velocity
    #[inline]
    pub fn velocity_std(&self) -> f64 {
        self.p[1][1].sqrt()
    }
    
    /// Verifica se sinal está revertendo (velocity negativa)
    #[inline]
    pub fn is_reverting(&self) -> bool {
        self.tick_count >= self.min_ticks && self.x[1] < 0.0
    }
    
    /// Verifica se reversão é confirmada com threshold mínimo
    #[inline]
    pub fn is_reverting_with_threshold(&self, min_negative_velocity: f64) -> bool {
        self.tick_count >= self.min_ticks && self.x[1] < min_negative_velocity
    }
    
    /// Retorna contador de ticks
    #[inline]
    pub fn tick_count(&self) -> u64 {
        self.tick_count
    }
    
    /// Verifica se está warmed up
    #[inline]
    pub fn is_warmed_up(&self) -> bool {
        self.tick_count >= self.min_ticks
    }
    
    /// Reseta o filtro
    pub fn reset(&mut self) {
        self.x = [0.0, 0.0];
        self.p = [[1.0, 0.0], [0.0, 0.1]];
        self.p_pred = [[1.0, 0.0], [0.0, 0.1]];
        self.q = [[1e-5, 0.0], [0.0, 1e-6]];
        self.r = 1e-4;
        self.last_innovation = 0.0;
        self.last_k = [0.0, 0.0];
        self.tick_count = 0;
    }
}

impl Default for PositionVelocityKalman {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod pv_kalman_tests {
    use super::*;
    
    #[test]
    fn test_p_pred_initialization() {
        let pv = PositionVelocityKalman::new();
        assert!((pv.p_pred[0][0] - 1.0).abs() < 0.001);
        assert!((pv.p_pred[1][1] - 0.1).abs() < 0.001);
    }
    
    #[test]
    fn test_p_pred_after_predict() {
        let mut pv = PositionVelocityKalman::new();
        pv.predict();
        // p_pred deve ser atualizado após predict
        assert!(pv.p_pred[0][0] > 1.0); // Aumentou com Q
    }
    
    #[test]
    fn test_velocity_detection() {
        let mut pv = PositionVelocityKalman::new();
        
        // Simular sinal subindo
        for i in 0..20 {
            pv.step_adaptive(i as f64 * 0.1);
        }
        
        // Velocity deve ser positiva
        assert!(pv.velocity() > 0.0);
        assert!(!pv.is_reverting());
        
        // Simular reversão
        for i in (0..15).rev() {
            pv.step_adaptive(2.0 - (15 - i) as f64 * 0.15);
        }
        
        // Velocity deve ser negativa
        assert!(pv.velocity() < 0.0);
        assert!(pv.is_reverting());
    }
    
    #[test]
    fn test_adaptive_q_r() {
        let mut pv = PositionVelocityKalman::new();
        let initial_r = pv.r();
        let initial_q00 = pv.q()[0][0];
        
        // Processar dados com variância
        for i in 0..50 {
            let noise = if i % 2 == 0 { 0.1 } else { -0.1 };
            pv.step_adaptive(i as f64 * 0.05 + noise);
        }
        
        // Q e R devem ter adaptado
        assert!((pv.r() - initial_r).abs() > 1e-10 || pv.r() >= pv.r_min);
        assert!((pv.q()[0][0] - initial_q00).abs() > 1e-12 || pv.q()[0][0] >= pv.q_min);
    }
    
    #[test]
    fn test_symmetrize() {
        let mut pv = PositionVelocityKalman::new();
        
        // Forçar assimetria
        pv.p[0][1] = 0.01;
        pv.p[1][0] = 0.02;
        
        let corrected = pv.symmetrize_covariance();
        assert!(corrected);
        assert!((pv.p[0][1] - pv.p[1][0]).abs() < 1e-15);
    }
    
    #[test]
    fn test_health_check() {
        let pv = PositionVelocityKalman::new();
        let (healthy, health) = pv.check_covariance_health();
        
        assert!(healthy);
        assert!(health.is_positive_definite);
        assert!(health.is_symmetric);
        assert!(health.determinant > 0.0);
    }
    
    #[test]
    fn test_warmup() {
        let mut pv = PositionVelocityKalman::with_config(PVKalmanConfig {
            min_ticks: 10,
            ..Default::default()
        });
        
        for i in 0..9 {
            let result = pv.step_adaptive(i as f64);
            assert!(!result.is_valid);
            assert!(!pv.is_warmed_up());
        }
        
        let result = pv.step_adaptive(9.0);
        assert!(result.is_valid);
        assert!(pv.is_warmed_up());
    }
    
    #[test]
    fn test_reset() {
        let mut pv = PositionVelocityKalman::new();
        
        for i in 0..50 {
            pv.step_adaptive(i as f64);
        }
        
        pv.reset();
        
        assert_eq!(pv.tick_count(), 0);
        assert!((pv.position() - 0.0).abs() < 1e-10);
        assert!((pv.velocity() - 0.0).abs() < 1e-10);
        assert!((pv.p[0][0] - 1.0).abs() < 0.001);
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

    #[test]
    fn test_kalman_2d_p_pred_initialization() {
        let kf = KalmanFilter2D::default_tick_engine();
        
        // p_pred deve estar inicializado
        assert!((kf.p_pred[0][0] - 1.0).abs() < 0.001);
        assert!((kf.p_pred[1][1] - 0.01).abs() < 0.001);
    }

    #[test]
    fn test_kalman_2d_p_pred_after_predict() {
        let mut kf = KalmanFilter2D::default_tick_engine();
        
        // Após predict, p_pred deve ser P + Q
        kf.predict();
        
        // p_pred[0][0] = p_initial[0][0] + q[0][0] = 1.0 + 1e-6
        assert!((kf.p_pred[0][0] - (1.0 + 1e-6)).abs() < 1e-9);
    }

    #[test]
    fn test_kalman_2d_reset_includes_p_pred() {
        let mut kf = KalmanFilter2D::default_tick_engine();
        
        // Modificar estado
        kf.step(100.0, [100.0, 1.0]);
        kf.step(101.0, [100.5, 1.0]);
        
        // Reset
        kf.reset();
        
        // p_pred deve estar resetado
        assert!((kf.p_pred[0][0] - 1.0).abs() < 0.001);
        assert!((kf.p_pred[1][1] - 0.01).abs() < 0.001);
    }
}
