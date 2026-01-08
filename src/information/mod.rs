/// Shannon entropy: H(X) = -Σ p(x) log p(x)
/// 
/// Measures uncertainty/information content of a distribution.
/// Higher entropy = more uncertain.
#[inline]
pub fn entropy<const N: usize>(pmf: &[f64; N]) -> f64 {
    let mut h = 0.0;
    for &p in pmf {
        if p > 1e-300 {
            h -= p * p.ln();
        }
    }
    h
}

/// Normalized entropy (0 to 1)
#[inline]
pub fn entropy_normalized<const N: usize>(pmf: &[f64; N]) -> f64 {
    entropy(pmf) / (N as f64).ln()
}

/// KL divergence: D_KL(P||Q) = Σ p(x) log(p(x)/q(x))
/// 
/// Measures how much distribution P diverges from Q.
/// Always non-negative, zero iff P = Q.
#[inline]
pub fn kl_divergence<const N: usize>(p: &[f64; N], q: &[f64; N]) -> f64 {
    let mut kl = 0.0;
    for i in 0..N {
        if p[i] > 1e-300 && q[i] > 1e-300 {
            kl += p[i] * (p[i] / q[i]).ln();
        }
    }
    kl
}

/// Cross-entropy: H(P, Q) = -Σ p(x) log q(x)
#[inline]
pub fn cross_entropy<const N: usize>(p: &[f64; N], q: &[f64; N]) -> f64 {
    let mut ce = 0.0;
    for i in 0..N {
        if p[i] > 1e-300 && q[i] > 1e-300 {
            ce -= p[i] * q[i].ln();
        }
    }
    ce
}

/// Mutual information: I(X;Y) = Σ p(x,y) log(p(x,y)/(p(x)p(y)))
/// 
/// Measures dependence between two variables.
/// Zero iff X and Y are independent.
pub fn mutual_information<const M: usize, const N: usize>(
    joint_pmf: &[[f64; N]; M],
) -> f64 {
    // Compute marginals
    let mut px = [0.0; M];
    let mut py = [0.0; N];
    
    for i in 0..M {
        for j in 0..N {
            px[i] += joint_pmf[i][j];
            py[j] += joint_pmf[i][j];
        }
    }
    
    // Compute MI
    let mut mi = 0.0;
    for i in 0..M {
        for j in 0..N {
            let pxy = joint_pmf[i][j];
            if pxy > 1e-300 && px[i] > 1e-300 && py[j] > 1e-300 {
                mi += pxy * (pxy / (px[i] * py[j])).ln();
            }
        }
    }
    
    mi
}

/// Jensen-Shannon divergence: symmetric version of KL
/// 
/// JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
/// where M = 0.5 * (P + Q)
#[inline]
pub fn jensen_shannon_divergence<const N: usize>(p: &[f64; N], q: &[f64; N]) -> f64 {
    // Compute midpoint
    let mut m = [0.0; N];
    for i in 0..N {
        m[i] = 0.5 * (p[i] + q[i]);
    }
    
    0.5 * kl_divergence(p, &m) + 0.5 * kl_divergence(q, &m)
}

/// Fisher information (for Gaussian)
/// 
/// I(θ) = 1/σ²
#[inline]
pub fn fisher_information_gaussian(variance: f64) -> f64 {
    1.0 / variance
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_entropy() {
        // Uniform distribution: maximum entropy
        let uniform = [0.25, 0.25, 0.25, 0.25];
        let h_uniform = entropy(&uniform);
        assert!((h_uniform - (4.0_f64).ln()).abs() < 1e-10);
        
        // Deterministic: zero entropy
        let deterministic = [1.0, 0.0, 0.0, 0.0];
        let h_det = entropy(&deterministic);
        assert!(h_det < 1e-10);
    }
    
    #[test]
    fn test_kl_divergence() {
        let p = [0.5, 0.5];
        let q = [0.5, 0.5];
        
        // Same distribution: zero divergence
        assert!(kl_divergence(&p, &q) < 1e-10);
        
        let r = [0.9, 0.1];
        // Different: positive divergence
        assert!(kl_divergence(&p, &r) > 0.0);
    }
    
    #[test]
    fn test_mutual_information() {
        // Independent variables: MI = 0
        let independent = [
            [0.25, 0.25],
            [0.25, 0.25],
        ];
        let mi_indep = mutual_information(&independent);
        assert!(mi_indep.abs() < 1e-10);
        
        // Perfectly correlated: MI > 0
        let correlated = [
            [0.5, 0.0],
            [0.0, 0.5],
        ];
        let mi_corr = mutual_information(&correlated);
        assert!(mi_corr > 0.5);
    }
}