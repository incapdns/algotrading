use std::sync::OnceLock;

static CPU_FEATURES: OnceLock<CpuFeatures> = OnceLock::new();

#[derive(Debug, Clone, Copy)]
pub struct CpuFeatures {
    pub has_avx2: bool,
    pub has_avx512f: bool,
    pub has_fma: bool,
}

impl CpuFeatures {
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                has_avx2: is_x86_feature_detected!("avx2"),
                has_avx512f: is_x86_feature_detected!("avx512f"),
                has_fma: is_x86_feature_detected!("fma"),
            }
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self {
                has_avx2: false,
                has_avx512f: false,
                has_fma: false,
            }
        }
    }
}

pub fn cpu_features() -> &'static CpuFeatures {
    CPU_FEATURES.get_or_init(CpuFeatures::detect)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_feature_detection() {
        let features = cpu_features();
        println!("CPU Features: {:?}", features);
        #[cfg(target_arch = "x86_64")]
        assert!(features.has_avx2, "WARNING! Your CPU should support AVX2");
    }
}