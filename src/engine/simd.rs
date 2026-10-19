/// Hardware SIMD Feature Auto-Detection across all major CPU architectures:
/// - x86_64: AVX-512 (F, CD, VL, BW, DQ), AVX2, FMA, AVX, SSE4.2, SSE4.1, SSSE3, SSE2
/// - AArch64 (ARM64): ARM NEON, SVE (Scalable Vector Extension), SVE2, FP16, DotProd
/// - RISC-V (riscv64): RVV (Standard Vector Extension 'V'), Zve32f, Zve64d
#[derive(Debug, Clone)]
pub struct SimdCapabilities {
    pub arch: &'static str,
    pub flags: Vec<&'static str>,
}

impl SimdCapabilities {
    pub fn detect() -> Self {
        let mut flags = Vec::new();

        #[cfg(target_arch = "x86_64")]
        {
            // --- x86_64 SIMD Hierarchy ---
            // 1. AVX-512 family
            if is_x86_feature_detected!("avx512f") { flags.push("AVX-512F"); }
            if is_x86_feature_detected!("avx512bw") { flags.push("AVX-512BW"); }
            if is_x86_feature_detected!("avx512cd") { flags.push("AVX-512CD"); }
            if is_x86_feature_detected!("avx512dq") { flags.push("AVX-512DQ"); }
            if is_x86_feature_detected!("avx512vl") { flags.push("AVX-512VL"); }

            // 2. AVX2 & FMA (Zen 1/2/3, Intel Haswell - Comet Lake)
            if is_x86_feature_detected!("avx2") { flags.push("AVX2"); }
            if is_x86_feature_detected!("fma") { flags.push("FMA3"); }
            if is_x86_feature_detected!("avx") { flags.push("AVX"); }

            // 3. Legacy SSE fallback (older Intel Nehalem / AMD K10)
            if is_x86_feature_detected!("sse4.2") { flags.push("SSE4.2"); }
            if is_x86_feature_detected!("sse4.1") { flags.push("SSE4.1"); }
            if is_x86_feature_detected!("ssse3") { flags.push("SSSE3"); }
            if is_x86_feature_detected!("sse2") { flags.push("SSE2"); }

            Self {
                arch: "x86_64",
                flags,
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // --- AArch64 (ARM64) SIMD Hierarchy ---
            // NEON is baseline on ARMv8-A (Apple Silicon M1-M4, AWS Graviton 2/3/4, Raspberry Pi 4/5)
            flags.push("ARM-NEON");

            #[cfg(target_feature = "sve")]
            flags.push("ARM-SVE");

            #[cfg(target_feature = "sve2")]
            flags.push("ARM-SVE2");

            #[cfg(target_feature = "fp16")]
            flags.push("ARM-FP16");

            #[cfg(target_feature = "dotprod")]
            flags.push("ARM-DotProd");

            Self {
                arch: "aarch64",
                flags,
            }
        }

        #[cfg(target_arch = "riscv64")]
        {
            // --- RISC-V 64-bit Vector Hierarchy ---
            #[cfg(target_feature = "v")]
            flags.push("RISC-V-Vector (RVV 1.0)");

            #[cfg(target_feature = "zba")]
            flags.push("Zba (Address Gen)");

            #[cfg(target_feature = "zbb")]
            flags.push("Zbb (Basic Bit-manip)");

            if flags.is_empty() {
                flags.push("RISC-V Scalar Fallback");
            }

            Self {
                arch: "riscv64",
                flags,
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "riscv64")))]
        {
            flags.push("Generic-Scalar");
            Self {
                arch: "unknown",
                flags,
            }
        }
    }

    pub fn summary(&self) -> String {
        format!("Arch: {} [{}]", self.arch, self.flags.join(", "))
    }
}

/// Hardware-accelerated vector dot-product calculation for 512-D embeddings.
/// Automatically executes the fastest vector path supported by the target silicon.
pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector lengths must match");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                return dot_product_avx2_fma(a, b);
            }
        }
    }

    // Portable auto-vectorized loop (LLVM translates this into ARM NEON or RISC-V vector ops)
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_product_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 8;
    let mut sum_vec = _mm256_setzero_ps();

    let mut p_a = a.as_ptr();
    let mut p_b = b.as_ptr();

    for _ in 0..chunks {
        let va = _mm256_loadu_ps(p_a);
        let vb = _mm256_loadu_ps(p_b);
        // Fused Multiply-Add: sum = (va * vb) + sum
        sum_vec = _mm256_fmadd_ps(va, vb, sum_vec);

        p_a = p_a.add(8);
        p_b = p_b.add(8);
    }

    // Horizontal reduction of 8-float 256-bit register
    let mut buffer = [0.0f32; 8];
    _mm256_storeu_ps(buffer.as_mut_ptr(), sum_vec);
    let mut total: f32 = buffer.iter().sum();

    // Remainder loop
    for i in (chunks * 8)..len {
        total += *a.get_unchecked(i) * *b.get_unchecked(i);
    }

    total
}
