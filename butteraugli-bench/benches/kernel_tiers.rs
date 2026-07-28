//! Per-kernel SIMD-tier isolation for butteraugli's hot kernels.
//!
//! `tier_isolation.rs` measures the whole `compare` end-to-end. That cannot
//! reveal a single kernel that is SLOWER than its own scalar fallback — the
//! faster kernels average it away. That exact failure mode was found and fixed
//! in garb, zensim, zentone, zenpng and zenresize during the 2026-07-28
//! aarch64 sweep, so butteraugli's kernels are checked individually rather
//! than assumed healthy from the end-to-end ratio.
//!
//! This matters here more than in most crates: butteraugli is the inner loop
//! of jxl-encoder's quantization loop, which runs it 2-4x per encode.
//!
//! NOTE: on aarch64 NEON is BASELINE, so the "scalar" arm is still fully
//! autovectorized by LLVM. A ratio near 1.00 therefore does NOT mean a kernel
//! is missing — it means both arms compiled to equivalent work.
//!
//! Run: `cargo bench -p butteraugli-bench --bench kernel_tiers`
//! Do NOT pass `-C target-cpu=native`: that pins the tier at compile time,
//! after which it cannot be disabled and this bench skips rather than
//! silently reporting the SIMD path under both labels.

use butteraugli::blur::{blur_mirrored_5x5, gaussian_blur};
use butteraugli::image::{BufferPool, ImageF};
use butteraugli::malta::malta_diff_map;
use zenbench::black_box;

#[cfg(target_arch = "aarch64")]
type TierToken = archmage::NeonToken;
#[cfg(target_arch = "x86_64")]
type TierToken = archmage::X64V3Token;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const TIER_NAME: &str = if cfg!(target_arch = "aarch64") {
    "neon"
} else {
    "v3(avx2)"
};

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn set_simd(enabled: bool) -> bool {
    TierToken::dangerously_disable_token_process_wide(!enabled).is_ok()
}
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn set_simd(_enabled: bool) -> bool {
    false
}

/// Structured noise. A flat or gradient fill would give the Malta filter
/// degenerate diffs and understate exactly the kernel being measured.
fn img(w: usize, h: usize, seed: u32) -> ImageF {
    let mut im = ImageF::new(w, h);
    let mut s = seed | 1;
    for y in 0..h {
        let row = im.row_mut(y);
        for px in row.iter_mut().take(w) {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *px = (s >> 8) as f32 / 16_777_216.0;
        }
    }
    im
}

zenbench::main!(|suite| {
    if !set_simd(true) || !set_simd(false) {
        eprintln!(
            "[kernel_tiers] no toggleable SIMD tier here, or the tier is \
             compile-time guaranteed (drop -C target-cpu=native). Skipping."
        );
        return;
    }
    set_simd(true);
    eprintln!("[kernel_tiers] comparing {TIER_NAME} vs forced scalar");

    for &(label, w, h) in &[("512x512", 512usize, 512usize), ("1920x1080", 1920, 1080)] {
        // Malta: the asymmetric artifact filter, the heaviest single kernel.
        suite.compare(format!("malta_diff_map/{label}"), |group| {
            for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
                group.bench(arm, move |b| {
                    let a = img(w, h, 7);
                    let c = img(w, h, 13);
                    let pool = BufferPool::new();
                    b.iter(|| {
                        // Toggling inside the closure is required because
                        // zenbench interleaves the arms. It costs one atomic
                        // store per iteration and applies to both arms
                        // equally, so it cannot bias the comparison.
                        set_simd(simd);
                        malta_diff_map(
                            black_box(&a),
                            black_box(&c),
                            1.0,
                            1.0,
                            2.0,
                            false,
                            &pool,
                        )
                    })
                });
            }
        });

        // Gaussian blur at several sigmas: the kernel radius (and so the
        // inner-loop trip count) scales with sigma, so one sigma would pin
        // only one point on that curve.
        for sigma in [1.2f32, 3.0, 7.0] {
            suite.compare(format!("gaussian_blur/s{sigma}/{label}"), |group| {
                for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
                    group.bench(arm, move |b| {
                        let a = img(w, h, 7);
                        let pool = BufferPool::new();
                        b.iter(|| {
                            set_simd(simd);
                            gaussian_blur(black_box(&a), sigma, &pool)
                        })
                    });
                }
            });
        }

        // The fixed 5x5 mirrored blur used by the mask stage.
        suite.compare(format!("blur_mirrored_5x5/{label}"), |group| {
            for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
                group.bench(arm, move |b| {
                    let a = img(w, h, 7);
                    let weights = [0.4f32, 0.25, 0.05];
                    let pool = BufferPool::new();
                    b.iter(|| {
                        set_simd(simd);
                        blur_mirrored_5x5(black_box(&a), &weights, &pool)
                    })
                });
            }
        });
    }
    set_simd(true);
});
