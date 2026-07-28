//! SIMD-tier isolation: the native top tier vs the same code forced to scalar.
//!
//! `bench_compare` measures this crate against libjxl's C++ implementation.
//! That answers "are we competitive", not "is our SIMD worth anything". This
//! bench answers the second question by running the identical Rust pipeline
//! with the native SIMD token disabled, so a kernel that is slower than the
//! scalar fallback it replaces shows up instead of hiding behind a favourable
//! comparison against C++.
//!
//! Single-threaded on both arms — a rayon pool would let thread scheduling
//! noise swamp the SIMD delta.
//!
//! Run: `cargo bench -p butteraugli-bench --bench tier_isolation`
//! Do NOT build with `-C target-cpu=native`: that makes the tier compile-time
//! guaranteed, after which it cannot be disabled and this bench skips rather
//! than silently reporting the SIMD path under both labels.

use butteraugli::{ButteraugliParams, ButteraugliReference};
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

/// Size sweep: butteraugli's blur and malta passes change character with image
/// size (kernel-bound at small sizes, bandwidth-bound at large), so a single
/// size cannot tell you whether the SIMD paths are earning their keep.
const SIZES: &[(&str, usize, usize)] = &[
    ("256x256", 256, 256),
    ("512x512", 512, 512),
    ("1920x1080", 1920, 1080),
    ("3840x2160", 3840, 2160),
];

fn make_test_planes(w: usize, h: usize) -> ([Vec<f32>; 3], [Vec<f32>; 3]) {
    let n = w * h;
    let mut src = [
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    ];
    let mut dst = [
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    ];
    for i in 0..n {
        let x = (i % w) * 255 / w;
        let y = (i / w) * 255 / h;
        let r = x as u8;
        let g = y as u8;
        let b = (x as u8).wrapping_add(y as u8);
        src[0].push(linear_srgb::default::srgb_u8_to_linear(r));
        src[1].push(linear_srgb::default::srgb_u8_to_linear(g));
        src[2].push(linear_srgb::default::srgb_u8_to_linear(b));
        dst[0].push(linear_srgb::default::srgb_u8_to_linear(r.saturating_add(5)));
        dst[1].push(linear_srgb::default::srgb_u8_to_linear(g.saturating_add(3)));
        dst[2].push(linear_srgb::default::srgb_u8_to_linear(b));
    }
    (src, dst)
}

zenbench::main!(|suite| {
    if !set_simd(true) || !set_simd(false) {
        eprintln!(
            "[tier_isolation] no toggleable SIMD tier on this target, or the tier \
             is compile-time guaranteed (drop -C target-cpu=native, and ensure \
             archmage/testable_dispatch is enabled). Skipping."
        );
        return;
    }
    set_simd(true);
    eprintln!("[tier_isolation] comparing {TIER_NAME} vs forced scalar (1 thread)");

    for &(label, w, h) in SIZES {
        suite.compare(format!("butteraugli_{label}"), |group| {
            let (src, dst) = make_test_planes(w, h);
            for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
                let src = src.clone();
                let dst = dst.clone();
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(1)
                    .build()
                    .unwrap();
                group.bench(arm, move |b| {
                    b.iter(|| {
                        // Toggling here costs one atomic store per iteration and
                        // applies to both arms equally, so it cannot bias the
                        // comparison; it must be inside the closure because
                        // zenbench interleaves the arms.
                        set_simd(simd);
                        pool.install(|| {
                            let params = ButteraugliParams::default();
                            let reference = ButteraugliReference::new_linear_planar(
                                black_box(&src[0]),
                                black_box(&src[1]),
                                black_box(&src[2]),
                                w,
                                h,
                                w,
                                params,
                            )
                            .unwrap();
                            let result = reference
                                .compare_linear_planar(
                                    black_box(&dst[0]),
                                    black_box(&dst[1]),
                                    black_box(&dst[2]),
                                    w,
                                )
                                .unwrap();
                            black_box(result.score)
                        })
                    })
                });
            }
        });
    }

    set_simd(true);
});
