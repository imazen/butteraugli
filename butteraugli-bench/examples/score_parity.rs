//! Print Rust vs C++ butteraugli scores on synthetic test inputs.
//! Use to confirm a libjxl patch hasn't broken score parity.
//!
//! Coverage targets things our in-tree butteraugli tests already cover but
//! at the cross-implementation level (Rust ↔ libjxl-FFI):
//!
//! - Sizes that bracket every kernel-size transition. butteraugli derives
//!   kernel size from sigma (`len = 2*max(1, 2.25*sigma)+1`) for fixed
//!   `kSigma{Lf,Hf,Uhf}`. The case-N unrolls live at sizes 7/13/15/33,
//!   so dims around {7,13,15,33} probe the boundary between border and
//!   interior code, and dims < 32 probe the V-pass top/bottom-border
//!   overlap region. Asymmetric shapes flush both H and V border code.
//!
//! - Three content classes (smooth gradient, sharp edge, high-frequency
//!   random). High-frequency content tends to surface drift the smooth
//!   case can hide because everything is band-limited.

use butteraugli::{ButteraugliParams, ButteraugliReference};

#[cfg(has_cpp_butteraugli)]
unsafe extern "C" {
    fn butteraugli_from_linear_planes(
        src0: *const f32,
        src1: *const f32,
        src2: *const f32,
        dst0: *const f32,
        dst1: *const f32,
        dst2: *const f32,
        width: usize,
        height: usize,
    ) -> f64;
}

#[derive(Clone, Copy)]
enum Pattern {
    /// Diagonal gradient + per-pixel sub-8-LSB perturbation. Smooth, low-noise.
    Gradient,
    /// Solid background with a vertical edge at x=w/2; perturbation only on the
    /// right half, scaled larger so the mid-frequency band is exercised.
    Edge,
    /// LCG random per-pixel noise, plus a different LCG random distortion.
    /// Maximum frequency content; stresses the malta high-frequency path.
    Random,
}

fn perturb(s: u32, base: u8, max_delta: u8) -> u8 {
    base.saturating_add((s & ((max_delta as u32 * 2) - 1)) as u8)
        .saturating_sub(max_delta)
}

fn make_test_planes(
    w: usize,
    h: usize,
    seed: u32,
    pattern: Pattern,
) -> ([Vec<f32>; 3], [Vec<f32>; 3]) {
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
    let mut s_src = seed;
    let mut s_dst = seed.wrapping_mul(2654435761).wrapping_add(0xC0FFEE);
    for i in 0..n {
        s_src = s_src.wrapping_mul(1664525).wrapping_add(1013904223);
        s_dst = s_dst.wrapping_mul(22695477).wrapping_add(1);
        let x = i % w;
        let y = i / w;

        let (r, g, b, r2, g2, b2): (u8, u8, u8, u8, u8, u8) = match pattern {
            Pattern::Gradient => {
                let r = (x * 255 / w) as u8;
                let g = (y * 255 / h) as u8;
                let b = r.wrapping_add(g);
                (
                    r,
                    g,
                    b,
                    r.saturating_add((s_src & 7) as u8),
                    g.saturating_add(((s_src >> 4) & 5) as u8),
                    b.saturating_add(((s_src >> 8) & 3) as u8),
                )
            }
            Pattern::Edge => {
                let half = w / 2;
                let r = if x < half { 64 } else { 192 };
                let g = if x < half { 80 } else { 176 };
                let b = if x < half { 100 } else { 156 };
                let delta = if x < half { 0 } else { 16 };
                (
                    r,
                    g,
                    b,
                    r.saturating_add(delta),
                    g.saturating_add(delta),
                    b.saturating_sub(delta / 2),
                )
            }
            Pattern::Random => {
                let r = (s_src & 0xFF) as u8;
                let g = ((s_src >> 8) & 0xFF) as u8;
                let b = ((s_src >> 16) & 0xFF) as u8;
                (
                    r,
                    g,
                    b,
                    perturb(s_dst, r, 32),
                    perturb(s_dst >> 5, g, 24),
                    perturb(s_dst >> 11, b, 32),
                )
            }
        };
        src[0].push(linear_srgb::default::srgb_u8_to_linear(r));
        src[1].push(linear_srgb::default::srgb_u8_to_linear(g));
        src[2].push(linear_srgb::default::srgb_u8_to_linear(b));
        dst[0].push(linear_srgb::default::srgb_u8_to_linear(r2));
        dst[1].push(linear_srgb::default::srgb_u8_to_linear(g2));
        dst[2].push(linear_srgb::default::srgb_u8_to_linear(b2));
    }
    (src, dst)
}

fn main() {
    // Sizes derived from butteraugli/tests/capture_cpp_scores.rs::test_cases().
    // The small bracket covers every kernel-size boundary: 8 (just over the
    // libjxl ≥8 floor), 9, 15/16/17 around case-15 offset, 23/24, 31/32/33
    // around case-33 offset, then 47/48/64. Asymmetric mixes the H and V
    // border code and would have caught the iter1 bug.
    let sizes: &[(usize, usize)] = &[
        // square — every kernel boundary
        (8, 8),
        (9, 9),
        (15, 15),
        (16, 16),
        (17, 17),
        (23, 23),
        (24, 24),
        (31, 31),
        (32, 32),
        (33, 33),
        (47, 47),
        (48, 48),
        (64, 64),
        // asymmetric — exercises H and V borders independently
        (23, 31),
        (31, 23),
        (47, 33),
        (33, 47),
        (96, 128),
        (128, 96),
        // larger / production-realistic
        (256, 256),
        (512, 512),
        (1280, 720),
        (1920, 1080),
    ];
    let patterns: &[(Pattern, &str)] = &[
        (Pattern::Gradient, "gradient"),
        (Pattern::Edge, "edge"),
        (Pattern::Random, "random"),
    ];

    println!(
        "{:>10} {:>10} {:>14} {:>14} {:>14} {:>10}",
        "pattern", "size", "rust", "cpp", "abs_diff", "rel%"
    );

    let mut max_rel = 0.0_f64;
    let mut max_rel_label = String::new();
    let mut total = 0usize;
    for (i, &(w, h)) in sizes.iter().enumerate() {
        for &(pattern, pname) in patterns {
            // Stable per-(size, pattern) seed so reruns are deterministic.
            let seed = (i as u32).wrapping_mul(2654435761) ^ (pname.len() as u32 * 7919);
            let (src, dst) = make_test_planes(w, h, seed, pattern);

            let params = ButteraugliParams::default();
            let reference = ButteraugliReference::new_linear_planar(
                &src[0], &src[1], &src[2], w, h, w, params,
            )
            .unwrap();
            let rust_score = reference
                .compare_linear_planar(&dst[0], &dst[1], &dst[2], w)
                .unwrap()
                .score;

            #[cfg(has_cpp_butteraugli)]
            let cpp_score = unsafe {
                butteraugli_from_linear_planes(
                    src[0].as_ptr(),
                    src[1].as_ptr(),
                    src[2].as_ptr(),
                    dst[0].as_ptr(),
                    dst[1].as_ptr(),
                    dst[2].as_ptr(),
                    w,
                    h,
                )
            };
            #[cfg(not(has_cpp_butteraugli))]
            let cpp_score = f64::NAN;

            let abs = (rust_score as f64 - cpp_score).abs();
            // Avoid div-by-zero on identical-image cases.
            let rel = if cpp_score.abs() > 1e-9 {
                abs / cpp_score.abs() * 100.0
            } else {
                0.0
            };
            if rel > max_rel {
                max_rel = rel;
                max_rel_label = format!("{} {}x{}", pname, w, h);
            }
            total += 1;
            println!(
                "{:>10} {:>10} {:>14.6} {:>14.6} {:>14.2e} {:>9.4}%",
                pname,
                format!("{}x{}", w, h),
                rust_score,
                cpp_score,
                abs,
                rel
            );
        }
    }
    println!();
    println!(
        "{} cases, max relative diff: {:.4}% ({})",
        total, max_rel, max_rel_label
    );
}
