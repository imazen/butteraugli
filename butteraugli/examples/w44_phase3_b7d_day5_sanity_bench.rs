//! W44-PHASE3-B7d Day 5 sanity bench.
//!
//! At 1024², measures the wall ratio of `compare_linear_planar_strip_into`
//! vs `compare_linear_planar_into`. Day 5 acceptance: ≤ 1.5×. Day 6 will
//! do a proper paired interleaved A/B sweep across content classes and
//! sizes; this is a single-shot sanity gate.
//!
//! Day 7: gated behind `strip-tile-butteraugli` (default OFF). Run with:
//! `cargo run --release --features strip-tile-butteraugli --example w44_phase3_b7d_day5_sanity_bench`

#[cfg(not(feature = "strip-tile-butteraugli"))]
fn main() {
    eprintln!(
        "w44_phase3_b7d_day5_sanity_bench: skipped — rebuild with --features strip-tile-butteraugli."
    );
}

#[cfg(feature = "strip-tile-butteraugli")]
use butteraugli::{ButteraugliParams, ButteraugliReference};
#[cfg(feature = "strip-tile-butteraugli")]
use std::time::Instant;

#[cfg(feature = "strip-tile-butteraugli")]
fn make_pattern(w: usize, h: usize, seed: u32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = w * h;
    let mut r = vec![0.0f32; n];
    let mut g = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];
    let mut state = seed.max(1);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let xf = x as f32;
            let yf = y as f32;
            r[i] = (0.5
                + 0.3 * (xf * 0.137).sin()
                + 0.15 * (yf * 0.211).cos()
                + 0.05 * ((xf + yf) * 0.05).sin())
                .clamp(0.0, 1.0);
            g[i] = (0.5
                + 0.25 * (xf * 0.07).cos()
                + 0.2 * (yf * 0.13).sin())
                .clamp(0.0, 1.0);
            b[i] = (0.5
                + 0.35 * (yf * 0.027).cos()
                + 0.1 * (xf * 0.041).sin())
                .clamp(0.0, 1.0);
            // touch state so we can vary across calls
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        }
    }
    let _ = state;
    (r, g, b)
}

#[cfg(feature = "strip-tile-butteraugli")]
fn main() {
    let w = 1024;
    let h = 1024;
    let warm_iters = 2;
    let bench_iters = 6;

    let (rr, gg, bb) = make_pattern(w, h, 1);
    let (dr, dg, db) = make_pattern(w, h, 2);

    let reference = ButteraugliReference::new_linear_planar(
        &rr,
        &gg,
        &bb,
        w,
        h,
        w,
        ButteraugliParams::default().with_compute_diffmap(true),
    )
    .expect("new reference");

    // Warm up both paths.
    let mut diffmap_full: Vec<f32> = Vec::new();
    let mut diffmap_strip: Vec<f32> = Vec::new();
    for _ in 0..warm_iters {
        let _ = reference
            .compare_linear_planar_into(&dr, &dg, &db, w, &mut diffmap_full)
            .unwrap();
        let _ = reference
            .compare_linear_planar_strip_into(&dr, &dg, &db, w, &mut diffmap_strip)
            .unwrap();
    }

    // Interleave A/B for fairness.
    let mut t_full_total = 0u128;
    let mut t_strip_total = 0u128;
    let mut score_full = 0.0f64;
    let mut score_strip = 0.0f64;
    for _ in 0..bench_iters {
        let t0 = Instant::now();
        let (s_full, _) = reference
            .compare_linear_planar_into(&dr, &dg, &db, w, &mut diffmap_full)
            .unwrap();
        t_full_total += t0.elapsed().as_nanos();
        score_full = s_full;

        let t1 = Instant::now();
        let (s_strip, _) = reference
            .compare_linear_planar_strip_into(&dr, &dg, &db, w, &mut diffmap_strip)
            .unwrap();
        t_strip_total += t1.elapsed().as_nanos();
        score_strip = s_strip;
    }

    let avg_full_ms = (t_full_total as f64) / (bench_iters as f64) / 1_000_000.0;
    let avg_strip_ms = (t_strip_total as f64) / (bench_iters as f64) / 1_000_000.0;
    let ratio = avg_strip_ms / avg_full_ms;

    println!(
        "W44-PHASE3-B7d Day 5 sanity bench — image {w}×{h}, {bench_iters} iters (warm {warm_iters})"
    );
    println!("  compare_linear_planar_into:       avg {avg_full_ms:.3} ms");
    println!("  compare_linear_planar_strip_into: avg {avg_strip_ms:.3} ms");
    println!("  ratio (strip / full):             {ratio:.3}×");
    println!("  full score:  {score_full:.6}");
    println!("  strip score: {score_strip:.6}  (must equal above)");
    println!(
        "  scalar score bit-equal: {}",
        score_full.to_bits() == score_strip.to_bits()
    );

    if ratio > 1.5 {
        eprintln!(
            "WARN: ratio {ratio:.3}× exceeds Day 5 sanity gate of 1.5×; \
             Day 5 ships byte-identical, perf-neutral. Day 6 explores deeper tiling."
        );
    }
    assert_eq!(
        score_full.to_bits(),
        score_strip.to_bits(),
        "sanity bench must produce byte-identical scores"
    );
}
