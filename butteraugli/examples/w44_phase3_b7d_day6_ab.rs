//! W44-PHASE3-B7d Day 6 paired A/B bench.
//!
//! Honest measurement of `compare_linear_planar_strip_into` vs
//! `compare_linear_planar_into` across:
//!   - 4 sizes: 256², 512², 1024², 2048²
//!   - 2 fixtures per size: "smooth" (low-freq sin/cos field) and "noisy"
//!     (high-freq pseudo-random)
//!
//! 8 cells × 2 modes × N iterations (warm + bench). Interleaved per iter to
//! minimise CPU-frequency / thermal bias. Median + p25/p75 reported.
//!
//! Outputs a TSV to stdout. Optionally, with `JXL_B7D_STAGE_TIMING=1` in env,
//! the lib also emits per-stage timing lines on stderr (one line per stage
//! per call); parse those after the run to get Phase 2 attribution.
//!
//! Day 7: gated behind `strip-tile-butteraugli` (default OFF). Run:
//!   cargo run --release --features strip-tile-butteraugli --example w44_phase3_b7d_day6_ab
//!   JXL_B7D_STAGE_TIMING=1 cargo run --release --features strip-tile-butteraugli --example w44_phase3_b7d_day6_ab 2>stage.log

fn main() {
    #[cfg(not(feature = "strip-tile-butteraugli"))]
    {
        eprintln!(
            "w44_phase3_b7d_day6_ab: skipped — rebuild with --features strip-tile-butteraugli."
        );
    }
    #[cfg(feature = "strip-tile-butteraugli")]
    {
        strip_bench::run();
    }
}

#[cfg(feature = "strip-tile-butteraugli")]
mod strip_bench {

use butteraugli::{ButteraugliParams, ButteraugliReference};
use std::time::Instant;

fn make_smooth(w: usize, h: usize, seed: u32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = w * h;
    let mut r = vec![0.0f32; n];
    let mut g = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];
    let phase = seed as f32 * 0.1;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let xf = x as f32;
            let yf = y as f32;
            r[i] = (0.5
                + 0.30 * (xf * 0.013 + phase).sin()
                + 0.15 * (yf * 0.021 + phase).cos()
                + 0.05 * ((xf + yf) * 0.005).sin())
                .clamp(0.0, 1.0);
            g[i] = (0.5
                + 0.25 * (xf * 0.007 + phase).cos()
                + 0.20 * (yf * 0.013 + phase).sin())
                .clamp(0.0, 1.0);
            b[i] = (0.5
                + 0.35 * (yf * 0.003 + phase).cos()
                + 0.10 * (xf * 0.004 + phase).sin())
                .clamp(0.0, 1.0);
        }
    }
    (r, g, b)
}

fn make_noisy(w: usize, h: usize, seed: u32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = w * h;
    let mut r = vec![0.0f32; n];
    let mut g = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];
    let mut state = seed.max(1) as u64;
    // High-frequency pseudo-noise via LCG. Same shape as content with sharp
    // edges (e.g. text rasters, screenshots) that exercise malta + EHF kernels.
    for i in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let rb = ((state >> 24) & 0xFF) as f32 / 255.0;
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let gb = ((state >> 24) & 0xFF) as f32 / 255.0;
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let bb = ((state >> 24) & 0xFF) as f32 / 255.0;
        // Mix some low-frequency structure so it's not pure noise (which is
        // unrealistic — real images have both).
        let y = (i / w) as f32;
        let x = (i % w) as f32;
        let lf = 0.3 * ((x + y) * 0.005).sin();
        r[i] = (rb * 0.7 + lf + 0.15).clamp(0.0, 1.0);
        g[i] = (gb * 0.7 + lf + 0.15).clamp(0.0, 1.0);
        b[i] = (bb * 0.7 + lf + 0.15).clamp(0.0, 1.0);
    }
    (r, g, b)
}

fn perturb_distorted(
    r: &[f32],
    g: &[f32],
    b: &[f32],
    w: usize,
    h: usize,
    delta: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = w * h;
    let mut dr = vec![0.0f32; n];
    let mut dg = vec![0.0f32; n];
    let mut db = vec![0.0f32; n];
    for i in 0..n {
        // Add a low-amplitude sinusoidal perturbation to make the distorted
        // differ from the reference. This is just to produce non-zero scores;
        // the actual numeric content doesn't affect what we're measuring (wall
        // ratio).
        let y = (i / w) as f32;
        let x = (i % w) as f32;
        let d = delta * ((x * 0.31 + y * 0.13).sin() + 0.7);
        dr[i] = (r[i] + d * 0.05).clamp(0.0, 1.0);
        dg[i] = (g[i] - d * 0.04).clamp(0.0, 1.0);
        db[i] = (b[i] + d * 0.03).clamp(0.0, 1.0);
    }
    (dr, dg, db)
}

fn percentile(sorted: &[u128], pct: f64) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * pct).round() as usize;
    sorted[idx]
}

fn bench_cell(
    label: &str,
    w: usize,
    h: usize,
    fixture: &str,
    warm: usize,
    iters: usize,
) -> (
    f64, // strip / full ratio (median)
    u128, // full median ns
    u128, // strip median ns
    u128, // full p25 ns
    u128, // full p75 ns
    u128, // strip p25 ns
    u128, // strip p75 ns
    f64, // score full
    f64, // score strip
) {
    let (rr, gg, bb) = match fixture {
        "smooth" => make_smooth(w, h, 1),
        "noisy" => make_noisy(w, h, 1),
        _ => panic!("unknown fixture {fixture}"),
    };
    let (dr, dg, db) = perturb_distorted(&rr, &gg, &bb, w, h, 0.03);

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

    let mut diffmap_full: Vec<f32> = Vec::new();
    let mut diffmap_strip: Vec<f32> = Vec::new();

    // Warm both paths.
    for _ in 0..warm {
        let _ = reference
            .compare_linear_planar_into(&dr, &dg, &db, w, &mut diffmap_full)
            .unwrap();
        let _ = reference
            .compare_linear_planar_strip_into(&dr, &dg, &db, w, &mut diffmap_strip)
            .unwrap();
    }

    // Interleaved A/B/A/B/... per iter.
    let mut full_times = Vec::with_capacity(iters);
    let mut strip_times = Vec::with_capacity(iters);
    let mut last_full_score = 0.0f64;
    let mut last_strip_score = 0.0f64;
    for _ in 0..iters {
        let t0 = Instant::now();
        let (s_full, _) = reference
            .compare_linear_planar_into(&dr, &dg, &db, w, &mut diffmap_full)
            .unwrap();
        full_times.push(t0.elapsed().as_nanos());
        last_full_score = s_full;

        let t1 = Instant::now();
        let (s_strip, _) = reference
            .compare_linear_planar_strip_into(&dr, &dg, &db, w, &mut diffmap_strip)
            .unwrap();
        strip_times.push(t1.elapsed().as_nanos());
        last_strip_score = s_strip;
    }
    full_times.sort_unstable();
    strip_times.sort_unstable();
    let full_med = percentile(&full_times, 0.5);
    let strip_med = percentile(&strip_times, 0.5);
    let full_p25 = percentile(&full_times, 0.25);
    let full_p75 = percentile(&full_times, 0.75);
    let strip_p25 = percentile(&strip_times, 0.25);
    let strip_p75 = percentile(&strip_times, 0.75);
    let ratio = strip_med as f64 / full_med as f64;
    println!(
        "B7D_DAY6\t{label}\t{w}x{h}\t{fixture}\tfull_ms={:.3}\tstrip_ms={:.3}\tratio={:.3}",
        full_med as f64 / 1_000_000.0,
        strip_med as f64 / 1_000_000.0,
        ratio,
    );
    assert_eq!(
        last_full_score.to_bits(),
        last_strip_score.to_bits(),
        "Day 5 invariant: strip score must bit-match full",
    );
    (
        ratio,
        full_med,
        strip_med,
        full_p25,
        full_p75,
        strip_p25,
        strip_p75,
        last_full_score,
        last_strip_score,
    )
}

pub fn run() {
    let cells = [
        // (label, w, h, fixture, warm, iters)
        ("256_smooth", 256, 256, "smooth", 5, 30),
        ("256_noisy", 256, 256, "noisy", 5, 30),
        ("512_smooth", 512, 512, "smooth", 4, 20),
        ("512_noisy", 512, 512, "noisy", 4, 20),
        ("1024_smooth", 1024, 1024, "smooth", 3, 12),
        ("1024_noisy", 1024, 1024, "noisy", 3, 12),
        ("2048_smooth", 2048, 2048, "smooth", 2, 6),
        ("2048_noisy", 2048, 2048, "noisy", 2, 6),
    ];

    // Header for TSV digestion.
    println!(
        "#cell\twidth\theight\tfixture\tfull_med_ns\tstrip_med_ns\tfull_p25_ns\tfull_p75_ns\tstrip_p25_ns\tstrip_p75_ns\tratio_med\tscore_full\tscore_strip"
    );

    for (label, w, h, fixture, warm, iters) in cells {
        let (
            ratio,
            full_med,
            strip_med,
            full_p25,
            full_p75,
            strip_p25,
            strip_p75,
            sf,
            ss,
        ) = bench_cell(label, w, h, fixture, warm, iters);
        println!(
            "{label}\t{w}\t{h}\t{fixture}\t{full_med}\t{strip_med}\t{full_p25}\t{full_p75}\t{strip_p25}\t{strip_p75}\t{ratio:.4}\t{sf:.6}\t{ss:.6}",
        );
    }
}

} // mod strip_bench
