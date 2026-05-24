//! W44-PHASE3-B7d Day 6 per-stage attribution probe.
//!
//! Runs `compare_linear_planar_into` and `compare_linear_planar_strip_into`
//! at 1024×1024 with `JXL_B7D_STAGE_TIMING=1` enabled so the lib emits one
//! stage-timing line per stage per call. This binary uses 1 warmup + 8 iters
//! per mode, interleaved, and prints the raw timing output to stderr.
//!
//! Day 7: gated behind `strip-tile-butteraugli` (default OFF). Pipe stderr to
//! a file and post-process:
//!   cargo run --release --features strip-tile-butteraugli --example w44_phase3_b7d_day6_stage_probe 2>stages.tsv

#[cfg(not(feature = "strip-tile-butteraugli"))]
fn main() {
    eprintln!(
        "w44_phase3_b7d_day6_stage_probe: skipped — rebuild with --features strip-tile-butteraugli."
    );
}

#[cfg(feature = "strip-tile-butteraugli")]
use butteraugli::{ButteraugliParams, ButteraugliReference};

#[cfg(feature = "strip-tile-butteraugli")]
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
            r[i] = (0.5 + 0.30 * (xf * 0.013 + phase).sin() + 0.15 * (yf * 0.021 + phase).cos()
                + 0.05 * ((xf + yf) * 0.005).sin())
                .clamp(0.0, 1.0);
            g[i] = (0.5 + 0.25 * (xf * 0.007 + phase).cos() + 0.20 * (yf * 0.013 + phase).sin())
                .clamp(0.0, 1.0);
            b[i] = (0.5 + 0.35 * (yf * 0.003 + phase).cos() + 0.10 * (xf * 0.004 + phase).sin())
                .clamp(0.0, 1.0);
        }
    }
    (r, g, b)
}

#[cfg(feature = "strip-tile-butteraugli")]
fn perturb(r: &[f32], g: &[f32], b: &[f32], w: usize, h: usize, delta: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = w * h;
    let mut dr = vec![0.0f32; n];
    let mut dg = vec![0.0f32; n];
    let mut db = vec![0.0f32; n];
    for i in 0..n {
        let y = (i / w) as f32;
        let x = (i % w) as f32;
        let d = delta * ((x * 0.31 + y * 0.13).sin() + 0.7);
        dr[i] = (r[i] + d * 0.05).clamp(0.0, 1.0);
        dg[i] = (g[i] - d * 0.04).clamp(0.0, 1.0);
        db[i] = (b[i] + d * 0.03).clamp(0.0, 1.0);
    }
    (dr, dg, db)
}

#[cfg(feature = "strip-tile-butteraugli")]
fn main() {
    // FORCE stage timing on so this binary always emits per-stage data even
    // if the caller forgot the env var.
    // SAFETY: process-private env mutation prior to any threads spawning.
    unsafe { std::env::set_var("JXL_B7D_STAGE_TIMING", "1"); }
    let w = 1024;
    let h = 1024;
    let warm = 1;
    let iters = 8;

    let (rr, gg, bb) = make_smooth(w, h, 1);
    let (dr, dg, db) = perturb(&rr, &gg, &bb, w, h, 0.03);

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

    // Warm.
    for _ in 0..warm {
        eprintln!("# warm full");
        let _ = reference.compare_linear_planar_into(&dr, &dg, &db, w, &mut diffmap_full).unwrap();
        eprintln!("# warm strip");
        let _ = reference.compare_linear_planar_strip_into(&dr, &dg, &db, w, &mut diffmap_strip).unwrap();
    }

    for i in 0..iters {
        eprintln!("# iter {i} full");
        let _ = reference.compare_linear_planar_into(&dr, &dg, &db, w, &mut diffmap_full).unwrap();
        eprintln!("# iter {i} strip");
        let _ = reference.compare_linear_planar_strip_into(&dr, &dg, &db, w, &mut diffmap_strip).unwrap();
    }

    eprintln!("# done");
}
