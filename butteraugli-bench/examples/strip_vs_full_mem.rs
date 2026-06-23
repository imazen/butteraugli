//! Strip-wise vs full-image butteraugli: measured memory-vs-speed.
//!
//! Runs ONE comparison of two procedurally-generated linear-RGB f32 images
//! (worst-case high-entropy content) either full-image or strip-wise, and
//! prints the score + wall time. Run each mode under `/usr/bin/time -v` to
//! capture the per-process max RSS — that is the memory axis. This isolates
//! the strip APPROACH's peak-memory vs speed tradeoff; the warm-reference
//! buttloop (precompute once, compare N times) amortizes the full path's
//! precompute across iters, so its strip cost is higher than this one-shot
//! lower bound.
//!
//! Usage: strip_vs_full_mem <full|strip> <megapixels> [strip_height=256]

use std::time::Instant;

use butteraugli::{ButteraugliParams, ImgVec};
use rgb::RGB;

fn gen_img(w: usize, h: usize, seed: u32) -> ImgVec<RGB<f32>> {
    let mut v = Vec::with_capacity(w * h);
    let mut s = seed.wrapping_mul(2_246_822_519).wrapping_add(1);
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        (s >> 8) as f32 / (1u32 << 24) as f32 // [0,1)
    };
    for _ in 0..w * h {
        v.push(RGB::new(next(), next(), next()));
    }
    ImgVec::new(v, w, h)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mode = a.get(1).map(String::as_str).unwrap_or("full");
    let mp: f64 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(16.0);
    let strip_height: u32 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(256);

    // Square-ish dims for the requested megapixels.
    let side = ((mp * 1_000_000.0).sqrt()).round() as usize;
    let (w, h) = (side, side);

    let img1 = gen_img(w, h, 1);
    // Distorted = ref with a deterministic small perturbation so the metric
    // does real work (non-zero diffmap).
    let img2 = gen_img(w, h, 2);

    let params = ButteraugliParams::new().with_compute_diffmap(true);

    eprintln!(
        "strip_vs_full_mem: mode={mode} {w}x{h} ({:.1} MP) strip_height={strip_height}",
        (w * h) as f64 / 1e6
    );
    let t0 = Instant::now();
    let result = match mode {
        "strip" => butteraugli::butteraugli_linear_strip(
            img1.as_ref(),
            img2.as_ref(),
            &params,
            strip_height,
        ),
        _ => butteraugli::butteraugli_linear(img1.as_ref(), img2.as_ref(), &params),
    };
    let wall = t0.elapsed().as_secs_f64() * 1000.0;
    let score = result.map(|r| r.score).unwrap_or(f64::NAN);
    println!(
        "mode={mode}\tw={w}\th={h}\tmp={:.1}\tstrip_height={strip_height}\tscore={score:.6}\twall_ms={wall:.1}",
        (w * h) as f64 / 1e6
    );
}
