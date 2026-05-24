//! W44-PHASE3-B7d Day 5 — 50-image byte-identical parity test for
//! `ButteraugliReference::compare_linear_planar_strip_into` vs
//! `ButteraugliReference::compare_linear_planar_into`.
//!
//! Synthesises 50 representative fixtures (mix of gradients, noise, and
//! sin-wave patterns at varied sizes spanning 64² → 1024²) and asserts that
//! the strip-tiled compare path produces BIT-IDENTICAL output for BOTH the
//! scalar score AND every pixel of the diffmap. If ANY pixel diverges by
//! 1 ULP the test fails — ULP-equivalent is NOT acceptable per the
//! `byte-identical-strict` Day 5 acceptance gate.
//!
//! The IIR blur backend uses a slightly different op-order that introduces
//! per-call FMA-rounding noise (see existing `iir-blur` gate in
//! `precompute::tests::test_precompute_matches_full_compute`); this test
//! gates the iir-blur backend out for the same reason.
//!
//! This test exercises ONLY the strip-tiled per-pixel fusion stage (Stage 3
//! in the Day 5 nomenclature). The upstream stages (opsin / separate_frequencies
//! / malta / mask) remain on the full-image path because their internal kernels
//! have halos > 0 with mirror-reflect boundaries that diverge under naive
//! strip tiling — Day 6+ explores that separately with measurable ULP
//! tolerance.

#![cfg(not(target_arch = "wasm32"))]
#![cfg(not(feature = "iir-blur"))]

use butteraugli::{ButteraugliParams, ButteraugliReference};

/// Fixture kinds — each covers a distinct content class so the 50 cells
/// exercise a representative span of butteraugli's per-pixel arithmetic.
#[derive(Clone, Copy)]
enum FixtureKind {
    /// Smooth horizontal gradient (low-frequency dominated; HF / UHF stay
    /// small).
    GradientH,
    /// Smooth vertical gradient.
    GradientV,
    /// Diagonal gradient.
    GradientD,
    /// Pseudo-random noise via a deterministic xorshift PRNG (HF / UHF
    /// dominate; exercises the malta + mask correction paths).
    Noise,
    /// Mixed sin-wave pattern (broad-spectrum; exercises all 4 bands +
    /// non-linear cascade transitions).
    SinMix,
    /// Sparse impulses on a flat background (delta functions in the spectral
    /// sense — every band non-zero).
    Impulses,
}

/// Deterministic PRNG (xorshift32) — keeps the test self-contained and
/// reproducible across runs.
fn xorshift32(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

/// Build a (width, height, kind, seed) → linear-f32 planar (R, G, B) tuple.
fn synthesize(width: usize, height: usize, kind: FixtureKind, seed: u32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = width * height;
    let mut r = vec![0.0f32; n];
    let mut g = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];
    let mut prng = seed.max(1);
    match kind {
        FixtureKind::GradientH => {
            for y in 0..height {
                for x in 0..width {
                    let t = x as f32 / (width.max(1) - 1).max(1) as f32;
                    r[y * width + x] = t;
                    g[y * width + x] = 1.0 - t;
                    b[y * width + x] = (t * 2.0 - 1.0).abs();
                }
            }
        }
        FixtureKind::GradientV => {
            for y in 0..height {
                let t = y as f32 / (height.max(1) - 1).max(1) as f32;
                for x in 0..width {
                    r[y * width + x] = t;
                    g[y * width + x] = (1.0 - t) * 0.7;
                    b[y * width + x] = (t * 2.0 - 1.0).abs() * 0.6;
                }
            }
        }
        FixtureKind::GradientD => {
            for y in 0..height {
                for x in 0..width {
                    let t = ((x + y) as f32) / ((width + height).max(1) - 1).max(1) as f32;
                    r[y * width + x] = t;
                    g[y * width + x] = (t * 3.14159).sin().abs();
                    b[y * width + x] = 1.0 - t;
                }
            }
        }
        FixtureKind::Noise => {
            for i in 0..n {
                r[i] = (xorshift32(&mut prng) as f32 / u32::MAX as f32).clamp(0.0, 1.0);
                g[i] = (xorshift32(&mut prng) as f32 / u32::MAX as f32).clamp(0.0, 1.0);
                b[i] = (xorshift32(&mut prng) as f32 / u32::MAX as f32).clamp(0.0, 1.0);
            }
        }
        FixtureKind::SinMix => {
            for y in 0..height {
                for x in 0..width {
                    let i = y * width + x;
                    let xf = x as f32;
                    let yf = y as f32;
                    r[i] = (0.5
                        + 0.3 * (xf * 0.137).sin()
                        + 0.15 * (yf * 0.211).cos()
                        + 0.05 * ((xf + yf) * 0.05).sin())
                        .clamp(0.0, 1.0);
                    g[i] = (0.5
                        + 0.25 * (xf * 0.07).cos()
                        + 0.2 * (yf * 0.13).sin()
                        + 0.05 * ((xf - yf) * 0.083).sin())
                        .clamp(0.0, 1.0);
                    b[i] = (0.5
                        + 0.35 * (yf * 0.027).cos()
                        + 0.1 * (xf * 0.041).sin()
                        + 0.05 * ((xf * yf) as f32 * 0.0001).sin())
                        .clamp(0.0, 1.0);
                }
            }
        }
        FixtureKind::Impulses => {
            // Flat background with sparse impulses.
            for i in 0..n {
                r[i] = 0.5;
                g[i] = 0.5;
                b[i] = 0.5;
            }
            let stride = (n / 64).max(1);
            for i in (0..n).step_by(stride) {
                let v = (xorshift32(&mut prng) as f32 / u32::MAX as f32).clamp(0.0, 1.0);
                r[i] = v;
                g[i] = 1.0 - v;
                b[i] = (v * 2.0).fract();
            }
        }
    }
    (r, g, b)
}

/// Apply a small perceptible distortion to a planar (R, G, B) tuple,
/// returning the distorted version.
fn distort(r: &[f32], g: &[f32], b: &[f32], seed: u32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut prng = seed.max(1);
    let n = r.len();
    let mut dr = vec![0.0f32; n];
    let mut dg = vec![0.0f32; n];
    let mut db = vec![0.0f32; n];
    for i in 0..n {
        let noise =
            ((xorshift32(&mut prng) as f32 / u32::MAX as f32) - 0.5) * 0.07;
        dr[i] = (r[i] + noise).clamp(0.0, 1.0);
        let noise =
            ((xorshift32(&mut prng) as f32 / u32::MAX as f32) - 0.5) * 0.05;
        dg[i] = (g[i] + noise).clamp(0.0, 1.0);
        let noise =
            ((xorshift32(&mut prng) as f32 / u32::MAX as f32) - 0.5) * 0.06;
        db[i] = (b[i] + noise).clamp(0.0, 1.0);
    }
    (dr, dg, db)
}

fn fixture_kind(idx: usize) -> FixtureKind {
    match idx % 6 {
        0 => FixtureKind::GradientH,
        1 => FixtureKind::GradientV,
        2 => FixtureKind::GradientD,
        3 => FixtureKind::Noise,
        4 => FixtureKind::SinMix,
        _ => FixtureKind::Impulses,
    }
}

/// Return 50 (width, height) cells spanning 64² → 1024². Distribution:
/// - 12 cells at 64-96 (≤ MIN_SIZE_FOR_SUBSAMPLE boundary cases)
/// - 12 cells at 128-256
/// - 16 cells at 384-768
/// - 10 cells at 800-1024
fn cells_50() -> Vec<(usize, usize)> {
    vec![
        (64, 64),
        (72, 64),
        (64, 80),
        (80, 80),
        (96, 64),
        (64, 96),
        (96, 96),
        (95, 87),    // odd dims
        (88, 73),    // odd dims
        (96, 73),
        (84, 96),
        (90, 90),
        (128, 128),
        (160, 128),
        (128, 192),
        (192, 192),
        (200, 200),
        (200, 144),
        (144, 200),
        (256, 256),
        (227, 215),  // odd
        (231, 187),  // odd
        (192, 256),
        (256, 192),
        (384, 384),
        (400, 300),
        (300, 400),
        (512, 256),
        (256, 512),
        (512, 384),
        (384, 512),
        (511, 257),  // odd
        (513, 511),  // odd
        (512, 512),
        (640, 480),
        (480, 640),
        (768, 432),
        (432, 768),
        (640, 640),
        (768, 768),
        (800, 600),
        (1024, 512),
        (512, 1024),
        (1024, 768),
        (768, 1024),
        (959, 717),  // odd
        (1023, 1023),// odd
        (1024, 1024),
        (1024, 700),
        (820, 1020),
    ]
}

#[test]
fn strip_parity_50_images_byte_identical() {
    let cells = cells_50();
    assert_eq!(cells.len(), 50, "test should cover exactly 50 cells");

    let mut failures: Vec<String> = Vec::new();
    for (idx, (w, h)) in cells.into_iter().enumerate() {
        let kind = fixture_kind(idx);
        let ref_seed = (idx as u32) * 13 + 1;
        let dist_seed = (idx as u32) * 19 + 7;
        let (rr, gg, bb) = synthesize(w, h, kind, ref_seed);
        let (dr, dg, db) = distort(&rr, &gg, &bb, dist_seed);

        let reference = match ButteraugliReference::new_linear_planar(
            &rr,
            &gg,
            &bb,
            w,
            h,
            w,
            ButteraugliParams::default().with_compute_diffmap(true),
        ) {
            Ok(r) => r,
            Err(e) => {
                failures.push(format!(
                    "cell {idx} ({w}×{h}): new_linear_planar failed: {e:?}"
                ));
                continue;
            }
        };

        let mut diffmap_full: Vec<f32> = Vec::new();
        let (score_full, pnorm_full) =
            match reference.compare_linear_planar_into(&dr, &dg, &db, w, &mut diffmap_full) {
                Ok(r) => r,
                Err(e) => {
                    failures.push(format!(
                        "cell {idx} ({w}×{h}): compare_linear_planar_into failed: {e:?}"
                    ));
                    continue;
                }
            };

        let mut diffmap_strip: Vec<f32> = Vec::new();
        let (score_strip, pnorm_strip) = match reference
            .compare_linear_planar_strip_into(&dr, &dg, &db, w, &mut diffmap_strip)
        {
            Ok(r) => r,
            Err(e) => {
                failures.push(format!(
                    "cell {idx} ({w}×{h}): compare_linear_planar_strip_into failed: {e:?}"
                ));
                continue;
            }
        };

        // Byte-identical scalar score (f64 bit-equal).
        if score_full.to_bits() != score_strip.to_bits() {
            failures.push(format!(
                "cell {idx} ({w}×{h} kind {kind:?}): scalar score divergence — \
                 full={score_full} strip={score_strip} bits_full={:#018x} bits_strip={:#018x}",
                score_full.to_bits(),
                score_strip.to_bits(),
                kind = kind as u32,
            ));
            continue;
        }

        if pnorm_full.to_bits() != pnorm_strip.to_bits() {
            failures.push(format!(
                "cell {idx} ({w}×{h}): pnorm_3 divergence — \
                 full={pnorm_full} strip={pnorm_strip}"
            ));
            continue;
        }

        // Byte-identical per-pixel diffmap (f32 bit-equal on every pixel).
        if diffmap_full.len() != diffmap_strip.len() {
            failures.push(format!(
                "cell {idx} ({w}×{h}): diffmap length mismatch full={} strip={}",
                diffmap_full.len(),
                diffmap_strip.len()
            ));
            continue;
        }
        for (pix_idx, (&a, &c)) in diffmap_full.iter().zip(diffmap_strip.iter()).enumerate() {
            if a.to_bits() != c.to_bits() {
                let py = pix_idx / w;
                let px = pix_idx % w;
                failures.push(format!(
                    "cell {idx} ({w}×{h}): pixel ({px},{py}) divergence — \
                     full={a} ({:#010x}) strip={c} ({:#010x})",
                    a.to_bits(),
                    c.to_bits()
                ));
                break;
            }
        }
    }

    if !failures.is_empty() {
        panic!(
            "{} of 50 cells diverged byte-identical:\n  {}",
            failures.len(),
            failures.join("\n  ")
        );
    }
}
