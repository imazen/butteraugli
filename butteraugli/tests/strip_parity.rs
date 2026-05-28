//! Strip-vs-full parity tests for butteraugli.
//!
//! Verifies that `butteraugli_strip`, `butteraugli_linear_strip`, and
//! `ButteraugliReference::compare_strip` produce scores within atomic
//! tolerance of the corresponding full-image paths. Butteraugli's
//! blurs are FIR (finite impulse response), so with sufficient halo
//! the strip blur output is bit-identical to the full-image blur
//! inside the strip's interior region. Score parity follows from
//! exact diffmap parity at the interior pixels.

use butteraugli::{
    ButteraugliParams, ButteraugliReference, Img, ImgRef, RGB, RGB8, butteraugli,
    butteraugli_linear, butteraugli_linear_strip, butteraugli_strip,
};

fn gradient_rgb8(width: usize, height: usize, seed: u32) -> Vec<RGB8> {
    let mut data = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let r = ((x as u32)
                .wrapping_mul(7)
                .wrapping_add((y as u32).wrapping_mul(13))
                .wrapping_add(seed))
                & 0xff;
            let g = ((x as u32)
                .wrapping_mul(11)
                .wrapping_add((y as u32).wrapping_mul(3))
                .wrapping_add(seed.wrapping_add(50)))
                & 0xff;
            let b = ((x as u32)
                .wrapping_mul(5)
                .wrapping_add((y as u32).wrapping_mul(17))
                .wrapping_add(seed.wrapping_add(100)))
                & 0xff;
            data.push(RGB8::new(r as u8, g as u8, b as u8));
        }
    }
    data
}

fn gradient_linear(width: usize, height: usize, seed: u32) -> Vec<RGB<f32>> {
    gradient_rgb8(width, height, seed)
        .into_iter()
        .map(|p| RGB::new(p.r as f32 / 255.0, p.g as f32 / 255.0, p.b as f32 / 255.0))
        .collect()
}

const SCORE_TOLERANCE: f64 = 0.01;

#[test]
fn strip_parity_identical_64x64() {
    let pixels = gradient_rgb8(64, 64, 0);
    let img = Img::new(pixels, 64, 64);
    let params = ButteraugliParams::default();
    let full = butteraugli(img.as_ref(), img.as_ref(), &params).unwrap();
    let strip = butteraugli_strip(img.as_ref(), img.as_ref(), &params, 16).unwrap();
    assert!(
        (full.score - strip.score).abs() < 0.001,
        "identical 64x64 strip {:.6} vs full {:.6}",
        strip.score,
        full.score
    );
    // Identical images: both must be ~0
    assert!(full.score < 0.001);
    assert!(strip.score < 0.001);
}

#[test]
fn strip_parity_different_64x64() {
    let source = Img::new(gradient_rgb8(64, 64, 0), 64, 64);
    let distorted = Img::new(gradient_rgb8(64, 64, 1), 64, 64);
    let params = ButteraugliParams::default();
    let full = butteraugli(source.as_ref(), distorted.as_ref(), &params).unwrap();
    let strip = butteraugli_strip(source.as_ref(), distorted.as_ref(), &params, 16).unwrap();
    assert!(
        (full.score - strip.score).abs() < SCORE_TOLERANCE,
        "64x64 strip score {:.6} vs full {:.6} differs by {:.6}",
        strip.score,
        full.score,
        (full.score - strip.score).abs()
    );
}

#[test]
fn strip_parity_512x512_multiresolution() {
    let source = Img::new(gradient_rgb8(512, 512, 7), 512, 512);
    let distorted = Img::new(gradient_rgb8(512, 512, 8), 512, 512);
    let params = ButteraugliParams::default();
    let full = butteraugli(source.as_ref(), distorted.as_ref(), &params).unwrap();
    for strip_h in [64u32, 128, 256] {
        let strip =
            butteraugli_strip(source.as_ref(), distorted.as_ref(), &params, strip_h).unwrap();
        assert!(
            (full.score - strip.score).abs() < SCORE_TOLERANCE,
            "512x512 strip_h={strip_h} score {:.6} vs full {:.6}",
            strip.score,
            full.score
        );
        assert!(
            (full.pnorm_3 - strip.pnorm_3).abs() < SCORE_TOLERANCE,
            "512x512 strip_h={strip_h} pnorm {:.6} vs full {:.6}",
            strip.pnorm_3,
            full.pnorm_3
        );
    }
}

#[test]
fn strip_parity_1024x1024() {
    let source = Img::new(gradient_rgb8(1024, 1024, 11), 1024, 1024);
    let distorted = Img::new(gradient_rgb8(1024, 1024, 12), 1024, 1024);
    let params = ButteraugliParams::default();
    let full = butteraugli(source.as_ref(), distorted.as_ref(), &params).unwrap();
    let strip = butteraugli_strip(source.as_ref(), distorted.as_ref(), &params, 128).unwrap();
    assert!(
        (full.score - strip.score).abs() < SCORE_TOLERANCE,
        "1024x1024 strip {:.6} vs full {:.6}",
        strip.score,
        full.score
    );
}

#[test]
fn strip_linear_parity_512x512() {
    let source = Img::new(gradient_linear(512, 512, 17), 512, 512);
    let distorted = Img::new(gradient_linear(512, 512, 18), 512, 512);
    let params = ButteraugliParams::default();
    let full = butteraugli_linear(source.as_ref(), distorted.as_ref(), &params).unwrap();
    let strip =
        butteraugli_linear_strip(source.as_ref(), distorted.as_ref(), &params, 128).unwrap();
    assert!(
        (full.score - strip.score).abs() < SCORE_TOLERANCE,
        "512x512 linear strip {:.6} vs full {:.6}",
        strip.score,
        full.score
    );
}

#[test]
fn warm_ref_strip_parity_512x512() {
    let width = 512;
    let height = 512;
    let pixels1 = gradient_rgb8(width, height, 21);
    let pixels2 = gradient_rgb8(width, height, 22);
    let img1 = Img::new(pixels1.clone(), width, height);
    let img2 = Img::new(pixels2.clone(), width, height);
    let params = ButteraugliParams::default();
    let full = butteraugli(img1.as_ref(), img2.as_ref(), &params).unwrap();
    let rgb1_bytes: Vec<u8> = pixels1.iter().flat_map(|p| [p.r, p.g, p.b]).collect();
    let rgb2_bytes: Vec<u8> = pixels2.iter().flat_map(|p| [p.r, p.g, p.b]).collect();
    let reference =
        ButteraugliReference::new(&rgb1_bytes, width, height, params.clone()).expect("reference");
    let strip = reference.compare_strip(&rgb2_bytes, 128).unwrap();
    assert!(
        (full.score - strip.score).abs() < SCORE_TOLERANCE,
        "compare_strip {:.6} vs full {:.6}",
        strip.score,
        full.score
    );
}

#[test]
fn warm_ref_strip_matches_compare() {
    let width = 256;
    let height = 256;
    let pixels: Vec<RGB8> = gradient_rgb8(width, height, 99);
    let rgb_bytes: Vec<u8> = pixels.iter().flat_map(|p| [p.r, p.g, p.b]).collect();
    let params = ButteraugliParams::default();
    let reference =
        ButteraugliReference::new(&rgb_bytes, width, height, params.clone()).expect("reference");
    let compare = reference.compare(&rgb_bytes).unwrap();
    let strip = reference.compare_strip(&rgb_bytes, 64).unwrap();
    // Identical inputs round-trip to ~0 in both paths.
    assert!(compare.score < 0.001);
    assert!(strip.score < 0.001);
    assert!(
        (compare.score - strip.score).abs() < 0.001,
        "compare {:.6} vs compare_strip {:.6}",
        compare.score,
        strip.score
    );
}

#[test]
fn strip_height_below_minimum_errors() {
    let img = Img::new(gradient_rgb8(64, 64, 0), 64, 64);
    let params = ButteraugliParams::default();
    let err = butteraugli_strip(img.as_ref(), img.as_ref(), &params, 4)
        .expect_err("strip_height=4 < MIN must error");
    let msg = format!("{err}");
    assert!(msg.contains("too small") || msg.contains("size"));
}

#[test]
fn strip_mismatched_dimensions_errors() {
    let a = Img::new(gradient_rgb8(64, 64, 0), 64, 64);
    let b = Img::new(gradient_rgb8(32, 32, 0), 32, 32);
    let params = ButteraugliParams::default();
    assert!(butteraugli_strip(a.as_ref(), b.as_ref(), &params, 16).is_err());
}

#[test]
fn strip_diffmap_output() {
    // When compute_diffmap is enabled, the strip path should
    // assemble a full-image diffmap from the per-strip diffmaps.
    let width = 128;
    let height = 128;
    let source = Img::new(gradient_rgb8(width, height, 31), width, height);
    let distorted = Img::new(gradient_rgb8(width, height, 32), width, height);
    let params = ButteraugliParams::default().with_compute_diffmap(true);
    let strip = butteraugli_strip(source.as_ref(), distorted.as_ref(), &params, 32).unwrap();
    let dm = strip.diffmap.expect("diffmap requested");
    assert_eq!(dm.width(), width);
    assert_eq!(dm.height(), height);
    // diffmap_out's max-norm should equal `score`.
    let dm_max = dm.buf().iter().copied().fold(0.0_f32, f32::max);
    assert!((dm_max as f64 - strip.score).abs() < 1e-6);
}

#[test]
fn butteraugli_strip_imgref_passthrough() {
    // butteraugli_strip should accept ImgRef just like butteraugli.
    let pixels: Vec<RGB8> = gradient_rgb8(48, 48, 5);
    let img: ImgRef<'_, RGB8> = ImgRef::new(&pixels, 48, 48);
    let params = ButteraugliParams::default();
    let result = butteraugli_strip(img, img, &params, 16).unwrap();
    assert!(result.score < 0.001);
}
