//! Print Rust vs C++ butteraugli scores on synthetic test inputs.
//! Use to confirm a libjxl patch hasn't broken score parity.

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

fn make_test_planes(w: usize, h: usize, seed: u32) -> ([Vec<f32>; 3], [Vec<f32>; 3]) {
    let n = w * h;
    let mut src = [Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n)];
    let mut dst = [Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n)];
    let mut s = seed;
    for i in 0..n {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        let r = ((i % w) * 255 / w) as u8;
        let g = ((i / w) * 255 / h) as u8;
        let b = r.wrapping_add(g);
        let r2 = r.saturating_add((s & 7) as u8);
        let g2 = g.saturating_add(((s >> 4) & 5) as u8);
        let b2 = b.saturating_add(((s >> 8) & 3) as u8);
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
    // Small-image cases (8 <= dim < 32) probe the V-border overlap region
    // (ysize < kernel size = 2*offset+1). butteraugli's largest kernel is
    // size 33 (offset 16), so anything with a dim < 32 stresses that path
    // — matching what the in-tree ButteraugliBlurEquivalence test covers
    // but at the score level, so future cross-implementation drift here
    // won't go silent. Asymmetric shapes flush both H and V border code.
    //
    // libjxl short-circuits images smaller than 8×8 in DiffmapPsychoImage,
    // so 8×8 is the smallest meaningful case.
    let cases: &[(usize, usize, u32)] = &[
        (8, 8, 11),
        (12, 12, 22),
        (16, 24, 33),
        (24, 16, 44),
        (31, 31, 55),
        (256, 256, 42),
        (512, 512, 123),
        (1280, 720, 7),
        (1920, 1080, 99),
    ];

    println!(
        "{:>10} {:>14} {:>14} {:>14} {:>10}",
        "size", "rust", "cpp", "abs_diff", "rel%"
    );

    let mut max_rel = 0.0_f64;
    for &(w, h, seed) in cases {
        let (src, dst) = make_test_planes(w, h, seed);

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
                src[0].as_ptr(), src[1].as_ptr(), src[2].as_ptr(),
                dst[0].as_ptr(), dst[1].as_ptr(), dst[2].as_ptr(),
                w, h,
            )
        };
        #[cfg(not(has_cpp_butteraugli))]
        let cpp_score = f64::NAN;

        let abs = (rust_score as f64 - cpp_score).abs();
        let rel = abs / cpp_score.abs() * 100.0;
        max_rel = max_rel.max(rel);
        println!(
            "{:>10} {:>14.6} {:>14.6} {:>14.2e} {:>9.4}%",
            format!("{}x{}", w, h),
            rust_score,
            cpp_score,
            abs,
            rel
        );
    }
    println!();
    println!("max relative diff: {:.4}%", max_rel);
}
