//! Cross-architecture parity test for butteraugli SIMD implementations.
//!
//! Verifies that butteraugli produces identical scores on all architectures by
//! comparing against reference scores captured on x86_64 (AVX2). Each score is
//! stored as the raw u64 bit pattern of an f64, enabling bit-exact verification.
//!
//! Run with: `cargo test --test cross_arch_parity`
//! Cross-test: `cross test -p butteraugli --test cross_arch_parity --target aarch64-unknown-linux-gnu`

mod common;

use butteraugli::{ButteraugliParams, Img, butteraugli};
use common::generators::{generate_image_pair, parse_dimensions, rgb_bytes_to_pixels};

/// Minimum dimension requirement for butteraugli
const MIN_DIMENSION: usize = 8;

/// Maximum allowed relative difference between architectures.
///
/// FMA rounding differences between x86 AVX2 and ARM NEON can produce
/// tiny variations (~7e-6 in f32 intermediates). After accumulation into
/// the final f64 score, we allow up to 0.01% (1e-4) relative difference.
/// Low-scoring images (quantize distortions, ~0.4 score) amplify tiny
/// absolute differences into ~5e-5 relative, so 1e-4 provides headroom.
/// Real algorithm bugs would show >1% difference.
const MAX_RELATIVE_DIFF: f64 = 1e-4;

/// Reference scores captured on x86_64 with AVX2, stored as f64 bit patterns.
/// Each entry is (test_case_name, f64::to_bits() value).
const X86_REFERENCE_SCORES: &[(&str, u64)] = &[
    ("uniform_gray_128_shift_10_8x8", 0x402F88E5E0000000),
    ("uniform_gray_128_shift_50_8x8", 0x4052A74100000000),
    ("uniform_gray_128_shift_10_9x9", 0x402F88E1C0000000),
    ("uniform_gray_128_shift_50_9x9", 0x4052A742E0000000),
    ("uniform_gray_128_shift_10_15x15", 0x4035492E80000000),
    ("uniform_gray_128_shift_50_15x15", 0x40592E95C0000000),
    ("uniform_gray_128_shift_10_16x16", 0x4035493920000000),
    ("uniform_gray_128_shift_50_16x16", 0x40592E9F80000000),
    ("uniform_gray_128_shift_10_17x17", 0x40354928C0000000),
    ("uniform_gray_128_shift_50_17x17", 0x40592E9400000000),
    ("uniform_gray_128_shift_10_23x23", 0x4035492780000000),
    ("uniform_gray_128_shift_50_23x23", 0x40592E8EC0000000),
    ("uniform_gray_128_shift_10_24x24", 0x4035492340000000),
    ("uniform_gray_128_shift_50_24x24", 0x40592E88A0000000),
    ("uniform_gray_128_shift_10_31x31", 0x4035492E80000000),
    ("uniform_gray_128_shift_50_31x31", 0x40592E9340000000),
    ("uniform_gray_128_shift_10_32x32", 0x4035492E40000000),
    ("uniform_gray_128_shift_50_32x32", 0x40592E9500000000),
    ("uniform_red_shift_20_16x16", 0x403F24C500000000),
    ("uniform_green_shift_20_16x16", 0x4044F409E0000000),
    ("uniform_blue_shift_20_16x16", 0x4028B48A40000000),
    ("uniform_red_shift_20_23x23", 0x403F24C060000000),
    ("uniform_green_shift_20_23x23", 0x4044F3F5A0000000),
    ("uniform_blue_shift_20_23x23", 0x4028B488C0000000),
    ("uniform_red_shift_20_32x32", 0x403F24BD80000000),
    ("uniform_green_shift_20_32x32", 0x4044F3F700000000),
    ("uniform_blue_shift_20_32x32", 0x4028B48640000000),
    ("gradient_h_shift_15_8x8", 0x4016ADA460000000),
    ("gradient_v_shift_15_8x8", 0x4016ADA360000000),
    ("gradient_h_shift_15_9x9", 0x401746AAE0000000),
    ("gradient_v_shift_15_9x9", 0x401746AC60000000),
    ("gradient_h_shift_15_15x15", 0x4021790A00000000),
    ("gradient_v_shift_15_15x15", 0x4021790BC0000000),
    ("gradient_h_shift_15_16x16", 0x40223EBDA0000000),
    ("gradient_v_shift_15_16x16", 0x40223EBB40000000),
    ("gradient_h_shift_15_17x17", 0x4023023D00000000),
    ("gradient_v_shift_15_17x17", 0x4023023C40000000),
    ("gradient_h_shift_15_23x23", 0x40270DA020000000),
    ("gradient_v_shift_15_23x23", 0x40270DA000000000),
    ("gradient_h_shift_15_24x24", 0x402874DA40000000),
    ("gradient_v_shift_15_24x24", 0x402874DC20000000),
    ("gradient_h_shift_15_31x31", 0x402981BD40000000),
    ("gradient_v_shift_15_31x31", 0x402981BB00000000),
    ("gradient_h_shift_15_32x32", 0x402AC30740000000),
    ("gradient_v_shift_15_32x32", 0x402AC30060000000),
    ("gradient_h_shift_15_33x33", 0x402EC5B8A0000000),
    ("gradient_v_shift_15_33x33", 0x402EC5BA40000000),
    ("gradient_h_shift_15_47x47", 0x402F1365C0000000),
    ("gradient_v_shift_15_47x47", 0x402F136D00000000),
    ("gradient_diag_shift_20_16x16", 0x402BCB33E0000000),
    ("gradient_diag_shift_20_23x31", 0x4033D01E80000000),
    ("gradient_diag_shift_20_32x32", 0x4033B1FCE0000000),
    ("gradient_diag_shift_20_47x33", 0x4034C29EA0000000),
    ("color_gradient_shift_10_16x16", 0x40176B8F40000000),
    ("color_gradient_shift_10_23x23", 0x401E011100000000),
    ("color_gradient_shift_10_32x32", 0x4021FA2000000000),
    ("color_gradient_shift_10_33x47", 0x402428E2C0000000),
    ("checkerboard_vs_inverse_1px_8x8", 0x40197F9420000000),
    ("checkerboard_vs_inverse_2px_8x8", 0x402438AA40000000),
    ("checkerboard_shift_10_8x8", 0x400E98AFC0000000),
    ("checkerboard_vs_inverse_1px_15x15", 0x402142B400000000),
    ("checkerboard_vs_inverse_2px_15x15", 0x4023D0C240000000),
    ("checkerboard_shift_10_15x15", 0x401433EB00000000),
    ("checkerboard_vs_inverse_1px_16x16", 0x4018BE5400000000),
    ("checkerboard_vs_inverse_2px_16x16", 0x4023CC7D60000000),
    ("checkerboard_shift_10_16x16", 0x40144558C0000000),
    ("checkerboard_vs_inverse_1px_23x23", 0x402120B960000000),
    ("checkerboard_vs_inverse_2px_23x23", 0x4023C4C7C0000000),
    ("checkerboard_shift_10_23x23", 0x4014332600000000),
    ("checkerboard_vs_inverse_1px_32x32", 0x4018BEC780000000),
    ("checkerboard_vs_inverse_2px_32x32", 0x4023C386E0000000),
    ("checkerboard_shift_10_32x32", 0x4014427720000000),
    ("checkerboard_vs_inverse_1px_33x33", 0x4021221AE0000000),
    ("checkerboard_vs_inverse_2px_33x33", 0x4023C3B2C0000000),
    ("checkerboard_shift_10_33x33", 0x401432DA00000000),
    ("checkerboard_vs_inverse_4px_32x32", 0x4043E1A740000000),
    ("checkerboard_vs_inverse_8px_32x32", 0x4052B31540000000),
    ("checkerboard_vs_inverse_4px_47x47", 0x4043B78780000000),
    ("checkerboard_vs_inverse_8px_47x47", 0x40523FD540000000),
    ("checkerboard_vs_inverse_4px_64x64", 0x4043E48120000000),
    ("checkerboard_vs_inverse_8px_64x64", 0x4052B45B00000000),
    ("stripes_h_2px_shift_15_16x16", 0x401FC5A9A0000000),
    ("stripes_v_2px_shift_15_16x16", 0x401FC5AB40000000),
    ("stripes_h_2px_shift_15_23x23", 0x401EE14C80000000),
    ("stripes_v_2px_shift_15_23x23", 0x401EE14D00000000),
    ("stripes_h_2px_shift_15_32x32", 0x401FCA0180000000),
    ("stripes_v_2px_shift_15_32x32", 0x401FC9FFE0000000),
    ("stripes_h_2px_shift_15_33x47", 0x401EE4AB80000000),
    ("stripes_v_2px_shift_15_33x47", 0x401EC60940000000),
    ("sine_1x1_shift_10_32x32", 0x401B287360000000),
    ("sine_2x2_shift_10_32x32", 0x40183D6AE0000000),
    ("sine_4x4_shift_10_32x32", 0x40173B9CA0000000),
    ("sine_1x1_shift_10_33x33", 0x401BF8FD40000000),
    ("sine_2x2_shift_10_33x33", 0x40185F3680000000),
    ("sine_4x4_shift_10_33x33", 0x40173C6580000000),
    ("sine_1x1_shift_10_47x47", 0x4021DEAE40000000),
    ("sine_2x2_shift_10_47x47", 0x401A3BAF60000000),
    ("sine_4x4_shift_10_47x47", 0x40178B2C40000000),
    ("sine_1x1_shift_10_64x64", 0x40232A3C20000000),
    ("sine_2x2_shift_10_64x64", 0x401D126DC0000000),
    ("sine_4x4_shift_10_64x64", 0x401852FA80000000),
    ("radial_shift_15_16x16", 0x4022C8E480000000),
    ("radial_shift_15_23x23", 0x402312D340000000),
    ("radial_shift_15_32x32", 0x40251430E0000000),
    ("radial_shift_15_47x47", 0x4028DD33E0000000),
    ("edge_v_shift_10_16x16", 0x4015CF3AA0000000),
    ("edge_h_shift_10_16x16", 0x4015CF3B80000000),
    ("edge_v_vs_blur_16x16", 0x40111B9E40000000),
    ("edge_v_shift_10_23x31", 0x40169B08E0000000),
    ("edge_h_shift_10_23x31", 0x40184584C0000000),
    ("edge_v_vs_blur_23x31", 0x4016545060000000),
    ("edge_v_shift_10_32x32", 0x40187B0EE0000000),
    ("edge_h_shift_10_32x32", 0x40187B1340000000),
    ("edge_v_vs_blur_32x32", 0x401217A620000000),
    ("edge_v_shift_10_47x33", 0x4026752340000000),
    ("edge_h_shift_10_47x33", 0x4018BBC100000000),
    ("edge_v_vs_blur_47x33", 0x4016C28D00000000),
    ("random_seed0_shift_10_16x16", 0x40146B5C80000000),
    ("random_seed0_noise_20_16x16", 0x3FFDD9C4C0000000),
    ("random_seed0_shift_10_23x23", 0x4013E52A20000000),
    ("random_seed0_noise_20_23x23", 0x3FFADC2900000000),
    ("random_seed0_shift_10_32x32", 0x4013F50400000000),
    ("random_seed0_noise_20_32x32", 0x3FFC725880000000),
    ("random_seed0_shift_10_33x47", 0x40144AAC00000000),
    ("random_seed0_noise_20_33x47", 0x3FF9F065E0000000),
    ("random_seed0_shift_10_47x33", 0x40145BBE00000000),
    ("random_seed0_noise_20_47x33", 0x40005FB540000000),
    ("random_seed1_shift_10_16x16", 0x401427AAC0000000),
    ("random_seed1_noise_20_16x16", 0x3FFD02EA80000000),
    ("random_seed1_shift_10_23x23", 0x40141F6580000000),
    ("random_seed1_noise_20_23x23", 0x3FF9AD7960000000),
    ("random_seed1_shift_10_32x32", 0x401405A000000000),
    ("random_seed1_noise_20_32x32", 0x3FFD1A7440000000),
    ("random_seed1_shift_10_33x47", 0x40144515C0000000),
    ("random_seed1_noise_20_33x47", 0x3FFC265280000000),
    ("random_seed1_shift_10_47x33", 0x4014197F40000000),
    ("random_seed1_noise_20_47x33", 0x400142CC20000000),
    ("random_seed2_shift_10_16x16", 0x4013E89E80000000),
    ("random_seed2_noise_20_16x16", 0x3FF5ECAA80000000),
    ("random_seed2_shift_10_23x23", 0x4013D73BA0000000),
    ("random_seed2_noise_20_23x23", 0x3FF913D8C0000000),
    ("random_seed2_shift_10_32x32", 0x4014578100000000),
    ("random_seed2_noise_20_32x32", 0x3FFBB54440000000),
    ("random_seed2_shift_10_33x47", 0x40141684C0000000),
    ("random_seed2_noise_20_33x47", 0x40006755C0000000),
    ("random_seed2_shift_10_47x33", 0x4014348F00000000),
    ("random_seed2_noise_20_47x33", 0x3FFEE4A940000000),
    ("random_seed3_shift_10_16x16", 0x401430C460000000),
    ("random_seed3_noise_20_16x16", 0x3FF5BBAB40000000),
    ("random_seed3_shift_10_23x23", 0x4014211380000000),
    ("random_seed3_noise_20_23x23", 0x3FFC40F700000000),
    ("random_seed3_shift_10_32x32", 0x40147E60A0000000),
    ("random_seed3_noise_20_32x32", 0x3FFEDE96A0000000),
    ("random_seed3_shift_10_33x47", 0x4014835580000000),
    ("random_seed3_noise_20_33x47", 0x3FFDB69B00000000),
    ("random_seed3_shift_10_47x33", 0x40144275E0000000),
    ("random_seed3_noise_20_47x33", 0x3FFA6CF020000000),
    ("random_seed4_shift_10_16x16", 0x4014A94480000000),
    ("random_seed4_noise_20_16x16", 0x3FF6F3BC00000000),
    ("random_seed4_shift_10_23x23", 0x40147856E0000000),
    ("random_seed4_noise_20_23x23", 0x3FF8F1F5E0000000),
    ("random_seed4_shift_10_32x32", 0x40141336C0000000),
    ("random_seed4_noise_20_32x32", 0x3FFB77C860000000),
    ("random_seed4_shift_10_33x47", 0x4013F41100000000),
    ("random_seed4_noise_20_33x47", 0x3FFDBB7340000000),
    ("random_seed4_shift_10_47x33", 0x40145CC380000000),
    ("random_seed4_noise_20_47x33", 0x3FFB454720000000),
    ("random_mid_contrast_1.2_32x32", 0x400A0D1340000000),
    ("random_mid_gamma_0.9_32x32", 0x400E6F90A0000000),
    ("random_mid_blur_32x32", 0x4022CD2500000000),
    ("random_mid_quantize_32_32x32", 0x3FDB8C61A0000000),
    ("random_mid_contrast_1.2_47x47", 0x400C7D1980000000),
    ("random_mid_gamma_0.9_47x47", 0x400E96EB00000000),
    ("random_mid_blur_47x47", 0x4022F254E0000000),
    ("random_mid_quantize_32_47x47", 0x3FDC3C0BC0000000),
    ("random_mid_contrast_1.2_64x64", 0x400A6A5C80000000),
    ("random_mid_gamma_0.9_64x64", 0x400ED05560000000),
    ("random_mid_blur_64x64", 0x40233D3F40000000),
    ("random_mid_quantize_32_64x64", 0x3FDF8E0D80000000),
    ("color_grad_channel_swap_16x16", 0x4041EB8C40000000),
    ("color_grad_hue_shift_16x16", 0x4048263F00000000),
    ("color_grad_channel_swap_23x23", 0x4049D56B00000000),
    ("color_grad_hue_shift_23x23", 0x4050B28F00000000),
    ("color_grad_channel_swap_32x32", 0x4050A838A0000000),
    ("color_grad_hue_shift_32x32", 0x4056015C80000000),
    ("color_grad_channel_swap_47x33", 0x4055FC0240000000),
    ("color_grad_hue_shift_47x33", 0x405B88E600000000),
    ("random_color_channel_swap_32x32", 0x401948CE40000000),
    ("random_color_hue_shift_32x32", 0x402436CCC0000000),
    ("random_color_channel_swap_47x47", 0x4017925880000000),
    ("random_color_hue_shift_47x47", 0x4024FA6DE0000000),
];

enum CaseResult {
    BitExact,
    WithinTolerance(f64),
    Failed(String),
}

/// Run a single test case: generate image pair, compute score, compare to x86 reference.
fn run_case(name: &str, expected_bits: u64) -> CaseResult {
    let (width, height) = match parse_dimensions(name) {
        Some(d) => d,
        None => return CaseResult::Failed(format!("{name}: could not parse dimensions")),
    };

    if width < MIN_DIMENSION || height < MIN_DIMENSION {
        return CaseResult::Failed(format!(
            "{name}: dimensions {width}x{height} below minimum {MIN_DIMENSION}"
        ));
    }

    let (img_a, img_b) = match generate_image_pair(name, width, height) {
        Some(pair) => pair,
        None => return CaseResult::Failed(format!("{name}: unknown image pattern")),
    };

    let pixels_a = rgb_bytes_to_pixels(&img_a);
    let pixels_b = rgb_bytes_to_pixels(&img_b);
    let img_a = Img::new(pixels_a, width, height);
    let img_b = Img::new(pixels_b, width, height);

    let params = ButteraugliParams::default();
    let result = match butteraugli(img_a.as_ref(), img_b.as_ref(), &params) {
        Ok(r) => r,
        Err(e) => return CaseResult::Failed(format!("{name}: butteraugli error: {e}")),
    };

    let expected = f64::from_bits(expected_bits);
    let actual = result.score;
    let actual_bits = actual.to_bits();

    if actual_bits == expected_bits {
        return CaseResult::BitExact;
    }

    // Check relative difference
    let diff = (actual - expected).abs();
    let rel = if expected.abs() > 1e-15 {
        diff / expected.abs()
    } else {
        diff
    };

    if rel > MAX_RELATIVE_DIFF {
        CaseResult::Failed(format!(
            "{name}: score mismatch — expected {expected:.10} (0x{expected_bits:016X}), \
             got {actual:.10} (0x{actual_bits:016X}), \
             rel_diff={rel:.2e} ({:.6}%)",
            rel * 100.0
        ))
    } else {
        CaseResult::WithinTolerance(rel)
    }
}

#[test]
fn test_cross_arch_parity_all() {
    let mut bit_exact = 0usize;
    let mut within_tolerance = 0usize;
    let mut failed = 0usize;
    let mut max_rel_diff = 0.0f64;
    let mut failures = Vec::new();

    for &(name, expected_bits) in X86_REFERENCE_SCORES {
        match run_case(name, expected_bits) {
            CaseResult::BitExact => bit_exact += 1,
            CaseResult::WithinTolerance(rel) => {
                within_tolerance += 1;
                if rel > max_rel_diff {
                    max_rel_diff = rel;
                }
            }
            CaseResult::Failed(msg) => {
                failed += 1;
                failures.push(msg);
            }
        }
    }

    let total = X86_REFERENCE_SCORES.len();
    eprintln!("\nCross-architecture parity results:");
    eprintln!("  Total:            {total}");
    eprintln!("  Bit-exact:        {bit_exact}");
    eprintln!("  Within tolerance: {within_tolerance}");
    eprintln!("  Failed:           {failed}");
    if within_tolerance > 0 {
        eprintln!("  Max relative diff: {max_rel_diff:.2e}");
    }

    for msg in &failures {
        eprintln!("  FAIL: {msg}");
    }

    assert_eq!(
        failed, 0,
        "{failed}/{total} cross-arch parity tests failed (tolerance: {MAX_RELATIVE_DIFF:.0e})"
    );
}
