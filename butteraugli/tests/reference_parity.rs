//! Tests that verify butteraugli scores against hard-coded C++ reference values
//! from libjxl's `butteraugli_main`.
//!
//! Reference data was captured from `butteraugli_main` (recursive Comparator path),
//! NOT from the standalone `butteraugli` library (different entry point, different scores
//! for some synthetic patterns).
//!
//! Run with: `cargo test --test reference_parity`
//!
//! Reference scores were captured from the FIR blur path. The iir-blur feature
//! changes scores by 0.1–5% on real photos and far more on tiny synthetics, so
//! this whole file is FIR-only — `cargo test --features iir-blur` skips it.
#![cfg(not(feature = "iir-blur"))]

mod common;

use butteraugli::{ButteraugliParams, Img, butteraugli, reference_data};
use common::generators::{generate_image_pair, rgb_bytes_to_pixels};

/// Minimum dimension requirement for butteraugli
const MIN_DIMENSION: usize = 8;

/// Maximum allowed relative difference vs C++ butteraugli_main.
/// Observed max is ~0.002% (FMA rounding noise). 0.1% gives comfortable margin.
const SCORE_TOLERANCE: f64 = 0.001;

/// Test a single reference case against Rust butteraugli.
fn test_reference_case(case: &reference_data::ReferenceCase) -> Option<f64> {
    if case.width < MIN_DIMENSION || case.height < MIN_DIMENSION {
        return None;
    }

    let (img_a, img_b) = generate_image_pair(case.name, case.width, case.height)?;

    let pixels_a = rgb_bytes_to_pixels(&img_a);
    let pixels_b = rgb_bytes_to_pixels(&img_b);
    let img_a = Img::new(pixels_a, case.width, case.height);
    let img_b = Img::new(pixels_b, case.width, case.height);

    let params = ButteraugliParams::default()
        .with_intensity_target(reference_data::REFERENCE_INTENSITY_TARGET);
    let result = butteraugli(img_a.as_ref(), img_b.as_ref(), &params).expect("valid test input");

    let score_diff = (result.score - case.expected_score).abs();
    let score_rel = if case.expected_score > 0.001 {
        score_diff / case.expected_score
    } else {
        score_diff
    };

    Some(score_rel)
}

#[test]
fn test_all_reference_cases() {
    let mut passed = 0;
    let mut failed = 0;
    let mut skipped = 0;
    let mut max_rel = 0.0f64;
    let mut max_name = "";

    for case in reference_data::REFERENCE_CASES {
        match test_reference_case(case) {
            None => {
                skipped += 1;
                continue;
            }
            Some(score_rel) => {
                if score_rel > max_rel {
                    max_rel = score_rel;
                    max_name = case.name;
                }
                if score_rel < SCORE_TOLERANCE {
                    passed += 1;
                } else {
                    failed += 1;
                    eprintln!(
                        "FAIL {}: diff={:.4}% (tolerance: {:.2}%)",
                        case.name,
                        score_rel * 100.0,
                        SCORE_TOLERANCE * 100.0
                    );
                }
            }
        }
    }

    eprintln!(
        "\nReference parity: {} passed, {} failed, {} skipped\n\
         Max relative diff: {:.4}% on '{}'",
        passed,
        failed,
        skipped,
        max_rel * 100.0,
        max_name
    );

    assert_eq!(
        failed,
        0,
        "All reference cases must pass at {:.2}% tolerance",
        SCORE_TOLERANCE * 100.0
    );
}

#[test]
fn test_uniform_cases() {
    for case in reference_data::REFERENCE_CASES {
        if !case.name.starts_with("uniform_") {
            continue;
        }
        if let Some(rel) = test_reference_case(case) {
            assert!(
                rel < SCORE_TOLERANCE,
                "{}: score differs by {:.4}%",
                case.name,
                rel * 100.0
            );
        }
    }
}

#[test]
fn test_gradient_cases() {
    for case in reference_data::REFERENCE_CASES {
        if !case.name.starts_with("gradient_") {
            continue;
        }
        if let Some(rel) = test_reference_case(case) {
            assert!(
                rel < SCORE_TOLERANCE,
                "{}: score differs by {:.4}%",
                case.name,
                rel * 100.0
            );
        }
    }
}

#[test]
fn test_reference_data_integrity() {
    #[allow(clippy::assertions_on_constants)]
    {
        assert!(
            reference_data::REFERENCE_CASE_COUNT > 100,
            "Expected at least 100 reference cases, got {}",
            reference_data::REFERENCE_CASE_COUNT
        );
    }

    assert_eq!(
        reference_data::REFERENCE_CASES.len(),
        reference_data::REFERENCE_CASE_COUNT,
        "Reference case count mismatch"
    );

    for case in reference_data::REFERENCE_CASES {
        assert!(
            case.expected_score >= 0.0,
            "Expected score should be non-negative for '{}'",
            case.name
        );
    }
}
