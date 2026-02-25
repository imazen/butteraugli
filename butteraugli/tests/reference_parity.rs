//! Tests that verify butteraugli scores against hard-coded C++ reference values.
//!
//! These tests use pre-captured C++ butteraugli scores for synthetic test images,
//! allowing parity verification without requiring jpegli-sys at runtime.
//!
//! Run with: `cargo test --test reference_parity`

mod common;

use butteraugli::{ButteraugliParams, Img, butteraugli, reference_data};
use common::generators::{generate_image_pair, rgb_bytes_to_pixels};

// ============================================================================
// Tests
// ============================================================================

/// Minimum dimension requirement for butteraugli
const MIN_DIMENSION: usize = 8;

/// Test a single reference case against Rust butteraugli.
fn test_reference_case(case: &reference_data::ReferenceCase, tolerance: f64) {
    // Skip cases that don't meet minimum dimension requirement
    if case.width < MIN_DIMENSION || case.height < MIN_DIMENSION {
        return;
    }

    let (img_a, img_b) = match generate_image_pair(case.name, case.width, case.height) {
        Some(pair) => pair,
        None => {
            eprintln!("SKIP: Could not generate image pair for '{}'", case.name);
            return;
        }
    };

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

    assert!(
        score_rel < tolerance,
        "{}: score differs by {:.2}% (Rust={:.6}, C++={:.6})",
        case.name,
        score_rel * 100.0,
        result.score,
        case.expected_score
    );
}

#[test]
fn test_all_reference_cases_loose() {
    // Loose tolerance: 20% relative difference
    // This catches major regressions while allowing for implementation differences
    let tolerance = 0.20;
    let mut passed = 0;
    let mut failed = 0;
    let mut skipped = 0;

    for case in reference_data::REFERENCE_CASES {
        // Skip cases that don't meet minimum dimension requirement
        if case.width < MIN_DIMENSION || case.height < MIN_DIMENSION {
            skipped += 1;
            continue;
        }

        if generate_image_pair(case.name, case.width, case.height).is_none() {
            skipped += 1;
            continue;
        }

        let (img_a, img_b) = generate_image_pair(case.name, case.width, case.height).unwrap();
        let pixels_a = rgb_bytes_to_pixels(&img_a);
        let pixels_b = rgb_bytes_to_pixels(&img_b);
        let img_a = Img::new(pixels_a, case.width, case.height);
        let img_b = Img::new(pixels_b, case.width, case.height);
        let params = ButteraugliParams::default()
            .with_intensity_target(reference_data::REFERENCE_INTENSITY_TARGET);
        let result =
            butteraugli(img_a.as_ref(), img_b.as_ref(), &params).expect("valid test input");

        let score_diff = (result.score - case.expected_score).abs();
        let score_rel = if case.expected_score > 0.001 {
            score_diff / case.expected_score
        } else {
            score_diff
        };

        if score_rel < tolerance {
            passed += 1;
        } else {
            failed += 1;
            eprintln!(
                "FAIL {}: Rust={:.4} C++={:.4} diff={:.2}%",
                case.name,
                result.score,
                case.expected_score,
                score_rel * 100.0
            );
        }
    }

    eprintln!(
        "\nReference parity: {} passed, {} failed, {} skipped (tolerance: {:.0}%)",
        passed,
        failed,
        skipped,
        tolerance * 100.0
    );

    // Allow up to 10% of tests to fail at loose tolerance
    let failure_rate = failed as f64 / (passed + failed) as f64;
    assert!(
        failure_rate < 0.10,
        "Too many reference tests failed: {}/{} ({:.1}%)",
        failed,
        passed + failed,
        failure_rate * 100.0
    );
}

#[test]
fn test_uniform_cases_strict() {
    // Uniform images should match very closely
    let tolerance = 0.05;

    for case in reference_data::REFERENCE_CASES {
        if !case.name.starts_with("uniform_") {
            continue;
        }
        test_reference_case(case, tolerance);
    }
}

#[test]
fn test_gradient_cases() {
    let tolerance = 0.15;

    for case in reference_data::REFERENCE_CASES {
        if !case.name.starts_with("gradient_") {
            continue;
        }
        test_reference_case(case, tolerance);
    }
}

#[test]
fn test_reference_data_integrity() {
    // Verify the reference data is loaded correctly
    // Note: This is a compile-time check, but we keep it as a runtime assertion
    // to document the expected minimum number of test cases
    #[allow(clippy::assertions_on_constants)]
    {
        assert!(
            reference_data::REFERENCE_CASE_COUNT > 100,
            "Expected at least 100 reference cases, got {}",
            reference_data::REFERENCE_CASE_COUNT
        );
    }

    assert!(
        reference_data::REFERENCE_CASES.len() == reference_data::REFERENCE_CASE_COUNT,
        "Reference case count mismatch"
    );

    // Check that all expected scores are positive
    for case in reference_data::REFERENCE_CASES {
        assert!(
            case.expected_score >= 0.0,
            "Expected score should be non-negative for '{}'",
            case.name
        );
    }
}
