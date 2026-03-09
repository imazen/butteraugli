//! Captures reference scores from libjxl's butteraugli_main for all synthetic test cases.
//!
//! This test is #[ignore] by default — it requires the C++ butteraugli_main binary.
//! Run with:
//!   cargo test --test capture_cpp_scores -- --ignored --nocapture 2>&1 | tee /tmp/capture.log
//!
//! It generates PPM image pairs, runs butteraugli_main on each, and writes
//! the results to /tmp/butteraugli_reference_data.rs for pasting into
//! butteraugli/src/reference_data.rs.

mod common;

use common::generators::generate_image_pair;
use std::io::Write;
use std::process::Command;

const BUTTERAUGLI_MAIN: &str = "/home/lilith/work/jxl-efforts/libjxl/build/tools/butteraugli_main";
const INTENSITY_TARGET: &str = "80";
const PPM_DIR: &str = "/tmp/butteraugli_ref";

/// Write raw RGB data as a PPM file (no ICC profile, interpreted as sRGB by butteraugli_main).
fn write_ppm(path: &str, data: &[u8], width: usize, height: usize) {
    assert_eq!(data.len(), width * height * 3);
    let mut f = std::fs::File::create(path).expect("create ppm");
    write!(f, "P6\n{} {}\n255\n", width, height).unwrap();
    f.write_all(data).unwrap();
}

/// Run butteraugli_main and return the score.
fn run_cpp(ref_path: &str, dist_path: &str) -> f64 {
    let output = Command::new(BUTTERAUGLI_MAIN)
        .arg(ref_path)
        .arg(dist_path)
        .arg("--intensity_target")
        .arg(INTENSITY_TARGET)
        .output()
        .expect("failed to run butteraugli_main");

    assert!(
        output.status.success(),
        "butteraugli_main failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // First line is the score
    stdout
        .lines()
        .next()
        .expect("no output")
        .trim()
        .parse()
        .expect("failed to parse score")
}

/// All test case specifications: (pattern_prefix, shift/param, sizes).
/// The name is constructed as "{pattern}_{W}x{H}".
fn test_cases() -> Vec<(String, usize, usize)> {
    let mut cases = Vec::new();

    // Square sizes
    let sq_small: &[(usize, usize)] = &[
        (8, 8),
        (9, 9),
        (15, 15),
        (16, 16),
        (17, 17),
        (23, 23),
        (24, 24),
        (31, 31),
        (32, 32),
        (33, 33),
        (47, 47),
        (48, 48),
        (64, 64),
    ];
    let sq_medium: &[(usize, usize)] = &[(128, 128), (192, 192), (256, 256)];
    let sq_large: &[(usize, usize)] = &[(384, 384), (512, 512)];

    // Non-square sizes
    let nonsq: &[(usize, usize)] = &[(23, 31), (31, 23), (47, 33), (33, 47), (128, 96), (96, 128)];

    // All sizes combined
    let all_small: Vec<(usize, usize)> = sq_small
        .iter()
        .copied()
        .chain(nonsq.iter().copied())
        .collect();
    // Medium + large
    let all_medium: Vec<(usize, usize)> = sq_medium.to_vec();
    let all_large: Vec<(usize, usize)> = sq_large.to_vec();

    // Sizes for most patterns: small + medium
    let standard: Vec<(usize, usize)> = all_small
        .iter()
        .copied()
        .chain(all_medium.iter().copied())
        .collect();
    // Extended: standard + large
    let extended: Vec<(usize, usize)> = standard
        .iter()
        .copied()
        .chain(all_large.iter().copied())
        .collect();

    // Helper: add cases for a pattern at given sizes (min dimension filter)
    let mut add = |prefix: &str, sizes: &[(usize, usize)], min_dim: usize| {
        for &(w, h) in sizes {
            if w >= min_dim && h >= min_dim {
                cases.push((format!("{}_{w}x{h}", prefix), w, h));
            }
        }
    };

    // === Uniform gray shifts ===
    for shift in [1, 5, 10, 20, 50] {
        add(&format!("uniform_gray_128_shift_{shift}"), &extended, 8);
    }

    // === Uniform color shifts ===
    for color in ["red", "green", "blue"] {
        add(&format!("uniform_{color}_shift_20"), &standard, 8);
    }

    // === Gradient patterns ===
    for dir in ["h", "v"] {
        add(&format!("gradient_{dir}_shift_15"), &extended, 8);
    }
    add("gradient_diag_shift_20", &standard, 8);
    add("color_gradient_shift_10", &standard, 8);

    // === Checkerboard vs inverse ===
    for block in [1, 2] {
        add(&format!("checkerboard_vs_inverse_{block}px"), &standard, 8);
    }
    for block in [4, 8] {
        // Need at least 2*block for meaningful pattern
        add(
            &format!("checkerboard_vs_inverse_{block}px"),
            &standard,
            block * 2,
        );
    }

    // === Checkerboard shift ===
    add("checkerboard_shift_10", &standard, 8);

    // === Stripes ===
    add("stripes_h_2px_shift_15", &standard, 8);
    add("stripes_v_2px_shift_15", &standard, 8);

    // === Sine waves ===
    for freq in ["1x1", "2x2", "4x4"] {
        add(&format!("sine_{freq}_shift_10"), &standard, 16);
    }

    // === Radial ===
    add("radial_shift_15", &standard, 8);

    // === Edges ===
    add("edge_v_shift_10", &standard, 8);
    add("edge_h_shift_10", &standard, 8);
    add("edge_v_vs_blur", &standard, 8);

    // === Random seeded ===
    for seed_idx in 0..5 {
        add(&format!("random_seed{seed_idx}_shift_10"), &standard, 8);
    }
    for seed_idx in 0..3 {
        add(&format!("random_seed{seed_idx}_noise_20"), &standard, 8);
    }

    // === Random midrange distortions ===
    add("random_mid_contrast_1.2", &standard, 16);
    add("random_mid_gamma_0.9", &standard, 16);
    add("random_mid_blur", &standard, 16);
    add("random_mid_quantize_32", &standard, 16);

    // === Color distortions ===
    add("color_grad_channel_swap", &standard, 8);
    add("color_grad_hue_shift", &standard, 8);
    add("random_color_channel_swap", &standard, 16);
    add("random_color_hue_shift", &standard, 16);

    cases
}

#[test]
#[ignore]
fn capture_all_cpp_scores() {
    // Verify butteraugli_main exists
    assert!(
        std::path::Path::new(BUTTERAUGLI_MAIN).exists(),
        "butteraugli_main not found at {BUTTERAUGLI_MAIN}"
    );

    std::fs::create_dir_all(PPM_DIR).expect("create ppm dir");

    let cases = test_cases();
    eprintln!("Generating {} test cases...", cases.len());

    let mut results: Vec<(String, usize, usize, f64)> = Vec::new();
    let mut skipped = 0;
    let failed = 0;

    for (i, (name, w, h)) in cases.iter().enumerate() {
        if i % 50 == 0 {
            eprintln!("  [{}/{}] {}", i, cases.len(), name);
        }

        let pair = match generate_image_pair(name, *w, *h) {
            Some(p) => p,
            None => {
                eprintln!("  SKIP: unrecognized pattern '{}'", name);
                skipped += 1;
                continue;
            }
        };

        let ref_path = format!("{PPM_DIR}/{name}_a.ppm");
        let dist_path = format!("{PPM_DIR}/{name}_b.ppm");
        write_ppm(&ref_path, &pair.0, *w, *h);
        write_ppm(&dist_path, &pair.1, *w, *h);

        let score = run_cpp(&ref_path, &dist_path);

        results.push((name.clone(), *w, *h, score));

        // Clean up PPMs to save disk
        let _ = std::fs::remove_file(&ref_path);
        let _ = std::fs::remove_file(&dist_path);
    }

    if failed > 0 || skipped > 0 {
        eprintln!(
            "\n{} captured, {} skipped, {} failed",
            results.len(),
            skipped,
            failed
        );
    }

    // Write Rust source file
    let out_path = "/tmp/butteraugli_reference_data.rs";
    let mut f = std::fs::File::create(out_path).expect("create output");

    writeln!(
        f,
        "//! Reference scores captured from libjxl butteraugli_main."
    )
    .unwrap();
    writeln!(f, "//!").unwrap();
    writeln!(
        f,
        "//! Generated by: cargo test --test capture_cpp_scores -- --ignored"
    )
    .unwrap();
    writeln!(
        f,
        "//! butteraugli_main: /home/lilith/work/jxl-efforts/libjxl/build/tools/butteraugli_main"
    )
    .unwrap();
    writeln!(
        f,
        "//! libjxl commit: d2c7032e0ddb5e5590852292ed635b9bc7ab1e12 (2026-02-22)"
    )
    .unwrap();
    writeln!(f, "//! intensity_target: {INTENSITY_TARGET}").unwrap();
    writeln!(f, "//! Image format: PPM (nonlinear sRGB, no ICC profiles)").unwrap();
    writeln!(f, "//! Date: {}", chrono::Utc::now().format("%Y-%m-%d")).unwrap();
    writeln!(f).unwrap();

    writeln!(
        f,
        "/// Intensity target used when capturing reference scores."
    )
    .unwrap();
    writeln!(
        f,
        "pub const REFERENCE_INTENSITY_TARGET: f32 = {INTENSITY_TARGET}.0;"
    )
    .unwrap();
    writeln!(f).unwrap();

    writeln!(
        f,
        "/// A reference test case with expected butteraugli score from C++ libjxl."
    )
    .unwrap();
    writeln!(f, "#[derive(Debug)]").unwrap();
    writeln!(f, "pub struct ReferenceCase {{").unwrap();
    writeln!(f, "    pub name: &'static str,").unwrap();
    writeln!(f, "    pub width: usize,").unwrap();
    writeln!(f, "    pub height: usize,").unwrap();
    writeln!(f, "    pub expected_score: f64,").unwrap();
    writeln!(f, "}}").unwrap();
    writeln!(f).unwrap();

    writeln!(f, "/// Number of reference cases.").unwrap();
    writeln!(
        f,
        "pub const REFERENCE_CASE_COUNT: usize = {};",
        results.len()
    )
    .unwrap();
    writeln!(f).unwrap();

    writeln!(
        f,
        "/// All reference cases with C++ butteraugli_main scores."
    )
    .unwrap();
    writeln!(f, "pub const REFERENCE_CASES: &[ReferenceCase] = &[").unwrap();

    for (name, w, h, score) in &results {
        writeln!(f, "    ReferenceCase {{").unwrap();
        writeln!(f, "        name: \"{name}\",").unwrap();
        writeln!(f, "        width: {w},").unwrap();
        writeln!(f, "        height: {h},").unwrap();
        // Use full precision
        writeln!(f, "        expected_score: {score:.15},").unwrap();
        writeln!(f, "    }},").unwrap();
    }

    writeln!(f, "];").unwrap();

    eprintln!("\nWrote {} reference cases to {out_path}", results.len());
    eprintln!("Copy to butteraugli/src/reference_data.rs after review.");
}
