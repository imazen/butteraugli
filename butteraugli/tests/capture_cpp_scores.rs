//! Captures reference scores from libjxl's butteraugli_main for all synthetic test cases.
//!
//! This test is #[ignore] by default — it requires the C++ butteraugli_main binary.
//! Run with:
//!   cargo test --test capture_cpp_scores -- --ignored --nocapture 2>&1 | tee /tmp/capture.log
//!
//! It generates PPM image pairs, runs butteraugli_main on each, computes diffmap
//! stats from Rust, and writes the results to /tmp/butteraugli_reference_data.rs.

mod common;

use butteraugli::{ButteraugliParams, Img, butteraugli};
use common::generators::{generate_image_pair, rgb_bytes_to_pixels};
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
    stdout
        .lines()
        .next()
        .expect("no output")
        .trim()
        .parse()
        .expect("failed to parse score")
}

struct DiffmapStats {
    min: f32,
    max: f32,
    mean: f32,
    std: f32,
}

/// Compute diffmap stats from Rust butteraugli.
fn compute_rust_stats(img_a: &[u8], img_b: &[u8], w: usize, h: usize) -> DiffmapStats {
    let pixels_a = rgb_bytes_to_pixels(img_a);
    let pixels_b = rgb_bytes_to_pixels(img_b);
    let img_a = Img::new(pixels_a, w, h);
    let img_b = Img::new(pixels_b, w, h);

    let params = ButteraugliParams::default()
        .with_intensity_target(INTENSITY_TARGET.parse::<f32>().unwrap())
        .with_compute_diffmap(true);
    let result = butteraugli(img_a.as_ref(), img_b.as_ref(), &params).expect("valid input");
    let diffmap = result.diffmap.expect("diffmap requested");
    let buf = diffmap.buf();

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    for &v in buf {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
        sum += v as f64;
    }
    let n = buf.len() as f64;
    let mean = sum / n;

    let mut var_sum = 0.0f64;
    for &v in buf {
        let d = v as f64 - mean;
        var_sum += d * d;
    }
    let std = (var_sum / n).sqrt() as f32;

    DiffmapStats {
        min,
        max,
        mean: mean as f32,
        std,
    }
}

/// All test case specifications.
fn test_cases() -> Vec<(String, usize, usize)> {
    let mut cases = Vec::new();

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
    let nonsq: &[(usize, usize)] = &[(23, 31), (31, 23), (47, 33), (33, 47), (128, 96), (96, 128)];

    let all_small: Vec<(usize, usize)> = sq_small
        .iter()
        .copied()
        .chain(nonsq.iter().copied())
        .collect();
    let all_medium: Vec<(usize, usize)> = sq_medium.to_vec();
    let all_large: Vec<(usize, usize)> = sq_large.to_vec();

    let standard: Vec<(usize, usize)> = all_small
        .iter()
        .copied()
        .chain(all_medium.iter().copied())
        .collect();
    let extended: Vec<(usize, usize)> = standard
        .iter()
        .copied()
        .chain(all_large.iter().copied())
        .collect();

    let mut add = |prefix: &str, sizes: &[(usize, usize)], min_dim: usize| {
        for &(w, h) in sizes {
            if w >= min_dim && h >= min_dim {
                cases.push((format!("{}_{w}x{h}", prefix), w, h));
            }
        }
    };

    for shift in [1, 5, 10, 20, 50] {
        add(&format!("uniform_gray_128_shift_{shift}"), &extended, 8);
    }
    for color in ["red", "green", "blue"] {
        add(&format!("uniform_{color}_shift_20"), &standard, 8);
    }
    for dir in ["h", "v"] {
        add(&format!("gradient_{dir}_shift_15"), &extended, 8);
    }
    add("gradient_diag_shift_20", &standard, 8);
    add("color_gradient_shift_10", &standard, 8);

    for block in [1, 2] {
        add(&format!("checkerboard_vs_inverse_{block}px"), &standard, 8);
    }
    for block in [4, 8] {
        add(
            &format!("checkerboard_vs_inverse_{block}px"),
            &standard,
            block * 2,
        );
    }
    add("checkerboard_shift_10", &standard, 8);

    add("stripes_h_2px_shift_15", &standard, 8);
    add("stripes_v_2px_shift_15", &standard, 8);

    for freq in ["1x1", "2x2", "4x4"] {
        add(&format!("sine_{freq}_shift_10"), &standard, 16);
    }

    add("radial_shift_15", &standard, 8);

    add("edge_v_shift_10", &standard, 8);
    add("edge_h_shift_10", &standard, 8);
    add("edge_v_vs_blur", &standard, 8);

    for seed_idx in 0..5 {
        add(&format!("random_seed{seed_idx}_shift_10"), &standard, 8);
    }
    for seed_idx in 0..3 {
        add(&format!("random_seed{seed_idx}_noise_20"), &standard, 8);
    }

    add("random_mid_contrast_1.2", &standard, 16);
    add("random_mid_gamma_0.9", &standard, 16);
    add("random_mid_blur", &standard, 16);
    add("random_mid_quantize_32", &standard, 16);

    add("color_grad_channel_swap", &standard, 8);
    add("color_grad_hue_shift", &standard, 8);
    add("random_color_channel_swap", &standard, 16);
    add("random_color_hue_shift", &standard, 16);

    cases
}

struct CaseResult {
    name: String,
    width: usize,
    height: usize,
    cpp_score: f64,
    stats: DiffmapStats,
}

#[test]
#[ignore]
fn capture_all_cpp_scores() {
    assert!(
        std::path::Path::new(BUTTERAUGLI_MAIN).exists(),
        "butteraugli_main not found at {BUTTERAUGLI_MAIN}"
    );

    std::fs::create_dir_all(PPM_DIR).expect("create ppm dir");

    let cases = test_cases();
    eprintln!("Generating {} test cases...", cases.len());

    let mut results: Vec<CaseResult> = Vec::new();
    let mut skipped = 0;

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

        // C++ score
        let ref_path = format!("{PPM_DIR}/{name}_a.ppm");
        let dist_path = format!("{PPM_DIR}/{name}_b.ppm");
        write_ppm(&ref_path, &pair.0, *w, *h);
        write_ppm(&dist_path, &pair.1, *w, *h);
        let cpp_score = run_cpp(&ref_path, &dist_path);
        let _ = std::fs::remove_file(&ref_path);
        let _ = std::fs::remove_file(&dist_path);

        // Rust diffmap stats
        let stats = compute_rust_stats(&pair.0, &pair.1, *w, *h);

        results.push(CaseResult {
            name: name.clone(),
            width: *w,
            height: *h,
            cpp_score,
            stats,
        });
    }

    if skipped > 0 {
        eprintln!("\n{} captured, {} skipped", results.len(), skipped);
    }

    // Write Rust source file
    let out_path = "/tmp/butteraugli_reference_data.rs";
    let mut f = std::fs::File::create(out_path).expect("create output");

    writeln!(
        f,
        "//! Reference scores from libjxl butteraugli_main + diffmap stats from Rust."
    )
    .unwrap();
    writeln!(f, "//!").unwrap();
    writeln!(
        f,
        "//! Scores: captured from C++ butteraugli_main (ground truth for parity testing)."
    )
    .unwrap();
    writeln!(
        f,
        "//! Stats: computed from Rust butteraugli diffmap (regression detection)."
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

    writeln!(f, "/// Diffmap statistics computed from Rust butteraugli.").unwrap();
    writeln!(f, "#[derive(Debug, Clone, Copy)]").unwrap();
    writeln!(f, "#[non_exhaustive]").unwrap();
    writeln!(f, "pub struct DiffmapStats {{").unwrap();
    writeln!(f, "    pub min: f32,").unwrap();
    writeln!(f, "    pub max: f32,").unwrap();
    writeln!(f, "    pub mean: f32,").unwrap();
    writeln!(f, "    pub std: f32,").unwrap();
    writeln!(f, "}}").unwrap();
    writeln!(f).unwrap();

    writeln!(
        f,
        "/// A reference test case with C++ score and Rust diffmap stats."
    )
    .unwrap();
    writeln!(f, "#[derive(Debug)]").unwrap();
    writeln!(f, "#[non_exhaustive]").unwrap();
    writeln!(f, "pub struct ReferenceCase {{").unwrap();
    writeln!(f, "    pub name: &'static str,").unwrap();
    writeln!(f, "    pub width: usize,").unwrap();
    writeln!(f, "    pub height: usize,").unwrap();
    writeln!(
        f,
        "    /// Ground truth score from C++ libjxl butteraugli_main."
    )
    .unwrap();
    writeln!(f, "    pub expected_score: f64,").unwrap();
    writeln!(
        f,
        "    /// Diffmap statistics from Rust butteraugli (for regression detection)."
    )
    .unwrap();
    writeln!(f, "    pub expected_stats: DiffmapStats,").unwrap();
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

    writeln!(f, "/// All reference cases.").unwrap();
    writeln!(f, "pub const REFERENCE_CASES: &[ReferenceCase] = &[").unwrap();

    for r in &results {
        writeln!(f, "    ReferenceCase {{").unwrap();
        writeln!(f, "        name: \"{}\",", r.name).unwrap();
        writeln!(f, "        width: {},", r.width).unwrap();
        writeln!(f, "        height: {},", r.height).unwrap();
        writeln!(f, "        expected_score: {:.15},", r.cpp_score).unwrap();
        writeln!(f, "        expected_stats: DiffmapStats {{").unwrap();
        writeln!(f, "            min: {:?},", r.stats.min).unwrap();
        writeln!(f, "            max: {:?},", r.stats.max).unwrap();
        writeln!(f, "            mean: {:?},", r.stats.mean).unwrap();
        writeln!(f, "            std: {:?},", r.stats.std).unwrap();
        writeln!(f, "        }},").unwrap();
        writeln!(f, "    }},").unwrap();
    }

    writeln!(f, "];").unwrap();

    eprintln!("\nWrote {} reference cases to {out_path}", results.len());
}
