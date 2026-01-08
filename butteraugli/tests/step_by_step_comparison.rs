#![cfg(feature = "cpp-parity")]
//! Step-by-step comparison of Rust vs C++ butteraugli pipeline.
//!
//! This test traces through each stage of the butteraugli algorithm
//! comparing intermediate values to locate where divergence occurs.

use butteraugli::blur::gaussian_blur;
use butteraugli::image::ImageF;
use butteraugli::opsin::srgb_to_xyb_butteraugli;
use butteraugli::psycho::separate_frequencies;
use jpegli_internals_sys::{
    butteraugli_blur, butteraugli_compare_full, butteraugli_opsin_dynamics,
    butteraugli_separate_frequencies, BUTTERAUGLI_OK,
};

// ============================================================================
// Test image generators
// ============================================================================

/// 64x64 checkerboard - known to have ~15% divergence
fn generate_checkerboard_64x64() -> Vec<u8> {
    (0..64)
        .flat_map(|y| {
            (0..64).flat_map(move |x| {
                let v = if (x + y) % 2 == 0 { 50u8 } else { 200u8 };
                [v, v, v]
            })
        })
        .collect()
}

/// 64x64 uniform gray
fn generate_uniform_64x64(val: u8) -> Vec<u8> {
    vec![val; 64 * 64 * 3]
}

/// 64x64 horizontal gradient
fn generate_h_gradient_64x64() -> Vec<u8> {
    (0..64)
        .flat_map(|_y| {
            (0..64).flat_map(|x| {
                let v = (x * 255 / 63) as u8;
                [v, v, v]
            })
        })
        .collect()
}

// ============================================================================
// Helper functions
// ============================================================================

fn srgb_to_linear_rust(srgb: &[u8]) -> Vec<f32> {
    srgb.iter()
        .map(|&v| {
            let x = v as f32 / 255.0;
            if x <= 0.04045 {
                x / 12.92
            } else {
                ((x + 0.055) / 1.055).powf(2.4)
            }
        })
        .collect()
}

fn stats(data: &[f32]) -> (f32, f32, f32) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = data.iter().sum();
    let mean = sum / data.len() as f32;
    (min, max, mean)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn mean_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() {
        return 0.0;
    }
    let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
    sum / a.len() as f32
}

// ============================================================================
// Stage 1: Blur comparison
// ============================================================================

#[test]
fn test_stage1_blur() {
    println!("\n=== Stage 1: Blur Comparison ===\n");

    let width = 64usize;
    let height = 64usize;

    // Create test pattern - a gradient
    let input: Vec<f32> = (0..height)
        .flat_map(|y| (0..width).map(move |x| (x + y) as f32 / (width + height) as f32))
        .collect();

    for sigma in [1.564f32, 3.225, 7.156] {
        // Rust blur
        let input_img = ImageF::from_vec(input.clone(), width, height);
        let rust_blurred = gaussian_blur(&input_img, sigma);
        let mut rust_flat = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                rust_flat.push(rust_blurred.get(x, y));
            }
        }

        // C++ blur
        let mut cpp_blurred = vec![0.0f32; width * height];
        let result = unsafe {
            butteraugli_blur(
                input.as_ptr(),
                width,
                height,
                sigma,
                cpp_blurred.as_mut_ptr(),
            )
        };

        if result != BUTTERAUGLI_OK {
            println!(
                "  sigma={:.3}: C++ butteraugli_blur failed with code {}",
                sigma, result
            );
            continue;
        }

        let max_diff = max_abs_diff(&rust_flat, &cpp_blurred);
        let mean_diff = mean_abs_diff(&rust_flat, &cpp_blurred);

        let (r_min, r_max, r_mean) = stats(&rust_flat);
        let (c_min, c_max, c_mean) = stats(&cpp_blurred);

        println!("  sigma={:.3}:", sigma);
        println!(
            "    Rust: min={:.4} max={:.4} mean={:.4}",
            r_min, r_max, r_mean
        );
        println!(
            "    C++:  min={:.4} max={:.4} mean={:.4}",
            c_min, c_max, c_mean
        );
        println!(
            "    max_diff={:.6} mean_diff={:.6} ({:.2}%)",
            max_diff,
            mean_diff,
            mean_diff / c_mean.abs().max(0.001) * 100.0
        );
    }
}

// ============================================================================
// Stage 2: XYB conversion comparison (via OpsinDynamicsImage)
// ============================================================================

#[test]
fn test_stage2_xyb_conversion() {
    println!("\n=== Stage 2: XYB Conversion (OpsinDynamicsImage) ===\n");

    let width = 64usize;
    let height = 64usize;
    let intensity_target = 80.0f32;

    for (name, srgb) in [
        ("Uniform Gray", generate_uniform_64x64(128)),
        ("H Gradient", generate_h_gradient_64x64()),
        ("Checkerboard", generate_checkerboard_64x64()),
    ] {
        let linear = srgb_to_linear_rust(&srgb);

        // Rust XYB
        let rust_xyb = srgb_to_xyb_butteraugli(&srgb, width, height, intensity_target);

        // C++ XYB (SIMPLIFIED wrapper: no blur, no bias, just mix+cbrt)
        let mut cpp_xyb = vec![0.0f32; width * height * 3];
        let result = unsafe {
            butteraugli_opsin_dynamics(
                linear.as_ptr(),
                width,
                height,
                intensity_target,
                cpp_xyb.as_mut_ptr(),
            )
        };

        if result != BUTTERAUGLI_OK {
            println!("  {}: C++ opsin_dynamics failed", name);
            continue;
        }

        // Compare X, Y, B channels
        for (ch, ch_name) in [(0, "X"), (1, "Y"), (2, "B")] {
            let mut rust_flat = Vec::with_capacity(width * height);
            let mut cpp_flat = Vec::with_capacity(width * height);
            for y in 0..height {
                for x in 0..width {
                    rust_flat.push(rust_xyb.plane(ch).get(x, y));
                    cpp_flat.push(cpp_xyb[(y * width + x) * 3 + ch]);
                }
            }

            let max_diff = max_abs_diff(&rust_flat, &cpp_flat);
            let mean_diff = mean_abs_diff(&rust_flat, &cpp_flat);
            let (_, _, r_mean) = stats(&rust_flat);
            let (_, _, c_mean) = stats(&cpp_flat);

            println!(
                "  {} - {} channel: max_diff={:.6} mean_diff={:.6} (Rust mean={:.4} C++ mean={:.4})",
                name, ch_name, max_diff, mean_diff, r_mean, c_mean
            );
        }
    }
}

// ============================================================================
// Stage 3: Frequency separation comparison
// ============================================================================

#[test]
fn test_stage3_frequency_separation() {
    println!("\n=== Stage 3: Frequency Separation ===\n");

    let width = 64usize;
    let height = 64usize;
    let intensity_target = 80.0f32;

    for (name, srgb) in [
        ("Uniform Gray", generate_uniform_64x64(128)),
        ("H Gradient", generate_h_gradient_64x64()),
        ("Checkerboard", generate_checkerboard_64x64()),
    ] {
        println!("  {}:", name);

        // Convert sRGB to linear RGB for C++ (C++ does its own XYB conversion)
        let linear = srgb_to_linear_rust(&srgb);

        // Get Rust XYB and frequency separation
        let rust_xyb = srgb_to_xyb_butteraugli(&srgb, width, height, intensity_target);
        let rust_psycho = separate_frequencies(&rust_xyb);

        // C++ frequency separation
        let mut cpp_lf_x = vec![0.0f32; width * height];
        let mut cpp_lf_y = vec![0.0f32; width * height];
        let mut cpp_lf_b = vec![0.0f32; width * height];
        let mut cpp_mf_x = vec![0.0f32; width * height];
        let mut cpp_mf_y = vec![0.0f32; width * height];
        let mut cpp_mf_b = vec![0.0f32; width * height];
        let mut cpp_hf_x = vec![0.0f32; width * height];
        let mut cpp_hf_y = vec![0.0f32; width * height];
        let mut cpp_uhf_x = vec![0.0f32; width * height];
        let mut cpp_uhf_y = vec![0.0f32; width * height];

        // C++ takes linear RGB (interleaved) and does its own XYB conversion
        let result = unsafe {
            butteraugli_separate_frequencies(
                linear.as_ptr(),
                width,
                height,
                intensity_target,
                cpp_lf_x.as_mut_ptr(),
                cpp_lf_y.as_mut_ptr(),
                cpp_lf_b.as_mut_ptr(),
                cpp_mf_x.as_mut_ptr(),
                cpp_mf_y.as_mut_ptr(),
                cpp_mf_b.as_mut_ptr(),
                cpp_hf_x.as_mut_ptr(),
                cpp_hf_y.as_mut_ptr(),
                cpp_uhf_x.as_mut_ptr(),
                cpp_uhf_y.as_mut_ptr(),
            )
        };

        if result != BUTTERAUGLI_OK {
            println!("    C++ separate_frequencies failed with code {}", result);
            continue;
        }

        // Compare each frequency band
        let band_names = [
            "LF_X", "LF_Y", "LF_B", "MF_X", "MF_Y", "MF_B", "HF_X", "HF_Y", "UHF_X", "UHF_Y",
        ];
        let cpp_bands: [&Vec<f32>; 10] = [
            &cpp_lf_x, &cpp_lf_y, &cpp_lf_b, &cpp_mf_x, &cpp_mf_y, &cpp_mf_b, &cpp_hf_x, &cpp_hf_y,
            &cpp_uhf_x, &cpp_uhf_y,
        ];

        for (i, band_name) in band_names.iter().enumerate() {
            // Get rust plane based on index
            let mut rust_flat = Vec::with_capacity(width * height);
            for y in 0..height {
                for x in 0..width {
                    let val = match i {
                        0 => rust_psycho.lf.plane(0).get(x, y),
                        1 => rust_psycho.lf.plane(1).get(x, y),
                        2 => rust_psycho.lf.plane(2).get(x, y),
                        3 => rust_psycho.mf.plane(0).get(x, y),
                        4 => rust_psycho.mf.plane(1).get(x, y),
                        5 => rust_psycho.mf.plane(2).get(x, y),
                        6 => rust_psycho.hf[0].get(x, y),
                        7 => rust_psycho.hf[1].get(x, y),
                        8 => rust_psycho.uhf[0].get(x, y),
                        9 => rust_psycho.uhf[1].get(x, y),
                        _ => 0.0,
                    };
                    rust_flat.push(val);
                }
            }
            let cpp_data = cpp_bands[i];

            let max_diff = max_abs_diff(&rust_flat, cpp_data);
            let mean_diff = mean_abs_diff(&rust_flat, cpp_data);
            let (_, _, r_mean) = stats(&rust_flat);
            let (_, _, c_mean) = stats(cpp_data);
            let rel_diff = if c_mean.abs() > 0.001 {
                mean_diff / c_mean.abs() * 100.0
            } else {
                0.0
            };

            // Only print if there's significant difference
            if max_diff > 0.001 || rel_diff > 1.0 {
                println!(
                    "    {}: max={:.6} mean={:.6} ({:.1}%) R_mean={:.4} C_mean={:.4}",
                    band_name, max_diff, mean_diff, rel_diff, r_mean, c_mean
                );
            }
        }
    }
}

// ============================================================================
// Full pipeline comparison
// ============================================================================

#[test]
fn test_full_pipeline_divergence_location() {
    println!("\n=== Full Pipeline Divergence Location ===\n");
    println!("Testing checkerboard pattern (known 15% divergence case)\n");

    let width = 64usize;
    let height = 64usize;
    let intensity_target = 80.0f32;

    let img1 = generate_checkerboard_64x64();
    let img2: Vec<u8> = img1.iter().map(|&v| v.saturating_add(20)).collect();

    let linear1 = srgb_to_linear_rust(&img1);
    let linear2 = srgb_to_linear_rust(&img2);

    // Get final scores from C++
    let mut cpp_score = 0.0f64;
    let mut cpp_diffmap = vec![0.0f32; width * height];
    let result = unsafe {
        butteraugli_compare_full(
            linear1.as_ptr(),
            linear2.as_ptr(),
            width,
            height,
            1.0,
            1.0,
            intensity_target,
            &mut cpp_score,
            cpp_diffmap.as_mut_ptr(),
        )
    };
    assert_eq!(result, BUTTERAUGLI_OK);

    // Rust computation
    let rust_xyb1 = srgb_to_xyb_butteraugli(&img1, width, height, intensity_target);
    let rust_xyb2 = srgb_to_xyb_butteraugli(&img2, width, height, intensity_target);
    let rust_psycho1 = separate_frequencies(&rust_xyb1);
    let rust_psycho2 = separate_frequencies(&rust_xyb2);

    println!("Stage-by-stage analysis:");

    // Check frequency bands difference
    println!("\n  Frequency band differences (img1 vs img2):");
    let bands = [
        "LF_X", "LF_Y", "LF_B", "MF_X", "MF_Y", "MF_B", "HF_X", "HF_Y", "UHF_X", "UHF_Y",
    ];

    for (i, band_name) in bands.iter().enumerate() {
        let mut diff_vals = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let v1 = match i {
                    0 => rust_psycho1.lf.plane(0).get(x, y),
                    1 => rust_psycho1.lf.plane(1).get(x, y),
                    2 => rust_psycho1.lf.plane(2).get(x, y),
                    3 => rust_psycho1.mf.plane(0).get(x, y),
                    4 => rust_psycho1.mf.plane(1).get(x, y),
                    5 => rust_psycho1.mf.plane(2).get(x, y),
                    6 => rust_psycho1.hf[0].get(x, y),
                    7 => rust_psycho1.hf[1].get(x, y),
                    8 => rust_psycho1.uhf[0].get(x, y),
                    9 => rust_psycho1.uhf[1].get(x, y),
                    _ => 0.0,
                };
                let v2 = match i {
                    0 => rust_psycho2.lf.plane(0).get(x, y),
                    1 => rust_psycho2.lf.plane(1).get(x, y),
                    2 => rust_psycho2.lf.plane(2).get(x, y),
                    3 => rust_psycho2.mf.plane(0).get(x, y),
                    4 => rust_psycho2.mf.plane(1).get(x, y),
                    5 => rust_psycho2.mf.plane(2).get(x, y),
                    6 => rust_psycho2.hf[0].get(x, y),
                    7 => rust_psycho2.hf[1].get(x, y),
                    8 => rust_psycho2.uhf[0].get(x, y),
                    9 => rust_psycho2.uhf[1].get(x, y),
                    _ => 0.0,
                };
                diff_vals.push((v1 - v2).abs());
            }
        }
        let (min, max, mean) = stats(&diff_vals);
        if max > 0.001 {
            println!(
                "    {}: min={:.6} max={:.6} mean={:.6}",
                band_name, min, max, mean
            );
        }
    }

    // Final score comparison
    println!("\n  Final scores:");
    println!("    C++ score: {:.6}", cpp_score);

    // Check diffmap stats
    let (c_min, c_max, c_mean) = stats(&cpp_diffmap);
    println!(
        "\n  C++ diffmap: min={:.6} max={:.6} mean={:.6}",
        c_min, c_max, c_mean
    );
}

// Diagnostic test for width-dependent divergence
#[test]
fn test_width_dependent_frequency_divergence() {
    println!("\n=== Width-Dependent Frequency Divergence ===\n");

    let height = 24usize;
    let intensity_target = 80.0f32;

    // Test widths: 22 (mod4=2, fails) vs 24 (mod4=0, passes)
    for width in [22, 24] {
        println!("  Width {} (mod4={}):", width, width % 4);

        // Create edge pattern (vertical edge in middle)
        let srgb: Vec<u8> = (0..height)
            .flat_map(|_y| {
                (0..width).flat_map(|x| {
                    let v = if x < width / 2 { 50u8 } else { 200u8 };
                    [v, v, v]
                })
            })
            .collect();

        let linear = srgb_to_linear_rust(&srgb);
        let rust_xyb = srgb_to_xyb_butteraugli(&srgb, width, height, intensity_target);
        let rust_psycho = separate_frequencies(&rust_xyb);

        let mut cpp_hf_x = vec![0.0f32; width * height];
        let mut cpp_hf_y = vec![0.0f32; width * height];
        let mut cpp_uhf_x = vec![0.0f32; width * height];
        let mut cpp_uhf_y = vec![0.0f32; width * height];

        let result = unsafe {
            butteraugli_separate_frequencies(
                linear.as_ptr(),
                width,
                height,
                intensity_target,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                cpp_hf_x.as_mut_ptr(),
                cpp_hf_y.as_mut_ptr(),
                cpp_uhf_x.as_mut_ptr(),
                cpp_uhf_y.as_mut_ptr(),
            )
        };

        if result != BUTTERAUGLI_OK {
            println!("    C++ separate_frequencies failed with code {}", result);
            continue;
        }

        // Compare HF and UHF bands
        for (band_name, rust_data, cpp_data) in [
            ("HF_X", &rust_psycho.hf[0], &cpp_hf_x),
            ("HF_Y", &rust_psycho.hf[1], &cpp_hf_y),
            ("UHF_X", &rust_psycho.uhf[0], &cpp_uhf_x),
            ("UHF_Y", &rust_psycho.uhf[1], &cpp_uhf_y),
        ] {
            let mut rust_flat = Vec::with_capacity(width * height);
            for y in 0..height {
                for x in 0..width {
                    rust_flat.push(rust_data.get(x, y));
                }
            }

            let max_diff = max_abs_diff(&rust_flat, cpp_data);
            let mean_diff = mean_abs_diff(&rust_flat, cpp_data);
            let (_, _, r_mean) = stats(&rust_flat);
            let (_, _, c_mean) = stats(cpp_data);
            let rel_diff = if c_mean.abs() > 0.001 {
                mean_diff / c_mean.abs() * 100.0
            } else {
                mean_diff * 100.0
            };

            println!(
                "    {}: max={:.6} mean={:.6} ({:.1}%) R_mean={:.6} C_mean={:.6}",
                band_name, max_diff, mean_diff, rel_diff, r_mean, c_mean
            );
        }
    }
}

// Test Malta filter diff map for different widths
#[test]
fn test_malta_diff_map_width_dependent() {
    use butteraugli::malta::malta_diff_map;

    println!("\n=== Malta DiffMap Width-Dependent Test ===\n");

    let height = 24usize;

    for width in [22, 24] {
        println!("  Width {} (mod4={}):", width, width % 4);

        // Create two test images: uniform vs with edge
        let img0 = butteraugli::image::ImageF::filled(width, height, 0.5);
        let mut img1 = butteraugli::image::ImageF::filled(width, height, 0.5);

        // Add edge to img1
        for y in 0..height {
            for x in 0..width {
                if x > width / 2 {
                    img1.set(x, y, 0.7);
                }
            }
        }

        // Compute Malta diff maps
        let diff_hf = malta_diff_map(&img0, &img1, 1.0, 1.0, 1.0, true); // LF variant
        let diff_uhf = malta_diff_map(&img0, &img1, 1.0, 1.0, 1.0, false); // HF variant

        // Compute statistics
        let mut sum_hf = 0.0f32;
        let mut max_hf = 0.0f32;
        let mut sum_uhf = 0.0f32;
        let mut max_uhf = 0.0f32;

        for y in 0..height {
            for x in 0..width {
                let v_hf = diff_hf.get(x, y);
                let v_uhf = diff_uhf.get(x, y);
                sum_hf += v_hf;
                sum_uhf += v_uhf;
                if v_hf > max_hf {
                    max_hf = v_hf;
                }
                if v_uhf > max_uhf {
                    max_uhf = v_uhf;
                }
            }
        }

        let mean_hf = sum_hf / (width * height) as f32;
        let mean_uhf = sum_uhf / (width * height) as f32;

        println!(
            "    Malta LF:  sum={:.6} mean={:.8} max={:.6}",
            sum_hf, mean_hf, max_hf
        );
        println!(
            "    Malta HF:  sum={:.6} mean={:.8} max={:.6}",
            sum_uhf, mean_uhf, max_uhf
        );
    }
}

// Test diffmap comparison between Rust and C++ for different widths
#[test]
fn test_diffmap_comparison_width_dependent() {
    println!("\n=== Diffmap Comparison Width-Dependent ===\n");

    let height = 24usize;
    let intensity_target = 80.0f32;

    for width in [22, 24] {
        println!("  Width {} (mod4={}):", width, width % 4);

        // Create test images: edge pattern
        let srgb1: Vec<u8> = (0..height)
            .flat_map(|_y| {
                (0..width).flat_map(|x| {
                    let v = if x < width / 2 { 50u8 } else { 200u8 };
                    [v, v, v]
                })
            })
            .collect();

        // Second image: blurred/shifted edge
        let srgb2: Vec<u8> = srgb1.iter().map(|&v| v.saturating_add(10)).collect();

        let linear1 = srgb_to_linear_rust(&srgb1);
        let linear2 = srgb_to_linear_rust(&srgb2);

        // C++ diffmap
        let mut cpp_score = 0.0f64;
        let mut cpp_diffmap = vec![0.0f32; width * height];
        let result = unsafe {
            butteraugli_compare_full(
                linear1.as_ptr(),
                linear2.as_ptr(),
                width,
                height,
                1.0,
                1.0,
                intensity_target,
                &mut cpp_score,
                cpp_diffmap.as_mut_ptr(),
            )
        };

        if result != BUTTERAUGLI_OK {
            println!("    C++ failed with code {}", result);
            continue;
        }

        // Rust diffmap
        let params = butteraugli::ButteraugliParams::default();
        let rust_result = butteraugli::compute_butteraugli(&srgb1, &srgb2, width, height, &params)
            .expect("Rust butteraugli failed");
        let rust_score = rust_result.score;

        // Compare diffmaps
        if let Some(ref rust_diffmap) = rust_result.diffmap {
            let mut max_diff = 0.0f32;
            let mut sum_rust = 0.0f32;
            let mut sum_cpp = 0.0f32;
            let mut max_rust = 0.0f32;
            let mut max_cpp = 0.0f32;

            for y in 0..height {
                for x in 0..width {
                    let rv = rust_diffmap.get(x, y);
                    let cv = cpp_diffmap[y * width + x];
                    sum_rust += rv;
                    sum_cpp += cv;
                    if rv > max_rust {
                        max_rust = rv;
                    }
                    if cv > max_cpp {
                        max_cpp = cv;
                    }
                    let diff = (rv - cv).abs();
                    if diff > max_diff {
                        max_diff = diff;
                    }
                }
            }

            let mean_rust = sum_rust / (width * height) as f32;
            let mean_cpp = sum_cpp / (width * height) as f32;

            let rel_diff = ((rust_score as f64 - cpp_score) / cpp_score * 100.0).abs();
            println!(
                "    Score: Rust={:.6} C++={:.6} diff={:.1}%",
                rust_score, cpp_score, rel_diff
            );
            println!("    Diffmap sum: Rust={:.4} C++={:.4}", sum_rust, sum_cpp);
            println!(
                "    Diffmap mean: Rust={:.6} C++={:.6}",
                mean_rust, mean_cpp
            );
            println!("    Diffmap max: Rust={:.6} C++={:.6}", max_rust, max_cpp);
            println!("    Diffmap max_diff={:.6}", max_diff);
        }
    }
}

// Test with embedded pattern (matches failing test)
#[test]
fn test_embedded_pattern_diffmap_comparison() {
    println!("\n=== Embedded Pattern Diffmap Comparison ===\n");

    let height = 24usize;
    let intensity_target = 80.0f32;

    // Create a fixed 16x16 pattern with edge
    fn create_pattern_16x16() -> Vec<u8> {
        let mut data = vec![0u8; 16 * 16 * 3];
        for y in 0..16 {
            for x in 0..16 {
                let val = if x < 8 { 50 } else { 200 };
                data[(y * 16 + x) * 3] = val;
                data[(y * 16 + x) * 3 + 1] = val;
                data[(y * 16 + x) * 3 + 2] = val;
            }
        }
        data
    }

    fn apply_blur_simple(img: &[u8], width: usize, height: usize) -> Vec<u8> {
        // Simple box blur
        let mut out = img.to_vec();
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                for c in 0..3 {
                    let mut sum = 0u32;
                    for dy in 0..3 {
                        for dx in 0..3 {
                            sum += img[((y + dy - 1) * width + (x + dx - 1)) * 3 + c] as u32;
                        }
                    }
                    out[(y * width + x) * 3 + c] = (sum / 9) as u8;
                }
            }
        }
        out
    }

    fn embed_in_size(pattern: &[u8], total_width: usize, total_height: usize) -> Vec<u8> {
        let mut data = vec![128u8; total_width * total_height * 3];
        let ox = (total_width - 16) / 2;
        let oy = (total_height - 16) / 2;
        for y in 0..16 {
            for x in 0..16 {
                let src_idx = (y * 16 + x) * 3;
                let dst_idx = ((oy + y) * total_width + (ox + x)) * 3;
                data[dst_idx] = pattern[src_idx];
                data[dst_idx + 1] = pattern[src_idx + 1];
                data[dst_idx + 2] = pattern[src_idx + 2];
            }
        }
        data
    }

    let pattern = create_pattern_16x16();
    let blurred_pattern = apply_blur_simple(&pattern, 16, 16);

    for width in [22, 24] {
        println!("  Width {} (mod4={}):", width, width % 4);

        let img_a = embed_in_size(&pattern, width, height);
        let img_b = embed_in_size(&blurred_pattern, width, height);

        let linear1 = srgb_to_linear_rust(&img_a);
        let linear2 = srgb_to_linear_rust(&img_b);

        // C++ score
        let mut cpp_score = 0.0f64;
        let result = unsafe {
            butteraugli_compare_full(
                linear1.as_ptr(),
                linear2.as_ptr(),
                width,
                height,
                1.0,
                1.0,
                intensity_target,
                &mut cpp_score,
                std::ptr::null_mut(),
            )
        };

        if result != BUTTERAUGLI_OK {
            println!("    C++ failed with code {}", result);
            continue;
        }

        // Rust score
        let params = butteraugli::ButteraugliParams::default();
        let rust_result = butteraugli::compute_butteraugli(&img_a, &img_b, width, height, &params)
            .expect("Rust butteraugli failed");

        let rel_diff = ((rust_result.score as f64 - cpp_score) / cpp_score * 100.0).abs();
        println!(
            "    Score: Rust={:.6} C++={:.6} diff={:.1}%",
            rust_result.score, cpp_score, rel_diff
        );
    }
}

// Test frequency bands for embedded pattern
#[test]
fn test_embedded_pattern_frequency_bands() {
    println!("\n=== Embedded Pattern Frequency Bands ===\n");

    let height = 24usize;
    let intensity_target = 80.0f32;

    fn create_pattern_16x16() -> Vec<u8> {
        let mut data = vec![0u8; 16 * 16 * 3];
        for y in 0..16 {
            for x in 0..16 {
                let val = if x < 8 { 50 } else { 200 };
                data[(y * 16 + x) * 3] = val;
                data[(y * 16 + x) * 3 + 1] = val;
                data[(y * 16 + x) * 3 + 2] = val;
            }
        }
        data
    }

    fn embed_in_size(pattern: &[u8], total_width: usize, total_height: usize) -> Vec<u8> {
        let mut data = vec![128u8; total_width * total_height * 3];
        let ox = (total_width - 16) / 2;
        let oy = (total_height - 16) / 2;
        for y in 0..16 {
            for x in 0..16 {
                let src_idx = (y * 16 + x) * 3;
                let dst_idx = ((oy + y) * total_width + (ox + x)) * 3;
                data[dst_idx] = pattern[src_idx];
                data[dst_idx + 1] = pattern[src_idx + 1];
                data[dst_idx + 2] = pattern[src_idx + 2];
            }
        }
        data
    }

    let pattern = create_pattern_16x16();

    for width in [22, 24] {
        println!(
            "  Width {} (mod4={}, border={} pixels):",
            width,
            width % 4,
            (width - 16) / 2
        );

        let srgb = embed_in_size(&pattern, width, height);
        let linear = srgb_to_linear_rust(&srgb);

        // Rust frequency separation
        let rust_xyb = srgb_to_xyb_butteraugli(&srgb, width, height, intensity_target);
        let rust_psycho = separate_frequencies(&rust_xyb);

        // C++ frequency separation
        let mut cpp_hf_x = vec![0.0f32; width * height];
        let mut cpp_hf_y = vec![0.0f32; width * height];
        let mut cpp_uhf_x = vec![0.0f32; width * height];
        let mut cpp_uhf_y = vec![0.0f32; width * height];

        let result = unsafe {
            butteraugli_separate_frequencies(
                linear.as_ptr(),
                width,
                height,
                intensity_target,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                cpp_hf_x.as_mut_ptr(),
                cpp_hf_y.as_mut_ptr(),
                cpp_uhf_x.as_mut_ptr(),
                cpp_uhf_y.as_mut_ptr(),
            )
        };

        if result != BUTTERAUGLI_OK {
            println!("    C++ failed");
            continue;
        }

        // Compare HF and UHF bands
        for (band_name, rust_data, cpp_data) in [
            ("HF_X", &rust_psycho.hf[0], &cpp_hf_x),
            ("HF_Y", &rust_psycho.hf[1], &cpp_hf_y),
            ("UHF_X", &rust_psycho.uhf[0], &cpp_uhf_x),
            ("UHF_Y", &rust_psycho.uhf[1], &cpp_uhf_y),
        ] {
            let mut rust_sum = 0.0f32;
            let mut cpp_sum = 0.0f32;
            let mut max_diff = 0.0f32;

            for y in 0..height {
                for x in 0..width {
                    let rv = rust_data.get(x, y);
                    let cv = cpp_data[y * width + x];
                    rust_sum += rv.abs();
                    cpp_sum += cv.abs();
                    if (rv - cv).abs() > max_diff {
                        max_diff = (rv - cv).abs();
                    }
                }
            }

            println!(
                "    {}: Rust_sum={:.4} C++_sum={:.4} max_diff={:.6}",
                band_name, rust_sum, cpp_sum, max_diff
            );
        }
    }
}

// ============================================================================
// Direct blur comparison for border cases
// ============================================================================

#[test]
fn test_blur_border_handling_sizes() {
    println!("\n=== Blur Border Handling at Different Widths ===\n");

    let height = 24usize;

    // Create simple edge pattern: left half = 0.0, right half = 1.0
    fn create_edge_pattern(width: usize, height: usize) -> Vec<f32> {
        let mut data = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                data[y * width + x] = if x >= width / 2 { 1.0 } else { 0.0 };
            }
        }
        data
    }

    // Test with sigmas used in frequency separation
    let sigmas = [
        (1.564f32, "UHF sigma"),
        (3.225f32, "HF sigma"),
        (7.156f32, "LF sigma"),
    ];

    for width in [18, 20, 22, 24, 26] {
        println!("  Width {} (mod4={}):", width, width % 4);

        let input = create_edge_pattern(width, height);
        let input_img = ImageF::from_vec(input.clone(), width, height);

        for (sigma, name) in &sigmas {
            // Rust blur
            let rust_blurred = gaussian_blur(&input_img, *sigma);
            let mut rust_flat = Vec::with_capacity(width * height);
            for y in 0..height {
                for x in 0..width {
                    rust_flat.push(rust_blurred.get(x, y));
                }
            }

            // C++ blur
            let mut cpp_blurred = vec![0.0f32; width * height];
            let result = unsafe {
                butteraugli_blur(
                    input.as_ptr(),
                    width,
                    height,
                    *sigma,
                    cpp_blurred.as_mut_ptr(),
                )
            };

            if result != BUTTERAUGLI_OK {
                println!("    {}: C++ blur failed", name);
                continue;
            }

            let max_diff = max_abs_diff(&rust_flat, &cpp_blurred);
            let mean_diff = mean_abs_diff(&rust_flat, &cpp_blurred);

            // Check border pixels specifically
            let mut border_max_diff = 0.0f32;
            let kernel_radius = (2.25 * sigma).max(1.0) as usize;
            for y in 0..height {
                for x in 0..width {
                    let is_border = x < kernel_radius
                        || x >= width - kernel_radius
                        || y < kernel_radius
                        || y >= height - kernel_radius;
                    if is_border {
                        let diff = (rust_flat[y * width + x] - cpp_blurred[y * width + x]).abs();
                        if diff > border_max_diff {
                            border_max_diff = diff;
                        }
                    }
                }
            }

            println!(
                "    {}: max_diff={:.6} mean_diff={:.6} border_max={:.6}",
                name, max_diff, mean_diff, border_max_diff
            );
        }
    }
}

// ============================================================================
// XYB conversion comparison for different widths
// ============================================================================

#[test]
fn test_xyb_conversion_width_dependent() {
    println!("\n=== XYB Conversion Width Dependence ===\n");

    let height = 24usize;
    let intensity_target = 80.0f32;

    fn create_pattern_16x16() -> Vec<u8> {
        let mut data = vec![0u8; 16 * 16 * 3];
        for y in 0..16 {
            for x in 0..16 {
                let val = if x < 8 { 50 } else { 200 };
                data[(y * 16 + x) * 3] = val;
                data[(y * 16 + x) * 3 + 1] = val;
                data[(y * 16 + x) * 3 + 2] = val;
            }
        }
        data
    }

    fn embed_in_size(pattern: &[u8], total_width: usize, total_height: usize) -> Vec<u8> {
        let mut data = vec![128u8; total_width * total_height * 3];
        let ox = (total_width - 16) / 2;
        let oy = (total_height - 16) / 2;
        for y in 0..16 {
            for x in 0..16 {
                let src_idx = (y * 16 + x) * 3;
                let dst_idx = ((oy + y) * total_width + (ox + x)) * 3;
                data[dst_idx] = pattern[src_idx];
                data[dst_idx + 1] = pattern[src_idx + 1];
                data[dst_idx + 2] = pattern[src_idx + 2];
            }
        }
        data
    }

    let pattern = create_pattern_16x16();

    for width in [22, 24] {
        println!(
            "  Width {} (mod4={}, border={} pixels):",
            width,
            width % 4,
            (width - 16) / 2
        );

        let srgb = embed_in_size(&pattern, width, height);
        let linear = srgb_to_linear_rust(&srgb);

        // Rust XYB
        let rust_xyb = srgb_to_xyb_butteraugli(&srgb, width, height, intensity_target);

        // C++ XYB (SIMPLIFIED wrapper: no blur, no bias, just mix+cbrt)
        let mut cpp_xyb = vec![0.0f32; width * height * 3];
        let result = unsafe {
            butteraugli_opsin_dynamics(
                linear.as_ptr(),
                width,
                height,
                intensity_target,
                cpp_xyb.as_mut_ptr(),
            )
        };

        if result != BUTTERAUGLI_OK {
            println!("    C++ opsin_dynamics failed");
            continue;
        }

        // Compare XYB channels
        for (ch, ch_name) in [(0, "X"), (1, "Y"), (2, "B")] {
            let mut rust_sum = 0.0f32;
            let mut cpp_sum = 0.0f32;
            let mut max_diff = 0.0f32;
            let mut border_max_diff = 0.0f32;
            let border = (width - 16) / 2;

            for y in 0..height {
                for x in 0..width {
                    let rv = rust_xyb.plane(ch).get(x, y);
                    let cv = cpp_xyb[(y * width + x) * 3 + ch];
                    rust_sum += rv.abs();
                    cpp_sum += cv.abs();
                    let diff = (rv - cv).abs();
                    if diff > max_diff {
                        max_diff = diff;
                    }

                    // Check border pixels specifically
                    let is_border =
                        x < border || x >= width - border || y < border || y >= height - border;
                    if is_border && diff > border_max_diff {
                        border_max_diff = diff;
                    }
                }
            }

            println!(
                "    {}: max_diff={:.6} border_max={:.6} (Rust_sum={:.2} C++_sum={:.2})",
                ch_name, max_diff, border_max_diff, rust_sum, cpp_sum
            );
        }
    }
}

// ============================================================================
// XYB blur comparison with embedded pattern
// ============================================================================

#[test]
fn test_xyb_blur_embedded_pattern() {
    println!("\n=== XYB Blur (sigma=1.2) with Embedded Pattern ===\n");

    let height = 24usize;

    fn create_pattern_16x16() -> Vec<u8> {
        let mut data = vec![0u8; 16 * 16 * 3];
        for y in 0..16 {
            for x in 0..16 {
                let val = if x < 8 { 50 } else { 200 };
                data[(y * 16 + x) * 3] = val;
                data[(y * 16 + x) * 3 + 1] = val;
                data[(y * 16 + x) * 3 + 2] = val;
            }
        }
        data
    }

    fn embed_in_size(pattern: &[u8], total_width: usize, total_height: usize) -> Vec<u8> {
        let mut data = vec![128u8; total_width * total_height * 3];
        let ox = (total_width - 16) / 2;
        let oy = (total_height - 16) / 2;
        for y in 0..16 {
            for x in 0..16 {
                let src_idx = (y * 16 + x) * 3;
                let dst_idx = ((oy + y) * total_width + (ox + x)) * 3;
                data[dst_idx] = pattern[src_idx];
                data[dst_idx + 1] = pattern[src_idx + 1];
                data[dst_idx + 2] = pattern[src_idx + 2];
            }
        }
        data
    }

    let pattern = create_pattern_16x16();
    let sigma_xyb = 1.2f32;

    for width in [22, 24] {
        println!("  Width {} (mod4={}):", width, width % 4);

        let srgb = embed_in_size(&pattern, width, height);
        let linear = srgb_to_linear_rust(&srgb);

        // Convert to single-channel linear Y (use green as proxy)
        let linear_y: Vec<f32> = (0..width * height).map(|i| linear[i * 3 + 1]).collect();

        // Rust blur with sigma=1.2 - test both with and without padding
        let input_no_pad = ImageF::from_vec(linear_y.clone(), width, height);
        let rust_blurred_no_pad = gaussian_blur(&input_no_pad, sigma_xyb);

        // Also try with padded input (ImageF::new has stride alignment)
        let mut input_padded = ImageF::new(width, height);
        for y in 0..height {
            for x in 0..width {
                input_padded.set(x, y, linear_y[y * width + x]);
            }
        }
        let rust_blurred_padded = gaussian_blur(&input_padded, sigma_xyb);

        let mut rust_flat_no_pad = Vec::with_capacity(width * height);
        let mut rust_flat_padded = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                rust_flat_no_pad.push(rust_blurred_no_pad.get(x, y));
                rust_flat_padded.push(rust_blurred_padded.get(x, y));
            }
        }

        // C++ blur with sigma=1.2
        let mut cpp_blurred = vec![0.0f32; width * height];
        let result = unsafe {
            butteraugli_blur(
                linear_y.as_ptr(),
                width,
                height,
                sigma_xyb,
                cpp_blurred.as_mut_ptr(),
            )
        };

        if result != BUTTERAUGLI_OK {
            println!("    C++ blur failed");
            continue;
        }

        let max_diff_no_pad = max_abs_diff(&rust_flat_no_pad, &cpp_blurred);
        let max_diff_padded = max_abs_diff(&rust_flat_padded, &cpp_blurred);
        let rust_stride_diff = max_abs_diff(&rust_flat_no_pad, &rust_flat_padded);

        println!("    Blur sigma={:.1}:", sigma_xyb);
        println!("      no_pad vs C++: max_diff={:.6}", max_diff_no_pad);
        println!("      padded vs C++: max_diff={:.6}", max_diff_padded);
        println!(
            "      Rust no_pad vs padded: max_diff={:.6}",
            rust_stride_diff
        );

        // Check specific edge positions where max diff occurs
        if max_diff_padded > 0.0001 {
            // Find the position of max diff
            let mut max_diff_pos = (0, 0);
            let mut max_diff_val = 0.0f32;
            for y in 0..height {
                for x in 0..width {
                    let diff = (rust_flat_padded[y * width + x] - cpp_blurred[y * width + x]).abs();
                    if diff > max_diff_val {
                        max_diff_val = diff;
                        max_diff_pos = (x, y);
                    }
                }
            }
            let (mx, my) = max_diff_pos;
            println!(
                "      Max diff at ({},{}): Rust={:.6} C++={:.6}",
                mx,
                my,
                rust_flat_padded[my * width + mx],
                cpp_blurred[my * width + mx]
            );
        }
    }
}

// ============================================================================
// Full frequency band comparison to find where divergence starts
// ============================================================================

#[test]
fn test_all_frequency_bands_divergence() {
    println!("\n=== All Frequency Bands Divergence Analysis ===\n");

    let height = 24usize;
    let intensity_target = 80.0f32;

    fn create_pattern_16x16() -> Vec<u8> {
        let mut data = vec![0u8; 16 * 16 * 3];
        for y in 0..16 {
            for x in 0..16 {
                let val = if x < 8 { 50 } else { 200 };
                data[(y * 16 + x) * 3] = val;
                data[(y * 16 + x) * 3 + 1] = val;
                data[(y * 16 + x) * 3 + 2] = val;
            }
        }
        data
    }

    fn embed_in_size(pattern: &[u8], total_width: usize, total_height: usize) -> Vec<u8> {
        let mut data = vec![128u8; total_width * total_height * 3];
        let ox = (total_width - 16) / 2;
        let oy = (total_height - 16) / 2;
        for y in 0..16 {
            for x in 0..16 {
                let src_idx = (y * 16 + x) * 3;
                let dst_idx = ((oy + y) * total_width + (ox + x)) * 3;
                data[dst_idx] = pattern[src_idx];
                data[dst_idx + 1] = pattern[src_idx + 1];
                data[dst_idx + 2] = pattern[src_idx + 2];
            }
        }
        data
    }

    let pattern = create_pattern_16x16();

    for width in [22, 24] {
        println!("  Width {} (mod4={}):", width, width % 4);

        let srgb = embed_in_size(&pattern, width, height);
        let linear = srgb_to_linear_rust(&srgb);

        // Rust frequency separation
        let rust_xyb = srgb_to_xyb_butteraugli(&srgb, width, height, intensity_target);
        let rust_psycho = separate_frequencies(&rust_xyb);

        // C++ frequency separation - get ALL bands
        let mut cpp_lf_x = vec![0.0f32; width * height];
        let mut cpp_lf_y = vec![0.0f32; width * height];
        let mut cpp_lf_b = vec![0.0f32; width * height];
        let mut cpp_mf_x = vec![0.0f32; width * height];
        let mut cpp_mf_y = vec![0.0f32; width * height];
        let mut cpp_mf_b = vec![0.0f32; width * height];
        let mut cpp_hf_x = vec![0.0f32; width * height];
        let mut cpp_hf_y = vec![0.0f32; width * height];
        let mut cpp_uhf_x = vec![0.0f32; width * height];
        let mut cpp_uhf_y = vec![0.0f32; width * height];

        let result = unsafe {
            butteraugli_separate_frequencies(
                linear.as_ptr(),
                width,
                height,
                intensity_target,
                cpp_lf_x.as_mut_ptr(),
                cpp_lf_y.as_mut_ptr(),
                cpp_lf_b.as_mut_ptr(),
                cpp_mf_x.as_mut_ptr(),
                cpp_mf_y.as_mut_ptr(),
                cpp_mf_b.as_mut_ptr(),
                cpp_hf_x.as_mut_ptr(),
                cpp_hf_y.as_mut_ptr(),
                cpp_uhf_x.as_mut_ptr(),
                cpp_uhf_y.as_mut_ptr(),
            )
        };

        if result != BUTTERAUGLI_OK {
            println!("    C++ failed");
            continue;
        }

        // Compare ALL bands in order of processing: LF -> MF -> HF -> UHF
        let bands: &[(&str, &ImageF, &[f32])] = &[
            ("LF_X", rust_psycho.lf.plane(0), &cpp_lf_x),
            ("LF_Y", rust_psycho.lf.plane(1), &cpp_lf_y),
            ("LF_B", rust_psycho.lf.plane(2), &cpp_lf_b),
            ("MF_X", rust_psycho.mf.plane(0), &cpp_mf_x),
            ("MF_Y", rust_psycho.mf.plane(1), &cpp_mf_y),
            ("MF_B", rust_psycho.mf.plane(2), &cpp_mf_b),
            ("HF_X", &rust_psycho.hf[0], &cpp_hf_x),
            ("HF_Y", &rust_psycho.hf[1], &cpp_hf_y),
            ("UHF_X", &rust_psycho.uhf[0], &cpp_uhf_x),
            ("UHF_Y", &rust_psycho.uhf[1], &cpp_uhf_y),
        ];

        for (band_name, rust_plane, cpp_data) in bands {
            let mut max_diff = 0.0f32;
            let mut max_diff_x = 0;
            let mut max_diff_y = 0;
            let mut rust_sum = 0.0f32;
            let mut cpp_sum = 0.0f32;

            for y in 0..height {
                for x in 0..width {
                    let rv = rust_plane.get(x, y);
                    let cv = cpp_data[y * width + x];
                    rust_sum += rv.abs();
                    cpp_sum += cv.abs();
                    let diff = (rv - cv).abs();
                    if diff > max_diff {
                        max_diff = diff;
                        max_diff_x = x;
                        max_diff_y = y;
                    }
                }
            }

            let rel_diff = if cpp_sum > 0.001 {
                (rust_sum - cpp_sum).abs() / cpp_sum * 100.0
            } else {
                0.0
            };

            if max_diff > 0.0001 {
                println!(
                    "    {}: max_diff={:.6} at ({},{}) sum_diff={:.2}%",
                    band_name, max_diff, max_diff_x, max_diff_y, rel_diff
                );
            } else {
                println!("    {}: MATCH (max_diff={:.6})", band_name, max_diff);
            }
        }
    }
}
