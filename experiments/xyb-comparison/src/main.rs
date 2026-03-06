//! XYB Color Space Comparison Experiment
//!
//! Compares XYB conversions across implementations:
//! 1. **JPEG XL Codec XYB** (via `yuvxyb` crate) — cbrt-based, used by jpegli/jxl encoders
//! 2. **Butteraugli Internal Opsin** (via `butteraugli` crate) — log-gamma, sensitivity adaptation
//! 3. **Archived google/butteraugli** — inline reference from the discontinued standalone repo
//!
//! All three are commonly called "XYB" but produce very different values.

use butteraugli::opsin;

fn main() {
    // Test colors: linear RGB in [0, 1]
    let test_colors: &[(&str, [f32; 3])] = &[
        ("black", [0.0, 0.0, 0.0]),
        ("white", [1.0, 1.0, 1.0]),
        ("mid gray (linear 0.5)", [0.5, 0.5, 0.5]),
        ("mid gray (sRGB ~0.216)", [0.2158605, 0.2158605, 0.2158605]),
        ("pure red", [1.0, 0.0, 0.0]),
        ("pure green", [0.0, 1.0, 0.0]),
        ("pure blue", [0.0, 0.0, 1.0]),
        ("skin tone", [0.73, 0.50, 0.38]),
        ("sky blue", [0.25, 0.55, 0.85]),
        ("dark (1%)", [0.01, 0.01, 0.01]),
        ("bright (99%)", [0.99, 0.99, 0.99]),
        ("saturated yellow", [1.0, 1.0, 0.0]),
        ("saturated cyan", [0.0, 1.0, 1.0]),
        ("saturated magenta", [1.0, 0.0, 1.0]),
    ];

    println!("╔══════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                        XYB Color Space Comparison                               ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════════╣");
    println!("║ Impl 1: JPEG XL Codec XYB  (yuvxyb crate) — cbrt, 0.5*(L±M)                   ║");
    println!("║ Impl 2: Butteraugli Opsin   (butteraugli)  — log gamma, L±M, sensitivity adapt ║");
    println!("║ Impl 3: google/butteraugli  (archived)     — rational poly gamma, L±M          ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Run all test colors through all three implementations
    for &(name, rgb) in test_colors {
        let [r, g, b] = rgb;

        let codec = codec_xyb(r, g, b);
        let codec_yuvxyb = codec_xyb_via_yuvxyb(r, g, b);
        let ba = butteraugli_opsin_single_pixel(r, g, b, 80.0);
        let google = google_butteraugli_xyb(r, g, b);

        println!("── {} ── (linear RGB = [{:.3}, {:.3}, {:.3}])", name, r, g, b);
        println!(
            "  Codec XYB (inline):   X={:+10.6}  Y={:10.6}  B={:10.6}",
            codec.0, codec.1, codec.2
        );
        println!(
            "  Codec XYB (yuvxyb):   X={:+10.6}  Y={:10.6}  B={:10.6}",
            codec_yuvxyb.0, codec_yuvxyb.1, codec_yuvxyb.2
        );
        let yuvxyb_diff = max_abs_diff_3(codec, codec_yuvxyb);
        if yuvxyb_diff > 1e-7 {
            println!("    ^ yuvxyb vs inline max diff: {:.2e}", yuvxyb_diff);
        } else {
            println!("    ^ yuvxyb matches inline (diff < 1e-7)");
        }
        println!(
            "  Butteraugli opsin:    X={:+10.6}  Y={:10.6}  B={:10.6}  (no blur, single pixel)",
            ba.0, ba.1, ba.2
        );
        println!(
            "  google/butteraugli:   X={:+10.6}  Y={:10.6}  B={:10.6}",
            google.0, google.1, google.2
        );

        // Show deltas
        let codec_vs_ba = max_abs_diff_3(codec, ba);
        let codec_vs_google = max_abs_diff_3(codec, google);
        let ba_vs_google = max_abs_diff_3(ba, google);
        println!(
            "  Deltas:  codec↔ba={:.4}  codec↔google={:.4}  ba↔google={:.4}",
            codec_vs_ba, codec_vs_google, ba_vs_google
        );
        println!();
    }

    // Also print the matrix/bias/transfer differences
    println!("═══════════════════════════════════════════════════════════════");
    println!("Matrix Comparison (row 0 / row 1 / row 2):");
    println!();
    println!("  JPEG XL Codec:");
    println!("    [0.30000, 0.62200, 0.07800]  sum=1.000");
    println!("    [0.23000, 0.69200, 0.07800]  sum=1.000");
    println!("    [0.24342, 0.20477, 0.55181]  sum=1.000");
    println!("    bias = [0.00379, 0.00379, 0.00379]");
    println!();
    println!("  libjxl Butteraugli:");
    println!("    [0.29957, 0.63373, 0.07771]  sum={:.3}", 0.29957 + 0.63373 + 0.07771);
    println!("    [0.22159, 0.69391, 0.09873]  sum={:.3}", 0.22159 + 0.69391 + 0.09873);
    println!("    [0.02000, 0.02000, 0.20480]  sum={:.3}", 0.02 + 0.02 + 0.2048);
    println!("    bias = [1.75575, 1.75575, 12.22645]");
    println!();
    println!("  google/butteraugli (archived):");
    println!("    [0.25446, 0.48824, 0.06353]  sum={:.3}", 0.25446 + 0.48824 + 0.06353);
    println!("    [0.19521, 0.56802, 0.08608]  sum={:.3}", 0.19521 + 0.56802 + 0.08608);
    println!("    [0.07375, 0.06142, 0.24417]  sum={:.3}", 0.07375 + 0.06142 + 0.24417);
    println!("    bias = [1.01681, 1.15101, 1.20482]");
    println!();
    println!("  Transfer functions:");
    println!("    Codec:              cbrt(x)");
    println!("    libjxl butteraugli: 13.34 * ln(x + 9.971) - 23.16  (log-based)");
    println!("    google/butteraugli: rational polynomial (degree 5/5)");
    println!();
    println!("  X,Y formulas:");
    println!("    Codec:              X = 0.5*(L-M), Y = 0.5*(L+M)");
    println!("    libjxl butteraugli: X = L-M,       Y = L+M");
    println!("    google/butteraugli: X = L-M,       Y = L+M");
}

fn max_abs_diff_3(a: (f32, f32, f32), b: (f32, f32, f32)) -> f32 {
    (a.0 - b.0)
        .abs()
        .max((a.1 - b.1).abs())
        .max((a.2 - b.2).abs())
}

// ============================================================================
// Implementation 1: JPEG XL Codec XYB (inline reference)
// ============================================================================

/// JPEG XL codec XYB — the color space used by jpegli, jxl-rs, yuvxyb, libjxl encoder/decoder.
/// Matrix from libjxl opsin_params.h, transfer = cbrt, X = 0.5*(L-M), Y = 0.5*(L+M).
fn codec_xyb(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // OpsinAbsorbance matrix (rows sum to 1.0)
    const M: [[f32; 3]; 3] = [
        [0.30, 0.622, 0.078],
        [0.23, 0.692, 0.078],
        [0.243_422_69, 0.204_767_45, 0.551_809_86],
    ];
    const BIAS: f32 = 0.003_793_073_4;
    let neg_cbrt_bias: f32 = -BIAS.cbrt();

    // Step 1: matrix * RGB + bias
    let opsin0 = (M[0][0] * r + M[0][1] * g + M[0][2] * b + BIAS).max(0.0);
    let opsin1 = (M[1][0] * r + M[1][1] * g + M[1][2] * b + BIAS).max(0.0);
    let opsin2 = (M[2][0] * r + M[2][1] * g + M[2][2] * b + BIAS).max(0.0);

    // Step 2: cbrt + subtract cbrt(bias)
    let l = opsin0.cbrt() + neg_cbrt_bias;
    let m = opsin1.cbrt() + neg_cbrt_bias;
    let s = opsin2.cbrt() + neg_cbrt_bias;

    // Step 3: opponent transform with 0.5 scaling
    let x = 0.5 * (l - m);
    let y = 0.5 * (l + m);
    (x, y, s)
}

// ============================================================================
// Implementation 1b: JPEG XL Codec XYB via yuvxyb crate (cross-check)
// ============================================================================

fn codec_xyb_via_yuvxyb(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let lrgb = yuvxyb::LinearRgb::new(vec![[r, g, b]], 1, 1).unwrap();
    let xyb = yuvxyb::Xyb::from(lrgb);
    let p = xyb.data()[0];
    (p[0], p[1], p[2])
}

// ============================================================================
// Implementation 2: Butteraugli Internal Opsin (single pixel, no blur)
// ============================================================================

/// Butteraugli's OpsinDynamicsImage without the blur/sensitivity adaptation step.
/// This shows the raw opsin matrix + gamma, comparable to a single isolated pixel
/// (where blur = the pixel itself, so sensitivity = gamma(opsin)/opsin).
fn butteraugli_opsin_single_pixel(r: f32, g: f32, b: f32, intensity_target: f32) -> (f32, f32, f32) {
    let r = r * intensity_target;
    let g = g * intensity_target;
    let b = b * intensity_target;

    // For a single pixel with no neighbors, blur = original.
    // So pre_mixed = cur_mixed, and sensitivity = gamma(pre) / pre.
    // Final = cur_mixed * sensitivity = cur_mixed * gamma(cur_mixed) / cur_mixed = gamma(cur_mixed).
    // But we must clamp first.
    let (pre0, pre1, pre2) = opsin::opsin_absorbance(r, g, b, true);

    // Sensitivity = gamma(pre) / pre (both from blurred, which = original for single pixel)
    let min_val: f32 = 1e-4;
    let sens0 = (opsin::gamma(pre0) / pre0.max(min_val)).max(min_val);
    let sens1 = (opsin::gamma(pre1) / pre1.max(min_val)).max(min_val);
    let sens2 = (opsin::gamma(pre2) / pre2.max(min_val)).max(min_val);

    // Apply sensitivity to original (= blurred for single pixel)
    let (cur0_raw, cur1_raw, cur2_raw) = opsin::opsin_absorbance(r, g, b, false);
    let cur0 = (cur0_raw * sens0).max(1.755_748_4);
    let cur1 = (cur1_raw * sens1).max(1.755_748_4);
    let cur2 = (cur2_raw * sens2).max(12.226_455);

    // XYB: no 0.5 scaling
    let x = cur0 - cur1;
    let y = cur0 + cur1;
    (x, y, cur2)
}

// ============================================================================
// Implementation 3: Archived google/butteraugli (inline from discontinued repo)
// ============================================================================

/// XYB conversion from the archived google/butteraugli repository.
/// Different matrix, different biases, rational polynomial gamma.
/// This codebase is discontinued and diverged significantly from libjxl.
fn google_butteraugli_xyb(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // google/butteraugli OpsinAbsorbance matrix
    const M: [[f64; 3]; 3] = [
        [0.254462330846, 0.488238255095, 0.0635278003854],
        [0.195214015766, 0.568019861857, 0.0860755536007],
        [0.07374607900105684, 0.06142425304154509, 0.24416850520714256],
    ];
    const BIAS: [f64; 3] = [1.01681026909, 1.1510118369, 1.20481945273];

    // Scale by 255 (google/butteraugli expects [0, 255] input, no intensity_target)
    let r = r as f64 * 255.0;
    let g = g as f64 * 255.0;
    let b = b as f64 * 255.0;

    // For single pixel: blur = original, same as libjxl butteraugli logic
    let pre0 = (M[0][0] * r + M[0][1] * g + M[0][2] * b + BIAS[0]).max(BIAS[0]);
    let pre1 = (M[1][0] * r + M[1][1] * g + M[1][2] * b + BIAS[1]).max(BIAS[1]);
    let pre2 = (M[2][0] * r + M[2][1] * g + M[2][2] * b + BIAS[2]).max(BIAS[2]);

    let gamma0 = google_gamma_polynomial(pre0);
    let gamma1 = google_gamma_polynomial(pre1);
    let gamma2 = google_gamma_polynomial(pre2);

    let sens0 = gamma0 / pre0.max(1e-4);
    let sens1 = gamma1 / pre1.max(1e-4);
    let sens2 = gamma2 / pre2.max(1e-4);

    let cur0 = (M[0][0] * r + M[0][1] * g + M[0][2] * b + BIAS[0]) * sens0;
    let cur1 = (M[1][0] * r + M[1][1] * g + M[1][2] * b + BIAS[1]) * sens1;
    let cur2 = (M[2][0] * r + M[2][1] * g + M[2][2] * b + BIAS[2]) * sens2;

    let cur0 = cur0.max(BIAS[0]);
    let cur1 = cur1.max(BIAS[1]);
    let cur2 = cur2.max(BIAS[2]);

    // XYB: no 0.5 scaling (same as libjxl butteraugli)
    let x = (cur0 - cur1) as f32;
    let y = (cur0 + cur1) as f32;
    let b_out = cur2 as f32;
    (x, y, b_out)
}

/// Rational polynomial gamma from google/butteraugli.
/// This is GammaPolynomial from the archived repo (NOT the libjxl Gamma function).
///
/// Domain: [0.971783, 590.188894]
/// Chebyshev-like evaluation with explicit coefficients.
fn google_gamma_polynomial(v: f64) -> f64 {
    // Coefficients from google/butteraugli's GammaPolynomial
    // Numerator (degree 5)
    const N: [f64; 6] = [
        98.7821300963361,
        164.273222212631,
        92.948112871376,
        33.8165311212688,
        6.91626704983562,
        0.556380877028234,
    ];
    // Denominator (degree 5, monic-ish)
    const D: [f64; 6] = [
        1.0,
        1.64339473427892,
        0.89392405219969,
        0.298947051776379,
        0.0507146002577288,
        0.00226495093949756,
    ];

    // Domain mapping: [low, high] -> [-1, 1]
    const LOW: f64 = 0.971783;
    const HIGH: f64 = 590.188894;

    // Map v to [-1, 1] for Chebyshev-like evaluation
    let x = (v - LOW) / (HIGH - LOW) * 2.0 - 1.0;

    // Evaluate numerator and denominator using Horner's method
    let mut num = N[5];
    for i in (0..5).rev() {
        num = num * x + N[i];
    }
    let mut den = D[5];
    for i in (0..5).rev() {
        den = den * x + D[i];
    }

    num / den
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify yuvxyb crate matches our inline JPEG XL codec XYB implementation.
    #[test]
    fn yuvxyb_matches_inline_codec_xyb() {
        let colors: &[[f32; 3]] = &[
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.73, 0.50, 0.38],
            [0.25, 0.55, 0.85],
            [0.01, 0.01, 0.01],
            [0.99, 0.99, 0.99],
        ];

        for &[r, g, b] in colors {
            let inline = codec_xyb(r, g, b);
            let yuvxyb = codec_xyb_via_yuvxyb(r, g, b);
            let diff = max_abs_diff_3(inline, yuvxyb);
            assert!(
                diff < 1e-6,
                "yuvxyb diverges from inline codec XYB at RGB=[{r}, {g}, {b}]: diff={diff:.2e}"
            );
        }
    }

    /// Verify all three implementations produce fundamentally different results.
    /// This confirms they are truly different color spaces, not just rounding differences.
    #[test]
    fn three_implementations_are_fundamentally_different() {
        // Use white (the most "normal" color)
        let (r, g, b) = (1.0_f32, 1.0, 1.0);

        let codec = codec_xyb(r, g, b);
        let ba = butteraugli_opsin_single_pixel(r, g, b, 80.0);
        let google = google_butteraugli_xyb(r, g, b);

        // Codec Y < 1.0 (normalized for compression)
        assert!(codec.1 < 1.0, "Codec Y should be < 1.0, got {}", codec.1);

        // Butteraugli Y >> 1.0 (absolute photoreceptor response)
        assert!(ba.1 > 50.0, "Butteraugli Y should be >> 1.0, got {}", ba.1);

        // google/butteraugli Y >> butteraugli Y (different scaling)
        assert!(
            google.1 > ba.1,
            "google/butteraugli Y ({}) should exceed libjxl butteraugli Y ({})",
            google.1,
            ba.1
        );

        // Codec and butteraugli must differ by at least 100x on Y channel
        let ratio = ba.1 / codec.1;
        assert!(
            ratio > 100.0,
            "Butteraugli/Codec Y ratio should be > 100x, got {ratio:.1}"
        );
    }

    /// Verify codec XYB is zero for black (bias cancellation).
    #[test]
    fn codec_xyb_black_is_zero() {
        let (x, y, b) = codec_xyb(0.0, 0.0, 0.0);
        assert!(x.abs() < 1e-7, "X should be ~0 for black, got {x}");
        assert!(y.abs() < 1e-7, "Y should be ~0 for black, got {y}");
        assert!(b.abs() < 1e-7, "B should be ~0 for black, got {b}");
    }

    /// Verify butteraugli opsin is NOT zero for black (large biases).
    #[test]
    fn butteraugli_opsin_black_has_large_bias() {
        let (x, y, b) = butteraugli_opsin_single_pixel(0.0, 0.0, 0.0, 80.0);
        // Y = cur0 + cur1, where both come from bias ~1.756 * sensitivity
        assert!(y > 20.0, "Butteraugli Y for black should be > 20 (bias), got {y}");
        assert!(b > 20.0, "Butteraugli B for black should be > 20 (bias), got {b}");
        // X should be ~0 for neutral colors
        assert!(x.abs() < 1.0, "Butteraugli X for black should be ~0, got {x}");
    }

    /// Verify codec XYB: gray has X=0 (achromatic).
    #[test]
    fn codec_xyb_gray_has_zero_x() {
        for gray in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let (x, _y, _b) = codec_xyb(gray, gray, gray);
            assert!(
                x.abs() < 1e-7,
                "Codec X should be 0 for gray={gray}, got {x}"
            );
        }
    }

    /// Verify codec XYB: Y = B for gray (equal L/M cone response → equal Y and S).
    #[test]
    fn codec_xyb_gray_y_equals_b() {
        for gray in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let (_x, y, b) = codec_xyb(gray, gray, gray);
            assert!(
                (y - b).abs() < 1e-6,
                "Codec Y should equal B for gray={gray}: Y={y}, B={b}"
            );
        }
    }

    /// Verify the transfer function difference: cbrt vs log-gamma.
    /// At the same opsin value, cbrt and gamma produce very different outputs.
    #[test]
    fn transfer_functions_diverge() {
        let test_val = 10.0_f32; // reasonable opsin value

        let cbrt_result = test_val.cbrt();
        let gamma_result = opsin::gamma(test_val);

        // cbrt(10) ≈ 2.15
        assert!(
            (cbrt_result - 2.154).abs() < 0.01,
            "cbrt(10) should be ~2.15, got {cbrt_result}"
        );
        // gamma(10) uses log, should be very different
        assert!(
            (gamma_result - cbrt_result).abs() > 0.1,
            "Gamma and cbrt should diverge significantly at v=10"
        );
    }
}
