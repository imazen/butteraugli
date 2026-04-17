//! IIR (recursive) Gaussian blur — Charalampidis 2016.
//!
//! Approximates a Gaussian convolution in O(N) per pixel **independent of sigma**,
//! versus the FIR path which is O(sigma) per pixel. Roughly 4–10× fewer operations
//! at butteraugli's sigmas (1.56–7.16).
//!
//! "Recursive Implementation of the Gaussian Filter Using Truncated Cosine
//! Functions", D. Charalampidis, IEEE Trans. Signal Processing, 2016.
//!
//! **NOT bit-exact with the FIR path.** Real-photo score deviation is 0.1–5%
//! (libjxl `butteraugli_main` reference, GB82 corpus Q75, 576×576). On small
//! synthetic images the gap blows up because IIR uses zero-padding boundary
//! conditions while FIR uses clamp-to-edge — gated behind `iir-blur` for that
//! reason.

use crate::image::{BufferPool, ImageF};
use archmage::{autoversion, incant, magetypes};
use core::f64::consts::PI;
use magetypes::simd::generic::f32x8 as GenericF32x8;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
use magetypes::simd::v4::f32x16 as MtF32x16;

/// IIR weights for one Gaussian sigma. Three parallel 2-pole sections (k=1,3,5),
/// constants derived in f64 for numerical stability and stored as f32.
#[derive(Clone, Copy, Debug)]
pub(crate) struct IirCoeffs {
    /// Filter "radius" (paper N): warm-up extent before the visible output region.
    pub radius: i32,
    /// Per-section input coefficients (`n2[k]` from the paper).
    pub mul_in: [f32; 3],
    /// Per-section state coefficient (`d1[k]` from the paper).
    pub mul_prev: [f32; 3],
}

impl IirCoeffs {
    pub fn for_sigma(sigma: f32) -> Self {
        let sigma = sigma as f64;

        // Eq. (57): N = round(3.2795 * sigma + 0.2546).
        let radius = 3.2795_f64.mul_add(sigma, 0.2546).round();

        // Table I: omega_k = (2k-1) * pi / (2N), k=1,3,5.
        let pi_div_2r = PI / (2.0 * radius);
        let omega = [pi_div_2r, 3.0 * pi_div_2r, 5.0 * pi_div_2r];

        // Eq. (37): p_k.
        let p = [
            1.0 / (0.5 * omega[0]).tan(),
            -1.0 / (0.5 * omega[1]).tan(),
            1.0 / (0.5 * omega[2]).tan(),
        ];

        // Eq. (44): r_k.
        let r = [
            p[0] * p[0] / omega[0].sin(),
            -p[1] * p[1] / omega[1].sin(),
            p[2] * p[2] / omega[2].sin(),
        ];

        // Eq. (50): rho_k.
        let neg_half_sigma2 = -0.5 * sigma * sigma;
        let recip_radius = 1.0 / radius;
        let rho = [
            (neg_half_sigma2 * omega[0] * omega[0]).exp() * recip_radius,
            (neg_half_sigma2 * omega[1] * omega[1]).exp() * recip_radius,
            (neg_half_sigma2 * omega[2] * omega[2]).exp() * recip_radius,
        ];

        // Eq. (52): zeta_15, zeta_35.
        let d_13 = p[0].mul_add(r[1], -r[0] * p[1]);
        let d_35 = p[1].mul_add(r[2], -r[1] * p[2]);
        let d_51 = p[2].mul_add(r[0], -r[2] * p[0]);
        let recip_d13 = 1.0 / d_13;
        let zeta_15 = d_35 * recip_d13;
        let zeta_35 = d_51 * recip_d13;

        // Eq. (56): solve A * beta = gamma.
        let g0 = 1.0;
        let g1 = radius.mul_add(radius, -sigma * sigma);
        let g2 = zeta_15.mul_add(rho[0], zeta_35 * rho[1]) + rho[2];
        let beta = solve_3x3(
            [
                [p[0], p[1], p[2]],
                [r[0], r[1], r[2]],
                [zeta_15, zeta_35, 1.0],
            ],
            [g0, g1, g2],
        );

        debug_assert!(
            (beta[2].mul_add(p[2], beta[0].mul_add(p[0], beta[1] * p[1])) - 1.0).abs() < 1e-9
        );

        let mul_in = [
            (-beta[0] * (omega[0] * (radius + 1.0)).cos()) as f32,
            (-beta[1] * (omega[1] * (radius + 1.0)).cos()) as f32,
            (-beta[2] * (omega[2] * (radius + 1.0)).cos()) as f32,
        ];
        let mul_prev = [
            (-2.0 * omega[0].cos()) as f32,
            (-2.0 * omega[1].cos()) as f32,
            (-2.0 * omega[2].cos()) as f32,
        ];

        Self {
            radius: radius as i32,
            mul_in,
            mul_prev,
        }
    }
}

fn solve_3x3(a: [[f64; 3]; 3], b: [f64; 3]) -> [f64; 3] {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    let inv_det = 1.0 / det;

    let x0 = b[0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (b[1] * a[2][2] - a[1][2] * b[2])
        + a[0][2] * (b[1] * a[2][1] - a[1][1] * b[2]);
    let x1 = a[0][0] * (b[1] * a[2][2] - a[1][2] * b[2])
        - b[0] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * b[2] - b[1] * a[2][0]);
    let x2 = a[0][0] * (a[1][1] * b[2] - b[1] * a[2][1])
        - a[0][1] * (a[1][0] * b[2] - b[1] * a[2][0])
        + b[0] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);

    [x0 * inv_det, x1 * inv_det, x2 * inv_det]
}

// ---------------------------------------------------------------------------
// Horizontal pass — scalar IIR per row, dispatched with #[autoversion] for FMA.
// ---------------------------------------------------------------------------

#[allow(unused_imports)] // autoversion fallback path triggers a false positive on i686
#[autoversion(v4, v3, neon, wasm128, scalar)]
fn horizontal_pass(input: &[f32], output: &mut [f32], width: usize, coeffs: &IirCoeffs) {
    debug_assert_eq!(input.len(), output.len());
    debug_assert_eq!(input.len() % width, 0);
    for (in_row, out_row) in input
        .chunks_exact(width)
        .zip(output.chunks_exact_mut(width))
    {
        horizontal_row(in_row, out_row, coeffs);
    }
}

#[inline(always)]
fn horizontal_row(input: &[f32], output: &mut [f32], coeffs: &IirCoeffs) {
    let width = input.len() as isize;
    let big_n = coeffs.radius as isize;

    let mi1 = coeffs.mul_in[0];
    let mi3 = coeffs.mul_in[1];
    let mi5 = coeffs.mul_in[2];
    let mp1 = coeffs.mul_prev[0];
    let mp3 = coeffs.mul_prev[1];
    let mp5 = coeffs.mul_prev[2];

    let mut prev_1 = 0f32;
    let mut prev_3 = 0f32;
    let mut prev_5 = 0f32;
    let mut prev2_1 = 0f32;
    let mut prev2_3 = 0f32;
    let mut prev2_5 = 0f32;

    let mut n = -big_n + 1;
    while n < width {
        let left = n - big_n - 1;
        let right = n + big_n - 1;
        let left_val = if left >= 0 && left < width {
            input[left as usize]
        } else {
            0f32
        };
        let right_val = if right >= 0 && right < width {
            input[right as usize]
        } else {
            0f32
        };
        let sum = left_val + right_val;

        let out_1 = sum.mul_add(mi1, -mp1.mul_add(prev_1, prev2_1));
        let out_3 = sum.mul_add(mi3, -mp3.mul_add(prev_3, prev2_3));
        let out_5 = sum.mul_add(mi5, -mp5.mul_add(prev_5, prev2_5));

        prev2_1 = prev_1;
        prev2_3 = prev_3;
        prev2_5 = prev_5;
        prev_1 = out_1;
        prev_3 = out_3;
        prev_5 = out_5;

        if n >= 0 {
            output[n as usize] = out_1 + out_3 + out_5;
        }
        n += 1;
    }
}

// ---------------------------------------------------------------------------
// Vertical pass — SIMD across columns. Each lane is an independent IIR column.
// ---------------------------------------------------------------------------

fn vertical_pass(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    coeffs: &IirCoeffs,
) {
    incant!(
        vertical_pass_inner(input, output, width, height, coeffs),
        [v4, v3, neon, wasm128, scalar]
    )
}

// AVX-512 path: 16 columns at a time. On Zen 4 this is the same throughput as
// 2× v3 f32x8 (zmm splits to 2× ymm μops), but cuts loop overhead in half and
// frees more registers for IIR state, which matters for serial recurrences.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
fn vertical_pass_inner_v4(
    token: archmage::X64V4Token,
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    coeffs: &IirCoeffs,
) {
    const LANES: usize = 16;
    let big_n = coeffs.radius as isize;
    let height_i = height as isize;
    let groups = width / LANES;

    let mi1 = MtF32x16::splat(token, coeffs.mul_in[0]);
    let mi3 = MtF32x16::splat(token, coeffs.mul_in[1]);
    let mi5 = MtF32x16::splat(token, coeffs.mul_in[2]);
    let mp1 = MtF32x16::splat(token, coeffs.mul_prev[0]);
    let mp3 = MtF32x16::splat(token, coeffs.mul_prev[1]);
    let mp5 = MtF32x16::splat(token, coeffs.mul_prev[2]);
    let zeroes = MtF32x16::zero(token);

    for g in 0..groups {
        let col = g * LANES;
        let mut prev_1 = zeroes;
        let mut prev_3 = zeroes;
        let mut prev_5 = zeroes;
        let mut prev2_1 = zeroes;
        let mut prev2_3 = zeroes;
        let mut prev2_5 = zeroes;

        let mut n = -big_n + 1;
        while n < height_i {
            let top = n - big_n - 1;
            let bottom = n + big_n - 1;

            let top_v = if top >= 0 && top < height_i {
                MtF32x16::from_array(
                    token,
                    input[top as usize * width + col..][..LANES]
                        .try_into()
                        .unwrap(),
                )
            } else {
                zeroes
            };
            let bot_v = if bottom >= 0 && bottom < height_i {
                MtF32x16::from_array(
                    token,
                    input[bottom as usize * width + col..][..LANES]
                        .try_into()
                        .unwrap(),
                )
            } else {
                zeroes
            };
            let sum = top_v + bot_v;

            let acc1 = prev_1.mul_add(mp1, prev2_1);
            let acc3 = prev_3.mul_add(mp3, prev2_3);
            let acc5 = prev_5.mul_add(mp5, prev2_5);
            let out1 = sum.mul_add(mi1, -acc1);
            let out3 = sum.mul_add(mi3, -acc3);
            let out5 = sum.mul_add(mi5, -acc5);

            prev2_1 = prev_1;
            prev2_3 = prev_3;
            prev2_5 = prev_5;
            prev_1 = out1;
            prev_3 = out3;
            prev_5 = out5;

            if n >= 0 {
                let result = out1 + out3 + out5;
                let dst = n as usize * width + col;
                output[dst..dst + LANES].copy_from_slice(&result.to_array());
            }
            n += 1;
        }
    }

    // Scalar remainder for columns past the last full LANES group.
    let scalar_start = groups * LANES;
    if scalar_start < width {
        vertical_pass_scalar_columns(input, output, width, height, scalar_start, coeffs);
    }
}

#[magetypes(v3, neon, wasm128, scalar)]
fn vertical_pass_inner(
    token: Token,
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    coeffs: &IirCoeffs,
) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    const LANES: usize = 8;

    let big_n = coeffs.radius as isize;
    let height_i = height as isize;
    let groups = width / LANES;

    let mi1 = f32x8::splat(token, coeffs.mul_in[0]);
    let mi3 = f32x8::splat(token, coeffs.mul_in[1]);
    let mi5 = f32x8::splat(token, coeffs.mul_in[2]);
    let mp1 = f32x8::splat(token, coeffs.mul_prev[0]);
    let mp3 = f32x8::splat(token, coeffs.mul_prev[1]);
    let mp5 = f32x8::splat(token, coeffs.mul_prev[2]);
    let zeroes = f32x8::zero(token);

    // Process one column group at a time, all rows in a tight inner loop.
    // State stays in 6 SIMD registers across the n loop — no per-iteration
    // memory traffic for state, so loop is bound only by FMA latency.
    for g in 0..groups {
        let col = g * LANES;
        let mut prev_1 = zeroes;
        let mut prev_3 = zeroes;
        let mut prev_5 = zeroes;
        let mut prev2_1 = zeroes;
        let mut prev2_3 = zeroes;
        let mut prev2_5 = zeroes;

        let mut n = -big_n + 1;
        while n < height_i {
            let top = n - big_n - 1;
            let bottom = n + big_n - 1;

            let top_v = if top >= 0 && top < height_i {
                f32x8::from_array(
                    token,
                    input[top as usize * width + col..][..LANES]
                        .try_into()
                        .unwrap(),
                )
            } else {
                zeroes
            };
            let bot_v = if bottom >= 0 && bottom < height_i {
                f32x8::from_array(
                    token,
                    input[bottom as usize * width + col..][..LANES]
                        .try_into()
                        .unwrap(),
                )
            } else {
                zeroes
            };
            let sum = top_v + bot_v;

            // out = sum * mi - mp * prev - prev2
            let acc1 = prev_1.mul_add(mp1, prev2_1);
            let acc3 = prev_3.mul_add(mp3, prev2_3);
            let acc5 = prev_5.mul_add(mp5, prev2_5);
            let out1 = sum.mul_add(mi1, -acc1);
            let out3 = sum.mul_add(mi3, -acc3);
            let out5 = sum.mul_add(mi5, -acc5);

            prev2_1 = prev_1;
            prev2_3 = prev_3;
            prev2_5 = prev_5;
            prev_1 = out1;
            prev_3 = out3;
            prev_5 = out5;

            if n >= 0 {
                let result = out1 + out3 + out5;
                let dst = n as usize * width + col;
                output[dst..dst + LANES].copy_from_slice(&result.to_array());
            }
            n += 1;
        }
    }

    // Scalar remainder for columns past the last full LANES group.
    let scalar_start = groups * LANES;
    if scalar_start < width {
        vertical_pass_scalar_columns(input, output, width, height, scalar_start, coeffs);
    }
}

#[allow(unused_imports)]
#[autoversion(v4, v3, neon, wasm128, scalar)]
fn vertical_pass_scalar_columns(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    start_x: usize,
    coeffs: &IirCoeffs,
) {
    let big_n = coeffs.radius as isize;
    let height_i = height as isize;

    let mi1 = coeffs.mul_in[0];
    let mi3 = coeffs.mul_in[1];
    let mi5 = coeffs.mul_in[2];
    let mp1 = coeffs.mul_prev[0];
    let mp3 = coeffs.mul_prev[1];
    let mp5 = coeffs.mul_prev[2];

    for x in start_x..width {
        let mut prev_1 = 0f32;
        let mut prev_3 = 0f32;
        let mut prev_5 = 0f32;
        let mut prev2_1 = 0f32;
        let mut prev2_3 = 0f32;
        let mut prev2_5 = 0f32;

        let mut n = -big_n + 1;
        while n < height_i {
            let top = n - big_n - 1;
            let bottom = n + big_n - 1;
            let top_v = if top >= 0 && top < height_i {
                input[top as usize * width + x]
            } else {
                0f32
            };
            let bot_v = if bottom >= 0 && bottom < height_i {
                input[bottom as usize * width + x]
            } else {
                0f32
            };
            let sum = top_v + bot_v;

            let out_1 = sum.mul_add(mi1, -mp1.mul_add(prev_1, prev2_1));
            let out_3 = sum.mul_add(mi3, -mp3.mul_add(prev_3, prev2_3));
            let out_5 = sum.mul_add(mi5, -mp5.mul_add(prev_5, prev2_5));

            prev2_1 = prev_1;
            prev2_3 = prev_3;
            prev2_5 = prev_5;
            prev_1 = out_1;
            prev_3 = out_3;
            prev_5 = out_5;

            if n >= 0 {
                output[n as usize * width + x] = out_1 + out_3 + out_5;
            }
            n += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry point.
// ---------------------------------------------------------------------------

/// Apply IIR Gaussian blur to an image. Mirrors `gaussian_blur` signature.
pub fn gaussian_blur_iir(input: &ImageF, sigma: f32, pool: &BufferPool) -> ImageF {
    if sigma <= 0.0 {
        return input.clone();
    }
    let coeffs = IirCoeffs::for_sigma(sigma);
    let width = input.width();
    let height = input.height();

    let mut temp = ImageF::from_pool_dirty(width, height, pool);
    horizontal_pass(input.data(), temp.data_mut(), width, &coeffs);

    let mut output = ImageF::from_pool_dirty(width, height, pool);
    vertical_pass(temp.data(), output.data_mut(), width, height, &coeffs);
    temp.recycle(pool);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iir_horizontal_impulse_dc_gain() {
        for sigma in [1.564f32, 2.7, 3.225, 7.156] {
            let coeffs = IirCoeffs::for_sigma(sigma);
            let mut input = vec![0f32; 256];
            input[128] = 1.0;
            let mut output = vec![0f32; 256];
            horizontal_row(&input, &mut output, &coeffs);
            let sum: f32 = output.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "sigma={sigma}: 1D impulse sum {sum}, expected ~1.0",
            );
            let peak_idx = output
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;
            assert_eq!(peak_idx, 128, "sigma={sigma}: peak should be centered");
        }
    }

    #[test]
    fn iir_2d_impulse_dc_gain() {
        for sigma in [1.564f32, 2.7, 3.225, 7.156] {
            let pool = BufferPool::new();
            let mut img = ImageF::filled(128, 128, 0.0);
            img.set(64, 64, 1.0);
            let blurred = gaussian_blur_iir(&img, sigma, &pool);
            let sum: f32 = (0..128).flat_map(|y| blurred.row(y).iter().copied()).sum();
            assert!(
                (sum - 1.0).abs() < 0.02,
                "sigma={sigma}: 2D impulse sum {sum}, expected ~1.0",
            );
        }
    }

    #[test]
    fn iir_dc_constant_center() {
        let pool = BufferPool::new();
        let img = ImageF::filled(64, 64, 0.5);
        let blurred = gaussian_blur_iir(&img, 2.7, &pool);
        let center = blurred.get(32, 32);
        assert!(
            (center - 0.5).abs() < 1e-3,
            "center should be ~0.5, got {center}",
        );
    }
}
