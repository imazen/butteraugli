//! Main butteraugli difference computation.
//!
//! This module ties together all the components to compute the
//! perceptual difference between two images.

use crate::consts::{
    NORM1_HF, NORM1_HF_X, NORM1_MF, NORM1_MF_X, NORM1_UHF, NORM1_UHF_X, WMUL, W_HF_MALTA,
    W_HF_MALTA_X, W_MF_MALTA, W_MF_MALTA_X, W_UHF_MALTA, W_UHF_MALTA_X,
};
use crate::image::{BufferPool, Image3F, ImageF};
use crate::malta::malta_diff_map;
use crate::mask::{
    combine_channels_for_masking, compute_mask as compute_mask_from_images, mask_dc_y, mask_y,
};
use crate::opsin::linear_rgb_to_xyb_butteraugli;
use crate::psycho::{separate_frequencies, PsychoImage};
use crate::ButteraugliParams;
use imgref::ImgRef;
use rgb::{RGB, RGB8};

/// Internal result type for diff module (uses ImageF, not ImgVec).
pub(crate) struct InternalResult {
    pub score: f64,
    pub diffmap: Option<ImageF>,
}

/// Minimum image dimension for multi-resolution processing.
/// Images smaller than this are handled without recursion.
const MIN_SIZE_FOR_MULTIRESOLUTION: usize = 8;

/// Converts linear RGB f32 buffer to XYB Image3F using butteraugli's OpsinDynamicsImage.
fn linear_rgb_to_xyb_image(
    rgb: &[f32],
    width: usize,
    height: usize,
    intensity_target: f32,
    pool: &BufferPool,
) -> Image3F {
    linear_rgb_to_xyb_butteraugli(rgb, width, height, intensity_target, pool)
}

/// Subsamples an Image3F by 2x using box filter averaging.
///
/// Each 2x2 block of pixels is averaged into a single pixel.
/// Edge cases for odd dimensions are handled by scaling the edge values.
#[allow(dead_code)]
fn subsample_2x(input: &Image3F) -> Image3F {
    let in_width = input.width();
    let in_height = input.height();
    let out_width = in_width.div_ceil(2);
    let out_height = in_height.div_ceil(2);

    let mut output = Image3F::new(out_width, out_height);

    // Initialize to zero (already done by Image3F::new)

    // Accumulate 2x2 blocks
    for c in 0..3 {
        for y in 0..in_height {
            for x in 0..in_width {
                let val = input.plane(c).get(x, y);
                let ox = x / 2;
                let oy = y / 2;
                let prev = output.plane(c).get(ox, oy);
                output.plane_mut(c).set(ox, oy, prev + 0.25 * val);
            }
        }

        // Handle odd width - last column only has half the samples
        if (in_width & 1) != 0 {
            let last_col = out_width - 1;
            for y in 0..out_height {
                let prev = output.plane(c).get(last_col, y);
                output.plane_mut(c).set(last_col, y, prev * 2.0);
            }
        }

        // Handle odd height - last row only has half the samples
        if (in_height & 1) != 0 {
            let last_row = out_height - 1;
            for x in 0..out_width {
                let prev = output.plane(c).get(x, last_row);
                output.plane_mut(c).set(x, last_row, prev * 2.0);
            }
        }
    }

    output
}

/// Converts sRGB u8 buffer to linear f32.
fn srgb_u8_to_linear_f32(rgb: &[u8]) -> Vec<f32> {
    rgb.iter()
        .map(|&v| crate::opsin::srgb_to_linear(v))
        .collect()
}

/// Adds a supersampled (upscaled 2x) diffmap to the destination.
///
/// This blends the lower-resolution analysis with the higher-resolution one
/// using a heuristic mixing value to reduce noise from lower resolutions.
fn add_supersampled_2x(src: &ImageF, weight: f32, dest: &mut ImageF) {
    let width = dest.width();
    let height = dest.height();

    // Heuristic from C++: lower resolution images have less error
    const K_HEURISTIC_MIXING_VALUE: f32 = 0.3;

    let blend = 1.0 - K_HEURISTIC_MIXING_VALUE * weight;
    for y in 0..height {
        let src_y = (y / 2).min(src.height() - 1);
        let src_row = src.row(src_y);
        let dst_row = dest.row_mut(y);
        let src_w = src.width();
        for x in 0..width {
            let src_val = src_row[(x / 2).min(src_w - 1)];
            dst_row[x] = dst_row[x] * blend + weight * src_val;
        }
    }
}

/// L2 difference (symmetric).
///
/// Computes squared difference weighted by w and adds to diffmap.
fn l2_diff(i0: &ImageF, i1: &ImageF, w: f32, diffmap: &mut ImageF) {
    archmage::incant!(l2_diff(i0, i1, w, diffmap));
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn l2_diff_v4(token: archmage::X64V4Token, i0: &ImageF, i1: &ImageF, w: f32, diffmap: &mut ImageF) {
    use magetypes::simd::f32x16;

    let width = i0.width();
    let height = i0.height();
    let w_simd = f32x16::splat(token, w);

    for y in 0..height {
        let row0 = i0.row(y);
        let row1 = i1.row(y);
        let row_diff = diffmap.row_mut(y);

        // SIMD path: process 16 elements at a time
        let chunks = row_diff.len() / 16;
        for i in 0..chunks {
            let x = i * 16;
            let v0 = f32x16::load(token, row0[x..x + 16].try_into().unwrap());
            let v1 = f32x16::load(token, row1[x..x + 16].try_into().unwrap());
            let curr = f32x16::load(token, row_diff[x..x + 16].try_into().unwrap());

            let diff = v0 - v1;
            let result = diff * diff * w_simd + curr;

            result.store((&mut row_diff[x..x + 16]).try_into().unwrap());
        }

        // Scalar tail
        let simd_width = chunks * 16;
        for x in simd_width..width {
            let diff = row0[x] - row1[x];
            row_diff[x] += diff * diff * w;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn l2_diff_v3(token: archmage::X64V3Token, i0: &ImageF, i1: &ImageF, w: f32, diffmap: &mut ImageF) {
    use magetypes::simd::f32x8;

    let width = i0.width();
    let height = i0.height();
    let w_simd = f32x8::splat(token, w);

    for y in 0..height {
        let row0 = i0.row(y);
        let row1 = i1.row(y);
        let row_diff = diffmap.row_mut(y);

        // SIMD path: process 8 elements at a time
        let chunks = row_diff.len() / 8;
        for i in 0..chunks {
            let x = i * 8;
            let v0 = f32x8::load(token, row0[x..x + 8].try_into().unwrap());
            let v1 = f32x8::load(token, row1[x..x + 8].try_into().unwrap());
            let curr = f32x8::load(token, row_diff[x..x + 8].try_into().unwrap());

            let diff = v0 - v1;
            let result = diff * diff * w_simd + curr;

            result.store((&mut row_diff[x..x + 8]).try_into().unwrap());
        }

        // Scalar tail
        let simd_width = chunks * 8;
        for x in simd_width..width {
            let diff = row0[x] - row1[x];
            row_diff[x] += diff * diff * w;
        }
    }
}

fn l2_diff_scalar(
    _token: archmage::ScalarToken,
    i0: &ImageF,
    i1: &ImageF,
    w: f32,
    diffmap: &mut ImageF,
) {
    let width = i0.width();
    let height = i0.height();

    for y in 0..height {
        let row0 = i0.row(y);
        let row1 = i1.row(y);
        let row_diff = diffmap.row_mut(y);

        for x in 0..width {
            let diff = row0[x] - row1[x];
            row_diff[x] += diff * diff * w;
        }
    }
}

/// L2 difference asymmetric.
///
/// This penalizes artifacts (original < reconstructed) more than blur
/// (original > reconstructed). Based on C++ L2DiffAsymmetric.
///
/// # Arguments
/// * `i0` - Original image
/// * `i1` - Reconstructed image
/// * `w_0gt1` - Weight when original > reconstructed (penalize blur)
/// * `w_0lt1` - Weight when original < reconstructed (penalize artifacts)
/// * `diffmap` - Output difference map (accumulated)
#[multiversed::multiversed("x86-64-v4", "x86-64-v3", "x86-64-v2", "arm64")]
fn l2_diff_asymmetric(i0: &ImageF, i1: &ImageF, w_0gt1: f32, w_0lt1: f32, diffmap: &mut ImageF) {
    if w_0gt1 == 0.0 && w_0lt1 == 0.0 {
        return;
    }

    let width = i0.width();
    let height = i0.height();
    let vw_0gt1 = w_0gt1 * 0.8;
    let vw_0lt1 = w_0lt1 * 0.8;

    for y in 0..height {
        let row0 = i0.row(y);
        let row1 = i1.row(y);
        let row_diff = diffmap.row_mut(y);

        for x in 0..width {
            let val0 = row0[x];
            let val1 = row1[x];

            // Primary symmetric quadratic objective
            let diff = val0 - val1;
            let mut total = row_diff[x] + diff * diff * vw_0gt1;

            // Secondary half-open quadratic objectives
            let fabs0 = val0.abs();
            let too_small = 0.4 * fabs0;
            let too_big = fabs0;

            let v = if val0 < 0.0 {
                if val1 > -too_small {
                    val1 + too_small
                } else if val1 < -too_big {
                    -val1 - too_big
                } else {
                    0.0
                }
            } else {
                if val1 < too_small {
                    too_small - val1
                } else if val1 > too_big {
                    val1 - too_big
                } else {
                    0.0
                }
            };

            total += vw_0lt1 * v * v;
            row_diff[x] = total;
        }
    }
}

// WMUL weights are imported from crate::consts
// These match C++ butteraugli.cc:
// [HF_X, HF_Y, HF_B, MF_X, MF_Y, MF_B, LF_X, LF_Y, LF_B]
// Note: WMUL is f64 array, but we need f32 for pixel operations

/// Computes difference between two PsychoImages using Malta filter.
///
/// This is the core butteraugli algorithm that applies:
/// 1. Malta edge-aware filter for UHF, HF, MF differences
/// 2. L2DiffAsymmetric for HF channels
/// 3. L2Diff for MF and LF channels
fn compute_psycho_diff_malta(
    ps0: &PsychoImage,
    ps1: &PsychoImage,
    hf_asymmetry: f32,
    _xmul: f32,
) -> Image3F {
    let width = ps0.width();
    let height = ps0.height();

    // Block diff AC accumulates Malta and L2 differences
    let mut block_diff_ac = Image3F::new(width, height);

    // Apply Malta filter for UHF (uses full Malta, not LF variant)
    // UHF Y channel
    let uhf_y_diff = malta_diff_map(
        &ps0.uhf[1],
        &ps1.uhf[1],
        W_UHF_MALTA * hf_asymmetry as f64,
        W_UHF_MALTA / hf_asymmetry as f64,
        NORM1_UHF,
        false, // use full Malta
    );
    {
        let ac = block_diff_ac.plane_mut(1);
        for y in 0..height {
            let src = uhf_y_diff.row(y);
            let dst = ac.row_mut(y);
            for x in 0..width {
                dst[x] += src[x];
            }
        }
    }

    // UHF X channel
    let uhf_x_diff = malta_diff_map(
        &ps0.uhf[0],
        &ps1.uhf[0],
        W_UHF_MALTA_X * hf_asymmetry as f64,
        W_UHF_MALTA_X / hf_asymmetry as f64,
        NORM1_UHF_X,
        false,
    );
    {
        let ac = block_diff_ac.plane_mut(0);
        for y in 0..height {
            let src = uhf_x_diff.row(y);
            let dst = ac.row_mut(y);
            for x in 0..width {
                dst[x] += src[x];
            }
        }
    }

    // Apply Malta LF filter for HF
    let sqrt_hf_asym = hf_asymmetry.sqrt();

    // HF Y channel (LF Malta)
    let hf_y_diff = malta_diff_map(
        &ps0.hf[1],
        &ps1.hf[1],
        W_HF_MALTA * sqrt_hf_asym as f64,
        W_HF_MALTA / sqrt_hf_asym as f64,
        NORM1_HF,
        true, // use LF Malta
    );
    {
        let ac = block_diff_ac.plane_mut(1);
        for y in 0..height {
            let src = hf_y_diff.row(y);
            let dst = ac.row_mut(y);
            for x in 0..width {
                dst[x] += src[x];
            }
        }
    }

    // HF X channel (LF Malta)
    let hf_x_diff = malta_diff_map(
        &ps0.hf[0],
        &ps1.hf[0],
        W_HF_MALTA_X * sqrt_hf_asym as f64,
        W_HF_MALTA_X / sqrt_hf_asym as f64,
        NORM1_HF_X,
        true,
    );
    {
        let ac = block_diff_ac.plane_mut(0);
        for y in 0..height {
            let src = hf_x_diff.row(y);
            let dst = ac.row_mut(y);
            for x in 0..width {
                dst[x] += src[x];
            }
        }
    }

    // Apply Malta LF filter for MF
    // MF Y channel
    let mf_y_diff = malta_diff_map(
        ps0.mf.plane(1),
        ps1.mf.plane(1),
        W_MF_MALTA,
        W_MF_MALTA,
        NORM1_MF,
        true,
    );
    {
        let ac = block_diff_ac.plane_mut(1);
        for y in 0..height {
            let src = mf_y_diff.row(y);
            let dst = ac.row_mut(y);
            for x in 0..width {
                dst[x] += src[x];
            }
        }
    }

    // MF X channel
    let mf_x_diff = malta_diff_map(
        ps0.mf.plane(0),
        ps1.mf.plane(0),
        W_MF_MALTA_X,
        W_MF_MALTA_X,
        NORM1_MF_X,
        true,
    );
    {
        let ac = block_diff_ac.plane_mut(0);
        for y in 0..height {
            let src = mf_x_diff.row(y);
            let dst = ac.row_mut(y);
            for x in 0..width {
                dst[x] += src[x];
            }
        }
    }

    // Add L2DiffAsymmetric for HF channels (X and Y, no blue)
    l2_diff_asymmetric(
        &ps0.hf[0],
        &ps1.hf[0],
        WMUL[0] as f32 * hf_asymmetry,
        WMUL[0] as f32 / hf_asymmetry,
        block_diff_ac.plane_mut(0),
    );
    l2_diff_asymmetric(
        &ps0.hf[1],
        &ps1.hf[1],
        WMUL[1] as f32 * hf_asymmetry,
        WMUL[1] as f32 / hf_asymmetry,
        block_diff_ac.plane_mut(1),
    );

    // Add L2Diff for MF channels (all three)
    l2_diff(
        ps0.mf.plane(0),
        ps1.mf.plane(0),
        WMUL[3] as f32,
        block_diff_ac.plane_mut(0),
    );
    l2_diff(
        ps0.mf.plane(1),
        ps1.mf.plane(1),
        WMUL[4] as f32,
        block_diff_ac.plane_mut(1),
    );
    l2_diff(
        ps0.mf.plane(2),
        ps1.mf.plane(2),
        WMUL[5] as f32,
        block_diff_ac.plane_mut(2),
    );

    block_diff_ac
}

/// Computes the mask from two PsychoImages.
///
/// Matches C++ MaskPsychoImage (butteraugli.cc lines 1250-1264).
/// Returns the computed mask and optionally accumulates AC differences.
fn mask_psycho_image(
    ps0: &PsychoImage,
    ps1: &PsychoImage,
    diff_ac: Option<&mut ImageF>,
    pool: &BufferPool,
) -> ImageF {
    let width = ps0.width();
    let height = ps0.height();

    // Combine HF and UHF channels for masking
    let mut mask0 = ImageF::new(width, height);
    let mut mask1 = ImageF::new(width, height);
    combine_channels_for_masking(&ps0.hf, &ps0.uhf, &mut mask0);
    combine_channels_for_masking(&ps1.hf, &ps1.uhf, &mut mask1);

    // Compute mask using DiffPrecompute, blur, and FuzzyErosion
    compute_mask_from_images(&mask0, &mask1, diff_ac, pool)
}

/// Combines channels to produce final diffmap.
///
/// Matches C++ CombineChannelsToDiffmap (butteraugli.cc lines 1289-1315).
/// Applies MaskY for AC differences and MaskDcY for DC differences.
#[multiversed::multiversed("x86-64-v4", "x86-64-v3", "x86-64-v2", "arm64")]
fn combine_channels_to_diffmap(
    mask: &ImageF,
    block_diff_dc: &Image3F,
    block_diff_ac: &Image3F,
    xmul: f32,
) -> ImageF {
    let width = mask.width();
    let height = mask.height();
    let mut diffmap = ImageF::new(width, height);

    for y in 0..height {
        let mask_row = mask.row(y);
        let dc0 = block_diff_dc.plane(0).row(y);
        let dc1 = block_diff_dc.plane(1).row(y);
        let dc2 = block_diff_dc.plane(2).row(y);
        let ac0 = block_diff_ac.plane(0).row(y);
        let ac1 = block_diff_ac.plane(1).row(y);
        let ac2 = block_diff_ac.plane(2).row(y);
        let out = diffmap.row_mut(y);

        for x in 0..width {
            let val = mask_row[x] as f64;

            // Compute masking factors from the mask value
            // MaskY is used for AC, MaskDcY is used for DC
            let maskval = mask_y(val) as f32;
            let dc_maskval = mask_dc_y(val) as f32;

            // Apply xmul to X channel (index 0) and sum with mask
            let dc_masked = dc0[x] * xmul * dc_maskval + dc1[x] * dc_maskval + dc2[x] * dc_maskval;
            let ac_masked = ac0[x] * xmul * maskval + ac1[x] * maskval + ac2[x] * maskval;

            // Final diffmap value is sqrt of sum
            out[x] = (dc_masked + ac_masked).sqrt();
        }
    }

    diffmap
}

/// Computes the global score from a difference map.
///
/// C++ ButteraugliScoreFromDiffmap (butteraugli.cc lines 1952-1962)
/// returns the maximum value in the diffmap. The diffmap already has
/// the global scaling applied via MaskY/MaskDcY.
#[multiversed::multiversed("x86-64-v4", "x86-64-v3", "x86-64-v2", "arm64")]
fn compute_score_from_diffmap(diffmap: &ImageF) -> f64 {
    let width = diffmap.width();
    let height = diffmap.height();
    let num_pixels = width * height;

    if num_pixels == 0 {
        return 0.0;
    }

    // Find maximum difference value (C++ butteraugli approach)
    let mut max_val = 0.0f32;

    for y in 0..height {
        let row = diffmap.row(y);
        for x in 0..width {
            if row[x] > max_val {
                max_val = row[x];
            }
        }
    }

    // No additional scaling needed - MaskY/MaskDcY already include GLOBAL_SCALE
    max_val as f64
}

/// Computes butteraugli diffmap with multiresolution (sRGB u8 input).
///
/// Converts sRGB to linear first, then delegates to the linear path.
/// This ensures subsampling happens in linear space (not gamma-compressed sRGB).
fn compute_diffmap_multiresolution(
    rgb1: &[u8],
    rgb2: &[u8],
    width: usize,
    height: usize,
    params: &ButteraugliParams,
    pool: &BufferPool,
) -> ImageF {
    let linear1 = srgb_u8_to_linear_f32(rgb1);
    let linear2 = srgb_u8_to_linear_f32(rgb2);
    compute_diffmap_multiresolution_linear(&linear1, &linear2, width, height, params, pool)
}

/// Subsamples linear RGB f32 buffer by 2x for multi-resolution processing.
fn subsample_linear_rgb_2x(rgb: &[f32], width: usize, height: usize) -> (Vec<f32>, usize, usize) {
    let out_width = width.div_ceil(2);
    let out_height = height.div_ceil(2);
    let mut output = vec![0.0f32; out_width * out_height * 3];

    // Simple averaging of 2x2 blocks
    for oy in 0..out_height {
        for ox in 0..out_width {
            let mut r_sum = 0.0f32;
            let mut g_sum = 0.0f32;
            let mut b_sum = 0.0f32;
            let mut count = 0.0f32;

            for dy in 0..2 {
                for dx in 0..2 {
                    let ix = ox * 2 + dx;
                    let iy = oy * 2 + dy;
                    if ix < width && iy < height {
                        let idx = (iy * width + ix) * 3;
                        r_sum += rgb[idx];
                        g_sum += rgb[idx + 1];
                        b_sum += rgb[idx + 2];
                        count += 1.0;
                    }
                }
            }

            if count > 0.0 {
                let out_idx = (oy * out_width + ox) * 3;
                output[out_idx] = r_sum / count;
                output[out_idx + 1] = g_sum / count;
                output[out_idx + 2] = b_sum / count;
            }
        }
    }

    (output, out_width, out_height)
}

/// Computes the diffmap for a single resolution level (linear RGB input).
fn compute_diffmap_single_resolution_linear(
    rgb1: &[f32],
    rgb2: &[f32],
    width: usize,
    height: usize,
    params: &ButteraugliParams,
    pool: &BufferPool,
) -> ImageF {
    // Convert to XYB using butteraugli's OpsinDynamicsImage
    let xyb1 = linear_rgb_to_xyb_image(rgb1, width, height, params.intensity_target(), pool);
    let xyb2 = linear_rgb_to_xyb_image(rgb2, width, height, params.intensity_target(), pool);

    // Perform frequency decomposition
    let ps1 = separate_frequencies(&xyb1, pool);
    let ps2 = separate_frequencies(&xyb2, pool);

    // Compute AC differences using Malta filter
    let mut block_diff_ac =
        compute_psycho_diff_malta(&ps1, &ps2, params.hf_asymmetry(), params.xmul());

    // Compute mask from both PsychoImages (also accumulates some AC differences)
    let mask = mask_psycho_image(&ps1, &ps2, Some(block_diff_ac.plane_mut(1)), pool);

    // Compute DC (LF) differences
    let mut block_diff_dc = Image3F::new(width, height);
    for c in 0..3 {
        let w = WMUL[6 + c] as f32;
        let dc = block_diff_dc.plane_mut(c);
        for y in 0..height {
            let lf1 = ps1.lf.plane(c).row(y);
            let lf2 = ps2.lf.plane(c).row(y);
            let dst = dc.row_mut(y);
            for x in 0..width {
                let d = lf1[x] - lf2[x];
                dst[x] = d * d * w;
            }
        }
    }

    // Combine channels to final diffmap using MaskY/MaskDcY
    combine_channels_to_diffmap(&mask, &block_diff_dc, &block_diff_ac, params.xmul())
}

/// Computes butteraugli diffmap with single-level multiresolution (linear RGB input).
///
/// Matches C++ ButteraugliComparator::Diffmap: computes at full resolution,
/// then adds ONE sub-level at half resolution via AddSupersampled2x.
/// The C++ creates a recursive tree in Make() but Diffmap() only uses
/// the immediate sub-level (via DiffmapOpsinDynamicsImage, which doesn't recurse).
fn compute_diffmap_multiresolution_linear(
    rgb1: &[f32],
    rgb2: &[f32],
    width: usize,
    height: usize,
    params: &ButteraugliParams,
    pool: &BufferPool,
) -> ImageF {
    const MIN_SIZE_FOR_SUBSAMPLE: usize = 15;

    // Compute sub-level diffmap at half resolution (single level, not recursive)
    let mut sub_diffmap = None;
    if !params.single_resolution()
        && width >= MIN_SIZE_FOR_SUBSAMPLE
        && height >= MIN_SIZE_FOR_SUBSAMPLE
    {
        let (sub_rgb1, sw, sh) = subsample_linear_rgb_2x(rgb1, width, height);
        let (sub_rgb2, _, _) = subsample_linear_rgb_2x(rgb2, width, height);

        // Single level only — matches C++ Diffmap behavior
        sub_diffmap = Some(compute_diffmap_single_resolution_linear(
            &sub_rgb1, &sub_rgb2, sw, sh, params, pool,
        ));
    }

    // Compute diffmap at full resolution
    let mut diffmap =
        compute_diffmap_single_resolution_linear(rgb1, rgb2, width, height, params, pool);

    // Add supersampled sub-level contribution
    if let Some(ref sub) = sub_diffmap {
        add_supersampled_2x(sub, 0.5, &mut diffmap);
    }

    diffmap
}

/// Main implementation of butteraugli comparison (sRGB u8 input).
///
/// Converts sRGB to linear f32 and delegates to the linear path.
/// This ensures all subsampling happens in linear space.
pub fn compute_butteraugli_impl(
    rgb1: &[u8],
    rgb2: &[u8],
    width: usize,
    height: usize,
    params: &ButteraugliParams,
) -> InternalResult {
    assert_eq!(rgb1.len(), width * height * 3);
    assert_eq!(rgb2.len(), width * height * 3);

    // Handle identical images case
    if rgb1 == rgb2 {
        return InternalResult {
            score: 0.0,
            diffmap: Some(ImageF::new(width, height)),
        };
    }

    // Convert sRGB u8 → linear f32, then use the linear path.
    // This is critical: subsampling must happen in linear space, not sRGB.
    let linear1 = srgb_u8_to_linear_f32(rgb1);
    let linear2 = srgb_u8_to_linear_f32(rgb2);

    compute_butteraugli_linear_impl(&linear1, &linear2, width, height, params)
}

/// Main implementation of butteraugli comparison (linear RGB f32 input).
///
/// This matches the C++ butteraugli API which expects linear RGB float input.
pub fn compute_butteraugli_linear_impl(
    rgb1: &[f32],
    rgb2: &[f32],
    width: usize,
    height: usize,
    params: &ButteraugliParams,
) -> InternalResult {
    assert_eq!(rgb1.len(), width * height * 3);
    assert_eq!(rgb2.len(), width * height * 3);

    // Handle identical images case
    if rgb1 == rgb2 {
        return InternalResult {
            score: 0.0,
            diffmap: Some(ImageF::new(width, height)),
        };
    }

    let pool = BufferPool::new();

    // Handle very small images without multi-resolution
    let diffmap = if width < MIN_SIZE_FOR_MULTIRESOLUTION || height < MIN_SIZE_FOR_MULTIRESOLUTION {
        compute_diffmap_single_resolution_linear(rgb1, rgb2, width, height, params, &pool)
    } else {
        compute_diffmap_multiresolution_linear(rgb1, rgb2, width, height, params, &pool)
    };

    // Compute global score
    let score = compute_score_from_diffmap(&diffmap);

    InternalResult {
        score,
        diffmap: Some(diffmap),
    }
}

/// Computes the diffmap for a single resolution level from XYB images.
fn compute_diffmap_single_resolution_xyb(
    xyb1: &Image3F,
    xyb2: &Image3F,
    params: &ButteraugliParams,
    pool: &BufferPool,
) -> ImageF {
    let width = xyb1.width();
    let height = xyb1.height();

    // Perform frequency decomposition
    let ps1 = separate_frequencies(xyb1, pool);
    let ps2 = separate_frequencies(xyb2, pool);

    // Compute AC differences using Malta filter
    let mut block_diff_ac =
        compute_psycho_diff_malta(&ps1, &ps2, params.hf_asymmetry(), params.xmul());

    // Compute mask from both PsychoImages (also accumulates some AC differences)
    let mask = mask_psycho_image(&ps1, &ps2, Some(block_diff_ac.plane_mut(1)), pool);

    // Compute DC (LF) differences
    let mut block_diff_dc = Image3F::new(width, height);
    for c in 0..3 {
        let w = WMUL[6 + c] as f32;
        let dc = block_diff_dc.plane_mut(c);
        for y in 0..height {
            let lf1 = ps1.lf.plane(c).row(y);
            let lf2 = ps2.lf.plane(c).row(y);
            let dst = dc.row_mut(y);
            for x in 0..width {
                let d = lf1[x] - lf2[x];
                dst[x] = d * d * w;
            }
        }
    }

    // Combine channels to final diffmap using MaskY/MaskDcY
    combine_channels_to_diffmap(&mask, &block_diff_dc, &block_diff_ac, params.xmul())
}

/// Implementation of butteraugli comparison for ImgRef<RGB8>.
pub(crate) fn compute_butteraugli_imgref(
    img1: ImgRef<RGB8>,
    img2: ImgRef<RGB8>,
    params: &ButteraugliParams,
    compute_diffmap: bool,
) -> InternalResult {
    let width = img1.width();
    let height = img1.height();

    // Convert ImgRef to contiguous u8 slice (handles stride)
    let rgb1 = imgref_rgb8_to_slice(img1);
    let rgb2 = imgref_rgb8_to_slice(img2);

    // Use the existing proven implementation with multiresolution support
    let mut result = compute_butteraugli_impl(&rgb1, &rgb2, width, height, params);

    // Drop diffmap if not requested
    if !compute_diffmap {
        result.diffmap = None;
    }

    result
}

/// Implementation of butteraugli comparison for ImgRef<RGB<f32>>.
pub(crate) fn compute_butteraugli_linear_imgref(
    img1: ImgRef<RGB<f32>>,
    img2: ImgRef<RGB<f32>>,
    params: &ButteraugliParams,
    compute_diffmap: bool,
) -> InternalResult {
    let width = img1.width();
    let height = img1.height();

    // Convert ImgRef to contiguous f32 slice (handles stride)
    let rgb1 = imgref_rgbf32_to_slice(img1);
    let rgb2 = imgref_rgbf32_to_slice(img2);

    // Use the existing proven implementation with multiresolution support
    let mut result = compute_butteraugli_linear_impl(&rgb1, &rgb2, width, height, params);

    // Drop diffmap if not requested
    if !compute_diffmap {
        result.diffmap = None;
    }

    result
}

/// Converts ImgRef<RGB8> to a contiguous Vec<u8> in RGB order.
fn imgref_rgb8_to_slice(img: ImgRef<RGB8>) -> Vec<u8> {
    let width = img.width();
    let height = img.height();
    let mut out = Vec::with_capacity(width * height * 3);

    for row in img.rows() {
        for px in row {
            out.push(px.r);
            out.push(px.g);
            out.push(px.b);
        }
    }

    out
}

/// Converts ImgRef<RGB<f32>> to a contiguous Vec<f32> in RGB order.
fn imgref_rgbf32_to_slice(img: ImgRef<RGB<f32>>) -> Vec<f32> {
    let width = img.width();
    let height = img.height();
    let mut out = Vec::with_capacity(width * height * 3);

    for row in img.rows() {
        for px in row {
            out.push(px.r);
            out.push(px.g);
            out.push(px.b);
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_images() {
        let width = 32;
        let height = 32;
        let rgb: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();

        let result =
            compute_butteraugli_impl(&rgb, &rgb, width, height, &ButteraugliParams::default());

        assert!(
            result.score < 0.001,
            "Identical images should have score ~0, got {}",
            result.score
        );
    }

    #[test]
    fn test_slightly_different_images() {
        let width = 32;
        let height = 32;
        let rgb1: Vec<u8> = vec![128; width * height * 3];
        let mut rgb2 = rgb1.clone();
        // Change one pixel slightly
        rgb2[0] = 129;
        rgb2[1] = 129;
        rgb2[2] = 129;

        let result =
            compute_butteraugli_impl(&rgb1, &rgb2, width, height, &ButteraugliParams::default());

        // Small difference should have low score
        assert!(
            result.score < 2.0,
            "Small difference should have low score, got {}",
            result.score
        );
    }

    #[test]
    fn test_very_different_images() {
        let width = 32;
        let height = 32;
        let rgb1: Vec<u8> = vec![0; width * height * 3];
        let rgb2: Vec<u8> = vec![255; width * height * 3];

        let result =
            compute_butteraugli_impl(&rgb1, &rgb2, width, height, &ButteraugliParams::default());

        // Very different images should have non-zero score
        // Note: uniform images (all black vs all white) have limited frequency content,
        // so the score may be lower than expected for natural images
        assert!(
            result.score > 0.01,
            "Very different images should have non-zero score, got {}",
            result.score
        );
    }

    #[test]
    fn test_diffmap_dimensions() {
        let width = 64;
        let height = 48;
        let rgb1: Vec<u8> = vec![100; width * height * 3];
        let rgb2: Vec<u8> = vec![150; width * height * 3];

        let result =
            compute_butteraugli_impl(&rgb1, &rgb2, width, height, &ButteraugliParams::default());

        let diffmap = result.diffmap.unwrap();
        assert_eq!(diffmap.width(), width);
        assert_eq!(diffmap.height(), height);
    }

    #[test]
    fn test_l2_diff_asymmetric() {
        let width = 16;
        let height = 16;
        let i0 = ImageF::filled(width, height, 1.0);
        let i1 = ImageF::filled(width, height, 0.5);
        let mut diffmap = ImageF::new(width, height);

        l2_diff_asymmetric(&i0, &i1, 1.0, 1.0, &mut diffmap);

        // Should have non-zero difference
        let mut sum = 0.0;
        for y in 0..height {
            for x in 0..width {
                sum += diffmap.get(x, y);
            }
        }
        assert!(sum > 0.0, "L2 diff should be non-zero for different images");
    }

    #[test]
    fn test_add_supersampled_2x() {
        let src = ImageF::filled(4, 4, 1.0);
        let mut dest = ImageF::filled(8, 8, 2.0);

        add_supersampled_2x(&src, 0.5, &mut dest);

        // Should have blended values
        // new = old * (1 - 0.3 * 0.5) + 0.5 * 1.0 = 2.0 * 0.85 + 0.5 = 1.7 + 0.5 = 2.2
        let val = dest.get(0, 0);
        assert!((val - 2.2).abs() < 0.01, "Expected ~2.2, got {val}");
    }

    #[test]
    fn test_multiresolution_small_image() {
        // Very small image should not recurse
        let width = 4;
        let height = 4;
        let rgb1: Vec<u8> = vec![128; width * height * 3];
        let rgb2: Vec<u8> = vec![140; width * height * 3];

        let result =
            compute_butteraugli_impl(&rgb1, &rgb2, width, height, &ButteraugliParams::default());

        assert!(result.score > 0.0, "Should have non-zero score");
    }
}
