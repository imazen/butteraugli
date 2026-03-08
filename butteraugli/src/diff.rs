//! Main butteraugli difference computation.
//!
//! This module ties together all the components to compute the
//! perceptual difference between two images.

use crate::ButteraugliParams;
use crate::consts::{
    NORM1_HF, NORM1_HF_X, NORM1_MF, NORM1_MF_X, NORM1_UHF, NORM1_UHF_X, W_HF_MALTA, W_HF_MALTA_X,
    W_MF_MALTA, W_MF_MALTA_X, W_UHF_MALTA, W_UHF_MALTA_X, WMUL,
};
use crate::image::{BufferPool, Image3F, ImageF};
use crate::malta::malta_diff_map;
use crate::mask::compute_mask_from_hf_uhf;
use crate::opsin::linear_rgb_to_xyb_butteraugli;
use crate::psycho::{PsychoImage, separate_frequencies};
use imgref::ImgRef;
use rgb::{RGB, RGB8};

/// Conditional parallelism: uses rayon::join when the `rayon` feature is enabled,
/// otherwise runs both closures sequentially on the current thread.
#[cfg(feature = "rayon")]
pub(crate) fn maybe_join<A, B, RA, RB>(a: A, b: B) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    RA: Send,
    RB: Send,
{
    rayon::join(a, b)
}

#[cfg(not(feature = "rayon"))]
pub(crate) fn maybe_join<A, B, RA, RB>(a: A, b: B) -> (RA, RB)
where
    A: FnOnce() -> RA,
    B: FnOnce() -> RB,
{
    (a(), b())
}

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

/// Converts sRGB u8 buffer to linear f32.
#[cfg(test)]
fn srgb_u8_to_linear_f32(rgb: &[u8]) -> Vec<f32> {
    let lut = &*crate::opsin::SRGB_TO_LINEAR_LUT;
    rgb.iter().map(|&v| lut[v as usize]).collect()
}

/// Adds a supersampled (upscaled 2x) diffmap to the destination.
///
/// This blends the lower-resolution analysis with the higher-resolution one
/// using a heuristic mixing value to reduce noise from lower resolutions.
/// Adds supersampled diffmap contribution (2× upsampling with blending).
///
/// Processes pairs of destination pixels that share the same source pixel,
/// enabling sequential access on both src and dst for better vectorization.
#[archmage::autoversion]
fn add_supersampled_2x(_token: archmage::SimdToken, src: &ImageF, weight: f32, dest: &mut ImageF) {
    let dest_width = dest.width();
    let dest_height = dest.height();
    const K_HEURISTIC_MIXING_VALUE: f32 = 0.3;
    let blend = 1.0 - K_HEURISTIC_MIXING_VALUE * weight;
    let src_w = src.width();
    let src_h = src.height();

    for y in 0..dest_height {
        let src_y = (y / 2).min(src_h - 1);
        let src_row = src.row(src_y);
        let dst_row = dest.row_mut(y);

        // Process pairs of dest pixels that share the same source pixel.
        let n_pairs = (dest_width / 2).min(src_w);
        for (pair, &sv) in dst_row[..n_pairs * 2]
            .chunks_exact_mut(2)
            .zip(src_row[..n_pairs].iter())
        {
            let ws = weight * sv;
            pair[0] = pair[0].mul_add(blend, ws);
            pair[1] = pair[1].mul_add(blend, ws);
        }

        // Handle odd trailing pixel (when dest_width is odd)
        if dest_width > n_pairs * 2 {
            let sv = src_row[(dest_width / 2).min(src_w - 1)];
            dst_row[dest_width - 1] = dst_row[dest_width - 1].mul_add(blend, weight * sv);
        }
    }
}

/// Accumulates two source images into a destination: dst[x] += a[x] + b[x].
#[archmage::autoversion]
fn accumulate_two(_token: archmage::SimdToken, a: &ImageF, b: &ImageF, dst: &mut ImageF) {
    let height = a.height();
    for y in 0..height {
        let ra = a.row(y);
        let rb = b.row(y);
        let rd = dst.row_mut(y);
        for ((d, &va), &vb) in rd.iter_mut().zip(ra.iter()).zip(rb.iter()) {
            *d += va + vb;
        }
    }
}

/// L2 difference (symmetric).
///
/// Computes squared difference weighted by w and adds to diffmap.
#[archmage::autoversion]
fn l2_diff(_token: archmage::SimdToken, i0: &ImageF, i1: &ImageF, w: f32, diffmap: &mut ImageF) {
    let height = i0.height();

    for y in 0..height {
        let row0 = i0.row(y);
        let row1 = i1.row(y);
        let row_diff = diffmap.row_mut(y);

        for ((d, &v0), &v1) in row_diff.iter_mut().zip(row0.iter()).zip(row1.iter()) {
            let diff = v0 - v1;
            *d = (diff * diff).mul_add(w, *d);
        }
    }
}

/// L2 difference (symmetric, write-only).
///
/// Like `l2_diff` but overwrites diffmap instead of accumulating.
/// Use when diffmap is uninitialized or dirty — avoids needing zeroed memory.
#[archmage::autoversion]
fn l2_diff_write(
    _token: archmage::SimdToken,
    i0: &ImageF,
    i1: &ImageF,
    w: f32,
    diffmap: &mut ImageF,
) {
    let height = i0.height();

    for y in 0..height {
        let row0 = i0.row(y);
        let row1 = i1.row(y);
        let row_diff = diffmap.row_mut(y);

        for ((d, &v0), &v1) in row_diff.iter_mut().zip(row0.iter()).zip(row1.iter()) {
            let diff = v0 - v1;
            *d = diff * diff * w;
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
#[archmage::autoversion]
fn l2_diff_asymmetric(
    _token: archmage::SimdToken,
    i0: &ImageF,
    i1: &ImageF,
    w_0gt1: f32,
    w_0lt1: f32,
    diffmap: &mut ImageF,
) {
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

            let diff = val0 - val1;
            let total = (diff * diff).mul_add(vw_0gt1, row_diff[x]);

            // Branch-free asymmetric penalty:
            // Flip val1 to match val0's sign direction, then clamp.
            let fabs0 = val0.abs();
            let too_small = 0.4 * fabs0;
            let sign = 1.0f32.copysign(val0);
            let sv1 = val1 * sign;
            let v = (too_small - sv1).max(0.0) + (sv1 - fabs0).max(0.0);

            row_diff[x] = (v * v).mul_add(vw_0lt1, total);
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
    pool: &BufferPool,
) -> Image3F {
    let width = ps0.width();
    let height = ps0.height();

    let sqrt_hf_asym = hf_asymmetry.sqrt();

    // Parallel: compute all 6 independent Malta diff maps
    let ((uhf_y_diff, uhf_x_diff), ((hf_y_diff, hf_x_diff), (mf_y_diff, mf_x_diff))) = maybe_join(
        || {
            maybe_join(
                || {
                    malta_diff_map(
                        &ps0.uhf[1],
                        &ps1.uhf[1],
                        W_UHF_MALTA * hf_asymmetry as f64,
                        W_UHF_MALTA / hf_asymmetry as f64,
                        NORM1_UHF,
                        false,
                        pool,
                    )
                },
                || {
                    malta_diff_map(
                        &ps0.uhf[0],
                        &ps1.uhf[0],
                        W_UHF_MALTA_X * hf_asymmetry as f64,
                        W_UHF_MALTA_X / hf_asymmetry as f64,
                        NORM1_UHF_X,
                        false,
                        pool,
                    )
                },
            )
        },
        || {
            maybe_join(
                || {
                    maybe_join(
                        || {
                            malta_diff_map(
                                &ps0.hf[1],
                                &ps1.hf[1],
                                W_HF_MALTA * sqrt_hf_asym as f64,
                                W_HF_MALTA / sqrt_hf_asym as f64,
                                NORM1_HF,
                                true,
                                pool,
                            )
                        },
                        || {
                            malta_diff_map(
                                &ps0.hf[0],
                                &ps1.hf[0],
                                W_HF_MALTA_X * sqrt_hf_asym as f64,
                                W_HF_MALTA_X / sqrt_hf_asym as f64,
                                NORM1_HF_X,
                                true,
                                pool,
                            )
                        },
                    )
                },
                || {
                    maybe_join(
                        || {
                            malta_diff_map(
                                ps0.mf.plane(1),
                                ps1.mf.plane(1),
                                W_MF_MALTA,
                                W_MF_MALTA,
                                NORM1_MF,
                                true,
                                pool,
                            )
                        },
                        || {
                            malta_diff_map(
                                ps0.mf.plane(0),
                                ps1.mf.plane(0),
                                W_MF_MALTA_X,
                                W_MF_MALTA_X,
                                NORM1_MF_X,
                                true,
                                pool,
                            )
                        },
                    )
                },
            )
        },
    );

    // Use UHF Malta results directly as accumulators (no zero-init + add_to needed)
    let mut plane_x = uhf_x_diff;
    let mut plane_y = uhf_y_diff;

    // Fuse HF + MF Malta into single accumulation pass per channel
    accumulate_two(&hf_y_diff, &mf_y_diff, &mut plane_y);
    accumulate_two(&hf_x_diff, &mf_x_diff, &mut plane_x);

    // Add L2DiffAsymmetric for HF channels (X and Y, no blue)
    l2_diff_asymmetric(
        &ps0.hf[0],
        &ps1.hf[0],
        WMUL[0] as f32 * hf_asymmetry,
        WMUL[0] as f32 / hf_asymmetry,
        &mut plane_x,
    );
    l2_diff_asymmetric(
        &ps0.hf[1],
        &ps1.hf[1],
        WMUL[1] as f32 * hf_asymmetry,
        WMUL[1] as f32 / hf_asymmetry,
        &mut plane_y,
    );

    // Add L2Diff for MF channels (all three)
    l2_diff(
        ps0.mf.plane(0),
        ps1.mf.plane(0),
        WMUL[3] as f32,
        &mut plane_x,
    );
    l2_diff(
        ps0.mf.plane(1),
        ps1.mf.plane(1),
        WMUL[4] as f32,
        &mut plane_y,
    );

    // B channel: only L2Diff — use write-only variant (no zero-init needed)
    let mut plane_b = ImageF::new_uninit(width, height);
    l2_diff_write(
        ps0.mf.plane(2),
        ps1.mf.plane(2),
        WMUL[5] as f32,
        &mut plane_b,
    );

    Image3F::from_planes(plane_x, plane_y, plane_b)
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
    // Fused combine_channels + diff_precompute eliminates intermediate buffers
    compute_mask_from_hf_uhf(&ps0.hf, &ps0.uhf, &ps1.hf, &ps1.uhf, diff_ac, pool)
}

/// Combines AC channels with inline DC diff computation from LF planes.
///
/// Fuses compute_lf_diff + combine_channels_to_diffmap into a single pass,
/// eliminating 3 intermediate DC diff plane allocations and memory traffic.
#[archmage::autoversion]
fn combine_channels_to_diffmap_fused(
    _token: archmage::SimdToken,
    mask: &ImageF,
    lf1: &Image3F,
    lf2: &Image3F,
    block_diff_ac: &Image3F,
    xmul: f32,
) -> ImageF {
    let width = mask.width();
    let height = mask.height();
    let mut diffmap = ImageF::new_uninit(width, height);
    let dc_w0 = WMUL[6] as f32;
    let dc_w1 = WMUL[7] as f32;
    let dc_w2 = WMUL[8] as f32;

    // Precompute f32 mask constants for SIMD-friendly inner loop
    let global_scale = crate::consts::GLOBAL_SCALE;
    let my_mul = crate::consts::MASK_Y_MUL as f32;
    let my_scaler = crate::consts::MASK_Y_SCALER as f32;
    let my_offset = crate::consts::MASK_Y_OFFSET as f32;
    let mdc_mul = crate::consts::MASK_DC_Y_MUL as f32;
    let mdc_scaler = crate::consts::MASK_DC_Y_SCALER as f32;
    let mdc_offset = crate::consts::MASK_DC_Y_OFFSET as f32;

    for y in 0..height {
        let mask_row = mask.row(y);
        let lf1_0 = lf1.plane(0).row(y);
        let lf1_1 = lf1.plane(1).row(y);
        let lf1_2 = lf1.plane(2).row(y);
        let lf2_0 = lf2.plane(0).row(y);
        let lf2_1 = lf2.plane(1).row(y);
        let lf2_2 = lf2.plane(2).row(y);
        let ac0 = block_diff_ac.plane(0).row(y);
        let ac1 = block_diff_ac.plane(1).row(y);
        let ac2 = block_diff_ac.plane(2).row(y);
        let out = diffmap.row_mut(y);

        for x in 0..width {
            let val = mask_row[x];

            // mask_y in f32: (global_scale * (1 + mul / (scaler * val + offset)))²
            let c_y = my_mul / my_scaler.mul_add(val, my_offset);
            let r_y = global_scale.mul_add(c_y, global_scale);
            let maskval = r_y * r_y;

            // mask_dc_y in f32
            let c_dc = mdc_mul / mdc_scaler.mul_add(val, mdc_offset);
            let r_dc = global_scale.mul_add(c_dc, global_scale);
            let dc_maskval = r_dc * r_dc;

            // DC diff computed inline: d*d*w for each channel
            let d0 = lf1_0[x] - lf2_0[x];
            let d1 = lf1_1[x] - lf2_1[x];
            let d2 = lf1_2[x] - lf2_2[x];
            let dc_masked = (d0 * d0 * dc_w0 * xmul).mul_add(
                dc_maskval,
                (d1 * d1 * dc_w1).mul_add(dc_maskval, d2 * d2 * dc_w2 * dc_maskval),
            );

            let ac_masked =
                (ac0[x] * xmul).mul_add(maskval, ac1[x].mul_add(maskval, ac2[x] * maskval));

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
#[archmage::autoversion]
fn compute_score_from_diffmap(_token: archmage::SimdToken, diffmap: &ImageF) -> f64 {
    let width = diffmap.width();
    let height = diffmap.height();

    if width * height == 0 {
        return 0.0;
    }

    // Use 8 independent max accumulators to break the loop-carried dependency,
    // enabling LLVM to vectorize with vmaxps (packed max).
    let mut lanes = [0.0f32; 8];
    for y in 0..height {
        let row = diffmap.row(y);
        for chunk in row.chunks_exact(8) {
            for (m, &v) in lanes.iter_mut().zip(chunk.iter()) {
                if v > *m {
                    *m = v;
                }
            }
        }
        for &v in row.chunks_exact(8).remainder() {
            if v > lanes[0] {
                lanes[0] = v;
            }
        }
    }

    let mut max_val = lanes[0];
    for &m in &lanes[1..] {
        if m > max_val {
            max_val = m;
        }
    }
    max_val as f64
}

/// Subsamples linear RGB f32 buffer by 2x for multi-resolution processing.
fn subsample_linear_rgb_2x(rgb: &[f32], width: usize, height: usize) -> (Vec<f32>, usize, usize) {
    let out_width = width.div_ceil(2);
    let out_height = height.div_ceil(2);
    let mut output = vec![0.0f32; out_width * out_height * 3];

    // Interior: full 2x2 blocks (no boundary checks needed)
    let interior_w = width / 2;
    let interior_h = height / 2;
    let inv4 = 0.25f32;

    for oy in 0..interior_h {
        let iy = oy * 2;
        let row0 = iy * width * 3;
        let row1 = (iy + 1) * width * 3;
        for ox in 0..interior_w {
            let ix = ox * 2;
            let i00 = row0 + ix * 3;
            let i10 = row0 + (ix + 1) * 3;
            let i01 = row1 + ix * 3;
            let i11 = row1 + (ix + 1) * 3;
            let out_idx = (oy * out_width + ox) * 3;
            output[out_idx] = (rgb[i00] + rgb[i10] + rgb[i01] + rgb[i11]) * inv4;
            output[out_idx + 1] =
                (rgb[i00 + 1] + rgb[i10 + 1] + rgb[i01 + 1] + rgb[i11 + 1]) * inv4;
            output[out_idx + 2] =
                (rgb[i00 + 2] + rgb[i10 + 2] + rgb[i01 + 2] + rgb[i11 + 2]) * inv4;
        }
    }

    // Right edge column (if width is odd)
    if out_width > interior_w {
        let ox = interior_w;
        let ix = ox * 2;
        for oy in 0..interior_h {
            let iy = oy * 2;
            let i00 = (iy * width + ix) * 3;
            let i01 = ((iy + 1) * width + ix) * 3;
            let out_idx = (oy * out_width + ox) * 3;
            let inv2 = 0.5f32;
            output[out_idx] = (rgb[i00] + rgb[i01]) * inv2;
            output[out_idx + 1] = (rgb[i00 + 1] + rgb[i01 + 1]) * inv2;
            output[out_idx + 2] = (rgb[i00 + 2] + rgb[i01 + 2]) * inv2;
        }
    }

    // Bottom edge row (if height is odd)
    if out_height > interior_h {
        let oy = interior_h;
        let iy = oy * 2;
        let row0 = iy * width * 3;
        for ox in 0..interior_w {
            let ix = ox * 2;
            let i00 = row0 + ix * 3;
            let i10 = row0 + (ix + 1) * 3;
            let out_idx = (oy * out_width + ox) * 3;
            let inv2 = 0.5f32;
            output[out_idx] = (rgb[i00] + rgb[i10]) * inv2;
            output[out_idx + 1] = (rgb[i00 + 1] + rgb[i10 + 1]) * inv2;
            output[out_idx + 2] = (rgb[i00 + 2] + rgb[i10 + 2]) * inv2;
        }
        // Bottom-right corner (if both odd)
        if out_width > interior_w {
            let ox = interior_w;
            let ix = ox * 2;
            let i00 = row0 + ix * 3;
            let out_idx = (oy * out_width + ox) * 3;
            output[out_idx] = rgb[i00];
            output[out_idx + 1] = rgb[i00 + 1];
            output[out_idx + 2] = rgb[i00 + 2];
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
) -> ImageF {
    let intensity_target = params.intensity_target();

    // Parallel: XYB conversion + frequency decomposition for both images
    let (ps1, ps2) = maybe_join(
        || {
            let pool = BufferPool::new();
            let xyb = linear_rgb_to_xyb_image(rgb1, width, height, intensity_target, &pool);
            separate_frequencies(&xyb, &pool)
        },
        || {
            let pool = BufferPool::new();
            let xyb = linear_rgb_to_xyb_image(rgb2, width, height, intensity_target, &pool);
            separate_frequencies(&xyb, &pool)
        },
    );

    // Compute AC differences using Malta filter (internally parallelized)
    let pool = BufferPool::new();
    let mut block_diff_ac =
        compute_psycho_diff_malta(&ps1, &ps2, params.hf_asymmetry(), params.xmul(), &pool);
    let mask = mask_psycho_image(&ps1, &ps2, Some(block_diff_ac.plane_mut(1)), &pool);

    // Combine channels to final diffmap (DC diff computed inline from LF planes)
    combine_channels_to_diffmap_fused(&mask, &ps1.lf, &ps2.lf, &block_diff_ac, params.xmul())
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
) -> ImageF {
    const MIN_SIZE_FOR_SUBSAMPLE: usize = 15;

    let need_sub = !params.single_resolution()
        && width >= MIN_SIZE_FOR_SUBSAMPLE
        && height >= MIN_SIZE_FOR_SUBSAMPLE;

    if need_sub {
        // Parallel: compute full-res and half-res diffmaps simultaneously
        let (sub_diffmap, mut diffmap) = maybe_join(
            || {
                let (sub_rgb1, sw, sh) = subsample_linear_rgb_2x(rgb1, width, height);
                let (sub_rgb2, _, _) = subsample_linear_rgb_2x(rgb2, width, height);
                compute_diffmap_single_resolution_linear(&sub_rgb1, &sub_rgb2, sw, sh, params)
            },
            || compute_diffmap_single_resolution_linear(rgb1, rgb2, width, height, params),
        );

        add_supersampled_2x(&sub_diffmap, 0.5, &mut diffmap);
        diffmap
    } else {
        compute_diffmap_single_resolution_linear(rgb1, rgb2, width, height, params)
    }
}

/// Main implementation of butteraugli comparison (sRGB u8 input).
///
/// Converts sRGB to linear f32 and delegates to the linear path.
/// This ensures all subsampling happens in linear space.
#[cfg(test)]
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

    // Handle very small images without multi-resolution
    let diffmap = if width < MIN_SIZE_FOR_MULTIRESOLUTION || height < MIN_SIZE_FOR_MULTIRESOLUTION {
        compute_diffmap_single_resolution_linear(rgb1, rgb2, width, height, params)
    } else {
        compute_diffmap_multiresolution_linear(rgb1, rgb2, width, height, params)
    };

    // Compute global score
    let score = compute_score_from_diffmap(&diffmap);

    InternalResult {
        score,
        diffmap: Some(diffmap),
    }
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

    // Fused conversion: ImgRef<RGB8> → linear f32 in one pass (avoids
    // intermediate Vec<u8> and Vec<f32> allocations)
    let linear1 = imgref_srgb_to_linear_f32(img1);
    let linear2 = imgref_srgb_to_linear_f32(img2);

    let mut result = compute_butteraugli_linear_impl(&linear1, &linear2, width, height, params);

    if !compute_diffmap {
        result.diffmap = None;
    }

    result
}

/// Converts ImgRef<RGB8> directly to interleaved linear f32, skipping
/// intermediate Vec<u8> allocation.
pub(crate) fn imgref_srgb_to_linear_f32(img: ImgRef<RGB8>) -> Vec<f32> {
    let lut = &*crate::opsin::SRGB_TO_LINEAR_LUT;
    let width = img.width();
    let height = img.height();
    let mut result = Vec::with_capacity(width * height * 3);
    for row in img.rows() {
        for px in row {
            result.push(lut[px.r as usize]);
            result.push(lut[px.g as usize]);
            result.push(lut[px.b as usize]);
        }
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
    let rgb1 = imgref_rgbf32_to_f32_vec(img1);
    let rgb2 = imgref_rgbf32_to_f32_vec(img2);

    // Use the existing proven implementation with multiresolution support
    let mut result = compute_butteraugli_linear_impl(&rgb1, &rgb2, width, height, params);

    // Drop diffmap if not requested
    if !compute_diffmap {
        result.diffmap = None;
    }

    result
}

/// Converts ImgRef<RGB<f32>> to a contiguous Vec<f32> in RGB order.
pub(crate) fn imgref_rgbf32_to_f32_vec(img: ImgRef<RGB<f32>>) -> Vec<f32> {
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
