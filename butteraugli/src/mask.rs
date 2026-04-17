//! Visual masking functions for butteraugli.
//!
//! Visual masking is the phenomenon where the visibility of one feature
//! is reduced by the presence of another feature. This module implements
//! the masking computations used in butteraugli.
//!
//! Key functions:
//! - `combine_channels_for_masking`: Combines HF and UHF channels
//! - `diff_precompute`: Applies sqrt-like transform for numerical stability
//! - `fuzzy_erosion`: Finds smooth areas using weighted minimum
//! - `mask_y`: Converts mask value to AC masking factor
//! - `mask_dc_y`: Converts mask value to DC masking factor

use crate::blur::gaussian_blur;
use crate::consts::{
    COMBINE_CHANNELS_MULS, GLOBAL_SCALE, MASK_BIAS, MASK_DC_Y_MUL, MASK_DC_Y_OFFSET,
    MASK_DC_Y_SCALER, MASK_MUL, MASK_RADIUS, MASK_TO_ERROR_MUL, MASK_Y_MUL, MASK_Y_OFFSET,
    MASK_Y_SCALER,
};
use crate::image::{BufferPool, ImageF};

/// Combines HF and UHF channels for masking computation.
///
/// Only X and Y components are involved in masking. B's influence
/// is considered less important in the high frequency area.
///
/// Matches C++ CombineChannelsForMasking (butteraugli.cc lines 1108-1132).
#[archmage::autoversion(v4x, v4, v3, neon, wasm128, scalar)]
pub fn combine_channels_for_masking(
    _token: archmage::SimdToken,
    hf: &[ImageF; 2],
    uhf: &[ImageF; 2],
    out: &mut ImageF,
) {
    let width = hf[0].width();
    let height = hf[0].height();

    for y in 0..height {
        let row_y_hf = hf[1].row(y);
        let row_y_uhf = uhf[1].row(y);
        let row_x_hf = hf[0].row(y);
        let row_x_uhf = uhf[0].row(y);
        let row_out = out.row_mut(y);

        for x in 0..width {
            // C++: xdiff = (row_x_uhf[x] + row_x_hf[x]) * muls[0]
            let xdiff = (row_x_uhf[x] + row_x_hf[x]) * COMBINE_CHANNELS_MULS[0];
            // C++: ydiff = row_y_uhf[x] * muls[1] + row_y_hf[x] * muls[2]
            let ydiff = row_y_uhf[x].mul_add(
                COMBINE_CHANNELS_MULS[1],
                row_y_hf[x] * COMBINE_CHANNELS_MULS[2],
            );
            // C++: row[x] = sqrt(xdiff * xdiff + ydiff * ydiff)
            row_out[x] = xdiff.mul_add(xdiff, ydiff * ydiff).sqrt();
        }
    }
}

/// Precomputes difference values for masking.
///
/// Applies sqrt-like transformation to make values more perceptually uniform.
/// Matches C++ DiffPrecompute (butteraugli.cc lines 1134-1147).
#[archmage::autoversion(v4x, v4, v3, neon, wasm128, scalar)]
pub fn diff_precompute(
    _token: archmage::SimdToken,
    xyb: &ImageF,
    mul: f32,
    bias_arg: f32,
    out: &mut ImageF,
) {
    let width = xyb.width();
    let height = xyb.height();
    let bias = mul * bias_arg;
    let sqrt_bias = bias.sqrt();

    for y in 0..height {
        let row_in = xyb.row(y);
        let row_out = out.row_mut(y);
        for x in 0..width {
            // C++: sqrt(mul * abs(row_in[x]) + bias) - sqrt_bias
            row_out[x] = mul.mul_add(row_in[x].abs(), bias).sqrt() - sqrt_bias;
        }
    }
}

/// Stores the three smallest values encountered.
/// Matches C++ StoreMin3 (butteraugli.cc lines 1155-1168).
#[inline]
fn store_min3(v: f32, min0: &mut f32, min1: &mut f32, min2: &mut f32) {
    if v < *min2 {
        if v < *min0 {
            *min2 = *min1;
            *min1 = *min0;
            *min0 = v;
        } else if v < *min1 {
            *min2 = *min1;
            *min1 = v;
        } else {
            *min2 = v;
        }
    }
}

/// Performs fuzzy erosion to find smooth areas.
///
/// Look for smooth areas near the area of degradation.
/// If the areas are generally smooth, don't apply masking.
///
/// Matches C++ FuzzyErosion (butteraugli.cc lines 1170-1208).
pub fn fuzzy_erosion(from: &ImageF, to: &mut ImageF) {
    let width = from.width();
    let height = from.height();
    const K: usize = 3;

    // Border rows: y < K or y >= height - K (need Option checks for up/down)
    for y in 0..K.min(height) {
        fuzzy_erosion_border_row(from, to, y, width, height);
    }

    // Interior rows: all 3 neighbor rows exist
    for y in K..height.saturating_sub(K) {
        let row_c = from.row(y);
        let row_up = from.row(y - K);
        let row_dn = from.row(y + K);
        let out_row = to.row_mut(y);

        // Left border: x < K
        for x in 0..K.min(width) {
            fuzzy_erosion_pixel_interior_y(row_c, row_up, row_dn, out_row, x, width);
        }

        // Interior: branch-free min/max, auto-vectorizable
        fuzzy_erosion_interior_row(row_c, row_up, row_dn, out_row, width);

        // Right border: x >= width - K
        for x in width.saturating_sub(K)..width {
            if x >= K {
                fuzzy_erosion_pixel_interior_y(row_c, row_up, row_dn, out_row, x, width);
            }
        }
    }

    // Bottom border rows
    for y in height.saturating_sub(K)..height {
        if y >= K {
            fuzzy_erosion_border_row(from, to, y, width, height);
        }
    }
}

/// Branch-free min3 update: inserts v into the sorted triple (min0, min1, min2).
///
/// Uses min/max instead of branches for SIMD-friendly compilation.
#[inline]
fn update_min3(v: f32, min0: f32, min1: f32, min2: f32) -> (f32, f32, f32) {
    let new0 = min0.min(v);
    let pushed = min0.max(v);
    let new1 = min1.min(pushed);
    let pushed2 = min1.max(pushed);
    let new2 = min2.min(pushed2);
    (new0, new1, new2)
}

/// Auto-vectorized interior row processing for fuzzy erosion.
///
/// Processes x in [K, width-K) where all 9 neighbors are guaranteed to exist.
/// Uses branch-free min/max operations that compile to SIMD vminps/vmaxps.
#[archmage::autoversion(v4x, v4, v3, neon, wasm128, scalar)]
fn fuzzy_erosion_interior_row(
    _token: archmage::SimdToken,
    row_c: &[f32],
    row_up: &[f32],
    row_dn: &[f32],
    out: &mut [f32],
    width: usize,
) {
    const K: usize = 3;
    let end = width.saturating_sub(K);
    // Pre-slice to eliminate bounds checks in the hot loop
    let c_left = &row_c[..end.saturating_sub(K)];
    let c_mid = &row_c[K..end];
    let c_right = &row_c[K + K..];
    let u_left = &row_up[..end.saturating_sub(K)];
    let u_mid = &row_up[K..end];
    let u_right = &row_up[K + K..];
    let d_left = &row_dn[..end.saturating_sub(K)];
    let d_mid = &row_dn[K..end];
    let d_right = &row_dn[K + K..];
    let out_slice = &mut out[K..end];

    for i in 0..out_slice.len() {
        let c = c_mid[i];
        let init = 2.0 * c;
        let (m0, m1, m2) = (c, init, init);

        let (m0, m1, m2) = update_min3(c_left[i], m0, m1, m2);
        let (m0, m1, m2) = update_min3(u_left[i], m0, m1, m2);
        let (m0, m1, m2) = update_min3(d_left[i], m0, m1, m2);
        let (m0, m1, m2) = update_min3(c_right[i], m0, m1, m2);
        let (m0, m1, m2) = update_min3(u_right[i], m0, m1, m2);
        let (m0, m1, m2) = update_min3(d_right[i], m0, m1, m2);
        let (m0, m1, m2) = update_min3(u_mid[i], m0, m1, m2);
        let (m0, m1, _m2) = update_min3(d_mid[i], m0, m1, m2);

        out_slice[i] = (0.45f32).mul_add(m0, (0.3f32).mul_add(m1, 0.25 * _m2));
    }
}

/// Process a single pixel with all 3 vertical neighbor rows available.
/// Still needs x-boundary checks.
#[inline]
fn fuzzy_erosion_pixel_interior_y(
    row_c: &[f32],
    row_up: &[f32],
    row_dn: &[f32],
    out_row: &mut [f32],
    x: usize,
    width: usize,
) {
    const K: usize = 3;
    let mut min0 = row_c[x];
    let mut min1 = 2.0 * min0;
    let mut min2 = min1;

    if x >= K {
        store_min3(row_c[x - K], &mut min0, &mut min1, &mut min2);
        store_min3(row_up[x - K], &mut min0, &mut min1, &mut min2);
        store_min3(row_dn[x - K], &mut min0, &mut min1, &mut min2);
    }
    if x + K < width {
        store_min3(row_c[x + K], &mut min0, &mut min1, &mut min2);
        store_min3(row_up[x + K], &mut min0, &mut min1, &mut min2);
        store_min3(row_dn[x + K], &mut min0, &mut min1, &mut min2);
    }
    store_min3(row_up[x], &mut min0, &mut min1, &mut min2);
    store_min3(row_dn[x], &mut min0, &mut min1, &mut min2);

    out_row[x] = 0.45 * min0 + 0.3 * min1 + 0.25 * min2;
}

/// Process a border row where up/down neighbors may not exist.
#[inline(never)]
fn fuzzy_erosion_border_row(from: &ImageF, to: &mut ImageF, y: usize, width: usize, height: usize) {
    const K: usize = 3;
    let row_c = from.row(y);
    let row_up = if y >= K { Some(from.row(y - K)) } else { None };
    let row_dn = if y + K < height {
        Some(from.row(y + K))
    } else {
        None
    };
    let out_row = to.row_mut(y);

    for x in 0..width {
        let mut min0 = row_c[x];
        let mut min1 = 2.0 * min0;
        let mut min2 = min1;

        if x >= K {
            store_min3(row_c[x - K], &mut min0, &mut min1, &mut min2);
            if let Some(r) = row_up {
                store_min3(r[x - K], &mut min0, &mut min1, &mut min2);
            }
            if let Some(r) = row_dn {
                store_min3(r[x - K], &mut min0, &mut min1, &mut min2);
            }
        }
        if x + K < width {
            store_min3(row_c[x + K], &mut min0, &mut min1, &mut min2);
            if let Some(r) = row_up {
                store_min3(r[x + K], &mut min0, &mut min1, &mut min2);
            }
            if let Some(r) = row_dn {
                store_min3(r[x + K], &mut min0, &mut min1, &mut min2);
            }
        }
        if let Some(r) = row_up {
            store_min3(r[x], &mut min0, &mut min1, &mut min2);
        }
        if let Some(r) = row_dn {
            store_min3(r[x], &mut min0, &mut min1, &mut min2);
        }

        out_row[x] = 0.45 * min0 + 0.3 * min1 + 0.25 * min2;
    }
}

/// Converts mask value to AC masking multiplier.
///
/// Matches C++ MaskY (butteraugli.cc lines 1266-1273).
#[inline]
pub fn mask_y(delta: f64) -> f64 {
    let c = MASK_Y_MUL / (MASK_Y_SCALER * delta + MASK_Y_OFFSET);
    let retval = GLOBAL_SCALE as f64 * (1.0 + c);
    retval * retval
}

/// Converts mask value to DC masking multiplier.
///
/// Matches C++ MaskDcY (butteraugli.cc lines 1275-1282).
#[inline]
pub fn mask_dc_y(delta: f64) -> f64 {
    let c = MASK_DC_Y_MUL / (MASK_DC_Y_SCALER * delta + MASK_DC_Y_OFFSET);
    let retval = GLOBAL_SCALE as f64 * (1.0 + c);
    retval * retval
}

/// Fused combine_channels_for_masking + diff_precompute.
///
/// Reads 4 input planes (hf[0], hf[1], uhf[0], uhf[1]), writes 1 output plane.
/// Eliminates intermediate buffer and one full read+write pass per image.
#[archmage::autoversion(v4x, v4, v3, neon, wasm128, scalar)]
fn combine_and_precompute(
    _token: archmage::SimdToken,
    hf: &[ImageF; 2],
    uhf: &[ImageF; 2],
    out: &mut ImageF,
) {
    let width = hf[0].width();
    let height = hf[0].height();
    let bias = MASK_MUL * MASK_BIAS;
    let sqrt_bias = bias.sqrt();

    for y in 0..height {
        let row_y_hf = hf[1].row(y);
        let row_y_uhf = uhf[1].row(y);
        let row_x_hf = hf[0].row(y);
        let row_x_uhf = uhf[0].row(y);
        let row_out = out.row_mut(y);

        for x in 0..width {
            let xdiff = (row_x_uhf[x] + row_x_hf[x]) * COMBINE_CHANNELS_MULS[0];
            let ydiff = row_y_uhf[x].mul_add(
                COMBINE_CHANNELS_MULS[1],
                row_y_hf[x] * COMBINE_CHANNELS_MULS[2],
            );
            let combined = xdiff.mul_add(xdiff, ydiff * ydiff).sqrt();
            // combined >= 0, so abs() in diff_precompute is a no-op
            row_out[x] = MASK_MUL.mul_add(combined, bias).sqrt() - sqrt_bias;
        }
    }
}

/// Computes mask directly from HF/UHF frequency bands.
///
/// Fuses combine_channels_for_masking + diff_precompute into a single pass,
/// eliminating two intermediate ImageF allocations and ~4MB memory traffic.
pub fn compute_mask_from_hf_uhf(
    hf0: &[ImageF; 2],
    uhf0: &[ImageF; 2],
    hf1: &[ImageF; 2],
    uhf1: &[ImageF; 2],
    diff_ac: Option<&mut ImageF>,
    pool: &BufferPool,
) -> ImageF {
    let width = hf0[0].width();
    let height = hf0[0].height();

    // Fused combine + precompute for image 0
    let mut diff0 = ImageF::from_pool_dirty(width, height, pool);
    combine_and_precompute(hf0, uhf0, &mut diff0);

    // Fused combine + precompute for image 1
    let mut diff1 = ImageF::from_pool_dirty(width, height, pool);
    combine_and_precompute(hf1, uhf1, &mut diff1);

    // Blur diff0 and diff1
    let blurred0 = gaussian_blur(&diff0, MASK_RADIUS, pool);
    let blurred1 = gaussian_blur(&diff1, MASK_RADIUS, pool);
    diff0.recycle(pool);
    diff1.recycle(pool);

    // FuzzyErosion on blurred0 — result IS the mask (no copy needed)
    let mut mask = ImageF::from_pool_dirty(width, height, pool);
    fuzzy_erosion(&blurred0, &mut mask);

    // Accumulate mask-to-error difference into diff_ac if requested
    if let Some(ac) = diff_ac {
        accumulate_mask_to_error(&blurred0, &blurred1, ac);
    }

    blurred0.recycle(pool);
    blurred1.recycle(pool);
    mask
}

/// Autoversioned mask-to-error accumulation: ac[x] += MUL * (b0[x] - b1[x])^2.
#[archmage::autoversion(v4x, v4, v3, neon, wasm128, scalar)]
fn accumulate_mask_to_error(
    _token: archmage::SimdToken,
    b0: &ImageF,
    b1: &ImageF,
    ac: &mut ImageF,
) {
    let height = b0.height();
    for y in 0..height {
        let row0 = b0.row(y);
        let row1 = b1.row(y);
        let ac_row = ac.row_mut(y);
        for ((a, &v0), &v1) in ac_row.iter_mut().zip(row0.iter()).zip(row1.iter()) {
            let diff = v0 - v1;
            *a = (diff * diff).mul_add(MASK_TO_ERROR_MUL, *a);
        }
    }
}

/// Precomputed reference-side mask data.
///
/// Stores the results of combine+precompute → blur → fuzzy_erosion for the
/// reference image's HF/UHF bands, so these can be reused across multiple
/// comparisons without recomputation.
#[derive(Clone)]
pub struct PrecomputedMask {
    /// Fuzzy-eroded blurred mask (used as the final mask in combine_channels_to_diffmap)
    pub mask: ImageF,
    /// Blurred combined channels (used for accumulate_mask_to_error)
    pub blurred: ImageF,
}

/// Precomputes the reference-side mask from HF/UHF frequency bands.
///
/// This performs combine_and_precompute + gaussian_blur + fuzzy_erosion on the
/// reference image only. The result can be stored and reused for every comparison.
pub fn precompute_reference_mask(
    hf: &[ImageF; 2],
    uhf: &[ImageF; 2],
    pool: &BufferPool,
) -> PrecomputedMask {
    let width = hf[0].width();
    let height = hf[0].height();

    let mut diff = ImageF::from_pool_dirty(width, height, pool);
    combine_and_precompute(hf, uhf, &mut diff);

    let blurred = gaussian_blur(&diff, MASK_RADIUS, pool);
    diff.recycle(pool);

    let mut mask = ImageF::from_pool_dirty(width, height, pool);
    fuzzy_erosion(&blurred, &mut mask);

    PrecomputedMask { mask, blurred }
}

/// Applies the distorted-side mask correction using precomputed reference data.
///
/// Runs combine_and_precompute + blur on the distorted image's HF/UHF bands,
/// then accumulates the mask-to-error difference into `diff_ac` if provided.
/// Does NOT copy the precomputed mask — callers should use `precomputed.mask`
/// directly as a read-only reference.
pub fn apply_mask_correction_precomputed(
    precomputed: &PrecomputedMask,
    hf1: &[ImageF; 2],
    uhf1: &[ImageF; 2],
    diff_ac: Option<&mut ImageF>,
    pool: &BufferPool,
) {
    let width = hf1[0].width();
    let height = hf1[0].height();

    // Only compute the distorted side
    let mut diff1 = ImageF::from_pool_dirty(width, height, pool);
    combine_and_precompute(hf1, uhf1, &mut diff1);

    let blurred1 = gaussian_blur(&diff1, MASK_RADIUS, pool);
    diff1.recycle(pool);

    // Accumulate mask-to-error using precomputed reference blur
    if let Some(ac) = diff_ac {
        accumulate_mask_to_error(&precomputed.blurred, &blurred1, ac);
    }

    blurred1.recycle(pool);
}

/// Computes mask from both images' psychovisual representations.
///
/// Matches C++ Mask function (butteraugli.cc lines 1212-1247).
pub fn compute_mask(
    mask0: &ImageF,
    mask1: &ImageF,
    diff_ac: Option<&mut ImageF>,
    pool: &BufferPool,
) -> ImageF {
    let width = mask0.width();
    let height = mask0.height();

    // DiffPrecompute for mask0
    let mut diff0 = ImageF::from_pool_dirty(width, height, pool);
    diff_precompute(mask0, MASK_MUL, MASK_BIAS, &mut diff0);

    // DiffPrecompute for mask1
    let mut diff1 = ImageF::from_pool_dirty(width, height, pool);
    diff_precompute(mask1, MASK_MUL, MASK_BIAS, &mut diff1);

    // Blur diff0 and diff1
    let blurred0 = gaussian_blur(&diff0, MASK_RADIUS, pool);
    let blurred1 = gaussian_blur(&diff1, MASK_RADIUS, pool);
    diff0.recycle(pool);
    diff1.recycle(pool);

    // FuzzyErosion on blurred0 — result IS the mask (no copy needed)
    let mut mask = ImageF::from_pool_dirty(width, height, pool);
    fuzzy_erosion(&blurred0, &mut mask);

    // Accumulate mask-to-error difference into diff_ac if requested
    if let Some(ac) = diff_ac {
        accumulate_mask_to_error(&blurred0, &blurred1, ac);
    }

    blurred0.recycle(pool);
    blurred1.recycle(pool);
    mask
}

/// Applies visual masking based on local contrast.
///
/// Higher local contrast means differences are less visible (masked).
pub fn apply_masking(diff: &ImageF, mask: &ImageF, out: &mut ImageF) {
    let width = diff.width();
    let height = diff.height();

    for y in 0..height {
        let row_diff = diff.row(y);
        let row_mask = mask.row(y);
        let row_out = out.row_mut(y);

        for x in 0..width {
            // Higher mask value means lower sensitivity
            let sensitivity = 1.0 / (1.0 + row_mask[x]);
            row_out[x] = row_diff[x] * sensitivity;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_min3() {
        let mut min0 = 10.0f32;
        let mut min1 = 20.0f32;
        let mut min2 = 30.0f32;

        store_min3(5.0, &mut min0, &mut min1, &mut min2);
        assert!((min0 - 5.0).abs() < 0.001);
        assert!((min1 - 10.0).abs() < 0.001);
        assert!((min2 - 20.0).abs() < 0.001);

        store_min3(15.0, &mut min0, &mut min1, &mut min2);
        assert!((min0 - 5.0).abs() < 0.001);
        assert!((min1 - 10.0).abs() < 0.001);
        assert!((min2 - 15.0).abs() < 0.001);
    }

    #[test]
    fn test_fuzzy_erosion() {
        let mut from = ImageF::new(16, 16);
        // Create a pattern with one bright spot
        from.set(8, 8, 10.0);
        from.set(5, 5, 5.0);

        let mut to = ImageF::new(16, 16);
        fuzzy_erosion(&from, &mut to);

        // The output should be smoother
        // The bright spot should be somewhat reduced
        assert!(to.get(8, 8) <= 10.0);
    }

    #[test]
    fn test_diff_precompute() {
        let input = ImageF::filled(16, 16, 1.0);
        let mut output = ImageF::new(16, 16);

        diff_precompute(&input, 1.0, 0.01, &mut output);

        // All values should be positive
        for y in 0..16 {
            for x in 0..16 {
                assert!(output.get(x, y) >= 0.0);
            }
        }
    }

    #[test]
    fn test_mask_y() {
        // Test MaskY function
        let result = mask_y(1.0);
        assert!(result > 0.0);
        assert!(result.is_finite());

        // Higher delta should result in lower masking
        let result_high = mask_y(10.0);
        assert!(result_high < result);
    }

    #[test]
    fn test_mask_dc_y() {
        // Test MaskDcY function
        let result = mask_dc_y(1.0);
        assert!(result > 0.0);
        assert!(result.is_finite());

        // Higher delta should result in lower masking
        let result_high = mask_dc_y(10.0);
        assert!(result_high < result);
    }

    #[test]
    fn test_fuzzy_erosion_weights() {
        // Test that weights sum to 1.0
        let weights_sum: f64 = 0.45 + 0.3 + 0.25;
        assert!((weights_sum - 1.0).abs() < 0.001);
    }
}

#[test]
fn test_mask_y_cpp_values() {
    // Verify MaskY matches C++ implementation
    // C++ calculation for delta=1.0:
    // c = 2.5485944793 / (0.451936922203 * 1.0 + 0.829591754942) = 1.989
    // retval = 0.0709 * (1.0 + 1.989) = 0.2119
    // return 0.2119^2 = 0.0449

    let result = mask_y(1.0);
    println!("MaskY(1.0) = {result}");

    // Calculate expected value
    let offset = 0.829591754942;
    let scaler = 0.451936922203;
    let mul = 2.5485944793;
    let global_scale = 1.0 / (17.83 * 0.790799174);

    let c = mul / (scaler * 1.0 + offset);
    let retval = global_scale * (1.0 + c);
    let expected = retval * retval;

    println!("Expected: {expected}, c={c}, retval={retval}, global_scale={global_scale}");

    assert!(
        (result - expected).abs() < 1e-6,
        "MaskY(1.0) = {result}, expected {expected}"
    );
}
