//! Multi-scale psychovisual decomposition for butteraugli.
//!
//! The PsychoImage struct holds frequency-decomposed versions of an image:
//! - UHF (Ultra High Frequency): Very fine details
//! - HF (High Frequency): Fine edges and textures
//! - MF (Medium Frequency): Larger-scale textures
//! - LF (Low Frequency): Smooth gradients and base colors
//!
//! This decomposition allows butteraugli to weight different spatial
//! frequencies according to human visual sensitivity.

use crate::blur::gaussian_blur;
use crate::consts::{
    ADD_HF_RANGE, ADD_MF_RANGE, BMUL_LF_TO_VALS, MAXCLAMP_HF, MAXCLAMP_UHF, MUL_Y_HF, MUL_Y_UHF,
    REMOVE_HF_RANGE, REMOVE_MF_RANGE, REMOVE_UHF_RANGE, SIGMA_HF, SIGMA_LF, SIGMA_UHF, SUPPRESS_S,
    SUPPRESS_XY, XMUL_LF_TO_VALS, Y_TO_B_MUL_LF_TO_VALS, YMUL_LF_TO_VALS,
};
use crate::diff::maybe_join;
use crate::image::{BufferPool, Image3F, ImageF};

/// Multi-scale psychovisual decomposition of an image.
///
/// Each frequency band captures different spatial features:
/// - `uhf`: Ultra high frequency (X, Y channels only)
/// - `hf`: High frequency (X, Y channels only)
/// - `mf`: Medium frequency (X, Y, B channels)
/// - `lf`: Low frequency (X, Y, B channels)
#[derive(Debug, Clone)]
pub struct PsychoImage {
    /// Ultra high frequency components (X, Y channels)
    pub uhf: [ImageF; 2],
    /// High frequency components (X, Y channels)
    pub hf: [ImageF; 2],
    /// Medium frequency components (X, Y, B channels)
    pub mf: Image3F,
    /// Low frequency components (X, Y, B channels)
    pub lf: Image3F,
}

impl PsychoImage {
    /// Creates a new PsychoImage with empty/zero images.
    #[cfg(test)]
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        // All planes are fully overwritten by separate_frequencies, skip zeroing
        Self {
            uhf: [
                ImageF::new_uninit(width, height),
                ImageF::new_uninit(width, height),
            ],
            hf: [
                ImageF::new_uninit(width, height),
                ImageF::new_uninit(width, height),
            ],
            mf: Image3F::new_uninit(width, height),
            lf: Image3F::new_uninit(width, height),
        }
    }

    /// Creates a new PsychoImage using pooled buffers (dirty, caller must overwrite).
    #[must_use]
    pub(crate) fn from_pool(width: usize, height: usize, pool: &BufferPool) -> Self {
        Self {
            uhf: [
                ImageF::from_pool_dirty(width, height, pool),
                ImageF::from_pool_dirty(width, height, pool),
            ],
            hf: [
                ImageF::from_pool_dirty(width, height, pool),
                ImageF::from_pool_dirty(width, height, pool),
            ],
            mf: Image3F::from_pool_dirty(width, height, pool),
            lf: Image3F::from_pool_dirty(width, height, pool),
        }
    }

    /// Recycles all internal buffers back to the pool.
    pub(crate) fn recycle(self, pool: &BufferPool) {
        let [uhf0, uhf1] = self.uhf;
        uhf0.recycle(pool);
        uhf1.recycle(pool);
        let [hf0, hf1] = self.hf;
        hf0.recycle(pool);
        hf1.recycle(pool);
        self.mf.recycle(pool);
        self.lf.recycle(pool);
    }

    /// Width of the images.
    #[must_use]
    pub fn width(&self) -> usize {
        self.lf.width()
    }

    /// Height of the images.
    #[must_use]
    pub fn height(&self) -> usize {
        self.lf.height()
    }
}

/// Removes values in a small range around zero.
///
/// Makes the area around zero less important by clamping small values to zero.
#[inline]
fn remove_range_around_zero(x: f32, range: f32) -> f32 {
    if x > range {
        x - range
    } else if x < -range {
        x + range
    } else {
        0.0
    }
}

/// Amplifies values in a small range around zero.
///
/// Makes the area around zero more important by doubling small values.
#[inline]
fn amplify_range_around_zero(x: f32, range: f32) -> f32 {
    if x > range {
        x + range
    } else if x < -range {
        x - range
    } else {
        x * 2.0
    }
}

/// Maximum clamp function from C++ butteraugli.
///
/// Compresses extreme values to prevent outliers from dominating.
/// Note: uses manual multiply+add instead of mul_add to avoid fmaf
/// library call overhead (this runs outside #[rite] context).
#[inline]
fn maximum_clamp(v: f32, max_val: f32) -> f32 {
    const MUL: f32 = 0.724_216_146;
    if v >= max_val {
        (v - max_val) * MUL + max_val
    } else if v <= -max_val {
        (v + max_val) * MUL + (-max_val)
    } else {
        v
    }
}

/// Converts low-frequency XYB to "vals" space for comparison.
///
/// Vals space can be converted to L2-norm space through visual masking.
fn xyb_low_freq_to_vals(lf: &mut Image3F) {
    let width = lf.width();
    let height = lf.height();
    let (p0, p1, p2) = lf.planes_mut();

    for y in 0..height {
        let row_x = p0.row_mut(y);
        let row_y = p1.row_mut(y);
        let row_b = p2.row_mut(y);

        for x in 0..width {
            let vx = row_x[x];
            let vy = row_y[x];
            let vb = row_b[x];

            let b = (Y_TO_B_MUL_LF_TO_VALS as f32) * vy + vb;
            row_b[x] = b * BMUL_LF_TO_VALS as f32;
            row_x[x] = vx * XMUL_LF_TO_VALS as f32;
            row_y[x] = vy * YMUL_LF_TO_VALS as f32;
        }
    }
}

/// Suppresses X channel based on Y channel values.
///
/// High Y (luminance) values reduce sensitivity to X (chroma) differences.
#[archmage::autoversion]
fn suppress_x_by_y(_token: archmage::SimdToken, in_y: &ImageF, inout_x: &mut ImageF) {
    let height = in_y.height();
    let s = SUPPRESS_S as f32;
    let one_minus_s = 1.0 - s;
    let yw = SUPPRESS_XY as f32;

    for y in 0..height {
        let row_y = in_y.row(y);
        let row_x = inout_x.row_mut(y);

        for (vx, &vy) in row_x.iter_mut().zip(row_y.iter()) {
            let scaler = (yw / vy.mul_add(vy, yw)).mul_add(one_minus_s, s);
            *vx *= scaler;
        }
    }
}

/// Applies remove_range_around_zero: dst[x] = copysign(max(|src[x]| - range, 0), src[x]).
///
/// Branch-free formulation for SIMD vectorization.
#[archmage::autoversion]
fn apply_remove_range(_token: archmage::SimdToken, src: &ImageF, range: f32, dst: &mut ImageF) {
    for y in 0..src.height() {
        let row_in = src.row(y);
        let row_out = dst.row_mut(y);
        for (out, &v) in row_out.iter_mut().zip(row_in.iter()) {
            // Branch-free: copysign(max(|v| - range, 0.0), v)
            let abs_v = v.abs();
            let reduced = abs_v - range;
            let clamped = if reduced > 0.0 { reduced } else { 0.0 };
            *out = clamped.copysign(v);
        }
    }
}

/// Applies amplify_range_around_zero: dst[x] = src[x] + copysign(min(|src[x]|, range), src[x]).
///
/// Branch-free formulation for SIMD vectorization.
#[archmage::autoversion]
fn apply_amplify_range(_token: archmage::SimdToken, src: &ImageF, range: f32, dst: &mut ImageF) {
    for y in 0..src.height() {
        let row_in = src.row(y);
        let row_out = dst.row_mut(y);
        for (out, &v) in row_out.iter_mut().zip(row_in.iter()) {
            // Branch-free: v + copysign(min(|v|, range), v)
            let abs_v = v.abs();
            let boost = if abs_v < range { abs_v } else { range };
            *out = v + boost.copysign(v);
        }
    }
}

/// Subtracts two images: dst[x] = a[x] - b[x].
#[archmage::autoversion]
fn subtract_images(_token: archmage::SimdToken, a: &ImageF, b: &ImageF, dst: &mut ImageF) {
    for y in 0..a.height() {
        let ra = a.row(y);
        let rb = b.row(y);
        let rd = dst.row_mut(y);
        for ((d, &va), &vb) in rd.iter_mut().zip(ra.iter()).zip(rb.iter()) {
            *d = va - vb;
        }
    }
}

/// Minimum pixel count to parallelize blur planes within frequency separation.
/// Below this threshold, the overhead of spawning tasks exceeds the benefit.
const MIN_PIXELS_FOR_BLUR_PARALLEL: usize = 768 * 768;

/// Separates LF (low frequency) and MF (medium frequency) components.
fn separate_lf_and_mf(xyb: &Image3F, lf: &mut Image3F, mf: &mut Image3F, pool: &BufferPool) {
    let sigma = SIGMA_LF as f32;
    let width = xyb.width();
    let height = xyb.height();

    if width * height >= MIN_PIXELS_FOR_BLUR_PARALLEL {
        // Blur all 3 planes in parallel (each plane is independent)
        let (lf0, lf1, lf2) = lf.planes_mut();
        let (mf0, mf1, mf2) = mf.planes_mut();

        // Swap blurred result directly into LF (no copy), compute MF = orig - LF
        let blur_plane = |xyb_plane: &ImageF, lf_out: &mut ImageF, mf_out: &mut ImageF| {
            let mut blurred = gaussian_blur(xyb_plane, sigma, pool);
            // Swap blurred into lf_out — both have same dimensions, avoids copy
            core::mem::swap(lf_out, &mut blurred);
            blurred.recycle(pool); // recycle the old dirty lf_out buffer
            subtract_images(xyb_plane, lf_out, mf_out);
        };

        maybe_join(
            || blur_plane(xyb.plane(0), lf0, mf0),
            || {
                maybe_join(
                    || blur_plane(xyb.plane(1), lf1, mf1),
                    || blur_plane(xyb.plane(2), lf2, mf2),
                )
            },
        );
    } else {
        for i in 0..3 {
            let mut blurred = gaussian_blur(xyb.plane(i), sigma, pool);
            // Swap blurred into LF plane — avoids full-image copy
            let lf_plane = lf.plane_mut(i);
            core::mem::swap(lf_plane, &mut blurred);
            blurred.recycle(pool);
            // MF = original - LF
            subtract_images(xyb.plane(i), lf.plane(i), mf.plane_mut(i));
        }
    }

    // Convert LF to vals space
    xyb_low_freq_to_vals(lf);
}

/// Processes one MF→HF channel: blur, subtract, apply range function.
///
/// Fused approach: blur once, then compute HF = original - blurred and
/// MF = range(blurred) in two passes, eliminating 2 full-image copies.
fn separate_mf_hf_channel(
    mf_plane: &mut ImageF,
    hf_plane: &mut ImageF,
    sigma: f32,
    range: f32,
    use_amplify: bool,
    pool: &BufferPool,
) {
    // Blur the original MF plane
    let blurred = gaussian_blur(mf_plane, sigma, pool);

    // HF = orig - blurred (autoversioned SIMD subtraction)
    subtract_images(mf_plane, &blurred, hf_plane);

    // MF = range_adjusted(blurred)
    if use_amplify {
        apply_amplify_range(&blurred, range, mf_plane);
    } else {
        apply_remove_range(&blurred, range, mf_plane);
    }

    blurred.recycle(pool);
}

/// Separates MF (medium frequency) and HF (high frequency) components.
fn separate_mf_and_hf(mf: &mut Image3F, hf: &mut [ImageF; 2], pool: &BufferPool) {
    let width = mf.width();
    let height = mf.height();
    let sigma = SIGMA_HF as f32;

    if width * height >= MIN_PIXELS_FOR_BLUR_PARALLEL {
        // Split MF planes and HF arrays for parallel mutable access
        let (mf0, mf1, mf2) = mf.planes_mut();
        let (hf_x_slice, hf_y_slice) = hf.split_at_mut(1);
        let hf_x = &mut hf_x_slice[0];
        let hf_y = &mut hf_y_slice[0];

        maybe_join(
            || separate_mf_hf_channel(mf0, hf_x, sigma, REMOVE_MF_RANGE as f32, false, pool),
            || {
                maybe_join(
                    || separate_mf_hf_channel(mf1, hf_y, sigma, ADD_MF_RANGE as f32, true, pool),
                    || {
                        let mut blurred_b = gaussian_blur(mf2, sigma, pool);
                        core::mem::swap(mf2, &mut blurred_b);
                        blurred_b.recycle(pool);
                    },
                )
            },
        );

        suppress_x_by_y(hf_y, hf_x);
    } else {
        // Sequential path for small images
        for i in 0..2 {
            let range = if i == 0 {
                REMOVE_MF_RANGE
            } else {
                ADD_MF_RANGE
            };
            let use_amplify = i == 1;
            separate_mf_hf_channel(
                mf.plane_mut(i),
                &mut hf[i],
                sigma,
                range as f32,
                use_amplify,
                pool,
            );
        }
        let mut blurred_b = gaussian_blur(mf.plane(2), sigma, pool);
        core::mem::swap(mf.plane_mut(2), &mut blurred_b);
        blurred_b.recycle(pool);
        let (hf_x, hf_y) = hf.split_at_mut(1);
        suppress_x_by_y(&hf_y[0], &mut hf_x[0]);
    }
}

/// Separates HF (high frequency) and UHF (ultra high frequency) components.
fn separate_hf_and_uhf(hf: &mut [ImageF; 2], uhf: &mut [ImageF; 2], pool: &BufferPool) {
    let width = hf[0].width();
    let height = hf[0].height();
    let sigma = SIGMA_UHF as f32;

    if width * height >= MIN_PIXELS_FOR_BLUR_PARALLEL {
        // Split arrays for parallel mutable access
        let (hf_x_slice, hf_y_slice) = hf.split_at_mut(1);
        let (uhf_x_slice, uhf_y_slice) = uhf.split_at_mut(1);

        maybe_join(
            || {
                let hf_x = &mut hf_x_slice[0];
                let uhf_x = &mut uhf_x_slice[0];
                let blurred = gaussian_blur(hf_x, sigma, pool);

                // Fused: UHF = range(orig - blurred), HF = range(blurred)
                for y in 0..height {
                    let row_orig = hf_x.row(y);
                    let row_blurred = blurred.row(y);
                    let row_uhf = uhf_x.row_mut(y);
                    for x in 0..width {
                        let uhf_val = row_orig[x] - row_blurred[x];
                        row_uhf[x] = remove_range_around_zero(uhf_val, REMOVE_UHF_RANGE as f32);
                    }
                    let row_hf = hf_x.row_mut(y);
                    for x in 0..width {
                        row_hf[x] =
                            remove_range_around_zero(row_blurred[x], REMOVE_HF_RANGE as f32);
                    }
                }
                blurred.recycle(pool);
            },
            || {
                let hf_y = &mut hf_y_slice[0];
                let uhf_y = &mut uhf_y_slice[0];
                let blurred = gaussian_blur(hf_y, sigma, pool);

                // Fused: UHF = clamp(orig - clamp(blurred)) * scale,
                //        HF = amplify_range(clamp(blurred) * scale)
                for y in 0..height {
                    let row_orig = hf_y.row(y);
                    let row_blurred = blurred.row(y);
                    let row_uhf = uhf_y.row_mut(y);
                    for x in 0..width {
                        let hf_clamped = maximum_clamp(row_blurred[x], MAXCLAMP_HF as f32);
                        let uhf_val = row_orig[x] - hf_clamped;
                        let uhf_clamped = maximum_clamp(uhf_val, MAXCLAMP_UHF as f32);
                        row_uhf[x] = uhf_clamped * MUL_Y_UHF as f32;
                    }
                    let row_hf = hf_y.row_mut(y);
                    for x in 0..width {
                        let hf_clamped = maximum_clamp(row_blurred[x], MAXCLAMP_HF as f32);
                        row_hf[x] = amplify_range_around_zero(
                            hf_clamped * MUL_Y_HF as f32,
                            ADD_HF_RANGE as f32,
                        );
                    }
                }
                blurred.recycle(pool);
            },
        );
    } else {
        // Sequential path for small images
        for i in 0..2 {
            let blurred = gaussian_blur(&hf[i], sigma, pool);

            // Fused: UHF = adjusted(orig - blurred), HF = adjusted(blurred)
            for y in 0..height {
                let row_orig = hf[i].row(y);
                let row_blurred = blurred.row(y);
                let row_uhf = uhf[i].row_mut(y);
                for x in 0..width {
                    if i == 0 {
                        let uhf_val = row_orig[x] - row_blurred[x];
                        row_uhf[x] = remove_range_around_zero(uhf_val, REMOVE_UHF_RANGE as f32);
                    } else {
                        let hf_clamped = maximum_clamp(row_blurred[x], MAXCLAMP_HF as f32);
                        let uhf_val = row_orig[x] - hf_clamped;
                        let uhf_clamped = maximum_clamp(uhf_val, MAXCLAMP_UHF as f32);
                        row_uhf[x] = uhf_clamped * MUL_Y_UHF as f32;
                    }
                }
                let row_hf = hf[i].row_mut(y);
                for x in 0..width {
                    if i == 0 {
                        row_hf[x] =
                            remove_range_around_zero(row_blurred[x], REMOVE_HF_RANGE as f32);
                    } else {
                        let hf_clamped = maximum_clamp(row_blurred[x], MAXCLAMP_HF as f32);
                        row_hf[x] = amplify_range_around_zero(
                            hf_clamped * MUL_Y_HF as f32,
                            ADD_HF_RANGE as f32,
                        );
                    }
                }
            }
            blurred.recycle(pool);
        }
    }
}

/// Performs the full frequency decomposition on an XYB image.
///
/// This is the main entry point for creating a PsychoImage from
/// an XYB color-space image.
pub fn separate_frequencies(xyb: &Image3F, pool: &BufferPool) -> PsychoImage {
    let width = xyb.width();
    let height = xyb.height();

    let mut ps = PsychoImage::from_pool(width, height, pool);

    // Separate into LF and MF
    separate_lf_and_mf(xyb, &mut ps.lf, &mut ps.mf, pool);

    // Separate MF into MF and HF
    separate_mf_and_hf(&mut ps.mf, &mut ps.hf, pool);

    // Separate HF into HF and UHF
    separate_hf_and_uhf(&mut ps.hf, &mut ps.uhf, pool);

    ps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remove_range_around_zero() {
        assert!((remove_range_around_zero(0.5, 0.1) - 0.4).abs() < 0.001);
        assert!((remove_range_around_zero(-0.5, 0.1) - (-0.4)).abs() < 0.001);
        assert!((remove_range_around_zero(0.05, 0.1) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_amplify_range_around_zero() {
        assert!((amplify_range_around_zero(0.5, 0.1) - 0.6).abs() < 0.001);
        assert!((amplify_range_around_zero(-0.5, 0.1) - (-0.6)).abs() < 0.001);
        assert!((amplify_range_around_zero(0.05, 0.1) - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_maximum_clamp() {
        assert!((maximum_clamp(5.0, 10.0) - 5.0).abs() < 0.001);
        assert!(maximum_clamp(15.0, 10.0) < 15.0);
        assert!(maximum_clamp(15.0, 10.0) > 10.0);
    }

    #[test]
    fn test_psycho_image_creation() {
        let ps = PsychoImage::new(100, 50);
        assert_eq!(ps.width(), 100);
        assert_eq!(ps.height(), 50);
    }

    #[test]
    fn test_frequency_separation() {
        // Create a simple XYB image
        let pool = BufferPool::new();
        let mut xyb = Image3F::new(32, 32);
        for y in 0..32 {
            for x in 0..32 {
                // Add some variation
                let val = (x + y) as f32 / 64.0;
                xyb.plane_mut(0).set(x, y, val * 0.1);
                xyb.plane_mut(1).set(x, y, val);
                xyb.plane_mut(2).set(x, y, val * 0.5);
            }
        }

        let ps = separate_frequencies(&xyb, &pool);

        // Just verify it runs and produces valid data
        assert_eq!(ps.width(), 32);
        assert_eq!(ps.height(), 32);
    }
}
