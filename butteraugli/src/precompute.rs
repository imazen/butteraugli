//! Precomputed reference data for fast repeated butteraugli comparisons.
//!
//! When comparing multiple distorted images against the same reference image,
//! precompute the reference data once and reuse it for ~40-50% speedup.
//!
//! # Example
//!
//! ```
//! use butteraugli::{ButteraugliReference, ButteraugliParams};
//!
//! // Load reference image (8-bit sRGB)
//! let width = 64;
//! let height = 64;
//! let reference_rgb: Vec<u8> = vec![128; width * height * 3];
//!
//! // Precompute reference data once
//! let reference = ButteraugliReference::new(&reference_rgb, width, height, ButteraugliParams::default())
//!     .expect("valid image");
//!
//! // Compare against multiple distorted images
//! for quality in [90, 80, 70] {
//!     let distorted_rgb: Vec<u8> = vec![120; width * height * 3]; // simulated distortion
//!     let result = reference.compare(&distorted_rgb).expect("valid distorted image");
//!     println!("Quality {}: butteraugli score = {:.3}", quality, result.score);
//! }
//! ```

use crate::image::{Image3F, ImageF};
use crate::opsin::{linear_rgb_to_xyb_butteraugli, srgb_to_xyb_butteraugli};
use crate::psycho::{separate_frequencies, PsychoImage};
use crate::{ButteraugliError, ButteraugliParams, ButteraugliResult};

/// Minimum image dimension for multi-resolution processing.
const MIN_SIZE_FOR_MULTIRESOLUTION: usize = 8;

/// Minimum size for computing half-resolution (matches C++ threshold).
const MIN_SIZE_FOR_SUBSAMPLE: usize = 15;

/// Precomputed data for a single resolution level.
#[derive(Clone)]
struct ScaleData {
    /// XYB image (needed for mask computation which uses original XYB values)
    xyb: Image3F,
    /// Frequency-decomposed psychovisual image
    psycho: PsychoImage,
}

/// Precomputed butteraugli reference data for fast repeated comparisons.
///
/// This struct stores precomputed frequency decomposition and XYB conversion
/// for the reference image, allowing you to quickly compare multiple distorted
/// images against the same reference without recomputing reference-side data.
///
/// Ideal for:
/// - Simulated annealing optimization
/// - Batch quality assessment
/// - Encoder tuning loops
#[derive(Clone)]
pub struct ButteraugliReference {
    /// Full resolution precomputed data
    full: ScaleData,
    /// Half resolution precomputed data (for multiresolution, if image large enough)
    half: Option<ScaleData>,
    /// Original image dimensions
    width: usize,
    height: usize,
    /// Parameters used for precomputation
    params: ButteraugliParams,
}

impl ButteraugliReference {
    /// Precompute reference data from an sRGB u8 image.
    ///
    /// # Arguments
    /// * `rgb` - Reference image (sRGB u8, 3 bytes per pixel, row-major RGB order)
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `params` - Butteraugli comparison parameters
    ///
    /// # Errors
    /// Returns an error if:
    /// - Buffer size doesn't match dimensions
    /// - Image is smaller than 8x8 pixels
    pub fn new(
        rgb: &[u8],
        width: usize,
        height: usize,
        params: ButteraugliParams,
    ) -> Result<Self, ButteraugliError> {
        let expected_size = width * height * 3;

        if width < MIN_SIZE_FOR_MULTIRESOLUTION || height < MIN_SIZE_FOR_MULTIRESOLUTION {
            return Err(ButteraugliError::InvalidDimensions { width, height });
        }

        if rgb.len() != expected_size {
            return Err(ButteraugliError::InvalidBufferSize {
                expected: expected_size,
                actual: rgb.len(),
            });
        }

        // Precompute full resolution
        let xyb = srgb_to_xyb_butteraugli(rgb, width, height, params.intensity_target());
        let psycho = separate_frequencies(&xyb);
        let full = ScaleData { xyb, psycho };

        // Precompute half resolution if image is large enough
        let half = if width >= MIN_SIZE_FOR_SUBSAMPLE && height >= MIN_SIZE_FOR_SUBSAMPLE {
            let (sub_rgb, sw, sh) = subsample_rgb_2x(rgb, width, height);
            let sub_xyb = srgb_to_xyb_butteraugli(&sub_rgb, sw, sh, params.intensity_target());
            let sub_psycho = separate_frequencies(&sub_xyb);
            Some(ScaleData {
                xyb: sub_xyb,
                psycho: sub_psycho,
            })
        } else {
            None
        };

        Ok(Self {
            full,
            half,
            width,
            height,
            params,
        })
    }

    /// Precompute reference data from a linear RGB f32 image.
    ///
    /// # Arguments
    /// * `rgb` - Reference image (linear RGB f32, 3 floats per pixel, row-major, 0.0-1.0 range)
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `params` - Butteraugli comparison parameters
    ///
    /// # Errors
    /// Returns an error if:
    /// - Buffer size doesn't match dimensions
    /// - Image is smaller than 8x8 pixels
    pub fn new_linear(
        rgb: &[f32],
        width: usize,
        height: usize,
        params: ButteraugliParams,
    ) -> Result<Self, ButteraugliError> {
        let expected_size = width * height * 3;

        if width < MIN_SIZE_FOR_MULTIRESOLUTION || height < MIN_SIZE_FOR_MULTIRESOLUTION {
            return Err(ButteraugliError::InvalidDimensions { width, height });
        }

        if rgb.len() != expected_size {
            return Err(ButteraugliError::InvalidBufferSize {
                expected: expected_size,
                actual: rgb.len(),
            });
        }

        // Precompute full resolution
        let xyb = linear_rgb_to_xyb_butteraugli(rgb, width, height, params.intensity_target());
        let psycho = separate_frequencies(&xyb);
        let full = ScaleData { xyb, psycho };

        // Precompute half resolution if image is large enough
        let half = if width >= MIN_SIZE_FOR_SUBSAMPLE && height >= MIN_SIZE_FOR_SUBSAMPLE {
            let (sub_rgb, sw, sh) = subsample_linear_rgb_2x(rgb, width, height);
            let sub_xyb =
                linear_rgb_to_xyb_butteraugli(&sub_rgb, sw, sh, params.intensity_target());
            let sub_psycho = separate_frequencies(&sub_xyb);
            Some(ScaleData {
                xyb: sub_xyb,
                psycho: sub_psycho,
            })
        } else {
            None
        };

        Ok(Self {
            full,
            half,
            width,
            height,
            params,
        })
    }

    /// Compare a distorted sRGB image against the precomputed reference.
    ///
    /// This is faster than `compute_butteraugli` when comparing multiple
    /// distorted images against the same reference because the reference-side
    /// XYB conversion and frequency decomposition are already done.
    ///
    /// # Arguments
    /// * `rgb` - Distorted image (sRGB u8, 3 bytes per pixel, row-major RGB order)
    ///
    /// # Errors
    /// Returns an error if:
    /// - Buffer size doesn't match reference dimensions
    pub fn compare(&self, rgb: &[u8]) -> Result<ButteraugliResult, ButteraugliError> {
        let expected_size = self.width * self.height * 3;

        if rgb.len() != expected_size {
            return Err(ButteraugliError::InvalidBufferSize {
                expected: expected_size,
                actual: rgb.len(),
            });
        }

        Ok(self.compare_impl(rgb))
    }

    /// Compare a distorted linear RGB image against the precomputed reference.
    ///
    /// # Arguments
    /// * `rgb` - Distorted image (linear RGB f32, 3 floats per pixel, row-major, 0.0-1.0 range)
    ///
    /// # Errors
    /// Returns an error if:
    /// - Buffer size doesn't match reference dimensions
    pub fn compare_linear(&self, rgb: &[f32]) -> Result<ButteraugliResult, ButteraugliError> {
        let expected_size = self.width * self.height * 3;

        if rgb.len() != expected_size {
            return Err(ButteraugliError::InvalidBufferSize {
                expected: expected_size,
                actual: rgb.len(),
            });
        }

        Ok(self.compare_linear_impl(rgb))
    }

    /// Width of the reference image.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Height of the reference image.
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Parameters used for this reference.
    #[must_use]
    pub fn params(&self) -> &ButteraugliParams {
        &self.params
    }

    /// Internal comparison implementation for sRGB input.
    fn compare_impl(&self, rgb: &[u8]) -> ButteraugliResult {
        // Convert distorted image to XYB and compute frequency decomposition
        let xyb2 =
            srgb_to_xyb_butteraugli(rgb, self.width, self.height, self.params.intensity_target());
        let ps2 = separate_frequencies(&xyb2);

        // Compute diffmap at full resolution using precomputed reference
        let mut diffmap = compute_diffmap_with_precomputed(
            &self.full.psycho,
            &ps2,
            self.width,
            self.height,
            &self.params,
        );

        // Add half-resolution contribution if available
        if let Some(ref half) = self.half {
            let (sub_rgb, sw, sh) = subsample_rgb_2x(rgb, self.width, self.height);
            let sub_xyb = srgb_to_xyb_butteraugli(&sub_rgb, sw, sh, self.params.intensity_target());
            let sub_ps2 = separate_frequencies(&sub_xyb);

            let sub_diffmap =
                compute_diffmap_with_precomputed(&half.psycho, &sub_ps2, sw, sh, &self.params);

            add_supersampled_2x(&sub_diffmap, 0.5, &mut diffmap);
        }

        let score = compute_score_from_diffmap(&diffmap);

        ButteraugliResult {
            score,
            diffmap: Some(diffmap),
        }
    }

    /// Internal comparison implementation for linear RGB input.
    fn compare_linear_impl(&self, rgb: &[f32]) -> ButteraugliResult {
        let xyb2 = linear_rgb_to_xyb_butteraugli(
            rgb,
            self.width,
            self.height,
            self.params.intensity_target(),
        );
        let ps2 = separate_frequencies(&xyb2);

        let mut diffmap = compute_diffmap_with_precomputed(
            &self.full.psycho,
            &ps2,
            self.width,
            self.height,
            &self.params,
        );

        if let Some(ref half) = self.half {
            let (sub_rgb, sw, sh) = subsample_linear_rgb_2x(rgb, self.width, self.height);
            let sub_xyb =
                linear_rgb_to_xyb_butteraugli(&sub_rgb, sw, sh, self.params.intensity_target());
            let sub_ps2 = separate_frequencies(&sub_xyb);

            let sub_diffmap =
                compute_diffmap_with_precomputed(&half.psycho, &sub_ps2, sw, sh, &self.params);

            add_supersampled_2x(&sub_diffmap, 0.5, &mut diffmap);
        }

        let score = compute_score_from_diffmap(&diffmap);

        ButteraugliResult {
            score,
            diffmap: Some(diffmap),
        }
    }
}

// ============================================================================
// Internal functions (factored out from diff.rs for reuse)
// ============================================================================

use crate::consts::{
    NORM1_HF, NORM1_HF_X, NORM1_MF, NORM1_MF_X, NORM1_UHF, NORM1_UHF_X, WMUL, W_HF_MALTA,
    W_HF_MALTA_X, W_MF_MALTA, W_MF_MALTA_X, W_UHF_MALTA, W_UHF_MALTA_X,
};
use crate::malta::malta_diff_map;
use crate::mask::{
    combine_channels_for_masking, compute_mask as compute_mask_from_images, mask_dc_y, mask_y,
};

/// Computes diffmap using precomputed reference PsychoImage.
fn compute_diffmap_with_precomputed(
    ps1: &PsychoImage,
    ps2: &PsychoImage,
    width: usize,
    height: usize,
    params: &ButteraugliParams,
) -> ImageF {
    // Compute AC differences using Malta filter
    let mut block_diff_ac =
        compute_psycho_diff_malta(ps1, ps2, params.hf_asymmetry(), params.xmul());

    // Compute mask from both PsychoImages
    let mask = mask_psycho_image(ps1, ps2, Some(block_diff_ac.plane_mut(1)));

    // Compute DC (LF) differences
    let mut block_diff_dc = Image3F::new(width, height);
    for c in 0..3 {
        compute_lf_diff(
            ps1.lf.plane(c),
            ps2.lf.plane(c),
            WMUL[6 + c] as f32,
            block_diff_dc.plane_mut(c),
        );
    }

    // Combine channels to final diffmap
    combine_channels_to_diffmap(&mask, &block_diff_dc, &block_diff_ac, params.xmul())
}

/// Computes LF (DC) squared difference - multiversioned for autovectorization.
#[multiversion::multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+avx+avx2+bmi1+bmi2+cmpxchg16b+f16c+fma+fxsr+lzcnt+movbe+popcnt+sse+sse2+sse3+sse4.1+sse4.2+ssse3+xsave",
    "x86_64+avx+avx2+bmi1+bmi2+cmpxchg16b+f16c+fma+fxsr+lzcnt+movbe+popcnt+sse+sse2+sse3+sse4.1+sse4.2+ssse3+xsave",
    "x86_64+cmpxchg16b+fxsr+popcnt+sse+sse2+sse3+sse4.1+sse4.2+ssse3",
))]
fn compute_lf_diff(p1: &ImageF, p2: &ImageF, w: f32, out: &mut ImageF) {
    let width = p1.width();
    let height = p1.height();

    for y in 0..height {
        let row1 = p1.row(y);
        let row2 = p2.row(y);
        let row_out = out.row_mut(y);

        for x in 0..width {
            let d = row1[x] - row2[x];
            row_out[x] = d * d * w;
        }
    }
}

/// Computes difference between two PsychoImages using Malta filter.
fn compute_psycho_diff_malta(
    ps0: &PsychoImage,
    ps1: &PsychoImage,
    hf_asymmetry: f32,
    _xmul: f32,
) -> Image3F {
    let width = ps0.width();
    let height = ps0.height();

    let mut block_diff_ac = Image3F::new(width, height);

    // UHF Y channel
    let uhf_y_diff = malta_diff_map(
        &ps0.uhf[1],
        &ps1.uhf[1],
        W_UHF_MALTA * hf_asymmetry as f64,
        W_UHF_MALTA / hf_asymmetry as f64,
        NORM1_UHF,
        false,
    );
    // TODO(simd): Accumulation loop - candidate for wide crate
    for y in 0..height {
        for x in 0..width {
            let v = block_diff_ac.plane(1).get(x, y) + uhf_y_diff.get(x, y);
            block_diff_ac.plane_mut(1).set(x, y, v);
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
    for y in 0..height {
        for x in 0..width {
            let v = block_diff_ac.plane(0).get(x, y) + uhf_x_diff.get(x, y);
            block_diff_ac.plane_mut(0).set(x, y, v);
        }
    }

    // HF channels
    let sqrt_hf_asym = hf_asymmetry.sqrt();

    let hf_y_diff = malta_diff_map(
        &ps0.hf[1],
        &ps1.hf[1],
        W_HF_MALTA * sqrt_hf_asym as f64,
        W_HF_MALTA / sqrt_hf_asym as f64,
        NORM1_HF,
        true,
    );
    for y in 0..height {
        for x in 0..width {
            let v = block_diff_ac.plane(1).get(x, y) + hf_y_diff.get(x, y);
            block_diff_ac.plane_mut(1).set(x, y, v);
        }
    }

    let hf_x_diff = malta_diff_map(
        &ps0.hf[0],
        &ps1.hf[0],
        W_HF_MALTA_X * sqrt_hf_asym as f64,
        W_HF_MALTA_X / sqrt_hf_asym as f64,
        NORM1_HF_X,
        true,
    );
    for y in 0..height {
        for x in 0..width {
            let v = block_diff_ac.plane(0).get(x, y) + hf_x_diff.get(x, y);
            block_diff_ac.plane_mut(0).set(x, y, v);
        }
    }

    // MF channels
    let mf_y_diff = malta_diff_map(
        ps0.mf.plane(1),
        ps1.mf.plane(1),
        W_MF_MALTA,
        W_MF_MALTA,
        NORM1_MF,
        true,
    );
    for y in 0..height {
        for x in 0..width {
            let v = block_diff_ac.plane(1).get(x, y) + mf_y_diff.get(x, y);
            block_diff_ac.plane_mut(1).set(x, y, v);
        }
    }

    let mf_x_diff = malta_diff_map(
        ps0.mf.plane(0),
        ps1.mf.plane(0),
        W_MF_MALTA_X,
        W_MF_MALTA_X,
        NORM1_MF_X,
        true,
    );
    for y in 0..height {
        for x in 0..width {
            let v = block_diff_ac.plane(0).get(x, y) + mf_x_diff.get(x, y);
            block_diff_ac.plane_mut(0).set(x, y, v);
        }
    }

    // L2DiffAsymmetric for HF channels
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

    // L2Diff for MF channels
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

/// Computes mask from two PsychoImages.
fn mask_psycho_image(ps0: &PsychoImage, ps1: &PsychoImage, diff_ac: Option<&mut ImageF>) -> ImageF {
    let width = ps0.width();
    let height = ps0.height();

    let mut mask0 = ImageF::new(width, height);
    let mut mask1 = ImageF::new(width, height);
    combine_channels_for_masking(&ps0.hf, &ps0.uhf, &mut mask0);
    combine_channels_for_masking(&ps1.hf, &ps1.uhf, &mut mask1);

    compute_mask_from_images(&mask0, &mask1, diff_ac)
}

/// Combines channels to produce final diffmap - multiversioned for autovectorization.
#[multiversion::multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+avx+avx2+bmi1+bmi2+cmpxchg16b+f16c+fma+fxsr+lzcnt+movbe+popcnt+sse+sse2+sse3+sse4.1+sse4.2+ssse3+xsave",
    "x86_64+avx+avx2+bmi1+bmi2+cmpxchg16b+f16c+fma+fxsr+lzcnt+movbe+popcnt+sse+sse2+sse3+sse4.1+sse4.2+ssse3+xsave",
    "x86_64+cmpxchg16b+fxsr+popcnt+sse+sse2+sse3+sse4.1+sse4.2+ssse3",
))]
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
        for x in 0..width {
            let val = mask.get(x, y) as f64;
            let maskval = mask_y(val) as f32;
            let dc_maskval = mask_dc_y(val) as f32;

            let diff_dc = [
                block_diff_dc.plane(0).get(x, y),
                block_diff_dc.plane(1).get(x, y),
                block_diff_dc.plane(2).get(x, y),
            ];
            let diff_ac = [
                block_diff_ac.plane(0).get(x, y),
                block_diff_ac.plane(1).get(x, y),
                block_diff_ac.plane(2).get(x, y),
            ];

            let diff_ac_scaled = [diff_ac[0] * xmul, diff_ac[1], diff_ac[2]];
            let diff_dc_scaled = [diff_dc[0] * xmul, diff_dc[1], diff_dc[2]];

            let dc_masked = diff_dc_scaled[0] * dc_maskval
                + diff_dc_scaled[1] * dc_maskval
                + diff_dc_scaled[2] * dc_maskval;
            let ac_masked = diff_ac_scaled[0] * maskval
                + diff_ac_scaled[1] * maskval
                + diff_ac_scaled[2] * maskval;

            diffmap.set(x, y, (dc_masked + ac_masked).sqrt());
        }
    }

    diffmap
}

/// L2 difference (symmetric) - multiversioned for autovectorization.
#[multiversion::multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+avx+avx2+bmi1+bmi2+cmpxchg16b+f16c+fma+fxsr+lzcnt+movbe+popcnt+sse+sse2+sse3+sse4.1+sse4.2+ssse3+xsave",
    "x86_64+avx+avx2+bmi1+bmi2+cmpxchg16b+f16c+fma+fxsr+lzcnt+movbe+popcnt+sse+sse2+sse3+sse4.1+sse4.2+ssse3+xsave",
    "x86_64+cmpxchg16b+fxsr+popcnt+sse+sse2+sse3+sse4.1+sse4.2+ssse3",
))]
fn l2_diff(i0: &ImageF, i1: &ImageF, w: f32, diffmap: &mut ImageF) {
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

/// L2 difference asymmetric - multiversioned for autovectorization.
#[multiversion::multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+avx+avx2+bmi1+bmi2+cmpxchg16b+f16c+fma+fxsr+lzcnt+movbe+popcnt+sse+sse2+sse3+sse4.1+sse4.2+ssse3+xsave",
    "x86_64+avx+avx2+bmi1+bmi2+cmpxchg16b+f16c+fma+fxsr+lzcnt+movbe+popcnt+sse+sse2+sse3+sse4.1+sse4.2+ssse3+xsave",
    "x86_64+cmpxchg16b+fxsr+popcnt+sse+sse2+sse3+sse4.1+sse4.2+ssse3",
))]
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

            let diff = val0 - val1;
            let mut total = row_diff[x] + diff * diff * vw_0gt1;

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

/// Computes global score from diffmap - multiversioned for autovectorization.
#[multiversion::multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+avx+avx2+bmi1+bmi2+cmpxchg16b+f16c+fma+fxsr+lzcnt+movbe+popcnt+sse+sse2+sse3+sse4.1+sse4.2+ssse3+xsave",
    "x86_64+avx+avx2+bmi1+bmi2+cmpxchg16b+f16c+fma+fxsr+lzcnt+movbe+popcnt+sse+sse2+sse3+sse4.1+sse4.2+ssse3+xsave",
    "x86_64+cmpxchg16b+fxsr+popcnt+sse+sse2+sse3+sse4.1+sse4.2+ssse3",
))]
fn compute_score_from_diffmap(diffmap: &ImageF) -> f64 {
    let width = diffmap.width();
    let height = diffmap.height();

    if width * height == 0 {
        return 0.0;
    }

    let mut max_val = 0.0f32;
    for y in 0..height {
        let row = diffmap.row(y);
        for x in 0..width {
            let v = row[x];
            if v > max_val {
                max_val = v;
            }
        }
    }

    max_val as f64
}

/// Adds supersampled diffmap contribution.
fn add_supersampled_2x(src: &ImageF, weight: f32, dest: &mut ImageF) {
    let width = dest.width();
    let height = dest.height();
    const K_HEURISTIC_MIXING_VALUE: f32 = 0.3;

    // TODO(simd): Upsampling and blending are vectorizable
    for y in 0..height {
        for x in 0..width {
            let src_x = x / 2;
            let src_y = y / 2;
            let src_val = src.get(src_x.min(src.width() - 1), src_y.min(src.height() - 1));

            let prev = dest.get(x, y);
            let mixed = prev * (1.0 - K_HEURISTIC_MIXING_VALUE * weight) + weight * src_val;
            dest.set(x, y, mixed);
        }
    }
}

/// Subsamples sRGB buffer by 2x.
fn subsample_rgb_2x(rgb: &[u8], width: usize, height: usize) -> (Vec<u8>, usize, usize) {
    let out_width = width.div_ceil(2);
    let out_height = height.div_ceil(2);
    let mut output = vec![0u8; out_width * out_height * 3];

    for oy in 0..out_height {
        for ox in 0..out_width {
            let mut r_sum = 0u32;
            let mut g_sum = 0u32;
            let mut b_sum = 0u32;
            let mut count = 0u32;

            for dy in 0..2 {
                for dx in 0..2 {
                    let ix = ox * 2 + dx;
                    let iy = oy * 2 + dy;
                    if ix < width && iy < height {
                        let idx = (iy * width + ix) * 3;
                        r_sum += rgb[idx] as u32;
                        g_sum += rgb[idx + 1] as u32;
                        b_sum += rgb[idx + 2] as u32;
                        count += 1;
                    }
                }
            }

            if count > 0 {
                let out_idx = (oy * out_width + ox) * 3;
                output[out_idx] = (r_sum / count) as u8;
                output[out_idx + 1] = (g_sum / count) as u8;
                output[out_idx + 2] = (b_sum / count) as u8;
            }
        }
    }

    (output, out_width, out_height)
}

/// Subsamples linear RGB buffer by 2x.
fn subsample_linear_rgb_2x(rgb: &[f32], width: usize, height: usize) -> (Vec<f32>, usize, usize) {
    let out_width = width.div_ceil(2);
    let out_height = height.div_ceil(2);
    let mut output = vec![0.0f32; out_width * out_height * 3];

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precompute_creation() {
        let width = 64;
        let height = 64;
        let rgb: Vec<u8> = vec![128; width * height * 3];

        let reference =
            ButteraugliReference::new(&rgb, width, height, ButteraugliParams::default())
                .expect("should create reference");

        assert_eq!(reference.width(), width);
        assert_eq!(reference.height(), height);
        assert!(
            reference.half.is_some(),
            "should have half-resolution data for 64x64"
        );
    }

    #[test]
    fn test_precompute_small_image_no_half() {
        let width = 12;
        let height = 12;
        let rgb: Vec<u8> = vec![128; width * height * 3];

        let reference =
            ButteraugliReference::new(&rgb, width, height, ButteraugliParams::default())
                .expect("should create reference");

        assert!(
            reference.half.is_none(),
            "should not have half-resolution for 12x12"
        );
    }

    #[test]
    fn test_precompute_too_small() {
        let width = 4;
        let height = 4;
        let rgb: Vec<u8> = vec![128; width * height * 3];

        let result = ButteraugliReference::new(&rgb, width, height, ButteraugliParams::default());
        assert!(matches!(
            result,
            Err(ButteraugliError::InvalidDimensions { .. })
        ));
    }

    #[test]
    fn test_precompute_wrong_buffer_size() {
        let rgb: Vec<u8> = vec![128; 100]; // Wrong size

        let result = ButteraugliReference::new(&rgb, 64, 64, ButteraugliParams::default());
        assert!(matches!(
            result,
            Err(ButteraugliError::InvalidBufferSize { .. })
        ));
    }

    #[test]
    fn test_compare_dimension_mismatch() {
        let width = 64;
        let height = 64;
        let rgb: Vec<u8> = vec![128; width * height * 3];

        let reference =
            ButteraugliReference::new(&rgb, width, height, ButteraugliParams::default())
                .expect("should create reference");

        // Try to compare with wrong-sized image
        let wrong_rgb: Vec<u8> = vec![128; 32 * 32 * 3];
        let result = reference.compare(&wrong_rgb);
        assert!(matches!(
            result,
            Err(ButteraugliError::InvalidBufferSize { .. })
        ));
    }

    #[test]
    fn test_precompute_identical_images() {
        let width = 32;
        let height = 32;
        let rgb: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();

        let reference =
            ButteraugliReference::new(&rgb, width, height, ButteraugliParams::default())
                .expect("should create reference");

        let result = reference.compare(&rgb).expect("should compare");

        assert!(
            result.score < 0.001,
            "identical images should have score ~0, got {}",
            result.score
        );
    }

    #[test]
    fn test_precompute_different_images() {
        let width = 32;
        let height = 32;
        let rgb1: Vec<u8> = vec![100; width * height * 3];
        let rgb2: Vec<u8> = vec![150; width * height * 3];

        let reference =
            ButteraugliReference::new(&rgb1, width, height, ButteraugliParams::default())
                .expect("should create reference");

        let result = reference.compare(&rgb2).expect("should compare");

        assert!(
            result.score > 0.0,
            "different images should have non-zero score"
        );
    }

    #[test]
    fn test_precompute_matches_full_compute() {
        // This is the critical parity test
        let width = 48;
        let height = 48;

        // Create a gradient image for more interesting comparison
        let rgb1: Vec<u8> = (0..width * height)
            .flat_map(|i| {
                let x = i % width;
                let y = i / width;
                [(x * 5) as u8, (y * 5) as u8, 128]
            })
            .collect();

        // Create a slightly distorted version
        let rgb2: Vec<u8> = rgb1.iter().map(|&v| v.saturating_add(5)).collect();

        // Compute using precomputed reference
        let reference =
            ButteraugliReference::new(&rgb1, width, height, ButteraugliParams::default())
                .expect("should create reference");
        let precomputed_result = reference.compare(&rgb2).expect("should compare");

        // Compute using full method
        let full_result =
            crate::compute_butteraugli(&rgb1, &rgb2, width, height, &ButteraugliParams::default())
                .expect("should compute");

        // Scores should match exactly
        assert!(
            (precomputed_result.score - full_result.score).abs() < 1e-10,
            "precomputed score {} should match full score {}",
            precomputed_result.score,
            full_result.score
        );

        // Diffmaps should also match
        let precomputed_diffmap = precomputed_result.diffmap.as_ref().unwrap();
        let full_diffmap = full_result.diffmap.as_ref().unwrap();

        assert_eq!(precomputed_diffmap.width(), full_diffmap.width());
        assert_eq!(precomputed_diffmap.height(), full_diffmap.height());

        for y in 0..height {
            for x in 0..width {
                let pre = precomputed_diffmap.get(x, y);
                let full = full_diffmap.get(x, y);
                assert!(
                    (pre - full).abs() < 1e-6,
                    "diffmap mismatch at ({}, {}): precomputed={}, full={}",
                    x,
                    y,
                    pre,
                    full
                );
            }
        }
    }

    #[test]
    fn test_precompute_linear_matches_full() {
        let width = 32;
        let height = 32;

        // Create linear RGB data
        let rgb1: Vec<f32> = (0..width * height)
            .flat_map(|i| {
                let x = (i % width) as f32 / width as f32;
                let y = (i / width) as f32 / height as f32;
                [x, y, 0.5]
            })
            .collect();

        let rgb2: Vec<f32> = rgb1.iter().map(|&v| (v * 0.95).min(1.0)).collect();

        // Compute using precomputed reference
        let reference =
            ButteraugliReference::new_linear(&rgb1, width, height, ButteraugliParams::default())
                .expect("should create reference");
        let precomputed_result = reference.compare_linear(&rgb2).expect("should compare");

        // Compute using full method
        let full_result = crate::compute_butteraugli_linear(
            &rgb1,
            &rgb2,
            width,
            height,
            &ButteraugliParams::default(),
        )
        .expect("should compute");

        assert!(
            (precomputed_result.score - full_result.score).abs() < 1e-10,
            "precomputed score {} should match full score {}",
            precomputed_result.score,
            full_result.score
        );
    }

    #[test]
    fn test_precompute_with_multiresolution() {
        // Test with image large enough for multiresolution
        let width = 64;
        let height = 64;

        let rgb1: Vec<u8> = (0..width * height)
            .flat_map(|i| {
                let x = i % width;
                [(x * 4) as u8, 128, 128]
            })
            .collect();

        let rgb2: Vec<u8> = rgb1.iter().map(|&v| v.saturating_add(10)).collect();

        let reference =
            ButteraugliReference::new(&rgb1, width, height, ButteraugliParams::default())
                .expect("should create reference");

        assert!(reference.half.is_some(), "should have multiresolution data");

        let precomputed_result = reference.compare(&rgb2).expect("should compare");
        let full_result =
            crate::compute_butteraugli(&rgb1, &rgb2, width, height, &ButteraugliParams::default())
                .expect("should compute");

        assert!(
            (precomputed_result.score - full_result.score).abs() < 1e-10,
            "precomputed score {} should match full score {} (with multiresolution)",
            precomputed_result.score,
            full_result.score
        );
    }
}
