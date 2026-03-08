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

use crate::diff::maybe_join;
use crate::image::{BufferPool, Image3F, ImageF};
use crate::opsin::{linear_planar_to_xyb_butteraugli, linear_rgb_to_xyb_butteraugli};
use crate::psycho::{PsychoImage, separate_frequencies};
use crate::{ButteraugliError, ButteraugliParams, ButteraugliResult, check_finite_f32};

/// Minimum image dimension for multi-resolution processing.
const MIN_SIZE_FOR_MULTIRESOLUTION: usize = 8;

/// Minimum size for computing half-resolution (matches C++ threshold).
const MIN_SIZE_FOR_SUBSAMPLE: usize = 15;

/// Precomputed data for a single resolution level.
#[derive(Clone)]
struct ScaleData {
    /// Frequency-decomposed psychovisual image
    psycho: PsychoImage,
}

/// Precomputed butteraugli reference data for fast repeated comparisons.
///
/// This struct stores precomputed frequency decomposition and XYB conversion
/// for the reference image, allowing you to quickly compare multiple distorted
/// images against the same reference without recomputing reference-side data.
///
/// Uses single-level multiresolution matching C++ `ButteraugliComparator::Diffmap`:
/// the full-resolution diffmap plus one half-resolution sub-level.
///
/// Ideal for:
/// - Simulated annealing optimization
/// - Batch quality assessment
/// - Encoder tuning loops
#[derive(Clone)]
pub struct ButteraugliReference {
    /// Full resolution precomputed data
    full: ScaleData,
    /// Half resolution precomputed data (single sub-level for multiresolution)
    half: Option<ScaleData>,
    /// Original image dimensions
    width: usize,
    height: usize,
    /// Parameters used for precomputation
    params: ButteraugliParams,
}

/// Converts sRGB u8 buffer to linear f32.
fn srgb_u8_to_linear_f32(rgb: &[u8]) -> Vec<f32> {
    let lut = &*crate::opsin::SRGB_TO_LINEAR_LUT;
    rgb.iter().map(|&v| lut[v as usize]).collect()
}

impl ButteraugliReference {
    /// Precompute reference data from an sRGB u8 image.
    ///
    /// Internally converts sRGB to linear RGB, then delegates to the linear path.
    /// This ensures subsampling at all resolution levels happens in linear space.
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
        params.validate()?;

        let expected_size = width
            .checked_mul(height)
            .and_then(|wh| wh.checked_mul(3))
            .ok_or(ButteraugliError::DimensionOverflow { width, height })?;

        if width < MIN_SIZE_FOR_MULTIRESOLUTION || height < MIN_SIZE_FOR_MULTIRESOLUTION {
            return Err(ButteraugliError::InvalidDimensions { width, height });
        }

        if rgb.len() != expected_size {
            return Err(ButteraugliError::InvalidBufferSize {
                expected: expected_size,
                actual: rgb.len(),
            });
        }

        // Convert sRGB u8 → linear f32 and delegate.
        // Subsampling must happen in linear space, not gamma-compressed sRGB.
        let linear = srgb_u8_to_linear_f32(rgb);
        // Skip re-validation in new_linear — params already validated above.
        Self::new_linear_validated(&linear, width, height, params)
    }

    /// Precompute reference data from a linear RGB f32 image.
    ///
    /// Creates full-resolution data plus a single half-resolution sub-level,
    /// matching C++ `ButteraugliComparator::Diffmap` behavior.
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
        params.validate()?;
        Self::new_linear_validated(rgb, width, height, params)
    }

    /// Internal constructor that skips param validation (caller must have
    /// already called `params.validate()`).
    fn new_linear_validated(
        rgb: &[f32],
        width: usize,
        height: usize,
        params: ButteraugliParams,
    ) -> Result<Self, ButteraugliError> {
        let expected_size = width
            .checked_mul(height)
            .and_then(|wh| wh.checked_mul(3))
            .ok_or(ButteraugliError::DimensionOverflow { width, height })?;

        if width < MIN_SIZE_FOR_MULTIRESOLUTION || height < MIN_SIZE_FOR_MULTIRESOLUTION {
            return Err(ButteraugliError::InvalidDimensions { width, height });
        }

        if rgb.len() != expected_size {
            return Err(ButteraugliError::InvalidBufferSize {
                expected: expected_size,
                actual: rgb.len(),
            });
        }

        check_finite_f32(rgb, "linear rgb")?;

        let need_half = !params.single_resolution()
            && width >= MIN_SIZE_FOR_SUBSAMPLE
            && height >= MIN_SIZE_FOR_SUBSAMPLE;
        let intensity_target = params.intensity_target();

        // Run full-res and half-res in parallel (each with its own BufferPool)
        let (full, half) = maybe_join(
            || {
                let pool = BufferPool::new();
                let xyb =
                    linear_rgb_to_xyb_butteraugli(rgb, width, height, intensity_target, &pool);
                let psycho = separate_frequencies(&xyb, &pool);
                ScaleData { psycho }
            },
            || {
                if need_half {
                    let pool = BufferPool::new();
                    let (sub_rgb, sw, sh) = subsample_linear_rgb_2x(rgb, width, height);
                    let sub_xyb =
                        linear_rgb_to_xyb_butteraugli(&sub_rgb, sw, sh, intensity_target, &pool);
                    let sub_psycho = separate_frequencies(&sub_xyb, &pool);
                    Some(ScaleData { psycho: sub_psycho })
                } else {
                    None
                }
            },
        );

        Ok(Self {
            full,
            half,
            width,
            height,
            params,
        })
    }

    /// Creates a new reference from planar linear RGB data.
    ///
    /// Takes three separate channel slices (R, G, B) with the given stride.
    /// This avoids the interleave/de-interleave overhead when the caller
    /// already has planar data (e.g., from an encoder's reconstruction buffer).
    ///
    /// # Arguments
    /// * `r`, `g`, `b` - Per-channel planar data (stride * height elements each)
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `stride` - Pixels per row (>= width, for alignment padding)
    /// * `params` - Butteraugli parameters
    ///
    /// # Errors
    /// Returns an error if:
    /// - Image is smaller than 8x8 pixels
    /// - Any channel buffer is too small for the given stride and height
    pub fn new_linear_planar(
        r: &[f32],
        g: &[f32],
        b: &[f32],
        width: usize,
        height: usize,
        stride: usize,
        params: ButteraugliParams,
    ) -> Result<Self, ButteraugliError> {
        params.validate()?;

        if width < MIN_SIZE_FOR_MULTIRESOLUTION || height < MIN_SIZE_FOR_MULTIRESOLUTION {
            return Err(ButteraugliError::InvalidDimensions { width, height });
        }

        let min_size = stride
            .checked_mul(height)
            .ok_or(ButteraugliError::DimensionOverflow { width, height })?;
        if r.len() < min_size || g.len() < min_size || b.len() < min_size {
            return Err(ButteraugliError::InvalidBufferSize {
                expected: min_size,
                actual: r.len().min(g.len()).min(b.len()),
            });
        }

        check_finite_f32(&r[..min_size], "planar r")?;
        check_finite_f32(&g[..min_size], "planar g")?;
        check_finite_f32(&b[..min_size], "planar b")?;

        let need_half = !params.single_resolution()
            && width >= MIN_SIZE_FOR_SUBSAMPLE
            && height >= MIN_SIZE_FOR_SUBSAMPLE;
        let intensity_target = params.intensity_target();

        // Run full-res and half-res in parallel (each with its own BufferPool)
        let (full, half) = maybe_join(
            || {
                let pool = BufferPool::new();
                let xyb = linear_planar_to_xyb_butteraugli(
                    r,
                    g,
                    b,
                    width,
                    height,
                    stride,
                    intensity_target,
                    &pool,
                );
                let psycho = separate_frequencies(&xyb, &pool);
                ScaleData { psycho }
            },
            || {
                if need_half {
                    let pool = BufferPool::new();
                    let (sub_r, sub_g, sub_b, sw, sh) =
                        subsample_planar_rgb_2x(r, g, b, width, height, stride);
                    let sub_xyb = linear_planar_to_xyb_butteraugli(
                        &sub_r,
                        &sub_g,
                        &sub_b,
                        sw,
                        sh,
                        sw,
                        intensity_target,
                        &pool,
                    );
                    let sub_psycho = separate_frequencies(&sub_xyb, &pool);
                    Some(ScaleData { psycho: sub_psycho })
                } else {
                    None
                }
            },
        );

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

        let result = self.compare_impl(rgb);
        if !result.score.is_finite() {
            return Err(ButteraugliError::NonFiniteResult);
        }
        Ok(result)
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

        check_finite_f32(rgb, "compare linear rgb")?;

        let result = self.compare_linear_impl(rgb);
        if !result.score.is_finite() {
            return Err(ButteraugliError::NonFiniteResult);
        }
        Ok(result)
    }

    /// Compare a distorted planar linear RGB image against the precomputed reference.
    ///
    /// Takes three separate channel slices with stride, avoiding the
    /// interleave/de-interleave overhead of `compare_linear`.
    ///
    /// # Arguments
    /// * `r`, `g`, `b` - Per-channel planar data (stride * height elements each)
    /// * `stride` - Pixels per row (>= width)
    ///
    /// # Errors
    /// Returns an error if any channel buffer is too small for the given stride and height.
    pub fn compare_linear_planar(
        &self,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        stride: usize,
    ) -> Result<ButteraugliResult, ButteraugliError> {
        let min_size = stride * self.height;
        if r.len() < min_size || g.len() < min_size || b.len() < min_size {
            return Err(ButteraugliError::InvalidBufferSize {
                expected: min_size,
                actual: r.len().min(g.len()).min(b.len()),
            });
        }

        check_finite_f32(&r[..min_size], "compare planar r")?;
        check_finite_f32(&g[..min_size], "compare planar g")?;
        check_finite_f32(&b[..min_size], "compare planar b")?;

        let result = self.compare_linear_planar_impl(r, g, b, stride);
        if !result.score.is_finite() {
            return Err(ButteraugliError::NonFiniteResult);
        }
        Ok(result)
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

    /// Precompute reference data from an `ImgRef<RGB8>` (sRGB).
    ///
    /// Convenience wrapper around [`new`](Self::new) that accepts `imgref` types.
    ///
    /// # Errors
    /// Returns an error if the image is smaller than 8x8 pixels.
    pub fn from_srgb(
        img: imgref::ImgRef<rgb::RGB8>,
        params: ButteraugliParams,
    ) -> Result<Self, ButteraugliError> {
        let linear = crate::diff::imgref_srgb_to_linear_f32(img);
        Self::new_linear(&linear, img.width(), img.height(), params)
    }

    /// Precompute reference data from an `ImgRef<RGB<f32>>` (linear RGB).
    ///
    /// Convenience wrapper around [`new_linear`](Self::new_linear) that accepts `imgref` types.
    ///
    /// # Errors
    /// Returns an error if the image is smaller than 8x8 pixels.
    pub fn from_linear(
        img: imgref::ImgRef<rgb::RGB<f32>>,
        params: ButteraugliParams,
    ) -> Result<Self, ButteraugliError> {
        let rgb = crate::diff::imgref_rgbf32_to_f32_vec(img);
        Self::new_linear(&rgb, img.width(), img.height(), params)
    }

    /// Compare a distorted sRGB image (as `ImgRef<RGB8>`) against the reference.
    ///
    /// # Errors
    /// Returns an error if dimensions don't match the reference.
    pub fn compare_srgb(
        &self,
        img: imgref::ImgRef<rgb::RGB8>,
    ) -> Result<ButteraugliResult, ButteraugliError> {
        if img.width() != self.width || img.height() != self.height {
            return Err(ButteraugliError::DimensionMismatch {
                w1: self.width,
                h1: self.height,
                w2: img.width(),
                h2: img.height(),
            });
        }
        let linear = crate::diff::imgref_srgb_to_linear_f32(img);
        self.compare_linear(&linear)
    }

    /// Compare a distorted linear RGB image (as `ImgRef<RGB<f32>>`) against the reference.
    ///
    /// # Errors
    /// Returns an error if dimensions don't match the reference.
    pub fn compare_linear_imgref(
        &self,
        img: imgref::ImgRef<rgb::RGB<f32>>,
    ) -> Result<ButteraugliResult, ButteraugliError> {
        if img.width() != self.width || img.height() != self.height {
            return Err(ButteraugliError::DimensionMismatch {
                w1: self.width,
                h1: self.height,
                w2: img.width(),
                h2: img.height(),
            });
        }
        let rgb = crate::diff::imgref_rgbf32_to_f32_vec(img);
        self.compare_linear(&rgb)
    }

    /// Internal comparison implementation for sRGB input.
    ///
    /// Converts sRGB to linear and delegates to the linear path.
    fn compare_impl(&self, rgb: &[u8]) -> ButteraugliResult {
        let linear = srgb_u8_to_linear_f32(rgb);
        self.compare_linear_impl(&linear)
    }

    /// Internal comparison implementation for linear RGB input.
    ///
    /// Computes full-resolution diffmap using precomputed reference, then adds
    /// a single half-resolution sub-level via AddSupersampled2x. This matches
    /// C++ `ButteraugliComparator::Diffmap` which only uses one sub-level.
    fn compare_linear_impl(&self, rgb: &[f32]) -> ButteraugliResult {
        let intensity_target = self.params.intensity_target();
        let width = self.width;
        let height = self.height;
        let params = &self.params;
        let full_psycho = &self.full.psycho;
        let half_ref = self.half.as_ref();

        // Run full-res and half-res in parallel
        let (mut diffmap, sub_diffmap) = maybe_join(
            || {
                let pool = BufferPool::new();
                let xyb2 =
                    linear_rgb_to_xyb_butteraugli(rgb, width, height, intensity_target, &pool);
                let ps2 = separate_frequencies(&xyb2, &pool);
                compute_diffmap_with_precomputed(full_psycho, &ps2, width, height, params, &pool)
            },
            || {
                half_ref.map(|half| {
                    let pool = BufferPool::new();
                    let (sub_rgb, sw, sh) = subsample_linear_rgb_2x(rgb, width, height);
                    let sub_xyb =
                        linear_rgb_to_xyb_butteraugli(&sub_rgb, sw, sh, intensity_target, &pool);
                    let sub_ps = separate_frequencies(&sub_xyb, &pool);
                    compute_diffmap_with_precomputed(&half.psycho, &sub_ps, sw, sh, params, &pool)
                })
            },
        );

        if let Some(sub) = sub_diffmap {
            add_supersampled_2x(&sub, 0.5, &mut diffmap);
        }

        let score = compute_score_from_diffmap(&diffmap);

        ButteraugliResult {
            score,
            diffmap: Some(diffmap.into_imgvec()),
        }
    }

    /// Internal comparison implementation for planar linear RGB input.
    fn compare_linear_planar_impl(
        &self,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        stride: usize,
    ) -> ButteraugliResult {
        let intensity_target = self.params.intensity_target();
        let width = self.width;
        let height = self.height;
        let params = &self.params;
        let full_psycho = &self.full.psycho;
        let half_ref = self.half.as_ref();

        // Run full-res and half-res in parallel
        let (mut diffmap, sub_diffmap) = maybe_join(
            || {
                let pool = BufferPool::new();
                let xyb2 = linear_planar_to_xyb_butteraugli(
                    r,
                    g,
                    b,
                    width,
                    height,
                    stride,
                    intensity_target,
                    &pool,
                );
                let ps2 = separate_frequencies(&xyb2, &pool);
                compute_diffmap_with_precomputed(full_psycho, &ps2, width, height, params, &pool)
            },
            || {
                half_ref.map(|half| {
                    let pool = BufferPool::new();
                    let (sub_r, sub_g, sub_b, sw, sh) =
                        subsample_planar_rgb_2x(r, g, b, width, height, stride);
                    let sub_xyb = linear_planar_to_xyb_butteraugli(
                        &sub_r,
                        &sub_g,
                        &sub_b,
                        sw,
                        sh,
                        sw,
                        intensity_target,
                        &pool,
                    );
                    let sub_ps = separate_frequencies(&sub_xyb, &pool);
                    compute_diffmap_with_precomputed(&half.psycho, &sub_ps, sw, sh, params, &pool)
                })
            },
        );

        if let Some(sub) = sub_diffmap {
            add_supersampled_2x(&sub, 0.5, &mut diffmap);
        }

        let score = compute_score_from_diffmap(&diffmap);

        ButteraugliResult {
            score,
            diffmap: Some(diffmap.into_imgvec()),
        }
    }
}

// ============================================================================
// Internal functions (factored out from diff.rs for reuse)
// ============================================================================

use crate::consts::{
    NORM1_HF, NORM1_HF_X, NORM1_MF, NORM1_MF_X, NORM1_UHF, NORM1_UHF_X, W_HF_MALTA, W_HF_MALTA_X,
    W_MF_MALTA, W_MF_MALTA_X, W_UHF_MALTA, W_UHF_MALTA_X, WMUL,
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
    pool: &BufferPool,
) -> ImageF {
    // Compute AC differences using Malta filter
    let mut block_diff_ac =
        compute_psycho_diff_malta(ps1, ps2, params.hf_asymmetry(), params.xmul());

    // Compute mask from both PsychoImages
    let mask = mask_psycho_image(ps1, ps2, Some(block_diff_ac.plane_mut(1)), pool);

    // Compute DC (LF) differences (fully overwritten by compute_lf_diff)
    let mut block_diff_dc = Image3F::new_uninit(width, height);
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

/// Computes LF (DC) squared difference - autoversioned for autovectorization.
#[archmage::autoversion]
fn compute_lf_diff(
    _token: archmage::SimdToken,
    p1: &ImageF,
    p2: &ImageF,
    w: f32,
    out: &mut ImageF,
) {
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

/// Adds src into dst element-wise.
fn add_to(src: &ImageF, dst: &mut ImageF) {
    let width = src.width();
    let height = src.height();
    for y in 0..height {
        let s = src.row(y);
        let d = dst.row_mut(y);
        for x in 0..width {
            d[x] += s[x];
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
    let sqrt_hf_asym = hf_asymmetry.sqrt();

    // Run Y-channel and X-channel Malta computations in parallel
    let (plane_y, plane_x) = maybe_join(
        || {
            // Y channel: UHF_Y + HF_Y + MF_Y Malta + L2 diffs
            let mut ac_y = ImageF::new(width, height);

            let (uhf_y, (hf_y, mf_y)) = maybe_join(
                || {
                    malta_diff_map(
                        &ps0.uhf[1],
                        &ps1.uhf[1],
                        W_UHF_MALTA * hf_asymmetry as f64,
                        W_UHF_MALTA / hf_asymmetry as f64,
                        NORM1_UHF,
                        false,
                    )
                },
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
                            )
                        },
                        || {
                            malta_diff_map(
                                ps0.mf.plane(1),
                                ps1.mf.plane(1),
                                W_MF_MALTA,
                                W_MF_MALTA,
                                NORM1_MF,
                                true,
                            )
                        },
                    )
                },
            );

            add_to(&uhf_y, &mut ac_y);
            add_to(&hf_y, &mut ac_y);
            add_to(&mf_y, &mut ac_y);

            l2_diff_asymmetric(
                &ps0.hf[1],
                &ps1.hf[1],
                WMUL[1] as f32 * hf_asymmetry,
                WMUL[1] as f32 / hf_asymmetry,
                &mut ac_y,
            );
            l2_diff(ps0.mf.plane(1), ps1.mf.plane(1), WMUL[4] as f32, &mut ac_y);

            ac_y
        },
        || {
            // X channel: UHF_X + HF_X + MF_X Malta + L2 diffs
            let mut ac_x = ImageF::new(width, height);

            let (uhf_x, (hf_x, mf_x)) = maybe_join(
                || {
                    malta_diff_map(
                        &ps0.uhf[0],
                        &ps1.uhf[0],
                        W_UHF_MALTA_X * hf_asymmetry as f64,
                        W_UHF_MALTA_X / hf_asymmetry as f64,
                        NORM1_UHF_X,
                        false,
                    )
                },
                || {
                    maybe_join(
                        || {
                            malta_diff_map(
                                &ps0.hf[0],
                                &ps1.hf[0],
                                W_HF_MALTA_X * sqrt_hf_asym as f64,
                                W_HF_MALTA_X / sqrt_hf_asym as f64,
                                NORM1_HF_X,
                                true,
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
                            )
                        },
                    )
                },
            );

            add_to(&uhf_x, &mut ac_x);
            add_to(&hf_x, &mut ac_x);
            add_to(&mf_x, &mut ac_x);

            l2_diff_asymmetric(
                &ps0.hf[0],
                &ps1.hf[0],
                WMUL[0] as f32 * hf_asymmetry,
                WMUL[0] as f32 / hf_asymmetry,
                &mut ac_x,
            );
            l2_diff(ps0.mf.plane(0), ps1.mf.plane(0), WMUL[3] as f32, &mut ac_x);

            ac_x
        },
    );

    // B channel L2Diff (small, run inline)
    let mut plane_b = ImageF::new(width, height);
    l2_diff(
        ps0.mf.plane(2),
        ps1.mf.plane(2),
        WMUL[5] as f32,
        &mut plane_b,
    );

    Image3F::from_planes(plane_x, plane_y, plane_b)
}

/// Computes mask from two PsychoImages.
fn mask_psycho_image(
    ps0: &PsychoImage,
    ps1: &PsychoImage,
    diff_ac: Option<&mut ImageF>,
    pool: &BufferPool,
) -> ImageF {
    let width = ps0.width();
    let height = ps0.height();

    let mut mask0 = ImageF::new_uninit(width, height);
    let mut mask1 = ImageF::new_uninit(width, height);
    combine_channels_for_masking(&ps0.hf, &ps0.uhf, &mut mask0);
    combine_channels_for_masking(&ps1.hf, &ps1.uhf, &mut mask1);

    compute_mask_from_images(&mask0, &mask1, diff_ac, pool)
}

/// Combines channels to produce final diffmap - autoversioned for autovectorization.
#[archmage::autoversion]
fn combine_channels_to_diffmap(
    _token: archmage::SimdToken,
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
            let maskval = mask_y(val) as f32;
            let dc_maskval = mask_dc_y(val) as f32;

            let dc_masked = dc0[x] * xmul * dc_maskval + dc1[x] * dc_maskval + dc2[x] * dc_maskval;
            let ac_masked = ac0[x] * xmul * maskval + ac1[x] * maskval + ac2[x] * maskval;

            out[x] = (dc_masked + ac_masked).sqrt();
        }
    }

    diffmap
}

/// L2 difference (symmetric) - autoversioned for autovectorization.
#[archmage::autoversion]
fn l2_diff(_token: archmage::SimdToken, i0: &ImageF, i1: &ImageF, w: f32, diffmap: &mut ImageF) {
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

/// L2 difference asymmetric - autoversioned for autovectorization.
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

/// Computes global score from diffmap - autoversioned for autovectorization.
#[archmage::autoversion]
fn compute_score_from_diffmap(_token: archmage::SimdToken, diffmap: &ImageF) -> f64 {
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

    let blend = 1.0 - K_HEURISTIC_MIXING_VALUE * weight;
    let src_w = src.width();
    for y in 0..height {
        let src_y = (y / 2).min(src.height() - 1);
        let src_row = src.row(src_y);
        let dst_row = dest.row_mut(y);
        for x in 0..width {
            let src_val = src_row[(x / 2).min(src_w - 1)];
            dst_row[x] = dst_row[x] * blend + weight * src_val;
        }
    }
}

/// Subsamples linear RGB buffer by 2x.
fn subsample_linear_rgb_2x(rgb: &[f32], width: usize, height: usize) -> (Vec<f32>, usize, usize) {
    let out_width = width.div_ceil(2);
    let out_height = height.div_ceil(2);
    let out_size = out_width * out_height * 3;
    let mut output = vec![0.0f32; out_size];

    // Fast interior: all 2x2 blocks fully within bounds
    let interior_w = width / 2;
    let interior_h = height / 2;
    let inv4 = 0.25f32;

    for oy in 0..interior_h {
        let row0 = oy * 2 * width * 3;
        let row1 = (oy * 2 + 1) * width * 3;
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
            let row0 = oy * 2 * width * 3 + ix * 3;
            let row1 = (oy * 2 + 1) * width * 3 + ix * 3;
            let out_idx = (oy * out_width + ox) * 3;
            output[out_idx] = (rgb[row0] + rgb[row1]) * 0.5;
            output[out_idx + 1] = (rgb[row0 + 1] + rgb[row1 + 1]) * 0.5;
            output[out_idx + 2] = (rgb[row0 + 2] + rgb[row1 + 2]) * 0.5;
        }
    }

    // Bottom edge row (if height is odd)
    if out_height > interior_h {
        let oy = interior_h;
        let row0 = oy * 2 * width * 3;
        for ox in 0..interior_w {
            let ix = ox * 2;
            let i00 = row0 + ix * 3;
            let i10 = row0 + (ix + 1) * 3;
            let out_idx = (oy * out_width + ox) * 3;
            output[out_idx] = (rgb[i00] + rgb[i10]) * 0.5;
            output[out_idx + 1] = (rgb[i00 + 1] + rgb[i10 + 1]) * 0.5;
            output[out_idx + 2] = (rgb[i00 + 2] + rgb[i10 + 2]) * 0.5;
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

/// Subsamples planar linear RGB by 2x for multi-resolution processing.
fn subsample_planar_rgb_2x(
    r: &[f32],
    g: &[f32],
    b: &[f32],
    width: usize,
    height: usize,
    stride: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, usize, usize) {
    let out_width = width.div_ceil(2);
    let out_height = height.div_ceil(2);
    let out_size = out_width * out_height;
    let mut out_r = vec![0.0f32; out_size];
    let mut out_g = vec![0.0f32; out_size];
    let mut out_b = vec![0.0f32; out_size];

    // Fast interior: all 2x2 blocks are fully within bounds
    let interior_w = width / 2;
    let interior_h = height / 2;
    let inv4 = 0.25f32;

    for oy in 0..interior_h {
        let row0 = oy * 2 * stride;
        let row1 = (oy * 2 + 1) * stride;
        for ox in 0..interior_w {
            let ix = ox * 2;
            let i00 = row0 + ix;
            let i10 = row0 + ix + 1;
            let i01 = row1 + ix;
            let i11 = row1 + ix + 1;
            let out_idx = oy * out_width + ox;
            out_r[out_idx] = (r[i00] + r[i10] + r[i01] + r[i11]) * inv4;
            out_g[out_idx] = (g[i00] + g[i10] + g[i01] + g[i11]) * inv4;
            out_b[out_idx] = (b[i00] + b[i10] + b[i01] + b[i11]) * inv4;
        }
    }

    // Right edge column (if width is odd)
    if out_width > interior_w {
        let ox = interior_w;
        let ix = ox * 2;
        for oy in 0..interior_h {
            let row0 = oy * 2 * stride;
            let row1 = (oy * 2 + 1) * stride;
            let out_idx = oy * out_width + ox;
            out_r[out_idx] = (r[row0 + ix] + r[row1 + ix]) * 0.5;
            out_g[out_idx] = (g[row0 + ix] + g[row1 + ix]) * 0.5;
            out_b[out_idx] = (b[row0 + ix] + b[row1 + ix]) * 0.5;
        }
    }

    // Bottom edge row (if height is odd)
    if out_height > interior_h {
        let oy = interior_h;
        let row0 = oy * 2 * stride;
        for ox in 0..interior_w {
            let ix = ox * 2;
            let out_idx = oy * out_width + ox;
            out_r[out_idx] = (r[row0 + ix] + r[row0 + ix + 1]) * 0.5;
            out_g[out_idx] = (g[row0 + ix] + g[row0 + ix + 1]) * 0.5;
            out_b[out_idx] = (b[row0 + ix] + b[row0 + ix + 1]) * 0.5;
        }
        // Bottom-right corner (if both odd)
        if out_width > interior_w {
            let ox = interior_w;
            let ix = ox * 2;
            let out_idx = oy * out_width + ox;
            out_r[out_idx] = r[row0 + ix];
            out_g[out_idx] = g[row0 + ix];
            out_b[out_idx] = b[row0 + ix];
        }
    }

    (out_r, out_g, out_b, out_width, out_height)
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
            "should have sub-level data for 64x64"
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
            "should not have sub-level for 12x12"
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

        // Compute using new API
        use crate::{Img, RGB8, butteraugli};
        let pixels1: Vec<RGB8> = rgb1
            .chunks_exact(3)
            .map(|c| RGB8::new(c[0], c[1], c[2]))
            .collect();
        let pixels2: Vec<RGB8> = rgb2
            .chunks_exact(3)
            .map(|c| RGB8::new(c[0], c[1], c[2]))
            .collect();
        let img1 = Img::new(pixels1, width, height);
        let img2 = Img::new(pixels2, width, height);

        let params = ButteraugliParams::default().with_compute_diffmap(true);
        let full_result =
            butteraugli(img1.as_ref(), img2.as_ref(), &params).expect("should compute");

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
                let idx = y * width + x;
                let pre = precomputed_diffmap.buf()[idx];
                let full = full_diffmap.buf()[idx];
                assert!(
                    (pre - full).abs() < 1e-6,
                    "diffmap mismatch at ({x}, {y}): precomputed={pre}, full={full}"
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

        // Compute using new API
        use crate::{Img, RGB, butteraugli_linear};
        let pixels1: Vec<RGB<f32>> = rgb1
            .chunks_exact(3)
            .map(|c| RGB::new(c[0], c[1], c[2]))
            .collect();
        let pixels2: Vec<RGB<f32>> = rgb2
            .chunks_exact(3)
            .map(|c| RGB::new(c[0], c[1], c[2]))
            .collect();
        let img1 = Img::new(pixels1, width, height);
        let img2 = Img::new(pixels2, width, height);

        let full_result =
            butteraugli_linear(img1.as_ref(), img2.as_ref(), &ButteraugliParams::default())
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

        assert!(
            reference.half.is_some(),
            "should have half-resolution sub-level"
        );

        let precomputed_result = reference.compare(&rgb2).expect("should compare");

        // Compute using new API
        use crate::{Img, RGB8, butteraugli};
        let pixels1: Vec<RGB8> = rgb1
            .chunks_exact(3)
            .map(|c| RGB8::new(c[0], c[1], c[2]))
            .collect();
        let pixels2: Vec<RGB8> = rgb2
            .chunks_exact(3)
            .map(|c| RGB8::new(c[0], c[1], c[2]))
            .collect();
        let img1 = Img::new(pixels1, width, height);
        let img2 = Img::new(pixels2, width, height);

        let full_result = butteraugli(img1.as_ref(), img2.as_ref(), &ButteraugliParams::default())
            .expect("should compute");

        assert!(
            (precomputed_result.score - full_result.score).abs() < 1e-10,
            "precomputed score {} should match full score {} (with multiresolution)",
            precomputed_result.score,
            full_result.score
        );
    }
}
