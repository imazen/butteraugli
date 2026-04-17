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
use crate::mask::PrecomputedMask;
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
    /// Precomputed reference-side mask (blur + fuzzy_erosion)
    mask: PrecomputedMask,
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
/// A persistent `BufferPool` is maintained across compare calls. After the first
/// comparison, all subsequent comparisons reuse previously allocated temporary
/// buffers, eliminating mmap/munmap overhead and reducing memset to only the
/// buffers that need zeroing (accumulators).
///
/// Ideal for:
/// - Simulated annealing optimization
/// - Batch quality assessment
/// - Encoder tuning loops
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
    /// Persistent buffer pool — reused across compare calls to avoid re-allocation
    pool: BufferPool,
}

impl Clone for ButteraugliReference {
    fn clone(&self) -> Self {
        Self {
            full: self.full.clone(),
            half: self.half.clone(),
            width: self.width,
            height: self.height,
            params: self.params.clone(),
            pool: BufferPool::new(), // fresh empty pool for the clone
        }
    }
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

        // Run full-res and half-res in parallel (each with its own BufferPool).
        // The full-res pool is returned and reused for compare calls.
        let ((full, reuse_pool), half) = maybe_join(
            || {
                let pool = BufferPool::new();
                let xyb =
                    linear_rgb_to_xyb_butteraugli(rgb, width, height, intensity_target, &pool);
                let psycho = separate_frequencies(&xyb, &pool);
                let mask = crate::mask::precompute_reference_mask(&psycho.hf, &psycho.uhf, &pool);
                xyb.recycle(&pool);
                (ScaleData { psycho, mask }, pool)
            },
            || {
                if need_half {
                    let pool = BufferPool::new();
                    let (sub_rgb, sw, sh) = subsample_linear_rgb_2x(rgb, width, height);
                    let sub_xyb =
                        linear_rgb_to_xyb_butteraugli(&sub_rgb, sw, sh, intensity_target, &pool);
                    let sub_psycho = separate_frequencies(&sub_xyb, &pool);
                    let sub_mask = crate::mask::precompute_reference_mask(
                        &sub_psycho.hf,
                        &sub_psycho.uhf,
                        &pool,
                    );
                    Some(ScaleData {
                        psycho: sub_psycho,
                        mask: sub_mask,
                    })
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
            pool: reuse_pool,
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

        // Run full-res and half-res in parallel (each with its own BufferPool).
        // The full-res pool is returned and reused for compare calls, so the
        // first compare_linear_planar call reuses pre-warmed buffers instead
        // of allocating + zeroing fresh memory.
        let ((full, reuse_pool), half) = maybe_join(
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
                let mask = crate::mask::precompute_reference_mask(&psycho.hf, &psycho.uhf, &pool);
                // Recycle xyb now — its buffers go back to pool for reuse
                xyb.recycle(&pool);
                (ScaleData { psycho, mask }, pool)
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
                    let sub_mask = crate::mask::precompute_reference_mask(
                        &sub_psycho.hf,
                        &sub_psycho.uhf,
                        &pool,
                    );
                    Some(ScaleData {
                        psycho: sub_psycho,
                        mask: sub_mask,
                    })
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
            pool: reuse_pool,
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
        let full_mask = &self.full.mask;
        let half_ref = self.half.as_ref();
        let pool = &self.pool;

        // Run full-res and half-res in parallel (shared pool via Mutex)
        let (mut diffmap, sub_diffmap) = maybe_join(
            || {
                let xyb2 =
                    linear_rgb_to_xyb_butteraugli(rgb, width, height, intensity_target, pool);
                let ps2 = separate_frequencies(&xyb2, pool);
                let dm =
                    compute_diffmap_with_precomputed(full_psycho, &ps2, full_mask, params, pool);
                ps2.recycle(pool);
                xyb2.recycle(pool);
                dm
            },
            || {
                half_ref.map(|half| {
                    let (sub_rgb, sw, sh) = subsample_linear_rgb_2x(rgb, width, height);
                    let sub_xyb =
                        linear_rgb_to_xyb_butteraugli(&sub_rgb, sw, sh, intensity_target, pool);
                    let sub_ps = separate_frequencies(&sub_xyb, pool);
                    let dm = compute_diffmap_with_precomputed(
                        &half.psycho,
                        &sub_ps,
                        &half.mask,
                        params,
                        pool,
                    );
                    sub_ps.recycle(pool);
                    sub_xyb.recycle(pool);
                    dm
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
        let full_mask = &self.full.mask;
        let half_ref = self.half.as_ref();
        let pool = &self.pool;

        // Run full-res and half-res in parallel (shared pool via Mutex)
        let (mut diffmap, sub_diffmap) = maybe_join(
            || {
                let xyb2 = linear_planar_to_xyb_butteraugli(
                    r,
                    g,
                    b,
                    width,
                    height,
                    stride,
                    intensity_target,
                    pool,
                );
                let ps2 = separate_frequencies(&xyb2, pool);
                let dm =
                    compute_diffmap_with_precomputed(full_psycho, &ps2, full_mask, params, pool);
                ps2.recycle(pool);
                xyb2.recycle(pool);
                dm
            },
            || {
                half_ref.map(|half| {
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
                        pool,
                    );
                    let sub_ps = separate_frequencies(&sub_xyb, pool);
                    let dm = compute_diffmap_with_precomputed(
                        &half.psycho,
                        &sub_ps,
                        &half.mask,
                        params,
                        pool,
                    );
                    sub_ps.recycle(pool);
                    sub_xyb.recycle(pool);
                    dm
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

/// Computes diffmap using precomputed reference PsychoImage and precomputed mask.
fn compute_diffmap_with_precomputed(
    ps1: &PsychoImage,
    ps2: &PsychoImage,
    precomputed_mask: &PrecomputedMask,
    params: &ButteraugliParams,
    pool: &BufferPool,
) -> ImageF {
    // Compute AC differences using Malta filter
    let mut block_diff_ac =
        compute_psycho_diff_malta(ps1, ps2, params.hf_asymmetry(), params.xmul(), pool);

    // Apply distorted-side mask correction (blur + mask-to-error accumulation)
    crate::mask::apply_mask_correction_precomputed(
        precomputed_mask,
        &ps2.hf,
        &ps2.uhf,
        Some(block_diff_ac.plane_mut(1)),
        pool,
    );

    // Use precomputed mask directly (no copy needed — read-only reference)
    let diffmap = combine_channels_to_diffmap_fused(
        &precomputed_mask.mask,
        &ps1.lf,
        &ps2.lf,
        &block_diff_ac,
        params.xmul(),
    );

    // Recycle temporaries back to pool
    block_diff_ac.recycle(pool);

    diffmap
}

/// Computes difference between two PsychoImages using Malta filter.
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

    // Run Y-channel and X-channel Malta computations in parallel
    let (plane_y, plane_x) = maybe_join(
        || {
            // Y channel: UHF_Y + HF_Y + MF_Y Malta + L2 diffs
            let (uhf_y, (hf_y, mf_y)) = maybe_join(
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
                                ps0.mf.plane(1),
                                ps1.mf.plane(1),
                                W_MF_MALTA,
                                W_MF_MALTA,
                                NORM1_MF,
                                true,
                                pool,
                            )
                        },
                    )
                },
            );

            // Use uhf_y directly as accumulator (no zero-init + add_to needed)
            let mut ac_y = uhf_y;
            // Fuse hf_y + mf_y into a single accumulation pass
            accumulate_two(&hf_y, &mf_y, &mut ac_y);
            hf_y.recycle(pool);
            mf_y.recycle(pool);

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
            let (uhf_x, (hf_x, mf_x)) = maybe_join(
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
            );

            // Use uhf_x directly as accumulator
            let mut ac_x = uhf_x;
            // Fuse hf_x + mf_x into a single accumulation pass
            accumulate_two(&hf_x, &mf_x, &mut ac_x);
            hf_x.recycle(pool);
            mf_x.recycle(pool);

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

    // B channel L2Diff — write-only variant (no zero-init needed)
    let mut plane_b = ImageF::from_pool_dirty(width, height, pool);
    l2_diff_write(
        ps0.mf.plane(2),
        ps1.mf.plane(2),
        WMUL[5] as f32,
        &mut plane_b,
    );

    Image3F::from_planes(plane_x, plane_y, plane_b)
}

/// Combines AC channels with inline DC diff computation from LF planes.
///
/// Fuses compute_lf_diff + combine_channels_to_diffmap into a single pass,
/// eliminating 3 intermediate DC diff plane allocations and 6MB memory traffic.
#[archmage::autoversion(v4x, v4, v3, neon, wasm128, scalar)]
fn combine_channels_to_diffmap_fused(
    _token: archmage::SimdToken,
    mask: &ImageF,
    lf1: &Image3F,
    lf2: &Image3F,
    block_diff_ac: &Image3F,
    xmul: f32,
) -> ImageF {
    use crate::consts::{
        MASK_DC_Y_MUL, MASK_DC_Y_OFFSET, MASK_DC_Y_SCALER, MASK_Y_MUL, MASK_Y_OFFSET, MASK_Y_SCALER,
    };

    let width = mask.width();
    let height = mask.height();
    let mut diffmap = ImageF::new_uninit(width, height);
    let dc_w0 = WMUL[6] as f32;
    let dc_w1 = WMUL[7] as f32;
    let dc_w2 = WMUL[8] as f32;

    // Precompute f32 mask constants for SIMD-friendly inner loop
    let global_scale = crate::consts::GLOBAL_SCALE;
    let my_mul = MASK_Y_MUL as f32;
    let my_scaler = MASK_Y_SCALER as f32;
    let my_offset = MASK_Y_OFFSET as f32;
    let mdc_mul = MASK_DC_Y_MUL as f32;
    let mdc_scaler = MASK_DC_Y_SCALER as f32;
    let mdc_offset = MASK_DC_Y_OFFSET as f32;

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

/// Accumulates two source images into a destination: dst[x] += a[x] + b[x].
#[archmage::autoversion(v4x, v4, v3, neon, wasm128, scalar)]
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

/// L2 difference (symmetric) - autoversioned for autovectorization.
#[archmage::autoversion(v4x, v4, v3, neon, wasm128, scalar)]
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

/// L2 difference (symmetric, write-only) - autoversioned for autovectorization.
///
/// Like `l2_diff` but overwrites diffmap instead of accumulating.
/// Use when diffmap is uninitialized or dirty.
#[archmage::autoversion(v4x, v4, v3, neon, wasm128, scalar)]
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

/// L2 difference asymmetric - autoversioned for autovectorization.
#[archmage::autoversion(v4x, v4, v3, neon, wasm128, scalar)]
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

    let height = i0.height();
    let vw_0gt1 = w_0gt1 * 0.8;
    let vw_0lt1 = w_0lt1 * 0.8;

    for y in 0..height {
        let row0 = i0.row(y);
        let row1 = i1.row(y);
        let row_diff = diffmap.row_mut(y);

        for ((d, &val0), &val1) in row_diff.iter_mut().zip(row0.iter()).zip(row1.iter()) {
            let diff = val0 - val1;
            let total = (diff * diff).mul_add(vw_0gt1, *d);

            // Branch-free asymmetric penalty:
            // Flip val1 to match val0's sign direction, then clamp.
            let fabs0 = val0.abs();
            let too_small = 0.4 * fabs0;
            let sign = 1.0f32.copysign(val0);
            let sv1 = val1 * sign;
            // v = max(too_small - sv1, 0) + max(sv1 - fabs0, 0)
            let v = (too_small - sv1).max(0.0) + (sv1 - fabs0).max(0.0);

            *d = (v * v).mul_add(vw_0lt1, total);
        }
    }
}

/// Computes global score from diffmap - autoversioned for autovectorization.
///
/// Uses 8 independent max accumulators to break the loop-carried dependency,
/// enabling LLVM to vectorize with vmaxps (8-wide packed max).
/// Diffmap values are guaranteed non-NaN (output of sqrt of validated inputs).
#[archmage::autoversion(v4x, v4, v3, neon, wasm128, scalar)]
fn compute_score_from_diffmap(_token: archmage::SimdToken, diffmap: &ImageF) -> f64 {
    let width = diffmap.width();
    let height = diffmap.height();

    if width * height == 0 {
        return 0.0;
    }

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

    // Horizontal reduction of 8 lanes
    let mut max_val = lanes[0];
    for &m in &lanes[1..] {
        if m > max_val {
            max_val = m;
        }
    }
    max_val as f64
}

/// Adds supersampled diffmap contribution (2× upsampling with blending).
///
/// Processes pairs of destination pixels that share the same source pixel,
/// enabling sequential access on both src and dst for better vectorization.
#[archmage::autoversion(v4x, v4, v3, neon, wasm128, scalar)]
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
        // For standard half→full upsample, this covers all or all-but-one pixels.
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

/// Subsamples a single planar channel by 2x (interior only).
///
/// Uses zip+chunks_exact for bounds-check-free sequential access.
#[archmage::autoversion(v4x, v4, v3, neon, wasm128, scalar)]
fn subsample_channel_2x_interior(
    _token: archmage::SimdToken,
    src: &[f32],
    out: &mut [f32],
    stride: usize,
    interior_w: usize,
    interior_h: usize,
    out_width: usize,
) {
    let inv4 = 0.25f32;
    for oy in 0..interior_h {
        let row0_start = oy * 2 * stride;
        let row1_start = (oy * 2 + 1) * stride;
        let src_row0 = &src[row0_start..row0_start + interior_w * 2];
        let src_row1 = &src[row1_start..row1_start + interior_w * 2];
        let dst_row = &mut out[oy * out_width..oy * out_width + interior_w];

        for ((d, pair0), pair1) in dst_row
            .iter_mut()
            .zip(src_row0.chunks_exact(2))
            .zip(src_row1.chunks_exact(2))
        {
            *d = (pair0[0] + pair0[1] + pair1[0] + pair1[1]) * inv4;
        }
    }
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

    let interior_w = width / 2;
    let interior_h = height / 2;

    // Fast interior: per-channel SIMD-friendly loop
    subsample_channel_2x_interior(r, &mut out_r, stride, interior_w, interior_h, out_width);
    subsample_channel_2x_interior(g, &mut out_g, stride, interior_w, interior_h, out_width);
    subsample_channel_2x_interior(b, &mut out_b, stride, interior_w, interior_h, out_width);

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

    // Checks bit-identical equality between the precompute path and the standalone
    // path. With iir-blur the two paths use slightly different op orders inside
    // the IIR recursion (different blur call sites trigger different inlining),
    // producing ~1e-5 FMA-rounding differences that exceed this 1e-10 tolerance.
    // The IIR feature is documented as approximate; gating to FIR-only.
    #[cfg(not(feature = "iir-blur"))]
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
