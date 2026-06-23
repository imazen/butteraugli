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

use enough::Stop;

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

impl ScaleData {
    /// Heap bytes backing one resolution level: the 10-plane psycho
    /// pyramid + the 2-plane mask = 12 `ImageF` at this scale.
    #[must_use]
    fn byte_size(&self) -> usize {
        self.psycho.byte_size() + self.mask.byte_size()
    }

    /// A-priori byte estimate for a level at `width × height`, matching the
    /// layout `byte_size` measures (`stride = round_up(width, 16)`, 12
    /// planes). Kept in lockstep with [`Self::byte_size`] by
    /// `estimated_reference_bytes_matches_precompute`.
    #[must_use]
    fn estimated_byte_size(width: usize, height: usize) -> usize {
        // ImageF aligns stride to 16 floats (64 B) — see `ImageF::new`.
        let stride = (width + 15) & !15;
        let plane = stride * height * core::mem::size_of::<f32>();
        // PsychoImage(uhf2 + hf2 + mf3 + lf3 = 10) + PrecomputedMask(2).
        12 * plane
    }
}

/// Retained reference-side source data for the strip walker.
///
/// The strip walker (`compare_strip`, `compare_linear_strip`) needs to
/// slice strip-shaped windows from the reference source. Storing the
/// original sRGB bytes (`Srgb`) costs `width * height * 3` bytes —
/// 4× less than storing the pre-converted linear `Vec<f32>` (`Linear`)
/// that 0.9.3 retained for every `new()`-built reference. The
/// per-strip sRGB→linear conversion (LUT-based, identical to the
/// distorted-side conversion the strip walker already runs) recovers
/// the linear bytes on demand without inflating the persistent
/// footprint. The 4× savings closes most of the 0.9.3 warm-ref
/// peak-heap regression (192 MB → 48 MB at 16 MP, 480 MB → 120 MB at
/// 40 MP).
#[derive(Clone)]
enum ReferenceSource {
    /// sRGB u8 bytes from `new()`. Stored as the original input — 3
    /// bytes per pixel.
    Srgb(Vec<u8>),
    /// Linear f32 from `new_linear()`. 12 bytes per pixel; no
    /// compression opportunity beyond clone elision since the input was
    /// already linear.
    Linear(Vec<f32>),
}

impl ReferenceSource {
    /// Heap bytes retained for the strip-walker source copy.
    #[must_use]
    fn byte_size(&self) -> usize {
        match self {
            ReferenceSource::Srgb(v) => v.capacity(),
            ReferenceSource::Linear(v) => v.capacity() * core::mem::size_of::<f32>(),
        }
    }
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
    /// Retained reference-side source data for the strip walker.
    ///
    /// `None` when the reference was constructed via
    /// `new_linear_planar` (planar-only path that doesn't retain the
    /// interleaved source). `Some(Srgb(_))` for `new` constructions
    /// (stores the original u8 bytes — 3 B/pixel). `Some(Linear(_))`
    /// for `new_linear` constructions (12 B/pixel — no compression
    /// opportunity since the input was already linear). 0.9.3 stored
    /// linear f32 for both u8 and f32 constructors, costing 12 B/pixel
    /// even for the sRGB path that already had cheaper u8 bytes
    /// available — the 4× sRGB→linear blow-up dominated the warm-ref
    /// heap regression sized by the CPU sweep on 2026-05-28.
    source: Option<ReferenceSource>,
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
            source: self.source.clone(),
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
        // 0.9.4 memory fix: retain the original u8 bytes (3 B/pixel)
        // for the strip walker instead of the 12 B/pixel linear f32
        // clone that 0.9.3 stashed. The strip walker re-derives the
        // linear bytes on demand using the same LUT path it already
        // applies to the distorted side.
        let src = Some(ReferenceSource::Srgb(rgb.to_vec()));
        // Skip re-validation in new_linear — params already validated above.
        Self::new_linear_validated(&linear, width, height, params, src)
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
        // Linear-RGB callers don't have a cheaper representation to
        // stash (f32 is already the smallest non-lossy form). Retain a
        // clone for the strip walker.
        let src = Some(ReferenceSource::Linear(rgb.to_vec()));
        Self::new_linear_validated(rgb, width, height, params, src)
    }

    /// Internal constructor that skips param validation (caller must have
    /// already called `params.validate()`).
    ///
    /// The `source` argument carries the retained strip-walker source
    /// (`Srgb`/`Linear`/`None`) — kept distinct from `rgb` so callers
    /// like `new()` can stash the cheap u8 form while still passing the
    /// linear f32 view this constructor needs to build the precompute.
    fn new_linear_validated(
        rgb: &[f32],
        width: usize,
        height: usize,
        params: ButteraugliParams,
        source: Option<ReferenceSource>,
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
                    let (sub_rgb, sw, sh) = subsample_linear_rgb_2x(rgb, width, height, &pool);
                    let sub_xyb =
                        linear_rgb_to_xyb_butteraugli(&sub_rgb, sw, sh, intensity_target, &pool);
                    pool.put(sub_rgb); // B7b: return subsample buffer to pool
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
            source,
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
                        subsample_planar_rgb_2x(r, g, b, width, height, stride, &pool);
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
                    // B7b: return subsample buffers to pool for reuse
                    pool.put(sub_r);
                    pool.put(sub_g);
                    pool.put(sub_b);
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
            // Planar constructor doesn't have interleaved source;
            // the strip walker will surface a clear error if
            // compare_strip is called on a planar-constructed
            // reference.
            source: None,
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
        self.compare_with_stop(rgb, &enough::Unstoppable)
    }

    /// Cancellable variant of [`Self::compare`].
    ///
    /// `stop` is checked once at the outermost per-scale boundary of the
    /// warm-reference compute, before any per-pixel work; a cancelled token
    /// returns [`ButteraugliError::Cancelled`]. [`enough::Unstoppable`] makes
    /// this behave identically to [`Self::compare`] at zero cost.
    ///
    /// # Errors
    /// As [`Self::compare`], plus [`ButteraugliError::Cancelled`] if `stop`
    /// signals cancellation.
    pub fn compare_with_stop(
        &self,
        rgb: &[u8],
        stop: &dyn Stop,
    ) -> Result<ButteraugliResult, ButteraugliError> {
        let expected_size = self.width * self.height * 3;

        if rgb.len() != expected_size {
            return Err(ButteraugliError::InvalidBufferSize {
                expected: expected_size,
                actual: rgb.len(),
            });
        }

        let result = self.compare_impl(rgb, stop)?;
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
        self.compare_linear_with_stop(rgb, &enough::Unstoppable)
    }

    /// Cancellable variant of [`Self::compare_linear`].
    ///
    /// `stop` is checked once at the outermost per-scale boundary of the
    /// warm-reference compute, before any per-pixel work; a cancelled token
    /// returns [`ButteraugliError::Cancelled`]. [`enough::Unstoppable`] makes
    /// this behave identically to [`Self::compare_linear`] at zero cost.
    ///
    /// # Errors
    /// As [`Self::compare_linear`], plus [`ButteraugliError::Cancelled`] if
    /// `stop` signals cancellation.
    pub fn compare_linear_with_stop(
        &self,
        rgb: &[f32],
        stop: &dyn Stop,
    ) -> Result<ButteraugliResult, ButteraugliError> {
        let expected_size = self.width * self.height * 3;

        if rgb.len() != expected_size {
            return Err(ButteraugliError::InvalidBufferSize {
                expected: expected_size,
                actual: rgb.len(),
            });
        }

        check_finite_f32(rgb, "compare linear rgb")?;

        let result = self.compare_linear_impl(rgb, stop)?;
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
        // Mirror the checked_mul in `new_linear_planar` — a stride coming from
        // an adversarial caller can otherwise overflow the per-channel buffer
        // size on 32-bit targets, panicking before the buffer length check.
        let min_size =
            stride
                .checked_mul(self.height)
                .ok_or(ButteraugliError::DimensionOverflow {
                    width: self.width,
                    height: self.height,
                })?;
        if r.len() < min_size || g.len() < min_size || b.len() < min_size {
            return Err(ButteraugliError::InvalidBufferSize {
                expected: min_size,
                actual: r.len().min(g.len()).min(b.len()),
            });
        }

        check_finite_f32(&r[..min_size], "compare planar r")?;
        check_finite_f32(&g[..min_size], "compare planar g")?;
        check_finite_f32(&b[..min_size], "compare planar b")?;

        // Non-cancellable entry point — `Unstoppable` is zero-cost.
        let result = self.compare_linear_planar_impl(r, g, b, stride, &enough::Unstoppable)?;
        if !result.score.is_finite() {
            return Err(ButteraugliError::NonFiniteResult);
        }
        Ok(result)
    }

    /// Compare a distorted planar linear RGB image, writing the diffmap
    /// into a caller-owned `Vec<f32>`.
    ///
    /// This is the buffer-recycling variant of [`Self::compare_linear_planar`] —
    /// the caller passes a persistent `Vec<f32>` that is reused across
    /// successive compares (e.g. across butteraugli-loop iterations in an
    /// encoder), avoiding the per-call fresh `width * height * 4 B`
    /// allocation that the [`ButteraugliResult::diffmap`] return path
    /// produces.
    ///
    /// On entry, `diffmap_out` may be any size; it is resized to
    /// `self.width() * self.height()` and overwritten with the diffmap.
    /// Returns the score + p-norm components; the diffmap lives in
    /// `diffmap_out` after the call.
    ///
    /// Bit-identical to `compare_linear_planar` modulo the buffer
    /// management (B7a, 2026-05-23).
    ///
    /// # Errors
    /// Same as [`Self::compare_linear_planar`].
    pub fn compare_linear_planar_into(
        &self,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        stride: usize,
        diffmap_out: &mut Vec<f32>,
    ) -> Result<(f64, f64), ButteraugliError> {
        let min_size =
            stride
                .checked_mul(self.height)
                .ok_or(ButteraugliError::DimensionOverflow {
                    width: self.width,
                    height: self.height,
                })?;
        if r.len() < min_size || g.len() < min_size || b.len() < min_size {
            return Err(ButteraugliError::InvalidBufferSize {
                expected: min_size,
                actual: r.len().min(g.len()).min(b.len()),
            });
        }

        check_finite_f32(&r[..min_size], "compare planar r")?;
        check_finite_f32(&g[..min_size], "compare planar g")?;
        check_finite_f32(&b[..min_size], "compare planar b")?;

        // Non-cancellable entry point — `Unstoppable` is zero-cost.
        let (score, pnorm_3) = self.compare_linear_planar_impl_into(
            r,
            g,
            b,
            stride,
            diffmap_out,
            &enough::Unstoppable,
        )?;
        if !score.is_finite() {
            return Err(ButteraugliError::NonFiniteResult);
        }
        Ok((score, pnorm_3))
    }

    /// Interleaved linear-RGB source data, if retained AND already
    /// stored in linear form.
    ///
    /// `Some(buf)` when the reference was built via `new_linear`
    /// (linear bytes stored as-is); `None` for `new` (stores u8 sRGB —
    /// strip walker must call [`Self::source_linear_rgb_owned`] to
    /// materialise the linear form on demand) and for
    /// `new_linear_planar` (planar constructor does not retain
    /// interleaved source).
    ///
    /// `#[doc(hidden)]` because the buffer's layout is an
    /// implementation detail shared between the precompute and
    /// strip modules. External callers should not rely on the
    /// signature.
    ///
    /// 0.9.4 behavior change: pre-0.9.4 this returned `Some(_)` for
    /// both `new` and `new_linear`-built references. After the memory
    /// fix, `new`-built references store the cheaper u8 form — the
    /// strip walker uses [`Self::source_linear_rgb_owned`] instead,
    /// which is a no-op clone when the source is already linear and a
    /// LUT-based conversion when the source is u8.
    #[doc(hidden)]
    #[must_use]
    pub fn source_linear_rgb(&self) -> Option<&[f32]> {
        match self.source.as_ref()? {
            ReferenceSource::Linear(buf) => Some(buf.as_slice()),
            ReferenceSource::Srgb(_) => None,
        }
    }

    /// Owned linear-RGB source for the strip walker — materialises the
    /// linear bytes from whichever storage form was retained.
    ///
    /// Returns `None` only when the reference was constructed via
    /// `new_linear_planar` (planar constructor doesn't retain
    /// interleaved source data at all).
    ///
    /// Allocates `width * height * 3 * 4 B` when the source is the u8
    /// sRGB form (`new()`-built); zero-copies via clone when the source
    /// is already linear f32 (`new_linear()`-built). The strip walker
    /// invokes this once per `compare_strip` call and drops the result
    /// when the per-call walk completes, so the linear buffer is alive
    /// only during the strip-walk window — not retained across calls.
    ///
    /// `#[doc(hidden)]` for the same reasons as
    /// [`Self::source_linear_rgb`].
    #[doc(hidden)]
    #[must_use]
    pub fn source_linear_rgb_owned(&self) -> Option<Vec<f32>> {
        match self.source.as_ref()? {
            ReferenceSource::Linear(buf) => Some(buf.clone()),
            ReferenceSource::Srgb(bytes) => Some(srgb_u8_to_linear_f32(bytes)),
        }
    }

    /// Drops the retained reference-side source data, freeing the
    /// per-pixel retention cost.
    ///
    /// After calling this, [`Self::compare_strip`] /
    /// [`Self::compare_linear_strip`] and the `_srgb` /
    /// `_linear_imgref` variants will return
    /// [`ButteraugliError::InvalidParameter`] for the `reference`
    /// argument — the non-strip [`Self::compare`] /
    /// [`Self::compare_linear`] / [`Self::compare_linear_planar`]
    /// paths are unaffected.
    ///
    /// Use this when the caller has determined that no strip
    /// dispatch will follow on this reference and wants to reclaim
    /// the `width * height * 3` (u8) or `width * height * 12` (f32)
    /// bytes that the source clone occupies.
    pub fn drop_strip_source(&mut self) {
        self.source = None;
    }

    /// Frees the persistent buffer pool, releasing any cached buffers
    /// held between `compare` calls.
    ///
    /// `compare` calls following `shrink_to_fit` will re-allocate
    /// transient buffers from the OS instead of reusing pooled ones,
    /// which trades a one-time allocation cost (~tens of ms at 16 MP)
    /// for the pool footprint (~tens to hundreds of MB at 16 MP+).
    ///
    /// Cached precomputed reference data (XYB pyramid, masks, retained
    /// source) is NOT affected — the warm-ref speedup over a cold
    /// `butteraugli()` call still applies.
    pub fn shrink_to_fit(&mut self) {
        self.pool.clear();
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

    /// A-priori estimate of the heap bytes a reference's **persistent
    /// precompute** will occupy, computed from dimensions + params alone —
    /// *before* the reference is built. This is the multi-resolution psycho
    /// pyramid + masks (full level, plus the half-resolution level when
    /// `params` enables multi-resolution and the image is large enough); it
    /// is the dominant, long-lived allocation of a `ButteraugliReference`
    /// and the figure a memory budget should reserve before constructing one
    /// inside a quantization loop.
    ///
    /// Exact for the planar / linear / sRGB constructors' precompute: equal
    /// to the full+half `ScaleData` byte total an actual reference reports
    /// (validated by the `estimated_reference_bytes_matches_precompute`
    /// test). It does NOT include the retained strip-walker source or the
    /// transient compare-time buffer pool — those are not part of the
    /// persistent precompute and are accounted separately by
    /// [`Self::memory_bytes`] on a live reference.
    ///
    /// `0` for degenerate sizes (`width == 0 || height == 0`).
    #[must_use]
    pub fn estimated_reference_bytes(
        width: usize,
        height: usize,
        params: &ButteraugliParams,
    ) -> usize {
        if width == 0 || height == 0 {
            return 0;
        }
        let mut total = ScaleData::estimated_byte_size(width, height);
        // Mirror the `need_half` gate in `new_linear_planar` /
        // `new`: a half-resolution level is built unless single-resolution
        // is requested and only when both dims clear the subsample floor.
        let need_half = !params.single_resolution()
            && width >= MIN_SIZE_FOR_SUBSAMPLE
            && height >= MIN_SIZE_FOR_SUBSAMPLE;
        if need_half {
            total += ScaleData::estimated_byte_size(width.div_ceil(2), height.div_ceil(2));
        }
        total
    }

    /// Heap bytes of the **persistent precompute** actually held by this
    /// live reference (full + optional half `ScaleData`). This is what
    /// [`Self::estimated_reference_bytes`] predicts a-priori.
    #[must_use]
    pub fn precompute_bytes(&self) -> usize {
        self.full.byte_size() + self.half.as_ref().map_or(0, ScaleData::byte_size)
    }

    /// Total heap bytes this reference currently retains: the persistent
    /// precompute ([`Self::precompute_bytes`]) plus the retained
    /// strip-walker source and any idle buffers cached in the internal
    /// buffer pool. Useful for memory introspection / budget accounting
    /// across batches.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.precompute_bytes()
            + self.source.as_ref().map_or(0, ReferenceSource::byte_size)
            + self.pool.retained_bytes()
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
        self.compare_srgb_with_stop(img, &enough::Unstoppable)
    }

    /// Cancellable variant of [`Self::compare_srgb`].
    ///
    /// `stop` is checked once at the outermost per-scale boundary of the
    /// warm-reference compute, before any per-pixel work; a cancelled token
    /// returns [`ButteraugliError::Cancelled`]. [`enough::Unstoppable`] makes
    /// this behave identically to [`Self::compare_srgb`] at zero cost.
    ///
    /// # Errors
    /// As [`Self::compare_srgb`], plus [`ButteraugliError::Cancelled`] if
    /// `stop` signals cancellation.
    pub fn compare_srgb_with_stop(
        &self,
        img: imgref::ImgRef<rgb::RGB8>,
        stop: &dyn Stop,
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
        self.compare_linear_with_stop(&linear, stop)
    }

    /// Compare a distorted linear RGB image (as `ImgRef<RGB<f32>>`) against the reference.
    ///
    /// # Errors
    /// Returns an error if dimensions don't match the reference.
    pub fn compare_linear_imgref(
        &self,
        img: imgref::ImgRef<rgb::RGB<f32>>,
    ) -> Result<ButteraugliResult, ButteraugliError> {
        self.compare_linear_imgref_with_stop(img, &enough::Unstoppable)
    }

    /// Cancellable variant of [`Self::compare_linear_imgref`].
    ///
    /// `stop` is checked once at the outermost per-scale boundary of the
    /// warm-reference compute, before any per-pixel work; a cancelled token
    /// returns [`ButteraugliError::Cancelled`]. [`enough::Unstoppable`] makes
    /// this behave identically to [`Self::compare_linear_imgref`] at zero cost.
    ///
    /// # Errors
    /// As [`Self::compare_linear_imgref`], plus [`ButteraugliError::Cancelled`]
    /// if `stop` signals cancellation.
    pub fn compare_linear_imgref_with_stop(
        &self,
        img: imgref::ImgRef<rgb::RGB<f32>>,
        stop: &dyn Stop,
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
        self.compare_linear_with_stop(&rgb, stop)
    }

    /// Internal comparison implementation for sRGB input.
    ///
    /// Converts sRGB to linear and delegates to the linear path.
    fn compare_impl(
        &self,
        rgb: &[u8],
        stop: &dyn Stop,
    ) -> Result<ButteraugliResult, ButteraugliError> {
        let linear = srgb_u8_to_linear_f32(rgb);
        self.compare_linear_impl(&linear, stop)
    }

    /// Internal comparison implementation for linear RGB input.
    ///
    /// Computes full-resolution diffmap using precomputed reference, then adds
    /// a single half-resolution sub-level via AddSupersampled2x. This matches
    /// C++ `ButteraugliComparator::Diffmap` which only uses one sub-level.
    ///
    /// The cooperative-cancellation check lives here, at the outermost
    /// per-scale boundary — *before* the full-res + half-res scale dispatch
    /// (`maybe_join`) below — so a `cancel()` between warm-ref compares is
    /// honoured without ever entering the per-pixel kernels in
    /// `opsin`/`psycho`/`mask`/`malta`/`blur`. Those inner loops carry no check.
    fn compare_linear_impl(
        &self,
        rgb: &[f32],
        stop: &dyn Stop,
    ) -> Result<ButteraugliResult, ButteraugliError> {
        // Cooperative cancellation: outermost per-scale boundary — checked
        // before the scale dispatch below, never inside the kernels.
        stop.check().map_err(ButteraugliError::Cancelled)?;

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
                    let (sub_rgb, sw, sh) = subsample_linear_rgb_2x(rgb, width, height, pool);
                    let sub_xyb =
                        linear_rgb_to_xyb_butteraugli(&sub_rgb, sw, sh, intensity_target, pool);
                    pool.put(sub_rgb); // B7b: return subsample buffer to pool
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
            sub.recycle(pool); // B7a: sub_diffmap buffer back to pool
        }

        let (score, pnorm_3) = compute_score_from_diffmap(&diffmap);

        Ok(ButteraugliResult {
            score,
            pnorm_3,
            diffmap: Some(diffmap.into_imgvec()),
        })
    }

    /// Internal comparison implementation for planar linear RGB input,
    /// writing the diffmap into a caller-owned `Vec<f32>` (B7a, 2026-05-23).
    ///
    /// Mirrors `compare_linear_planar_impl` but recycles the final diffmap
    /// buffer via the persistent `BufferPool` and copies the result into the
    /// caller's Vec. The caller's Vec is resized to `width * height`.
    fn compare_linear_planar_impl_into(
        &self,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        stride: usize,
        diffmap_out: &mut Vec<f32>,
        stop: &dyn Stop,
    ) -> Result<(f64, f64), ButteraugliError> {
        // Cooperative cancellation: outermost per-scale boundary — checked
        // before the scale dispatch below, never inside the kernels.
        stop.check().map_err(ButteraugliError::Cancelled)?;

        let intensity_target = self.params.intensity_target();
        let width = self.width;
        let height = self.height;
        let params = &self.params;
        let full_psycho = &self.full.psycho;
        let full_mask = &self.full.mask;
        let half_ref = self.half.as_ref();
        let pool = &self.pool;

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
                        subsample_planar_rgb_2x(r, g, b, width, height, stride, pool);
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
                    pool.put(sub_r);
                    pool.put(sub_g);
                    pool.put(sub_b);
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
            sub.recycle(pool);
        }

        let (score, pnorm_3) = compute_score_from_diffmap(&diffmap);

        // B7a: copy into caller's Vec (or move via owned buf if tight-stride)
        // then recycle the internal ImageF.data back to the pool. The caller's
        // Vec retains its existing capacity for next-iter reuse.
        let needed = width * height;
        if diffmap_out.capacity() < needed {
            diffmap_out.reserve(needed - diffmap_out.len());
        }
        // Resize so element write via index/slicing is valid.
        diffmap_out.resize(needed, 0.0);
        let dst = &mut diffmap_out[..needed];
        if diffmap.stride() == width {
            // No padding — straight memcpy from packed data.
            dst.copy_from_slice(&diffmap.data()[..needed]);
        } else {
            // Padded — copy row by row to strip stride padding.
            for y in 0..height {
                let src_row = diffmap.row(y);
                let dst_row = &mut dst[y * width..(y + 1) * width];
                dst_row.copy_from_slice(src_row);
            }
        }
        diffmap.recycle(pool);

        Ok((score, pnorm_3))
    }

    /// Internal comparison implementation for planar linear RGB input.
    fn compare_linear_planar_impl(
        &self,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        stride: usize,
        stop: &dyn Stop,
    ) -> Result<ButteraugliResult, ButteraugliError> {
        // Cooperative cancellation: outermost per-scale boundary — checked
        // before the scale dispatch below, never inside the kernels.
        stop.check().map_err(ButteraugliError::Cancelled)?;

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
                        subsample_planar_rgb_2x(r, g, b, width, height, stride, pool);
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
                    // B7b: return subsample buffers to pool for reuse
                    pool.put(sub_r);
                    pool.put(sub_g);
                    pool.put(sub_b);
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
            sub.recycle(pool); // B7a: sub_diffmap buffer back to pool
        }

        let (score, pnorm_3) = compute_score_from_diffmap(&diffmap);

        Ok(ButteraugliResult {
            score,
            pnorm_3,
            diffmap: Some(diffmap.into_imgvec()),
        })
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

    // Use precomputed mask directly (no copy needed — read-only reference).
    // B7a (2026-05-23): diffmap output now sourced from BufferPool so the
    // ~4 MB/call allocation at 1024² is recycled across buttloop iters.
    let diffmap = combine_channels_to_diffmap_fused(
        &precomputed_mask.mask,
        &ps1.lf,
        &ps2.lf,
        &block_diff_ac,
        params.xmul(),
        pool,
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
#[archmage::autoversion]
fn combine_channels_to_diffmap_fused(
    _token: archmage::SimdToken,
    mask: &ImageF,
    lf1: &Image3F,
    lf2: &Image3F,
    block_diff_ac: &Image3F,
    xmul: f32,
    pool: &BufferPool,
) -> ImageF {
    use crate::consts::{
        MASK_DC_Y_MUL, MASK_DC_Y_OFFSET, MASK_DC_Y_SCALER, MASK_Y_MUL, MASK_Y_OFFSET, MASK_Y_SCALER,
    };

    let width = mask.width();
    let height = mask.height();
    // B7a: pool-backed allocation, recycled on subsequent compare calls.
    // Every pixel is written before being read so dirty is safe.
    let mut diffmap = ImageF::from_pool_dirty(width, height, pool);
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

/// L2 difference (symmetric) - autoversioned for autovectorization.
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

/// L2 difference (symmetric, write-only) - autoversioned for autovectorization.
///
/// Like `l2_diff` but overwrites diffmap instead of accumulating.
/// Use when diffmap is uninitialized or dirty.
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

/// Computes max-norm score AND libjxl 3-norm aggregation from a diffmap in
/// a single pass. See `diff.rs::compute_score_from_diffmap` for the full
/// rationale; both implementations are kept in sync.
#[archmage::autoversion]
fn compute_score_from_diffmap(_token: archmage::SimdToken, diffmap: &ImageF) -> (f64, f64) {
    let width = diffmap.width();
    let height = diffmap.height();

    if width * height == 0 {
        return (0.0, 0.0);
    }

    let mut max_lanes = [0.0f32; 8];
    let mut sum_p3 = [0.0f64; 8];
    let mut sum_p6 = [0.0f64; 8];
    let mut sum_p12 = [0.0f64; 8];

    for y in 0..height {
        let row = diffmap.row(y);
        for chunk in row.chunks_exact(8) {
            for i in 0..8 {
                let v = chunk[i];
                if v > max_lanes[i] {
                    max_lanes[i] = v;
                }
                let d = v as f64;
                let d3 = d * d * d;
                sum_p3[i] += d3;
                let d6 = d3 * d3;
                sum_p6[i] += d6;
                sum_p12[i] += d6 * d6;
            }
        }
        for &v in row.chunks_exact(8).remainder() {
            if v > max_lanes[0] {
                max_lanes[0] = v;
            }
            let d = v as f64;
            let d3 = d * d * d;
            sum_p3[0] += d3;
            let d6 = d3 * d3;
            sum_p6[0] += d6;
            sum_p12[0] += d6 * d6;
        }
    }

    let mut max_val = max_lanes[0];
    for &m in &max_lanes[1..] {
        if m > max_val {
            max_val = m;
        }
    }

    let total_p3: f64 = sum_p3.iter().sum();
    let total_p6: f64 = sum_p6.iter().sum();
    let total_p12: f64 = sum_p12.iter().sum();
    let one_per_pixels = 1.0_f64 / ((width * height) as f64);
    let v0 = (one_per_pixels * total_p3).powf(1.0 / 3.0);
    let v1 = (one_per_pixels * total_p6).powf(1.0 / 6.0);
    let v2 = (one_per_pixels * total_p12).powf(1.0 / 12.0);
    let pnorm_3 = (v0 + v1 + v2) / 3.0;

    (max_val as f64, pnorm_3)
}

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
///
/// B7b (2026-05-23): output buffer is sourced from the caller's `BufferPool`
/// — the ~3 MB allocation per call at 1024² is recycled across buttloop
/// iters. The interior loop + right/bottom edge handlers together write
/// every cell of the output (even for odd dimensions), so no zero-fill is
/// required even though `pool.take` returns potentially-stale data.
fn subsample_linear_rgb_2x(
    rgb: &[f32],
    width: usize,
    height: usize,
    pool: &BufferPool,
) -> (Vec<f32>, usize, usize) {
    let out_width = width.div_ceil(2);
    let out_height = height.div_ceil(2);
    let out_size = out_width * out_height * 3;
    let mut output = pool.take(out_size);

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
#[archmage::autoversion]
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
///
/// B7b (2026-05-23): output buffers are sourced from the caller's
/// `BufferPool` so the 3× ~1 MB allocations per call at 1024² are recycled
/// across buttloop iters. Caller is responsible for `pool.put`-ing the
/// returned Vecs after consumption. The interior loop + right/bottom edge
/// handlers together write every cell, so no zero-fill is needed even for
/// odd dimensions where `pool.take` may return stale data.
fn subsample_planar_rgb_2x(
    r: &[f32],
    g: &[f32],
    b: &[f32],
    width: usize,
    height: usize,
    stride: usize,
    pool: &BufferPool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, usize, usize) {
    let out_width = width.div_ceil(2);
    let out_height = height.div_ceil(2);
    let out_size = out_width * out_height;
    let out_r = pool.take(out_size);
    let out_g = pool.take(out_size);
    let out_b = pool.take(out_size);
    let (mut out_r, mut out_g, mut out_b) = (out_r, out_g, out_b);

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

    /// The a-priori `estimated_reference_bytes` must EXACTLY equal the
    /// persistent precompute an actual reference reports, across sizes that
    /// span the `need_half` gate (small single-res, multi-res, odd dims that
    /// exercise `div_ceil` + stride alignment) and across both
    /// constructors. This pins the estimate to the real allocation so the
    /// jxl-encoder budget guard reserves the right figure (issue
    /// imazen/jxl-encoder#93) — not a guessed constant.
    #[test]
    fn estimated_reference_bytes_matches_precompute() {
        // (width, height) chosen to hit: below subsample floor (single-res),
        // just above it, stride-aligned, and odd dims (div_ceil + padding).
        for &(w, h) in &[(8, 8), (14, 20), (15, 15), (37, 41), (64, 64), (100, 73)] {
            let params = ButteraugliParams::default();
            // sRGB constructor (retains a source; precompute is source-free).
            let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i * 7 % 251) as u8).collect();
            let reference = ButteraugliReference::new(&rgb, w, h, params.clone())
                .expect("reference should build");
            let est = ButteraugliReference::estimated_reference_bytes(w, h, &params);
            assert_eq!(
                est,
                reference.precompute_bytes(),
                "estimate must equal actual precompute at {w}x{h}",
            );
            // memory_bytes includes the retained sRGB source + pool, so it is
            // strictly >= the precompute-only estimate.
            assert!(
                reference.memory_bytes() >= est,
                "memory_bytes {} must be >= precompute estimate {est} at {w}x{h}",
                reference.memory_bytes(),
            );
            // half-level presence must agree with the estimate's gate.
            let need_half = w >= MIN_SIZE_FOR_SUBSAMPLE && h >= MIN_SIZE_FOR_SUBSAMPLE;
            assert_eq!(reference.half.is_some(), need_half, "half gate at {w}x{h}");
        }

        // single_resolution=true suppresses the half level in both the
        // estimate and the actual build.
        let params = ButteraugliParams::new().with_single_resolution(true);
        let rgb: Vec<u8> = vec![128; 64 * 64 * 3];
        let reference =
            ButteraugliReference::new(&rgb, 64, 64, params.clone()).expect("build single-res");
        assert!(reference.half.is_none());
        assert_eq!(
            ButteraugliReference::estimated_reference_bytes(64, 64, &params),
            reference.precompute_bytes(),
        );

        // Degenerate dims estimate to 0 without panicking.
        assert_eq!(
            ButteraugliReference::estimated_reference_bytes(0, 64, &params),
            0
        );
    }

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

    // W44-phase3-B8 (2026-05-23): regression test for the iir-blur
    // gaussian_blur_iir stride bug discovered in B7a+b. Two consecutive
    // compare_linear_planar calls on the same reference must produce
    // byte-identical scores under BOTH the FIR and IIR blur backends.
    //
    // Pre-fix repro on `cargo test --features iir-blur`: 3 calls yielded
    // wildly different scores (e.g. 40137 → 1264 → 119212) because
    // `gaussian_blur_iir` used `chunks_exact(width)` to walk the input
    // buffer instead of stride-aware row addressing. With `from_pool_dirty`
    // returning buffers with stale padding bytes between width and stride,
    // the misaligned IIR result depended on whatever previous call had
    // left in the padding columns. See blur_iir.rs for the fix.
    #[test]
    fn test_compare_linear_planar_iir_determinism_repro() {
        let width = 48;
        let height = 48;
        let r1: Vec<f32> = (0..width * height)
            .map(|i| ((i % 32) as f32) / 32.0)
            .collect();
        let g1: Vec<f32> = (0..width * height)
            .map(|i| ((i % 16) as f32) / 16.0)
            .collect();
        let b1: Vec<f32> = (0..width * height)
            .map(|i| ((i % 8) as f32) / 8.0)
            .collect();
        let r2: Vec<f32> = r1.iter().map(|&v| (v * 0.97).min(1.0)).collect();
        let g2: Vec<f32> = g1.iter().map(|&v| (v * 0.97).min(1.0)).collect();
        let b2: Vec<f32> = b1.iter().map(|&v| (v * 0.97).min(1.0)).collect();

        let reference = ButteraugliReference::new_linear_planar(
            &r1,
            &g1,
            &b1,
            width,
            height,
            width,
            ButteraugliParams::default().with_compute_diffmap(true),
        )
        .expect("new reference");

        let mut scores = Vec::new();
        for _ in 0..3 {
            let res = reference
                .compare_linear_planar(&r2, &g2, &b2, width)
                .expect("compare_linear_planar");
            scores.push(res.score);
        }
        eprintln!("scores across 3 calls: {scores:?}");
        for w in scores.windows(2) {
            let delta = (w[0] - w[1]).abs();
            assert!(
                delta < 1e-6,
                "consecutive scores must be deterministic: {} vs {} (delta {})",
                w[0],
                w[1],
                delta
            );
        }
    }

    #[test]
    fn test_compare_linear_planar_into_matches_owned() {
        let width = 48;
        let height = 48;

        let r1: Vec<f32> = (0..width * height)
            .map(|i| ((i % 32) as f32) / 32.0)
            .collect();
        let g1: Vec<f32> = (0..width * height)
            .map(|i| ((i % 16) as f32) / 16.0)
            .collect();
        let b1: Vec<f32> = (0..width * height)
            .map(|i| ((i % 8) as f32) / 8.0)
            .collect();
        let r2: Vec<f32> = r1.iter().map(|&v| (v * 0.97).min(1.0)).collect();
        let g2: Vec<f32> = g1.iter().map(|&v| (v * 0.97).min(1.0)).collect();
        let b2: Vec<f32> = b1.iter().map(|&v| (v * 0.97).min(1.0)).collect();

        let reference = ButteraugliReference::new_linear_planar(
            &r1,
            &g1,
            &b1,
            width,
            height,
            width,
            ButteraugliParams::default().with_compute_diffmap(true),
        )
        .expect("should create reference");

        // Owned variant
        let owned = reference
            .compare_linear_planar(&r2, &g2, &b2, width)
            .expect("compare_linear_planar");
        let owned_diffmap = owned.diffmap.as_ref().expect("diffmap present");

        // Into variant — first call with empty Vec
        let mut diffmap_out: Vec<f32> = Vec::new();
        let (score_into, pnorm_into) = reference
            .compare_linear_planar_into(&r2, &g2, &b2, width, &mut diffmap_out)
            .expect("compare_linear_planar_into");

        assert_eq!(diffmap_out.len(), width * height);
        assert!(
            (owned.score - score_into).abs() < 1e-12,
            "score mismatch owned={} into={}",
            owned.score,
            score_into
        );
        assert!(
            (owned.pnorm_3 - pnorm_into).abs() < 1e-12,
            "pnorm3 mismatch owned={} into={}",
            owned.pnorm_3,
            pnorm_into
        );
        for (i, (&a, &b)) in owned_diffmap
            .buf()
            .iter()
            .zip(diffmap_out.iter())
            .enumerate()
        {
            assert!((a - b).abs() < 1e-7, "diffmap[{i}]: owned={a}, into={b}");
        }

        // Second call should reuse the existing Vec capacity (no allocation
        // observed externally, but check that the contents update correctly).
        let cap_before = diffmap_out.capacity();
        let (score_into_2, _) = reference
            .compare_linear_planar_into(&r2, &g2, &b2, width, &mut diffmap_out)
            .expect("compare_linear_planar_into second call");
        let cap_after = diffmap_out.capacity();
        assert_eq!(
            cap_before, cap_after,
            "capacity should not grow across reuse"
        );
        assert!(
            (score_into - score_into_2).abs() < 1e-12,
            "deterministic across reuse"
        );
    }

    #[test]
    fn test_compare_linear_planar_into_resizes_undersized() {
        let width = 16;
        let height = 16;
        let r = vec![0.5f32; width * height];
        let g = vec![0.5f32; width * height];
        let b = vec![0.5f32; width * height];

        let reference = ButteraugliReference::new_linear_planar(
            &r,
            &g,
            &b,
            width,
            height,
            width,
            ButteraugliParams::default().with_compute_diffmap(true),
        )
        .expect("create");

        let mut diffmap_out: Vec<f32> = vec![999.0; 7];
        let _ = reference
            .compare_linear_planar_into(&r, &g, &b, width, &mut diffmap_out)
            .expect("into");
        assert_eq!(diffmap_out.len(), width * height);
    }

    #[test]
    fn test_compare_linear_planar_into_rejects_short_buffers() {
        let width = 16;
        let height = 16;
        let r = vec![0.5f32; width * height];
        let g = vec![0.5f32; width * height];
        let b = vec![0.5f32; width * height];
        let reference = ButteraugliReference::new_linear_planar(
            &r,
            &g,
            &b,
            width,
            height,
            width,
            ButteraugliParams::default(),
        )
        .expect("create");

        let mut diffmap_out: Vec<f32> = Vec::new();
        let short = vec![0.5f32; 4];
        let result = reference.compare_linear_planar_into(&short, &g, &b, width, &mut diffmap_out);
        assert!(matches!(
            result,
            Err(ButteraugliError::InvalidBufferSize { .. })
        ));
    }
}
