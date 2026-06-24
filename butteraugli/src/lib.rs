//! # Butteraugli
//!
//! Butteraugli is a perceptual image quality metric developed by Google.
//! This is a Rust port of the butteraugli algorithm from libjxl.
//!
//! The metric is based on:
//! - Opsin: dynamics of photosensitive chemicals in the retina
//! - XYB: hybrid opponent/trichromatic color space
//! - Visual masking: how features hide other features
//! - Multi-scale analysis: UHF, HF, MF, LF frequency components
//!
//! ## Quality Thresholds
//!
//! - Score < 1.0: Images are perceived as identical
//! - Score 1.0-2.0: Subtle differences may be noticeable
//! - Score > 2.0: Visible difference between images
//!
//! ## Example
//!
//! ```rust
//! use butteraugli::{butteraugli, ButteraugliParams};
//! use imgref::Img;
//! use rgb::RGB8;
//!
//! // Create two 8x8 RGB images. Images smaller than 8x8 (down to 1x1)
//! // are reflect(mirror)-padded up to the 8x8 floor and scored too.
//! let width = 8;
//! let height = 8;
//! let pixels: Vec<RGB8> = (0..width * height)
//!     .map(|i| RGB8::new((i % 256) as u8, ((i * 2) % 256) as u8, ((i * 3) % 256) as u8))
//!     .collect();
//!
//! let img1 = Img::new(pixels.clone(), width, height);
//! let img2 = Img::new(pixels, width, height); // Identical images
//!
//! let params = ButteraugliParams::default();
//! let result = butteraugli(img1.as_ref(), img2.as_ref(), &params).unwrap();
//!
//! // Identical images should have score ~0
//! assert!(result.score < 0.01);
//! ```
//!
//! ## Input scaling and `intensity_target`
//!
//! Butteraugli models absolute luminance. Input pixel values are mapped to
//! nits (cd/m²) before the opsin stage:
//!
//! - [`butteraugli`] takes gamma-encoded sRGB u8; each channel is decoded
//!   with the sRGB EOTF to linear `[0.0, 1.0]`.
//! - [`butteraugli_linear`] takes linear RGB f32 directly.
//!
//! In both cases linear `1.0` is then scaled to
//! [`ButteraugliParams::intensity_target`] nits (default `80.0`, the SDR
//! convention used by libjxl's `butteraugli_main`). Linear values above
//! `1.0` are accepted and map above `intensity_target` nits; for HDR
//! content, scale your data so `1.0` is the mastering/display peak and set
//! `intensity_target` to that peak in nits. Negative values are clamped to
//! `0.0` inside the opsin stage; NaN/Inf inputs are rejected with
//! [`ButteraugliError::NonFiniteResult`].
//!
//! ## Features
//!
//! - **`rayon`** *(default)*: multi-threaded blur and Malta passes
//! - **`avx512`** *(default)*: AVX-512 runtime dispatch (used only when the CPU supports it)
//! - **`iir-blur`**: O(N) recursive Gaussian instead of FIR convolution — faster on
//!   non-AVX-512 hardware but NOT score-parity with libjxl; off by default
//! - **`unsafe-performance`**: unchecked indexing in hot loops (pre-validated ranges)
//! - **`internals`**: expose internal modules for testing/benchmarking (unstable API)
//!
//! ## References
//!
//! - <https://github.com/google/butteraugli>
//! - <https://github.com/libjxl/libjxl>

#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
// Allow C++ constant formats (ported from libjxl with exact values)
#![allow(clippy::unreadable_literal)]
#![allow(clippy::inconsistent_digit_grouping)]
#![allow(clippy::excessive_precision)]
// Allow mul_add style from C++ (may affect numerical parity)
#![allow(clippy::suboptimal_flops)]
// Allow common patterns in numerical code ported from C++
#![allow(clippy::many_single_char_names)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::manual_saturating_arithmetic)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_possible_wrap)]
// These are nice-to-have but not critical for initial release
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::collapsible_else_if)]
#![allow(clippy::if_not_else)]
#![allow(clippy::imprecise_flops)]
#![allow(clippy::implicit_saturating_sub)]
#![allow(clippy::useless_let_if_seq)]
// archmage uses _token parameters implicitly via proc macros
#![allow(clippy::used_underscore_binding)]
// archmage 0.9.20 deprecates the SimdToken parameter on #[autoversion] functions
// and the implicit `scalar` fallback in incant! tier lists. Both removals are
// mechanical (~20 functions across 6 modules + every incant! call site). Tracking
// as a follow-up so this PR stays focused on the iir-blur feature.
// TODO(post-iir-blur): migrate per archmage 0.9.20 deprecation notes, then remove.
#![allow(deprecated)]

// Internal modules - exposed with "internals" feature for testing/benchmarking.
// The pub(crate) variants allow dead_code because these modules contain items
// that are only reachable via the "internals" feature or cpp-parity tests.
#[cfg(feature = "internals")]
pub mod blur;
#[cfg(not(feature = "internals"))]
#[allow(dead_code)]
pub(crate) mod blur;

#[cfg(feature = "iir-blur")]
#[allow(dead_code)]
pub(crate) mod blur_iir;

#[cfg(feature = "internals")]
pub mod consts;
#[cfg(not(feature = "internals"))]
#[allow(dead_code)]
pub(crate) mod consts;

mod diff;

#[cfg(feature = "internals")]
pub mod image;
#[cfg(not(feature = "internals"))]
#[allow(dead_code)]
pub(crate) mod image;

#[cfg(feature = "internals")]
pub mod malta;
#[cfg(not(feature = "internals"))]
pub(crate) mod malta;

#[cfg(feature = "internals")]
pub mod mask;
#[cfg(not(feature = "internals"))]
#[allow(dead_code)]
pub(crate) mod mask;

#[cfg(feature = "internals")]
pub mod opsin;
#[cfg(not(feature = "internals"))]
#[allow(dead_code)]
pub(crate) mod opsin;

pub mod precompute;
pub use precompute::ButteraugliReference;

mod strip;
pub use strip::{
    ButteraugliStripConfig, HALO_ROWS_DEFAULT, MIN_STRIP_HEIGHT, butteraugli_linear_strip,
    butteraugli_linear_strip_with_config, butteraugli_linear_strip_with_stop, butteraugli_strip,
    butteraugli_strip_with_config, butteraugli_strip_with_stop,
};

#[cfg(feature = "internals")]
pub mod psycho;
#[cfg(not(feature = "internals"))]
pub(crate) mod psycho;

// Re-export imgref and rgb types for convenience
pub use imgref::{Img, ImgRef, ImgVec};
pub use rgb::{RGB, RGB8};

/// Re-export of the [`enough`] cooperative-cancellation crate, so callers of
/// the `*_with_stop` entry points (e.g. [`butteraugli_with_stop`]) can name
/// [`enough::Stop`] / [`enough::Unstoppable`] / [`enough::StopReason`]
/// without taking a direct dependency.
pub use enough;

/// Reflect-101 index map (OpenCV `BORDER_REFLECT_101`): fold an
/// out-of-range index `i` back into `[0, n)` by mirroring at the borders
/// without repeating the edge sample. Identity for `i < n`, so the
/// original samples land at `[0, n)` after padding; `n <= 1` collapses
/// to 0 (a single row/column replicates).
#[inline]
fn reflect_index(i: usize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let mut k = i % period;
    if k >= n {
        k = period - k;
    }
    k
}

/// Reflect(mirror)-pad an `ImgRef<T>` up to `min` px on each axis so
/// butteraugli can score images below its 8×8 floor (the blur + edge
/// passes need ≥8 px). Returns `None` when the image is already ≥ `min`
/// on both axes (the caller scores the borrow directly) or is empty. The
/// original pixels occupy the top-left `w × h` region of the result, so
/// a diffmap computed on the padded image can be cropped back with
/// [`crop_diffmap`].
fn reflect_pad_to_min<T: Copy>(img: ImgRef<T>, min: usize) -> Option<ImgVec<T>> {
    let (w, h) = (img.width(), img.height());
    if w == 0 || h == 0 {
        return None;
    }
    let (pw, ph) = (w.max(min), h.max(min));
    if pw == w && ph == h {
        return None;
    }
    let rows: Vec<&[T]> = img.rows().collect();
    let mut out = Vec::with_capacity(pw * ph);
    for y in 0..ph {
        let srow = rows[reflect_index(y, h)];
        for x in 0..pw {
            out.push(srow[reflect_index(x, w)]);
        }
    }
    Some(Img::new(out, pw, ph))
}

/// Crop a per-pixel diffmap back to the logical `w × h` (top-left
/// sub-rectangle) after the inputs were reflect-padded by
/// [`reflect_pad_to_min`]. No-op when already the requested size.
fn crop_diffmap(dm: ImgVec<f32>, w: usize, h: usize) -> ImgVec<f32> {
    if dm.width() == w && dm.height() == h {
        return dm;
    }
    let (buf, cw, ch) = dm.as_ref().sub_image(0, 0, w, h).to_contiguous_buf();
    Img::new(buf.into_owned(), cw, ch)
}

/// Error type for butteraugli operations.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ButteraugliError {
    /// Image (or strip height) is below the supported minimum.
    ///
    /// [`butteraugli`] / [`butteraugli_linear`] return this only for empty
    /// (zero-dimension) inputs — sub-8px images are reflect-padded and
    /// scored. The strip API ([`butteraugli_strip`] and the
    /// `ButteraugliReference::compare_*strip*` methods) returns it for
    /// images smaller than 8×8 and for `strip_height <`
    /// [`MIN_STRIP_HEIGHT`] (reported as `width = strip_height`,
    /// `height = MIN_STRIP_HEIGHT`).
    #[non_exhaustive]
    ImageTooSmall {
        /// Image width.
        width: usize,
        /// Image height.
        height: usize,
    },
    /// Image dimensions don't match.
    #[non_exhaustive]
    DimensionMismatch {
        /// First image width.
        w1: usize,
        /// First image height.
        h1: usize,
        /// Second image width.
        w2: usize,
        /// Second image height.
        h2: usize,
    },
    /// Image dimensions are invalid (legacy variant).
    #[non_exhaustive]
    #[doc(hidden)]
    InvalidDimensions {
        /// Width provided.
        width: usize,
        /// Height provided.
        height: usize,
    },
    /// Buffer size doesn't match expected size (legacy variant).
    #[non_exhaustive]
    #[doc(hidden)]
    InvalidBufferSize {
        /// Expected buffer size.
        expected: usize,
        /// Actual buffer size.
        actual: usize,
    },
    /// A parameter value is out of valid range.
    #[non_exhaustive]
    InvalidParameter {
        /// Parameter name.
        name: &'static str,
        /// The invalid value.
        value: f64,
        /// Why it's invalid.
        reason: &'static str,
    },
    /// Image dimensions would overflow buffer size calculations.
    #[non_exhaustive]
    DimensionOverflow {
        /// Image width.
        width: usize,
        /// Image height.
        height: usize,
    },
    /// Score computation produced NaN or infinity (usually from non-finite input pixels).
    NonFiniteResult,
    /// The computation was cooperatively cancelled via the
    /// [`enough::Stop`] token passed to a `*_with_stop` entry point
    /// (e.g. [`butteraugli_with_stop`]). Carries the
    /// [`enough::StopReason`] (cancelled vs. timed out).
    Cancelled(enough::StopReason),
}

impl std::fmt::Display for ButteraugliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ImageTooSmall { width, height } => {
                write!(f, "image or strip too small: {width}x{height}")
            }
            Self::DimensionMismatch { w1, h1, w2, h2 } => {
                write!(f, "image dimensions don't match: {w1}x{h1} vs {w2}x{h2}")
            }
            Self::InvalidDimensions { width, height } => {
                write!(f, "invalid dimensions: {width}x{height} (minimum 8x8)")
            }
            Self::InvalidBufferSize { expected, actual } => {
                write!(
                    f,
                    "buffer size {actual} doesn't match expected size {expected}"
                )
            }
            Self::InvalidParameter {
                name,
                value,
                reason,
            } => {
                write!(f, "invalid parameter {name}={value}: {reason}")
            }
            Self::DimensionOverflow { width, height } => {
                write!(
                    f,
                    "image dimensions {width}x{height} overflow buffer size calculation"
                )
            }
            Self::Cancelled(reason) => {
                write!(f, "butteraugli computation cancelled: {reason}")
            }
            Self::NonFiniteResult => {
                write!(
                    f,
                    "score computation produced NaN or infinity (check input pixels)"
                )
            }
        }
    }
}

impl std::error::Error for ButteraugliError {}

/// Butteraugli comparison parameters.
///
/// Use the builder pattern to construct:
/// ```rust
/// use butteraugli::ButteraugliParams;
///
/// let params = ButteraugliParams::new()
///     .with_intensity_target(250.0)  // HDR display
///     .with_hf_asymmetry(1.5)        // Penalize new artifacts more
///     .with_compute_diffmap(true);   // Generate per-pixel difference map
/// ```
#[derive(Debug, Clone)]
pub struct ButteraugliParams {
    hf_asymmetry: f32,
    xmul: f32,
    intensity_target: f32,
    compute_diffmap: bool,
    single_resolution: bool,
}

impl Default for ButteraugliParams {
    fn default() -> Self {
        Self {
            hf_asymmetry: 1.0,
            xmul: 1.0,
            intensity_target: 80.0,
            compute_diffmap: false,
            single_resolution: false,
        }
    }
}

impl ButteraugliParams {
    /// Creates a new `ButteraugliParams` with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the intensity target — the luminance in nits (cd/m²) that a
    /// linear input value of `1.0` is mapped to before the opsin stage.
    ///
    /// Default is `80.0`, the SDR convention used by libjxl's
    /// `butteraugli_main`. For HDR content, scale your linear input so
    /// `1.0` is the mastering/display peak and set this to that peak in
    /// nits (e.g. `250.0`, `1000.0`). Internally every linear RGB sample
    /// is multiplied by this value, so it must be finite and positive
    /// (checked by [`Self::validate`]).
    #[must_use]
    pub fn with_intensity_target(mut self, intensity_target: f32) -> Self {
        self.intensity_target = intensity_target;
        self
    }

    /// Sets the HF asymmetry multiplier.
    /// Values > 1.0 penalize new high-frequency artifacts more than blurring.
    #[must_use]
    pub fn with_hf_asymmetry(mut self, hf_asymmetry: f32) -> Self {
        self.hf_asymmetry = hf_asymmetry;
        self
    }

    /// Sets the X channel multiplier.
    #[must_use]
    pub fn with_xmul(mut self, xmul: f32) -> Self {
        self.xmul = xmul;
        self
    }

    /// Sets whether to compute the per-pixel difference map.
    ///
    /// When `true`, the result will include an `ImgVec<f32>` difference map.
    /// When `false` (default), the diffmap field will be `None`, which is faster.
    #[must_use]
    pub fn with_compute_diffmap(mut self, compute_diffmap: bool) -> Self {
        self.compute_diffmap = compute_diffmap;
        self
    }

    /// Returns the HF asymmetry multiplier.
    #[must_use]
    pub fn hf_asymmetry(&self) -> f32 {
        self.hf_asymmetry
    }

    /// Returns the X channel multiplier.
    #[must_use]
    pub fn xmul(&self) -> f32 {
        self.xmul
    }

    /// Returns the intensity target in nits.
    #[must_use]
    pub fn intensity_target(&self) -> f32 {
        self.intensity_target
    }

    /// Returns whether to compute the per-pixel difference map.
    #[must_use]
    pub fn compute_diffmap(&self) -> bool {
        self.compute_diffmap
    }

    /// Skip the half-resolution pass for faster approximate results.
    ///
    /// The half-resolution pass contributes ~15% weight to the final diffmap.
    /// Skipping it saves ~25% of computation. Useful for encoder tuning loops
    /// where precise scores aren't needed.
    #[must_use]
    pub fn with_single_resolution(mut self, single_resolution: bool) -> Self {
        self.single_resolution = single_resolution;
        self
    }

    /// Returns whether single-resolution mode is enabled.
    #[must_use]
    pub fn single_resolution(&self) -> bool {
        self.single_resolution
    }

    /// Validates that all parameter values are in acceptable ranges.
    ///
    /// Called automatically by all public entry points. Returns an error if
    /// any parameter would cause division by zero, NaN propagation, or
    /// garbage results.
    ///
    /// # Errors
    ///
    /// Returns [`ButteraugliError::InvalidParameter`] if:
    /// - `hf_asymmetry` is not finite or not positive
    /// - `intensity_target` is not finite or not positive
    /// - `xmul` is not finite or is negative
    pub fn validate(&self) -> Result<(), ButteraugliError> {
        if !self.hf_asymmetry.is_finite() || self.hf_asymmetry <= 0.0 {
            return Err(ButteraugliError::InvalidParameter {
                name: "hf_asymmetry",
                value: self.hf_asymmetry as f64,
                reason: "must be finite and positive",
            });
        }
        if !self.intensity_target.is_finite() || self.intensity_target <= 0.0 {
            return Err(ButteraugliError::InvalidParameter {
                name: "intensity_target",
                value: self.intensity_target as f64,
                reason: "must be finite and positive",
            });
        }
        if !self.xmul.is_finite() || self.xmul < 0.0 {
            return Err(ButteraugliError::InvalidParameter {
                name: "xmul",
                value: self.xmul as f64,
                reason: "must be finite and non-negative",
            });
        }
        Ok(())
    }
}

/// Check that all values in a float slice are finite (not NaN or Inf).
pub(crate) fn check_finite_f32(
    data: &[f32],
    context: &'static str,
) -> Result<(), ButteraugliError> {
    for &v in data {
        if !v.is_finite() {
            return Err(ButteraugliError::NonFiniteResult);
        }
    }
    let _ = context;
    Ok(())
}

/// Check that all pixels in an `ImgRef<RGB<f32>>` are finite.
fn check_finite_rgb_imgref(img: ImgRef<RGB<f32>>) -> Result<(), ButteraugliError> {
    for row in img.rows() {
        for px in row {
            if !px.r.is_finite() || !px.g.is_finite() || !px.b.is_finite() {
                return Err(ButteraugliError::NonFiniteResult);
            }
        }
    }
    Ok(())
}

/// Quality threshold for "good" (images look the same).
pub const BUTTERAUGLI_GOOD: f64 = 1.0;

/// Quality threshold for "bad" (visible difference).
pub const BUTTERAUGLI_BAD: f64 = 2.0;

/// libjxl-style p-norm of a slice — the average of three p-norms at
/// exponents `p`, `2p`, `4p`. Used by [`ButteraugliResult::pnorm`] for
/// `p ≠ 3` (the `p == 3` case is precomputed during scoring).
fn pnorm_slice(diffmap: &[f32], p: f64) -> f64 {
    if diffmap.is_empty() {
        return f64::NAN;
    }
    let mut sum = [0.0_f64; 3];
    for &v in diffmap {
        let d = v as f64;
        let mut acc = d.powf(p);
        sum[0] += acc;
        acc *= acc;
        sum[1] += acc;
        acc *= acc;
        sum[2] += acc;
    }
    let one_per_pixels = 1.0_f64 / diffmap.len() as f64;
    let mut v = 0.0_f64;
    for (i, &s) in sum.iter().enumerate() {
        let exponent = 1.0_f64 / (p * f64::from(1u32 << i));
        v += (one_per_pixels * s).powf(exponent);
    }
    v / 3.0
}

/// Butteraugli image comparison result.
///
/// [`score`](Self::score) is the global max-norm — the historical
/// butteraugli score, with `<1.0 = good`, `>2.0 = bad` for pass/fail gating.
/// [`pnorm_3`](Self::pnorm_3) is the libjxl-style 3-norm aggregation reported
/// by `butteraugli_main --pnorm` and used in the Cloudinary CID22 paper —
/// useful for rate-distortion sweeps where averaging tail and bulk distortion
/// is more informative than max alone. Both are produced in a single fused
/// reduction pass during the comparison and available regardless of whether
/// `compute_diffmap` was enabled.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ButteraugliResult {
    /// Max-norm difference score. `<1.0` is "good", `>2.0` is "bad".
    /// Equivalent to libjxl's `ButteraugliScoreFromDiffmap`.
    pub score: f64,
    /// libjxl 3-norm aggregation — average of three p-norms at exponents
    /// 3, 6, 12, matching `lib/extras/metrics.cc:ComputeDistanceP` at p=3.
    /// Always populated; no extra allocation beyond the transient internal
    /// diffmap (which is freed before return when `compute_diffmap` is false).
    pub pnorm_3: f64,
    /// Per-pixel difference map (only present if `compute_diffmap` was true).
    pub diffmap: Option<ImgVec<f32>>,
}

impl ButteraugliResult {
    /// libjxl-style p-norm of the diffmap (average of three p-norms at
    /// `p`, `2p`, `4p`, matching `lib/extras/metrics.cc:ComputeDistanceP`).
    ///
    /// `p = 3.0` returns the precomputed [`Self::pnorm_3`] without touching
    /// the diffmap. Other `p` values require `compute_diffmap = true`;
    /// returns `None` if the diffmap isn't present.
    #[must_use]
    pub fn pnorm(&self, p: f64) -> Option<f64> {
        if (p - 3.0).abs() < 1e-6 {
            return Some(self.pnorm_3);
        }
        let dm = self.diffmap.as_ref()?;
        // ImgVec built via our diff path is always contiguous (stride == width).
        debug_assert_eq!(dm.buf().len(), dm.width() * dm.height());
        Some(pnorm_slice(dm.buf(), p))
    }

    /// Max-norm of the diffmap — the same value as [`Self::score`], named
    /// explicitly so call sites can be unambiguous about which aggregation
    /// they want when [`Self::pnorm_3`] is also in play.
    #[must_use]
    pub fn max_norm(&self) -> f64 {
        self.score
    }
}

/// Computes butteraugli score between two sRGB images.
///
/// This function accepts 8-bit sRGB data via `ImgRef<RGB8>` and internally converts
/// to linear RGB for perceptual comparison (linear `1.0` maps to
/// [`ButteraugliParams::intensity_target`] nits — see the crate-level
/// "Input scaling" notes).
///
/// Images smaller than 8×8 (down to 1×1) are reflect(mirror)-padded up to
/// the 8×8 floor and scored; the returned diffmap is cropped back to the
/// input dimensions.
///
/// # Arguments
/// * `img1` - First image (sRGB, supports stride via ImgRef)
/// * `img2` - Second image (sRGB, supports stride via ImgRef)
/// * `params` - Comparison parameters
///
/// # Returns
/// Butteraugli score and optional per-pixel difference map.
///
/// # Errors
/// Returns an error if:
/// - Image dimensions don't match
/// - Either dimension is zero
///
/// # Example
/// ```rust
/// use butteraugli::{butteraugli, ButteraugliParams, Img, RGB8};
///
/// let width = 16;
/// let height = 16;
/// let pixels: Vec<RGB8> = vec![RGB8::new(128, 128, 128); width * height];
/// let img = Img::new(pixels, width, height);
///
/// let result = butteraugli(img.as_ref(), img.as_ref(), &ButteraugliParams::default())?;
/// println!("Score: {}", result.score);
/// # Ok::<(), butteraugli::ButteraugliError>(())
/// ```
pub fn butteraugli(
    img1: ImgRef<RGB8>,
    img2: ImgRef<RGB8>,
    params: &ButteraugliParams,
) -> Result<ButteraugliResult, ButteraugliError> {
    butteraugli_with_stop(img1, img2, params, &enough::Unstoppable)
}

/// Like [`butteraugli`], but cooperatively cancellable via an
/// [`enough::Stop`] token.
///
/// `stop` is checked once at the outermost per-scale boundary of the core
/// compute (before any of the multi-scale processing blocks run); the
/// per-pixel kernels are never interrupted mid-scale. If the token signals a
/// stop, returns [`ButteraugliError::Cancelled`] carrying the
/// [`enough::StopReason`].
///
/// Pass [`enough::Unstoppable`] for a non-cancellable call (this is exactly
/// what [`butteraugli`] does — it is zero-cost).
///
/// # Arguments
/// * `img1` - First image (sRGB, supports stride via ImgRef)
/// * `img2` - Second image (sRGB, supports stride via ImgRef)
/// * `params` - Comparison parameters
/// * `stop` - Cooperative-cancellation token
///
/// # Errors
/// Returns an error if:
/// - Image dimensions don't match
/// - Either dimension is zero
/// - `stop` signals cancellation ([`ButteraugliError::Cancelled`])
pub fn butteraugli_with_stop(
    img1: ImgRef<RGB8>,
    img2: ImgRef<RGB8>,
    params: &ButteraugliParams,
    stop: &dyn enough::Stop,
) -> Result<ButteraugliResult, ButteraugliError> {
    params.validate()?;

    let (w1, h1) = (img1.width(), img1.height());
    let (w2, h2) = (img2.width(), img2.height());

    // Empty images can't be scored or reflect-padded.
    if w1 == 0 || h1 == 0 {
        return Err(ButteraugliError::ImageTooSmall {
            width: w1,
            height: h1,
        });
    }

    if w1 != w2 || h1 != h2 {
        return Err(ButteraugliError::DimensionMismatch { w1, h1, w2, h2 });
    }

    // Reject dimensions that would overflow `width * height * 3` (the
    // interleaved RGB element count). On 32-bit targets this is reachable with
    // benign-looking sizes; on 64-bit it can be reached when `imgref::Img` was
    // built from a stride that truncated. Matches the check in
    // `ButteraugliReference::new`.
    w1.checked_mul(h1).and_then(|wh| wh.checked_mul(3)).ok_or(
        ButteraugliError::DimensionOverflow {
            width: w1,
            height: h1,
        },
    )?;

    // Reflect(mirror)-pad sub-8px inputs up to the 8×8 floor so a small
    // image scores down to 1×1 instead of erroring. NO-OP at ≥8px
    // (scores the borrows directly); the diffmap is cropped back to the
    // logical extent below.
    let pad1 = reflect_pad_to_min(img1, 8);
    let pad2 = reflect_pad_to_min(img2, 8);
    let (i1, i2) = match (pad1.as_ref(), pad2.as_ref()) {
        (Some(a), Some(b)) => (a.as_ref(), b.as_ref()),
        _ => (img1, img2),
    };

    let result = diff::compute_butteraugli_imgref(i1, i2, params, params.compute_diffmap, stop)?;

    if !result.score.is_finite() {
        return Err(ButteraugliError::NonFiniteResult);
    }

    Ok(ButteraugliResult {
        score: result.score,
        pnorm_3: result.pnorm_3,
        diffmap: result
            .diffmap
            .map(image::ImageF::into_imgvec)
            .map(|dm| crop_diffmap(dm, w1, h1)),
    })
}

/// Computes butteraugli score between two linear RGB images.
///
/// This function accepts linear RGB float data via `ImgRef<RGB<f32>>`.
/// Use this for HDR content or when you already have linear RGB data.
///
/// **Scaling contract:** linear `1.0` maps to
/// [`ButteraugliParams::intensity_target`] nits (default `80.0` = SDR).
/// Values above `1.0` are accepted and map proportionally above
/// `intensity_target`; negatives are clamped to `0.0` in the opsin stage;
/// NaN/Inf pixels are rejected with [`ButteraugliError::NonFiniteResult`].
/// For HDR, scale your data so `1.0` is the mastering/display peak and set
/// `intensity_target` to that peak in nits.
///
/// Images smaller than 8×8 (down to 1×1) are reflect(mirror)-padded up to
/// the 8×8 floor and scored; the returned diffmap is cropped back to the
/// input dimensions.
///
/// # Arguments
/// * `img1` - First image (linear RGB, `1.0` = `intensity_target` nits)
/// * `img2` - Second image (linear RGB, `1.0` = `intensity_target` nits)
/// * `params` - Comparison parameters
///
/// # Returns
/// Butteraugli score and optional per-pixel difference map.
///
/// # Errors
/// Returns an error if:
/// - Image dimensions don't match
/// - Either dimension is zero
/// - Any input pixel is NaN or infinite
pub fn butteraugli_linear(
    img1: ImgRef<RGB<f32>>,
    img2: ImgRef<RGB<f32>>,
    params: &ButteraugliParams,
) -> Result<ButteraugliResult, ButteraugliError> {
    butteraugli_linear_with_stop(img1, img2, params, &enough::Unstoppable)
}

/// Like [`butteraugli_linear`], but cooperatively cancellable via an
/// [`enough::Stop`] token.
///
/// `stop` is checked once at the outermost per-scale boundary of the core
/// compute (before any of the multi-scale processing blocks run); the
/// per-pixel kernels are never interrupted mid-scale. If the token signals a
/// stop, returns [`ButteraugliError::Cancelled`] carrying the
/// [`enough::StopReason`].
///
/// Pass [`enough::Unstoppable`] for a non-cancellable call (this is exactly
/// what [`butteraugli_linear`] does — it is zero-cost).
///
/// # Arguments
/// * `img1` - First image (linear RGB, `1.0` = `intensity_target` nits)
/// * `img2` - Second image (linear RGB, `1.0` = `intensity_target` nits)
/// * `params` - Comparison parameters
/// * `stop` - Cooperative-cancellation token
///
/// # Errors
/// Returns an error if:
/// - Image dimensions don't match
/// - Either dimension is zero
/// - Any input pixel is NaN or infinite
/// - `stop` signals cancellation ([`ButteraugliError::Cancelled`])
pub fn butteraugli_linear_with_stop(
    img1: ImgRef<RGB<f32>>,
    img2: ImgRef<RGB<f32>>,
    params: &ButteraugliParams,
    stop: &dyn enough::Stop,
) -> Result<ButteraugliResult, ButteraugliError> {
    params.validate()?;

    let (w1, h1) = (img1.width(), img1.height());
    let (w2, h2) = (img2.width(), img2.height());

    // Empty images can't be scored or reflect-padded.
    if w1 == 0 || h1 == 0 {
        return Err(ButteraugliError::ImageTooSmall {
            width: w1,
            height: h1,
        });
    }

    if w1 != w2 || h1 != h2 {
        return Err(ButteraugliError::DimensionMismatch { w1, h1, w2, h2 });
    }

    // Reject dimensions that would overflow `width * height * 3`. See the
    // matching check in `butteraugli` above.
    w1.checked_mul(h1).and_then(|wh| wh.checked_mul(3)).ok_or(
        ButteraugliError::DimensionOverflow {
            width: w1,
            height: h1,
        },
    )?;

    check_finite_rgb_imgref(img1)?;
    check_finite_rgb_imgref(img2)?;

    // Reflect(mirror)-pad sub-8px inputs up to the 8×8 floor (see
    // `butteraugli`). NO-OP at ≥8px.
    let pad1 = reflect_pad_to_min(img1, 8);
    let pad2 = reflect_pad_to_min(img2, 8);
    let (i1, i2) = match (pad1.as_ref(), pad2.as_ref()) {
        (Some(a), Some(b)) => (a.as_ref(), b.as_ref()),
        _ => (img1, img2),
    };

    let result =
        diff::compute_butteraugli_linear_imgref(i1, i2, params, params.compute_diffmap, stop)?;

    if !result.score.is_finite() {
        return Err(ButteraugliError::NonFiniteResult);
    }

    Ok(ButteraugliResult {
        score: result.score,
        pnorm_3: result.pnorm_3,
        diffmap: result
            .diffmap
            .map(image::ImageF::into_imgvec)
            .map(|dm| crop_diffmap(dm, w1, h1)),
    })
}

/// Converts sRGB u8 value to linear RGB f32.
///
/// Apply this to each channel when converting sRGB images to linear RGB
/// for use with [`butteraugli_linear`].
#[must_use]
pub fn srgb_to_linear(v: u8) -> f32 {
    opsin::srgb_to_linear(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_images() {
        let width = 16;
        let height = 16;
        let pixels: Vec<RGB8> = (0..width * height)
            .map(|i| {
                RGB8::new(
                    (i % 256) as u8,
                    ((i * 2) % 256) as u8,
                    ((i * 3) % 256) as u8,
                )
            })
            .collect();
        let img = Img::new(pixels, width, height);

        let result = butteraugli(img.as_ref(), img.as_ref(), &ButteraugliParams::default())
            .expect("valid input");

        // Identical images should have score 0
        assert!(
            result.score < 0.001,
            "Identical images should have score ~0, got {}",
            result.score
        );
    }

    #[test]
    fn test_different_images() {
        let width = 16;
        let height = 16;
        let pixels1: Vec<RGB8> = vec![RGB8::new(0, 0, 0); width * height];
        let pixels2: Vec<RGB8> = vec![RGB8::new(255, 255, 255); width * height];
        let img1 = Img::new(pixels1, width, height);
        let img2 = Img::new(pixels2, width, height);

        let result = butteraugli(img1.as_ref(), img2.as_ref(), &ButteraugliParams::default())
            .expect("valid input");

        // Different images should have non-zero score
        assert!(
            result.score > 0.01,
            "Different images should have non-zero score, got {}",
            result.score
        );
    }

    #[test]
    fn test_dimension_mismatch() {
        let pixels1: Vec<RGB8> = vec![RGB8::new(0, 0, 0); 16 * 16];
        let pixels2: Vec<RGB8> = vec![RGB8::new(0, 0, 0); 8 * 8];
        let img1 = Img::new(pixels1, 16, 16);
        let img2 = Img::new(pixels2, 8, 8);

        let result = butteraugli(img1.as_ref(), img2.as_ref(), &ButteraugliParams::default());
        assert!(matches!(
            result,
            Err(ButteraugliError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_sub_8_reflect_pads_instead_of_rejecting() {
        // Sub-8px images are reflect(mirror)-padded up to the 8×8 floor
        // and scored (down to 1×1) rather than rejected. An identical
        // pair must still score ~0 like any identical pair.
        let pixels: Vec<RGB8> = (0..4 * 4)
            .map(|i| RGB8::new((i * 7) as u8, (i * 13) as u8, (i * 29) as u8))
            .collect();
        let img = Img::new(pixels, 4, 4);
        let same = butteraugli(img.as_ref(), img.as_ref(), &ButteraugliParams::default())
            .expect("sub-8 image must score (reflect-padded), not reject");
        assert!(
            same.score < 0.001,
            "identical 4×4 must score ~0, got {}",
            same.score
        );

        // 1×1 also scores instead of erroring.
        let one = Img::new(vec![RGB8::new(10, 20, 30)], 1, 1);
        let one_score = butteraugli(one.as_ref(), one.as_ref(), &ButteraugliParams::default())
            .expect("1×1 image must score")
            .score;
        assert!(
            one_score < 0.001,
            "identical 1×1 must score ~0, got {one_score}"
        );

        // A real sub-8 difference produces a finite, non-trivial score.
        let a = Img::new(vec![RGB8::new(0, 0, 0); 5 * 5], 5, 5);
        let b = Img::new(vec![RGB8::new(255, 255, 255); 5 * 5], 5, 5);
        let diff = butteraugli(a.as_ref(), b.as_ref(), &ButteraugliParams::default())
            .expect("5×5 differing pair must score")
            .score;
        assert!(
            diff.is_finite() && diff > 0.0,
            "5×5 black/white score {diff}"
        );
    }

    #[test]
    fn test_compute_diffmap_flag() {
        let width = 16;
        let height = 16;
        let pixels: Vec<RGB8> = vec![RGB8::new(128, 128, 128); width * height];
        let img = Img::new(pixels, width, height);

        // Without diffmap
        let params = ButteraugliParams::default();
        let result = butteraugli(img.as_ref(), img.as_ref(), &params).unwrap();
        assert!(result.diffmap.is_none());

        // With diffmap
        let params = ButteraugliParams::default().with_compute_diffmap(true);
        let result = butteraugli(img.as_ref(), img.as_ref(), &params).unwrap();
        assert!(result.diffmap.is_some());
        let diffmap = result.diffmap.unwrap();
        assert_eq!(diffmap.width(), width);
        assert_eq!(diffmap.height(), height);
    }

    // ================================================================
    // Parameter validation tests
    // ================================================================

    fn make_test_images() -> (ImgVec<RGB8>, ImgVec<RGB8>) {
        let width = 16;
        let height = 16;
        let pixels: Vec<RGB8> = vec![RGB8::new(128, 128, 128); width * height];
        let img = Img::new(pixels, width, height);
        (img.clone(), img)
    }

    #[test]
    fn test_zero_hf_asymmetry_returns_error() {
        let (img1, img2) = make_test_images();
        let params = ButteraugliParams::new().with_hf_asymmetry(0.0);
        let result = butteraugli(img1.as_ref(), img2.as_ref(), &params);
        assert!(
            matches!(
                result,
                Err(ButteraugliError::InvalidParameter {
                    name: "hf_asymmetry",
                    ..
                })
            ),
            "expected InvalidParameter for hf_asymmetry=0.0, got {result:?}"
        );
    }

    #[test]
    fn test_negative_hf_asymmetry_returns_error() {
        let (img1, img2) = make_test_images();
        let params = ButteraugliParams::new().with_hf_asymmetry(-1.0);
        let result = butteraugli(img1.as_ref(), img2.as_ref(), &params);
        assert!(
            matches!(
                result,
                Err(ButteraugliError::InvalidParameter {
                    name: "hf_asymmetry",
                    ..
                })
            ),
            "expected InvalidParameter for hf_asymmetry=-1.0, got {result:?}"
        );
    }

    #[test]
    fn test_zero_intensity_target_returns_error() {
        let (img1, img2) = make_test_images();
        let params = ButteraugliParams::new().with_intensity_target(0.0);
        let result = butteraugli(img1.as_ref(), img2.as_ref(), &params);
        assert!(
            matches!(
                result,
                Err(ButteraugliError::InvalidParameter {
                    name: "intensity_target",
                    ..
                })
            ),
            "expected InvalidParameter for intensity_target=0.0, got {result:?}"
        );
    }

    #[test]
    fn test_nan_xmul_returns_error() {
        let (img1, img2) = make_test_images();
        let params = ButteraugliParams::new().with_xmul(f32::NAN);
        let result = butteraugli(img1.as_ref(), img2.as_ref(), &params);
        assert!(
            matches!(
                result,
                Err(ButteraugliError::InvalidParameter { name: "xmul", .. })
            ),
            "expected InvalidParameter for xmul=NaN, got {result:?}"
        );
    }

    #[test]
    fn test_inf_hf_asymmetry_returns_error() {
        let (img1, img2) = make_test_images();
        let params = ButteraugliParams::new().with_hf_asymmetry(f32::INFINITY);
        let result = butteraugli(img1.as_ref(), img2.as_ref(), &params);
        assert!(
            matches!(
                result,
                Err(ButteraugliError::InvalidParameter {
                    name: "hf_asymmetry",
                    ..
                })
            ),
            "expected InvalidParameter for hf_asymmetry=Inf, got {result:?}"
        );
    }

    #[test]
    fn test_negative_xmul_returns_error() {
        let (img1, img2) = make_test_images();
        let params = ButteraugliParams::new().with_xmul(-0.5);
        let result = butteraugli(img1.as_ref(), img2.as_ref(), &params);
        assert!(
            matches!(
                result,
                Err(ButteraugliError::InvalidParameter { name: "xmul", .. })
            ),
            "expected InvalidParameter for xmul=-0.5, got {result:?}"
        );
    }

    #[test]
    fn test_zero_xmul_is_valid() {
        // xmul=0 is allowed (silences X channel, not a bug)
        let (img1, img2) = make_test_images();
        let params = ButteraugliParams::new().with_xmul(0.0);
        let result = butteraugli(img1.as_ref(), img2.as_ref(), &params);
        assert!(result.is_ok(), "xmul=0.0 should be valid, got {result:?}");
    }

    #[test]
    fn test_nan_pixels_returns_non_finite_result() {
        let width = 16;
        let height = 16;
        // All-NaN pixels propagate through the entire pipeline to produce NaN score.
        // A single NaN pixel may get absorbed by averaging/clamping in the algorithm,
        // but a fully-NaN image guarantees NaN propagation.
        let pixels1: Vec<RGB<f32>> = vec![RGB::new(f32::NAN, f32::NAN, f32::NAN); width * height];
        let pixels2: Vec<RGB<f32>> = vec![RGB::new(0.5, 0.5, 0.5); width * height];

        let img1 = Img::new(pixels1, width, height);
        let img2 = Img::new(pixels2, width, height);

        let result =
            butteraugli_linear(img1.as_ref(), img2.as_ref(), &ButteraugliParams::default());
        assert!(
            matches!(result, Err(ButteraugliError::NonFiniteResult)),
            "expected NonFiniteResult for NaN input pixels, got {result:?}"
        );
    }

    #[test]
    fn test_inf_pixels_returns_non_finite_result() {
        let width = 16;
        let height = 16;
        let pixels1: Vec<RGB<f32>> =
            vec![RGB::new(f32::INFINITY, f32::INFINITY, f32::INFINITY); width * height];
        let pixels2: Vec<RGB<f32>> = vec![RGB::new(0.5, 0.5, 0.5); width * height];

        let img1 = Img::new(pixels1, width, height);
        let img2 = Img::new(pixels2, width, height);

        let result =
            butteraugli_linear(img1.as_ref(), img2.as_ref(), &ButteraugliParams::default());
        assert!(
            matches!(result, Err(ButteraugliError::NonFiniteResult)),
            "expected NonFiniteResult for Inf input pixels, got {result:?}"
        );
    }

    #[test]
    fn test_default_params_still_valid() {
        assert!(ButteraugliParams::default().validate().is_ok());
    }

    #[test]
    fn test_validate_method_directly() {
        // Valid params
        assert!(
            ButteraugliParams::new()
                .with_hf_asymmetry(1.5)
                .with_intensity_target(250.0)
                .with_xmul(0.5)
                .validate()
                .is_ok()
        );

        // Each invalid param
        assert!(
            ButteraugliParams::new()
                .with_hf_asymmetry(0.0)
                .validate()
                .is_err()
        );
        assert!(
            ButteraugliParams::new()
                .with_hf_asymmetry(-1.0)
                .validate()
                .is_err()
        );
        assert!(
            ButteraugliParams::new()
                .with_hf_asymmetry(f32::NAN)
                .validate()
                .is_err()
        );
        assert!(
            ButteraugliParams::new()
                .with_hf_asymmetry(f32::INFINITY)
                .validate()
                .is_err()
        );
        assert!(
            ButteraugliParams::new()
                .with_intensity_target(0.0)
                .validate()
                .is_err()
        );
        assert!(
            ButteraugliParams::new()
                .with_intensity_target(-10.0)
                .validate()
                .is_err()
        );
        assert!(ButteraugliParams::new().with_xmul(-0.1).validate().is_err());
        assert!(
            ButteraugliParams::new()
                .with_xmul(f32::NAN)
                .validate()
                .is_err()
        );

        // xmul=0 is valid
        assert!(ButteraugliParams::new().with_xmul(0.0).validate().is_ok());
    }

    #[test]
    fn test_validation_on_linear_api() {
        let width = 16;
        let height = 16;
        let pixels: Vec<RGB<f32>> = vec![RGB::new(0.5, 0.5, 0.5); width * height];
        let img = Img::new(pixels, width, height);

        let params = ButteraugliParams::new().with_hf_asymmetry(0.0);
        let result = butteraugli_linear(img.as_ref(), img.as_ref(), &params);
        assert!(matches!(
            result,
            Err(ButteraugliError::InvalidParameter {
                name: "hf_asymmetry",
                ..
            })
        ));
    }

    #[test]
    fn test_validation_on_precompute_api() {
        let width = 32;
        let height = 32;
        let rgb: Vec<u8> = vec![128; width * height * 3];

        let params = ButteraugliParams::new().with_intensity_target(0.0);
        let result = ButteraugliReference::new(&rgb, width, height, params);
        assert!(matches!(
            result,
            Err(ButteraugliError::InvalidParameter {
                name: "intensity_target",
                ..
            })
        ));
    }

    #[test]
    fn test_validation_on_precompute_linear_api() {
        let width = 32;
        let height = 32;
        let rgb: Vec<f32> = vec![0.5; width * height * 3];

        let params = ButteraugliParams::new().with_hf_asymmetry(-1.0);
        let result = ButteraugliReference::new_linear(&rgb, width, height, params);
        assert!(matches!(
            result,
            Err(ButteraugliError::InvalidParameter {
                name: "hf_asymmetry",
                ..
            })
        ));
    }

    /// `compare_linear_planar` previously used unchecked `stride * self.height`
    /// while its sibling `new_linear_planar` used `checked_mul`. An adversarial
    /// stride can panic on the multiplication on 32-bit targets before the
    /// buffer length check fires. The fix returns `DimensionOverflow` instead.
    #[test]
    fn test_compare_linear_planar_rejects_stride_overflow() {
        let width = 16;
        let height = 16;
        let channel: Vec<f32> = vec![0.5; width * height];
        let reference = ButteraugliReference::new_linear_planar(
            &channel,
            &channel,
            &channel,
            width,
            height,
            width,
            ButteraugliParams::default(),
        )
        .expect("valid reference");

        // `stride * self.height` overflows usize: usize::MAX / height + 1.
        let bad_stride = usize::MAX / height + 1;
        let dummy: Vec<f32> = vec![0.0; 1];
        let err = reference
            .compare_linear_planar(&dummy, &dummy, &dummy, bad_stride)
            .expect_err("stride overflow must be rejected");
        assert!(
            matches!(err, ButteraugliError::DimensionOverflow { .. }),
            "expected DimensionOverflow, got {err:?}",
        );
    }

    #[test]
    fn test_validation_on_precompute_planar_api() {
        let width = 32;
        let height = 32;
        let channel: Vec<f32> = vec![0.5; width * height];

        let params = ButteraugliParams::new().with_xmul(f32::NAN);
        let result = ButteraugliReference::new_linear_planar(
            &channel, &channel, &channel, width, height, width, params,
        );
        assert!(matches!(
            result,
            Err(ButteraugliError::InvalidParameter { name: "xmul", .. })
        ));
    }

    // ================================================================
    // p-norm aggregation tests (libjxl ComputeDistanceP parity)
    // ================================================================

    #[test]
    fn test_pnorm_slice_empty_returns_nan() {
        assert!(pnorm_slice(&[], 3.0).is_nan());
    }

    #[test]
    fn test_pnorm_slice_uniform_returns_value() {
        // A uniform diffmap of value v should give pnorm == v for any p.
        for &v in &[0.5_f32, 1.0, 2.5, 7.3] {
            let dm = vec![v; 64];
            for &p in &[1.5_f64, 2.0, 3.0, 4.0] {
                let got = pnorm_slice(&dm, p);
                assert!(
                    (got - v as f64).abs() < 1e-6,
                    "uniform pnorm_slice(v={v}, p={p}) = {got}, want {v}"
                );
            }
        }
    }

    #[test]
    fn test_pnorm_slice_matches_libjxl_reference_formula() {
        // Mirrors lib/extras/metrics.cc:ComputeDistanceP (slow-path branch).
        let dm: Vec<f32> = (1..=20).map(|i| (i as f32) * 0.13).collect();
        let p = 3.0_f64;

        let mut sum1 = [0.0_f64; 3];
        for &v in &dm {
            let mut d2 = (v as f64).powf(p);
            sum1[0] += d2;
            d2 *= d2;
            sum1[1] += d2;
            d2 *= d2;
            sum1[2] += d2;
        }
        let one_per_pixels = 1.0 / dm.len() as f64;
        let mut want = 0.0_f64;
        for i in 0..3 {
            want += (one_per_pixels * sum1[i]).powf(1.0 / (p * f64::from(1u32 << i)));
        }
        want /= 3.0;

        let got = pnorm_slice(&dm, p);
        let rel_err = (got - want).abs() / want;
        assert!(
            rel_err < 1e-12,
            "pnorm_slice {got} vs reference formula {want}, rel_err {rel_err}"
        );
    }

    #[test]
    fn test_pnorm_method_returns_precomputed_for_p3_without_diffmap() {
        let res = ButteraugliResult {
            score: 0.5,
            pnorm_3: 0.42,
            diffmap: None,
        };
        // p=3 is precomputed and always available, even without diffmap.
        assert_eq!(res.pnorm(3.0), Some(0.42));
        // Other p values still require the diffmap.
        assert!(res.pnorm(2.5).is_none());
        assert!(res.pnorm(4.0).is_none());
    }

    #[test]
    fn test_pnorm_method_uses_diffmap_for_arbitrary_p() {
        let dm = ImgVec::new(vec![2.0_f32; 16 * 16], 16, 16);
        let res = ButteraugliResult {
            score: 2.0,
            pnorm_3: 99.0, // sentinel — should NOT be returned for p=4
            diffmap: Some(dm),
        };
        // p ≠ 3: must be computed from diffmap, not the sentinel.
        let got = res.pnorm(4.0).expect("diffmap is present");
        assert!((got - 2.0).abs() < 1e-9, "got {got}");
        // p = 3: must return the precomputed sentinel, not recompute.
        assert_eq!(res.pnorm(3.0), Some(99.0));
    }

    #[test]
    fn test_max_norm_alias() {
        let res = ButteraugliResult {
            score: 1.234,
            pnorm_3: 0.0,
            diffmap: None,
        };
        // Bit-exact equality: max_norm() is just a getter, must round-trip.
        assert_eq!(res.max_norm().to_bits(), 1.234_f64.to_bits());
    }

    #[test]
    fn test_pnorm_3_field_matches_slice_helper_on_diffmap() {
        // The precomputed pnorm_3 field must agree with reducing the returned
        // diffmap directly, otherwise the fused reduction has drifted.
        let width = 32;
        let height = 32;
        let pixels1: Vec<RGB8> = (0..width * height)
            .map(|i| RGB8::new((i % 256) as u8, ((i * 3) % 256) as u8, 0))
            .collect();
        let pixels2: Vec<RGB8> = (0..width * height)
            .map(|i| RGB8::new(((i + 50) % 256) as u8, ((i * 3 + 30) % 256) as u8, 0))
            .collect();
        let img1 = Img::new(pixels1, width, height);
        let img2 = Img::new(pixels2, width, height);

        let params = ButteraugliParams::default().with_compute_diffmap(true);
        let result = butteraugli(img1.as_ref(), img2.as_ref(), &params).expect("valid");

        let dm = result.diffmap.as_ref().expect("diffmap requested");
        let recomputed = pnorm_slice(dm.buf(), 3.0);

        let rel_err = (result.pnorm_3 - recomputed).abs() / recomputed.max(1e-12);
        assert!(
            rel_err < 1e-9,
            "fused pnorm_3 = {} vs slice helper pnorm(diffmap, 3) = {}, rel_err {rel_err}",
            result.pnorm_3,
            recomputed
        );
    }

    #[test]
    fn test_pnorm_3_available_without_diffmap() {
        // The whole point of fused reduction: pnorm_3 is populated even when
        // compute_diffmap=false (no allocation cost for the result).
        let width = 32;
        let height = 32;
        let pixels1: Vec<RGB8> = vec![RGB8::new(40, 40, 40); width * height];
        let pixels2: Vec<RGB8> = vec![RGB8::new(80, 80, 80); width * height];
        let img1 = Img::new(pixels1, width, height);
        let img2 = Img::new(pixels2, width, height);

        let params = ButteraugliParams::default(); // compute_diffmap = false
        let result = butteraugli(img1.as_ref(), img2.as_ref(), &params).expect("valid");

        assert!(result.diffmap.is_none(), "diffmap should not be returned");
        assert!(result.pnorm_3.is_finite(), "pnorm_3 must be populated");
        assert!(result.pnorm_3 >= 0.0);
        // pnorm method also works (precomputed path).
        assert_eq!(result.pnorm(3.0), Some(result.pnorm_3));
    }

    #[test]
    fn test_pnorm_3_zero_for_identical_images() {
        let width = 16;
        let height = 16;
        let pixels: Vec<RGB8> = vec![RGB8::new(128, 128, 128); width * height];
        let img = Img::new(pixels, width, height);

        let params = ButteraugliParams::default();
        let result = butteraugli(img.as_ref(), img.as_ref(), &params).expect("valid");
        assert!(result.pnorm_3.abs() < 1e-9);
        assert!(result.score < 0.001);
    }

    #[test]
    fn test_error_display() {
        let err = ButteraugliError::InvalidParameter {
            name: "hf_asymmetry",
            value: 0.0,
            reason: "must be finite and positive",
        };
        assert_eq!(
            err.to_string(),
            "invalid parameter hf_asymmetry=0: must be finite and positive"
        );

        let err = ButteraugliError::DimensionOverflow {
            width: 1000000,
            height: 1000000,
        };
        assert!(err.to_string().contains("overflow"));

        let err = ButteraugliError::NonFiniteResult;
        assert!(err.to_string().contains("NaN"));
    }

    /// Adversarial dimensions where `width * height * 3` overflows `usize`.
    ///
    /// Reachable on 32-bit targets with benign-looking sizes, and on 64-bit
    /// when an `imgref::Img` is constructed via `new_stride` (which stores
    /// width/height internally as `u32` but exposes them as `usize`, so a
    /// caller-supplied `u32::MAX - 1` survives construction).
    ///
    /// The standalone `butteraugli` and `butteraugli_linear` entry points
    /// must reject these with `DimensionOverflow` instead of panicking on
    /// the multiplication or down-stream allocation.
    #[test]
    fn test_butteraugli_rejects_dimension_overflow() {
        // u32::MAX-1 squared * 3 overflows usize on 64-bit. The buffer is
        // never read because the overflow check fires first; we just need
        // an `ImgRef` with these dimensions.
        let huge: usize = (u32::MAX - 1) as usize;
        let buf: Vec<RGB8> = Vec::new();
        // SAFETY: imgref's `new_stride` does not access the buffer, only
        // stores it. The overflow check in `butteraugli` returns Err before
        // any iteration. We use stride==width to satisfy `stride >= width`.
        let img = imgref::ImgRef::new_stride(&buf[..], huge, huge, huge);
        let err = butteraugli(img, img, &ButteraugliParams::default())
            .expect_err("dimension overflow must be rejected");
        assert!(
            matches!(err, ButteraugliError::DimensionOverflow { .. }),
            "expected DimensionOverflow, got {err:?}",
        );
    }

    #[test]
    fn test_butteraugli_linear_rejects_dimension_overflow() {
        let huge: usize = (u32::MAX - 1) as usize;
        let buf: Vec<RGB<f32>> = Vec::new();
        let img = imgref::ImgRef::new_stride(&buf[..], huge, huge, huge);
        let err = butteraugli_linear(img, img, &ButteraugliParams::default())
            .expect_err("dimension overflow must be rejected");
        assert!(
            matches!(err, ButteraugliError::DimensionOverflow { .. }),
            "expected DimensionOverflow, got {err:?}",
        );
    }
}
