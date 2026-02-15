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
//! // Create two 8x8 RGB images (must be 8x8 minimum)
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
//! ## Features
//!
//! - **`cli`**: Command-line tool (adds clap, image, serde_json)
//! - **`internals`**: Expose internal modules for testing/benchmarking (unstable API)
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

// Internal modules - exposed with "internals" feature for testing/benchmarking
#[cfg(feature = "internals")]
pub mod blur;
#[cfg(not(feature = "internals"))]
pub(crate) mod blur;

#[cfg(feature = "internals")]
pub mod consts;
#[cfg(not(feature = "internals"))]
pub(crate) mod consts;

mod diff;

#[cfg(feature = "internals")]
pub mod image;
#[cfg(not(feature = "internals"))]
pub(crate) mod image;

pub(crate) mod image_aligned;

#[cfg(feature = "internals")]
pub mod malta;
#[cfg(not(feature = "internals"))]
pub(crate) mod malta;

#[cfg(feature = "internals")]
pub mod mask;
#[cfg(not(feature = "internals"))]
pub(crate) mod mask;

#[cfg(feature = "internals")]
pub mod opsin;
#[cfg(not(feature = "internals"))]
pub(crate) mod opsin;

pub mod precompute;
// Re-export ButteraugliReference for convenience
pub use precompute::ButteraugliReference;

#[cfg(feature = "internals")]
pub mod psycho;
#[cfg(not(feature = "internals"))]
pub(crate) mod psycho;

pub(crate) mod xyb;

// C++ reference data for regression testing (auto-generated)
// Hidden from docs as this is internal test infrastructure
#[doc(hidden)]
pub mod reference_data;

// Re-export imgref and rgb types for convenience
pub use imgref::{Img, ImgRef, ImgVec};
pub use rgb::{RGB, RGB8};

/// Error type for butteraugli operations.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ButteraugliError {
    /// Image is too small (minimum 8x8).
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
}

impl std::fmt::Display for ButteraugliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ImageTooSmall { width, height } => {
                write!(f, "image too small: {width}x{height} (minimum 8x8)")
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

/// Controls which Malta patterns 13-16 are used in the HF/UHF bands.
///
/// In libjxl's butteraugli, patterns 13-16 are identical copies of patterns 8,7,6,5
/// (9-sample straight lines). The standalone google/butteraugli has different patterns
/// 13-16: 8-sample S-curves. See <https://github.com/libjxl/libjxl/issues/4623>.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MaltaVariant {
    /// Patterns 13-16 are copies of 8,7,6,5 (9 samples each).
    /// This matches libjxl's butteraugli implementation.
    #[default]
    Libjxl,
    /// Patterns 13-16 are original S-curves (8 samples each).
    /// This matches the standalone google/butteraugli implementation.
    StandaloneGoogle,
}

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
    malta_variant: MaltaVariant,
}

impl Default for ButteraugliParams {
    fn default() -> Self {
        Self {
            hf_asymmetry: 1.0,
            xmul: 1.0,
            intensity_target: 80.0,
            compute_diffmap: false,
            single_resolution: false,
            malta_variant: MaltaVariant::default(),
        }
    }
}

impl ButteraugliParams {
    /// Creates a new `ButteraugliParams` with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the intensity target (display brightness in nits).
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

    /// Sets which Malta pattern variant to use for HF/UHF bands.
    ///
    /// See [`MaltaVariant`] for details on the difference between libjxl
    /// and standalone google/butteraugli patterns 13-16.
    #[must_use]
    pub fn with_malta_variant(mut self, malta_variant: MaltaVariant) -> Self {
        self.malta_variant = malta_variant;
        self
    }

    /// Returns the Malta pattern variant.
    #[must_use]
    pub fn malta_variant(&self) -> MaltaVariant {
        self.malta_variant
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

/// Butteraugli image comparison result.
#[derive(Debug, Clone)]
pub struct ButteraugliResult {
    /// Global difference score. < 1.0 is "good", > 2.0 is "bad".
    pub score: f64,
    /// Per-pixel difference map (only present if `compute_diffmap` was true).
    pub diffmap: Option<ImgVec<f32>>,
}

/// Computes butteraugli score between two sRGB images.
///
/// This function accepts 8-bit sRGB data via `ImgRef<RGB8>` and internally converts
/// to linear RGB for perceptual comparison.
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
/// - Images are smaller than 8x8 pixels
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
    params.validate()?;

    let (w1, h1) = (img1.width(), img1.height());
    let (w2, h2) = (img2.width(), img2.height());

    if w1 < 8 || h1 < 8 {
        return Err(ButteraugliError::ImageTooSmall {
            width: w1,
            height: h1,
        });
    }

    if w1 != w2 || h1 != h2 {
        return Err(ButteraugliError::DimensionMismatch { w1, h1, w2, h2 });
    }

    let result = diff::compute_butteraugli_imgref(img1, img2, params, params.compute_diffmap);

    if !result.score.is_finite() {
        return Err(ButteraugliError::NonFiniteResult);
    }

    Ok(ButteraugliResult {
        score: result.score,
        diffmap: result.diffmap.map(image::ImageF::into_imgvec),
    })
}

/// Computes butteraugli score between two linear RGB images.
///
/// This function accepts linear RGB float data via `ImgRef<RGB<f32>>`.
/// Use this for HDR content or when you already have linear RGB data.
///
/// # Arguments
/// * `img1` - First image (linear RGB, values in 0.0-1.0 range)
/// * `img2` - Second image (linear RGB, values in 0.0-1.0 range)
/// * `params` - Comparison parameters
///
/// # Returns
/// Butteraugli score and optional per-pixel difference map.
///
/// # Errors
/// Returns an error if:
/// - Image dimensions don't match
/// - Images are smaller than 8x8 pixels
pub fn butteraugli_linear(
    img1: ImgRef<RGB<f32>>,
    img2: ImgRef<RGB<f32>>,
    params: &ButteraugliParams,
) -> Result<ButteraugliResult, ButteraugliError> {
    params.validate()?;

    let (w1, h1) = (img1.width(), img1.height());
    let (w2, h2) = (img2.width(), img2.height());

    if w1 < 8 || h1 < 8 {
        return Err(ButteraugliError::ImageTooSmall {
            width: w1,
            height: h1,
        });
    }

    if w1 != w2 || h1 != h2 {
        return Err(ButteraugliError::DimensionMismatch { w1, h1, w2, h2 });
    }

    check_finite_rgb_imgref(img1)?;
    check_finite_rgb_imgref(img2)?;

    let result =
        diff::compute_butteraugli_linear_imgref(img1, img2, params, params.compute_diffmap);

    if !result.score.is_finite() {
        return Err(ButteraugliError::NonFiniteResult);
    }

    Ok(ButteraugliResult {
        score: result.score,
        diffmap: result.diffmap.map(image::ImageF::into_imgvec),
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

// ============================================================================
// Legacy API (deprecated, will be removed in future versions)
// ============================================================================

/// Legacy result type that uses internal ImageF.
///
/// This is kept for backward compatibility during the transition period.
#[doc(hidden)]
pub struct LegacyButteraugliResult {
    /// Global difference score.
    pub score: f64,
    /// Per-pixel difference map.
    pub diffmap: Option<image::ImageF>,
}

/// Legacy function for backward compatibility.
///
/// Use [`butteraugli`] instead.
#[deprecated(since = "0.4.0", note = "Use butteraugli() with ImgRef<RGB8> instead")]
#[allow(clippy::missing_errors_doc)]
pub fn compute_butteraugli(
    rgb1: &[u8],
    rgb2: &[u8],
    width: usize,
    height: usize,
    params: &ButteraugliParams,
) -> Result<LegacyButteraugliResult, ButteraugliError> {
    params.validate()?;

    let expected_size = width
        .checked_mul(height)
        .and_then(|wh| wh.checked_mul(3))
        .ok_or(ButteraugliError::DimensionOverflow { width, height })?;

    if width < 8 || height < 8 {
        return Err(ButteraugliError::ImageTooSmall { width, height });
    }

    if rgb1.len() != expected_size || rgb2.len() != expected_size {
        return Err(ButteraugliError::DimensionMismatch {
            w1: width,
            h1: height,
            w2: if rgb2.len() == expected_size {
                width
            } else {
                0
            },
            h2: if rgb2.len() == expected_size {
                height
            } else {
                0
            },
        });
    }

    // Convert u8 slices to RGB8 and create Img
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

    let result = diff::compute_butteraugli_imgref(img1.as_ref(), img2.as_ref(), params, true);

    if !result.score.is_finite() {
        return Err(ButteraugliError::NonFiniteResult);
    }

    Ok(LegacyButteraugliResult {
        score: result.score,
        diffmap: result.diffmap,
    })
}

/// Legacy function for backward compatibility (linear RGB).
///
/// Use [`butteraugli_linear`] instead.
#[deprecated(
    since = "0.4.0",
    note = "Use butteraugli_linear() with ImgRef<RGB<f32>> instead"
)]
#[allow(clippy::missing_errors_doc)]
pub fn compute_butteraugli_linear(
    rgb1: &[f32],
    rgb2: &[f32],
    width: usize,
    height: usize,
    params: &ButteraugliParams,
) -> Result<LegacyButteraugliResult, ButteraugliError> {
    params.validate()?;

    let expected_size = width
        .checked_mul(height)
        .and_then(|wh| wh.checked_mul(3))
        .ok_or(ButteraugliError::DimensionOverflow { width, height })?;

    if width < 8 || height < 8 {
        return Err(ButteraugliError::ImageTooSmall { width, height });
    }

    if rgb1.len() != expected_size || rgb2.len() != expected_size {
        return Err(ButteraugliError::DimensionMismatch {
            w1: width,
            h1: height,
            w2: if rgb2.len() == expected_size {
                width
            } else {
                0
            },
            h2: if rgb2.len() == expected_size {
                height
            } else {
                0
            },
        });
    }

    check_finite_f32(rgb1, "rgb1")?;
    check_finite_f32(rgb2, "rgb2")?;

    // Convert f32 slices to RGB<f32> and create Img
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

    let result =
        diff::compute_butteraugli_linear_imgref(img1.as_ref(), img2.as_ref(), params, true);

    if !result.score.is_finite() {
        return Err(ButteraugliError::NonFiniteResult);
    }

    Ok(LegacyButteraugliResult {
        score: result.score,
        diffmap: result.diffmap,
    })
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
    fn test_too_small_dimensions() {
        let pixels: Vec<RGB8> = vec![RGB8::new(0, 0, 0); 4 * 4];
        let img = Img::new(pixels, 4, 4);

        let result = butteraugli(img.as_ref(), img.as_ref(), &ButteraugliParams::default());
        assert!(matches!(
            result,
            Err(ButteraugliError::ImageTooSmall { .. })
        ));
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
    #[allow(deprecated)]
    fn test_overflow_dimensions_returns_error() {
        // width * height * 3 would overflow usize on 64-bit
        let width = usize::MAX / 2;
        let height = 3;
        let rgb: Vec<u8> = vec![0; 8]; // doesn't matter, overflow comes first
        let result = compute_butteraugli(&rgb, &rgb, width, height, &ButteraugliParams::default());
        assert!(
            matches!(result, Err(ButteraugliError::DimensionOverflow { .. })),
            "expected DimensionOverflow, got {:?}",
            result.as_ref().err()
        );
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
        assert!(ButteraugliParams::new()
            .with_hf_asymmetry(1.5)
            .with_intensity_target(250.0)
            .with_xmul(0.5)
            .validate()
            .is_ok());

        // Each invalid param
        assert!(ButteraugliParams::new()
            .with_hf_asymmetry(0.0)
            .validate()
            .is_err());
        assert!(ButteraugliParams::new()
            .with_hf_asymmetry(-1.0)
            .validate()
            .is_err());
        assert!(ButteraugliParams::new()
            .with_hf_asymmetry(f32::NAN)
            .validate()
            .is_err());
        assert!(ButteraugliParams::new()
            .with_hf_asymmetry(f32::INFINITY)
            .validate()
            .is_err());
        assert!(ButteraugliParams::new()
            .with_intensity_target(0.0)
            .validate()
            .is_err());
        assert!(ButteraugliParams::new()
            .with_intensity_target(-10.0)
            .validate()
            .is_err());
        assert!(ButteraugliParams::new().with_xmul(-0.1).validate().is_err());
        assert!(ButteraugliParams::new()
            .with_xmul(f32::NAN)
            .validate()
            .is_err());

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
}
