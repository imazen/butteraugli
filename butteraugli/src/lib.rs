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
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ButteraugliError {
    /// Image is too small (minimum 8x8).
    ImageTooSmall {
        /// Image width.
        width: usize,
        /// Image height.
        height: usize,
    },
    /// Image dimensions don't match.
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
    #[doc(hidden)]
    InvalidDimensions {
        /// Width provided.
        width: usize,
        /// Height provided.
        height: usize,
    },
    /// Buffer size doesn't match expected size (legacy variant).
    #[doc(hidden)]
    InvalidBufferSize {
        /// Expected buffer size.
        expected: usize,
        /// Actual buffer size.
        actual: usize,
    },
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
}

impl Default for ButteraugliParams {
    fn default() -> Self {
        Self {
            hf_asymmetry: 1.0,
            xmul: 1.0,
            intensity_target: 80.0,
            compute_diffmap: false,
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

    let result =
        diff::compute_butteraugli_linear_imgref(img1, img2, params, params.compute_diffmap);

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
    let expected_size = width * height * 3;

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
    let expected_size = width * height * 3;

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
}
