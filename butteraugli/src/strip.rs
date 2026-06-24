//! Strip-wise butteraugli computation for bounded peak memory at very
//! large image sizes.
//!
//! At 40 MP the full-image butteraugli pipeline allocates several
//! image-sized `f32` planes (3-channel XYB, 3-channel UHF/HF/MF/LF,
//! mask, AC accumulators, diffmap, plus the half-resolution variants),
//! totalling ~7 GiB of working memory. The strip walker bounds that
//! to roughly `O(strip_height * width)` by processing the image in
//! horizontal strips, computing each strip's per-pixel diffmap, and
//! aggregating the max-norm + libjxl 3-norm reductions across strips.
//!
//! ## Algorithm
//!
//! Butteraugli's blurs are FIR (finite impulse response): the
//! `compute_kernel` function in `blur.rs` returns a finite-radius
//! kernel of half-width `ceil(M * sigma)` where `M = 2.25`. The
//! largest sigma in the pipeline is `SIGMA_LF = 7.156`, giving a
//! per-blur half-width of 17 rows. Chained operations sum their
//! half-widths:
//!
//! - `separate_frequencies` runs LF/HF/UHF blurs in parallel,
//!   contributing 17 rows of halo.
//! - `compute_psycho_diff_malta` applies a 9x9 Malta filter on the
//!   blurred frequencies → +4 rows.
//! - `mask_psycho_image` blurs with `MASK_RADIUS = 2.7` (half-width
//!   7) and fuzzy-erodes (3x3 stencil, +1) → +8 rows.
//! - `combine_channels_to_diffmap_fused` is per-pixel.
//!
//! Sum across the chain: ~25 rows at full resolution.
//!
//! Multi-resolution mode runs the same chain on a 2x-downsampled
//! image; its 25-row halo at half-res = 50 rows at full resolution.
//!
//! The default strip halo is [`HALO_ROWS_DEFAULT`] = 64 rows, which
//! covers both with margin. Strip boundaries align to multiples of 2
//! so the 2x-downsample is consistent across strips.
//!
//! ## Parity
//!
//! With the default halo, strip-mode produces a diffmap that is
//! bit-identical to the full-image diffmap inside the strip's
//! interior region (modulo the image-edge boundary handling, which
//! is the same whether the strip touches the image edge or not). The
//! max-norm score and libjxl 3-norm aggregate to the same values as
//! the full-image path.
//!
//! ## Example
//!
//! ```
//! use butteraugli::{butteraugli_strip, ButteraugliParams, Img, RGB8};
//!
//! let width = 32;
//! let height = 32;
//! let pixels: Vec<RGB8> = (0..width * height)
//!     .map(|i| RGB8::new((i % 256) as u8, ((i * 2) % 256) as u8, ((i * 3) % 256) as u8))
//!     .collect();
//! let img = Img::new(pixels, width, height);
//!
//! let result = butteraugli_strip(
//!     img.as_ref(),
//!     img.as_ref(),
//!     &ButteraugliParams::default(),
//!     16,
//! )?;
//! assert!(result.score < 0.001);
//! # Ok::<(), butteraugli::ButteraugliError>(())
//! ```

use enough::Stop;
use imgref::ImgRef;
use rgb::{RGB, RGB8};

use crate::diff::{
    self, compute_diffmap_multiresolution_linear, compute_diffmap_single_resolution_linear,
    imgref_rgbf32_to_f32_vec, imgref_srgb_to_linear_f32,
};
use crate::image::ImageF;
use crate::precompute::ButteraugliReference;
use crate::{ButteraugliError, ButteraugliParams, ButteraugliResult, check_finite_f32};

/// Default number of halo rows above and below each strip.
///
/// 64 rows comfortably covers the chained-blur halo (~25 rows at
/// full-res) plus the multi-resolution sub-level halo (~50 rows at
/// full-res) with margin. Inside a strip's interior the per-pixel
/// diffmap is bit-identical to the full-image path.
pub const HALO_ROWS_DEFAULT: usize = 64;

/// Strip boundaries align to a multiple of this value so the 2x
/// sub-sample mapping is consistent across strips.
const STRIP_ALIGNMENT: usize = 2;

/// Minimum supported strip height (in rows).
///
/// Below 8 rows the multi-resolution path returns truncated
/// frequencies; the strip walker rejects the configuration.
pub const MIN_STRIP_HEIGHT: usize = 8;

/// Configuration for strip-wise butteraugli computation.
#[derive(Debug, Clone, Copy)]
pub struct ButteraugliStripConfig {
    /// Number of rows above and below each strip's interior that are
    /// processed but excluded from the per-pixel score reduction.
    pub halo_rows: usize,
}

impl Default for ButteraugliStripConfig {
    fn default() -> Self {
        Self {
            halo_rows: HALO_ROWS_DEFAULT,
        }
    }
}

impl ButteraugliStripConfig {
    /// Create a strip config with the given halo size (rows).
    #[must_use]
    pub fn with_halo_rows(halo_rows: usize) -> Self {
        Self { halo_rows }
    }
}

/// Per-strip reduction accumulator. Holds the max-norm across all
/// interior pixels seen so far, plus the `d^3 / d^6 / d^12` sums
/// needed for libjxl's 3-norm aggregation.
#[derive(Debug, Default)]
struct StripReducer {
    max_val: f32,
    sum_p3: f64,
    sum_p6: f64,
    sum_p12: f64,
    pixels: u64,
}

impl StripReducer {
    /// Accumulate the interior rows of `diffmap`. `interior_y0`
    /// and `interior_y1` are strip-local row indices (0-based into
    /// `diffmap`).
    fn add_strip(&mut self, diffmap: &ImageF, interior_y0: usize, interior_y1: usize) {
        let width = diffmap.width();
        let mut max_lanes = [0.0f32; 8];
        let mut sum_p3 = [0.0f64; 8];
        let mut sum_p6 = [0.0f64; 8];
        let mut sum_p12 = [0.0f64; 8];

        for y in interior_y0..interior_y1 {
            let row = diffmap.row(y);
            // Mirror diff.rs::compute_score_from_diffmap's 8-lane
            // reduction so the per-strip sums add up identically to
            // the full-image reduction modulo associativity.
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

        let mut strip_max = max_lanes[0];
        for &m in &max_lanes[1..] {
            if m > strip_max {
                strip_max = m;
            }
        }
        if strip_max > self.max_val {
            self.max_val = strip_max;
        }
        self.sum_p3 += sum_p3.iter().sum::<f64>();
        self.sum_p6 += sum_p6.iter().sum::<f64>();
        self.sum_p12 += sum_p12.iter().sum::<f64>();
        self.pixels += (interior_y1 - interior_y0) as u64 * width as u64;
    }

    fn finalise(&self, total_pixels: u64) -> (f64, f64) {
        debug_assert_eq!(
            self.pixels, total_pixels,
            "strip reducer accumulated {} pixels but image has {}; \
             strip walker has a coverage bug",
            self.pixels, total_pixels
        );
        if total_pixels == 0 {
            return (0.0, 0.0);
        }
        let one_per_pixels = 1.0_f64 / total_pixels as f64;
        let v0 = (one_per_pixels * self.sum_p3).powf(1.0 / 3.0);
        let v1 = (one_per_pixels * self.sum_p6).powf(1.0 / 6.0);
        let v2 = (one_per_pixels * self.sum_p12).powf(1.0 / 12.0);
        let pnorm_3 = (v0 + v1 + v2) / 3.0;
        (self.max_val as f64, pnorm_3)
    }
}

/// Strip-wise butteraugli with sRGB u8 inputs.
///
/// `strip_height` is the number of rows in each strip's "interior"
/// (the rows that contribute to the final score); each strip is
/// actually processed at `strip_height + 2 * halo_rows` rows where
/// `halo_rows = HALO_ROWS_DEFAULT`.
///
/// At 40 MP with `strip_height = 256`, peak working memory drops from
/// the full-image's ~7 GiB to roughly `O(strip_height * width)` ≈
/// 1.5 GiB.
///
/// # Errors
/// - If image dimensions don't match.
/// - If images are smaller than 8x8.
/// - If `strip_height < MIN_STRIP_HEIGHT`.
pub fn butteraugli_strip(
    img1: ImgRef<RGB8>,
    img2: ImgRef<RGB8>,
    params: &ButteraugliParams,
    strip_height: u32,
) -> Result<ButteraugliResult, ButteraugliError> {
    butteraugli_strip_with_config(
        img1,
        img2,
        params,
        strip_height,
        ButteraugliStripConfig::default(),
    )
}

/// Like [`butteraugli_strip`], but cooperatively cancellable via an
/// [`enough::Stop`] token.
///
/// `stop` is checked once per strip, at the top of the strip loop (the
/// outermost per-strip boundary); the per-strip diffmap kernels are never
/// interrupted mid-strip. If the token signals a stop, returns
/// [`ButteraugliError::Cancelled`] carrying the [`enough::StopReason`].
///
/// Pass [`enough::Unstoppable`] for a non-cancellable call (this is exactly
/// what [`butteraugli_strip`] does — it is zero-cost). Uses the default
/// strip config (halo = [`HALO_ROWS_DEFAULT`]).
///
/// # Errors
/// As [`butteraugli_strip`], plus [`ButteraugliError::Cancelled`] if `stop`
/// signals cancellation.
pub fn butteraugli_strip_with_stop(
    img1: ImgRef<RGB8>,
    img2: ImgRef<RGB8>,
    params: &ButteraugliParams,
    strip_height: u32,
    stop: &dyn Stop,
) -> Result<ButteraugliResult, ButteraugliError> {
    butteraugli_strip_with_config_and_stop(
        img1,
        img2,
        params,
        strip_height,
        ButteraugliStripConfig::default(),
        stop,
    )
}

/// Strip-wise butteraugli with sRGB u8 inputs and explicit
/// configuration.
///
/// # Errors
/// As [`butteraugli_strip`].
pub fn butteraugli_strip_with_config(
    img1: ImgRef<RGB8>,
    img2: ImgRef<RGB8>,
    params: &ButteraugliParams,
    strip_height: u32,
    config: ButteraugliStripConfig,
) -> Result<ButteraugliResult, ButteraugliError> {
    butteraugli_strip_with_config_and_stop(
        img1,
        img2,
        params,
        strip_height,
        config,
        &enough::Unstoppable,
    )
}

/// Shared body for the sRGB strip entry points; `stop` is threaded to the
/// strip walker which checks it once per strip.
fn butteraugli_strip_with_config_and_stop(
    img1: ImgRef<RGB8>,
    img2: ImgRef<RGB8>,
    params: &ButteraugliParams,
    strip_height: u32,
    config: ButteraugliStripConfig,
    stop: &dyn Stop,
) -> Result<ButteraugliResult, ButteraugliError> {
    params.validate()?;
    validate_image_pair(img1, img2, strip_height as usize)?;
    let (width, height) = (img1.width(), img1.height());
    // Reject dimensions that would overflow `width * height * 3`.
    width
        .checked_mul(height)
        .and_then(|wh| wh.checked_mul(3))
        .ok_or(ButteraugliError::DimensionOverflow { width, height })?;

    let linear1 = imgref_srgb_to_linear_f32(img1);
    let linear2 = imgref_srgb_to_linear_f32(img2);

    run_strip_walker_linear(
        &linear1,
        &linear2,
        width,
        height,
        strip_height as usize,
        params,
        config.halo_rows,
        params.compute_diffmap(),
        stop,
    )
}

/// Strip-wise butteraugli with linear-RGB f32 inputs.
///
/// # Errors
/// As [`butteraugli_strip`].
pub fn butteraugli_linear_strip(
    img1: ImgRef<RGB<f32>>,
    img2: ImgRef<RGB<f32>>,
    params: &ButteraugliParams,
    strip_height: u32,
) -> Result<ButteraugliResult, ButteraugliError> {
    butteraugli_linear_strip_with_config(
        img1,
        img2,
        params,
        strip_height,
        ButteraugliStripConfig::default(),
    )
}

/// Like [`butteraugli_linear_strip`], but cooperatively cancellable via an
/// [`enough::Stop`] token.
///
/// `stop` is checked once per strip, at the top of the strip loop (the
/// outermost per-strip boundary); the per-strip diffmap kernels are never
/// interrupted mid-strip. If the token signals a stop, returns
/// [`ButteraugliError::Cancelled`] carrying the [`enough::StopReason`].
///
/// Pass [`enough::Unstoppable`] for a non-cancellable call (this is exactly
/// what [`butteraugli_linear_strip`] does — it is zero-cost). Uses the default
/// strip config (halo = [`HALO_ROWS_DEFAULT`]).
///
/// # Errors
/// As [`butteraugli_linear_strip`], plus [`ButteraugliError::Cancelled`] if
/// `stop` signals cancellation.
pub fn butteraugli_linear_strip_with_stop(
    img1: ImgRef<RGB<f32>>,
    img2: ImgRef<RGB<f32>>,
    params: &ButteraugliParams,
    strip_height: u32,
    stop: &dyn Stop,
) -> Result<ButteraugliResult, ButteraugliError> {
    butteraugli_linear_strip_with_config_and_stop(
        img1,
        img2,
        params,
        strip_height,
        ButteraugliStripConfig::default(),
        stop,
    )
}

/// Strip-wise butteraugli with linear-RGB f32 inputs and explicit
/// configuration.
///
/// # Errors
/// As [`butteraugli_strip`].
pub fn butteraugli_linear_strip_with_config(
    img1: ImgRef<RGB<f32>>,
    img2: ImgRef<RGB<f32>>,
    params: &ButteraugliParams,
    strip_height: u32,
    config: ButteraugliStripConfig,
) -> Result<ButteraugliResult, ButteraugliError> {
    butteraugli_linear_strip_with_config_and_stop(
        img1,
        img2,
        params,
        strip_height,
        config,
        &enough::Unstoppable,
    )
}

/// Shared body for the linear-RGB strip entry points; `stop` is threaded to the
/// strip walker which checks it once per strip.
fn butteraugli_linear_strip_with_config_and_stop(
    img1: ImgRef<RGB<f32>>,
    img2: ImgRef<RGB<f32>>,
    params: &ButteraugliParams,
    strip_height: u32,
    config: ButteraugliStripConfig,
    stop: &dyn Stop,
) -> Result<ButteraugliResult, ButteraugliError> {
    params.validate()?;
    let (width, height) = (img1.width(), img1.height());
    let (w2, h2) = (img2.width(), img2.height());
    if width < 8 || height < 8 {
        return Err(ButteraugliError::ImageTooSmall { width, height });
    }
    if width != w2 || height != h2 {
        return Err(ButteraugliError::DimensionMismatch {
            w1: width,
            h1: height,
            w2,
            h2,
        });
    }
    if (strip_height as usize) < MIN_STRIP_HEIGHT {
        return Err(ButteraugliError::ImageTooSmall {
            width: strip_height as usize,
            height: MIN_STRIP_HEIGHT,
        });
    }
    width
        .checked_mul(height)
        .and_then(|wh| wh.checked_mul(3))
        .ok_or(ButteraugliError::DimensionOverflow { width, height })?;

    let linear1 = imgref_rgbf32_to_f32_vec(img1);
    let linear2 = imgref_rgbf32_to_f32_vec(img2);
    check_finite_f32(&linear1, "linear rgb1")?;
    check_finite_f32(&linear2, "linear rgb2")?;

    run_strip_walker_linear(
        &linear1,
        &linear2,
        width,
        height,
        strip_height as usize,
        params,
        config.halo_rows,
        params.compute_diffmap(),
        stop,
    )
}

fn validate_image_pair(
    img1: ImgRef<RGB8>,
    img2: ImgRef<RGB8>,
    strip_height: usize,
) -> Result<(), ButteraugliError> {
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
    if strip_height < MIN_STRIP_HEIGHT {
        return Err(ButteraugliError::ImageTooSmall {
            width: strip_height,
            height: MIN_STRIP_HEIGHT,
        });
    }
    Ok(())
}

/// Strip walker for the linear-RGB path. Slices the interleaved RGB
/// buffer for each strip, calls the existing diffmap path on the
/// strip's rows (with halo), accumulates the strip's interior into a
/// single max + p-norm reducer, and produces the final score.
///
/// `stop` is checked once per strip, at the top of the strip loop (the
/// outermost per-strip boundary) — never inside the per-strip diffmap
/// kernels. A `cancel()` is therefore honoured at strip granularity.
#[allow(clippy::too_many_arguments)]
fn run_strip_walker_linear(
    rgb1: &[f32],
    rgb2: &[f32],
    width: usize,
    height: usize,
    strip_height: usize,
    params: &ButteraugliParams,
    halo: usize,
    want_diffmap: bool,
    stop: &dyn Stop,
) -> Result<ButteraugliResult, ButteraugliError> {
    let mut full_diffmap: Option<Vec<f32>> = if want_diffmap {
        Some(vec![0.0; width * height])
    } else {
        None
    };
    let mut reducer = StripReducer::default();

    let mut y = 0usize;
    while y < height {
        // Cooperative cancellation: outermost per-strip boundary — checked
        // before this strip's diffmap is computed, never inside the kernels.
        stop.check().map_err(ButteraugliError::Cancelled)?;

        let mut next_y = (y + strip_height).next_multiple_of(STRIP_ALIGNMENT);
        if next_y >= height || height - next_y < STRIP_ALIGNMENT {
            next_y = height;
        }
        let interior_start = y;
        let interior_end = next_y;
        let halo_above = halo.min(interior_start);
        let halo_below = halo.min(height - interior_end);
        let strip_y0 = interior_start - halo_above;
        let strip_y1 = interior_end + halo_below;
        let strip_h_full = strip_y1 - strip_y0;

        let rgb_offset = strip_y0 * width * 3;
        let rgb_len = strip_h_full * width * 3;
        let strip_rgb1 = &rgb1[rgb_offset..rgb_offset + rgb_len];
        let strip_rgb2 = &rgb2[rgb_offset..rgb_offset + rgb_len];

        let diffmap = if width < diff::MIN_SIZE_FOR_MULTIRESOLUTION
            || strip_h_full < diff::MIN_SIZE_FOR_MULTIRESOLUTION
        {
            compute_diffmap_single_resolution_linear(
                strip_rgb1,
                strip_rgb2,
                width,
                strip_h_full,
                params,
            )
        } else {
            compute_diffmap_multiresolution_linear(
                strip_rgb1,
                strip_rgb2,
                width,
                strip_h_full,
                params,
            )
        };

        let interior_y0_in_strip = interior_start - strip_y0;
        let interior_y1_in_strip = interior_end - strip_y0;
        reducer.add_strip(&diffmap, interior_y0_in_strip, interior_y1_in_strip);

        if let Some(out) = full_diffmap.as_mut() {
            for (dst_y, src_y) in
                (interior_start..interior_end).zip(interior_y0_in_strip..interior_y1_in_strip)
            {
                let dst_row = &mut out[dst_y * width..(dst_y + 1) * width];
                let src_row = diffmap.row(src_y);
                dst_row.copy_from_slice(src_row);
            }
        }

        y = next_y;
    }

    let total_pixels = (width as u64) * (height as u64);
    let (score, pnorm_3) = reducer.finalise(total_pixels);

    if !score.is_finite() {
        return Err(ButteraugliError::NonFiniteResult);
    }

    Ok(ButteraugliResult {
        score,
        pnorm_3,
        diffmap: full_diffmap.map(|buf| imgref::ImgVec::new(buf, width, height)),
    })
}

impl ButteraugliReference {
    /// Compare a distorted sRGB image against the precomputed
    /// reference using strip-bounded peak memory.
    ///
    /// Mirrors [`ButteraugliReference::compare`] but processes the
    /// distorted side in strips of `strip_height` rows (plus the
    /// default halo). The cached reference's per-resolution data is
    /// not used by the strip walker — the strip walker recomputes
    /// the reference-side blurs per strip so the dist and ref blurs
    /// share the same FIR boundary handling. This still saves the
    /// XYB conversion + half-resolution downsample (cached on the
    /// `ButteraugliReference`) for trivial cases, but the principal
    /// benefit is the bounded peak memory.
    ///
    /// # Errors
    /// - If the distorted image's buffer size doesn't match the
    ///   reference dimensions.
    /// - If `strip_height < MIN_STRIP_HEIGHT`.
    pub fn compare_strip(
        &self,
        rgb: &[u8],
        strip_height: u32,
    ) -> Result<ButteraugliResult, ButteraugliError> {
        self.compare_strip_with_config(rgb, strip_height, ButteraugliStripConfig::default())
    }

    /// Cancellable variant of [`Self::compare_strip`].
    ///
    /// `stop` is checked once per strip, at the outermost per-strip boundary
    /// of the strip walker, before that strip's diffmap is computed; a
    /// cancelled token returns [`ButteraugliError::Cancelled`].
    /// [`enough::Unstoppable`] makes this behave identically to
    /// [`Self::compare_strip`] at zero cost.
    ///
    /// # Errors
    /// As [`Self::compare_strip`], plus [`ButteraugliError::Cancelled`] if
    /// `stop` signals cancellation between strips.
    pub fn compare_strip_with_stop(
        &self,
        rgb: &[u8],
        strip_height: u32,
        stop: &dyn Stop,
    ) -> Result<ButteraugliResult, ButteraugliError> {
        self.compare_strip_with_config_and_stop(
            rgb,
            strip_height,
            ButteraugliStripConfig::default(),
            stop,
        )
    }

    /// Strip-bounded comparison with explicit configuration.
    ///
    /// # Errors
    /// As [`ButteraugliReference::compare_strip`].
    pub fn compare_strip_with_config(
        &self,
        rgb: &[u8],
        strip_height: u32,
        config: ButteraugliStripConfig,
    ) -> Result<ButteraugliResult, ButteraugliError> {
        self.compare_strip_with_config_and_stop(rgb, strip_height, config, &enough::Unstoppable)
    }

    /// Shared body for the sRGB strip entry points; `stop` is threaded to the
    /// strip walker (checked once per strip). The non-cancellable entry points
    /// pass [`enough::Unstoppable`].
    ///
    /// # Errors
    /// As [`ButteraugliReference::compare_strip`], plus
    /// [`ButteraugliError::Cancelled`] if `stop` signals cancellation.
    fn compare_strip_with_config_and_stop(
        &self,
        rgb: &[u8],
        strip_height: u32,
        config: ButteraugliStripConfig,
        stop: &dyn Stop,
    ) -> Result<ButteraugliResult, ButteraugliError> {
        let width = self.width();
        let height = self.height();
        let expected = width * height * 3;
        if rgb.len() != expected {
            return Err(ButteraugliError::InvalidBufferSize {
                expected,
                actual: rgb.len(),
            });
        }
        if (strip_height as usize) < MIN_STRIP_HEIGHT {
            return Err(ButteraugliError::ImageTooSmall {
                width: strip_height as usize,
                height: MIN_STRIP_HEIGHT,
            });
        }
        // 0.9.4: source materialisation is owned. `Srgb`-stored
        // references (the common path from `new()`) re-derive the
        // linear bytes via the LUT used elsewhere in the pipeline;
        // `Linear`-stored references hand back a clone. Allocated once
        // per call and dropped when the strip walk completes — no
        // persistent linear retention.
        let linear1 = self
            .source_linear_rgb_owned()
            .ok_or(ButteraugliError::InvalidParameter {
                name: "reference",
                value: 0.0,
                reason: "compare_strip requires a reference built via \
                     ButteraugliReference::new or new_linear (the planar \
                     constructor does not retain interleaved source data, \
                     and `drop_strip_source` was not previously called)",
            })?;
        let lut = &*crate::opsin::SRGB_TO_LINEAR_LUT;
        let linear2: Vec<f32> = rgb.iter().map(|&v| lut[v as usize]).collect();

        // `stop` is checked once per strip inside the walker; non-cancellable
        // callers pass `Unstoppable` (zero-cost).
        run_strip_walker_linear(
            &linear1,
            &linear2,
            width,
            height,
            strip_height as usize,
            self.params(),
            config.halo_rows,
            self.params().compute_diffmap(),
            stop,
        )
    }

    /// Compare a distorted linear-RGB image against the precomputed
    /// reference using strip-bounded peak memory.
    ///
    /// # Errors
    /// As [`ButteraugliReference::compare_strip`].
    pub fn compare_linear_strip(
        &self,
        rgb: &[f32],
        strip_height: u32,
    ) -> Result<ButteraugliResult, ButteraugliError> {
        self.compare_linear_strip_with_config(rgb, strip_height, ButteraugliStripConfig::default())
    }

    /// Cancellable variant of [`Self::compare_linear_strip`].
    ///
    /// `stop` is checked once per strip, at the outermost per-strip boundary
    /// of the strip walker, before that strip's diffmap is computed; a
    /// cancelled token returns [`ButteraugliError::Cancelled`].
    /// [`enough::Unstoppable`] makes this behave identically to
    /// [`Self::compare_linear_strip`] at zero cost.
    ///
    /// # Errors
    /// As [`Self::compare_linear_strip`], plus [`ButteraugliError::Cancelled`]
    /// if `stop` signals cancellation between strips.
    pub fn compare_linear_strip_with_stop(
        &self,
        rgb: &[f32],
        strip_height: u32,
        stop: &dyn Stop,
    ) -> Result<ButteraugliResult, ButteraugliError> {
        self.compare_linear_strip_with_config_and_stop(
            rgb,
            strip_height,
            ButteraugliStripConfig::default(),
            stop,
        )
    }

    /// Strip-bounded linear-RGB comparison with explicit configuration.
    ///
    /// # Errors
    /// As [`ButteraugliReference::compare_strip`].
    pub fn compare_linear_strip_with_config(
        &self,
        rgb: &[f32],
        strip_height: u32,
        config: ButteraugliStripConfig,
    ) -> Result<ButteraugliResult, ButteraugliError> {
        self.compare_linear_strip_with_config_and_stop(
            rgb,
            strip_height,
            config,
            &enough::Unstoppable,
        )
    }

    /// Shared body for the linear-RGB strip entry points; `stop` is threaded
    /// to the strip walker (checked once per strip). The non-cancellable entry
    /// points pass [`enough::Unstoppable`].
    ///
    /// # Errors
    /// As [`ButteraugliReference::compare_strip`], plus
    /// [`ButteraugliError::Cancelled`] if `stop` signals cancellation.
    fn compare_linear_strip_with_config_and_stop(
        &self,
        rgb: &[f32],
        strip_height: u32,
        config: ButteraugliStripConfig,
        stop: &dyn Stop,
    ) -> Result<ButteraugliResult, ButteraugliError> {
        let width = self.width();
        let height = self.height();
        let expected = width * height * 3;
        if rgb.len() != expected {
            return Err(ButteraugliError::InvalidBufferSize {
                expected,
                actual: rgb.len(),
            });
        }
        if (strip_height as usize) < MIN_STRIP_HEIGHT {
            return Err(ButteraugliError::ImageTooSmall {
                width: strip_height as usize,
                height: MIN_STRIP_HEIGHT,
            });
        }
        check_finite_f32(rgb, "compare_linear_strip rgb")?;
        // 0.9.4: see compare_strip for source-materialisation rationale.
        let linear1 = self
            .source_linear_rgb_owned()
            .ok_or(ButteraugliError::InvalidParameter {
                name: "reference",
                value: 0.0,
                reason: "compare_linear_strip requires a reference built via \
                     ButteraugliReference::new or new_linear (the planar \
                     constructor does not retain interleaved source data, \
                     and `drop_strip_source` was not previously called)",
            })?;
        // `stop` is checked once per strip inside the walker; non-cancellable
        // callers pass `Unstoppable` (zero-cost).
        run_strip_walker_linear(
            &linear1,
            rgb,
            width,
            height,
            strip_height as usize,
            self.params(),
            config.halo_rows,
            self.params().compute_diffmap(),
            stop,
        )
    }

    /// Compare a distorted sRGB image (as `ImgRef<RGB8>`) against the
    /// reference using strip-bounded peak memory.
    ///
    /// # Errors
    /// As [`ButteraugliReference::compare_strip`].
    pub fn compare_strip_srgb(
        &self,
        img: ImgRef<RGB8>,
        strip_height: u32,
    ) -> Result<ButteraugliResult, ButteraugliError> {
        self.compare_strip_srgb_with_stop(img, strip_height, &enough::Unstoppable)
    }

    /// Cancellable variant of [`Self::compare_strip_srgb`].
    ///
    /// `stop` is checked once per strip inside the strip walker; a cancelled
    /// token returns [`ButteraugliError::Cancelled`]. [`enough::Unstoppable`]
    /// makes this behave identically to [`Self::compare_strip_srgb`] at zero
    /// cost.
    ///
    /// # Errors
    /// As [`Self::compare_strip_srgb`], plus [`ButteraugliError::Cancelled`]
    /// if `stop` signals cancellation between strips.
    pub fn compare_strip_srgb_with_stop(
        &self,
        img: ImgRef<RGB8>,
        strip_height: u32,
        stop: &dyn Stop,
    ) -> Result<ButteraugliResult, ButteraugliError> {
        if img.width() != self.width() || img.height() != self.height() {
            return Err(ButteraugliError::DimensionMismatch {
                w1: self.width(),
                h1: self.height(),
                w2: img.width(),
                h2: img.height(),
            });
        }
        let linear = imgref_srgb_to_linear_f32(img);
        self.compare_linear_strip_with_stop(&linear, strip_height, stop)
    }

    /// Compare a distorted linear-RGB image (as `ImgRef<RGB<f32>>`)
    /// against the reference using strip-bounded peak memory.
    ///
    /// # Errors
    /// As [`ButteraugliReference::compare_strip`].
    pub fn compare_strip_linear_imgref(
        &self,
        img: ImgRef<RGB<f32>>,
        strip_height: u32,
    ) -> Result<ButteraugliResult, ButteraugliError> {
        self.compare_strip_linear_imgref_with_stop(img, strip_height, &enough::Unstoppable)
    }

    /// Cancellable variant of [`Self::compare_strip_linear_imgref`].
    ///
    /// `stop` is checked once per strip inside the strip walker; a cancelled
    /// token returns [`ButteraugliError::Cancelled`]. [`enough::Unstoppable`]
    /// makes this behave identically to [`Self::compare_strip_linear_imgref`]
    /// at zero cost.
    ///
    /// # Errors
    /// As [`Self::compare_strip_linear_imgref`], plus
    /// [`ButteraugliError::Cancelled`] if `stop` signals cancellation between
    /// strips.
    pub fn compare_strip_linear_imgref_with_stop(
        &self,
        img: ImgRef<RGB<f32>>,
        strip_height: u32,
        stop: &dyn Stop,
    ) -> Result<ButteraugliResult, ButteraugliError> {
        if img.width() != self.width() || img.height() != self.height() {
            return Err(ButteraugliError::DimensionMismatch {
                w1: self.width(),
                h1: self.height(),
                w2: img.width(),
                h2: img.height(),
            });
        }
        let linear = imgref_rgbf32_to_f32_vec(img);
        check_finite_f32(&linear, "linear rgb")?;
        self.compare_linear_strip_with_stop(&linear, strip_height, stop)
    }
}
