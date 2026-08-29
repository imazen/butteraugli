//! Planar linear-light scoring API (`linear-planes` feature).
//!
//! This module is the supported, documented entry point for callers that
//! already hold **planar linear-light f32** image data — encoders scoring a
//! reconstruction against a source, GPU backends validating against the CPU
//! reference, batch harnesses scoring many distorted images against one
//! reference. It is a thin, allocation-conscious wrapper over the same
//! reference-precompute and strip-walker code paths the default API uses;
//! every score it returns is bit-identical to the corresponding
//! [`ButteraugliReference`] / [`crate::butteraugli_linear_strip`] call (see
//! *Parity* below).
//!
//! It exists because the previous way to reach these code paths was the
//! `internals` feature, which exports seven whole modules of unstable
//! implementation detail. `linear-planes` exports six types with a stated
//! contract instead.
//!
//! # Input contract
//!
//! [`LinearPlanes`] carries everything the metric needs to interpret a
//! buffer, checked at construction:
//!
//! - **Three separate channel planes** (`r`, `g`, `b`), each `f32`.
//! - **`width` × `height`** in pixels, both ≥ 8. Unlike
//!   [`crate::butteraugli`], this API does *not* reflect-pad sub-8px input —
//!   it rejects it with [`ButteraugliError::ImageTooSmall`], because a
//!   caller holding planar buffers is in a position to pad deliberately.
//! - **`stride`** in *pixels* (not bytes), ≥ `width`. Each plane must hold at
//!   least `stride * height` elements. Row `y` of channel `c` is
//!   `c[y * stride .. y * stride + width]`; the `stride - width` trailing
//!   elements of each row are ignored by the metric but must exist and be
//!   finite.
//! - **[`LinearColorSpace`]**, which states how the numbers are to be read.
//!   Today the only variant is [`LinearColorSpace::LinearSrgb`]: linear-light
//!   (no transfer function applied), sRGB / Rec.709 primaries, D65 white
//!   point, `0.0` = black. Values are **not** clamped to `1.0` — see below.
//!
//! Non-finite samples (NaN / ±inf) anywhere in the used region are rejected
//! with [`ButteraugliError::NonFiniteResult`].
//!
//! # Luminance scaling
//!
//! Butteraugli models absolute luminance. A linear input value of `1.0` is
//! mapped to [`ScorerBuilder::with_intensity_target`] nits (cd/m²) before the
//! opsin stage — default `80.0`, the SDR convention libjxl's
//! `butteraugli_main` uses. Values above `1.0` are accepted and map above the
//! intensity target; for HDR content scale your planes so `1.0` is the
//! mastering/display peak and set the intensity target to that peak in nits.
//! Negative values are clamped to `0.0` inside the opsin stage.
//!
//! # Usage
//!
//! Build one [`Scorer`] per reference image, then score many distorted images
//! against it. The reference-side XYB conversion, frequency decomposition and
//! mask precompute happen once, in [`ScorerBuilder::build`]; each
//! [`Scorer::score`] call only pays for the distorted side.
//!
//! ```
//! use butteraugli::linear_planes::{LinearPlanes, Scorer};
//!
//! let (w, h) = (32usize, 32usize);
//! let flat = vec![0.5f32; w * h];
//! let reference = LinearPlanes::new(&flat, &flat, &flat, w, h)?;
//!
//! let scorer = Scorer::builder()
//!     .with_intensity_target(80.0)
//!     .with_diffmap(true)
//!     .build(&reference)?;
//!
//! let scores = scorer.score(&reference)?;
//! assert!(scores.max_norm < 0.01);
//! assert!(scores.diffmap.is_some());
//! # Ok::<(), butteraugli::ButteraugliError>(())
//! ```
//!
//! # Resolution and walk modes
//!
//! Two independent axes, both explicit:
//!
//! - [`Resolution`] selects the scale pyramid.
//!   [`Resolution::MultiScale`] (default) matches libjxl
//!   `ButteraugliComparator::Diffmap` — full resolution plus one
//!   half-resolution sub-level. [`Resolution::SingleScale`] skips the
//!   sub-level: ~25% less work, ~15% of the diffmap weight dropped, and
//!   **not** parity with `butteraugli_main`.
//! - [`Walk`] selects how the image is traversed.
//!   [`Walk::WholeImage`] (default) allocates image-sized working planes.
//!   [`Walk::Strip`] bounds peak working memory to
//!   `O(strip_rows × width)` by walking horizontal strips with a halo, at
//!   the cost of recomputing the halo rows per strip. The two walks agree to
//!   within f64 summation noise — see *Parity*.
//!
//! # Outputs
//!
//! [`Scores`] always carries both aggregations:
//! [`max_norm`](Scores::max_norm) (the historical butteraugli score, `<1.0`
//! good / `>2.0` bad) and [`pnorm_3`](Scores::pnorm_3) (the libjxl 3-norm
//! that `butteraugli_main --pnorm` reports). The per-pixel
//! [`diffmap`](Scores::diffmap) is produced only when
//! [`ScorerBuilder::with_diffmap`] was set.
//!
//! # Parity
//!
//! This module adds no arithmetic of its own. Each mode forwards to an
//! existing code path with identical arguments, so scores are bit-identical
//! (`==` on the returned `f64`, not within a tolerance) to:
//!
//! | mode | equivalent default-API call |
//! |---|---|
//! | [`Walk::WholeImage`] | [`ButteraugliReference::new_linear_planar`] + [`ButteraugliReference::compare_linear_planar`] |
//! | [`Walk::Strip`] | [`crate::butteraugli_linear_strip_with_config`] |
//!
//! `tests/linear_planes_parity.rs` asserts exactly that, with no tolerance.
//!
//! Strip mode versus whole-image mode is a *different* question, and the
//! answer is "equal, but not by construction". The FIR blurs have finite
//! support, so with the default halo each strip's interior diffmap is
//! bit-identical to the whole-image diffmap — but the reductions run in a
//! different order, so only the max-norm is exactly equal; the 3-norm carries
//! f64 summation-associativity noise. Measured on this API at
//! 64×128 / 128×256 / 256×512 with 16/32/64-row strips:
//! [`max_norm`](Scores::max_norm) relative difference `0.0` (exact),
//! [`pnorm_3`](Scores::pnorm_3) relative difference `1e-12` to `6e-12`. The
//! default-API equivalent is covered by `tests/strip_parity.rs`. Do not rely
//! on strip and whole-image 3-norms comparing `==`.
//!
//! Under the `iir-blur` feature not even the max-norm is exact: the recursive
//! Gaussian's impulse response is infinite, so no halo bounds it and each
//! strip's filter state differs from the whole-image state. Measured max-norm
//! divergence `2.7e-7` – `1.2e-5` relative on the same grid. This is a
//! property of the strip walker itself, not of this wrapper — the default-API
//! `butteraugli_linear_strip` shows the same numbers. See the *Parity*
//! section of `strip.rs`.
//!
//! # Memory
//!
//! [`Walk::WholeImage`] keeps only the reference precompute — no copy of the
//! input planes — so [`Scorer::reference_bytes`] is the precompute alone.
//! [`Walk::Strip`] additionally retains an interleaved linear copy of the
//! reference planes (`width * height * 3 * 4` bytes), because the strip
//! walker re-derives reference-side data per strip rather than holding a
//! whole-image precompute, and it interleaves the *distorted* planes into a
//! second buffer of the same size on **every** [`Scorer::score`] call. What
//! strip mode bounds is the transient working set — the psycho pyramid, masks
//! and accumulators — which is the term that dominates at large sizes; it does
//! not reduce, and slightly increases, the per-call input footprint. Choose it
//! when the whole-image working set is what you are bounding.

use enough::Stop;
use imgref::ImgVec;

use crate::precompute::ButteraugliReference;
use crate::strip::{ButteraugliStripConfig, run_strip_walker_linear};
use crate::{ButteraugliError, ButteraugliParams, ButteraugliResult};

/// Smallest image dimension this API accepts, on either axis.
///
/// The blur and edge passes need at least this many pixels. Callers holding
/// planar buffers are expected to pad deliberately rather than have the
/// metric guess; sub-minimum input is rejected with
/// [`ButteraugliError::ImageTooSmall`].
pub const MIN_DIMENSION: usize = 8;

/// How the `f32` samples in a [`LinearPlanes`] are to be interpreted.
///
/// Non-exhaustive: future variants may add other primaries or white points.
/// Match with a `_` arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum LinearColorSpace {
    /// Linear-light (no transfer function applied), sRGB / Rec.709 primaries,
    /// D65 white point. `0.0` is black; `1.0` is mapped to
    /// [`ScorerBuilder::with_intensity_target`] nits. Values above `1.0` are
    /// allowed and scale past the intensity target.
    ///
    /// This is what butteraugli's opsin stage assumes, and what libjxl's
    /// `butteraugli_main` feeds it after decoding sRGB input.
    #[default]
    LinearSrgb,
}

/// A borrowed view of one image as three strided linear-light `f32` planes.
///
/// Constructed through [`LinearPlanes::new`] (tightly packed) or
/// [`LinearPlanes::with_stride`] (padded rows). Construction validates
/// dimensions, stride, buffer lengths and finiteness, so a `LinearPlanes`
/// value is always scorable.
///
/// See the [module docs](self#input-contract) for the full contract.
#[derive(Debug, Clone, Copy)]
pub struct LinearPlanes<'a> {
    r: &'a [f32],
    g: &'a [f32],
    b: &'a [f32],
    width: usize,
    height: usize,
    stride: usize,
    color_space: LinearColorSpace,
}

impl<'a> LinearPlanes<'a> {
    /// Wrap three tightly packed planes (`stride == width`).
    ///
    /// # Errors
    /// See [`LinearPlanes::with_stride`].
    pub fn new(
        r: &'a [f32],
        g: &'a [f32],
        b: &'a [f32],
        width: usize,
        height: usize,
    ) -> Result<Self, ButteraugliError> {
        Self::with_stride(r, g, b, width, height, width)
    }

    /// Wrap three planes whose rows are `stride` pixels apart.
    ///
    /// `stride` is measured in **pixels**, not bytes, and must be ≥ `width`.
    /// Each plane must hold at least `stride * height` elements.
    ///
    /// # Errors
    /// - [`ButteraugliError::ImageTooSmall`] if `width` or `height` is below
    ///   [`MIN_DIMENSION`].
    /// - [`ButteraugliError::InvalidParameter`] (`name = "stride"`) if
    ///   `stride < width`.
    /// - [`ButteraugliError::DimensionOverflow`] if `stride * height`
    ///   overflows `usize`.
    /// - [`ButteraugliError::InvalidBufferSize`] if any plane is shorter than
    ///   `stride * height`.
    /// - [`ButteraugliError::NonFiniteResult`] if any sample in the used
    ///   region is NaN or infinite.
    pub fn with_stride(
        r: &'a [f32],
        g: &'a [f32],
        b: &'a [f32],
        width: usize,
        height: usize,
        stride: usize,
    ) -> Result<Self, ButteraugliError> {
        if width < MIN_DIMENSION || height < MIN_DIMENSION {
            return Err(ButteraugliError::ImageTooSmall { width, height });
        }
        if stride < width {
            return Err(ButteraugliError::InvalidParameter {
                name: "stride",
                value: stride as f64,
                reason: "must be >= width (stride is measured in pixels)",
            });
        }
        let needed = stride
            .checked_mul(height)
            .ok_or(ButteraugliError::DimensionOverflow { width, height })?;
        let shortest = r.len().min(g.len()).min(b.len());
        if shortest < needed {
            return Err(ButteraugliError::InvalidBufferSize {
                expected: needed,
                actual: shortest,
            });
        }
        crate::check_finite_f32(&r[..needed], "linear-planes r")?;
        crate::check_finite_f32(&g[..needed], "linear-planes g")?;
        crate::check_finite_f32(&b[..needed], "linear-planes b")?;
        Ok(Self {
            r,
            g,
            b,
            width,
            height,
            stride,
            color_space: LinearColorSpace::LinearSrgb,
        })
    }

    /// Restate the colour space these samples are in.
    ///
    /// Purely declarative — it does not convert. Reference and distorted
    /// planes must agree, or [`Scorer::score`] returns
    /// [`ButteraugliError::InvalidParameter`].
    #[must_use]
    pub fn with_color_space(mut self, color_space: LinearColorSpace) -> Self {
        self.color_space = color_space;
        self
    }

    /// Image width in pixels.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Image height in pixels.
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Distance between row starts, in pixels.
    #[must_use]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// How these samples are to be interpreted.
    #[must_use]
    pub fn color_space(&self) -> LinearColorSpace {
        self.color_space
    }

    /// The red plane, as passed in.
    #[must_use]
    pub fn red(&self) -> &'a [f32] {
        self.r
    }

    /// The green plane, as passed in.
    #[must_use]
    pub fn green(&self) -> &'a [f32] {
        self.g
    }

    /// The blue plane, as passed in.
    #[must_use]
    pub fn blue(&self) -> &'a [f32] {
        self.b
    }

    /// Interleave the used region into `width * height * 3` RGB f32 samples.
    fn to_interleaved(self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.width * self.height * 3);
        for y in 0..self.height {
            let row = y * self.stride;
            let (rr, gg, bb) = (
                &self.r[row..row + self.width],
                &self.g[row..row + self.width],
                &self.b[row..row + self.width],
            );
            for x in 0..self.width {
                out.push(rr[x]);
                out.push(gg[x]);
                out.push(bb[x]);
            }
        }
        out
    }
}

/// Which scale levels the metric runs.
///
/// Non-exhaustive; match with a `_` arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum Resolution {
    /// Full resolution plus one half-resolution sub-level — what libjxl's
    /// `ButteraugliComparator::Diffmap` (and therefore `butteraugli_main`)
    /// does. Use this whenever scores must be comparable to libjxl.
    #[default]
    MultiScale,
    /// Full resolution only. Roughly 25% less work; drops the ~15% of the
    /// diffmap weight the sub-level contributes, so scores are **not**
    /// comparable to `butteraugli_main`. Intended for encoder search loops
    /// where relative ordering matters more than absolute value.
    SingleScale,
}

/// Strip-walk tuning: strip height and halo depth, both in rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StripMode {
    rows: u32,
    halo_rows: usize,
}

impl StripMode {
    /// Walk the image in strips of `rows` interior rows.
    ///
    /// `rows` must be at least [`crate::MIN_STRIP_HEIGHT`]; smaller values are
    /// rejected by [`ScorerBuilder::build`]. Halo defaults to
    /// [`crate::HALO_ROWS_DEFAULT`], which covers the chained-blur reach at
    /// both scales.
    #[must_use]
    pub fn new(rows: u32) -> Self {
        Self {
            rows,
            halo_rows: crate::HALO_ROWS_DEFAULT,
        }
    }

    /// Override the halo depth (rows processed above and below each strip's
    /// interior but excluded from that strip's reduction).
    ///
    /// Lowering this below [`crate::HALO_ROWS_DEFAULT`] trades parity for
    /// speed: the halo has to cover the summed reach of the blur, Malta and
    /// mask passes (~25 rows at full resolution, ~50 once the half-resolution
    /// sub-level is included) or strip interiors stop matching the
    /// whole-image diffmap.
    #[must_use]
    pub fn with_halo_rows(mut self, halo_rows: usize) -> Self {
        self.halo_rows = halo_rows;
        self
    }

    /// Interior rows per strip.
    #[must_use]
    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// Halo rows above and below each strip's interior.
    #[must_use]
    pub fn halo_rows(&self) -> usize {
        self.halo_rows
    }
}

/// How the image is traversed.
///
/// Non-exhaustive; match with a `_` arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum Walk {
    /// Score the whole image at once. Fastest, and the reference precompute
    /// is reused across every [`Scorer::score`] call. Working-set memory
    /// scales with the image.
    #[default]
    WholeImage,
    /// Score in horizontal strips, bounding working-set memory to
    /// `O(strip_rows × width)`. Costs the halo rows' work per strip and
    /// retains an interleaved copy of the reference planes.
    Strip(StripMode),
}

/// Both butteraugli aggregations, plus the optional per-pixel diffmap.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Scores {
    /// Max-norm — the historical butteraugli score. `<1.0` reads as "looks
    /// the same", `>2.0` as "visibly different". Equivalent to libjxl's
    /// `ButteraugliScoreFromDiffmap` and to
    /// [`ButteraugliResult::score`](crate::ButteraugliResult::score).
    pub max_norm: f64,
    /// libjxl 3-norm: the average of p-norms at exponents 3, 6 and 12,
    /// matching `lib/extras/metrics.cc:ComputeDistanceP` at `p = 3` and what
    /// `butteraugli_main --pnorm` prints. Always populated, with or without a
    /// diffmap.
    pub pnorm_3: f64,
    /// Per-pixel difference map, `width × height`, present only when
    /// [`ScorerBuilder::with_diffmap`] was enabled.
    pub diffmap: Option<ImgVec<f32>>,
}

impl Scores {
    fn from_result(r: ButteraugliResult) -> Self {
        Self {
            max_norm: r.score,
            pnorm_3: r.pnorm_3,
            diffmap: r.diffmap,
        }
    }
}

/// Builder for a [`Scorer`].
///
/// Every knob has a documented default; `ScorerBuilder::default()` reproduces
/// libjxl `butteraugli_main` behaviour at SDR (80 nits, multi-scale, whole
/// image, no diffmap).
#[derive(Debug, Clone)]
pub struct ScorerBuilder {
    intensity_target: f32,
    hf_asymmetry: f32,
    xmul: f32,
    resolution: Resolution,
    walk: Walk,
    diffmap: bool,
}

impl Default for ScorerBuilder {
    fn default() -> Self {
        let d = ButteraugliParams::default();
        Self {
            intensity_target: d.intensity_target(),
            hf_asymmetry: d.hf_asymmetry(),
            xmul: d.xmul(),
            resolution: Resolution::MultiScale,
            walk: Walk::WholeImage,
            diffmap: false,
        }
    }
}

impl ScorerBuilder {
    /// A builder with libjxl-default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Luminance in nits (cd/m²) that a linear input value of `1.0` maps to.
    ///
    /// Default `80.0` (the SDR convention `butteraugli_main` uses). Must be
    /// finite and positive.
    #[must_use]
    pub fn with_intensity_target(mut self, nits: f32) -> Self {
        self.intensity_target = nits;
        self
    }

    /// High-frequency asymmetry multiplier. `> 1.0` penalises *added*
    /// high-frequency detail (ringing, blocking) more than *removed* detail
    /// (blur). Default `1.0`; must be finite and positive.
    #[must_use]
    pub fn with_hf_asymmetry(mut self, hf_asymmetry: f32) -> Self {
        self.hf_asymmetry = hf_asymmetry;
        self
    }

    /// X (red-green opponent) channel multiplier. Default `1.0`; must be
    /// finite and non-negative.
    #[must_use]
    pub fn with_xmul(mut self, xmul: f32) -> Self {
        self.xmul = xmul;
        self
    }

    /// Select the scale pyramid. Default [`Resolution::MultiScale`].
    #[must_use]
    pub fn with_resolution(mut self, resolution: Resolution) -> Self {
        self.resolution = resolution;
        self
    }

    /// Select the traversal. Default [`Walk::WholeImage`].
    #[must_use]
    pub fn with_walk(mut self, walk: Walk) -> Self {
        self.walk = walk;
        self
    }

    /// Produce [`Scores::diffmap`] on every score call. Default `false`
    /// (cheaper — the internal diffmap is still built and reduced, but is
    /// freed rather than copied out).
    #[must_use]
    pub fn with_diffmap(mut self, diffmap: bool) -> Self {
        self.diffmap = diffmap;
        self
    }

    /// The [`ButteraugliParams`] these settings correspond to.
    ///
    /// Useful for cross-checking against a default-API call site during
    /// migration.
    #[must_use]
    pub fn to_params(&self) -> ButteraugliParams {
        ButteraugliParams::new()
            .with_intensity_target(self.intensity_target)
            .with_hf_asymmetry(self.hf_asymmetry)
            .with_xmul(self.xmul)
            .with_compute_diffmap(self.diffmap)
            .with_single_resolution(self.resolution == Resolution::SingleScale)
    }

    /// Precompute reference-side data and return a reusable [`Scorer`].
    ///
    /// In [`Walk::WholeImage`] mode this runs the reference XYB conversion,
    /// frequency decomposition and mask precompute once; the planes are not
    /// retained. In [`Walk::Strip`] mode it retains an interleaved linear copy
    /// of the planes instead, which the strip walker consumes per strip.
    ///
    /// # Errors
    /// - [`ButteraugliError::InvalidParameter`] if any parameter is out of
    ///   range, or (`name = "strip_rows"`) if a [`StripMode`] asks for fewer
    ///   than [`crate::MIN_STRIP_HEIGHT`] rows.
    /// - [`ButteraugliError::DimensionOverflow`] if the image dimensions
    ///   overflow the internal buffer-size arithmetic.
    /// - Any error [`ButteraugliReference::new_linear_planar`] reports.
    pub fn build(&self, reference: &LinearPlanes<'_>) -> Result<Scorer, ButteraugliError> {
        let params = self.to_params();
        params.validate()?;
        let (width, height) = (reference.width, reference.height);
        width
            .checked_mul(height)
            .and_then(|wh| wh.checked_mul(3))
            .ok_or(ButteraugliError::DimensionOverflow { width, height })?;

        let backing = match self.walk {
            Walk::WholeImage => Backing::Warm(Box::new(ButteraugliReference::new_linear_planar(
                reference.r,
                reference.g,
                reference.b,
                width,
                height,
                reference.stride,
                params.clone(),
            )?)),
            Walk::Strip(mode) => {
                if (mode.rows as usize) < crate::MIN_STRIP_HEIGHT {
                    return Err(ButteraugliError::InvalidParameter {
                        name: "strip_rows",
                        value: f64::from(mode.rows),
                        reason: "must be >= butteraugli::MIN_STRIP_HEIGHT",
                    });
                }
                Backing::Strip {
                    reference: reference.to_interleaved(),
                    mode,
                }
            }
        };

        Ok(Scorer {
            backing,
            params,
            width,
            height,
            color_space: reference.color_space,
            walk: self.walk,
            diffmap: self.diffmap,
        })
    }
}

/// Reference-side state, shaped by the [`Walk`] mode.
enum Backing {
    /// Whole-image mode: the warm reference precompute. Boxed because it is
    /// much larger than the strip variant.
    Warm(Box<ButteraugliReference>),
    /// Strip mode: the interleaved linear reference the walker re-slices.
    Strip {
        reference: Vec<f32>,
        mode: StripMode,
    },
}

/// A reference image with its precompute, ready to score distorted images.
///
/// Build one with [`Scorer::builder`] (or [`Scorer::new`] for defaults), then
/// call [`Scorer::score`] repeatedly. `Scorer` is `Send + Sync`; scoring takes
/// `&self`, so one scorer can serve concurrent callers.
pub struct Scorer {
    backing: Backing,
    params: ButteraugliParams,
    width: usize,
    height: usize,
    color_space: LinearColorSpace,
    walk: Walk,
    diffmap: bool,
}

impl Scorer {
    /// Start configuring a scorer.
    #[must_use]
    pub fn builder() -> ScorerBuilder {
        ScorerBuilder::new()
    }

    /// Build a scorer with default settings (80 nits, multi-scale, whole
    /// image, no diffmap).
    ///
    /// # Errors
    /// As [`ScorerBuilder::build`].
    pub fn new(reference: &LinearPlanes<'_>) -> Result<Self, ButteraugliError> {
        ScorerBuilder::new().build(reference)
    }

    /// Reference image width in pixels.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Reference image height in pixels.
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// The traversal this scorer was built for.
    #[must_use]
    pub fn walk(&self) -> Walk {
        self.walk
    }

    /// The colour space the reference declared; distorted input must match.
    #[must_use]
    pub fn color_space(&self) -> LinearColorSpace {
        self.color_space
    }

    /// The [`ButteraugliParams`] this scorer scores with.
    #[must_use]
    pub fn params(&self) -> &ButteraugliParams {
        &self.params
    }

    /// Heap bytes retained on the reference side.
    ///
    /// Whole-image mode reports the precompute
    /// ([`ButteraugliReference::memory_bytes`]); strip mode reports the
    /// retained interleaved reference copy. Neither includes the transient
    /// per-score working set.
    #[must_use]
    pub fn reference_bytes(&self) -> usize {
        match &self.backing {
            Backing::Warm(r) => r.memory_bytes(),
            Backing::Strip { reference, .. } => reference.len() * size_of::<f32>(),
        }
    }

    /// Score `distorted` against the reference.
    ///
    /// # Errors
    /// - [`ButteraugliError::DimensionMismatch`] if `distorted` is not the
    ///   reference's size.
    /// - [`ButteraugliError::InvalidParameter`] (`name = "color_space"`) if
    ///   `distorted` declares a different [`LinearColorSpace`].
    /// - [`ButteraugliError::NonFiniteResult`] if the computation produces a
    ///   non-finite score.
    pub fn score(&self, distorted: &LinearPlanes<'_>) -> Result<Scores, ButteraugliError> {
        self.score_with_stop(distorted, &enough::Unstoppable)
    }

    /// Cancellable [`Scorer::score`].
    ///
    /// `stop` is polled at the outermost per-scale boundary in whole-image
    /// mode and once per strip in strip mode — before any per-pixel work in
    /// each case. A cancelled token returns
    /// [`ButteraugliError::Cancelled`]. [`enough::Unstoppable`] costs nothing.
    ///
    /// # Errors
    /// As [`Scorer::score`], plus [`ButteraugliError::Cancelled`].
    pub fn score_with_stop(
        &self,
        distorted: &LinearPlanes<'_>,
        stop: &dyn Stop,
    ) -> Result<Scores, ButteraugliError> {
        self.check_compatible(distorted)?;
        match &self.backing {
            // `compare_linear_planar_impl` skips the buffer-size and
            // finiteness validation that the public `compare_linear_planar*`
            // methods perform. That is sound here and deliberate:
            // `LinearPlanes` already proved every plane holds `stride * height`
            // finite samples at construction, and `check_compatible` above
            // proved the dimensions match. Re-scanning 3 * w * h samples per
            // score call cost +3.6% at 1024x1024 (measured 2026-08-28).
            Backing::Warm(reference) => reference
                .compare_linear_planar_impl(
                    distorted.r,
                    distorted.g,
                    distorted.b,
                    distorted.stride,
                    stop,
                )
                .and_then(|r| {
                    if r.score.is_finite() {
                        Ok(r)
                    } else {
                        Err(ButteraugliError::NonFiniteResult)
                    }
                })
                .map(|r| {
                    // The warm-reference path always materialises a diffmap
                    // internally and hands it back regardless of
                    // `compute_diffmap`. This API promises a diffmap only
                    // when one was asked for, so drop it otherwise.
                    let mut s = Scores::from_result(r);
                    if !self.diffmap {
                        s.diffmap = None;
                    }
                    s
                }),
            Backing::Strip { reference, mode } => run_strip_walker_linear(
                reference,
                &distorted.to_interleaved(),
                self.width,
                self.height,
                mode.rows as usize,
                &self.params,
                mode.halo_rows,
                self.diffmap,
                stop,
            )
            .map(Scores::from_result),
        }
    }

    /// Score `distorted`, writing the per-pixel diffmap into a caller-owned
    /// buffer instead of allocating one.
    ///
    /// `diffmap_out` is resized to `width * height` and fully overwritten;
    /// reusing one `Vec` across calls avoids a `width * height * 4`-byte
    /// allocation per score. The returned [`Scores`] has
    /// [`diffmap`](Scores::diffmap) set to `None` — the pixels are in
    /// `diffmap_out`.
    ///
    /// Available in both walk modes. In [`Walk::WholeImage`] mode the
    /// diffmap is written directly into `diffmap_out` regardless of the
    /// [`ScorerBuilder::with_diffmap`] setting. In [`Walk::Strip`] mode the
    /// walker builds its own diffmap and it is moved into `diffmap_out`, so
    /// the buffer is replaced rather than reused.
    ///
    /// # Errors
    /// As [`Scorer::score`].
    pub fn score_into(
        &self,
        distorted: &LinearPlanes<'_>,
        diffmap_out: &mut Vec<f32>,
    ) -> Result<Scores, ButteraugliError> {
        self.check_compatible(distorted)?;
        match &self.backing {
            // Same reasoning as `score_with_stop`: validation already done.
            Backing::Warm(reference) => {
                let (max_norm, pnorm_3) = reference.compare_linear_planar_impl_into(
                    distorted.r,
                    distorted.g,
                    distorted.b,
                    distorted.stride,
                    diffmap_out,
                    &enough::Unstoppable,
                )?;
                if !max_norm.is_finite() {
                    return Err(ButteraugliError::NonFiniteResult);
                }
                Ok(Scores {
                    max_norm,
                    pnorm_3,
                    diffmap: None,
                })
            }
            Backing::Strip { reference, mode } => {
                let result = run_strip_walker_linear(
                    reference,
                    &distorted.to_interleaved(),
                    self.width,
                    self.height,
                    mode.rows as usize,
                    &self.params,
                    mode.halo_rows,
                    true,
                    &enough::Unstoppable,
                )?;
                *diffmap_out = result.diffmap.map_or_else(Vec::new, imgref::Img::into_buf);
                Ok(Scores {
                    max_norm: result.score,
                    pnorm_3: result.pnorm_3,
                    diffmap: None,
                })
            }
        }
    }

    fn check_compatible(&self, distorted: &LinearPlanes<'_>) -> Result<(), ButteraugliError> {
        if distorted.width != self.width || distorted.height != self.height {
            return Err(ButteraugliError::DimensionMismatch {
                w1: self.width,
                h1: self.height,
                w2: distorted.width,
                h2: distorted.height,
            });
        }
        if distorted.color_space != self.color_space {
            return Err(ButteraugliError::InvalidParameter {
                name: "color_space",
                value: 0.0,
                reason: "distorted planes must declare the same LinearColorSpace as the reference",
            });
        }
        Ok(())
    }
}

impl core::fmt::Debug for Scorer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Scorer")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("color_space", &self.color_space)
            .field("walk", &self.walk)
            .field("diffmap", &self.diffmap)
            .field("reference_bytes", &self.reference_bytes())
            // `backing` (megabytes of precompute) and `params` (derivable
            // from the builder settings already shown) are deliberately
            // omitted — printing them would be noise, not information.
            .finish_non_exhaustive()
    }
}

/// Strip config for the given mode — kept so the mapping onto the default
/// API's [`ButteraugliStripConfig`] is one obvious line in
/// `docs/MIGRATION_0.9.4.md`.
impl From<StripMode> for ButteraugliStripConfig {
    fn from(mode: StripMode) -> Self {
        Self::with_halo_rows(mode.halo_rows)
    }
}

#[cfg(test)]
// Exact `f64` equality is the assertion these tests make: this module must
// forward to the existing code paths without perturbing a single bit. A
// tolerance would let a real divergence hide inside it.
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    fn ramp(w: usize, h: usize, stride: usize, bias: f32) -> Vec<f32> {
        let mut v = vec![0.0f32; stride * h];
        for y in 0..h {
            for x in 0..w {
                v[y * stride + x] = ((x * 7 + y * 13) % 251) as f32 / 251.0 * 0.9 + bias;
            }
        }
        v
    }

    #[test]
    fn rejects_sub_minimum_dimensions() {
        let p = vec![0.5f32; 64];
        let e = LinearPlanes::new(&p, &p, &p, 4, 4).unwrap_err();
        assert!(matches!(
            e,
            ButteraugliError::ImageTooSmall {
                width: 4,
                height: 4
            }
        ));
    }

    #[test]
    fn rejects_stride_below_width() {
        let p = vec![0.5f32; 1024];
        let e = LinearPlanes::with_stride(&p, &p, &p, 16, 16, 8).unwrap_err();
        assert!(matches!(
            e,
            ButteraugliError::InvalidParameter { name: "stride", .. }
        ));
    }

    #[test]
    fn rejects_short_plane() {
        let full = vec![0.5f32; 16 * 16];
        let short = vec![0.5f32; 16 * 15];
        let e = LinearPlanes::new(&full, &short, &full, 16, 16).unwrap_err();
        assert!(matches!(
            e,
            ButteraugliError::InvalidBufferSize {
                expected: 256,
                actual: 240
            }
        ));
    }

    #[test]
    fn rejects_non_finite_sample() {
        let good = vec![0.5f32; 16 * 16];
        let mut bad = good.clone();
        bad[100] = f32::NAN;
        assert!(LinearPlanes::new(&good, &bad, &good, 16, 16).is_err());
    }

    #[test]
    fn identical_planes_score_near_zero() {
        let (w, h) = (32, 32);
        let p = ramp(w, h, w, 0.05);
        let planes = LinearPlanes::new(&p, &p, &p, w, h).unwrap();
        let scorer = Scorer::new(&planes).unwrap();
        let s = scorer.score(&planes).unwrap();
        assert!(s.max_norm < 0.01, "max_norm = {}", s.max_norm);
        assert!(s.pnorm_3 < 0.01, "pnorm_3 = {}", s.pnorm_3);
        assert!(s.diffmap.is_none());
    }

    #[test]
    fn stride_padding_does_not_change_score() {
        let (w, h) = (32, 32);
        let tight_r = ramp(w, h, w, 0.05);
        let tight_d = ramp(w, h, w, 0.20);
        let pad_r = ramp(w, h, w + 11, 0.05);
        let pad_d = ramp(w, h, w + 11, 0.20);

        let tr = LinearPlanes::new(&tight_r, &tight_r, &tight_r, w, h).unwrap();
        let td = LinearPlanes::new(&tight_d, &tight_d, &tight_d, w, h).unwrap();
        let pr = LinearPlanes::with_stride(&pad_r, &pad_r, &pad_r, w, h, w + 11).unwrap();
        let pd = LinearPlanes::with_stride(&pad_d, &pad_d, &pad_d, w, h, w + 11).unwrap();

        let a = Scorer::new(&tr).unwrap().score(&td).unwrap();
        let b = Scorer::new(&pr).unwrap().score(&pd).unwrap();
        assert_eq!(a.max_norm, b.max_norm);
        assert_eq!(a.pnorm_3, b.pnorm_3);
    }

    #[test]
    fn dimension_mismatch_is_reported() {
        let (w, h) = (32, 32);
        let p = ramp(w, h, w, 0.05);
        let q = ramp(16, 16, 16, 0.05);
        let scorer = Scorer::new(&LinearPlanes::new(&p, &p, &p, w, h).unwrap()).unwrap();
        let other = LinearPlanes::new(&q, &q, &q, 16, 16).unwrap();
        assert!(matches!(
            scorer.score(&other).unwrap_err(),
            ButteraugliError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn diffmap_opt_in_produces_map() {
        let (w, h) = (32, 32);
        let reference = ramp(w, h, w, 0.05);
        let distorted = ramp(w, h, w, 0.20);
        let rp = LinearPlanes::new(&reference, &reference, &reference, w, h).unwrap();
        let dp = LinearPlanes::new(&distorted, &distorted, &distorted, w, h).unwrap();
        let scorer = Scorer::builder().with_diffmap(true).build(&rp).unwrap();
        let s = scorer.score(&dp).unwrap();
        let dm = s.diffmap.expect("diffmap requested");
        assert_eq!((dm.width(), dm.height()), (w, h));
    }

    #[test]
    fn score_into_matches_score_whole_image() {
        let (w, h) = (32, 32);
        let r = ramp(w, h, w, 0.05);
        let d = ramp(w, h, w, 0.20);
        let rp = LinearPlanes::new(&r, &r, &r, w, h).unwrap();
        let dp = LinearPlanes::new(&d, &d, &d, w, h).unwrap();
        let scorer = Scorer::builder().with_diffmap(true).build(&rp).unwrap();
        let owned = scorer.score(&dp).unwrap();
        let mut buf = Vec::new();
        let into = scorer.score_into(&dp, &mut buf).unwrap();
        assert_eq!(owned.max_norm, into.max_norm);
        assert_eq!(owned.pnorm_3, into.pnorm_3);
        assert_eq!(owned.diffmap.unwrap().buf().as_slice(), buf.as_slice());
    }

    #[test]
    fn strip_rows_below_minimum_rejected() {
        let (w, h) = (32, 32);
        let p = ramp(w, h, w, 0.05);
        let rp = LinearPlanes::new(&p, &p, &p, w, h).unwrap();
        let e = Scorer::builder()
            .with_walk(Walk::Strip(StripMode::new(4)))
            .build(&rp)
            .unwrap_err();
        assert!(matches!(
            e,
            ButteraugliError::InvalidParameter {
                name: "strip_rows",
                ..
            }
        ));
    }

    #[test]
    fn single_scale_differs_from_multi_scale() {
        let (w, h) = (32, 32);
        let r = ramp(w, h, w, 0.05);
        let d = ramp(w, h, w, 0.30);
        let rp = LinearPlanes::new(&r, &r, &r, w, h).unwrap();
        let dp = LinearPlanes::new(&d, &d, &d, w, h).unwrap();
        let multi = Scorer::new(&rp).unwrap().score(&dp).unwrap();
        let single = Scorer::builder()
            .with_resolution(Resolution::SingleScale)
            .build(&rp)
            .unwrap()
            .score(&dp)
            .unwrap();
        assert!(multi.max_norm > 0.0 && single.max_norm > 0.0);
        assert_ne!(multi.max_norm, single.max_norm);
    }

    #[test]
    fn builder_to_params_round_trips() {
        let p = Scorer::builder()
            .with_intensity_target(250.0)
            .with_hf_asymmetry(1.5)
            .with_xmul(0.75)
            .with_resolution(Resolution::SingleScale)
            .with_diffmap(true)
            .to_params();
        assert_eq!(p.intensity_target(), 250.0);
        assert_eq!(p.hf_asymmetry(), 1.5);
        assert_eq!(p.xmul(), 0.75);
        assert!(p.single_resolution());
        assert!(p.compute_diffmap());
    }
}
