//! `zenpsy-api` trait implementations for `butteraugli`.
//!
//! Provides [`ButteraugliMetric`] implementing [`zenpsy_api::Metric`] and
//! [`zenpsy_api::PrepareReference`] (returning owned [`PreparedButteraugli`]):
//! - **Score-only by default:** [`EvalOpts::new()`] computes `"butteraugli_max"`
//!   and `"butteraugli_pnorm3"` (`Direction::Lower`, perfect `0.0`) via a
//!   row-streaming reduction with **zero** full-resolution diffmap allocation.
//! - **Opt-in diffmap:** [`MapRequest::Native`] and [`MapRequest::Named("diffmap")`]
//!   compute and expose the per-pixel error map cropped to logical dimensions.
//!   Unknown named maps fail immediately with [`ErrorKind::NoMap`] before any
//!   allocation or computation.
//! - **Strict SDR BT.709 ingress:** Accepts full-range BT.709 sRGB (`RGB8`,
//!   `BGR8`, `RGBX8`, `BGRX8`, `RGBA8`, `BGRA8`, `RGB16`, `RGBA16`) and linear
//!   (`RGBF32`, `RGBAF32`, `RGB16`, `RGBA16`) inputs with `Orientation::Identity`
//!   and no HDR metadata / `DiffuseWhite`. Straight alpha is accepted only when
//!   every active pixel is fully opaque; translucent or premultiplied alpha is
//!   refused explicitly with [`ErrorKind::UnsupportedInput`].

use std::sync::OnceLock;

use enough::{Stop, StopReason};
use imgref::ImgVec;
use zenpsy_api::zenpixels::{
    AlphaMode, ColorSpace, Orientation, PixelView, PlaneFormat, PlaneRole, PlaneView, Primaries,
    SampleLayout, SampleStride, SampleType, SignalRange, Transfer,
};
use zenpsy_api::{
    Direction, ErrorKind, EvalOpts, Evaluation, MapGrid, MapRequest, MapView, Metric, MetricError,
    MetricId, MetricResult, PrepareReference, Score, ScoreSpec, WarmReference, fail, whereat,
};

use crate::{
    ButteraugliError, ButteraugliParams, ButteraugliReference, ButteraugliResult, Numerics,
};

static BUTTERAUGLI_MAX_SPEC: ScoreSpec =
    ScoreSpec::new("butteraugli_max", Direction::Lower).with_perfect(0.0);
static BUTTERAUGLI_PNORM3_SPEC: ScoreSpec =
    ScoreSpec::new("butteraugli_pnorm3", Direction::Lower).with_perfect(0.0);
static BUTTERAUGLI_DIFFMAP_SPEC: ScoreSpec =
    ScoreSpec::new("diffmap", Direction::Lower).with_perfect(0.0);

/// Latches the first observed [`StopReason`] so a one-shot stop token (e.g. a
/// transient timeout) is preserved across parallel branches and phase boundaries.
struct LatchingStop<'a> {
    inner: &'a dyn Stop,
    latched: OnceLock<StopReason>,
}

impl<'a> LatchingStop<'a> {
    fn new(inner: &'a dyn Stop) -> Self {
        Self {
            inner,
            latched: OnceLock::new(),
        }
    }

    fn reason(&self) -> Option<StopReason> {
        self.latched.get().copied()
    }
}

impl Stop for LatchingStop<'_> {
    fn check(&self) -> Result<(), StopReason> {
        if let Some(&reason) = self.latched.get() {
            return Err(reason);
        }
        if let Err(reason) = self.inner.check() {
            let _ = self.latched.set(reason);
            return Err(self.latched.get().copied().unwrap_or(reason));
        }
        Ok(())
    }

    fn may_stop(&self) -> bool {
        self.inner.may_stop() || self.latched.get().is_some()
    }
}

/// Configured Butteraugli scorer implementing [`zenpsy_api::Metric`] and
/// [`zenpsy_api::PrepareReference`].
#[derive(Clone, Debug, Default)]
pub struct ButteraugliMetric {
    params: ButteraugliParams,
}

impl ButteraugliMetric {
    /// Create a scorer with default [`ButteraugliParams`] (`Numerics::Fast`,
    /// `intensity_target = 80.0`, `hf_asymmetry = 1.0`, `xmul = 1.0`).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a scorer with explicit [`ButteraugliParams`].
    ///
    /// Note: [`ButteraugliParams::compute_diffmap`] is controlled per call by
    /// [`EvalOpts`] (`MapRequest::None` vs `Native` / `Named("diffmap")`).
    #[must_use]
    pub fn with_params(params: ButteraugliParams) -> Self {
        Self { params }
    }

    /// Configured parameters.
    #[must_use]
    pub fn params(&self) -> &ButteraugliParams {
        &self.params
    }
}

/// Native `zenpsy-api` assessment returned by [`ButteraugliMetric`].
///
/// Publishes `"butteraugli_max"` first (headline score, `Direction::Lower`,
/// perfect `0.0`) followed by `"butteraugli_pnorm3"` (`Direction::Lower`,
/// perfect `0.0`). Carries the per-pixel `"diffmap"` only when explicitly
/// requested.
#[derive(Clone, Debug)]
pub struct ButteraugliEvaluation {
    result: ButteraugliResult,
}

impl ButteraugliEvaluation {
    /// Complete native [`ButteraugliResult`].
    #[must_use]
    pub fn result(&self) -> &ButteraugliResult {
        &self.result
    }

    /// Max-norm butteraugli score (`0.0` = identical, `<1.0` = good, `>2.0` = visible).
    #[must_use]
    pub fn score(&self) -> f64 {
        self.result.score
    }

    /// Max-norm butteraugli score (alias for [`Self::score`]).
    #[must_use]
    pub fn max_norm(&self) -> f64 {
        self.result.score
    }

    /// libjxl-style 3-norm aggregation (`p = 3, 6, 12`).
    #[must_use]
    pub fn pnorm_3(&self) -> f64 {
        self.result.pnorm_3
    }

    /// Opt-in per-pixel error map cropped to logical image dimensions.
    #[must_use]
    pub fn diffmap(&self) -> Option<&ImgVec<f32>> {
        self.result.diffmap.as_ref()
    }

    /// Consume this evaluation and return its owned [`ButteraugliResult`].
    #[must_use]
    pub fn into_result(self) -> ButteraugliResult {
        self.result
    }
}

impl Evaluation for ButteraugliEvaluation {
    fn scores(&self) -> impl Iterator<Item = Score<'_>> {
        [
            Score::new(&BUTTERAUGLI_MAX_SPEC, self.result.score),
            Score::new(&BUTTERAUGLI_PNORM3_SPEC, self.result.pnorm_3),
        ]
        .into_iter()
    }

    fn maps(&self) -> impl Iterator<Item = MapView<'_>> {
        self.result.diffmap.iter().filter_map(|dm| {
            let w = u32::try_from(dm.width()).ok()?;
            let h = u32::try_from(dm.height()).ok()?;
            let plane = PlaneView::from_samples_strided(
                dm.buf(),
                w,
                h,
                SampleStride(dm.stride()),
                PlaneFormat::new(PlaneRole::Extra(0), SampleType::F32),
            )
            .ok()?;
            let grid = MapGrid::new(w, h).ok()?;
            Some(MapView::new(&BUTTERAUGLI_DIFFMAP_SPEC, plane, grid))
        })
    }
}

/// Owned prepared Butteraugli reference for repeated `zenpsy-api` comparisons
/// against one reference image.
pub struct PreparedButteraugli {
    orig_w: usize,
    orig_h: usize,
    params: ButteraugliParams,
    reference: ButteraugliReference,
}

impl std::fmt::Debug for PreparedButteraugli {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedButteraugli")
            .field("orig_w", &self.orig_w)
            .field("orig_h", &self.orig_h)
            .field("padded_w", &self.reference.width())
            .field("padded_h", &self.reference.height())
            .field("params", &self.params)
            .finish_non_exhaustive()
    }
}

impl WarmReference for PreparedButteraugli {
    type Output = ButteraugliEvaluation;

    fn evaluate(
        &mut self,
        candidate: &PixelView<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Output> {
        let want_diffmap = validate_map_request(opts.map_request())?;
        let latch = LatchingStop::new(opts.stop());
        check_stop(&latch)?;

        let cw = usize::try_from(candidate.width())
            .map_err(|_| err_msg(ErrorKind::Resource, "width overflows usize"))?;
        let ch = usize::try_from(candidate.height())
            .map_err(|_| err_msg(ErrorKind::Resource, "height overflows usize"))?;
        if cw != self.orig_w || ch != self.orig_h {
            return fail(
                MetricError::new(ErrorKind::DimensionMismatch)
                    .with_metric("butteraugli")
                    .with_message(format!(
                        "candidate dimensions {cw}x{ch} do not match reference {}x{}",
                        self.orig_w, self.orig_h
                    )),
            );
        }

        let (lin_cand, _, _, _, _) = view_to_padded_linear_rgb(
            candidate,
            self.params.numerics(),
            opts.max_pixels(),
            &latch,
        )?;
        check_stop(&latch)?;

        let mut result = self
            .reference
            .compare_linear_with_stop_mode(&lin_cand, want_diffmap, &latch)
            .map_err(|e| map_butteraugli_error(e, &latch))?;
        check_stop(&latch)?;

        if let Some(dm) = result.diffmap.take() {
            result.diffmap = Some(crate::crop_diffmap(dm, self.orig_w, self.orig_h));
        }
        Ok(ButteraugliEvaluation { result })
    }
}

impl Metric for ButteraugliMetric {
    type Output = ButteraugliEvaluation;

    fn id(&self) -> MetricId {
        let numerics_str = match self.params.numerics() {
            Numerics::Fast => "fast",
            Numerics::LibjxlExact => "libjxl_exact",
        };
        #[cfg(feature = "iir-blur")]
        let blur_str = match self.params.numerics() {
            Numerics::Fast => "iir",
            Numerics::LibjxlExact => "fir",
        };
        #[cfg(not(feature = "iir-blur"))]
        let blur_str = "fir";

        MetricId::new("butteraugli", zenpsy_api::implementation!(1))
            .with_reference("libjxl butteraugli")
            .with_config("numerics", numerics_str)
            .with_config("blur", blur_str)
            .with_config(
                "intensity_target",
                format!("{}", self.params.intensity_target()),
            )
            .with_config("hf_asymmetry", format!("{}", self.params.hf_asymmetry()))
            .with_config("xmul", format!("{}", self.params.xmul()))
            .with_config("single_resolution", self.params.single_resolution())
    }

    fn evaluate(
        &mut self,
        reference: &PixelView<'_>,
        candidate: &PixelView<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Output> {
        let want_diffmap = validate_map_request(opts.map_request())?;
        let latch = LatchingStop::new(opts.stop());
        check_stop(&latch)?;
        self.params
            .validate()
            .map_err(|e| map_butteraugli_error(e, &latch))?;

        if reference.width() != candidate.width() || reference.height() != candidate.height() {
            return fail(
                MetricError::new(ErrorKind::DimensionMismatch)
                    .with_metric("butteraugli")
                    .with_message(format!(
                        "image dimensions do not match: {}x{} vs {}x{}",
                        reference.width(),
                        reference.height(),
                        candidate.width(),
                        candidate.height()
                    )),
            );
        }

        let (lin1, orig_w, orig_h, pw, ph) = view_to_padded_linear_rgb(
            reference,
            self.params.numerics(),
            opts.max_pixels(),
            &latch,
        )?;
        let (lin2, _, _, _, _) = view_to_padded_linear_rgb(
            candidate,
            self.params.numerics(),
            opts.max_pixels(),
            &latch,
        )?;
        check_stop(&latch)?;

        let internal = crate::diff::compute_butteraugli_linear_impl_mode(
            &lin1,
            &lin2,
            pw,
            ph,
            &self.params,
            want_diffmap,
            &latch,
        )
        .map_err(|e| map_butteraugli_error(e, &latch))?;
        check_stop(&latch)?;

        if !internal.score.is_finite() || !internal.pnorm_3.is_finite() {
            return fail(
                MetricError::new(ErrorKind::InvalidInput)
                    .with_metric("butteraugli")
                    .with_message("score computation produced non-finite value"),
            );
        }

        let diffmap = internal
            .diffmap
            .map(crate::image::ImageF::into_imgvec)
            .map(|dm| crate::crop_diffmap(dm, orig_w, orig_h));

        Ok(ButteraugliEvaluation {
            result: ButteraugliResult {
                score: internal.score,
                pnorm_3: internal.pnorm_3,
                diffmap,
            },
        })
    }
}

impl PrepareReference for ButteraugliMetric {
    type Prepared<'a> = PreparedButteraugli;

    fn warm_reference(
        &mut self,
        reference: &PixelView<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Prepared<'_>> {
        let _ = validate_map_request(opts.map_request())?;
        let latch = LatchingStop::new(opts.stop());
        check_stop(&latch)?;
        self.params
            .validate()
            .map_err(|e| map_butteraugli_error(e, &latch))?;

        let (lin_ref, orig_w, orig_h, pw, ph) = view_to_padded_linear_rgb(
            reference,
            self.params.numerics(),
            opts.max_pixels(),
            &latch,
        )?;
        check_stop(&latch)?;

        let reference = ButteraugliReference::new_linear_with_stop(
            &lin_ref,
            pw,
            ph,
            self.params.clone(),
            false,
            &latch,
        )
        .map_err(|e| map_butteraugli_error(e, &latch))?;
        check_stop(&latch)?;

        Ok(PreparedButteraugli {
            orig_w,
            orig_h,
            params: self.params.clone(),
            reference,
        })
    }
}

#[track_caller]
fn err_msg(kind: ErrorKind, msg: &'static str) -> whereat::At<MetricError> {
    whereat::at(
        MetricError::new(kind)
            .with_metric("butteraugli")
            .with_message(msg),
    )
}

#[track_caller]
fn validate_map_request(req: MapRequest<'_>) -> MetricResult<bool> {
    match req {
        MapRequest::None => Ok(false),
        MapRequest::Native | MapRequest::Named("diffmap") => Ok(true),
        MapRequest::Named(_) => fail(
            MetricError::new(ErrorKind::NoMap)
                .with_metric("butteraugli")
                .with_message("butteraugli only publishes the 'diffmap' map"),
        ),
        _ => fail(
            MetricError::new(ErrorKind::UnsupportedRequest)
                .with_metric("butteraugli")
                .with_message("unsupported map request variant"),
        ),
    }
}

#[track_caller]
fn check_stop(latch: &LatchingStop<'_>) -> MetricResult<()> {
    latch
        .check()
        .map_err(|r| whereat::at(MetricError::stopped(r).with_metric("butteraugli")))
}

#[track_caller]
fn map_butteraugli_error(
    err: ButteraugliError,
    latch: &LatchingStop<'_>,
) -> whereat::At<MetricError> {
    if let Some(reason) = latch.reason() {
        return whereat::at(MetricError::stopped(reason).with_metric("butteraugli"));
    }
    match err {
        ButteraugliError::Cancelled(reason) => {
            whereat::at(MetricError::stopped(reason).with_metric("butteraugli"))
        }
        other => {
            let kind = match &other {
                ButteraugliError::DimensionMismatch { .. } => ErrorKind::DimensionMismatch,
                ButteraugliError::ImageTooSmall { .. }
                | ButteraugliError::InvalidDimensions { .. } => ErrorKind::TooSmall,
                ButteraugliError::DimensionOverflow { .. } => ErrorKind::Resource,
                ButteraugliError::InvalidParameter { .. } => ErrorKind::InvalidConfig,
                ButteraugliError::InvalidBufferSize { .. } | ButteraugliError::NonFiniteResult => {
                    ErrorKind::InvalidInput
                }
                ButteraugliError::Cancelled(_) => unreachable!(),
            };
            whereat::at(MetricError::from_detail(kind, other).with_metric("butteraugli"))
        }
    }
}

#[inline]
fn srgb_u16_to_linear(v: u16) -> f32 {
    let s = f32::from(v) / 65535.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

#[derive(Clone, Copy, Debug)]
enum IngressKind {
    SrgbRgb8,
    SrgbBgr8,
    SrgbRgba8 { check_opaque: bool },
    SrgbBgra8 { check_opaque: bool },
    SrgbRgb16,
    SrgbRgba16 { check_opaque: bool },
    LinearRgb16,
    LinearRgba16 { check_opaque: bool },
    LinearRgbF32,
    LinearRgbaF32 { check_opaque: bool },
}

#[track_caller]
fn classify_view(view: &PixelView<'_>) -> MetricResult<IngressKind> {
    if view.orientation() != Orientation::Identity {
        return Err(err_msg(
            ErrorKind::UnsupportedInput,
            "pending non-identity orientation must be applied before scoring",
        ));
    }
    if !view.hdr().is_empty() {
        return Err(err_msg(
            ErrorKind::UnsupportedInput,
            "HDR metadata is not accepted on Butteraugli SDR zenpsy-api ingress",
        ));
    }

    let fmt = view.format();
    let ColorSpace::Rgb {
        primaries: Primaries::Bt709,
        transfer: transfer @ (Transfer::Srgb | Transfer::Linear),
        range: SignalRange::Full,
        alpha,
        white: None,
    } = fmt.color
    else {
        return Err(err_msg(
            ErrorKind::UnsupportedInput,
            "requires full-range BT.709 RGB (sRGB or Linear transfer, no DiffuseWhite)",
        ));
    };

    let layout = fmt.layout;
    let kind = match (transfer, layout, alpha) {
        (Transfer::Srgb, SampleLayout::RGB8, AlphaMode::None) => IngressKind::SrgbRgb8,
        (Transfer::Srgb, SampleLayout::BGR8, AlphaMode::None) => IngressKind::SrgbBgr8,
        (Transfer::Srgb, SampleLayout::RGBX8, AlphaMode::Undefined)
        | (Transfer::Srgb, SampleLayout::RGBA8, AlphaMode::Opaque) => IngressKind::SrgbRgba8 {
            check_opaque: false,
        },
        (Transfer::Srgb, SampleLayout::RGBA8, AlphaMode::Straight) => {
            IngressKind::SrgbRgba8 { check_opaque: true }
        }
        (Transfer::Srgb, SampleLayout::BGRX8, AlphaMode::Undefined)
        | (Transfer::Srgb, SampleLayout::BGRA8, AlphaMode::Opaque) => IngressKind::SrgbBgra8 {
            check_opaque: false,
        },
        (Transfer::Srgb, SampleLayout::BGRA8, AlphaMode::Straight) => {
            IngressKind::SrgbBgra8 { check_opaque: true }
        }
        (Transfer::Srgb, SampleLayout::RGB16, AlphaMode::None) => IngressKind::SrgbRgb16,
        (Transfer::Srgb, SampleLayout::RGBA16, AlphaMode::Opaque) => IngressKind::SrgbRgba16 {
            check_opaque: false,
        },
        (Transfer::Srgb, SampleLayout::RGBA16, AlphaMode::Straight) => {
            IngressKind::SrgbRgba16 { check_opaque: true }
        }
        (Transfer::Linear, SampleLayout::RGB16, AlphaMode::None) => IngressKind::LinearRgb16,
        (Transfer::Linear, SampleLayout::RGBA16, AlphaMode::Opaque) => IngressKind::LinearRgba16 {
            check_opaque: false,
        },
        (Transfer::Linear, SampleLayout::RGBA16, AlphaMode::Straight) => {
            IngressKind::LinearRgba16 { check_opaque: true }
        }
        (Transfer::Linear, SampleLayout::RGBF32, AlphaMode::None) => IngressKind::LinearRgbF32,
        (Transfer::Linear, SampleLayout::RGBAF32, AlphaMode::Opaque) => {
            IngressKind::LinearRgbaF32 {
                check_opaque: false,
            }
        }
        (Transfer::Linear, SampleLayout::RGBAF32, AlphaMode::Straight) => {
            IngressKind::LinearRgbaF32 { check_opaque: true }
        }
        (_, _, AlphaMode::Premultiplied) => {
            return Err(err_msg(
                ErrorKind::UnsupportedInput,
                "premultiplied alpha is not supported; composite onto an opaque matte first",
            ));
        }
        _ => {
            return Err(err_msg(
                ErrorKind::UnsupportedInput,
                "unsupported pixel layout and alpha combination for Butteraugli ingress",
            ));
        }
    };
    Ok(kind)
}

fn view_to_padded_linear_rgb(
    view: &PixelView<'_>,
    numerics: Numerics,
    max_pixels: Option<u64>,
    latch: &LatchingStop<'_>,
) -> MetricResult<(Vec<f32>, usize, usize, usize, usize)> {
    check_stop(latch)?;
    let kind = classify_view(view)?;

    let w_u32 = view.width();
    let h_u32 = view.height();
    if w_u32 == 0 || h_u32 == 0 {
        return Err(err_msg(
            ErrorKind::TooSmall,
            "image dimensions must be non-zero",
        ));
    }

    let orig_w = usize::try_from(w_u32)
        .map_err(|_| err_msg(ErrorKind::Resource, "width overflows usize"))?;
    let orig_h = usize::try_from(h_u32)
        .map_err(|_| err_msg(ErrorKind::Resource, "height overflows usize"))?;
    let pw = orig_w.max(8);
    let ph = orig_h.max(8);

    if let Some(limit) = max_pixels {
        let orig_px = u64::from(w_u32)
            .checked_mul(u64::from(h_u32))
            .ok_or_else(|| err_msg(ErrorKind::Resource, "pixel count overflow"))?;
        if orig_px > limit {
            return fail(
                MetricError::new(ErrorKind::Resource)
                    .with_metric("butteraugli")
                    .with_message(format!(
                        "image has {orig_px} active pixels ({orig_w}x{orig_h}), exceeding max_pixels={limit}"
                    )),
            );
        }
    }

    let total_floats = pw
        .checked_mul(ph)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| err_msg(ErrorKind::Resource, "linear buffer size overflow"))?;
    let mut out = Vec::new();
    out.try_reserve_exact(total_floats).map_err(|_| {
        err_msg(
            ErrorKind::Resource,
            "out of memory allocating linear RGB buffer",
        )
    })?;
    out.resize(total_floats, 0.0f32);

    let lut = crate::exact::srgb_lut(numerics);
    let row_floats = pw * 3;

    for y in 0..orig_h {
        if y.is_multiple_of(64) {
            check_stop(latch)?;
        }
        let src_row = view.row(y as u32);
        let dst_row = &mut out[y * row_floats..y * row_floats + orig_w * 3];

        match kind {
            IngressKind::SrgbRgb8 => {
                for x in 0..orig_w {
                    let src = x * 3;
                    dst_row[src] = lut[src_row[src] as usize];
                    dst_row[src + 1] = lut[src_row[src + 1] as usize];
                    dst_row[src + 2] = lut[src_row[src + 2] as usize];
                }
            }
            IngressKind::SrgbBgr8 => {
                for x in 0..orig_w {
                    let src = x * 3;
                    dst_row[src] = lut[src_row[src + 2] as usize];
                    dst_row[src + 1] = lut[src_row[src + 1] as usize];
                    dst_row[src + 2] = lut[src_row[src] as usize];
                }
            }
            IngressKind::SrgbRgba8 { check_opaque } => {
                for x in 0..orig_w {
                    let src = x * 4;
                    if check_opaque && src_row[src + 3] != 255 {
                        return Err(err_msg(
                            ErrorKind::UnsupportedInput,
                            "translucent alpha pixel refused; composite onto an opaque matte first",
                        ));
                    }
                    let dst = x * 3;
                    dst_row[dst] = lut[src_row[src] as usize];
                    dst_row[dst + 1] = lut[src_row[src + 1] as usize];
                    dst_row[dst + 2] = lut[src_row[src + 2] as usize];
                }
            }
            IngressKind::SrgbBgra8 { check_opaque } => {
                for x in 0..orig_w {
                    let src = x * 4;
                    if check_opaque && src_row[src + 3] != 255 {
                        return Err(err_msg(
                            ErrorKind::UnsupportedInput,
                            "translucent alpha pixel refused; composite onto an opaque matte first",
                        ));
                    }
                    let dst = x * 3;
                    dst_row[dst] = lut[src_row[src + 2] as usize];
                    dst_row[dst + 1] = lut[src_row[src + 1] as usize];
                    dst_row[dst + 2] = lut[src_row[src] as usize];
                }
            }
            IngressKind::SrgbRgb16 => {
                for x in 0..orig_w {
                    let src = x * 6;
                    let red = u16::from_ne_bytes([src_row[src], src_row[src + 1]]);
                    let green = u16::from_ne_bytes([src_row[src + 2], src_row[src + 3]]);
                    let blue = u16::from_ne_bytes([src_row[src + 4], src_row[src + 5]]);
                    let dst = x * 3;
                    dst_row[dst] = srgb_u16_to_linear(red);
                    dst_row[dst + 1] = srgb_u16_to_linear(green);
                    dst_row[dst + 2] = srgb_u16_to_linear(blue);
                }
            }
            IngressKind::SrgbRgba16 { check_opaque } => {
                for x in 0..orig_w {
                    let src = x * 8;
                    if check_opaque {
                        let alpha_val = u16::from_ne_bytes([src_row[src + 6], src_row[src + 7]]);
                        if alpha_val != 65535 {
                            return Err(err_msg(
                                ErrorKind::UnsupportedInput,
                                "translucent alpha pixel refused; composite onto an opaque matte first",
                            ));
                        }
                    }
                    let red = u16::from_ne_bytes([src_row[src], src_row[src + 1]]);
                    let green = u16::from_ne_bytes([src_row[src + 2], src_row[src + 3]]);
                    let blue = u16::from_ne_bytes([src_row[src + 4], src_row[src + 5]]);
                    let dst = x * 3;
                    dst_row[dst] = srgb_u16_to_linear(red);
                    dst_row[dst + 1] = srgb_u16_to_linear(green);
                    dst_row[dst + 2] = srgb_u16_to_linear(blue);
                }
            }
            IngressKind::LinearRgb16 => {
                const INV_65535: f32 = 1.0 / 65535.0;
                for x in 0..orig_w {
                    let src = x * 6;
                    let red = u16::from_ne_bytes([src_row[src], src_row[src + 1]]);
                    let green = u16::from_ne_bytes([src_row[src + 2], src_row[src + 3]]);
                    let blue = u16::from_ne_bytes([src_row[src + 4], src_row[src + 5]]);
                    let dst = x * 3;
                    dst_row[dst] = f32::from(red) * INV_65535;
                    dst_row[dst + 1] = f32::from(green) * INV_65535;
                    dst_row[dst + 2] = f32::from(blue) * INV_65535;
                }
            }
            IngressKind::LinearRgba16 { check_opaque } => {
                const INV_65535: f32 = 1.0 / 65535.0;
                for x in 0..orig_w {
                    let src = x * 8;
                    if check_opaque {
                        let alpha_val = u16::from_ne_bytes([src_row[src + 6], src_row[src + 7]]);
                        if alpha_val != 65535 {
                            return Err(err_msg(
                                ErrorKind::UnsupportedInput,
                                "translucent alpha pixel refused; composite onto an opaque matte first",
                            ));
                        }
                    }
                    let red = u16::from_ne_bytes([src_row[src], src_row[src + 1]]);
                    let green = u16::from_ne_bytes([src_row[src + 2], src_row[src + 3]]);
                    let blue = u16::from_ne_bytes([src_row[src + 4], src_row[src + 5]]);
                    let dst = x * 3;
                    dst_row[dst] = f32::from(red) * INV_65535;
                    dst_row[dst + 1] = f32::from(green) * INV_65535;
                    dst_row[dst + 2] = f32::from(blue) * INV_65535;
                }
            }
            IngressKind::LinearRgbF32 => {
                for x in 0..orig_w {
                    let src = x * 12;
                    let red = f32::from_ne_bytes([
                        src_row[src],
                        src_row[src + 1],
                        src_row[src + 2],
                        src_row[src + 3],
                    ]);
                    let green = f32::from_ne_bytes([
                        src_row[src + 4],
                        src_row[src + 5],
                        src_row[src + 6],
                        src_row[src + 7],
                    ]);
                    let blue = f32::from_ne_bytes([
                        src_row[src + 8],
                        src_row[src + 9],
                        src_row[src + 10],
                        src_row[src + 11],
                    ]);
                    if !red.is_finite() || !green.is_finite() || !blue.is_finite() {
                        return Err(err_msg(
                            ErrorKind::InvalidInput,
                            "non-finite f32 sample in linear RGB input",
                        ));
                    }
                    let dst = x * 3;
                    dst_row[dst] = red;
                    dst_row[dst + 1] = green;
                    dst_row[dst + 2] = blue;
                }
            }
            IngressKind::LinearRgbaF32 { check_opaque } => {
                for x in 0..orig_w {
                    let src = x * 16;
                    let red = f32::from_ne_bytes([
                        src_row[src],
                        src_row[src + 1],
                        src_row[src + 2],
                        src_row[src + 3],
                    ]);
                    let green = f32::from_ne_bytes([
                        src_row[src + 4],
                        src_row[src + 5],
                        src_row[src + 6],
                        src_row[src + 7],
                    ]);
                    let blue = f32::from_ne_bytes([
                        src_row[src + 8],
                        src_row[src + 9],
                        src_row[src + 10],
                        src_row[src + 11],
                    ]);
                    if !red.is_finite() || !green.is_finite() || !blue.is_finite() {
                        return Err(err_msg(
                            ErrorKind::InvalidInput,
                            "non-finite f32 sample in linear RGBA input",
                        ));
                    }
                    if check_opaque {
                        let alpha_val = f32::from_ne_bytes([
                            src_row[src + 12],
                            src_row[src + 13],
                            src_row[src + 14],
                            src_row[src + 15],
                        ]);
                        if !alpha_val.is_finite() {
                            return Err(err_msg(
                                ErrorKind::InvalidInput,
                                "non-finite f32 alpha sample in linear RGBA input",
                            ));
                        }
                        if alpha_val.to_bits() != 1.0_f32.to_bits() {
                            return Err(err_msg(
                                ErrorKind::UnsupportedInput,
                                "translucent alpha pixel refused; composite onto an opaque matte first",
                            ));
                        }
                    }
                    let dst = x * 3;
                    dst_row[dst] = red;
                    dst_row[dst + 1] = green;
                    dst_row[dst + 2] = blue;
                }
            }
        }

        // Reflect-101 pad columns [orig_w..pw) using already-converted pixels.
        if pw > orig_w {
            let full_row = &mut out[y * row_floats..(y + 1) * row_floats];
            for x in orig_w..pw {
                let sx = crate::reflect_index(x, orig_w) * 3;
                let dx = x * 3;
                full_row[dx] = full_row[sx];
                full_row[dx + 1] = full_row[sx + 1];
                full_row[dx + 2] = full_row[sx + 2];
            }
        }
    }

    // Reflect-101 pad rows [orig_h..ph) by copying already-converted rows.
    for y in orig_h..ph {
        if y.is_multiple_of(64) {
            check_stop(latch)?;
        }
        let sy = crate::reflect_index(y, orig_h);
        let src_start = sy * row_floats;
        let dst_start = y * row_floats;
        out.copy_within(src_start..src_start + row_floats, dst_start);
    }

    check_stop(latch)?;
    Ok((out, orig_w, orig_h, pw, ph))
}
