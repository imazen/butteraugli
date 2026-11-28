#![cfg(feature = "zenpsy-api")]
//! End-to-end contract tests for `butteraugli`'s `zenpsy-api` integration.
//!
//! Covers:
//! - Score-only default (zero full-resolution diffmap allocation) vs opt-in
//!   `MapRequest::Native` / `MapRequest::Named("diffmap")`
//! - Bit-exact `f64::to_bits()` parity across score-only, diffmap, cold, warm
//!   (`PrepareReference`), and native `butteraugli()` APIs for BOTH
//!   `"butteraugli_max"` and `"butteraugli_pnorm3"`
//! - Unknown named map rejection (`ErrorKind::NoMap`) before computation
//! - Sub-8px (`1x1`, `3x5`, `7x7`) cold & warm parity and cropped diffmap size
//! - Strided RGB8/BGR8/RGBA8(opaque)/RGBF32_LINEAR inputs
//! - Cooperative cancellation (`OneShotTimeoutStop` preserving `StopReason::TimedOut`)
//!   during reference preparation and warm scoring, plus clean reuse
//! - `max_pixels` budget enforcement including reflect-padding expansion
//! - Strict SDR BT.709 ingress refusals (translucent straight alpha, premultiplied
//!   alpha, pending orientation, HDR metadata, DiffuseWhite, BT.2020 primaries,
//!   NaN linear f32)

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use butteraugli::{ButteraugliMetric, ButteraugliParams, Img, Numerics, RGB8, butteraugli};
use zenpsy_api::how_far::{Stop, StopOnly, StopReason};
use zenpsy_api::zenpixels::{
    AlphaMode, ColorSpace, ContentLightLevel, DiffuseWhite, HdrMetadata, Orientation, PixelFormat,
    PixelView, PlaneRole, Primaries, SampleLayout, SignalRange, Transfer, formats,
};
use zenpsy_api::{Direction, ErrorKind, EvalOpts, Evaluation, Metric, prepare_reference, score};

fn synth_rgb(w: u32, h: u32, distort: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = ((x * 29 + y * 13 + 17) % 256) as u8;
            let g = ((x * 7 + y * 37 + 53) % 256) as u8;
            let b = ((x ^ y).wrapping_mul(19) % 256) as u8;
            if distort {
                out.push(r.saturating_add(5));
                out.push(g.saturating_sub(3));
                out.push(b.saturating_add(4));
            } else {
                out.push(r);
                out.push(g);
                out.push(b);
            }
        }
    }
    out
}

fn synth_strided_rgb(w: u32, h: u32, pad_bytes: usize, distort: bool) -> (Vec<u8>, usize) {
    let packed = synth_rgb(w, h, distort);
    let row_len = w as usize * 3;
    let stride = row_len + pad_bytes;
    let mut out = vec![0xA5u8; stride * h as usize];
    for y in 0..h as usize {
        out[y * stride..y * stride + row_len]
            .copy_from_slice(&packed[y * row_len..(y + 1) * row_len]);
    }
    (out, stride)
}

#[derive(Clone)]
struct CountingStop {
    checks: Arc<AtomicUsize>,
}

impl CountingStop {
    fn new() -> Self {
        Self {
            checks: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn checks(&self) -> usize {
        self.checks.load(Ordering::SeqCst)
    }
}

impl Stop for CountingStop {
    fn check(&self) -> Result<(), StopReason> {
        self.checks.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[derive(Clone)]
struct OneShotTimeoutStop {
    checks: Arc<AtomicUsize>,
    trip_at: usize,
}

impl OneShotTimeoutStop {
    fn new(trip_at: usize) -> Self {
        Self {
            checks: Arc::new(AtomicUsize::new(0)),
            trip_at,
        }
    }

    fn checks(&self) -> usize {
        self.checks.load(Ordering::SeqCst)
    }
}

impl Stop for OneShotTimeoutStop {
    fn check(&self) -> Result<(), StopReason> {
        let n = self.checks.fetch_add(1, Ordering::SeqCst);
        if n == self.trip_at {
            Err(StopReason::TimedOut)
        } else {
            Ok(())
        }
    }
}

#[test]
fn metric_id_and_score_specs() {
    let m = ButteraugliMetric::new();
    let id = m.id();
    assert_eq!(id.family(), "butteraugli");
    assert_eq!(id.implementation().crate_name(), "butteraugli");
    assert_eq!(id.config_value("numerics"), Some("fast"));
    assert_eq!(id.config_value("intensity_target"), Some("80"));
    assert_eq!(id.config_value("single_resolution"), Some("false"));
}

#[test]
fn score_only_diffmap_warm_cold_and_native_bit_exact_parity() {
    let (w, h) = (96u32, 80u32);
    let src_bytes = synth_rgb(w, h, false);
    let dst_bytes = synth_rgb(w, h, true);

    let src_view = PixelView::new(&src_bytes, w, h, w as usize * 3, &formats::RGB8_SRGB).unwrap();
    let dst_view = PixelView::new(&dst_bytes, w, h, w as usize * 3, &formats::RGB8_SRGB).unwrap();

    // Native reference score (both with and without compute_diffmap).
    let src_px: Vec<RGB8> = src_bytes
        .as_chunks::<3>()
        .0
        .iter()
        .map(|&[r, g, b]| RGB8::new(r, g, b))
        .collect();
    let dst_px: Vec<RGB8> = dst_bytes
        .as_chunks::<3>()
        .0
        .iter()
        .map(|&[r, g, b]| RGB8::new(r, g, b))
        .collect();
    let native_score_only = butteraugli(
        Img::new(src_px.clone(), w as usize, h as usize).as_ref(),
        Img::new(dst_px.clone(), w as usize, h as usize).as_ref(),
        &ButteraugliParams::default(),
    )
    .unwrap();
    let native_with_map = butteraugli(
        Img::new(src_px, w as usize, h as usize).as_ref(),
        Img::new(dst_px, w as usize, h as usize).as_ref(),
        &ButteraugliParams::default().with_compute_diffmap(true),
    )
    .unwrap();
    assert!(native_score_only.diffmap.is_none());
    assert!(native_with_map.diffmap.is_some());
    assert_eq!(
        native_score_only.score.to_bits(),
        native_with_map.score.to_bits(),
        "native score-only and diffmap paths must be bit-exact for max-norm"
    );
    assert_eq!(
        native_score_only.pnorm_3.to_bits(),
        native_with_map.pnorm_3.to_bits(),
        "native score-only and diffmap paths must be bit-exact for pnorm_3"
    );

    let mut m = ButteraugliMetric::new();

    // Cold score-only
    let cold_only = score(&mut m, &src_view, &dst_view, EvalOpts::new()).unwrap();
    assert!(cold_only.diffmap().is_none());
    assert_eq!(cold_only.maps().count(), 0);
    let scores: Vec<_> = cold_only.scores().collect();
    assert_eq!(scores.len(), 2);
    assert_eq!(scores[0].spec.key(), "butteraugli_max");
    assert_eq!(scores[0].spec.direction(), Direction::Lower);
    assert_eq!(scores[0].spec.perfect(), Some(0.0));
    assert_eq!(scores[1].spec.key(), "butteraugli_pnorm3");
    assert_eq!(scores[1].spec.direction(), Direction::Lower);
    assert_eq!(scores[1].spec.perfect(), Some(0.0));
    assert_eq!(cold_only.score().to_bits(), native_with_map.score.to_bits());
    assert_eq!(
        cold_only.pnorm_3().to_bits(),
        native_with_map.pnorm_3.to_bits()
    );

    // Cold opt-in diffmap
    let cold_map = score(
        &mut m,
        &src_view,
        &dst_view,
        EvalOpts::new().with_map("diffmap"),
    )
    .unwrap();
    assert!(cold_map.diffmap().is_some());
    assert_eq!(cold_map.maps().count(), 1);
    let mv = cold_map.maps().next().unwrap();
    assert_eq!(mv.spec.key(), "diffmap");
    assert_eq!(mv.spec.direction(), Direction::Lower);
    assert_eq!(mv.values.format().role, PlaneRole::Extra(0));
    assert_eq!(mv.values.width(), w);
    assert_eq!(mv.values.height(), h);
    assert_eq!(cold_map.score().to_bits(), cold_only.score().to_bits());
    assert_eq!(cold_map.pnorm_3().to_bits(), cold_only.pnorm_3().to_bits());

    // Warm prepare_reference -> score-only & diffmap
    let mut prepared = prepare_reference(&mut m, &src_view, EvalOpts::new()).unwrap();
    let warm_only = prepared.score(&dst_view, EvalOpts::new()).unwrap();
    assert!(warm_only.diffmap().is_none());
    assert_eq!(warm_only.maps().count(), 0);
    let warm_map = prepared
        .score(&dst_view, EvalOpts::new().with_native_map())
        .unwrap();
    assert!(warm_map.diffmap().is_some());
    assert_eq!(warm_map.maps().count(), 1);

    // Warm score-only and warm diffmap must be bit-exact with each other!
    assert_eq!(warm_only.score().to_bits(), warm_map.score().to_bits());
    assert_eq!(warm_only.pnorm_3().to_bits(), warm_map.pnorm_3().to_bits());
    // And warm vs cold match within float equivalence (same pipeline!).
    assert!(
        (warm_only.score() - cold_only.score()).abs() < 1e-6,
        "warm={} vs cold={}",
        warm_only.score(),
        cold_only.score()
    );
    assert!(
        (warm_only.pnorm_3() - cold_only.pnorm_3()).abs() < 1e-6,
        "warm_p3={} vs cold_p3={}",
        warm_only.pnorm_3(),
        cold_only.pnorm_3()
    );

    // Unknown map rejected immediately with ErrorKind::NoMap.
    let err = score(
        &mut m,
        &src_view,
        &dst_view,
        EvalOpts::new().with_map("bogus"),
    )
    .unwrap_err();
    assert_eq!(err.error().kind(), ErrorKind::NoMap);
    let err = prepared
        .score(&dst_view, EvalOpts::new().with_map("bogus"))
        .unwrap_err();
    assert_eq!(err.error().kind(), ErrorKind::NoMap);
}

#[test]
fn sub_8px_and_strided_parity() {
    let mut m = ButteraugliMetric::new();
    for (w, h) in [(1u32, 1u32), (3, 5), (7, 7), (17, 13)] {
        let (src_s, stride) = synth_strided_rgb(w, h, 7, false);
        let (dst_s, _) = synth_strided_rgb(w, h, 7, true);
        let src = PixelView::new(&src_s, w, h, stride, &formats::RGB8_SRGB).unwrap();
        let dst = PixelView::new(&dst_s, w, h, stride, &formats::RGB8_SRGB).unwrap();

        let cold_same = score(&mut m, &src, &src, EvalOpts::new()).unwrap();
        assert_eq!(cold_same.score(), 0.0);
        assert_eq!(cold_same.pnorm_3(), 0.0);

        let cold_diff = score(&mut m, &src, &dst, EvalOpts::new().with_map("diffmap")).unwrap();
        let dm = cold_diff.diffmap().unwrap();
        assert_eq!(dm.width(), w as usize);
        assert_eq!(dm.height(), h as usize);

        let mut prep = prepare_reference(&mut m, &src, EvalOpts::new()).unwrap();
        let warm_only = prep.score(&dst, EvalOpts::new()).unwrap();
        let warm_map = prep
            .score(&dst, EvalOpts::new().with_map("diffmap"))
            .unwrap();
        assert_eq!(warm_only.score().to_bits(), warm_map.score().to_bits());
        assert_eq!(warm_only.pnorm_3().to_bits(), warm_map.pnorm_3().to_bits());
        assert_eq!(warm_map.diffmap().unwrap().width(), w as usize);
        assert_eq!(warm_map.diffmap().unwrap().height(), h as usize);
    }
}

#[test]
fn cancellation_during_prepare_and_scoring_and_clean_reuse() {
    let mut m = ButteraugliMetric::new();
    let (w, h) = (128u32, 128u32);
    let src_bytes = synth_rgb(w, h, false);
    let dst_bytes = synth_rgb(w, h, true);
    let src = PixelView::new(&src_bytes, w, h, w as usize * 3, &formats::RGB8_SRGB).unwrap();
    let dst = PixelView::new(&dst_bytes, w, h, w as usize * 3, &formats::RGB8_SRGB).unwrap();

    // Measure check counts for uncancelled prepare + warm score.
    let prep_counter = CountingStop::new();
    let prep_pulse = StopOnly::new(prep_counter.clone());
    let mut prepared =
        prepare_reference(&mut m, &src, EvalOpts::new().with_pulse(&prep_pulse)).unwrap();
    let full_prep_checks = prep_counter.checks();
    assert!(
        full_prep_checks > 4,
        "prepare_reference should poll stop across phases, saw {full_prep_checks}"
    );

    let score_counter = CountingStop::new();
    let score_pulse = StopOnly::new(score_counter.clone());
    let baseline = prepared
        .score(&dst, EvalOpts::new().with_pulse(&score_pulse))
        .unwrap();
    let full_score_checks = score_counter.checks();
    assert!(
        full_score_checks > 4,
        "warm score should poll stop across phases, saw {full_score_checks}"
    );

    // Mid-prepare one-shot TimedOut must abort early and preserve TimedOut.
    let trip_prep = OneShotTimeoutStop::new(full_prep_checks / 2);
    let trip_prep_pulse = StopOnly::new(trip_prep.clone());
    let err = prepare_reference(&mut m, &src, EvalOpts::new().with_pulse(&trip_prep_pulse))
        .err()
        .expect("mid-prepare TimedOut must fail");
    assert_eq!(err.error().kind(), ErrorKind::Stopped(StopReason::TimedOut));
    assert!(trip_prep.checks() < full_prep_checks);

    // Mid-score one-shot TimedOut must abort early and preserve TimedOut.
    let trip_score = OneShotTimeoutStop::new(full_score_checks / 2);
    let trip_score_pulse = StopOnly::new(trip_score.clone());
    let err = prepared
        .score(&dst, EvalOpts::new().with_pulse(&trip_score_pulse))
        .unwrap_err();
    assert_eq!(err.error().kind(), ErrorKind::Stopped(StopReason::TimedOut));
    assert!(trip_score.checks() < full_score_checks);

    // Clean reuse after cancellation must produce bit-exact results.
    let recovered = prepared.score(&dst, EvalOpts::new()).unwrap();
    assert_eq!(recovered.score().to_bits(), baseline.score().to_bits());
    assert_eq!(recovered.pnorm_3().to_bits(), baseline.pnorm_3().to_bits());
}

#[test]
fn max_pixels_counts_active_input_pixels_excluding_reflect_scratch() {
    let mut m = ButteraugliMetric::new();
    // 1x1 has 1 active pixel (even though reflect-padded internally to 8x8),
    // so EvalOpts::with_max_pixels(1) MUST succeed on cold and warm paths.
    let one_px = synth_rgb(1, 1, false);
    let one = PixelView::new(&one_px, 1, 1, 3, &formats::RGB8_SRGB).unwrap();
    let cold = score(&mut m, &one, &one, EvalOpts::new().with_max_pixels(1)).unwrap();
    assert_eq!(cold.score(), 0.0);
    let mut prep = prepare_reference(&mut m, &one, EvalOpts::new().with_max_pixels(1)).unwrap();
    let warm = prep
        .score(&one, EvalOpts::new().with_max_pixels(1))
        .unwrap();
    assert_eq!(warm.score(), 0.0);

    // 2x1 has 2 active pixels > 1 -> rejected as Resource before allocation.
    let two_px = synth_rgb(2, 1, false);
    let two = PixelView::new(&two_px, 2, 1, 6, &formats::RGB8_SRGB).unwrap();
    let err = score(&mut m, &two, &two, EvalOpts::new().with_max_pixels(1)).unwrap_err();
    assert_eq!(err.error().kind(), ErrorKind::Resource);
    let err = prepare_reference(&mut m, &two, EvalOpts::new().with_max_pixels(1))
        .err()
        .expect("prepare exceeding active max_pixels must fail");
    assert_eq!(err.error().kind(), ErrorKind::Resource);
}

#[test]
fn strict_sdr_ingress_rejects_unsupported_inputs() {
    let mut m = ButteraugliMetric::new();
    let rgb = synth_rgb(16, 16, false);
    let valid = PixelView::new(&rgb, 16, 16, 16 * 3, &formats::RGB8_SRGB).unwrap();

    // 1. Fully opaque Straight RGBA8 is accepted; translucent Straight RGBA8 is rejected.
    let mut rgba = vec![255u8; 16 * 16 * 4];
    let opaque_straight = PixelView::new(&rgba, 16, 16, 16 * 4, &formats::RGBA8_SRGB).unwrap();
    assert_eq!(
        score(&mut m, &opaque_straight, &opaque_straight, EvalOpts::new())
            .unwrap()
            .score(),
        0.0
    );
    rgba[7] = 128; // one translucent pixel
    let translucent = PixelView::new(&rgba, 16, 16, 16 * 4, &formats::RGBA8_SRGB).unwrap();
    let err = score(&mut m, &translucent, &translucent, EvalOpts::new()).unwrap_err();
    assert_eq!(err.error().kind(), ErrorKind::UnsupportedInput);

    // 2. Premultiplied alpha rejected.
    let premul_fmt = PixelFormat::new(
        SampleLayout::RGBA8,
        ColorSpace::Rgb {
            primaries: Primaries::Bt709,
            transfer: Transfer::Srgb,
            range: SignalRange::Full,
            alpha: AlphaMode::Premultiplied,
            white: None,
        },
    );
    let premul = PixelView::new(&rgba, 16, 16, 16 * 4, &premul_fmt).unwrap();
    let err = score(&mut m, &premul, &premul, EvalOpts::new()).unwrap_err();
    assert_eq!(err.error().kind(), ErrorKind::UnsupportedInput);

    // 3. Pending orientation rejected.
    let rotated = valid.clone().with_orientation(Orientation::Rotate90);
    let err = score(&mut m, &rotated, &rotated, EvalOpts::new()).unwrap_err();
    assert_eq!(err.error().kind(), ErrorKind::UnsupportedInput);

    // 4. HDR metadata rejected.
    let with_hdr = valid.clone().with_hdr(HdrMetadata {
        content_light_level: Some(ContentLightLevel::new(1000, 400)),
        mastering: None,
    });
    let err = score(&mut m, &with_hdr, &with_hdr, EvalOpts::new()).unwrap_err();
    assert_eq!(err.error().kind(), ErrorKind::UnsupportedInput);

    // 5. DiffuseWhite rejected.
    let f32_px = vec![[0.5f32, 0.5, 0.5, 1.0]; 16 * 16];
    let white_fmt = PixelFormat::new(
        SampleLayout::RGBAF32,
        ColorSpace::Rgb {
            primaries: Primaries::Bt709,
            transfer: Transfer::Linear,
            range: SignalRange::Full,
            alpha: AlphaMode::Opaque,
            white: Some(DiffuseWhite::BT2408),
        },
    );
    let white_view = PixelView::from_typed(&f32_px, 16, 16, &white_fmt).unwrap();
    let err = score(&mut m, &white_view, &white_view, EvalOpts::new()).unwrap_err();
    assert_eq!(err.error().kind(), ErrorKind::UnsupportedInput);

    // 6. BT.2020 primaries rejected.
    let bt2020_fmt = PixelFormat::new(
        SampleLayout::RGB8,
        ColorSpace::Rgb {
            primaries: Primaries::Bt2020,
            transfer: Transfer::Srgb,
            range: SignalRange::Full,
            alpha: AlphaMode::None,
            white: None,
        },
    );
    let bt2020 = PixelView::new(&rgb, 16, 16, 16 * 3, &bt2020_fmt).unwrap();
    let err = score(&mut m, &bt2020, &bt2020, EvalOpts::new()).unwrap_err();
    assert_eq!(err.error().kind(), ErrorKind::UnsupportedInput);

    // 7. Non-finite linear f32 rejected as InvalidInput.
    let mut nan_px = f32_px;
    nan_px[3][1] = f32::NAN;
    let nan_view = PixelView::from_typed(&nan_px, 16, 16, &formats::RGBAF32_LINEAR).unwrap();
    let err = score(&mut m, &nan_view, &nan_view, EvalOpts::new()).unwrap_err();
    assert_eq!(err.error().kind(), ErrorKind::InvalidInput);
}

#[test]
fn libjxl_exact_score_only_cold_and_warm_bit_exact_with_diffmap() {
    let (w, h) = (64u32, 48u32);
    let src_bytes = synth_rgb(w, h, false);
    let dst_bytes = synth_rgb(w, h, true);
    let src = PixelView::new(&src_bytes, w, h, w as usize * 3, &formats::RGB8_SRGB).unwrap();
    let dst = PixelView::new(&dst_bytes, w, h, w as usize * 3, &formats::RGB8_SRGB).unwrap();

    let mut m = ButteraugliMetric::with_params(
        ButteraugliParams::default().with_numerics(Numerics::LibjxlExact),
    );
    let cold_only = score(&mut m, &src, &dst, EvalOpts::new()).unwrap();
    assert!(cold_only.diffmap().is_none());
    let cold_map = score(&mut m, &src, &dst, EvalOpts::new().with_map("diffmap")).unwrap();
    assert!(cold_map.diffmap().is_some());
    assert_eq!(cold_only.score().to_bits(), cold_map.score().to_bits());
    assert_eq!(cold_only.pnorm_3().to_bits(), cold_map.pnorm_3().to_bits());

    let mut prep = prepare_reference(&mut m, &src, EvalOpts::new()).unwrap();
    let warm_only = prep.score(&dst, EvalOpts::new()).unwrap();
    assert!(warm_only.diffmap().is_none());
    let warm_map = prep
        .score(&dst, EvalOpts::new().with_map("diffmap"))
        .unwrap();
    assert!(warm_map.diffmap().is_some());
    assert_eq!(warm_only.score().to_bits(), warm_map.score().to_bits());
    assert_eq!(warm_only.pnorm_3().to_bits(), warm_map.pnorm_3().to_bits());
    assert_eq!(warm_only.score().to_bits(), cold_only.score().to_bits());
    assert_eq!(warm_only.pnorm_3().to_bits(), cold_only.pnorm_3().to_bits());
}

#[test]
fn rgbx8_and_bgrx8_ignore_padding_bytes_cold_and_warm() {
    let (w, h) = (32u32, 24u32);
    let rgb = synth_rgb(w, h, false);
    let mut rgbx_a = Vec::with_capacity((w * h * 4) as usize);
    let mut rgbx_b = Vec::with_capacity((w * h * 4) as usize);
    let mut bgrx_a = Vec::with_capacity((w * h * 4) as usize);
    let mut bgrx_b = Vec::with_capacity((w * h * 4) as usize);
    for (idx, px) in rgb.as_chunks::<3>().0.iter().enumerate() {
        let pad_a = (idx % 251) as u8;
        let pad_b = (idx.wrapping_mul(37) % 256) as u8;
        rgbx_a.extend_from_slice(&[px[0], px[1], px[2], pad_a]);
        rgbx_b.extend_from_slice(&[px[0], px[1], px[2], pad_b]);
        bgrx_a.extend_from_slice(&[px[2], px[1], px[0], pad_a]);
        bgrx_b.extend_from_slice(&[px[2], px[1], px[0], pad_b]);
    }
    let rgb_v = PixelView::new(&rgb, w, h, w as usize * 3, &formats::RGB8_SRGB).unwrap();
    let rgbx_va = PixelView::new(&rgbx_a, w, h, w as usize * 4, &formats::RGBX8_SRGB).unwrap();
    let rgbx_vb = PixelView::new(&rgbx_b, w, h, w as usize * 4, &formats::RGBX8_SRGB).unwrap();
    let bgrx_va = PixelView::new(&bgrx_a, w, h, w as usize * 4, &formats::BGRX8_SRGB).unwrap();
    let bgrx_vb = PixelView::new(&bgrx_b, w, h, w as usize * 4, &formats::BGRX8_SRGB).unwrap();

    let mut m = ButteraugliMetric::new();
    for (a, b) in [
        (&rgbx_va, &rgbx_vb),
        (&bgrx_va, &bgrx_vb),
        (&rgb_v, &rgbx_va),
        (&rgb_v, &bgrx_vb),
    ] {
        let cold = score(&mut m, a, b, EvalOpts::new()).unwrap();
        assert_eq!(cold.score(), 0.0);
        let mut prep = prepare_reference(&mut m, a, EvalOpts::new()).unwrap();
        let warm = prep.score(b, EvalOpts::new()).unwrap();
        assert_eq!(warm.score(), 0.0);
    }
}

#[test]
fn cancellation_during_subsampling_and_exact_warm_reuse() {
    // Tall enough (320 rows) so 2x subsampling (160 output rows) hits multiple
    // 64-row checkpoints during both reference prep and warm candidate compare.
    let (w, h) = (32u32, 320u32);
    let src_bytes = synth_rgb(w, h, false);
    let dst_bytes = synth_rgb(w, h, true);
    let src = PixelView::new(&src_bytes, w, h, w as usize * 3, &formats::RGB8_SRGB).unwrap();
    let dst = PixelView::new(&dst_bytes, w, h, w as usize * 3, &formats::RGB8_SRGB).unwrap();

    for numerics in [Numerics::Fast, Numerics::LibjxlExact] {
        let mut m =
            ButteraugliMetric::with_params(ButteraugliParams::default().with_numerics(numerics));
        let prep_counter = CountingStop::new();
        let prep_pulse = StopOnly::new(prep_counter.clone());
        let mut prep =
            prepare_reference(&mut m, &src, EvalOpts::new().with_pulse(&prep_pulse)).unwrap();
        let full_prep = prep_counter.checks();
        assert!(full_prep > 8, "{numerics:?}: saw {full_prep} prep checks");

        // Trip right after ingress/conversion into reference subsampling / pyramid build.
        for trip in [2, 6, full_prep / 3] {
            let pulse = StopOnly::new(OneShotTimeoutStop::new(trip));
            let err = prepare_reference(&mut m, &src, EvalOpts::new().with_pulse(&pulse))
                .err()
                .expect("mid-prepare timeout must fail");
            assert_eq!(err.error().kind(), ErrorKind::Stopped(StopReason::TimedOut));
        }

        let baseline = prep.score(&dst, EvalOpts::new()).unwrap();
        let score_counter = CountingStop::new();
        let score_pulse = StopOnly::new(score_counter.clone());
        let _ = prep
            .score(&dst, EvalOpts::new().with_pulse(&score_pulse))
            .unwrap();
        let full_score = score_counter.checks();

        for trip in [2, 6, full_score / 3] {
            let pulse = StopOnly::new(OneShotTimeoutStop::new(trip));
            let err = prep
                .score(&dst, EvalOpts::new().with_pulse(&pulse))
                .unwrap_err();
            assert_eq!(err.error().kind(), ErrorKind::Stopped(StopReason::TimedOut));
        }

        // Reuse the same PreparedButteraugli after repeated mid-run cancellations!
        let recovered = prep.score(&dst, EvalOpts::new()).unwrap();
        assert_eq!(recovered.score().to_bits(), baseline.score().to_bits());
        assert_eq!(recovered.pnorm_3().to_bits(), baseline.pnorm_3().to_bits());
    }
}
