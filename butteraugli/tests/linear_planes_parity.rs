//! `linear-planes` ↔ default-API parity.
//!
//! The `linear_planes` module adds no arithmetic: every mode forwards to an
//! existing code path with identical arguments. These tests pin that. They
//! compare with `assert_eq!` on `f64` — **not** a tolerance — because the
//! claim is bit-identity, not approximate agreement. If a future refactor
//! makes the wrapper take a different path, these fail loudly instead of
//! drifting inside a tolerance band.
//!
//! Run: `cargo test -p butteraugli --features linear-planes --test linear_planes_parity`

#![cfg(feature = "linear-planes")]
// Exact `f64` equality is the assertion this file makes: the `linear_planes`
// wrapper must forward to the existing code paths without perturbing a single
// bit. A tolerance would let a real divergence hide inside it.
#![allow(clippy::float_cmp)]

use butteraugli::linear_planes::{
    LinearPlanes, Resolution, Scorer, ScorerBuilder, StripMode, Walk,
};
use butteraugli::{
    ButteraugliParams, ButteraugliReference, ButteraugliStripConfig, Img, RGB,
    butteraugli_linear_strip_with_config,
};

/// One image expressed as an interleaved `RGB<f32>` pair, for comparing
/// against the interleaved strip free functions.
type InterleavedPair = (Img<Vec<RGB<f32>>>, Img<Vec<RGB<f32>>>);

/// Deterministic pseudo-image with structure at several frequencies, so the
/// Malta / mask / multi-scale stages all contribute.
fn plane(w: usize, h: usize, stride: usize, seed: u32) -> Vec<f32> {
    let mut v = vec![0.0f32; stride * h];
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    for y in 0..h {
        for x in 0..w {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = ((state >> 16) & 0xff) as f32 / 255.0;
            let ramp = (x as f32 / w as f32) * 0.5;
            let checker = if (x / 8 + y / 8) % 2 == 0 { 0.15 } else { 0.0 };
            v[y * stride + x] = (ramp + checker + noise * 0.2).clamp(0.0, 1.0);
        }
    }
    v
}

struct Pair {
    w: usize,
    h: usize,
    stride: usize,
    r_ref: Vec<f32>,
    g_ref: Vec<f32>,
    b_ref: Vec<f32>,
    r_dis: Vec<f32>,
    g_dis: Vec<f32>,
    b_dis: Vec<f32>,
}

impl Pair {
    fn new(w: usize, h: usize, stride: usize) -> Self {
        Self {
            w,
            h,
            stride,
            r_ref: plane(w, h, stride, 1),
            g_ref: plane(w, h, stride, 2),
            b_ref: plane(w, h, stride, 3),
            r_dis: plane(w, h, stride, 11),
            g_dis: plane(w, h, stride, 12),
            b_dis: plane(w, h, stride, 13),
        }
    }

    fn reference(&self) -> LinearPlanes<'_> {
        LinearPlanes::with_stride(
            &self.r_ref,
            &self.g_ref,
            &self.b_ref,
            self.w,
            self.h,
            self.stride,
        )
        .unwrap()
    }

    fn distorted(&self) -> LinearPlanes<'_> {
        LinearPlanes::with_stride(
            &self.r_dis,
            &self.g_dis,
            &self.b_dis,
            self.w,
            self.h,
            self.stride,
        )
        .unwrap()
    }

    /// Interleaved `RGB<f32>` views of the used region, for the strip
    /// free-function comparison.
    fn interleaved(&self) -> InterleavedPair {
        let mut a = Vec::with_capacity(self.w * self.h);
        let mut b = Vec::with_capacity(self.w * self.h);
        for y in 0..self.h {
            for x in 0..self.w {
                let i = y * self.stride + x;
                a.push(RGB::new(self.r_ref[i], self.g_ref[i], self.b_ref[i]));
                b.push(RGB::new(self.r_dis[i], self.g_dis[i], self.b_dis[i]));
            }
        }
        (Img::new(a, self.w, self.h), Img::new(b, self.w, self.h))
    }
}

fn whole_image_case(w: usize, h: usize, stride: usize, builder: ScorerBuilder) {
    let p = Pair::new(w, h, stride);
    let params = builder.to_params();

    // Clean API.
    let scorer = builder.build(&p.reference()).unwrap();
    let new = scorer.score(&p.distorted()).unwrap();

    // Default API, same inputs.
    let reference =
        ButteraugliReference::new_linear_planar(&p.r_ref, &p.g_ref, &p.b_ref, w, h, stride, params)
            .unwrap();
    let old = reference
        .compare_linear_planar(&p.r_dis, &p.g_dis, &p.b_dis, stride)
        .unwrap();

    assert_eq!(
        new.max_norm, old.score,
        "max_norm diverged at {w}x{h} stride {stride}"
    );
    assert_eq!(
        new.pnorm_3, old.pnorm_3,
        "pnorm_3 diverged at {w}x{h} stride {stride}"
    );
    assert!(new.max_norm > 0.0, "test images must actually differ");
}

#[test]
fn whole_image_matches_reference_api_default_params() {
    whole_image_case(64, 48, 64, Scorer::builder());
}

#[test]
fn whole_image_matches_reference_api_padded_stride() {
    whole_image_case(64, 48, 80, Scorer::builder());
}

#[test]
fn whole_image_matches_reference_api_single_scale() {
    whole_image_case(
        64,
        48,
        64,
        Scorer::builder().with_resolution(Resolution::SingleScale),
    );
}

#[test]
fn whole_image_matches_reference_api_hdr_intensity_target() {
    whole_image_case(
        64,
        48,
        64,
        Scorer::builder()
            .with_intensity_target(1000.0)
            .with_hf_asymmetry(1.5)
            .with_xmul(0.8),
    );
}

#[test]
fn whole_image_diffmap_matches_reference_api_pixel_for_pixel() {
    let (w, h, stride) = (64usize, 48usize, 72usize);
    let p = Pair::new(w, h, stride);

    let scorer = Scorer::builder()
        .with_diffmap(true)
        .build(&p.reference())
        .unwrap();
    let new = scorer.score(&p.distorted()).unwrap();

    let params = ButteraugliParams::new().with_compute_diffmap(true);
    let reference =
        ButteraugliReference::new_linear_planar(&p.r_ref, &p.g_ref, &p.b_ref, w, h, stride, params)
            .unwrap();
    let old = reference
        .compare_linear_planar(&p.r_dis, &p.g_dis, &p.b_dis, stride)
        .unwrap();

    let new_dm = new.diffmap.expect("diffmap requested");
    let old_dm = old.diffmap.expect("reference API returns a diffmap");
    assert_eq!((new_dm.width(), new_dm.height()), (w, h));
    assert_eq!(new_dm.buf().as_slice(), old_dm.buf().as_slice());
}

#[test]
fn score_into_matches_reference_api_into() {
    let (w, h, stride) = (64usize, 48usize, 64usize);
    let p = Pair::new(w, h, stride);

    let scorer = Scorer::builder()
        .with_diffmap(true)
        .build(&p.reference())
        .unwrap();
    let mut new_buf = Vec::new();
    let new = scorer.score_into(&p.distorted(), &mut new_buf).unwrap();
    assert!(
        new.diffmap.is_none(),
        "score_into writes into the caller buffer"
    );

    let reference = ButteraugliReference::new_linear_planar(
        &p.r_ref,
        &p.g_ref,
        &p.b_ref,
        w,
        h,
        stride,
        ButteraugliParams::new().with_compute_diffmap(true),
    )
    .unwrap();
    let mut old_buf = Vec::new();
    let (old_max, old_pnorm) = reference
        .compare_linear_planar_into(&p.r_dis, &p.g_dis, &p.b_dis, stride, &mut old_buf)
        .unwrap();

    assert_eq!(new.max_norm, old_max);
    assert_eq!(new.pnorm_3, old_pnorm);
    assert_eq!(new_buf, old_buf);
}

#[test]
fn diffmap_is_absent_unless_requested() {
    let p = Pair::new(64, 48, 64);
    let scorer = Scorer::new(&p.reference()).unwrap();
    let s = scorer.score(&p.distorted()).unwrap();
    assert!(s.diffmap.is_none());
    assert!(s.max_norm > 0.0 && s.pnorm_3 > 0.0);
}

fn strip_case(w: usize, h: usize, mode: StripMode, builder: ScorerBuilder) {
    let p = Pair::new(w, h, w);
    let params = builder.to_params();

    let scorer = builder
        .with_walk(Walk::Strip(mode))
        .build(&p.reference())
        .unwrap();
    let new = scorer.score(&p.distorted()).unwrap();

    let (a, b) = p.interleaved();
    let old = butteraugli_linear_strip_with_config(
        a.as_ref(),
        b.as_ref(),
        &params,
        mode.rows(),
        ButteraugliStripConfig::from(mode),
    )
    .unwrap();

    assert_eq!(
        new.max_norm, old.score,
        "strip max_norm diverged at {w}x{h}"
    );
    assert_eq!(
        new.pnorm_3, old.pnorm_3,
        "strip pnorm_3 diverged at {w}x{h}"
    );
    assert!(new.max_norm > 0.0, "test images must actually differ");
}

#[test]
fn strip_matches_free_function_default_halo() {
    strip_case(64, 96, StripMode::new(32), Scorer::builder());
}

#[test]
fn strip_matches_free_function_custom_halo() {
    strip_case(
        64,
        96,
        StripMode::new(16).with_halo_rows(24),
        Scorer::builder(),
    );
}

#[test]
fn strip_matches_free_function_single_scale() {
    strip_case(
        64,
        96,
        StripMode::new(32),
        Scorer::builder().with_resolution(Resolution::SingleScale),
    );
}

#[test]
fn strip_diffmap_matches_free_function() {
    let (w, h) = (64usize, 96usize);
    let mode = StripMode::new(32);
    let p = Pair::new(w, h, w);

    let scorer = Scorer::builder()
        .with_diffmap(true)
        .with_walk(Walk::Strip(mode))
        .build(&p.reference())
        .unwrap();
    let new = scorer.score(&p.distorted()).unwrap();

    let (a, b) = p.interleaved();
    let old = butteraugli_linear_strip_with_config(
        a.as_ref(),
        b.as_ref(),
        &ButteraugliParams::new().with_compute_diffmap(true),
        mode.rows(),
        ButteraugliStripConfig::from(mode),
    )
    .unwrap();

    let new_dm = new.diffmap.expect("diffmap requested");
    let old_dm = old.diffmap.expect("strip walker returns a diffmap");
    assert_eq!(new_dm.buf().as_slice(), old_dm.buf().as_slice());
}

#[test]
fn strip_reference_bytes_is_the_retained_interleaved_copy() {
    let (w, h) = (64usize, 96usize);
    let p = Pair::new(w, h, w);
    let scorer = Scorer::builder()
        .with_walk(Walk::Strip(StripMode::new(32)))
        .build(&p.reference())
        .unwrap();
    assert_eq!(scorer.reference_bytes(), w * h * 3 * size_of::<f32>());
}

#[test]
fn whole_image_reference_bytes_is_the_precompute() {
    let (w, h) = (64usize, 96usize);
    let p = Pair::new(w, h, w);
    let scorer = Scorer::new(&p.reference()).unwrap();
    let expected = ButteraugliReference::new_linear_planar(
        &p.r_ref,
        &p.g_ref,
        &p.b_ref,
        w,
        h,
        w,
        ButteraugliParams::default(),
    )
    .unwrap()
    .memory_bytes();
    assert_eq!(scorer.reference_bytes(), expected);
}

#[test]
fn scorer_is_reusable_across_many_distorted_images() {
    let (w, h) = (64usize, 48usize);
    let p = Pair::new(w, h, w);
    let scorer = Scorer::new(&p.reference()).unwrap();
    let first = scorer.score(&p.distorted()).unwrap();
    for _ in 0..4 {
        let again = scorer.score(&p.distorted()).unwrap();
        assert_eq!(first.max_norm, again.max_norm);
        assert_eq!(first.pnorm_3, again.pnorm_3);
    }
    // Identical planes must still score ~0 through the same warm reference.
    let same = scorer.score(&p.reference()).unwrap();
    assert!(same.max_norm < 0.01, "max_norm = {}", same.max_norm);
}

#[test]
fn cancellation_is_reported() {
    let p = Pair::new(64, 48, 64);
    let scorer = Scorer::new(&p.reference()).unwrap();
    let err = scorer
        .score_with_stop(&p.distorted(), &almost_enough::Stopper::cancelled())
        .unwrap_err();
    assert!(matches!(err, butteraugli::ButteraugliError::Cancelled(_)));
}

/// The module docs promise `Scorer` is `Send + Sync` so one scorer can serve
/// concurrent callers through `&self`. That holds only because
/// `ButteraugliReference`'s persistent `BufferPool` is a `Mutex`; pin it so a
/// future change to that type fails here instead of at a consumer's build.
#[test]
fn scorer_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Scorer>();
    assert_send_sync::<butteraugli::linear_planes::Scores>();
    assert_send_sync::<butteraugli::linear_planes::ScorerBuilder>();
}

/// One `Scorer`, many threads, `&self` scoring — same answer every time.
#[test]
fn scorer_scores_concurrently_through_shared_ref() {
    use std::sync::Arc;

    let p = Arc::new(Pair::new(64, 48, 64));
    let scorer = Arc::new(Scorer::new(&p.reference()).unwrap());
    let expected = scorer.score(&p.distorted()).unwrap().max_norm;

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let scorer = Arc::clone(&scorer);
            let p = Arc::clone(&p);
            std::thread::spawn(move || scorer.score(&p.distorted()).unwrap().max_norm)
        })
        .collect();
    for h in handles {
        assert_eq!(h.join().unwrap(), expected);
    }
}
