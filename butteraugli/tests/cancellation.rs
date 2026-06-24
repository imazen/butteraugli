//! Cooperative-cancellation tests for the `*_with_stop` entry points.
//!
//! A pre-cancelled [`almost_enough::Stopper::cancelled`] token must make every
//! `*_with_stop` function bail out with [`ButteraugliError::Cancelled`] at its
//! outermost per-scale / per-strip boundary, *before* doing any per-pixel work.
//! An [`enough::Unstoppable`] token must leave the result untouched (`Ok`).

use butteraugli::{
    ButteraugliError, ButteraugliParams, ButteraugliReference, Img, RGB, RGB8,
    butteraugli_linear_strip_with_stop, butteraugli_linear_with_stop, butteraugli_strip_with_stop,
    butteraugli_with_stop,
};

/// Small deterministic sRGB test image (≥8×8 so the strip API accepts it).
fn srgb_image(width: usize, height: usize, seed: usize) -> Img<Vec<RGB8>> {
    let pixels: Vec<RGB8> = (0..width * height)
        .map(|i| {
            let v = i + seed;
            RGB8::new(
                (v % 256) as u8,
                ((v * 2) % 256) as u8,
                ((v * 3) % 256) as u8,
            )
        })
        .collect();
    Img::new(pixels, width, height)
}

/// Flat row-major RGB `u8` bytes (3 per pixel) for the warm-reference
/// `ButteraugliReference::new` / `compare*` byte-slice entry points.
fn srgb_bytes(width: usize, height: usize, seed: usize) -> Vec<u8> {
    (0..width * height)
        .flat_map(|i| {
            let v = i + seed;
            [
                (v % 256) as u8,
                ((v * 2) % 256) as u8,
                ((v * 3) % 256) as u8,
            ]
        })
        .collect()
}

/// Same content as [`srgb_image`] but as linear-RGB f32 (cheap stand-in — the
/// cancellation check fires before any opsin/scaling work runs).
fn linear_image(width: usize, height: usize, seed: usize) -> Img<Vec<RGB<f32>>> {
    let pixels: Vec<RGB<f32>> = (0..width * height)
        .map(|i| {
            let v = (i + seed) as f32;
            RGB::new(
                (v % 256.0) / 255.0,
                ((v * 2.0) % 256.0) / 255.0,
                ((v * 3.0) % 256.0) / 255.0,
            )
        })
        .collect();
    Img::new(pixels, width, height)
}

#[test]
fn butteraugli_with_stop_cancelled() {
    let a = srgb_image(16, 16, 0);
    let b = srgb_image(16, 16, 5);
    let params = ButteraugliParams::default();

    let result = butteraugli_with_stop(
        a.as_ref(),
        b.as_ref(),
        &params,
        &almost_enough::Stopper::cancelled(),
    );

    assert!(
        matches!(result, Err(ButteraugliError::Cancelled(_))),
        "expected Cancelled, got {result:?}"
    );
}

#[test]
fn butteraugli_with_stop_unstoppable_ok() {
    let a = srgb_image(16, 16, 0);
    let b = srgb_image(16, 16, 5);
    let params = ButteraugliParams::default();

    let result = butteraugli_with_stop(a.as_ref(), b.as_ref(), &params, &enough::Unstoppable)
        .expect("Unstoppable must not cancel");

    assert!(result.score.is_finite());
    // Distinct images must produce a non-zero score (sanity: real work ran).
    assert!(
        result.score > 0.0,
        "expected non-zero score, got {}",
        result.score
    );
}

#[test]
fn butteraugli_with_stop_unstoppable_matches_butteraugli() {
    // The Unstoppable path must be byte-for-byte the plain `butteraugli`.
    let a = srgb_image(16, 16, 0);
    let b = srgb_image(16, 16, 5);
    let params = ButteraugliParams::default();

    let plain = butteraugli::butteraugli(a.as_ref(), b.as_ref(), &params).unwrap();
    let stopped =
        butteraugli_with_stop(a.as_ref(), b.as_ref(), &params, &enough::Unstoppable).unwrap();

    assert_eq!(plain.score, stopped.score);
}

#[test]
fn butteraugli_linear_with_stop_cancelled() {
    let a = linear_image(16, 16, 0);
    let b = linear_image(16, 16, 5);
    let params = ButteraugliParams::default();

    let result = butteraugli_linear_with_stop(
        a.as_ref(),
        b.as_ref(),
        &params,
        &almost_enough::Stopper::cancelled(),
    );

    assert!(
        matches!(result, Err(ButteraugliError::Cancelled(_))),
        "expected Cancelled, got {result:?}"
    );
}

#[test]
fn butteraugli_linear_with_stop_unstoppable_ok() {
    let a = linear_image(16, 16, 0);
    let b = linear_image(16, 16, 5);
    let params = ButteraugliParams::default();

    let result =
        butteraugli_linear_with_stop(a.as_ref(), b.as_ref(), &params, &enough::Unstoppable)
            .expect("Unstoppable must not cancel");

    assert!(result.score.is_finite());
}

#[test]
fn butteraugli_strip_with_stop_cancelled() {
    let a = srgb_image(32, 32, 0);
    let b = srgb_image(32, 32, 5);
    let params = ButteraugliParams::default();

    let result = butteraugli_strip_with_stop(
        a.as_ref(),
        b.as_ref(),
        &params,
        16,
        &almost_enough::Stopper::cancelled(),
    );

    assert!(
        matches!(result, Err(ButteraugliError::Cancelled(_))),
        "expected Cancelled, got {result:?}"
    );
}

#[test]
fn butteraugli_strip_with_stop_unstoppable_ok() {
    let a = srgb_image(32, 32, 0);
    let b = srgb_image(32, 32, 5);
    let params = ButteraugliParams::default();

    let result =
        butteraugli_strip_with_stop(a.as_ref(), b.as_ref(), &params, 16, &enough::Unstoppable)
            .expect("Unstoppable must not cancel");

    assert!(result.score.is_finite());
}

#[test]
fn butteraugli_strip_with_stop_unstoppable_matches_plain_strip() {
    let a = srgb_image(32, 32, 0);
    let b = srgb_image(32, 32, 5);
    let params = ButteraugliParams::default();

    let plain = butteraugli::butteraugli_strip(a.as_ref(), b.as_ref(), &params, 16).unwrap();
    let stopped =
        butteraugli_strip_with_stop(a.as_ref(), b.as_ref(), &params, 16, &enough::Unstoppable)
            .unwrap();

    assert_eq!(plain.score, stopped.score);
}

#[test]
fn butteraugli_linear_strip_with_stop_cancelled() {
    let a = linear_image(32, 32, 0);
    let b = linear_image(32, 32, 5);
    let params = ButteraugliParams::default();

    let result = butteraugli_linear_strip_with_stop(
        a.as_ref(),
        b.as_ref(),
        &params,
        16,
        &almost_enough::Stopper::cancelled(),
    );

    assert!(
        matches!(result, Err(ButteraugliError::Cancelled(_))),
        "expected Cancelled, got {result:?}"
    );
}

#[test]
fn butteraugli_linear_strip_with_stop_unstoppable_ok() {
    let a = linear_image(32, 32, 0);
    let b = linear_image(32, 32, 5);
    let params = ButteraugliParams::default();

    let result = butteraugli_linear_strip_with_stop(
        a.as_ref(),
        b.as_ref(),
        &params,
        16,
        &enough::Unstoppable,
    )
    .expect("Unstoppable must not cancel");

    assert!(result.score.is_finite());
}

#[test]
fn butteraugli_linear_strip_with_stop_unstoppable_matches_plain_strip() {
    let a = linear_image(32, 32, 0);
    let b = linear_image(32, 32, 5);
    let params = ButteraugliParams::default();

    let plain = butteraugli::butteraugli_linear_strip(a.as_ref(), b.as_ref(), &params, 16).unwrap();
    let stopped = butteraugli_linear_strip_with_stop(
        a.as_ref(),
        b.as_ref(),
        &params,
        16,
        &enough::Unstoppable,
    )
    .unwrap();

    assert_eq!(plain.score, stopped.score);
}

// ----------------------------------------------------------------------------
// Warm-reference (`ButteraugliReference`) batch path.
// ----------------------------------------------------------------------------

#[test]
fn reference_compare_with_stop_cancelled() {
    let reference =
        ButteraugliReference::new(&srgb_bytes(16, 16, 0), 16, 16, ButteraugliParams::default())
            .unwrap();
    let dist = srgb_bytes(16, 16, 5);

    let result = reference.compare_with_stop(&dist, &almost_enough::Stopper::cancelled());

    assert!(
        matches!(result, Err(ButteraugliError::Cancelled(_))),
        "expected Cancelled, got {result:?}"
    );
}

#[test]
fn reference_compare_with_stop_unstoppable_ok() {
    let reference =
        ButteraugliReference::new(&srgb_bytes(16, 16, 0), 16, 16, ButteraugliParams::default())
            .unwrap();
    let dist = srgb_bytes(16, 16, 5);

    let result = reference
        .compare_with_stop(&dist, &enough::Unstoppable)
        .expect("Unstoppable must not cancel");

    assert!(result.score.is_finite());
    // Distinct images must produce a non-zero score (sanity: real work ran).
    assert!(
        result.score > 0.0,
        "expected non-zero score, got {}",
        result.score
    );
}

#[test]
fn reference_compare_with_stop_unstoppable_matches_compare() {
    // The Unstoppable warm-ref path must be byte-for-byte the plain `compare`.
    let reference =
        ButteraugliReference::new(&srgb_bytes(16, 16, 0), 16, 16, ButteraugliParams::default())
            .unwrap();
    let dist = srgb_bytes(16, 16, 5);

    let plain = reference.compare(&dist).unwrap();
    let stopped = reference
        .compare_with_stop(&dist, &enough::Unstoppable)
        .unwrap();

    assert_eq!(plain.score, stopped.score);
}

#[test]
fn reference_compare_strip_with_stop_cancelled() {
    let reference =
        ButteraugliReference::new(&srgb_bytes(32, 32, 0), 32, 32, ButteraugliParams::default())
            .unwrap();
    let dist = srgb_bytes(32, 32, 5);

    let result = reference.compare_strip_with_stop(&dist, 16, &almost_enough::Stopper::cancelled());

    assert!(
        matches!(result, Err(ButteraugliError::Cancelled(_))),
        "expected Cancelled, got {result:?}"
    );
}

#[test]
fn reference_compare_strip_with_stop_unstoppable_matches_plain() {
    let reference =
        ButteraugliReference::new(&srgb_bytes(32, 32, 0), 32, 32, ButteraugliParams::default())
            .unwrap();
    let dist = srgb_bytes(32, 32, 5);

    let plain = reference.compare_strip(&dist, 16).unwrap();
    let stopped = reference
        .compare_strip_with_stop(&dist, 16, &enough::Unstoppable)
        .expect("Unstoppable must not cancel");

    assert_eq!(plain.score, stopped.score);
}
