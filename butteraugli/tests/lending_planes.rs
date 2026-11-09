//! Native lending preserves scores/maps without a caller map export allocation.
use butteraugli::{ButteraugliParams, ButteraugliReference, Img, Numerics};
fn planes(w: usize, h: usize, strides: [usize; 3], delta: f32, padding: f32) -> [Vec<f32>; 3] {
    core::array::from_fn(|c| {
        let mut p = vec![padding; (h - 1) * strides[c] + w];
        for y in 0..h {
            for x in 0..w {
                p[y * strides[c] + x] = ((x * 31 + y * 47 + c * 17) % 211 + 13) as f32 / 255.0
                    + if (x + y) % 7 == 0 { delta } else { 0.0 };
            }
        }
        p
    })
}
#[test]
fn independently_strided_minimal_views_match_legacy_scores_and_every_map_bit() {
    for numerics in [Numerics::Fast, Numerics::LibjxlExact] {
        for (w, h) in [(8, 8), (33, 25), (97, 73)] {
            let reference = planes(w, h, [w; 3], 0.0, 0.0);
            let candidate = planes(w, h, [w; 3], 0.02, 0.0);
            let params = ButteraugliParams::default()
                .with_numerics(numerics)
                .with_compute_diffmap(true);
            let legacy = ButteraugliReference::new_linear_planar(
                &reference[0],
                &reference[1],
                &reference[2],
                w,
                h,
                w,
                params.clone(),
            )
            .unwrap();
            let expected = legacy
                .compare_linear_planar(&candidate[0], &candidate[1], &candidate[2], w)
                .unwrap();
            let strides = [w + 3, w + 5, w + 7];
            let reference = planes(w, h, strides, 0.0, f32::NAN);
            let candidate = planes(w, h, strides, 0.02, f32::NAN);
            let views =
                core::array::from_fn(|c| Img::new_stride(&reference[c][..], w, h, strides[c]));
            let prepared = ButteraugliReference::new_linear_planes(views, params).unwrap();
            for _ in 0..2 {
                prepared
                    .compare_linear_planes_with(
                        core::array::from_fn(|c| {
                            Img::new_stride(&candidate[c][..], w, h, strides[c])
                        }),
                        &enough::Unstoppable,
                        |score, p3, map| {
                            assert_eq!(score.to_bits(), expected.score.to_bits());
                            assert_eq!(p3.to_bits(), expected.pnorm_3.to_bits());
                            assert_eq!((map.width(), map.height()), (w, h));
                            let expected = expected.diffmap.as_ref().unwrap();
                            for (a, b) in map.rows().zip(expected.rows()) {
                                for (a, b) in a.iter().zip(b) {
                                    assert_eq!(a.to_bits(), b.to_bits());
                                }
                            }
                        },
                    )
                    .unwrap();
            }
        }
    }
}
#[test]
fn active_nonfinite_samples_are_refused_without_invoking_consumer() {
    let p = vec![0.5; 64];
    let prepared = ButteraugliReference::new_linear_planes(
        [Img::new(&p[..], 8, 8); 3],
        ButteraugliParams::default(),
    )
    .unwrap();
    let mut bad = p.clone();
    bad[7] = f32::NAN;
    assert!(
        prepared
            .compare_linear_planes_with(
                [Img::new(&bad[..], 8, 8); 3],
                &enough::Unstoppable,
                |_, _, _| panic!("invalid input")
            )
            .is_err()
    );
}

#[test]
fn cancelled_lending_never_invokes_consumer_and_reference_remains_usable() {
    let p = vec![0.5; 64];
    let views = [Img::new(&p[..], 8, 8); 3];
    let reference =
        ButteraugliReference::new_linear_planes(views, ButteraugliParams::default()).unwrap();
    let stop = almost_enough::Stopper::cancelled();
    assert!(matches!(
        reference.compare_linear_planes_with(views, &stop, |_, _, _| panic!("cancelled")),
        Err(butteraugli::ButteraugliError::Cancelled(_))
    ));
    reference
        .compare_linear_planes_with(views, &enough::Unstoppable, |score, _, _| {
            assert!(score.is_finite())
        })
        .unwrap();
}
