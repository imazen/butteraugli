use super::*;
use crate::ButteraugliReference;

#[test]
fn exact_planar_ingresses_match_srgb_and_account_retained_storage() {
    let (w, h, stride) = (33, 35, 38);
    let a: Vec<_> = (0..w * h * 3)
        .map(|i| ((i * 71 + i / 13) % 256) as u8)
        .collect();
    let b: Vec<_> = a.iter().map(|v| v / 8 * 8).collect();
    let params = ButteraugliParams::new()
        .with_numerics(crate::Numerics::LibjxlExact)
        .with_compute_diffmap(true);
    let reference = ButteraugliReference::new(&a, w, h, params.clone()).unwrap();
    let mut ap = std::array::from_fn::<_, 3, _>(|_| vec![0.0; stride * h]);
    let mut bp = ap.clone();
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                ap[c][y * stride + x] =
                    srgb_lut(crate::Numerics::LibjxlExact)[a[(y * w + x) * 3 + c] as usize];
                bp[c][y * stride + x] =
                    srgb_lut(crate::Numerics::LibjxlExact)[b[(y * w + x) * 3 + c] as usize];
            }
        }
    }
    let planar = ButteraugliReference::new_linear_planar(
        &ap[0],
        &ap[1],
        &ap[2],
        w,
        h,
        stride,
        params.clone(),
    )
    .unwrap();
    assert_eq!(
        planar.precompute_bytes(),
        ButteraugliReference::estimated_reference_bytes(w, h, &params)
    );
    assert!(planar.memory_bytes() >= planar.precompute_bytes());
    let expected = reference.compare(&b).unwrap();
    let mut map = Vec::new();
    let (max, p3) = planar
        .compare_linear_planar_into(&bp[0], &bp[1], &bp[2], stride, &mut map)
        .unwrap();
    assert_eq!(
        (max.to_bits(), p3.to_bits()),
        (expected.score.to_bits(), expected.pnorm_3.to_bits())
    );
    assert_eq!(&map, expected.diffmap.as_ref().unwrap().buf());
    let (max, p3) = planar
        .compare_linear_planar_into(&ap[0], &ap[1], &ap[2], stride, &mut map)
        .unwrap();
    assert_eq!((max, p3), (0.0, 0.0));
    assert!(map.iter().all(|&v| v == 0.0));
    #[cfg(feature = "linear-planes")]
    {
        use crate::linear_planes::{LinearPlanes, Scorer};
        let a = LinearPlanes::with_stride(&ap[0], &ap[1], &ap[2], w, h, stride).unwrap();
        let b = LinearPlanes::with_stride(&bp[0], &bp[1], &bp[2], w, h, stride).unwrap();
        let scorer = Scorer::builder()
            .with_numerics(crate::Numerics::LibjxlExact)
            .with_diffmap(true)
            .build(&a)
            .unwrap();
        let scores = scorer.score_into(&b, &mut map).unwrap();
        assert_eq!(
            (scores.max_norm.to_bits(), scores.pnorm_3.to_bits()),
            (expected.score.to_bits(), expected.pnorm_3.to_bits())
        );
        assert_eq!(&map, expected.diffmap.as_ref().unwrap().buf());
        let zero = scorer.score_into(&a, &mut map).unwrap();
        assert_eq!((zero.max_norm, zero.pnorm_3), (0.0, 0.0));
        assert!(map.iter().all(|&v| v == 0.0));
    }
}
