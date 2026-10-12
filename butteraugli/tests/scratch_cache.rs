//! Cache configuration changes allocation behavior, never pixels or scores.
use butteraugli::{ButteraugliParams, ButteraugliReference};

#[test]
fn byte_budgets_preserve_strided_scores_and_bound_idle_storage() {
    for (w, h) in [(37, 41), (513, 259)] {
        let stride = w + 13;
        let src: Vec<f32> = (0..stride * h)
            .map(|i| ((i * 173 + i / 13) % 1024) as f32 / 1024.0)
            .collect();
        let run = || {
            let default = ButteraugliReference::new_linear_planar(
                &src,
                &src,
                &src,
                w,
                h,
                stride,
                ButteraugliParams::default(),
            )
            .unwrap();
            for bytes in [0, 1024, 1024 * 1024, 32 * 1024 * 1024] {
                let params = ButteraugliParams::default().with_scratch_cache_bytes(bytes);
                let mut configured = ButteraugliReference::new_linear_planar(
                    &src,
                    &src,
                    &src,
                    w,
                    h,
                    stride,
                    params.clone(),
                )
                .unwrap();
                let peak =
                    ButteraugliReference::estimated_planar_peak_bytes(w, h, &params).unwrap();
                assert!(configured.memory_bytes() - configured.precompute_bytes() <= bytes);
                for offset in [0.02_f32, 0.2, 0.03] {
                    let dst: Vec<f32> = src.iter().map(|v| (v + offset).min(1.0)).collect();
                    let expected = default
                        .compare_linear_planar(&dst, &dst, &dst, stride)
                        .unwrap();
                    for r in [&configured, &configured.clone()] {
                        let actual = r.compare_linear_planar(&dst, &dst, &dst, stride).unwrap();
                        assert_eq!(actual.score, expected.score);
                        assert_eq!(actual.pnorm_3, expected.pnorm_3);
                        assert_eq!(
                            actual.diffmap.unwrap().buf(),
                            expected.diffmap.as_ref().unwrap().buf()
                        );
                        assert!(r.memory_bytes() - r.precompute_bytes() <= bytes);
                        assert!(r.memory_bytes() < peak);
                    }
                }
                configured.shrink_to_fit();
                assert_eq!(configured.memory_bytes(), configured.precompute_bytes());
            }
        };
        #[cfg(feature = "rayon")]
        for threads in [1, 4] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(run);
        }
        #[cfg(not(feature = "rayon"))]
        run();
    }
}

#[cfg(feature = "linear-planes")]
#[test]
fn typed_whole_image_scorer_uses_the_same_cache_budget() {
    use butteraugli::linear_planes::{LinearPlanes, ScorerBuilder};
    let (w, h) = (65, 49);
    let src: Vec<f32> = (0..w * h).map(|i| (i * 7 % 251) as f32 / 251.0).collect();
    let dst: Vec<f32> = src.iter().map(|v| (v + 0.03).min(1.0)).collect();
    let planes = LinearPlanes::new(&src, &src, &src, w, h).unwrap();
    let target = LinearPlanes::new(&dst, &dst, &dst, w, h).unwrap();
    let default = ScorerBuilder::new().build(&planes).unwrap();
    let zero = ScorerBuilder::new()
        .with_scratch_cache_bytes(0)
        .build(&planes)
        .unwrap();
    let retained = zero.reference_bytes();
    for _ in 0..3 {
        let expected = default.score(&target).unwrap();
        let actual = zero.score(&target).unwrap();
        assert_eq!(actual.max_norm, expected.max_norm);
        assert_eq!(actual.pnorm_3, expected.pnorm_3);
        assert_eq!(zero.reference_bytes(), retained);
    }
}
