//! Accounting regression: exercise retained pools after repeated strided compares.
use butteraugli::{ButteraugliParams, ButteraugliReference};

#[test]
fn planar_peak_covers_retained_buffers_after_repeated_comparisons() {
    for (w, h) in [(8, 8), (14, 20), (15, 15), (37, 41), (256, 257), (513, 259)] {
        for single in [false, true] {
            let params = ButteraugliParams::new().with_single_resolution(single);
            let stride = w + 7;
            let data: Vec<f32> = (0..stride * h)
                .map(|i| ((i * 173 + i / 13) % 1024) as f32 / 1024.0)
                .collect();
            let reference = ButteraugliReference::new_linear_planar(
                &data,
                &data,
                &data,
                w,
                h,
                stride,
                params.clone(),
            )
            .unwrap();
            let peak = ButteraugliReference::estimated_planar_peak_bytes(w, h, &params).unwrap();
            assert!(peak > reference.memory_bytes());
            let mut output = Vec::new();
            for _ in 0..3 {
                reference
                    .compare_linear_planar_into(&data, &data, &data, stride, &mut output)
                    .unwrap();
                assert!(peak > reference.memory_bytes(), "{w}x{h}, single={single}");
                assert_eq!(output.len(), w * h);
            }
        }
    }
}

#[test]
fn planar_peak_checks_overflow_and_resolution_boundary() {
    let p = ButteraugliParams::new();
    for (w, h) in [(usize::MAX, 8), (8, usize::MAX), (usize::MAX / 16, 64)] {
        assert_eq!(
            ButteraugliReference::estimated_planar_peak_bytes(w, h, &p),
            None
        );
    }
    assert_eq!(
        ButteraugliReference::estimated_planar_peak_bytes(0, 100, &p),
        Some(0)
    );
    let single = p.clone().with_single_resolution(true);
    for (w, h) in [(8, 30), (14, 20)] {
        assert_eq!(
            ButteraugliReference::estimated_planar_peak_bytes(w, h, &p),
            ButteraugliReference::estimated_planar_peak_bytes(w, h, &single)
        );
    }
    assert!(
        ButteraugliReference::estimated_planar_peak_bytes(15, 15, &p)
            > ButteraugliReference::estimated_planar_peak_bytes(15, 15, &single)
    );
}

#[test]
fn automatic_admission_scales_then_matches_an_explicit_cap() {
    let automatic = ButteraugliParams::default();
    let capped = automatic
        .clone()
        .with_scratch_cache_bytes(256 * 1024 * 1024);
    let uncached = automatic.clone().with_scratch_cache_bytes(0);
    let larger = automatic
        .clone()
        .with_scratch_cache_bytes(1024 * 1024 * 1024);
    let peak = |w, h, params: &ButteraugliParams| {
        ButteraugliReference::estimated_planar_peak_bytes(w, h, params).unwrap()
    };
    // Small images get less admitted cache than a large explicit budget, but
    // retain useful scratch beyond construction's uncached comparison budget.
    assert!(peak(65, 49, &automatic) > peak(65, 49, &uncached));
    assert!(peak(65, 49, &automatic) < peak(65, 49, &capped));
    // Large-image automatic admission agrees with an explicit cap. Raising
    // the override still increases the estimate rather than silently clamping.
    for (w, h) in [(3840, 2160), (7680, 4320)] {
        assert_eq!(peak(w, h, &automatic), peak(w, h, &capped));
        assert!(peak(w, h, &automatic) < peak(w, h, &larger));
    }
}
