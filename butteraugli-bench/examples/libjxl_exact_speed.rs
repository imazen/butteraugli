//! Resident RGB8 comparisons. Zenbench owns warmup, calibration and sampling.
//! Decode occurs at setup. Each measured call includes result destruction.
use butteraugli::{ButteraugliParams, Img, Numerics, RGB8, butteraugli};
use std::{env, path::PathBuf, time::Duration};
use zenbench::black_box;
fn main() {
    let reference = image::open(env::var_os("BU_BENCH_REFERENCE").unwrap())
        .unwrap()
        .to_rgb8();
    let distorted = image::open(env::var_os("BU_BENCH_DISTORTED").unwrap())
        .unwrap()
        .to_rgb8();
    assert_eq!(reference.dimensions(), distorted.dimensions());
    let (w, h) = (reference.width() as usize, reference.height() as usize);
    let a: Vec<_> = reference
        .pixels()
        .map(|p| RGB8::new(p[0], p[1], p[2]))
        .collect();
    let b: Vec<_> = distorted
        .pixels()
        .map(|p| RGB8::new(p[0], p[1], p[2]))
        .collect();
    let result = zenbench::run(|suite| {
        suite.compare(format!("srgb-{w}x{h}"), |group| {
            group
                .config()
                .max_rounds(7)
                .min_rounds(7)
                .auto_rounds(false)
                .warmup_time(Duration::from_millis(500))
                .cache_firewall(false)
                .sample_target_ns(1)
                .min_sample_ns(1)
                .stack_jitter(false);
            for (label, mode) in [("fast", Numerics::Fast), ("exact", Numerics::LibjxlExact)] {
                let (a, b) = (a.clone(), b.clone());
                let params = ButteraugliParams::new()
                    .with_numerics(mode)
                    .with_compute_diffmap(true);
                group.bench(label, move |bencher| {
                    bencher.iter(|| {
                        black_box(
                            butteraugli(
                                Img::new(a.as_slice(), w, h),
                                Img::new(b.as_slice(), w, h),
                                &params,
                            )
                            .unwrap(),
                        )
                    })
                });
            }
        });
    });
    result
        .save(PathBuf::from(env::var_os("BU_BENCH_OUTPUT").unwrap()))
        .unwrap();
}
