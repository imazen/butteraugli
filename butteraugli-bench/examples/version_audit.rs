//! Interleaved 0.9.3/current comparisons with identical parameters and pixels.
//! Decode, JPEG distortion, linearization and warm-reference setup are untimed.
//! Run: version_audit IMAGE.png [ROUNDS]; paired zenbench results and scores on stdout.

use butteraugli::{ButteraugliParams, ButteraugliReference};
use butteraugli_published::{ButteraugliParams as OldParams, ButteraugliReference as OldRef};
use image::{ExtendedColorType, ImageFormat, codecs::jpeg::JpegEncoder, imageops::FilterType};
use std::{hint::black_box, sync::Arc, time::Duration};

#[path = "../../butteraugli/tests/common/generators.rs"]
mod generators;
#[allow(dead_code, clippy::excessive_precision, clippy::approx_constant)]
#[path = "../../butteraugli/tests/common/reference_data.rs"]
mod reference_data;

fn golden_audit() {
    let mut count = 0;
    let mut identical = 0;
    let mut abs_sum = [0.0_f64; 2];
    let mut max_relative = [0.0_f64; 2];
    let mut max_version_delta = 0.0_f64;
    for case in reference_data::REFERENCE_CASES {
        assert!(
            case.width >= 8 && case.height >= 8,
            "golden requires supported dimensions"
        );
        let (a, b) = generators::generate_image_pair(case.name, case.width, case.height)
            .expect("every captured generator must exist");
        if a == b {
            identical += 1;
            continue;
        }
        let a = generators::rgb_bytes_to_pixels(&a);
        let b = generators::rgb_bytes_to_pixels(&b);
        let a = butteraugli::Img::new(a, case.width, case.height);
        let b = butteraugli::Img::new(b, case.width, case.height);
        let old = butteraugli_published::butteraugli(a.as_ref(), b.as_ref(), &OldParams::default())
            .unwrap()
            .score;
        let new = butteraugli::butteraugli(a.as_ref(), b.as_ref(), &ButteraugliParams::default())
            .unwrap()
            .score;
        for (i, score) in [old, new].into_iter().enumerate() {
            let delta = (score - case.expected_score).abs();
            abs_sum[i] += delta;
            max_relative[i] = max_relative[i].max(delta / case.expected_score.abs().max(0.001));
        }
        max_version_delta = max_version_delta.max((new - old).abs());
        count += 1;
    }
    println!("distorted_pairs={count} identity_pairs_excluded={identical}");
    for (i, version) in ["0.9.3", "main"].into_iter().enumerate() {
        println!(
            "{version}: mean_abs_vs_cpp={:.17} max_relative_vs_cpp={:.17}",
            abs_sum[i] / count as f64,
            max_relative[i]
        );
    }
    println!("max_abs_version_delta={max_version_delta:.17}");
}

#[cfg(has_cpp_butteraugli)]
unsafe extern "C" {
    fn butteraugli_pnorm3_from_linear_planes(
        s0: *const f32,
        s1: *const f32,
        s2: *const f32,
        d0: *const f32,
        d1: *const f32,
        d2: *const f32,
        width: usize,
        height: usize,
        intensity_target: f32,
    ) -> f64;
}

fn prepare_pair(photo: &image::RgbImage, w: usize, h: usize) -> ([Vec<f32>; 3], [Vec<f32>; 3]) {
    let img = image::imageops::resize(photo, w as u32, h as u32, FilterType::Triangle);
    let mut jpg = Vec::new();
    JpegEncoder::new_with_quality(&mut jpg, 75)
        .encode(img.as_raw(), w as u32, h as u32, ExtendedColorType::Rgb8)
        .unwrap();
    let dst = image::load_from_memory_with_format(&jpg, ImageFormat::Jpeg)
        .unwrap()
        .to_rgb8();
    let planes = |bytes: &[u8]| -> [Vec<f32>; 3] {
        std::array::from_fn(|c| {
            bytes
                .as_chunks::<3>()
                .0
                .iter()
                .map(|p| linear_srgb::default::srgb_u8_to_linear(p[c]))
                .collect()
        })
    };
    let src = planes(img.as_raw());
    let dst = planes(dst.as_raw());
    (src, dst)
}

// Separate processes let /usr/bin/time -v measure each backend independently.
fn current_params(args: &[String]) -> ButteraugliParams {
    let params = ButteraugliParams::default();
    match args.iter().find_map(|s| s.strip_prefix("--cache-bytes=")) {
        Some(bytes) => {
            params.with_scratch_cache_bytes(bytes.parse().expect("cache bytes must be an integer"))
        }
        None => params,
    }
}

fn profile_version(args: &[String], version: &str) {
    let option = |name: &str, fallback: &str| {
        args.iter()
            .find_map(|s| s.strip_prefix(name))
            .unwrap_or(fallback)
            .to_owned()
    };
    let size = option("--size=", "1920x1080");
    let (w, h) = size.split_once('x').expect("size is WIDTHxHEIGHT");
    let (w, h): (usize, usize) = (w.parse().unwrap(), h.parse().unwrap());
    let threads: usize = option("--threads=", "8").parse().unwrap();
    let calls: usize = option("--calls=", "20").parse().unwrap();
    assert!(calls > 0);
    let photo = image::open(args.get(1).expect("provide a reference PNG"))
        .unwrap()
        .to_rgb8();
    let (src, dst) = prepare_pair(&photo, w, h);
    let workers = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    workers.install(|| {
        let run = |compare: &mut dyn FnMut() -> (f64, f64)| {
            black_box(compare()); // warm cache before the timed batch
            let start = std::time::Instant::now();
            let mut scores = (0.0, 0.0);
            for _ in 0..calls { scores = black_box(compare()); }
            println!("PROFILE version={version} size={size} threads={threads} calls={calls} wall_ms={:.6} score={:.17} pnorm_3={:.17}", start.elapsed().as_secs_f64()*1000.0, scores.0, scores.1);
        };
        match version {
            "published" => {
                let reference = OldRef::new_linear_planar(&src[0], &src[1], &src[2], w, h, w, OldParams::default()).unwrap();
                run(&mut || { let r = reference.compare_linear_planar(&dst[0], &dst[1], &dst[2], w).unwrap(); (r.score, r.pnorm_3) });
            }
            "current" => {
                let reference = ButteraugliReference::new_linear_planar(&src[0], &src[1], &src[2], w, h, w, current_params(args)).unwrap();
                run(&mut || { let r = reference.compare_linear_planar(&dst[0], &dst[1], &dst[2], w).unwrap(); (r.score, r.pnorm_3) });
                println!("RETAINED bytes={} precompute_bytes={}", reference.memory_bytes(), reference.precompute_bytes());
            }
            _ => panic!("profile version must be published or current"),
        }
    });
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    println!(
        "CONFIG scratch_cache_bytes={:?}",
        current_params(&args).scratch_cache_bytes()
    );
    if let Some(version) = args.iter().find_map(|s| s.strip_prefix("--profile=")) {
        profile_version(&args, version);
        return;
    }
    if args.get(1).is_some_and(|s| s == "--golden") {
        golden_audit();
        return;
    }
    let rounds: usize = args.get(2).map_or(7, |s| s.parse().unwrap());
    let scores_only = args.get(3).is_some_and(|s| s == "--scores-only");
    let filter = args.iter().find_map(|s| s.strip_prefix("--group="));
    assert!(
        rounds >= 3,
        "at least three alternating rounds are required"
    );
    let photo = image::open(args.get(1).expect("provide a reference PNG"))
        .unwrap()
        .to_rgb8();

    for (w, h) in [(512, 512), (1920, 1080), (3840, 2160)] {
        let (src, dst) = prepare_pair(&photo, w, h);
        #[cfg(has_cpp_butteraugli)]
        let cpp = unsafe {
            butteraugli_pnorm3_from_linear_planes(
                src[0].as_ptr(),
                src[1].as_ptr(),
                src[2].as_ptr(),
                dst[0].as_ptr(),
                dst[1].as_ptr(),
                dst[2].as_ptr(),
                w,
                h,
                80.0,
            )
        };
        #[cfg(not(has_cpp_butteraugli))]
        let cpp = f64::NAN; // Explicit Rust-only build; never report this as C++ parity.
        #[cfg(has_cpp_butteraugli)]
        assert!(cpp.is_finite() && cpp >= 0.0, "C++ oracle failed");
        let src = Arc::new(src);
        let dst = Arc::new(dst);
        for threads in [1, 8] {
            let pool = Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap(),
            );
            for mode in ["cold", "warm", "warm-map"] {
                let label = format!("{w}x{h}_{threads}t_{mode}");
                if filter.is_some_and(|f| !label.contains(f)) {
                    continue;
                }
                let map = mode == "warm-map";
                let params = current_params(&args).with_compute_diffmap(map);
                let old_params = OldParams::default().with_compute_diffmap(map);
                assert_eq!(params.intensity_target(), old_params.intensity_target());
                assert_eq!(params.single_resolution(), old_params.single_resolution());
                let current = pool.install(|| {
                    ButteraugliReference::new_linear_planar(
                        &src[0],
                        &src[1],
                        &src[2],
                        w,
                        h,
                        w,
                        params.clone(),
                    )
                    .unwrap()
                });
                let old = pool.install(|| {
                    OldRef::new_linear_planar(
                        &src[0],
                        &src[1],
                        &src[2],
                        w,
                        h,
                        w,
                        old_params.clone(),
                    )
                    .unwrap()
                });
                let old_score = pool.install(|| {
                    old.compare_linear_planar(&dst[0], &dst[1], &dst[2], w)
                        .unwrap()
                });
                let new_score = pool.install(|| {
                    current
                        .compare_linear_planar(&dst[0], &dst[1], &dst[2], w)
                        .unwrap()
                });
                println!(
                    "SCORE\t{w}\t{h}\t{threads}\t{mode}\t{:.17}\t{:.17}\t{:.17}\t{:.17}\t{cpp:.17}\t{}\t{}",
                    old_score.score,
                    new_score.score,
                    old_score.pnorm_3,
                    new_score.pnorm_3,
                    old_score.diffmap.is_some(),
                    new_score.diffmap.is_some()
                );
                drop(old_score);
                drop(new_score);
                if scores_only {
                    continue;
                }
                let gate = if args.iter().any(|arg| arg == "--ungated") {
                    zenbench::GateConfig::disabled()
                } else {
                    zenbench::GateConfig::strict()
                };
                let result = zenbench::run_gated(gate, |suite| {
                    suite.compare(label, |group| {
                        group
                            .config()
                            .max_rounds(rounds)
                            .min_rounds(rounds)
                            .auto_rounds(false)
                            .warmup_time(Duration::from_millis(100))
                            .max_time(Duration::from_secs(60));
                        group.config().min_iterations = 1;
                        group.config().max_iterations = 1;
                        group.baseline("0.9.3");
                        let s = src.clone();
                        let d = dst.clone();
                        let p = pool.clone();
                        group.bench("0.9.3", move |b| {
                            b.iter(|| {
                                p.install(|| {
                                    let reference;
                                    let r = if mode == "cold" {
                                        reference = OldRef::new_linear_planar(
                                            &s[0],
                                            &s[1],
                                            &s[2],
                                            w,
                                            h,
                                            w,
                                            old_params.clone(),
                                        )
                                        .unwrap();
                                        &reference
                                    } else {
                                        &old
                                    };
                                    black_box(
                                        r.compare_linear_planar(&d[0], &d[1], &d[2], w).unwrap(),
                                    )
                                })
                            })
                        });
                        let s = src.clone();
                        let d = dst.clone();
                        let p = pool.clone();
                        group.bench("main", move |b| {
                            b.iter(|| {
                                p.install(|| {
                                    let reference;
                                    let r = if mode == "cold" {
                                        reference = ButteraugliReference::new_linear_planar(
                                            &s[0],
                                            &s[1],
                                            &s[2],
                                            w,
                                            h,
                                            w,
                                            params.clone(),
                                        )
                                        .unwrap();
                                        &reference
                                    } else {
                                        &current
                                    };
                                    black_box(
                                        r.compare_linear_planar(&d[0], &d[1], &d[2], w).unwrap(),
                                    )
                                })
                            })
                        });
                    });
                });
                println!(
                    "GATE waits={} unreliable={}",
                    result.gate_waits, result.unreliable
                );
                assert!(!result.unreliable, "zenbench rejected this contended run");
                print!("{}", result.to_csv());
            }
        }
    }
}
