//! A/B benchmarks: butteraugli (Rust, this crate) vs libjxl::ButteraugliDiffmap (C++ FFI).
//!
//! Both implementations receive pre-linearized planar f32 buffers — the only work
//! measured is the butteraugli pipeline itself (XYB conversion, blur, masks, malta,
//! psycho, diffmap, score). sRGB→linear conversion happens once at setup.
//!
//! **Three variants per size:**
//! - `rust_butteraugli` — out-of-the-box (rayon multi-threaded, avx512 dispatch).
//! - `rust_butteraugli_st` — same crate, forced into a 1-thread rayon pool.
//!   This is the apples-to-apples algorithmic comparison vs the single-threaded
//!   C++ side. Improvements we want to upstream into libjxl should close the gap
//!   between this row and the C++ row.
//! - `cpp_libjxl` — libjxl ButteraugliDiffmap, x86-64-v2 baseline + AVX2/AVX3
//!   highway runtime dispatch (see `build.rs`).
//!
//! Run: `cargo bench -p butteraugli-bench --bench bench_compare`

use butteraugli::{ButteraugliParams, ButteraugliReference};
use zenbench::black_box;

#[cfg(has_cpp_butteraugli)]
unsafe extern "C" {
    fn butteraugli_from_linear_planes(
        src0: *const f32,
        src1: *const f32,
        src2: *const f32,
        dst0: *const f32,
        dst1: *const f32,
        dst2: *const f32,
        width: usize,
        height: usize,
    ) -> f64;
}

const SIZES: &[(&str, usize, usize)] = &[
    ("512x512", 512, 512),
    ("1280x720", 1280, 720),
    ("1920x1080", 1920, 1080),
    ("3840x2160", 3840, 2160),
];

/// Synthetic gradient pair — same shape zensim-bench uses for compute benchmarks.
/// Realistic enough to exercise the full butteraugli pipeline; no test-data dependency.
fn make_test_planes(w: usize, h: usize) -> ([Vec<f32>; 3], [Vec<f32>; 3]) {
    let n = w * h;
    let mut src = [
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    ];
    let mut dst = [
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    ];
    for i in 0..n {
        let x = (i % w) * 255 / w;
        let y = (i / w) * 255 / h;
        let r = x as u8;
        let g = y as u8;
        let b = (x as u8).wrapping_add(y as u8);
        let r2 = r.saturating_add(5);
        let g2 = g.saturating_add(3);
        let b2 = b;
        src[0].push(linear_srgb::default::srgb_u8_to_linear(r));
        src[1].push(linear_srgb::default::srgb_u8_to_linear(g));
        src[2].push(linear_srgb::default::srgb_u8_to_linear(b));
        dst[0].push(linear_srgb::default::srgb_u8_to_linear(r2));
        dst[1].push(linear_srgb::default::srgb_u8_to_linear(g2));
        dst[2].push(linear_srgb::default::srgb_u8_to_linear(b2));
    }
    (src, dst)
}

zenbench::main!(|suite| {
    for &(label, w, h) in SIZES {
        let group_name = format!("butteraugli_{label}");
        suite.compare(group_name, |group| {
            let (src, dst) = make_test_planes(w, h);

            // Rust (out-of-the-box): rayon multi-threaded, avx512 dispatch.
            {
                let src = src.clone();
                let dst = dst.clone();
                group.bench("rust_butteraugli", move |b| {
                    b.iter(|| {
                        let params = ButteraugliParams::default();
                        let reference = ButteraugliReference::new_linear_planar(
                            black_box(&src[0]),
                            black_box(&src[1]),
                            black_box(&src[2]),
                            w,
                            h,
                            w,
                            params,
                        )
                        .unwrap();
                        let result = reference
                            .compare_linear_planar(
                                black_box(&dst[0]),
                                black_box(&dst[1]),
                                black_box(&dst[2]),
                                w,
                            )
                            .unwrap();
                        black_box(result.score)
                    })
                });
            }

            // Rust single-threaded (1-thread rayon pool): apples-to-apples vs C++.
            {
                let src = src.clone();
                let dst = dst.clone();
                let st_pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(1)
                    .build()
                    .unwrap();
                group.bench("rust_butteraugli_st", move |b| {
                    b.iter(|| {
                        st_pool.install(|| {
                            let params = ButteraugliParams::default();
                            let reference = ButteraugliReference::new_linear_planar(
                                black_box(&src[0]),
                                black_box(&src[1]),
                                black_box(&src[2]),
                                w,
                                h,
                                w,
                                params,
                            )
                            .unwrap();
                            let result = reference
                                .compare_linear_planar(
                                    black_box(&dst[0]),
                                    black_box(&dst[1]),
                                    black_box(&dst[2]),
                                    w,
                                )
                                .unwrap();
                            black_box(result.score)
                        })
                    })
                });
            }

            // C++: libjxl::ButteraugliDiffmap via FFI shim.
            #[cfg(has_cpp_butteraugli)]
            {
                let src = src.clone();
                let dst = dst.clone();
                group.bench("cpp_libjxl", move |b| {
                    b.iter(|| {
                        let score = unsafe {
                            butteraugli_from_linear_planes(
                                black_box(src[0].as_ptr()),
                                black_box(src[1].as_ptr()),
                                black_box(src[2].as_ptr()),
                                black_box(dst[0].as_ptr()),
                                black_box(dst[1].as_ptr()),
                                black_box(dst[2].as_ptr()),
                                w,
                                h,
                            )
                        };
                        assert!(score > -900.0, "C++ butteraugli FFI failed: {score}");
                        black_box(score)
                    })
                });
            }
        });
    }
});
