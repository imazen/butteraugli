//! One-shot Butteraugli comparison on identical linear-light inputs.
//!
//! Build fmetrics with `zig build --release=fast`, then run with
//! `FMETRICS_LIB_DIR=/path/to/fmetrics/zig-out/lib` and the `fmetrics-ffi`
//! feature. Decode and JPEG distortion happen before the timed calls.

use butteraugli::{ButteraugliParams, ButteraugliReference};
use image::codecs::jpeg::JpegEncoder;
use image::{ExtendedColorType, ImageFormat, imageops::FilterType};
#[cfg(feature = "fmetrics-ffi")]
use std::ffi::c_void;
use std::time::Instant;

#[cfg(feature = "fmetrics-ffi")]
#[repr(C)]
struct FmetricsImg {
    data: *const c_void,
    width: u32,
    height: u32,
    stride: u32,
    format: i32,
    colorspace: i32,
    hdr: bool,
}

#[cfg(feature = "fmetrics-ffi")]
#[repr(C)]
struct FmetricsButteraugliOptions {
    intensity_target: f32,
    pnorm: i32,
}

#[cfg(feature = "fmetrics-ffi")]
#[repr(C)]
struct FmetricsWorkspace {
    _private: [u8; 0],
}

#[cfg(feature = "fmetrics-ffi")]
unsafe extern "C" {
    fn fmetrics_workspace_create() -> *mut FmetricsWorkspace;
    fn fmetrics_workspace_destroy(workspace: *mut FmetricsWorkspace);
    fn fmetrics_butteraugli_cmp(
        workspace: *mut FmetricsWorkspace,
        reference: *const FmetricsImg,
        distorted: *const FmetricsImg,
        options: *const FmetricsButteraugliOptions,
        result: *mut f64,
    ) -> i32;
}

#[cfg(has_cpp_butteraugli)]
unsafe extern "C" {
    fn butteraugli_pnorm3_from_linear_planes(
        src0: *const f32,
        src1: *const f32,
        src2: *const f32,
        dst0: *const f32,
        dst1: *const f32,
        dst2: *const f32,
        width: usize,
        height: usize,
        intensity_target: f32,
    ) -> f64;
}

struct Inputs {
    width: usize,
    height: usize,
    reference: [Vec<f32>; 3],
    distorted: [Vec<f32>; 3],
    reference_rgb: Vec<f32>,
    distorted_rgb: Vec<f32>,
}

fn planar_and_rgb(bytes: &[u8]) -> ([Vec<f32>; 3], Vec<f32>) {
    let n = bytes.len() / 3;
    let mut planar = std::array::from_fn(|_| Vec::with_capacity(n));
    let mut rgb = Vec::with_capacity(bytes.len());
    for px in bytes.as_chunks::<3>().0 {
        for c in 0..3 {
            let v = linear_srgb::default::srgb_u8_to_linear(px[c]);
            planar[c].push(v);
            rgb.push(v);
        }
    }
    (planar, rgb)
}

fn prepare(path: &str, quality: u8, resize: Option<(u32, u32)>) -> Inputs {
    let reference_img = image::open(path).expect("decode source").to_rgb8();
    let reference_img = match resize {
        Some((width, height)) => {
            image::imageops::resize(&reference_img, width, height, FilterType::Triangle)
        }
        None => reference_img,
    };
    let (width, height) = reference_img.dimensions();
    let mut jpeg = Vec::new();
    JpegEncoder::new_with_quality(&mut jpeg, quality)
        .encode(
            reference_img.as_raw(),
            width,
            height,
            ExtendedColorType::Rgb8,
        )
        .expect("encode distortion");
    let distorted_img = image::load_from_memory_with_format(&jpeg, ImageFormat::Jpeg)
        .expect("decode distortion")
        .to_rgb8();
    let (reference, reference_rgb) = planar_and_rgb(reference_img.as_raw());
    let (distorted, distorted_rgb) = planar_and_rgb(distorted_img.as_raw());
    Inputs {
        width: width as usize,
        height: height as usize,
        reference,
        distorted,
        reference_rgb,
        distorted_rgb,
    }
}

fn rust_score(input: &Inputs) -> f64 {
    let params = ButteraugliParams::default().with_intensity_target(203.0);
    let reference = ButteraugliReference::new_linear_planar(
        &input.reference[0],
        &input.reference[1],
        &input.reference[2],
        input.width,
        input.height,
        input.width,
        params,
    )
    .expect("Rust reference");
    reference
        .compare_linear_planar(
            &input.distorted[0],
            &input.distorted[1],
            &input.distorted[2],
            input.width,
        )
        .expect("Rust compare")
        .pnorm_3
}

#[cfg(feature = "fmetrics-ffi")]
fn fmetrics_score(input: &Inputs) -> f64 {
    let make_img = |data: &Vec<f32>| FmetricsImg {
        data: data.as_ptr().cast(),
        width: input.width as u32,
        height: input.height as u32,
        stride: (input.width * 3 * size_of::<f32>()) as u32,
        format: 2,     // FMETRICS_PIX_FMT_RGB_FLOAT
        colorspace: 2, // FMETRICS_COLORSPACE_LINEAR_SRGB
        hdr: false,
    };
    let reference = make_img(&input.reference_rgb);
    let distorted = make_img(&input.distorted_rgb);
    let options = FmetricsButteraugliOptions {
        intensity_target: 203.0,
        pnorm: 3,
    };
    let mut score = f64::NAN;
    // SAFETY: all FFI structs match fmetrics.h; the buffers outlive this call.
    unsafe {
        let workspace = fmetrics_workspace_create();
        assert!(!workspace.is_null(), "fmetrics workspace allocation");
        let status =
            fmetrics_butteraugli_cmp(workspace, &reference, &distorted, &options, &mut score);
        fmetrics_workspace_destroy(workspace);
        assert_eq!(status, 0, "fmetrics error {status}");
    }
    score
}

#[cfg(has_cpp_butteraugli)]
fn libjxl_score(input: &Inputs) -> f64 {
    let score = unsafe {
        butteraugli_pnorm3_from_linear_planes(
            input.reference[0].as_ptr(),
            input.reference[1].as_ptr(),
            input.reference[2].as_ptr(),
            input.distorted[0].as_ptr(),
            input.distorted[1].as_ptr(),
            input.distorted[2].as_ptr(),
            input.width,
            input.height,
            203.0,
        )
    };
    assert!(score >= 0.0, "libjxl Butteraugli error: {score}");
    score
}

fn main() {
    let mut args = std::env::args().skip(1);
    let backend = args.next().expect("rust|fmetrics|libjxl");
    let path = args.next().expect("source image path");
    let runs: usize = args.next().unwrap_or_else(|| "1".into()).parse().unwrap();
    let quality: u8 = args.next().unwrap_or_else(|| "75".into()).parse().unwrap();
    let resize = args.next().map(|size| {
        let (width, height) = size.split_once('x').expect("resize must be WxH");
        (width.parse().unwrap(), height.parse().unwrap())
    });
    let input = prepare(&path, quality, resize);
    // Keep both representations allocated across backends for comparable RSS.
    std::hint::black_box((&input.reference_rgb, &input.distorted_rgb));
    let call: fn(&Inputs) -> f64 = match backend.as_str() {
        "rust" => rust_score,
        #[cfg(feature = "fmetrics-ffi")]
        "fmetrics" => fmetrics_score,
        #[cfg(has_cpp_butteraugli)]
        "libjxl" => libjxl_score,
        _ => panic!("backend must be rust, fmetrics, or libjxl (when C++ FFI is built)"),
    };
    for _ in 0..runs {
        let start = Instant::now();
        let score = call(&input);
        println!(
            "backend={backend} image={path} size={}x{} quality={quality} score={score:.17} ms={:.3}",
            input.width,
            input.height,
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
}
