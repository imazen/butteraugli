//! Diffmap-level parity against libjxl's `butteraugli_main`, for both
//! `Numerics::Fast` and `Numerics::LibjxlExact`, under three sRGB ingress
//! choices.
//!
//! For every (reference, distortion, size) pair this writes 8-bit PNGs with no
//! color chunks plus two linear PFMs, runs `butteraugli_main --rawdistmap` on
//! all three, and compares the crate's diffmap, max-norm and 3-norm with each:
//!
//! - `png`: both sides decode the same u8 PNG with their own sRGB→linear step.
//! - `pfm-lcms`: both sides get linear floats from the lcms2 u8 LUT that
//!   fast-ssim2 captured from the standalone ssimulacra2 binary.
//! - `pfm-tf`: both sides get linear floats from libjxl's `TF_SRGB`
//!   rational polynomial (what `jxl_cms` uses for sRGB→linear sRGB).
//!
//! The two PFM arms take the transfer function out of the comparison. Matching
//! libjxl's own PNG distmap against a PFM arm identifies which linearization
//! `butteraugli_main` used.
//!
//! ```text
//! cargo run --release -p butteraugli --example libjxl_parity -- \
//!     --butteraugli-main /path/to/build/tools/butteraugli_main \
//!     --out ~/tmp/ba-parity/run [--cid22 12] [--clic 4]
//! ```

#[path = "data/lcms_srgb8_lut.rs"]
mod lcms_lut;

use butteraugli::{ButteraugliParams, Img, Numerics, RGB, RGB8, butteraugli, butteraugli_linear};
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

struct Image {
    w: usize,
    h: usize,
    px: Vec<u8>,
}

fn load_png(path: &Path) -> Image {
    let mut decoder = png::Decoder::new(BufReader::new(File::open(path).unwrap()));
    decoder.set_transformations(png::Transformations::normalize_to_color8());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    let (w, h) = (info.width as usize, info.height as usize);
    let src = &buf[..info.buffer_size()];
    let px = match info.color_type {
        png::ColorType::Rgb => src.to_vec(),
        png::ColorType::Rgba => src
            .chunks_exact(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        png::ColorType::Grayscale => src.iter().flat_map(|&v| [v, v, v]).collect(),
        png::ColorType::GrayscaleAlpha => src
            .chunks_exact(2)
            .flat_map(|c| [c[0], c[0], c[0]])
            .collect(),
        t => panic!("{}: unsupported color type {t:?}", path.display()),
    };
    Image { w, h, px }
}

fn save_png(path: &Path, img: &Image) {
    let mut enc = png::Encoder::new(
        BufWriter::new(File::create(path).unwrap()),
        img.w as u32,
        img.h as u32,
    );
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .unwrap()
        .write_image_data(&img.px)
        .unwrap();
}

/// Linear PFM, little-endian, bottom-up rows.
fn save_pfm(path: &Path, img: &Image, lut: &[f32; 256]) {
    let mut out = BufWriter::new(File::create(path).unwrap());
    write!(out, "PF\n{} {}\n-1.0\n", img.w, img.h).unwrap();
    for y in (0..img.h).rev() {
        for &v in &img.px[y * img.w * 3..(y + 1) * img.w * 3] {
            out.write_all(&lut[v as usize].to_le_bytes()).unwrap();
        }
    }
}

/// Grayscale PFM written by `butteraugli_main --rawdistmap`, returned top-down.
fn load_distmap(path: &Path, w: usize, h: usize) -> Vec<f32> {
    let bytes = fs::read(path).unwrap();
    let header = format!("Pf\n{w} {h}\n-1.0\n");
    assert!(
        bytes.starts_with(header.as_bytes()),
        "{}: unexpected header",
        path.display()
    );
    let data = &bytes[header.len()..];
    assert_eq!(data.len(), w * h * 4);
    let mut map = vec![0f32; w * h];
    for (row, chunk) in data.chunks_exact(w * 4).enumerate() {
        let y = h - 1 - row;
        for (x, b) in chunk.chunks_exact(4).enumerate() {
            map[y * w + x] = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        }
    }
    map
}

fn crop(img: &Image, w: usize, h: usize) -> Image {
    let (x0, y0) = ((img.w - w) / 2, (img.h - h) / 2);
    let mut px = Vec::with_capacity(w * h * 3);
    for y in y0..y0 + h {
        px.extend_from_slice(&img.px[(y * img.w + x0) * 3..(y * img.w + x0 + w) * 3]);
    }
    Image { w, h, px }
}

fn jpeg(img: &Image, quality: u8) -> Image {
    let mut comp = mozjpeg::Compress::new(mozjpeg::ColorSpace::JCS_RGB);
    comp.set_size(img.w, img.h);
    comp.set_quality(f32::from(quality));
    let mut encoded = Vec::new();
    let mut started = comp.start_compress(Cursor::new(&mut encoded)).unwrap();
    for row in img.px.chunks(img.w * 3) {
        started.write_scanlines(row).unwrap();
    }
    started.finish().unwrap();
    let mut dec = jpeg_decoder::Decoder::new(Cursor::new(encoded));
    let px = dec.decode().unwrap();
    assert_eq!(
        dec.info().unwrap().pixel_format,
        jpeg_decoder::PixelFormat::RGB24
    );
    Image {
        w: img.w,
        h: img.h,
        px,
    }
}

fn posterize(img: &Image) -> Image {
    Image {
        w: img.w,
        h: img.h,
        px: img.px.iter().map(|&v| v & !7).collect(),
    }
}

fn noise(img: &Image) -> Image {
    let px = img
        .px
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let mut z = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            z ^= z >> 29;
            z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z ^= z >> 32;
            (i32::from(v) + (z % 13) as i32 - 6).clamp(0, 255) as u8
        })
        .collect();
    Image {
        w: img.w,
        h: img.h,
        px,
    }
}

/// libjxl `TF_SRGB::DisplayFromEncoded` (cms/transfer_functions-inl.h) on
/// `u8 * (1/255)`, Horner steps fused as Highway's MulAdd.
fn libjxl_tf_srgb(v: u8) -> f32 {
    const P: [f32; 5] = [
        2.200248328e-4,
        1.043637593e-2,
        1.624820318e-1,
        7.961564959e-1,
        8.210152774e-1,
    ];
    const Q: [f32; 5] = [
        2.631846970e-1,
        1.076976492,
        4.987528350e-1,
        -5.512498495e-2,
        6.521209011e-3,
    ];
    let x = f32::from(v) * (1.0 / 255.0);
    if x <= 0.04045 {
        return x * (1.0 / 12.92);
    }
    let horner = |c: &[f32; 5]| c[..4].iter().rev().fold(c[4], |y, &k| y.mul_add(x, k));
    horner(&P) / horner(&Q)
}

/// The crate's 3-norm reduction, so map differences are isolated from
/// libjxl's lane-wise f64 summation order.
fn pnorm3(map: &[f32]) -> f64 {
    let mut sum = [0f64; 3];
    for &v in map {
        let mut d = f64::from(v).powi(3);
        sum[0] += d;
        d *= d;
        sum[1] += d;
        d *= d;
        sum[2] += d;
    }
    let inv = 1.0 / map.len() as f64;
    sum.iter()
        .enumerate()
        .map(|(i, &s)| (inv * s).powf(1.0 / (3.0 * f64::from(1u32 << i))))
        .sum::<f64>()
        / 3.0
}

fn max_norm(map: &[f32]) -> f64 {
    f64::from(map.iter().copied().fold(0f32, f32::max))
}

struct MapDiff {
    mismatched: usize,
    max_abs: f64,
    max_ulp: u32,
}

fn map_diff(a: &[f32], b: &[f32]) -> MapDiff {
    let mut d = MapDiff {
        mismatched: 0,
        max_abs: 0.0,
        max_ulp: 0,
    };
    for (&x, &y) in a.iter().zip(b) {
        if x.to_bits() != y.to_bits() {
            d.mismatched += 1;
            d.max_abs = d.max_abs.max(f64::from((x - y).abs()));
            d.max_ulp = d
                .max_ulp
                .max((x.to_bits() as i32).abs_diff(y.to_bits() as i32));
        }
    }
    d
}

struct Libjxl {
    printed_score: f64,
    map: Vec<f32>,
}

fn run_libjxl(
    bin: &Path,
    a: &Path,
    b: &Path,
    map_path: &Path,
    linear: bool,
    w: usize,
    h: usize,
) -> Libjxl {
    let mut cmd = Command::new(bin);
    cmd.arg(a)
        .arg(b)
        .arg("--rawdistmap")
        .arg(map_path)
        .args(["--pnorm", "3"]);
    if linear {
        cmd.args(["--colorspace", "RGB_D65_SRG_Rel_Lin"]);
    }
    let out = cmd.output().unwrap();
    assert!(
        out.status.success(),
        "{cmd:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).unwrap();
    let printed_score = stdout.lines().next().unwrap().trim().parse().unwrap();
    Libjxl {
        printed_score,
        map: load_distmap(map_path, w, h),
    }
}

fn lut_report(name: &str, a: &[f32; 256], b: &[f32; 256], out: &mut String) {
    let (mut n, mut ulp, mut abs) = (0, 0u32, 0f32);
    for (&x, &y) in a.iter().zip(b) {
        if x.to_bits() != y.to_bits() {
            n += 1;
            ulp = ulp.max(x.to_bits().abs_diff(y.to_bits()));
            abs = abs.max((x - y).abs());
        }
    }
    writeln!(
        out,
        "# lut {name}: {n}/256 entries differ, max {ulp} ULP, max abs {abs:e}"
    )
    .unwrap();
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .map(|i| args[i + 1].clone())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bin = PathBuf::from(arg(&args, "--butteraugli-main").expect("--butteraugli-main PATH"));
    let out_dir = PathBuf::from(arg(&args, "--out").expect("--out DIR"));
    let cid22_count: usize = arg(&args, "--cid22").map_or(12, |v| v.parse().unwrap());
    let clic_count: usize = arg(&args, "--clic").map_or(4, |v| v.parse().unwrap());
    let scratch = out_dir.join("pairs");
    fs::create_dir_all(&scratch).unwrap();

    let fast_lut: [f32; 256] = std::array::from_fn(|i| butteraugli::srgb_to_linear(i as u8));
    let tf_lut: [f32; 256] = std::array::from_fn(|i| libjxl_tf_srgb(i as u8));
    let lcms_lut = &lcms_lut::LINEAR_LUT_U8;

    let mut report = String::new();
    writeln!(report, "# butteraugli_main: {}", bin.display()).unwrap();
    lut_report("crate-fast vs libjxl-tf", &fast_lut, &tf_lut, &mut report);
    lut_report("crate-fast vs lcms2", &fast_lut, lcms_lut, &mut report);
    lut_report("libjxl-tf vs lcms2", &tf_lut, lcms_lut, &mut report);
    print!("{report}");

    let corpus = codec_corpus::Corpus::new().unwrap();
    let mut sources: Vec<(String, Image, Vec<(usize, usize)>)> = Vec::new();
    let pngs = |folder: &str| -> Vec<PathBuf> {
        let mut paths: Vec<PathBuf> = fs::read_dir(corpus.get(folder).unwrap())
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|e| e == "png"))
            .collect();
        paths.sort();
        paths
    };
    let stem = |p: &Path| p.file_stem().unwrap().to_string_lossy().into_owned();
    for path in pngs("CID22/CID22-512/validation")
        .into_iter()
        .take(cid22_count)
    {
        let img = load_png(&path);
        let sizes = vec![(7, 9), (16, 16), (65, 49), (256, 256), (img.w, img.h)];
        sources.push((format!("cid22/{}", stem(&path)), img, sizes));
    }
    for path in pngs("gb82") {
        let img = load_png(&path);
        let sizes = vec![(img.w, img.h)];
        sources.push((format!("gb82/{}", stem(&path)), img, sizes));
    }
    for path in pngs("clic2025/final-test").into_iter().take(clic_count) {
        let img = load_png(&path);
        let sizes = vec![(1024, 1024), (img.w, img.h)];
        sources.push((format!("clic2025/{}", &stem(&path)[..12]), img, sizes));
    }

    let mut tsv = String::from(
        "image\tw\th\tdistortion\tarm\tnumerics\tlibjxl_max\tcrate_max\tabs_dmax\tlibjxl_p3\tcrate_p3\tabs_dp3\tmap_mismatch\tmap_max_abs\tmap_max_ulp\n",
    );
    let mut cross =
        String::from("image\tw\th\tdistortion\tlibjxl_png_vs\tmap_mismatch\tmap_max_abs\n");
    for (name, src, sizes) in &sources {
        for &(w, h) in sizes {
            let reference = crop(src, w, h);
            let distortions: Vec<(&str, Image)> = vec![
                ("jpeg-q20", jpeg(&reference, 20)),
                ("jpeg-q50", jpeg(&reference, 50)),
                ("jpeg-q85", jpeg(&reference, 85)),
                ("posterize8", posterize(&reference)),
                ("noise6", noise(&reference)),
            ];
            let stem = scratch.join(format!("{}_{w}x{h}", name.replace('/', "_")));
            let path = |s: &str| PathBuf::from(format!("{}_{s}", stem.display()));
            save_png(&path("ref.png"), &reference);
            save_pfm(&path("ref-lcms.pfm"), &reference, lcms_lut);
            save_pfm(&path("ref-tf.pfm"), &reference, &tf_lut);
            for (dname, dist) in &distortions {
                save_png(&path(&format!("{dname}.png")), dist);
                save_pfm(&path(&format!("{dname}-lcms.pfm")), dist, lcms_lut);
                save_pfm(&path(&format!("{dname}-tf.pfm")), dist, &tf_lut);
                let map_path = path(&format!("{dname}-distmap.pfm"));
                let lj_png = run_libjxl(
                    &bin,
                    &path("ref.png"),
                    &path(&format!("{dname}.png")),
                    &map_path,
                    false,
                    w,
                    h,
                );
                let lj_lcms = run_libjxl(
                    &bin,
                    &path("ref-lcms.pfm"),
                    &path(&format!("{dname}-lcms.pfm")),
                    &map_path,
                    true,
                    w,
                    h,
                );
                let lj_tf = run_libjxl(
                    &bin,
                    &path("ref-tf.pfm"),
                    &path(&format!("{dname}-tf.pfm")),
                    &map_path,
                    true,
                    w,
                    h,
                );
                for (label, other) in [("pfm-lcms", &lj_lcms), ("pfm-tf", &lj_tf)] {
                    let d = map_diff(&lj_png.map, &other.map);
                    writeln!(
                        cross,
                        "{name}\t{w}\t{h}\t{dname}\t{label}\t{}\t{:e}",
                        d.mismatched, d.max_abs
                    )
                    .unwrap();
                }
                for (arm, lj, lut) in [
                    ("png", &lj_png, None),
                    ("pfm-lcms", &lj_lcms, Some(lcms_lut)),
                    ("pfm-tf", &lj_tf, Some(&tf_lut)),
                ] {
                    assert!(
                        (lj.printed_score - max_norm(&lj.map)).abs() < 1e-9,
                        "printed score != map max"
                    );
                    for numerics in [Numerics::Fast, Numerics::LibjxlExact] {
                        let params = ButteraugliParams::new()
                            .with_numerics(numerics)
                            .with_compute_diffmap(true);
                        let result = match lut {
                            None => {
                                let a: Vec<RGB8> = reference
                                    .px
                                    .chunks_exact(3)
                                    .map(|c| RGB8::new(c[0], c[1], c[2]))
                                    .collect();
                                let b: Vec<RGB8> = dist
                                    .px
                                    .chunks_exact(3)
                                    .map(|c| RGB8::new(c[0], c[1], c[2]))
                                    .collect();
                                butteraugli(Img::new(&a[..], w, h), Img::new(&b[..], w, h), &params)
                            }
                            Some(lut) => {
                                let lin = |img: &Image| -> Vec<RGB<f32>> {
                                    img.px
                                        .chunks_exact(3)
                                        .map(|c| {
                                            RGB::new(
                                                lut[c[0] as usize],
                                                lut[c[1] as usize],
                                                lut[c[2] as usize],
                                            )
                                        })
                                        .collect()
                                };
                                let (a, b) = (lin(&reference), lin(dist));
                                butteraugli_linear(
                                    Img::new(&a[..], w, h),
                                    Img::new(&b[..], w, h),
                                    &params,
                                )
                            }
                        }
                        .unwrap();
                        let crate_map: Vec<f32> =
                            result.diffmap.as_ref().unwrap().pixels().collect();
                        let d = map_diff(&lj.map, &crate_map);
                        let (lmax, lp3) = (max_norm(&lj.map), pnorm3(&lj.map));
                        writeln!(
                            tsv,
                            "{name}\t{w}\t{h}\t{dname}\t{arm}\t{numerics:?}\t{lmax:.17e}\t{:.17e}\t{:e}\t{lp3:.17e}\t{:.17e}\t{:e}\t{}\t{:e}\t{}",
                            result.score,
                            (result.score - lmax).abs(),
                            result.pnorm_3,
                            (result.pnorm_3 - lp3).abs(),
                            d.mismatched,
                            d.max_abs,
                            d.max_ulp
                        )
                        .unwrap();
                    }
                }
            }
            eprintln!("{name} {w}x{h} done");
        }
    }
    fs::write(out_dir.join("pairs.tsv"), &tsv).unwrap();
    fs::write(out_dir.join("libjxl_ingress.tsv"), &cross).unwrap();
    fs::write(out_dir.join("luts.txt"), &report).unwrap();
}
