//! Native RGB8 cost probes. Both arms consume the same decoded sRGB samples;
//! these modes reject higher precision input rather than narrowing it.
use super::student;
use butteraugli::{ButteraugliParams, Img, RGB8};
use image_io::{DynamicImage, ImageReader};
use std::{error::Error, hint::black_box, path::Path, time::Duration};
use zensim::{PixelFormat, StridedBytes, Zensim};

/// Persist the measured extractor's features without assigning a quality score.
/// RGB8 takes the measured native path; higher precision uses the shared ingress.
pub(super) fn export_edges(args: &[String]) -> Result<(), Box<dyn Error>> {
    use std::io::Write;
    if args.len() != 4 {
        return Err("usage: --export-edges REF DIST NEW.tsv".into());
    }
    let a = ImageReader::open(&args[1])?.decode()?;
    let b = ImageReader::open(&args[2])?.decode()?;
    if (a.width(), a.height()) != (b.width(), b.height()) {
        return Err("feature pair dimensions differ".into());
    }
    let (w, h) = (a.width() as usize, a.height() as usize);
    let features = match (a, b) {
        (DynamicImage::ImageRgb8(a), DynamicImage::ImageRgb8(b)) => extract(
            &student::edge_extractor(),
            StridedBytes::try_new(a.as_raw(), w, h, w * 3, PixelFormat::Srgb8Rgb)?,
            StridedBytes::try_new(b.as_raw(), w, h, w * 3, PixelFormat::Srgb8Rgb)?,
            true,
            true,
        )?,
        (a, b) => {
            let (a, _, _) = super::ingress::convert(a)?;
            let (b, _, _) = super::ingress::convert(b)?;
            let (a, b) = (student::rgba(&a), student::rgba(&b));
            extract(
                &student::edge_extractor(),
                StridedBytes::try_new(&a, w, h, w * 16, PixelFormat::LinearF32Rgba)?,
                StridedBytes::try_new(&b, w, h, w * 16, PixelFormat::LinearF32Rgba)?,
                true,
                true,
            )?
        }
    };
    let mut out = std::io::BufWriter::new(std::fs::File::create_new(&args[3])?);
    write!(out, "width\theight")?;
    for i in (0..228).filter(|&i| student::edge_feature(i)) {
        write!(out, "\tfeature_{i:03}")?;
    }
    writeln!(out)?;
    write!(out, "{w}\t{h}")?;
    for value in &features {
        write!(out, "\t{value:.17e}")?;
    }
    writeln!(out)?;
    out.flush()?;
    println!(
        "Persisted {} finite edge features for {w}x{h}",
        features.len()
    );
    Ok(())
}

/// Prepare twenty log-spaced reference sizes, capped at native size/4096.
/// The caller records source lineage and hashes and registers the variant set.
pub(super) fn render_dense(args: &[String]) -> Result<(), Box<dyn Error>> {
    use std::io::Write;
    if args.len() != 3 {
        return Err("usage: --render-dense SOURCE NEW_DIRECTORY".into());
    }
    let image = match ImageReader::open(&args[1])?.decode()? {
        DynamicImage::ImageRgb8(image) => image,
        DynamicImage::ImageLuma8(image) => DynamicImage::ImageLuma8(image).into_rgb8(),
        _ => {
            return Err("dense SDR renderer accepts RGB8 or losslessly expanded gray8 only".into());
        }
    };
    let (w, h) = image.dimensions();
    let limit = w.max(h).min(4096);
    if limit < 64 {
        return Err("source is too small for twenty distinct log-spaced sizes".into());
    }
    let output = Path::new(&args[2]);
    std::fs::create_dir(output)?;
    let mut progress = std::fs::File::create(output.join("progress.log"))?;
    let mut manifest = std::fs::File::create(output.join("renditions.tsv"))?;
    writeln!(manifest, "width\theight\tpath\tkernel")?;
    let mut seen = std::collections::BTreeSet::new();
    for i in 0..20 {
        let target = (32.0 * (f64::from(limit) / 32.0).powf(f64::from(i) / 19.0)).round() as u32;
        let target = target.min(limit);
        let (rw, rh) = if w >= h {
            (
                target,
                (u64::from(h) * u64::from(target) / u64::from(w)).max(1) as u32,
            )
        } else {
            (
                (u64::from(w) * u64::from(target) / u64::from(h)).max(1) as u32,
                target,
            )
        };
        if rw > w || rh > h || !seen.insert((rw, rh)) {
            return Err("invalid or duplicated rendition dimensions".into());
        }
        let path = output.join(format!("{rw}x{rh}.png"));
        let resized =
            image_io::imageops::resize(&image, rw, rh, image_io::imageops::FilterType::Lanczos3);
        resized.save(&path)?;
        writeln!(
            manifest,
            "{rw}\t{rh}\t{}\tlanczos3-encoded-srgb",
            path.display()
        )?;
        manifest.flush()?;
        writeln!(progress, "{}/20: persisted {rw}x{rh}", i + 1)?;
        progress.flush()?;
        println!("{}/20: persisted {rw}x{rh}", i + 1);
    }
    Ok(())
}

/// Exact center crops for resource probes; no resampling, synthesis or upscaling.
pub(super) fn crops(args: &[String]) -> Result<(), Box<dyn Error>> {
    use std::io::Write;
    if args.len() != 4 {
        return Err("usage: --resource-crops REF DIST NEW_DIRECTORY".into());
    }
    let (a, b) = (decode(&args[1])?, decode(&args[2])?);
    if a.dimensions() != b.dimensions() || a.width().min(a.height()) < 1024 {
        return Err("resource crop source pair must match and contain 1024-square crops".into());
    }
    let out = Path::new(&args[3]);
    std::fs::create_dir(out)?;
    let mut log = std::fs::File::create(out.join("progress.log"))?;
    let mut manifest = std::fs::File::create(out.join("crops.tsv"))?;
    writeln!(manifest, "width\theight\tx\ty\treference\tdistorted")?;
    for (w, h) in [(64, 64), (256, 256), (1024, 1024), a.dimensions()] {
        let (x, y) = ((a.width() - w) / 2, (a.height() - h) / 2);
        let name = format!("{w}x{h}");
        let (rp, dp) = (
            out.join(format!("{name}-ref.png")),
            out.join(format!("{name}-dist.png")),
        );
        let ac = image_io::imageops::crop_imm(&a, x, y, w, h).to_image();
        let bc = image_io::imageops::crop_imm(&b, x, y, w, h).to_image();
        if ac == bc {
            return Err(
                format!("identity crop at {name}; cannot benchmark comparison work").into(),
            );
        }
        ac.save(&rp)?;
        bc.save(&dp)?;
        writeln!(
            manifest,
            "{w}\t{h}\t{x}\t{y}\t{}\t{}",
            rp.display(),
            dp.display()
        )?;
        writeln!(log, "Persisted {name} at x={x}, y={y}")?;
        log.flush()?;
        println!("Persisted {name}");
    }
    Ok(())
}

fn decode(path: &str) -> Result<image_io::RgbImage, Box<dyn Error>> {
    match ImageReader::open(path)?.decode()? {
        DynamicImage::ImageRgb8(image) => Ok(image),
        _ => Err("native RGB8 probe requires RGB8 input; no precision conversion".into()),
    }
}

fn rgb(image: &image_io::RgbImage) -> Img<Vec<RGB8>> {
    Img::new(
        image
            .pixels()
            .map(|p| RGB8::new(p[0], p[1], p[2]))
            .collect(),
        image.width() as usize,
        image.height() as usize,
    )
}

fn extract(
    scorer: &Zensim,
    a: StridedBytes<'_>,
    b: StridedBytes<'_>,
    strips: bool,
    edges: bool,
) -> Result<Vec<f64>, Box<dyn Error>> {
    let result = if strips {
        // Edge profile: radius 5, one pass, four scales. 5 * 2^3 = 40
        // input rows cover the coarsest blur. Use 64 so a one-row final
        // interior still has a >=64-row reference (short references pad to
        // 64, but strip distorted planes do not). 256-row interiors align
        // with the pinned implementation's 32-row bands at every scale.
        scorer.compute_streaming_strips(&a, &b, 256, if edges { 64 } else { 128 })?
    } else if edges {
        scorer.compute(&a, &b)?
    } else {
        scorer.compute_all_features(&a, &b)?
    };
    let features = result.into_features();
    if features.len() != 228 || !features.iter().all(|v| v.is_finite()) {
        return Err("unexpected or nonfinite feature vector".into());
    }
    if edges {
        Ok(features
            .into_iter()
            .enumerate()
            .filter_map(|(i, v)| student::edge_feature(i).then_some(v))
            .collect())
    } else {
        Ok(features)
    }
}

pub(super) fn run(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.len() != 4 {
        return Err("usage: --bench-rgb8 REF DIST NEW.json | --memory-rgb8 teacher|features228|features228-strips REF DIST".into());
    }
    let memory = args[0] == "--memory-rgb8";
    let (rp, dp) = if memory {
        (&args[2], &args[3])
    } else {
        (&args[1], &args[2])
    };
    let (a, b) = (decode(rp)?, decode(dp)?);
    if a.dimensions() != b.dimensions() || a == b {
        return Err("resource measurement needs distinct images with matching dimensions".into());
    }
    let (w, h) = (a.width() as usize, a.height() as usize);
    let params = ButteraugliParams::default().with_compute_diffmap(true);
    if memory && args[1] == "teacher" {
        let (ra, rb) = (rgb(&a), rgb(&b));
        drop(a);
        drop(b);
        let result = butteraugli::butteraugli(ra.as_ref(), rb.as_ref(), &params)?;
        println!(
            "rgb8_teacher\t{w}\t{h}\t{}\t{}",
            result.score, result.pnorm_3
        );
        black_box(result);
        return Ok(());
    }
    let av = StridedBytes::try_new(a.as_raw(), w, h, w * 3, PixelFormat::Srgb8Rgb)?;
    let bv = StridedBytes::try_new(b.as_raw(), w, h, w * 3, PixelFormat::Srgb8Rgb)?;
    if memory {
        let strips = match args[1].as_str() {
            "features228" | "features168" => false,
            "features228-strips" | "features168-strips64" => true,
            _ => return Err("unsupported RGB8 memory arm".into()),
        };
        let edges = matches!(args[1].as_str(), "features168" | "features168-strips64");
        let scorer = if edges {
            student::edge_extractor()
        } else {
            student::extractor(228)
        };
        let features = extract(
            &scorer.with_parallel(edges || !strips),
            av,
            bv,
            strips,
            edges,
        )?;
        println!(
            "rgb8_{}\t{w}\t{h}\t{} features; no trained score",
            args[1],
            features.len()
        );
        black_box(features);
        return Ok(());
    }
    if Path::new(&args[3]).exists() {
        return Err("result output already exists".into());
    }
    let (ra, rb) = (rgb(&a), rgb(&b));
    let (as_strip, bs_strip) = (a.clone(), b.clone());
    let (as_edge, bs_edge) = (a.clone(), b.clone());
    let (as_edge_strip, bs_edge_strip) = (a.clone(), b.clone());
    let scorer = student::extractor(228);
    let strip_scorer = student::extractor(228).with_parallel(false);
    let edge_scorer = student::edge_extractor();
    let edge_strip_scorer = student::edge_extractor();
    let result = zenbench::run(|suite| {
        suite.compare(format!("cold_rgb8_pair_{w}x{h}"), |group| {
            group
                .config()
                .min_rounds(20)
                .max_rounds(40)
                .warmup_time(Duration::from_millis(200));
            group.bench("teacher_rgb8", move |bench| {
                bench.iter(|| {
                    butteraugli::butteraugli(
                        black_box(ra.as_ref()),
                        black_box(rb.as_ref()),
                        &params,
                    )
                    .unwrap()
                })
            });
            group.bench("features228_rgb8_only", move |bench| {
                let av =
                    StridedBytes::try_new(a.as_raw(), w, h, w * 3, PixelFormat::Srgb8Rgb).unwrap();
                let bv =
                    StridedBytes::try_new(b.as_raw(), w, h, w * 3, PixelFormat::Srgb8Rgb).unwrap();
                bench.iter(|| extract(&scorer, black_box(av), black_box(bv), false, false).unwrap())
            });
            group.bench("features228_rgb8_strips_only", move |bench| {
                let av =
                    StridedBytes::try_new(as_strip.as_raw(), w, h, w * 3, PixelFormat::Srgb8Rgb)
                        .unwrap();
                let bv =
                    StridedBytes::try_new(bs_strip.as_raw(), w, h, w * 3, PixelFormat::Srgb8Rgb)
                        .unwrap();
                bench.iter(|| {
                    extract(&strip_scorer, black_box(av), black_box(bv), true, false).unwrap()
                })
            });
            group.bench("features168_rgb8_only", move |bench| {
                let av =
                    StridedBytes::try_new(as_edge.as_raw(), w, h, w * 3, PixelFormat::Srgb8Rgb)
                        .unwrap();
                let bv =
                    StridedBytes::try_new(bs_edge.as_raw(), w, h, w * 3, PixelFormat::Srgb8Rgb)
                        .unwrap();
                bench.iter(|| {
                    extract(&edge_scorer, black_box(av), black_box(bv), false, true).unwrap()
                })
            });
            group.bench("features168_rgb8_strips64_only", move |bench| {
                let av = StridedBytes::try_new(
                    as_edge_strip.as_raw(),
                    w,
                    h,
                    w * 3,
                    PixelFormat::Srgb8Rgb,
                )
                .unwrap();
                let bv = StridedBytes::try_new(
                    bs_edge_strip.as_raw(),
                    w,
                    h,
                    w * 3,
                    PixelFormat::Srgb8Rgb,
                )
                .unwrap();
                bench.iter(|| {
                    extract(&edge_strip_scorer, black_box(av), black_box(bv), true, true).unwrap()
                });
            });
        });
    });
    result.save(&args[3])?;
    result.print_report();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimal_edge_halo_matches_whole_features_across_seams() {
        for (w, h) in [(65, 801), (128, 769), (128, 1024)] {
            let a: Vec<_> = (0..w * h * 3)
                .map(|i| ((i * 31 + i / (w * 3) * 7) % 256) as u8)
                .collect();
            let b: Vec<_> = a
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    if (i / (w * 3)) % 256 < 7 {
                        v.saturating_sub(17)
                    } else {
                        v
                    }
                })
                .collect();
            let av = StridedBytes::try_new(&a, w, h, w * 3, PixelFormat::Srgb8Rgb).unwrap();
            let bv = StridedBytes::try_new(&b, w, h, w * 3, PixelFormat::Srgb8Rgb).unwrap();
            let scorer = student::edge_extractor();
            let whole = extract(&scorer, av, bv, false, true).unwrap();
            let strips = extract(&scorer, av, bv, true, true).unwrap();
            for (i, (a, b)) in whole.iter().zip(strips).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0),
                    "{w}x{h} feature {i}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn native_features_accept_strided_rows() {
        let (w, h) = (17, 19);
        let a: Vec<_> = (0..w * h * 3).map(|i| (i * 31) as u8).collect();
        let b: Vec<_> = a.iter().map(|v| v.saturating_sub(7)).collect();
        let stride = w * 3 + 7;
        let padded = |v: &[u8]| {
            let mut out = vec![255; stride * h];
            for y in 0..h {
                out[y * stride..y * stride + w * 3].copy_from_slice(&v[y * w * 3..(y + 1) * w * 3]);
            }
            out
        };
        let (ap, bp) = (padded(&a), padded(&b));
        let view =
            |v, stride| StridedBytes::try_new(v, w, h, stride, PixelFormat::Srgb8Rgb).unwrap();
        let scorer = student::extractor(228);
        assert_eq!(
            extract(&scorer, view(&a, w * 3), view(&b, w * 3), false, false).unwrap(),
            extract(&scorer, view(&ap, stride), view(&bp, stride), false, false).unwrap()
        );
        let full = extract(&scorer, view(&a, w * 3), view(&b, w * 3), false, false).unwrap();
        let edge = student::edge_extractor();
        let reduced = extract(&edge, view(&ap, stride), view(&bp, stride), false, true).unwrap();
        let selected: Vec<_> = full
            .into_iter()
            .enumerate()
            .filter_map(|(i, v)| student::edge_feature(i).then_some(v))
            .collect();
        assert_eq!(reduced.len(), 168);
        assert_eq!(reduced, selected);
    }
}
