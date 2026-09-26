//! Native RGB8 cost probes. Both arms consume the same decoded sRGB samples;
//! these modes reject higher precision input rather than narrowing it.
use super::student;
use butteraugli::{ButteraugliParams, Img, RGB8};
use image_io::{DynamicImage, ImageReader};
use std::{error::Error, hint::black_box, path::Path, time::Duration};
use zensim::{PixelFormat, StridedBytes, Zensim};

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
) -> Result<Vec<f64>, Box<dyn Error>> {
    let result = if strips {
        scorer.compute_streaming_strips(&a, &b, 256, 128)?
    } else {
        scorer.compute_all_features(&a, &b)?
    };
    let features = result.into_features();
    if features.len() != 228 || !features.iter().all(|v| v.is_finite()) {
        return Err("unexpected or nonfinite feature vector".into());
    }
    Ok(features)
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
            "features228" => false,
            "features228-strips" => true,
            _ => return Err("unsupported RGB8 memory arm".into()),
        };
        let features = extract(
            &student::extractor(228).with_parallel(!strips),
            av,
            bv,
            strips,
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
    let scorer = student::extractor(228);
    let strip_scorer = student::extractor(228).with_parallel(false);
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
                bench.iter(|| extract(&scorer, black_box(av), black_box(bv), false).unwrap())
            });
            group.bench("features228_rgb8_strips_only", move |bench| {
                let av =
                    StridedBytes::try_new(as_strip.as_raw(), w, h, w * 3, PixelFormat::Srgb8Rgb)
                        .unwrap();
                let bv =
                    StridedBytes::try_new(bs_strip.as_raw(), w, h, w * 3, PixelFormat::Srgb8Rgb)
                        .unwrap();
                bench.iter(|| extract(&strip_scorer, black_box(av), black_box(bv), true).unwrap())
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
            extract(&scorer, view(&a, w * 3), view(&b, w * 3), false).unwrap(),
            extract(&scorer, view(&ap, stride), view(&bp, stride), false).unwrap()
        );
    }
}
