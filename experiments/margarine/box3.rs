//! Full-resolution approximation control. Shares the source of the scoring
//! stages with Butteraugli, replacing only general Gaussian blurs. This is
//! deliberately not a claim of calibrated fidelity or a 4x performance win.
#![forbid(unsafe_code)]
// Shared source includes helpers unused by this one-shot experimental entry.
#![allow(dead_code)]
// Preserve the same arithmetic/codegen lint decisions as butteraugli/lib.rs.
#![allow(clippy::manual_midpoint, clippy::chunks_exact_to_as_chunks)]
// Shared kernels use the same legacy autoversion signatures as lib.rs.
#![allow(deprecated)]

extern crate image as image_io;

#[path = "box_blur.rs"]
mod blur;
#[path = "../../butteraugli/src/consts.rs"]
#[allow(clippy::inconsistent_digit_grouping, clippy::excessive_precision)]
mod consts;
#[path = "../../butteraugli/src/diff.rs"]
mod diff;
#[path = "../../butteraugli/src/blur.rs"]
#[allow(clippy::implicit_saturating_sub, clippy::needless_range_loop)]
mod exact_blur;
#[path = "../../butteraugli/src/image.rs"]
mod image;
#[path = "../../butteraugli/src/malta.rs"]
#[allow(clippy::implicit_saturating_sub, clippy::needless_range_loop)]
mod malta;
#[path = "../../butteraugli/src/mask.rs"]
mod mask;
#[path = "../../butteraugli/src/opsin.rs"]
#[allow(clippy::excessive_precision, clippy::needless_range_loop)]
mod opsin;
#[path = "../../butteraugli/src/psycho.rs"]
#[allow(clippy::excessive_precision, clippy::needless_range_loop)]
mod psycho;

use butteraugli::{ButteraugliError, ButteraugliParams};
use std::error::Error;
use std::io::{BufWriter, Write};

fn load(path: &str) -> Result<(Vec<f32>, usize, usize), Box<dyn Error>> {
    let input = image_io::ImageReader::open(path)?.decode()?;
    if input.color() != image_io::ColorType::Rgb8 {
        return Err("experiment requires common-ingress RGB8".into());
    }
    let rgb = input.into_rgb8();
    let linear = rgb
        .as_raw()
        .iter()
        .map(|&v| opsin::srgb_to_linear(v))
        .collect();
    Ok((linear, rgb.width() as usize, rgb.height() as usize))
}

// Same p/2p/4p aggregation as butteraugli/src/lib.rs::pnorm_slice. Iterate
// logical rows because this experiment's ImageF can contain padded storage.
fn pnorm(map: &image::ImageF, p: f64) -> f64 {
    let mut sums = [0.0f64; 3];
    for y in 0..map.height() {
        for &value in map.row(y) {
            let mut acc = f64::from(value).powf(p);
            sums[0] += acc;
            acc *= acc;
            sums[1] += acc;
            acc *= acc;
            sums[2] += acc;
        }
    }
    let inv = 1.0 / (map.width() * map.height()) as f64;
    sums.iter()
        .enumerate()
        .map(|(i, &s)| (inv * s).powf(1.0 / (p * f64::from(1u32 << i))))
        .sum::<f64>()
        / 3.0
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: margarine-box3 REF DIST DIFFMAP.f32le".into());
    }
    let (reference, w, h) = load(&args[0])?;
    let (distorted, dw, dh) = load(&args[1])?;
    if (w, h) != (dw, dh) {
        return Err("image dimensions differ".into());
    }
    let result = diff::compute_butteraugli_linear_impl(
        &reference,
        &distorted,
        w,
        h,
        &ButteraugliParams::default(),
        &enough::Unstoppable,
    )?;
    let map = result.diffmap.as_ref().ok_or("missing diffmap")?;
    let mut out = BufWriter::new(std::fs::File::create_new(&args[2])?);
    for y in 0..map.height() {
        for value in map.row(y) {
            out.write_all(&value.to_le_bytes())?;
        }
    }
    out.flush()?;
    println!("mode\twidth\theight\tmax\tp1\tp2\tp3\tp6\tdiffmap");
    println!(
        "box3\t{w}\t{h}\t{:.17}\t{:.17}\t{:.17}\t{:.17}\t{:.17}\t{}",
        result.score,
        pnorm(map, 1.0),
        pnorm(map, 2.0),
        result.pnorm_3,
        pnorm(map, 6.0),
        args[2]
    );
    Ok(())
}

#[cfg(test)]
mod experiment_tests {
    use super::*;

    #[test]
    fn auxiliary_norms_match_public_butteraugli_pooling() {
        use butteraugli::{Img, RGB};
        let a = Img::new(vec![RGB::new(0.5, 0.5, 0.5); 32 * 32], 32, 32);
        let b = Img::new(
            (0..32 * 32)
                .map(|i| RGB::new(0.5 + (i % 5) as f32 / 100.0, 0.5, 0.5))
                .collect::<Vec<_>>(),
            32,
            32,
        );
        let teacher = butteraugli::butteraugli_linear(
            a.as_ref(),
            b.as_ref(),
            &ButteraugliParams::default().with_compute_diffmap(true),
        )
        .unwrap();
        let original = teacher.diffmap.as_ref().unwrap();
        let local = image::ImageF::from_vec(original.buf().to_vec(), 32, 32);
        for p in [1.0, 2.0, 6.0] {
            assert_eq!(pnorm(&local, p), teacher.pnorm(p).unwrap());
        }
    }

    #[test]
    fn retains_checkerboard_distortion_that_averaging_erases() {
        let reference = vec![0.5; 32 * 32 * 3];
        let distorted: Vec<_> = (0..32 * 32)
            .flat_map(|i| {
                let v = if (i % 32 + i / 32) % 2 == 0 {
                    0.25
                } else {
                    0.75
                };
                [v; 3]
            })
            .collect();
        let result = diff::compute_butteraugli_linear_impl(
            &reference,
            &distorted,
            32,
            32,
            &ButteraugliParams::default(),
            &enough::Unstoppable,
        )
        .unwrap();
        assert!(result.score.is_finite() && result.score > 0.0);
        assert!(result.pnorm_3.is_finite() && result.pnorm_3 > 0.0);
    }
}
