#![forbid(unsafe_code)]

extern crate image as image_io;
mod ingress;

use butteraugli::{ButteraugliParams, Img, ImgRef, RGB, butteraugli_linear};
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

// A negative control: no fitted constants and no full-resolution residual path.
// Average in linear light, with each odd-edge pixel counted once.
fn half_linear(input: ImgRef<'_, RGB<f32>>) -> Img<Vec<RGB<f32>>> {
    let w = input.width();
    let h = input.height();
    let mut output = Vec::with_capacity(w.div_ceil(2) * h.div_ceil(2));
    for y in (0..h).step_by(2) {
        for x in (0..w).step_by(2) {
            let mut sum = RGB::new(0.0, 0.0, 0.0);
            let mut n = 0.0;
            for yy in y..(y + 2).min(h) {
                for xx in x..(x + 2).min(w) {
                    let p = input.buf()[yy * input.stride() + xx];
                    sum.r += p.r;
                    sum.g += p.g;
                    sum.b += p.b;
                    n += 1.0;
                }
            }
            output.push(RGB::new(sum.r / n, sum.g / n, sum.b / n));
        }
    }
    Img::new(output, w.div_ceil(2), h.div_ceil(2))
}

fn load(path: &Path) -> Result<Img<Vec<RGB<f32>>>> {
    let (linear, w, h) = ingress::load(path)?;
    let pixels = linear
        .as_chunks::<3>()
        .0
        .iter()
        .map(|p| RGB::new(p[0], p[1], p[2]))
        .collect();
    Ok(Img::new(pixels, w, h))
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 || !matches!(args[0].as_str(), "teacher" | "half-control") {
        return Err(
            "usage: margarine-score teacher|half-control REF DIST DIFFMAP.f32le\n\
                    Assumes common sRGB ingress; output map is native scoring resolution."
                .into(),
        );
    }
    let mut reference = load(Path::new(&args[1]))?;
    let mut distorted = load(Path::new(&args[2]))?;
    if reference.width() != distorted.width() || reference.height() != distorted.height() {
        return Err("reference and distorted dimensions differ".into());
    }
    if args[0] == "half-control" {
        reference = half_linear(reference.as_ref());
        distorted = half_linear(distorted.as_ref());
    }
    let result = butteraugli_linear(
        reference.as_ref(),
        distorted.as_ref(),
        &ButteraugliParams::default().with_compute_diffmap(true),
    )?;
    let map = result.diffmap.as_ref().ok_or("missing requested diffmap")?;
    // create_new keeps a rerun from overwriting an earlier observation.
    let mut out = BufWriter::new(File::create_new(&args[3])?);
    for row in map.rows() {
        for value in row {
            out.write_all(&value.to_le_bytes())?;
        }
    }
    out.flush()?;
    println!("mode\twidth\theight\tmax\tp1\tp2\tp3\tp6\tdiffmap");
    println!(
        "{}\t{}\t{}\t{:.17}\t{:.17}\t{:.17}\t{:.17}\t{:.17}\t{}",
        args[0],
        map.width(),
        map.height(),
        result.score,
        result.pnorm(1.0).ok_or("missing p1")?,
        result.pnorm(2.0).ok_or("missing p2")?,
        result.pnorm_3,
        result.pnorm(6.0).ok_or("missing p6")?,
        args[3],
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strided_odd_edges_exclude_padding() {
        let tight: Vec<_> = (0..15).map(|i| RGB::new(i as f32, 0.0, 0.0)).collect();
        let mut padded = vec![RGB::new(f32::NAN, f32::NAN, f32::NAN); 21];
        for y in 0..3 {
            padded[y * 7..y * 7 + 5].copy_from_slice(&tight[y * 5..y * 5 + 5]);
        }
        let a = half_linear(Img::new(tight, 5, 3).as_ref());
        let b = half_linear(Img::new_stride(padded, 5, 3, 7).as_ref());
        assert_eq!(a.buf(), b.buf());
        assert_eq!(
            a.buf().iter().map(|p| p.r).collect::<Vec<_>>(),
            [3.0, 5.0, 6.5, 10.5, 12.5, 14.0]
        );
    }

    #[test]
    fn averaging_can_erase_a_real_distortion() {
        let reference = Img::new(vec![RGB::new(0.5, 0.5, 0.5); 32 * 32], 32, 32);
        let distorted = Img::new(
            (0..32 * 32)
                .map(|i| {
                    let v = if (i % 32 + i / 32) % 2 == 0 {
                        0.25
                    } else {
                        0.75
                    };
                    RGB::new(v, v, v)
                })
                .collect::<Vec<_>>(),
            32,
            32,
        );
        let params = ButteraugliParams::default();
        let full = butteraugli_linear(reference.as_ref(), distorted.as_ref(), &params).unwrap();
        let a = half_linear(reference.as_ref());
        let b = half_linear(distorted.as_ref());
        assert_eq!(a.buf(), b.buf());
        let half = butteraugli_linear(a.as_ref(), b.as_ref(), &params).unwrap();
        assert!(full.score > 0.0);
        assert_eq!(half.score, 0.0);
    }
}
