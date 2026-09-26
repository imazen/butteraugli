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
#[cfg(not(feature = "multirate"))]
mod blur;
#[path = "multirate_blur.rs"]
#[cfg(feature = "multirate")]
mod blur;

const CANDIDATE: &str = if cfg!(feature = "wide-malta") {
    if cfg!(feature = "row-malta") {
        "wide-row-malta"
    } else {
        "wide-full-malta"
    }
} else if cfg!(feature = "row-malta") {
    "row-malta"
} else if cfg!(feature = "full-malta") {
    if cfg!(feature = "coarse-gaussian") {
        "coarse-full-malta"
    } else {
        "full-malta"
    }
} else if cfg!(feature = "native-gaussian") {
    "native-gaussian"
} else if cfg!(feature = "native-mask") {
    "native-mask"
} else if cfg!(feature = "tiles") {
    if cfg!(feature = "planar") {
        "planar-tiles"
    } else {
        "tiles"
    }
} else if cfg!(feature = "lattice") {
    if cfg!(feature = "phase-rows") && cfg!(feature = "row-tiles") {
        "phase-tiles"
    } else if cfg!(feature = "phase-rows") {
        "phase-rows"
    } else if cfg!(feature = "row-tiles") {
        "row-tiles"
    } else if cfg!(feature = "row-psycho") {
        "row-psycho"
    } else if cfg!(feature = "coarse-gaussian") {
        "coarse-gaussian"
    } else if cfg!(feature = "stream-blur") {
        "stream-blur"
    } else if cfg!(feature = "planar") {
        "planar"
    } else {
        "lattice"
    }
} else if cfg!(feature = "bounded") {
    "bounded"
} else if cfg!(feature = "stable-peak") {
    "stable-peak"
} else if cfg!(feature = "reference-regions") {
    "reference-regions"
} else if cfg!(feature = "stratified") {
    if cfg!(feature = "peak-stratified") {
        "peak-stratified"
    } else if cfg!(feature = "anchored-pool") {
        "anchored-pool"
    } else {
        "stratified"
    }
} else if cfg!(feature = "refined2") {
    "refined2"
} else if cfg!(feature = "refined1") {
    "refined1"
} else if cfg!(feature = "refined") {
    "refined"
} else if cfg!(feature = "physical") {
    "physical"
} else if cfg!(feature = "perceptual") {
    "perceptual"
} else if cfg!(feature = "pooled") {
    "pooled"
} else if cfg!(feature = "sparse") {
    "sparse"
} else if cfg!(feature = "compact4") {
    "compact4"
} else if cfg!(feature = "compact") {
    "compact"
} else if cfg!(feature = "multirate") {
    "multirate"
} else {
    "box3"
};
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
#[allow(
    clippy::implicit_saturating_sub,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]
mod shared_malta;
#[cfg(not(any(
    feature = "compact4",
    feature = "sparse",
    feature = "lattice",
    feature = "physical"
)))]
use shared_malta as malta;
#[cfg(feature = "full-malta")]
mod full_malta;
#[cfg(feature = "physical")]
mod half_malta_bank;
#[cfg(all(feature = "full-malta", not(feature = "row-malta")))]
use full_malta as malta;
#[cfg(feature = "row-malta")]
mod malta {
    pub(crate) use crate::full_malta::sampled_rows_diff_map as malta_diff_map;
}
#[cfg(all(
    feature = "compact4",
    not(any(feature = "sparse", feature = "lattice", feature = "physical"))
))]
#[path = "directional_malta.rs"]
mod malta;
#[cfg(all(
    any(feature = "sparse", feature = "lattice"),
    not(any(feature = "physical", feature = "full-malta"))
))]
#[path = "sparse_malta.rs"]
mod malta;
#[cfg(all(feature = "physical", not(feature = "full-malta")))]
#[path = "physical_malta.rs"]
mod malta;
#[cfg(any(feature = "sparse", feature = "lattice", feature = "physical"))]
mod malta_bank;
#[path = "../../butteraugli/src/mask.rs"]
mod mask;
#[path = "../../butteraugli/src/opsin.rs"]
#[allow(clippy::excessive_precision, clippy::needless_range_loop)]
mod opsin;
#[path = "../../butteraugli/src/psycho.rs"]
#[allow(clippy::excessive_precision, clippy::needless_range_loop)]
mod shared_psycho;
#[cfg(any(not(feature = "compact"), feature = "physical"))]
use shared_psycho as psycho;
#[cfg(all(feature = "compact", not(feature = "physical")))]
#[path = "compact_psycho.rs"]
mod psycho;

use butteraugli::{ButteraugliError, ButteraugliParams};
use std::error::Error;
use std::io::{BufWriter, Write};

mod ingress;
use ingress::load;

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

#[cfg(feature = "bounded")]
mod bounded_diff;
mod learned;
#[cfg(any(feature = "pooled", feature = "perceptual"))]
mod paired_pool;
#[cfg(feature = "perceptual")]
mod perceptual_pool;
#[cfg(feature = "refined")]
mod refined;
#[path = "resources.rs"]
mod resources;
mod resources_rgb8;
#[cfg(feature = "stream-blur")]
mod stream_blur;
mod strips;
#[cfg(feature = "tiles")]
mod tiles;

fn candidate_encoded(
    a: &ingress::EncodedRows<'_>,
    b: &ingress::EncodedRows<'_>,
    rows: usize,
    params: &ButteraugliParams,
) -> Result<diff::InternalResult, Box<dyn Error>> {
    #[cfg(feature = "refined")]
    {
        refined::compute(a, b, rows, params)
    }
    #[cfg(all(feature = "perceptual", not(feature = "refined")))]
    {
        perceptual_pool::compute(a, b, rows, params)
    }
    #[cfg(all(
        feature = "pooled",
        not(any(feature = "perceptual", feature = "refined"))
    ))]
    {
        paired_pool::compute(a, b, rows, params)
    }
    #[cfg(not(any(feature = "pooled", feature = "perceptual")))]
    {
        strips::compute_encoded(a, b, rows, params)
    }
}
mod student;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args: Vec<_> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|arg| arg == "--bench-direct") {
        return resources_rgb8::bench_direct(&args);
    }
    if args.first().is_some_and(|arg| arg == "--memory-native") {
        if args.len() != 4 {
            return Err("usage: --memory-native ROWS REF DIST".into());
        }
        let a = ingress::decode(&args[2])?;
        let b = ingress::decode(&args[3])?;
        let a = ingress::EncodedRows::from_image(&a)?;
        let b = ingress::EncodedRows::from_image(&b)?;
        let result = candidate_encoded(&a, &b, args[1].parse()?, &ButteraugliParams::default())?;
        println!(
            "{CANDIDATE}-native-strip\t{}\t{}\t{}\t{}",
            a.width, a.height, result.score, result.pnorm_3
        );
        std::hint::black_box(result);
        return Ok(());
    }
    if args.first().is_some_and(|arg| arg == "--student") {
        return learned::run(&args);
    }
    if args.first().is_some_and(|arg| arg == "--resource-crops") {
        return resources_rgb8::crops(&args);
    }
    if args.first().is_some_and(|arg| arg == "--render-dense") {
        return resources_rgb8::render_dense(&args);
    }
    if args.first().is_some_and(|arg| arg == "--export-edges") {
        return resources_rgb8::export_edges(&args);
    }
    if matches!(
        args.first().map(String::as_str),
        Some("--bench-rgb8" | "--memory-rgb8")
    ) {
        return resources_rgb8::run(&args);
    }
    if args.first().is_some_and(|a| a == "--bench-student") {
        return resources_rgb8::bench_student(&args);
    }
    if matches!(
        args.first().map(String::as_str),
        Some("--bench" | "--bench-features" | "--memory")
    ) {
        return resources::run(&args);
    }
    let native_rows = if args.first().is_some_and(|a| a == "--native-strip") {
        if args.len() != 5 {
            return Err("usage: --native-strip ROWS REF DIST MAP".into());
        }
        let rows = args[1].parse::<usize>()?;
        args.drain(..2);
        Some(rows)
    } else {
        None
    };
    let strip = args.first().is_some_and(|a| a == "--strip");
    if strip {
        args.remove(0);
    }
    if args.len() != 3 {
        return Err("usage: margarine-box3 REF DIST DIFFMAP.f32le".into());
    }
    let (result, w, h) = if native_rows.is_some()
        || cfg!(any(
            feature = "pooled",
            feature = "perceptual",
            feature = "bounded"
        )) {
        let a = ingress::decode(&args[0])?;
        let b = ingress::decode(&args[1])?;
        let a = ingress::EncodedRows::from_image(&a)?;
        let b = ingress::EncodedRows::from_image(&b)?;
        (
            candidate_encoded(
                &a,
                &b,
                native_rows.unwrap_or(a.height),
                &ButteraugliParams::default(),
            )?,
            a.width,
            a.height,
        )
    } else {
        let (reference, w, h) = load(&args[0])?;
        let (distorted, dw, dh) = load(&args[1])?;
        if (w, h) != (dw, dh) {
            return Err("image dimensions differ".into());
        }
        let result = if strip {
            strips::compute(
                &reference,
                &distorted,
                w,
                h,
                3 * w,
                32,
                &ButteraugliParams::default(),
            )?
        } else {
            diff::compute_butteraugli_linear_impl(
                &reference,
                &distorted,
                w,
                h,
                &ButteraugliParams::default(),
                &enough::Unstoppable,
            )?
        };
        (result, w, h)
    };
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
        "{}\t{w}\t{h}\t{:.17}\t{:.17}\t{:.17}\t{:.17}\t{:.17}\t{}",
        if native_rows.is_some() {
            format!("{CANDIDATE}-native-strip")
        } else if strip {
            format!("{CANDIDATE}-strip")
        } else {
            CANDIDATE.to_owned()
        },
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

#[cfg(feature = "row-psycho")]
mod row_psycho;
