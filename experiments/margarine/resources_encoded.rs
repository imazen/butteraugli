//! Native-precision resource probes, including encoded-to-linear conversion.
use super::{ButteraugliParams, ingress};
use image_io::DynamicImage;
use std::{error::Error, hint::black_box, path::Path, time::Duration};

fn teacher(
    a: &DynamicImage,
    b: &DynamicImage,
) -> Result<butteraugli::ButteraugliResult, Box<dyn Error>> {
    let a = ingress::EncodedRows::from_image(a)?;
    let b = ingress::EncodedRows::from_image(b)?;
    let a = butteraugli::Img::new(a.linear_rgb(), a.width, a.height);
    let b = butteraugli::Img::new(b.linear_rgb(), b.width, b.height);
    Ok(butteraugli::butteraugli_linear(
        a.as_ref(),
        b.as_ref(),
        &ButteraugliParams::default().with_compute_diffmap(true),
    )?)
}

pub(super) fn run(args: &[String]) -> Result<(), Box<dyn Error>> {
    let memory = args[0] == "--memory-encoded-teacher";
    if args.len() != if memory { 3 } else { 5 } {
        return Err(
            "usage: --bench-encoded ROWS REF DIST NEW.json | --memory-encoded-teacher REF DIST"
                .into(),
        );
    }
    let offset = if memory { 1 } else { 2 };
    let a = ingress::decode(&args[offset])?;
    let b = ingress::decode(&args[offset + 1])?;
    ingress::EncodedRows::from_image(&a)?;
    ingress::EncodedRows::from_image(&b)?;
    if (a.width(), a.height()) != (b.width(), b.height()) || a == b {
        return Err("benchmark needs a distinct pair with matching dimensions".into());
    }
    if memory {
        let result = teacher(&a, &b)?;
        println!("teacher-native\t{}\t{}", result.score, result.pnorm_3);
        black_box(result);
        return Ok(());
    }
    let rows = args[1].parse::<usize>()?;
    if rows == 0 || Path::new(&args[4]).exists() {
        return Err("invalid row count or existing result".into());
    }
    let (ca, cb) = (a.clone(), b.clone());
    let result = zenbench::run(|suite| {
        suite.compare(
            format!("encoded_{}x{}_rows{rows}", a.width(), a.height()),
            |group| {
                group
                    .config()
                    .min_rounds(20)
                    .max_rounds(40)
                    .max_wall_time(Duration::from_secs(300))
                    .warmup_time(Duration::from_millis(200));
                group.bench("teacher_metric", move |bench| {
                    bench.iter(|| teacher(black_box(&a), black_box(&b)).unwrap())
                });
                group.bench(format!("{}_metric", super::CANDIDATE), move |bench| {
                    bench.iter(|| {
                        super::candidate_encoded(
                            &ingress::EncodedRows::from_image(black_box(&ca)).unwrap(),
                            &ingress::EncodedRows::from_image(black_box(&cb)).unwrap(),
                            rows,
                            &ButteraugliParams::default(),
                        )
                        .unwrap()
                    })
                });
                let (rp, dp) = (args[2].clone(), args[3].clone());
                group.bench("teacher_decode", move |bench| {
                    bench.iter(|| {
                        teacher(
                            &ingress::decode(&rp).unwrap(),
                            &ingress::decode(&dp).unwrap(),
                        )
                        .unwrap()
                    })
                });
                let (rp, dp) = (args[2].clone(), args[3].clone());
                group.bench(format!("{}_decode", super::CANDIDATE), move |bench| {
                    bench.iter(|| {
                        let (a, b) = (ingress::decode(&rp).unwrap(), ingress::decode(&dp).unwrap());
                        super::candidate_encoded(
                            &ingress::EncodedRows::from_image(&a).unwrap(),
                            &ingress::EncodedRows::from_image(&b).unwrap(),
                            rows,
                            &ButteraugliParams::default(),
                        )
                        .unwrap()
                    })
                });
            },
        );
    });
    result.save(&args[4])?;
    Ok(())
}
