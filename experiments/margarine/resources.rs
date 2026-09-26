//! Resource measurements on caller-supplied real pairs. Decoding and conversion
//! occur before zenbench timing. Memory mode measures one arm in a fresh process.
use super::*;
use butteraugli::{Img, RGB};
use std::hint::black_box;
use std::path::Path;
use std::time::Duration;

fn rgb(flat: &[f32]) -> Vec<RGB<f32>> {
    flat.chunks_exact(3)
        .map(|p| RGB::new(p[0], p[1], p[2]))
        .collect()
}

pub(super) fn run(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.len() != 4 {
        return Err(
            "usage: --bench REF DIST NEW_RESULTS.json | --memory teacher|box3 REF DIST".into(),
        );
    }
    let memory = args[0] == "--memory";
    let (ref_path, dist_path) = if memory {
        (&args[2], &args[3])
    } else {
        (&args[1], &args[2])
    };
    let (reference, w, h) = load(ref_path)?;
    let (distorted, dw, dh) = load(dist_path)?;
    if (w, h) != (dw, dh) || reference == distorted {
        return Err("resource measurement needs distinct images with matching dimensions".into());
    }
    let params = ButteraugliParams::default().with_compute_diffmap(true);
    if memory {
        let (score, p3) = match args[1].as_str() {
            "teacher" => {
                let a = Img::new(rgb(&reference), w, h);
                let b = Img::new(rgb(&distorted), w, h);
                drop(reference);
                drop(distorted);
                let result = butteraugli::butteraugli_linear(a.as_ref(), b.as_ref(), &params)?;
                black_box(&result);
                (result.score, result.pnorm_3)
            }
            "box3" => {
                let result = diff::compute_butteraugli_linear_impl(
                    &reference,
                    &distorted,
                    w,
                    h,
                    &params,
                    &enough::Unstoppable,
                )?;
                black_box(&result);
                (result.score, result.pnorm_3)
            }
            "features372" => {
                let a = student::rgba(&reference);
                drop(reference);
                let b = student::rgba(&distorted);
                drop(distorted);
                let features = student::extract(&student::extractor(), &a, &b, w, h, w * 16)?;
                println!(
                    "features372\t{w}\t{h}\t{} features; no trained score",
                    features.len()
                );
                black_box(features);
                return Ok(());
            }
            "box3-strip" => {
                let result = strips::compute(&reference, &distorted, w, h, 3 * w, 32, &params)?;
                black_box(&result);
                (result.score, result.pnorm_3)
            }
            _ => return Err("memory arm must be teacher, box3 or box3-strip".into()),
        };
        println!("{}\t{w}\t{h}\t{score}\t{p3}", args[1]);
        return Ok(());
    }
    if Path::new(&args[3]).exists() {
        return Err("result output already exists".into());
    }
    let a = Img::new(rgb(&reference), w, h);
    let b = Img::new(rgb(&distorted), w, h);
    let teacher_params = params.clone();
    let features = if args[0] == "--bench-features" {
        Some((
            student::extractor(),
            student::rgba(&reference),
            student::rgba(&distorted),
        ))
    } else {
        None
    };
    let result = zenbench::run(|suite| {
        suite.compare(format!("cold_pair_{w}x{h}"), |group| {
            group
                .config()
                .min_rounds(20)
                .max_rounds(40)
                .warmup_time(Duration::from_millis(200));
            group.bench("teacher", move |bench| {
                bench.iter(|| {
                    butteraugli::butteraugli_linear(
                        black_box(a.as_ref()),
                        black_box(b.as_ref()),
                        &teacher_params,
                    )
                    .unwrap()
                });
            });
            if let Some((scorer, a, b)) = features {
                group.bench("features372_only", move |bench| {
                    bench.iter(|| {
                        student::extract(&scorer, black_box(&a), black_box(&b), w, h, w * 16)
                            .unwrap()
                    });
                });
            }
            group.bench("box3", move |bench| {
                bench.iter(|| {
                    diff::compute_butteraugli_linear_impl(
                        black_box(&reference),
                        black_box(&distorted),
                        w,
                        h,
                        &params,
                        &enough::Unstoppable,
                    )
                    .unwrap()
                });
            });
        });
    });
    result.save(&args[3])?;
    result.print_report();
    Ok(())
}
