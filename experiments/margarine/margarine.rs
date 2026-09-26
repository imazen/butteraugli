//! Streamed Butteraugli-lineage approximation; qualification remains in progress.
#![forbid(unsafe_code)]
#![allow(dead_code, deprecated)]
#![allow(clippy::manual_midpoint, clippy::chunks_exact_to_as_chunks)]

include!("kernel.rs");
use std::io::{BufWriter, Write};

fn main() -> Result<(), Box<dyn Error>> {
    let mut paths = Vec::new();
    let mut all_scores = false;
    let mut diffmap = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                println!(
                    "Usage: margarine [--all-scores] [--diffmap NEW.f32le] REF DIST\n\
                    Encoded sRGB RGB/RGBA 8/16-bit; alpha must be opaque.\n\
                    Lower scores indicate less distortion. Output is JSON.\n\
                    Diffmaps contain row-major little-endian f32 values."
                );
                return Ok(());
            }
            "--all-scores" => all_scores = true,
            "--diffmap" => {
                if diffmap.is_some() {
                    return Err("--diffmap supplied more than once".into());
                }
                diffmap = Some(args.next().ok_or("--diffmap requires a new file path")?);
            }
            _ if arg.starts_with('-') => return Err(format!("unknown option: {arg}").into()),
            _ => paths.push(arg),
        }
    }
    if paths.len() != 2 {
        return Err("expected reference and distorted image paths; see --help".into());
    }
    if CANDIDATE != "simd-row-malta"
        || cfg!(any(
            feature = "coarse-gaussian",
            feature = "native-gaussian",
            feature = "native-mask",
            feature = "phase-rows",
            feature = "tiles",
            feature = "compact",
            feature = "sparse",
            feature = "pooled",
            feature = "perceptual",
            feature = "physical",
            feature = "simd-opsin",
            feature = "wide-malta"
        ))
    {
        return Err("this command requires the simd-malta,row-malta candidate without additional approximation features".into());
    }
    let reference = ingress::decode(&paths[0])?;
    let distorted = ingress::decode(&paths[1])?;
    let reference = ingress::EncodedRows::from_image(&reference)?;
    let distorted = ingress::EncodedRows::from_image(&distorted)?;
    if (reference.width, reference.height) != (distorted.width, distorted.height) {
        return Err("image dimensions differ".into());
    }
    let result = row_psycho::compute_geometry(
        &reference,
        &distorted,
        128,
        512,
        &ButteraugliParams::default(),
    )?;
    let map = result.diffmap.as_ref().ok_or("missing diffmap")?;
    if let Some(path) = diffmap {
        let mut out = BufWriter::new(std::fs::File::create_new(path)?);
        for y in 0..map.height() {
            for value in map.row(y) {
                out.write_all(&value.to_le_bytes())?;
            }
        }
        out.flush()?;
    }
    print!(
        "{{\"width\":{},\"height\":{},\"score\":{:.17},\"pooling\":\"max\",\"p3\":{:.17}",
        reference.width, reference.height, result.score, result.pnorm_3
    );
    if all_scores {
        print!(
            ",\"p1\":{:.17},\"p2\":{:.17},\"p6\":{:.17}",
            pnorm(map, 1.0),
            pnorm(map, 2.0),
            pnorm(map, 6.0)
        );
    }
    println!("}}");
    Ok(())
}
