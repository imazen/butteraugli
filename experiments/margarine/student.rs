//! Cost probe for a possible student feature extractor. No trained Margarine
//! model exists here, and the uniform-weight diagnostic is not a quality score.
use std::error::Error;
use std::sync::OnceLock;
use zensim::{AlphaMode, PixelFormat, StridedBytes, Zensim, ZensimProfile};

use zensim::profile::ProfileParams;

static WEIGHTS: [f64; 372] = [1.0; 372];
static PARAMS: OnceLock<ProfileParams> = OnceLock::new();

pub(crate) fn extractor() -> Zensim {
    let params = PARAMS.get_or_init(|| {
        ProfileParams::builder()
            .weights(&WEIGHTS)
            .extended_features(true)
            .compute_iw_features(true)
            .build()
    });
    Zensim::new(ZensimProfile::Custom {
        name: "margarine-feature-probe",
        params,
    })
}

// StridedBytes currently exposes linear f32 RGBA. Encode bytes safely without
// narrowing samples; this conversion occurs outside metric-only timing.
pub(crate) fn rgba(linear: &[f32]) -> Vec<u8> {
    let (pixels, tail) = linear.as_chunks::<3>();
    assert!(tail.is_empty());
    pixels
        .iter()
        .flat_map(|p| {
            [p[0], p[1], p[2], 1.0]
                .into_iter()
                .flat_map(f32::to_ne_bytes)
        })
        .collect()
}

pub(crate) fn extract(
    scorer: &Zensim,
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<Vec<f64>, Box<dyn Error>> {
    let a = StridedBytes::try_with_alpha_mode(
        reference,
        width,
        height,
        stride,
        PixelFormat::LinearF32Rgba,
        AlphaMode::Opaque,
    )?;
    let b = StridedBytes::try_with_alpha_mode(
        distorted,
        width,
        height,
        stride,
        PixelFormat::LinearF32Rgba,
        AlphaMode::Opaque,
    )?;
    let features = scorer.compute_all_features(&a, &b)?.into_features();
    if features.len() != 372 || !features.iter().all(|v| v.is_finite()) {
        return Err("unexpected or nonfinite feature vector".into());
    }
    Ok(features)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strided_linear_features_match_tight_input() {
        let w = 17;
        let h = 19;
        let a: Vec<_> = (0..w * h * 3)
            .map(|i| ((i * 31) % 101) as f32 / 101.0)
            .collect();
        let b: Vec<_> = a.iter().map(|v| v * 0.9).collect();
        let (a, b) = (rgba(&a), rgba(&b));
        let stride = w * 16 + 16;
        let pad = |input: &[u8]| {
            let mut out = vec![255; stride * h];
            for y in 0..h {
                out[y * stride..y * stride + w * 16]
                    .copy_from_slice(&input[y * w * 16..(y + 1) * w * 16]);
            }
            out
        };
        let scorer = extractor();
        assert_eq!(
            extract(&scorer, &a, &b, w, h, w * 16).unwrap(),
            extract(&scorer, &pad(&a), &pad(&b), w, h, stride).unwrap()
        );
    }
}
