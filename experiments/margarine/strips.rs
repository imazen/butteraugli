//! Bounded-height execution of the frozen box3 pipeline. This preserves its
//! spatial support; it is not a new perceptual approximation or a speed claim.
use super::*;

fn halo() -> usize {
    // RGB opsin preprocessing has radius 2. The longest band path traverses
    // LF, HF and UHF filters. Masking adds its blur plus fuzzy erosion's
    // 3-pixel offsets; Malta needs at most 4 pixels. Each scale now walks
    // its own strips, so support is measured in that scale’s pixels.
    let band = 2
        + blur::support(consts::SIGMA_LF as f32)
        + blur::support(consts::SIGMA_HF as f32)
        + blur::support(consts::SIGMA_UHF as f32);
    let local = 4.max(blur::support(consts::MASK_RADIUS) + 3);
    band + local
}

fn packed_strip(
    input: &[f32],
    w: usize,
    stride: usize,
    y0: usize,
    y1: usize,
) -> std::borrow::Cow<'_, [f32]> {
    if stride == 3 * w {
        std::borrow::Cow::Borrowed(&input[y0 * stride..y1 * stride])
    } else {
        let mut packed = Vec::with_capacity((y1 - y0) * 3 * w);
        for y in y0..y1 {
            packed.extend_from_slice(&input[y * stride..y * stride + 3 * w]);
        }
        std::borrow::Cow::Owned(packed)
    }
}

pub(super) fn compute(
    reference: &[f32],
    distorted: &[f32],
    w: usize,
    h: usize,
    stride: usize,
    rows: usize,
    params: &ButteraugliParams,
) -> Result<diff::InternalResult, Box<dyn Error>> {
    let row_width = w.checked_mul(3).ok_or("strip dimensions overflow")?;
    if w == 0 || h == 0 || rows == 0 || stride < row_width {
        return Err("invalid strip geometry".into());
    }
    let needed = (h - 1)
        .checked_mul(stride)
        .and_then(|n| n.checked_add(row_width))
        .ok_or("strip dimensions overflow")?;
    if reference.len() < needed || distorted.len() < needed {
        return Err("input too short for strip geometry".into());
    }
    compose(w, h, rows, params, |y0, y1| {
        (
            packed_strip(reference, w, stride, y0, y1),
            packed_strip(distorted, w, stride, y0, y1),
        )
    })
}

pub(super) fn compute_encoded(
    reference: &ingress::EncodedRows<'_>,
    distorted: &ingress::EncodedRows<'_>,
    rows: usize,
    params: &ButteraugliParams,
) -> Result<diff::InternalResult, Box<dyn Error>> {
    let (w, h) = (reference.width, reference.height);
    if rows == 0 || (w, h) != (distorted.width, distorted.height) {
        return Err("invalid encoded strip pair".into());
    }
    compose(w, h, rows, params, |y0, y1| {
        (
            reference.linear_strip(y0, y1).into(),
            distorted.linear_strip(y0, y1).into(),
        )
    })
}

fn single_scale(
    a: &[f32],
    b: &[f32],
    w: usize,
    h: usize,
    params: &ButteraugliParams,
    pool: &image::BufferPool,
) -> image::ImageF {
    if !cfg!(feature = "reuse") {
        return diff::compute_diffmap_single_resolution_linear(a, b, w, h, params);
    }
    let prepare = |rgb: &[f32]| {
        let xyb = opsin::linear_rgb_to_xyb_butteraugli(rgb, w, h, params.intensity_target(), pool);
        psycho::separate_frequencies_owned(xyb, pool)
    };
    let (a, b) = diff::maybe_join(|| prepare(a), || prepare(b));
    let mut ac =
        diff::compute_psycho_diff_malta(&a, &b, params.hf_asymmetry(), params.xmul(), pool);
    let mask = diff::mask_psycho_image(&a, &b, Some(ac.plane_mut(1)), pool);
    let map = diff::combine_channels_to_diffmap_fused(&mask, &a.lf, &b.lf, &ac, params.xmul());
    a.recycle(pool);
    b.recycle(pool);
    ac.recycle(pool);
    mask.recycle(pool);
    map
}

fn compose<'a>(
    w: usize,
    h: usize,
    rows: usize,
    params: &ButteraugliParams,
    mut load: impl FnMut(usize, usize) -> (std::borrow::Cow<'a, [f32]>, std::borrow::Cow<'a, [f32]>),
) -> Result<diff::InternalResult, Box<dyn Error>> {
    let mut scale = |factor: usize| {
        let sw = w.div_ceil(factor);
        let sh = h.div_ceil(factor);
        let mut map = image::ImageF::new(sw, sh);
        let pool = image::BufferPool::with_capacity(if cfg!(feature = "reuse") { 32 } else { 0 });
        let mut previous_height = 0;
        let halo = halo();
        let lattice = if cfg!(feature = "multirate") { 4 } else { 1 };
        for start in (0..sh).step_by(rows) {
            let end = start.saturating_add(rows).min(sh);
            let y0 = start.saturating_sub(halo) / lattice * lattice;
            let y1 = end
                .saturating_add(halo)
                .div_ceil(lattice)
                .saturating_mul(lattice)
                .min(sh);
            if y1 - y0 != previous_height {
                pool.clear();
                previous_height = y1 - y0;
            }
            let (a, b) = load(y0 * factor, (y1 * factor).min(h));
            let map_strip = if factor == 1 {
                single_scale(&a, &b, sw, y1 - y0, params, &pool)
            } else {
                let height = (y1 * factor).min(h) - y0 * factor;
                let (a, aw, ah) = diff::subsample_linear_rgb_2x(&a, w, height);
                let (b, bw, bh) = diff::subsample_linear_rgb_2x(&b, w, height);
                debug_assert_eq!((aw, ah), (sw, y1 - y0));
                debug_assert_eq!((bw, bh), (sw, y1 - y0));
                single_scale(&a, &b, sw, y1 - y0, params, &pool)
            };
            for y in start..end {
                map.row_mut(y).copy_from_slice(map_strip.row(y - y0));
            }
            map_strip.recycle(&pool);
        }
        map
    };
    // Keep only the small completed map alive while processing the full scale.
    // Original threshold and combination arithmetic are shared with Butteraugli.
    let sub = (!params.single_resolution() && w >= 15 && h >= 15).then(|| scale(2));
    let mut map = scale(1);
    if let Some(sub) = sub {
        diff::add_supersampled_2x(&sub, 0.5, &mut map);
    }
    let (score, pnorm_3) = diff::compute_score_from_diffmap(&map);
    Ok(diff::InternalResult {
        score,
        pnorm_3,
        diffmap: Some(map),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoded_strides_and_rgb16_match_linear_ingress() {
        use ingress::{EncodedRows, Samples};
        let (w, h, stride) = (33, 273, 33 * 4 + 7);
        let mut a = vec![0u16; stride * h];
        let mut b = a.clone();
        for y in 0..h {
            for x in 0..w {
                let i = y * stride + 4 * x;
                for c in 0..3 {
                    a[i + c] = ((x * 117 + y * 351 + c * 999) % 65536) as u16;
                    b[i + c] = a[i + c].saturating_add(if y % 64 < 3 { 1 } else { 0 });
                }
                a[i + 3] = 65535;
                b[i + 3] = 65535;
            }
        }
        let a = EncodedRows::new(Samples::U16(&a), w, h, stride, 4).unwrap();
        let b = EncodedRows::new(Samples::U16(&b), w, h, stride, 4).unwrap();
        let params = ButteraugliParams::default();
        let linear = compute(
            &a.linear_strip(0, h),
            &b.linear_strip(0, h),
            w,
            h,
            w * 3,
            64,
            &params,
        )
        .unwrap();
        let native = compute_encoded(&a, &b, 64, &params).unwrap();
        assert!(
            native.score > 0.0,
            "single-bit distortion must survive RGB16 ingress"
        );
        for y in 0..h {
            assert_eq!(
                linear.diffmap.as_ref().unwrap().row(y),
                native.diffmap.as_ref().unwrap().row(y)
            );
        }
    }

    #[test]
    fn rejects_overflow_and_short_strided_input() {
        let params = ButteraugliParams::default();
        assert!(compute(&[], &[], usize::MAX, 1, usize::MAX, 32, &params).is_err());
        assert!(compute(&[0.0; 12], &[0.0; 12], 2, 2, 8, 32, &params).is_err());
    }

    #[test]
    fn tiled_map_preserves_full_pipeline_across_boundaries_and_stride() {
        // Wide dynamic range and localized changes on both sides of seams.
        let (w, h, stride) = (31, 257, 31 * 3 + 7);
        let mut a = vec![f32::NAN; stride * h];
        let mut b = a.clone();
        for y in 0..h {
            for x in 0..3 * w {
                let v = ((x * 31 + y * 97 + x * y) % 997) as f32 / 997.0;
                a[y * stride + x] = v;
                b[y * stride + x] = if (y % 32).abs_diff(0) < 2 { v * 0.9 } else { v };
            }
        }
        let params = ButteraugliParams::default();
        let pa = packed_strip(&a, w, stride, 0, h);
        let pb = packed_strip(&b, w, stride, 0, h);
        let whole =
            diff::compute_butteraugli_linear_impl(&pa, &pb, w, h, &params, &enough::Unstoppable)
                .unwrap();
        for rows in [31, 64] {
            let strip = compute(&a, &b, w, h, stride, rows, &params).unwrap();
            for y in 0..h {
                for (&actual, &expected) in strip
                    .diffmap
                    .as_ref()
                    .unwrap()
                    .row(y)
                    .iter()
                    .zip(whole.diffmap.as_ref().unwrap().row(y))
                {
                    assert!(
                        (actual - expected).abs() <= 1e-5,
                        "rows={rows} y={y}: {actual} != {expected}"
                    );
                }
            }
        }
    }
}
