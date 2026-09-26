//! Bounded-height execution of the frozen box3 pipeline. This preserves its
//! spatial support; it is not a new perceptual approximation or a speed claim.
use super::*;

fn halo() -> usize {
    // RGB opsin preprocessing has radius 2. The longest band path traverses
    // LF, HF and UHF filters. Masking adds its blur plus fuzzy erosion's
    // 3-pixel offsets; Malta needs at most 4 pixels. The half-resolution
    // path doubles support; two extra rows cover downsampling/rounding.
    let band = 2
        + blur::support(consts::SIGMA_LF as f32)
        + blur::support(consts::SIGMA_HF as f32)
        + blur::support(consts::SIGMA_UHF as f32);
    let local = 4.max(blur::support(consts::MASK_RADIUS) + 3);
    2 * (band + local) + 2
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
    let mut map = image::ImageF::new(w, h);
    let halo = halo();
    for start in (0..h).step_by(rows) {
        let end = start.saturating_add(rows).min(h);
        // Align to the original 2x2 lattice, including odd final dimensions.
        let lattice = if cfg!(feature = "multirate") { 8 } else { 2 };
        let y0 = start.saturating_sub(halo) / lattice * lattice;
        let y1 = end
            .saturating_add(halo)
            .div_ceil(lattice)
            .saturating_mul(lattice)
            .min(h);
        let a = packed_strip(reference, w, stride, y0, y1);
        let b = packed_strip(distorted, w, stride, y0, y1);
        let strip = diff::compute_butteraugli_linear_impl(
            &a,
            &b,
            w,
            y1 - y0,
            params,
            &enough::Unstoppable,
        )?;
        let strip_map = strip.diffmap.as_ref().ok_or("missing strip map")?;
        for y in start..end {
            map.row_mut(y).copy_from_slice(strip_map.row(y - y0));
        }
    }
    let score = (0..h)
        .flat_map(|y| map.row(y))
        .copied()
        .fold(0.0f32, f32::max);
    Ok(diff::InternalResult {
        score: f64::from(score),
        pnorm_3: pnorm(&map, 3.0),
        diffmap: Some(map),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
