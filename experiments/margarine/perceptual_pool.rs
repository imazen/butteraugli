//! Native Butteraugli opsin adaptation before pair-dependent XYB reduction.
//! Coarse scoring retains the complete directional bank and shared masking.
use super::*;
use crate::image::{BufferPool, Image3F, ImageF};

pub(super) fn compute(
    a: &ingress::EncodedRows<'_>,
    b: &ingress::EncodedRows<'_>,
    rows: usize,
    params: &ButteraugliParams,
) -> Result<diff::InternalResult, Box<dyn Error>> {
    let (w, h) = (a.width, a.height);
    if rows == 0 || (w, h) != (b.width, b.height) {
        return Err("invalid perceptual pair".into());
    }
    let (mut pa, mut pb) = (
        Image3F::new(w.div_ceil(2), h.div_ceil(2)),
        Image3F::new(w.div_ceil(2), h.div_ceil(2)),
    );
    // Opsin's native adaptation filter has a two-row support radius.
    // Keep its work bounded even when the requested scoring strip is larger.
    let ingress_rows = 128;
    {
        let pool = BufferPool::with_capacity(24);
        for start in (0..h).step_by(ingress_rows) {
            let end = (start + ingress_rows).min(h);
            let y0 = start.saturating_sub(2);
            let y1 = (end + 2).min(h);
            let prepare = |image: &ingress::EncodedRows<'_>| {
                opsin::linear_rgb_to_xyb_butteraugli(
                    &image.linear_strip(y0, y1),
                    w,
                    y1 - y0,
                    params.intensity_target(),
                    &pool,
                )
            };
            let (a, b) = diff::maybe_join(|| prepare(a), || prepare(b));
            select(&a, &b, start - y0, end - y0, start / 2, &mut pa, &mut pb);
            a.recycle(&pool);
            b.recycle(&pool);
        }
    }
    let sub = if !params.single_resolution() && pa.width() >= 15 && pa.height() >= 15 {
        let pool = BufferPool::new();
        let reduce = |input: &Image3F| {
            Image3F::from_planes(
                blur::reduce(input.plane(0), 2, &pool),
                blur::reduce(input.plane(1), 2, &pool),
                blur::reduce(input.plane(2), 2, &pool),
            )
        };
        Some(score_level(&reduce(&pa), &reduce(&pb), rows, params))
    } else {
        None
    };
    let mut map = score_level(&pa, &pb, rows, params);
    if let Some(sub) = sub {
        diff::add_supersampled_2x(&sub, 0.5, &mut map);
    }
    Ok(paired_pool::finish(&map, w, h))
}

#[allow(clippy::too_many_arguments)]
fn select(
    a: &Image3F,
    b: &Image3F,
    y0: usize,
    y1: usize,
    dest_y: usize,
    pa: &mut Image3F,
    pb: &mut Image3F,
) {
    for y in (y0..y1).step_by(2) {
        for x in (0..a.width()).step_by(2) {
            let mut selected = (x, y);
            let mut largest = -1.0;
            for iy in y..(y + 2).min(y1) {
                for ix in x..(x + 2).min(a.width()) {
                    let mut energy = 0.0;
                    for c in 0..3 {
                        let delta = a.plane(c).row(iy)[ix] - b.plane(c).row(iy)[ix];
                        energy += delta * delta * consts::WMUL[3 + c] as f32;
                    }
                    if energy > largest {
                        largest = energy;
                        selected = (ix, iy);
                    }
                }
            }
            for c in 0..3 {
                pa.plane_mut(c).row_mut(dest_y + (y - y0) / 2)[x / 2] =
                    a.plane(c).row(selected.1)[selected.0];
                pb.plane_mut(c).row_mut(dest_y + (y - y0) / 2)[x / 2] =
                    b.plane(c).row(selected.1)[selected.0];
            }
        }
    }
}

fn crop(input: &Image3F, y0: usize, y1: usize, pool: &BufferPool) -> Image3F {
    let mut out = Image3F::from_pool_dirty(input.width(), y1 - y0, pool);
    for c in 0..3 {
        for y in y0..y1 {
            out.plane_mut(c)
                .row_mut(y - y0)
                .copy_from_slice(input.plane(c).row(y));
        }
    }
    out
}

fn score_level(a: &Image3F, b: &Image3F, rows: usize, params: &ButteraugliParams) -> ImageF {
    let (w, h) = (a.width(), a.height());
    let mut result = ImageF::new(w, h);
    let pool = BufferPool::with_capacity(32);
    let halo = strips::halo();
    let mut previous_height = 0;
    for start in (0..h).step_by(rows) {
        let end = (start + rows).min(h);
        let y0 = start.saturating_sub(halo) / 4 * 4;
        let y1 = (end + halo).div_ceil(4).saturating_mul(4).min(h);
        if previous_height != y1 - y0 {
            pool.clear();
            previous_height = y1 - y0;
        }
        let (a, b) = diff::maybe_join(
            || psycho::separate_frequencies_owned(crop(a, y0, y1, &pool), &pool),
            || psycho::separate_frequencies_owned(crop(b, y0, y1, &pool), &pool),
        );
        let mut ac =
            diff::compute_psycho_diff_malta(&a, &b, params.hf_asymmetry(), params.xmul(), &pool);
        let mask = diff::mask_psycho_image(&a, &b, Some(ac.plane_mut(1)), &pool);
        let map = diff::combine_channels_to_diffmap_fused(&mask, &a.lf, &b.lf, &ac, params.xmul());
        for y in start..end {
            result.row_mut(y).copy_from_slice(map.row(y - y0));
        }
        a.recycle(&pool);
        b.recycle(&pool);
        ac.recycle(&pool);
        mask.recycle(&pool);
        map.recycle(&pool);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ingress::{EncodedRows, Samples};
    #[test]
    fn native_ingress_and_scoring_seams_preserve_rgb16_error() {
        let (w, h, stride) = (33, 273, 33 * 3 + 7);
        let mut a = vec![0u16; stride * h];
        for y in 0..h {
            for x in 0..w * 3 {
                a[y * stride + x] = ((x * 117 + y * 351) % 65536) as u16;
            }
        }
        let mut b = a.clone();
        for y in [0, 1, 127, 128, 129, 255, 256, 272] {
            b[y * stride + 17 * 3] = b[y * stride + 17 * 3].saturating_add(1);
        }
        let a = EncodedRows::new(Samples::U16(&a), w, h, stride, 3).unwrap();
        let b = EncodedRows::new(Samples::U16(&b), w, h, stride, 3).unwrap();
        let whole = compute(&a, &b, h, &ButteraugliParams::default()).unwrap();
        let strip = compute(&a, &b, 31, &ButteraugliParams::default()).unwrap();
        assert!(strip.score > 0.0);
        for y in 0..h {
            assert_eq!(
                whole.diffmap.as_ref().unwrap().row(y),
                strip.diffmap.as_ref().unwrap().row(y)
            );
        }
    }
}
