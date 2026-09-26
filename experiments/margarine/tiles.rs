//! Bound both dimensions of the direct pipeline's working set. Each scale
//! preserves its global sampling lattice and only copies the valid interior.
use super::*;

const COLUMNS: usize = 256;

pub(super) fn compute(
    a: &ingress::EncodedRows<'_>,
    b: &ingress::EncodedRows<'_>,
    rows: usize,
    params: &ButteraugliParams,
) -> Result<diff::InternalResult, Box<dyn Error>> {
    let (w, h) = (a.width, a.height);
    let scale = |factor: usize| {
        let (sw, sh) = (w.div_ceil(factor), h.div_ceil(factor));
        let mut output = image::ImageF::new(sw, sh);
        let pool = image::BufferPool::with_capacity(32);
        let halo = strips::halo();
        let mut previous_shape = (0, 0);
        for y in (0..sh).step_by(rows) {
            let end_y = y.saturating_add(rows).min(sh);
            let y0 = y.saturating_sub(halo) / 4 * 4;
            let y1 = end_y
                .saturating_add(halo)
                .div_ceil(4)
                .saturating_mul(4)
                .min(sh);
            for x in (0..sw).step_by(COLUMNS) {
                let end_x = x.saturating_add(COLUMNS).min(sw);
                // Align native and subsampled SIMD lanes as well as factor-4 blur.
                let x0 = x.saturating_sub(halo) / 32 * 32;
                let x1 = end_x
                    .saturating_add(halo)
                    .div_ceil(4)
                    .saturating_mul(4)
                    .min(sw);
                let shape = (x1 - x0, y1 - y0);
                if shape != previous_shape {
                    pool.clear();
                    previous_shape = shape;
                }
                #[cfg(feature = "planar")]
                let tile =
                    strips::single_scale_encoded(a, b, factor, [x0, y0, x1, y1], params, &pool);
                #[cfg(not(feature = "planar"))]
                let tile = {
                    let load = |input: &ingress::EncodedRows<'_>| {
                        let (nx1, ny1) = ((x1 * factor).min(w), (y1 * factor).min(h));
                        let rgb = input.linear_region(x0 * factor, nx1, y0 * factor, ny1);
                        if factor == 1 {
                            rgb
                        } else {
                            let (small, rw, rh) = diff::subsample_linear_rgb_2x(
                                &rgb,
                                nx1 - x0 * factor,
                                ny1 - y0 * factor,
                            );
                            debug_assert_eq!((rw, rh), shape);
                            small
                        }
                    };
                    let (ar, br) = (load(a), load(b));
                    strips::single_scale(&ar, &br, shape.0, shape.1, params, &pool)
                };
                for oy in y..end_y {
                    output.row_mut(oy)[x..end_x]
                        .copy_from_slice(&tile.row(oy - y0)[x - x0..end_x - x0]);
                }
                tile.recycle(&pool);
            }
        }
        output
    };
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
    use ingress::{EncodedRows, Samples};
    #[test]
    fn tiles_preserve_strip_maps_across_both_axes_and_odd_rgb16_edges() {
        for (w, h) in [(33, 65), (355, 337), (640, 129)] {
            let stride = w * 3 + 7;
            let mut a = vec![0u16; stride * h];
            let mut b = a.clone();
            for y in 0..h {
                for x in 0..w {
                    for c in 0..3 {
                        let i = y * stride + x * 3 + c;
                        a[i] = ((x * 313 + y * 1997 + c * 10457) % 65536) as u16;
                        b[i] = a[i].saturating_add(((x + y + c) % 31) as u16);
                    }
                }
            }
            let a = EncodedRows::new(Samples::U16(&a), w, h, stride, 3).unwrap();
            let b = EncodedRows::new(Samples::U16(&b), w, h, stride, 3).unwrap();
            let params = ButteraugliParams::default();
            let expected = strips::compute(
                &a.linear_strip(0, h),
                &b.linear_strip(0, h),
                w,
                h,
                w * 3,
                64,
                &params,
            )
            .unwrap()
            .diffmap
            .unwrap();
            let actual = compute(&a, &b, 64, &params).unwrap().diffmap.unwrap();
            for y in 0..h {
                assert_eq!(actual.row(y), expected.row(y), "{w}x{h}, row {y}");
            }
        }
    }
}
