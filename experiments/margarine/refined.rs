//! Experimental exact-region correction of the paired-RGB proxy. No human
//! labels enter selection or correction: the original FIR metric supplies it.
use super::*;

const TILE: usize = 128;
const PATCHES: usize = if cfg!(feature = "refined1") {
    1
} else if cfg!(feature = "refined2") {
    2
} else {
    3
};
// Conservative finite support: opsin 2 + Gaussian radii 16+7+3 + mask 6+3,
// doubled for the half-resolution contribution, plus its sampling footprint.
const HALO: usize = 76;

#[derive(Clone, Copy)]
struct Region {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
}

fn select(map: &image::ImageF) -> Vec<(Region, f64)> {
    let mut tiles = Vec::new();
    for y in (0..map.height()).step_by(TILE) {
        for x in (0..map.width()).step_by(TILE) {
            let r = Region {
                x,
                y,
                w: TILE.min(map.width() - x),
                h: TILE.min(map.height() - y),
            };
            let mut energy = 0.0f64;
            for row in y..y + r.h {
                for &v in &map.row(row)[x..x + r.w] {
                    energy += f64::from(v).powi(3);
                }
            }
            tiles.push((energy, r));
        }
    }
    tiles.sort_by(|a, b| b.0.total_cmp(&a.0));
    if cfg!(feature = "stratified") && tiles.len() > 1 {
        // Keep the peak tile for localization, then represent the remaining
        // image with the tile nearest its mean cubic error per pixel. Area
        // weights stop the peak stratum dominating the global correction.
        let pixels = tiles[1..].iter().map(|(_, r)| r.w * r.h).sum::<usize>() as f64;
        let mean = tiles[1..].iter().map(|(energy, _)| energy).sum::<f64>() / pixels;
        let (_, representative) = tiles[1..]
            .iter()
            .min_by(|(ae, a), (be, b)| {
                (ae / (a.w * a.h) as f64 - mean)
                    .abs()
                    .total_cmp(&(be / (b.w * b.h) as f64 - mean).abs())
            })
            .unwrap();
        return vec![
            (tiles[0].1, 1.0),
            (
                *representative,
                pixels / (representative.w * representative.h) as f64,
            ),
        ];
    }
    tiles
        .into_iter()
        .take(PATCHES)
        .map(|(_, r)| (r, 1.0))
        .collect()
}

fn exact_region(
    a: &ingress::EncodedRows<'_>,
    b: &ingress::EncodedRows<'_>,
    region: Region,
    params: &ButteraugliParams,
) -> Result<Vec<f32>, Box<dyn Error>> {
    // Align both full/half-scale SIMD groups as well as the sampling phase.
    // Otherwise crop-relative vector tails can change FMA rounding.
    let x0 = region.x.saturating_sub(HALO) / 32 * 32;
    let y0 = region.y.saturating_sub(HALO) / 2 * 2;
    let x1 = (region.x + region.w + HALO).min(a.width);
    let y1 = (region.y + region.h + HALO).min(a.height);
    let load = |input: &ingress::EncodedRows<'_>| {
        let strip = input.linear_region(x0, x1, y0, y1);
        let pixels = strip
            .as_chunks::<3>()
            .0
            .iter()
            .map(|p| butteraugli::RGB::new(p[0], p[1], p[2]))
            .collect::<Vec<_>>();
        butteraugli::Img::new(pixels, x1 - x0, y1 - y0)
    };
    let (a, b) = (load(a), load(b));
    let result = butteraugli::butteraugli_linear(
        a.as_ref(),
        b.as_ref(),
        &params.clone().with_compute_diffmap(true),
    )?;
    let map = result.diffmap.ok_or("missing exact region map")?;
    let mut center = Vec::with_capacity(region.w * region.h);
    for y in region.y - y0..region.y - y0 + region.h {
        let start = y * map.stride() + region.x - x0;
        center.extend_from_slice(&map.buf()[start..start + region.w]);
    }
    Ok(center)
}

pub(super) fn compute(
    a: &ingress::EncodedRows<'_>,
    b: &ingress::EncodedRows<'_>,
    rows: usize,
    params: &ButteraugliParams,
) -> Result<diff::InternalResult, Box<dyn Error>> {
    if a.width.max(a.height) <= 64 {
        return paired_pool::compute(a, b, rows, params);
    }
    if a.width.max(a.height) <= 256 {
        return strips::compute_encoded(a, b, rows, params);
    }
    let mut map = paired_pool::compute_map(a, b, rows, params)?;
    let regions = select(&map);
    let mut patches = Vec::new();
    let (mut original, mut approximate) = (0.0f64, 0.0f64);
    for (region, weight) in regions {
        let exact = exact_region(a, b, region, params)?;
        for y in 0..region.h {
            for (x, &value) in exact[y * region.w..(y + 1) * region.w].iter().enumerate() {
                original += weight * f64::from(value).powi(3);
                approximate += weight * f64::from(map.row(region.y + y)[region.x + x]).powi(3);
            }
        }
        patches.push((region, exact));
    }
    let ratio = if approximate > 0.0 && original > 0.0 {
        (original / approximate).cbrt() as f32
    } else {
        1.0
    };
    for y in 0..map.height() {
        for value in map.row_mut(y) {
            *value *= ratio;
        }
    }
    for (r, exact) in patches {
        for y in 0..r.h {
            map.row_mut(r.y + y)[r.x..r.x + r.w].copy_from_slice(&exact[y * r.w..(y + 1) * r.w]);
        }
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
    fn finite_halo_matches_full_original_map_at_interior_and_edges() {
        let (w, h, stride) = (355, 337, 355 * 3 + 7);
        let mut a = vec![0u16; stride * h];
        let mut b = a.clone();
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let i = y * stride + x * 3 + c;
                    a[i] = ((x * 391 + y * 2137 + c * 10271) % 65536) as u16;
                    b[i] = a[i].saturating_add(((x * 7 + y * 11 + c) % 100) as u16);
                }
            }
        }
        let a = EncodedRows::new(Samples::U16(&a), w, h, stride, 3).unwrap();
        let b = EncodedRows::new(Samples::U16(&b), w, h, stride, 3).unwrap();
        let params = ButteraugliParams::default();
        let all = exact_region(&a, &b, Region { x: 0, y: 0, w, h }, &params).unwrap();
        for r in [
            Region {
                x: 0,
                y: 0,
                w: 128,
                h: 128,
            },
            Region {
                x: 128,
                y: 128,
                w: 128,
                h: 128,
            },
            Region {
                x: 256,
                y: 256,
                w: w - 256,
                h: h - 256,
            },
        ] {
            let patch = exact_region(&a, &b, r, &params).unwrap();
            for y in 0..r.h {
                assert_eq!(
                    &patch[y * r.w..(y + 1) * r.w],
                    &all[(r.y + y) * w + r.x..(r.y + y) * w + r.x + r.w]
                );
            }
        }
    }
}
