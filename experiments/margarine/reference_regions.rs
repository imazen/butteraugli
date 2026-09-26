//! Two reference-texture strata. The distorted image cannot change selection.
//! This is an analytical sampling experiment, without fitted thresholds.
use super::{Region, TILE, ingress};

pub(super) fn select(reference: &ingress::EncodedRows<'_>) -> Vec<(Region, f64)> {
    let mut tiles = Vec::new();
    for y in (0..reference.height).step_by(TILE) {
        for x in (0..reference.width).step_by(TILE) {
            let region = Region {
                x,
                y,
                w: TILE.min(reference.width - x),
                h: TILE.min(reference.height - y),
            };
            let mut energy = 0.0f64;
            for py in y..y + region.h {
                for px in x..x + region.w {
                    let current = reference.encoded_rgb(px, py);
                    let left = reference.encoded_rgb(px.saturating_sub(1), py);
                    let above = reference.encoded_rgb(px, py.saturating_sub(1));
                    for c in 0..3 {
                        energy += f64::from(current[c] - left[c]).powi(2)
                            + f64::from(current[c] - above[c]).powi(2);
                    }
                }
            }
            tiles.push((energy / (region.w * region.h) as f64, region));
        }
    }
    tiles.sort_by(|a, b| a.0.total_cmp(&b.0));
    let split = tiles.len().div_ceil(2);
    [&tiles[..split], &tiles[split..]]
        .into_iter()
        .filter(|stratum| !stratum.is_empty())
        .map(|stratum| {
            let area = stratum.iter().map(|(_, r)| r.w * r.h).sum::<usize>() as f64;
            let mean = stratum
                .iter()
                .map(|(energy, r)| energy * (r.w * r.h) as f64)
                .sum::<f64>()
                / area;
            let (_, representative) = stratum
                .iter()
                .min_by(|a, b| (a.0 - mean).abs().total_cmp(&(b.0 - mean).abs()))
                .unwrap();
            (
                *representative,
                area / (representative.w * representative.h) as f64,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ingress::{EncodedRows, Samples};

    #[test]
    fn reference_strata_preserve_area_and_ignore_stride_padding() {
        for (w, h) in [(1, 1), (129, 131), (355, 337)] {
            let stride = w * 3 + 9;
            let mut padded = vec![65535u16; stride * h];
            let mut packed = Vec::new();
            for y in 0..h {
                for x in 0..w {
                    for c in 0..3 {
                        let value = ((x * x + y * 313 + c * 197) % 65536) as u16;
                        padded[y * stride + x * 3 + c] = value;
                        packed.push(value);
                    }
                }
            }
            let a = EncodedRows::new(Samples::U16(&padded), w, h, stride, 3).unwrap();
            let b = EncodedRows::new(Samples::U16(&packed), w, h, w * 3, 3).unwrap();
            let selected = select(&a);
            assert_eq!(selected, select(&b));
            assert_eq!(selected.len(), (w.div_ceil(TILE) * h.div_ceil(TILE)).min(2));
            let area = selected
                .iter()
                .map(|(r, weight)| (r.w * r.h) as f64 * weight)
                .sum::<f64>();
            assert!((area - (w * h) as f64).abs() < 1e-8);
        }
    }
}
