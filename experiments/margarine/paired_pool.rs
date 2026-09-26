//! Pair-dependent reduction: retain the native sample with largest linear RGB
//! error in each 2x2 cell, selecting the same coordinate in both images.
//! This is an uncalibrated approximation, not ordinary image downsampling.
use super::*;

pub(super) fn compute(
    a: &ingress::EncodedRows<'_>,
    b: &ingress::EncodedRows<'_>,
    rows: usize,
    params: &ButteraugliParams,
) -> Result<diff::InternalResult, Box<dyn Error>> {
    let (w, h) = (a.width, a.height);
    if rows == 0 || (w, h) != (b.width, b.height) {
        return Err("invalid pooled pair".into());
    }
    let (pw, ph) = (w.div_ceil(2), h.div_ceil(2));
    let mut reference = vec![0.0; pw * ph * 3];
    let mut distorted = vec![0.0; pw * ph * 3];
    for y in (0..h).step_by(2) {
        let end = (y + 2).min(h);
        let ar = a.linear_strip(y, end);
        let br = b.linear_strip(y, end);
        pool_rows(
            &ar,
            &br,
            w,
            end - y,
            &mut reference[y / 2 * pw * 3..(y / 2 + 1) * pw * 3],
            &mut distorted[y / 2 * pw * 3..(y / 2 + 1) * pw * 3],
        );
    }
    let result = strips::compute(&reference, &distorted, pw, ph, pw * 3, rows, params)?;
    let coarse = result.diffmap.ok_or("missing pooled map")?;
    Ok(finish(&coarse, w, h))
}

pub(super) fn finish(coarse: &image::ImageF, w: usize, h: usize) -> diff::InternalResult {
    let mut map = image::ImageF::new(w, h);
    for y in 0..h {
        for (x, value) in map.row_mut(y).iter_mut().enumerate() {
            *value = coarse.row(y / 2)[x / 2];
        }
    }
    let (score, pnorm_3) = diff::compute_score_from_diffmap(&map);
    diff::InternalResult {
        score,
        pnorm_3,
        diffmap: Some(map),
    }
}

#[archmage::autoversion]
fn pool_rows(
    _token: archmage::SimdToken,
    a: &[f32],
    b: &[f32],
    w: usize,
    h: usize,
    reference: &mut [f32],
    distorted: &mut [f32],
) {
    for (ox, (ra, rb)) in reference
        .as_chunks_mut::<3>()
        .0
        .iter_mut()
        .zip(distorted.as_chunks_mut::<3>().0)
        .enumerate()
    {
        let mut largest = -1.0;
        let mut selected = 0;
        for y in 0..h {
            for x in ox * 2..(ox * 2 + 2).min(w) {
                let index = (y * w + x) * 3;
                let ar: &[f32; 3] = a[index..index + 3].try_into().unwrap();
                let br: &[f32; 3] = b[index..index + 3].try_into().unwrap();
                let dr = ar[0] - br[0];
                let dg = ar[1] - br[1];
                let db = ar[2] - br[2];
                let error = dr * dr + dg * dg + db * db;
                if error > largest {
                    largest = error;
                    selected = index;
                }
            }
        }
        ra.copy_from_slice(&a[selected..selected + 3]);
        rb.copy_from_slice(&b[selected..selected + 3]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ingress::{EncodedRows, Samples};

    #[test]
    fn every_cell_can_preserve_an_isolated_rgb16_low_bit_error() {
        let (w, h, stride) = (5, 5, 5 * 3 + 7);
        let a = vec![32000u16; stride * h];
        for y in 0..h {
            for x in 0..w {
                let mut b = a.clone();
                b[y * stride + x * 3] += 1;
                let a = EncodedRows::new(Samples::U16(&a), w, h, stride, 3).unwrap();
                let b = EncodedRows::new(Samples::U16(&b), w, h, stride, 3).unwrap();
                let score = compute(&a, &b, 2, &ButteraugliParams::default()).unwrap();
                assert!(score.score > 0.0, "lost ({x},{y})");
            }
        }
    }

    #[test]
    fn checkerboard_is_not_erased_and_odd_strip_seams_match() {
        let (w, h, stride) = (33, 273, 33 * 3 + 7);
        let a = vec![128u8; stride * h];
        let mut b = a.clone();
        for y in 0..h {
            for x in 0..w {
                b[y * stride + x * 3] = if (x + y) % 2 == 0 { 96 } else { 160 };
            }
        }
        let a = EncodedRows::new(Samples::U8(&a), w, h, stride, 3).unwrap();
        let b = EncodedRows::new(Samples::U8(&b), w, h, stride, 3).unwrap();
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
