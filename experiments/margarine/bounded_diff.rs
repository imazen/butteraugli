//! Preserve all six Malta banks and their accumulation order while releasing
//! intermediate maps after each channel. No perceptual constants are changed.
use crate::consts::*;
use crate::diff;
use crate::image::{BufferPool, Image3F, ImageF};
use crate::psycho::PsychoImage;

pub(crate) fn compute(a: &PsychoImage, b: &PsychoImage, asym: f32, pool: &BufferPool) -> Image3F {
    let channel = |c: usize| {
        let (wu, nu, wh, nh, wm, nm) = if c == 0 {
            (
                W_UHF_MALTA_X,
                NORM1_UHF_X,
                W_HF_MALTA_X,
                NORM1_HF_X,
                W_MF_MALTA_X,
                NORM1_MF_X,
            )
        } else {
            (
                W_UHF_MALTA,
                NORM1_UHF,
                W_HF_MALTA,
                NORM1_HF,
                W_MF_MALTA,
                NORM1_MF,
            )
        };
        let sqrt_asym = asym.sqrt();
        #[cfg(all(feature = "lattice", not(feature = "full-malta")))]
        use crate::malta::coarse_diff_map as bank;
        #[cfg(any(not(feature = "lattice"), feature = "full-malta"))]
        use crate::malta::malta_diff_map as bank;
        let uhf = bank(
            &a.uhf[c],
            &b.uhf[c],
            wu * asym as f64,
            wu / asym as f64,
            nu,
            false,
            pool,
        );
        let hf = bank(
            &a.hf[c],
            &b.hf[c],
            wh * sqrt_asym as f64,
            wh / sqrt_asym as f64,
            nh,
            true,
            pool,
        );
        let mf = bank(a.mf.plane(c), b.mf.plane(c), wm, wm, nm, true, pool);
        #[cfg(all(feature = "lattice", not(feature = "full-malta")))]
        let mut out = {
            let mut out = ImageF::from_pool_dirty(a.width(), a.height(), pool);
            crate::malta::reconstruct_sum([&uhf, &hf, &mf], &mut out);
            uhf.recycle(pool);
            out
        };
        #[cfg(any(not(feature = "lattice"), feature = "full-malta"))]
        let mut out = {
            let mut out = uhf;
            diff::accumulate_two(&hf, &mf, &mut out);
            out
        };
        hf.recycle(pool);
        mf.recycle(pool);
        diff::l2_diff_asymmetric(
            &a.hf[c],
            &b.hf[c],
            WMUL[c] as f32 * asym,
            WMUL[c] as f32 / asym,
            &mut out,
        );
        diff::l2_diff(a.mf.plane(c), b.mf.plane(c), WMUL[3 + c] as f32, &mut out);
        out
    };
    let (x, y) = diff::maybe_join(|| channel(0), || channel(1));
    let mut blue = ImageF::from_pool_dirty(a.width(), a.height(), pool);
    diff::l2_diff_write(a.mf.plane(2), b.mf.plane(2), WMUL[5] as f32, &mut blue);
    Image3F::from_planes(x, y, blue)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn channel_scheduling_preserves_all_planes_and_asymmetry() {
        for (w, h) in [(1, 1), (7, 9), (33, 31), (128, 129)] {
            let pool = BufferPool::new();
            let mut a = PsychoImage::new(w, h);
            let mut b = PsychoImage::new(w, h);
            for (salt, (aa, bb)) in a
                .uhf
                .iter_mut()
                .zip(&mut b.uhf)
                .chain(a.hf.iter_mut().zip(&mut b.hf))
                .enumerate()
            {
                for y in 0..h {
                    for x in 0..w {
                        aa.row_mut(y)[x] = ((x * 31 + y * 97 + salt * 3) % 101) as f32 * 0.13 - 5.0;
                        bb.row_mut(y)[x] = aa.row(y)[x] * 0.91 + ((x + y) % 3) as f32 * 0.1;
                    }
                }
            }
            for c in 0..3 {
                for y in 0..h {
                    for x in 0..w {
                        a.mf.plane_mut(c).row_mut(y)[x] =
                            ((x * 13 + y * 47 + c * 11) % 97) as f32 * 0.017;
                        b.mf.plane_mut(c).row_mut(y)[x] = a.mf.plane(c).row(y)[x] * 1.07;
                    }
                }
            }
            for asym in [0.5, 1.0, 1.7] {
                let expected = diff::compute_psycho_diff_malta(&a, &b, asym, 1.0, &pool);
                let actual = compute(&a, &b, asym, &pool);
                for c in 0..3 {
                    for y in 0..h {
                        assert_eq!(actual.plane(c).row(y), expected.plane(c).row(y));
                    }
                }
            }
        }
    }
}
