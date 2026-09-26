//! Experimental coarse LF/MF decomposition with native HF/UHF residuals.
//! Constants and nonlinear transforms come from Butteraugli. No fitted weights.
use crate::consts::{ADD_MF_RANGE, REMOVE_MF_RANGE, SIGMA_HF, SIGMA_LF};
use crate::image::{BufferPool, Image3F, ImageF};
use crate::{blur, exact_blur, shared_psycho as shared};
pub(crate) use shared::PsychoImage;

pub(crate) fn separate_frequencies_owned(xyb: Image3F, pool: &BufferPool) -> PsychoImage {
    let (w, h) = (xyb.width(), xyb.height());
    let mut ps = PsychoImage::from_pool(w, h, pool);
    for c in 0..3 {
        let coarse = blur::reduce(xyb.plane(c), 2, pool);
        // One area reduction plus reconstruction contributes 3/4 pixel².
        let sigma_lf = ((SIGMA_LF as f32).powi(2) - 0.75).sqrt() * 0.5;
        let lf = blur::gaussian_blur(&coarse, sigma_lf, pool);
        let mut residual = ImageF::from_pool_dirty(coarse.width(), coarse.height(), pool);
        shared::subtract_images(&coarse, &lf, &mut residual);
        // Already on the coarse lattice: no second area reduction here.
        let mf = exact_blur::gaussian_blur(&residual, SIGMA_HF as f32 * 0.5, pool);
        coarse.recycle(pool);
        residual.recycle(pool);

        let mut full_lf = blur::expand(&lf, w, h, 2, pool);
        let full_mf = blur::expand(&mf, w, h, 2, pool);
        lf.recycle(pool);
        mf.recycle(pool);
        if c < 2 {
            // Every native pixel contributes to the fine residual. In particular,
            // opposite-sign checkerboard artifacts survive coarse averaging.
            fine_residual(xyb.plane(c), &full_lf, &full_mf, &mut ps.hf[c]);
            if c == 0 {
                shared::apply_remove_range(&full_mf, REMOVE_MF_RANGE as f32, ps.mf.plane_mut(c));
            } else {
                shared::apply_amplify_range(&full_mf, ADD_MF_RANGE as f32, ps.mf.plane_mut(c));
            }
            full_mf.recycle(pool);
        } else {
            let old = std::mem::replace(ps.mf.plane_mut(c), full_mf);
            old.recycle(pool);
        }
        std::mem::swap(ps.lf.plane_mut(c), &mut full_lf);
        full_lf.recycle(pool);
    }
    xyb.recycle(pool);
    shared::xyb_low_freq_to_vals(&mut ps.lf);
    let (x, y) = ps.hf.split_at_mut(1);
    shared::suppress_x_by_y(&y[0], &mut x[0]);
    shared::separate_hf_and_uhf(&mut ps.hf, &mut ps.uhf, pool);
    ps
}

#[archmage::autoversion]
fn fine_residual(
    _token: archmage::SimdToken,
    source: &ImageF,
    lf: &ImageF,
    mf: &ImageF,
    output: &mut ImageF,
) {
    for y in 0..source.height() {
        for (((out, &source), &lf), &mf) in output
            .row_mut(y)
            .iter_mut()
            .zip(source.row(y))
            .zip(lf.row(y))
            .zip(mf.row(y))
        {
            *out = (source - lf) - mf;
        }
    }
}
