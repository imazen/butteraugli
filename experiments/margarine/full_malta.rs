//! Complete native Malta responses with each row's stencil range checked once.
//! The shared nonlinear difference transform and all sixteen lines are retained.
use crate::image::{BufferPool, ImageF};
use crate::malta_bank::{V, Window, hf_bank, lf_bank};

struct NativeWindow<'a> {
    rows: [&'a [f32; 16]; 9],
}
impl Window for NativeWindow<'_> {
    #[inline(always)]
    fn load(&self, dx: isize, dy: isize) -> V {
        let start = (dx + 4) as usize;
        let values: &[f32; 8] = self.rows[(dy + 4) as usize][start..start + 8]
            .try_into()
            .unwrap();
        V(*values)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn malta_diff_map(
    a: &ImageF,
    b: &ImageF,
    greater: f64,
    smaller: f64,
    norm: f64,
    lf: bool,
    pool: &BufferPool,
) -> ImageF {
    if a.width() < 8 {
        return crate::shared_malta::malta_diff_map(a, b, greater, smaller, norm, lf, pool);
    }
    let padded =
        crate::shared_malta::malta_scaled_differences(a, b, greater, smaller, norm, lf, pool);
    let mut out = ImageF::from_pool_dirty(a.width(), a.height(), pool);
    evaluate(&padded, lf, &mut out);
    padded.recycle(pool);
    out
}

#[archmage::autoversion]
fn evaluate(_token: archmage::SimdToken, padded: &ImageF, lf: bool, out: &mut ImageF) {
    let width = out.width();
    for y in 0..out.height() {
        let row = out.row_mut(y);
        for block in 0..width.div_ceil(8) {
            // The last block overlaps when width is not divisible by eight.
            let start = (block * 8).min(width - 8);
            let window = NativeWindow {
                rows: std::array::from_fn(|r| {
                    padded.row(y + r)[start..start + 16].try_into().unwrap()
                }),
            };
            let values = if lf {
                lf_bank(&window)
            } else {
                hf_bank(&window)
            };
            row[start..start + 8].copy_from_slice(&values.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_native_response_matches_shared_bank_at_borders_and_vector_tails() {
        let pool = BufferPool::new();
        for (w, h) in (1..34)
            .map(|w| (w, w + 2))
            .chain([(128, 129), (154, 151), (317, 129)])
        {
            let mut a = ImageF::new(w, h);
            let mut b = ImageF::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    a.row_mut(y)[x] = ((x * 31 + y * 97 + x * y) % 101) as f32 * 0.13 - 5.0;
                    b.row_mut(y)[x] = a.row(y)[x] * 0.91 + ((x + y) % 3) as f32 * 0.1;
                }
            }
            for lf in [false, true] {
                for (greater, smaller, norm) in [(1.0, 1.0, 0.5), (0.5, 1.7, 1.2), (1.7, 0.5, 2.0)]
                {
                    let expected = crate::shared_malta::malta_diff_map(
                        &a, &b, greater, smaller, norm, lf, &pool,
                    );
                    let actual = malta_diff_map(&a, &b, greater, smaller, norm, lf, &pool);
                    for y in 0..h {
                        if actual.row(y) != expected.row(y) {
                            let padded = crate::shared_malta::malta_scaled_differences(
                                &a, &b, greater, smaller, norm, lf, &pool,
                            );
                            for x in 0..w {
                                if actual.row(y)[x] != expected.row(y)[x] {
                                    let direct = if lf {
                                        crate::shared_malta::malta_unit_lf(&padded, x + 4, y + 4)
                                    } else {
                                        crate::shared_malta::malta_unit(&padded, x + 4, y + 4)
                                    };
                                    eprintln!(
                                        "x={x} weights={greater},{smaller},{norm} direct={direct} actual={} expected={}",
                                        actual.row(y)[x],
                                        expected.row(y)[x]
                                    );
                                }
                            }
                        }
                        assert_eq!(actual.row(y), expected.row(y), "{w}x{h}, lf={lf}, row {y}");
                    }
                }
            }
        }
    }
}
