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
    let padded =
        crate::shared_malta::malta_scaled_differences(a, b, greater, smaller, norm, lf, pool);
    let mut out = ImageF::from_pool_dirty(a.width(), a.height(), pool);
    evaluate(&padded, lf, &mut out);
    padded.recycle(pool);
    out
}

#[archmage::autoversion]
fn evaluate(_token: archmage::SimdToken, padded: &ImageF, lf: bool, out: &mut ImageF) {
    for y in 0..out.height() {
        let full = out.width() / 8 * 8;
        let row = out.row_mut(y);
        for (block, dst) in row[..full].as_chunks_mut::<8>().0.iter_mut().enumerate() {
            let start = block * 8;
            let window = NativeWindow {
                rows: std::array::from_fn(|r| {
                    padded.row(y + r)[start..start + 16].try_into().unwrap()
                }),
            };
            *dst = if lf {
                lf_bank(&window).0
            } else {
                hf_bank(&window).0
            };
        }
        if full != row.len() {
            let mut tail = [[0.0; 16]; 9];
            for (r, values) in tail.iter_mut().enumerate() {
                let src = &padded.row(y + r)[full..];
                let count = src.len().min(16);
                values[..count].copy_from_slice(&src[..count]);
            }
            let window = NativeWindow {
                rows: tail.each_ref(),
            };
            let values = if lf {
                lf_bank(&window)
            } else {
                hf_bank(&window)
            };
            let dst = &mut row[full..];
            dst.copy_from_slice(&values.0[..dst.len()]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_native_response_matches_shared_bank_at_borders_and_vector_tails() {
        let pool = BufferPool::new();
        for (w, h) in [(1, 1), (7, 9), (8, 10), (9, 11), (31, 33), (128, 129)] {
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
                        assert_eq!(actual.row(y), expected.row(y), "{w}x{h}, lf={lf}, row {y}");
                    }
                }
            }
        }
    }
}
