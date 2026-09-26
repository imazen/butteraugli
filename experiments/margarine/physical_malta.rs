//! Malta on a grid whose pixels span two original pixels. Every original tap
//! is bilinearly reconstructed at half its coordinate, then like terms combine.
use crate::half_malta_bank::{hf_bank, lf_bank};
use crate::image::{BufferPool, ImageF};
use crate::malta_bank::{V, Window as BankWindow};

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

struct Window<'a> {
    rows: [&'a [f32; 12]; 5],
}
impl BankWindow for Window<'_> {
    #[inline(always)]
    fn load(&self, dx: isize, dy: isize) -> V {
        let row = self.rows[(dy + 2) as usize];
        let start = (dx + 2) as usize;
        let values: &[f32; 8] = row[start..start + 8].try_into().unwrap();
        V(*values)
    }
}

#[archmage::autoversion]
fn evaluate(_token: archmage::SimdToken, padded: &ImageF, lf: bool, out: &mut ImageF) {
    for y in 0..out.height() {
        for (block, dst) in out.row_mut(y).chunks_mut(8).enumerate() {
            let rows = std::array::from_fn(|row| {
                let start = (y + 2 + row) * padded.stride() + 2 + block * 8;
                padded.data()[start..start + 12].try_into().unwrap()
            });
            let window = Window { rows };
            let value = if lf {
                lf_bank(&window)
            } else {
                hf_bank(&window)
            };
            dst.copy_from_slice(&value.0[..dst.len()]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn complete_bank_constant_response_is_preserved() {
        let row = [1.0; 12];
        let window = Window { rows: [&row; 5] };
        assert_eq!(hf_bank(&window).0, [1104.0; 8]);
        assert_eq!(lf_bank(&window).0, [400.0; 8]);
    }
}
