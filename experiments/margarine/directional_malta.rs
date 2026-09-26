//! Four-direction angular quadrature for Butteraugli's sixteen-line Malta bank.
//! Shares its asymmetric normalization and zero border extension. Multiplicity
//! is derived from tap counts to preserve a constant scaled difference.
use crate::image::{BufferPool, ImageF};

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
    crate::shared_malta::malta_diff_map_impl(
        a,
        b,
        greater,
        smaller,
        norm,
        lf,
        pool,
        |data, center, stride, width, lf, out| {
            debug_assert_eq!(out.len(), width);
            row(data, center, stride, lf, out);
        },
    )
}

#[archmage::autoversion]
fn row(
    _token: archmage::SimdToken,
    data: &[f32],
    center: usize,
    stride: usize,
    lf: bool,
    out: &mut [f32],
) {
    if lf {
        kernel(
            data,
            center,
            stride,
            [-4, -2, 0, 2, 4],
            [-3, -2, 0, 2, 3],
            4.0,
            out,
        );
    } else {
        let taps = [-4, -3, -2, -1, 0, 1, 2, 3, 4];
        // Full bank: ten 9-tap and six 7-tap lines. Four-direction bank:
        // two 9-tap and two 7-tap lines. Normalize their constant responses.
        kernel(
            data,
            center,
            stride,
            taps,
            [-3, -2, -1, 0, 1, 2, 3],
            1104.0 / 260.0,
            out,
        );
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn kernel<const A: usize, const D: usize>(
    data: &[f32],
    center: usize,
    stride: usize,
    axial: [isize; A],
    diagonal: [isize; D],
    multiplicity: f32,
    out: &mut [f32],
) {
    let stride = stride as isize;
    let full = out.len() / 8 * 8;
    for (block, dst) in out[..full].as_chunks_mut::<8>().0.iter_mut().enumerate() {
        let base = (center + block * 8) as isize;
        let mut sums = [[0.0f32; 8]; 4];
        for (direction, sum) in sums.iter_mut().enumerate() {
            let (taps, step): (&[isize], isize) = match direction {
                0 => (&axial, 1),
                1 => (&axial, stride),
                2 => (&diagonal, stride + 1),
                _ => (&diagonal, stride - 1),
            };
            for &tap in taps {
                let index = (base + tap * step) as usize;
                let values: &[f32; 8] = data[index..index + 8].try_into().unwrap();
                for lane in 0..8 {
                    sum[lane] += values[lane];
                }
            }
        }
        for lane in 0..8 {
            let mut value = 0.0;
            for sum in &sums {
                value += sum[lane] * sum[lane];
            }
            dst[lane] = multiplicity * value;
        }
    }
    for (x, dst) in out.iter_mut().enumerate().skip(full) {
        let base = (center + x) as isize;
        let mut sums = [0.0f32; 4];
        for (direction, sum) in sums.iter_mut().enumerate() {
            let (taps, step): (&[isize], isize) = match direction {
                0 => (&axial, 1),
                1 => (&axial, stride),
                2 => (&diagonal, stride + 1),
                _ => (&diagonal, stride - 1),
            };
            for &tap in taps {
                *sum += data[(base + tap * step) as usize];
            }
        }
        *dst = multiplicity * sums.iter().map(|sum| sum * sum).sum::<f32>();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_difference_keeps_full_bank_normalization() {
        let pool = BufferPool::new();
        let (mut a, mut b) = (ImageF::new(37, 19), ImageF::new(37, 19));
        for y in 0..19 {
            a.row_mut(y).fill(0.75);
            b.row_mut(y).fill(0.5);
        }
        for lf in [false, true] {
            let original = crate::shared_malta::malta_diff_map(&a, &b, 1.0, 1.0, 10.0, lf, &pool);
            let reduced = malta_diff_map(&a, &b, 1.0, 1.0, 10.0, lf, &pool);
            for y in 4..15 {
                for x in 4..33 {
                    let expected = original.row(y)[x];
                    assert!((reduced.row(y)[x] - expected).abs() <= expected * 2.0e-6);
                }
            }
        }
    }
}
