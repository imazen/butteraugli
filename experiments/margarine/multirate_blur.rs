//! Broad Gaussian filtering on a reduced lattice. Fine residuals and all
//! perceptual scoring stages still operate on the original pixel lattice.
//! This analytical experiment has no fitted constants or acceptance claim.
pub use crate::exact_blur::{blur_mirrored_5x5, compute_separable5_weights};
use crate::image::{BufferPool, ImageF};

fn geometry(sigma: f32) -> (usize, f32) {
    let factor = if sigma >= 6.0 {
        4
    } else if sigma >= 2.0 {
        2
    } else {
        1
    };
    // Area reduction contributes (f²-1)/12 variance. Linear reconstruction
    // contributes approximately (f²-1)/6, averaged over lattice phases.
    let variance = sigma * sigma - (factor * factor - 1) as f32 / 4.0;
    (factor, variance.max(0.0).sqrt() / factor as f32)
}

pub(crate) fn support(sigma: f32) -> usize {
    let (factor, reduced_sigma) = geometry(sigma);
    factor * ((2.25 * reduced_sigma).floor() as usize + 2)
}

#[archmage::autoversion]
fn reduce(_token: archmage::SimdToken, input: &ImageF, factor: usize, pool: &BufferPool) -> ImageF {
    let (w, h) = (input.width(), input.height());
    let mut result = ImageF::from_pool_dirty(w.div_ceil(factor), h.div_ceil(factor), pool);
    for oy in 0..result.height() {
        let y0 = oy * factor;
        let y1 = (y0 + factor).min(h);
        for (ox, out) in result.row_mut(oy).iter_mut().enumerate() {
            let x0 = ox * factor;
            let x1 = (x0 + factor).min(w);
            let mut sum = 0.0;
            for y in y0..y1 {
                for &v in &input.row(y)[x0..x1] {
                    sum += v;
                }
            }
            *out = sum / ((x1 - x0) * (y1 - y0)) as f32;
        }
    }
    result
}

fn coordinate(pixel: usize, factor: usize, length: usize) -> (usize, usize, f32) {
    let position = ((pixel as f32 + 0.5) / factor as f32 - 0.5).max(0.0);
    let a = (position as usize).min(length - 1);
    (a, (a + 1).min(length - 1), position - a as f32)
}

#[archmage::autoversion]
fn expand(
    _token: archmage::SimdToken,
    input: &ImageF,
    w: usize,
    h: usize,
    factor: usize,
    pool: &BufferPool,
) -> ImageF {
    let mut output = ImageF::from_pool_dirty(w, h, pool);
    let columns: Vec<_> = (0..w)
        .map(|x| coordinate(x, factor, input.width()))
        .collect();
    for y in 0..h {
        let (a, b, fy) = coordinate(y, factor, input.height());
        let (a, b) = (input.row(a), input.row(b));
        for (out, &(x0, x1, fx)) in output.row_mut(y).iter_mut().zip(&columns) {
            let top = a[x0] + fx * (a[x1] - a[x0]);
            let bottom = b[x0] + fx * (b[x1] - b[x0]);
            *out = top + fy * (bottom - top);
        }
    }
    output
}

pub fn gaussian_blur(input: &ImageF, sigma: f32, pool: &BufferPool) -> ImageF {
    let (factor, reduced_sigma) = geometry(sigma);
    if factor == 1 {
        return crate::exact_blur::gaussian_blur(input, sigma, pool);
    }
    let reduced = reduce(input, factor, pool);
    let filtered = crate::exact_blur::gaussian_blur(&reduced, reduced_sigma, pool);
    let output = expand(&filtered, input.width(), input.height(), factor, pool);
    reduced.recycle(pool);
    filtered.recycle(pool);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strided_odd_and_tiny_inputs_match_packed() {
        for (w, h) in [(1, 1), (3, 5), (31, 27)] {
            let stride = w + 7;
            let mut padded = vec![f32::NAN; stride * h];
            let mut tight = Vec::new();
            for y in 0..h {
                for x in 0..w {
                    let v = ((y * 13 + x * 7) % 19) as f32;
                    padded[y * stride + x] = v;
                    tight.push(v);
                }
            }
            let padded = ImageF::from_vec_padded(padded, w, h, stride);
            let tight = ImageF::from_vec(tight, w, h);
            let pool = BufferPool::new();
            for sigma in [1.5641633, 2.7, 3.224899, 7.1559334] {
                let a = gaussian_blur(&padded, sigma, &pool);
                let b = gaussian_blur(&tight, sigma, &pool);
                for y in 0..h {
                    assert_eq!(a.row(y), b.row(y));
                }
            }
        }
    }

    #[test]
    fn constant_and_centered_lattice() {
        let pool = BufferPool::new();
        let input = ImageF::filled(29, 33, 0.375);
        for sigma in [2.7, 3.224899, 7.1559334] {
            let output = gaussian_blur(&input, sigma, &pool);
            for y in 0..33 {
                for &v in output.row(y) {
                    assert!((v - 0.375).abs() < 1e-6);
                }
            }
        }
        assert_eq!(coordinate(1, 2, 3), (0, 1, 0.25));
        assert_eq!(coordinate(2, 2, 3), (0, 1, 0.75));
    }
}
