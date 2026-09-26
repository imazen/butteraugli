//! Broad Gaussian filtering on a reduced lattice. Fine residuals and all
//! perceptual scoring stages still operate on the original pixel lattice.
//! This analytical experiment has no fitted constants or acceptance claim.
pub use crate::exact_blur::{blur_mirrored_5x5, compute_separable5_weights};
use crate::image::{BufferPool, ImageF};

pub(crate) fn geometry(sigma: f32) -> (usize, f32) {
    if cfg!(feature = "native-gaussian") {
        return (1, sigma);
    }
    let factor = if sigma >= 6.0 {
        4
    } else if sigma >= 2.0 || (cfg!(feature = "coarse-gaussian") && sigma >= 1.0) {
        // At sigma=1, subtracting area/reconstruction variance leaves
        // reduced sigma=0.25. This experiment also reduces the fine band;
        // native residuals and all scoring stencils remain unchanged.
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
    let sigma = if cfg!(feature = "physical") {
        sigma * 0.5
    } else {
        sigma
    };
    let (factor, reduced_sigma) = geometry(sigma);
    factor * ((2.25 * reduced_sigma).floor() as usize + 2)
}

#[cfg(test)]
#[archmage::autoversion]
fn reduce_reference(
    _token: archmage::SimdToken,
    input: &ImageF,
    factor: usize,
    pool: &BufferPool,
) -> ImageF {
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

#[cfg(test)]
#[archmage::autoversion]
fn expand_reference(
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

#[inline(always)]
fn reduce_fixed<const F: usize>(input: &ImageF, pool: &BufferPool) -> ImageF {
    let (w, h) = (input.width(), input.height());
    let mut result = ImageF::from_pool_dirty(w.div_ceil(F), h.div_ceil(F), pool);
    for oy in 0..h.div_ceil(F) {
        let y0 = oy * F;
        let y1 = (y0 + F).min(h);
        let dst = result.row_mut(oy);
        let full = if y1 - y0 == F { w / F / 8 * 8 } else { 0 };
        for (block, out) in dst[..full].as_chunks_mut::<8>().0.iter_mut().enumerate() {
            let mut sums = [0.0f32; 8];
            for y in y0..y1 {
                let row = &input.row(y)[block * 8 * F..(block + 1) * 8 * F];
                for k in 0..F {
                    for lane in 0..8 {
                        sums[lane] += row[lane * F + k];
                    }
                }
            }
            for lane in 0..8 {
                out[lane] = sums[lane] / (F * F) as f32;
            }
        }
        for (ox, out) in dst.iter_mut().enumerate().skip(full) {
            let x0 = ox * F;
            let x1 = (x0 + F).min(w);
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

#[archmage::autoversion]
pub(crate) fn reduce(
    _token: archmage::SimdToken,
    input: &ImageF,
    factor: usize,
    pool: &BufferPool,
) -> ImageF {
    match factor {
        2 => reduce_fixed::<2>(input, pool),
        4 => reduce_fixed::<4>(input, pool),
        _ => unreachable!(),
    }
}

#[inline(always)]
pub(crate) fn expand_row<const F: usize>(a: &[f32], b: &[f32], fy: f32, out: &mut [f32]) {
    let left = (F / 2).min(out.len());
    out[..left].fill(a[0] + fy * (b[0] - a[0]));
    let interior = ((a.len() - 1) * F).min(out.len() - left) / F * F;
    // Eight independent coarse cells expose all interpolation lanes together.
    // Keep the per-pixel operation order identical to the reference expansion.
    let full = interior / (8 * F) * (8 * F);
    for block in 0..full / (8 * F) {
        let start = block * 8;
        let a: &[f32; 9] = a[start..start + 9].try_into().unwrap();
        let b: &[f32; 9] = b[start..start + 9].try_into().unwrap();
        let dst = &mut out[left + block * 8 * F..left + (block + 1) * 8 * F];
        for phase in 0..F {
            let fx = (phase as f32 + 0.5) / F as f32;
            let mut values = [0.0; 8];
            for i in 0..8 {
                let top = a[i] + fx * (a[i + 1] - a[i]);
                let bottom = b[i] + fx * (b[i + 1] - b[i]);
                values[i] = top + fy * (bottom - top);
            }
            for i in 0..8 {
                dst[i * F + phase] = values[i];
            }
        }
    }
    for ((dst, a), b) in out[left + full..left + interior]
        .as_chunks_mut::<F>()
        .0
        .iter_mut()
        .zip(a[full / F..].windows(2))
        .zip(b[full / F..].windows(2))
    {
        let [a0, a1]: [f32; 2] = a.try_into().unwrap();
        let [b0, b1]: [f32; 2] = b.try_into().unwrap();
        for (phase, d) in dst.iter_mut().enumerate() {
            let fx = (phase as f32 + 0.5) / F as f32;
            let top = a0 + fx * (a1 - a0);
            let bottom = b0 + fx * (b1 - b0);
            *d = top + fy * (bottom - top);
        }
    }
    for (x, d) in out.iter_mut().enumerate().skip(left + interior) {
        let (x0, x1, fx) = coordinate(x, F, a.len());
        let top = a[x0] + fx * (a[x1] - a[x0]);
        let bottom = b[x0] + fx * (b[x1] - b[x0]);
        *d = top + fy * (bottom - top);
    }
}

#[archmage::autoversion]
pub(crate) fn expand(
    _token: archmage::SimdToken,
    input: &ImageF,
    w: usize,
    h: usize,
    factor: usize,
    pool: &BufferPool,
) -> ImageF {
    let mut output = ImageF::from_pool_dirty(w, h, pool);
    for y in 0..h {
        let (a, b, fy) = coordinate(y, factor, input.height());
        let (a, b) = (input.row(a), input.row(b));
        match factor {
            2 => expand_row::<2>(a, b, fy, output.row_mut(y)),
            4 => expand_row::<4>(a, b, fy, output.row_mut(y)),
            _ => unreachable!(),
        }
    }
    output
}

pub fn gaussian_blur(input: &ImageF, sigma: f32, pool: &BufferPool) -> ImageF {
    let sigma = if cfg!(feature = "physical") {
        sigma * 0.5
    } else {
        sigma
    };
    let (factor, reduced_sigma) = geometry(sigma);
    if factor == 1 {
        return fir(input, sigma, pool);
    }
    let reduced = reduce(input, factor, pool);
    let filtered = fir(&reduced, reduced_sigma, pool);
    let output = expand(&filtered, input.width(), input.height(), factor, pool);
    reduced.recycle(pool);
    filtered.recycle(pool);
    output
}

fn fir(input: &ImageF, sigma: f32, pool: &BufferPool) -> ImageF {
    #[cfg(feature = "stream-blur")]
    {
        crate::stream_blur::gaussian_blur(input, sigma, pool)
    }
    #[cfg(not(feature = "stream-blur"))]
    {
        crate::exact_blur::gaussian_blur(input, sigma, pool)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resampling_matches_original_arithmetic_exactly() {
        let pool = BufferPool::new();
        for (w, h) in [(1, 1), (3, 5), (31, 27), (65, 69), (128, 129)] {
            let mut input = ImageF::new(w, h);
            for y in 0..h {
                for (x, v) in input.row_mut(y).iter_mut().enumerate() {
                    *v = (((x * 331 + y * 119 + x * y) % 997) as f32 - 498.0) * 0.113;
                }
            }
            for factor in [2, 4] {
                let a = reduce(&input, factor, &pool);
                let b = reduce_reference(&input, factor, &pool);
                for y in 0..a.height() {
                    assert_eq!(a.row(y), b.row(y), "reduce {w}x{h} factor={factor}");
                }
                let a = expand(&a, w, h, factor, &pool);
                let b = expand_reference(&b, w, h, factor, &pool);
                for y in 0..h {
                    assert_eq!(a.row(y), b.row(y), "expand {w}x{h} factor={factor} y={y}");
                }
            }
        }
    }

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
