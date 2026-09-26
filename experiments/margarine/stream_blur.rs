//! Separable FIR with a rolling horizontal-result buffer. Kernel coefficients
//! and clipped-edge normalization follow the shared Gaussian implementation.
//! Border arithmetic uses normalized coefficients at every lane; this can differ
//! by floating-point rounding from the shared scalar border tails.
use crate::image::{BufferPool, ImageF};

pub(crate) fn gaussian_blur(input: &ImageF, sigma: f32, pool: &BufferPool) -> ImageF {
    let kernel = crate::exact_blur::compute_kernel(sigma);
    // The experiment's largest kernel fits this row-view array. Preserve the
    // original implementation for other sigma values rather than truncate it.
    if kernel.len() > 64 {
        return crate::exact_blur::gaussian_blur(input, sigma, pool);
    }
    let inverse = 1.0 / kernel.iter().sum::<f32>();
    let scaled: Vec<_> = kernel.iter().map(|v| v * inverse).collect();
    let (w, h) = (input.width(), input.height());
    let radius = kernel.len() / 2;
    let retained = h.min(kernel.len());
    let mut ring = ImageF::from_pool_dirty(w, retained, pool);
    let mut output = ImageF::from_pool_dirty(w, h, pool);
    let mut next = 0;
    for y in 0..h {
        let start = y.saturating_sub(radius);
        let end = y.saturating_add(radius).saturating_add(1).min(h);
        while next < end {
            horizontal(
                input.row(next),
                &kernel,
                &scaled,
                ring.row_mut(next % retained),
            );
            next += 1;
        }
        let count = end - start;
        let mut views: [&[f32]; 64] = [&[]; 64];
        for (i, view) in views[..count].iter_mut().enumerate() {
            *view = ring.row((start + i) % retained);
        }
        let weights = &kernel[start + radius - y..end + radius - y];
        let mut border = [0.0; 64];
        let weights = if count == kernel.len() {
            &scaled[..]
        } else {
            let inverse = 1.0 / weights.iter().sum::<f32>();
            for (out, &weight) in border.iter_mut().zip(weights) {
                *out = weight * inverse;
            }
            &border[..count]
        };
        vertical(&views[..count], weights, output.row_mut(y));
    }
    ring.recycle(pool);
    output
}

#[archmage::autoversion]
pub(crate) fn horizontal(
    _token: archmage::SimdToken,
    input: &[f32],
    kernel: &[f32],
    scaled: &[f32],
    output: &mut [f32],
) {
    let radius = kernel.len() / 2;
    let begin = radius.min(input.len());
    let end = input.len().saturating_sub(radius).max(begin);
    for x in (0..begin).chain(end..input.len()) {
        let lo = x.saturating_sub(radius);
        let hi = (x + radius + 1).min(input.len());
        let weights = &kernel[lo + radius - x..hi + radius - x];
        let scale = 1.0 / weights.iter().sum::<f32>();
        output[x] = input[lo..hi]
            .iter()
            .zip(weights)
            .map(|(&a, &b)| a * b)
            .sum::<f32>()
            * scale;
    }
    let full = (end - begin) / 32 * 32;
    horizontal_blocks::<32>(
        input,
        scaled,
        begin.saturating_sub(radius),
        &mut output[begin..begin + full],
    );
    let tail = (end - begin - full) / 8 * 8;
    horizontal_blocks::<8>(
        input,
        scaled,
        (begin + full).saturating_sub(radius),
        &mut output[begin + full..begin + full + tail],
    );
    for x in begin + full + tail..end {
        output[x] = input[x - radius..x - radius + scaled.len()]
            .iter()
            .zip(scaled)
            .fold(0.0, |sum, (&value, &weight)| value.mul_add(weight, sum));
    }
}

// Keep each pixel's reduction order while exposing four independent AVX2
// accumulators (or eight NEON accumulators) to the instruction scheduler.
#[inline(always)]
fn horizontal_blocks<const N: usize>(
    input: &[f32],
    weights: &[f32],
    offset: usize,
    output: &mut [f32],
) {
    for (i, dst) in output.as_chunks_mut::<N>().0.iter_mut().enumerate() {
        let start = offset + i * N;
        let mut sums = [0.0f32; N];
        for (k, &weight) in weights.iter().enumerate() {
            let values: &[f32; N] = input[start + k..start + k + N].try_into().unwrap();
            for lane in 0..N {
                sums[lane] = values[lane].mul_add(weight, sums[lane]);
            }
        }
        *dst = sums;
    }
}

#[archmage::autoversion]
pub(crate) fn vertical(
    _token: archmage::SimdToken,
    rows: &[&[f32]],
    weights: &[f32],
    output: &mut [f32],
) {
    let full = output.len() / 32 * 32;
    vertical_blocks::<32>(rows, weights, 0, &mut output[..full]);
    let tail = (output.len() - full) / 8 * 8;
    vertical_blocks::<8>(rows, weights, full, &mut output[full..full + tail]);
    for (i, dst) in output[full + tail..].iter_mut().enumerate() {
        *dst = rows.iter().zip(weights).fold(0.0, |sum, (&row, &weight)| {
            row[full + tail + i].mul_add(weight, sum)
        });
    }
}

#[inline(always)]
fn vertical_blocks<const N: usize>(
    rows: &[&[f32]],
    weights: &[f32],
    offset: usize,
    output: &mut [f32],
) {
    for (i, dst) in output.as_chunks_mut::<N>().0.iter_mut().enumerate() {
        let start = offset + i * N;
        let mut sums = [0.0f32; N];
        for (&row, &weight) in rows.iter().zip(weights) {
            let values: &[f32; N] = row[start..start + N].try_into().unwrap();
            for lane in 0..N {
                sums[lane] = values[lane].mul_add(weight, sums[lane]);
            }
        }
        *dst = sums;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rolling_rows_match_full_separable_filter_with_rounding_bound() {
        let pool = BufferPool::new();
        for (w, h) in [(1, 1), (2, 3), (7, 9), (33, 35), (257, 131)] {
            let stride = w + 5;
            let mut values = vec![f32::NAN; stride * h];
            for y in 0..h {
                for x in 0..w {
                    values[y * stride + x] = ((x * 17 + y * 31) % 97) as f32 / 97.;
                }
            }
            let input = ImageF::from_vec_padded(values, w, h, stride);
            for sigma in [0.75, 1.5641633, 2.7, 3.224899, 7.1559334] {
                let expected = crate::exact_blur::gaussian_blur(&input, sigma, &pool);
                let actual = gaussian_blur(&input, sigma, &pool);
                for y in 0..h {
                    for (&a, &b) in actual.row(y).iter().zip(expected.row(y)) {
                        assert!(
                            (a - b).abs() <= 1e-6,
                            "{w}x{h}, sigma={sigma}, {a} versus {b}"
                        );
                    }
                }
            }
        }
    }
}
