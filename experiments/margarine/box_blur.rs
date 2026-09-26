//! Gaussian approximation by three separable running-sum boxes.
//! Box widths match the requested variance as closely as three adjacent odd
//! widths allow. The radius choice is analytic, not fitted to an image corpus.
//! Boundaries use cropped, renormalized boxes on each pass.

pub use crate::exact_blur::{blur_mirrored_5x5, compute_separable5_weights};
use crate::image::{BufferPool, ImageF};

fn radii(sigma: f32) -> [usize; 3] {
    let variance = f64::from(sigma).powi(2);
    // A width-w box has variance (w*w-1)/12; convolution adds variances.
    let ideal = (4.0 * variance + 1.0).sqrt();
    let lower = (ideal.floor() as usize).max(1) | 1;
    let lower = if lower as f64 > ideal {
        lower.saturating_sub(2).max(1)
    } else {
        lower
    };
    let upper = lower + 2;
    let low_variance = ((lower * lower) as f64 - 1.0) / 12.0;
    let high_variance = ((upper * upper) as f64 - 1.0) / 12.0;
    let n_low = ((3.0 * high_variance - variance) / (high_variance - low_variance))
        .round()
        .clamp(0.0, 3.0) as usize;
    std::array::from_fn(|i| if i < n_low { lower / 2 } else { upper / 2 })
}

pub(crate) fn support(sigma: f32) -> usize {
    radii(sigma).iter().sum()
}

fn box_pass(input: &ImageF, radius: usize, pool: &BufferPool) -> ImageF {
    let (w, h) = (input.width(), input.height());
    if radius == 0 {
        return input.clone();
    }
    let mut horizontal = ImageF::from_pool_dirty(w, h, pool);
    for y in 0..h {
        let src = input.row(y);
        let dst = horizontal.row_mut(y);
        let mut sum: f64 = src[..(radius + 1).min(w)]
            .iter()
            .map(|&v| f64::from(v))
            .sum();
        for (x, out) in dst.iter_mut().enumerate() {
            let lo = x.saturating_sub(radius);
            let hi = (x + radius + 1).min(w);
            *out = (sum / (hi - lo) as f64) as f32;
            if x >= radius {
                sum -= f64::from(src[x - radius]);
            }
            if x + radius + 1 < w {
                sum += f64::from(src[x + radius + 1]);
            }
        }
    }
    let mut output = ImageF::from_pool_dirty(w, h, pool);
    let mut sums = vec![0.0f64; w];
    for y in 0..(radius + 1).min(h) {
        for (sum, &value) in sums.iter_mut().zip(horizontal.row(y)) {
            *sum += f64::from(value);
        }
    }
    for y in 0..h {
        let count = (y + radius + 1).min(h) - y.saturating_sub(radius);
        for (out, &sum) in output.row_mut(y).iter_mut().zip(&sums) {
            *out = (sum / count as f64) as f32;
        }
        if y >= radius {
            for (sum, &value) in sums.iter_mut().zip(horizontal.row(y - radius)) {
                *sum -= f64::from(value);
            }
        }
        if y + radius + 1 < h {
            for (sum, &value) in sums.iter_mut().zip(horizontal.row(y + radius + 1)) {
                *sum += f64::from(value);
            }
        }
    }
    horizontal.recycle(pool);
    output
}

pub fn gaussian_blur(input: &ImageF, sigma: f32, pool: &BufferPool) -> ImageF {
    assert!(sigma.is_finite() && (0.0..=64.0).contains(&sigma));
    let r = radii(sigma);
    let mut output = box_pass(input, r[0], pool);
    for radius in &r[1..] {
        let next = box_pass(&output, *radius, pool);
        output.recycle(pool);
        output = next;
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn running_sum_matches_direct_clipped_box_with_stride_and_tiny_shapes() {
        for (w, h, radius) in [(1, 1, 7), (3, 5, 9), (17, 19, 3)] {
            let stride = w + 5;
            let mut pixels = vec![f32::NAN; stride * h];
            for y in 0..h {
                for x in 0..w {
                    pixels[y * stride + x] = ((x * 17 + y * 3) % 23) as f32;
                }
            }
            let input = ImageF::from_vec_padded(pixels, w, h, stride);
            let got = box_pass(&input, radius, &BufferPool::new());
            for y in 0..h {
                for x in 0..w {
                    let mut sum = 0.0;
                    let mut count = 0;
                    for yy in y.saturating_sub(radius)..(y + radius + 1).min(h) {
                        for xx in x.saturating_sub(radius)..(x + radius + 1).min(w) {
                            sum += f64::from(input.row(yy)[xx]);
                            count += 1;
                        }
                    }
                    let expected = (sum / count as f64) as f32;
                    assert!((got.row(y)[x] - expected).abs() < 2e-6);
                }
            }
        }
    }

    #[test]
    fn preserves_constants_including_edges() {
        let input = ImageF::filled(13, 7, 0.375);
        for sigma in [0.0, 0.5, 1.5641633, 3.224899, 7.1559334] {
            let output = gaussian_blur(&input, sigma, &BufferPool::new());
            for y in 0..7 {
                assert!(output.row(y).iter().all(|&v| v == 0.375));
            }
        }
    }
}
