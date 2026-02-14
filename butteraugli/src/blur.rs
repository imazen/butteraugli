//! Gaussian blur implementation for butteraugli.
//!
//! Butteraugli uses Gaussian blurs at various scales to separate
//! frequency bands. The blur is implemented as a separable convolution.
//!
//! The C++ butteraugli uses clamp-to-edge boundary handling with
//! re-normalization for border pixels. This module matches that behavior.
//!
//! Optimizations:
//! - Transpose during horizontal convolution for cache-friendly vertical pass
//! - Pre-normalized kernel weights for interior pixels (no division in inner loop)
//! - Separate fast path for interior pixels (no bounds checking)
//! - Explicit f32x8 SIMD for ~1.4x speedup

use crate::image::{BufferPool, ImageF};

/// Computes normalized separable 5x5 weights for a given sigma.
///
/// Returns [w0, w1, w2] where:
/// - w0 = center weight
/// - w1 = 1-pixel offset weight
/// - w2 = 2-pixel offset weight
///
/// The kernel is symmetric: [w2, w1, w0, w1, w2]
#[must_use]
pub fn compute_separable5_weights(sigma: f32) -> [f32; 3] {
    let kernel = compute_kernel(sigma);
    assert_eq!(kernel.len(), 5, "Separable5 requires kernel size 5");

    let sum: f32 = kernel.iter().sum();
    let scale = 1.0 / sum;

    [
        kernel[2] * scale, // w0: center
        kernel[1] * scale, // w1: offset 1
        kernel[0] * scale, // w2: offset 2
    ]
}

/// Computes a 1D Gaussian kernel for the given sigma.
///
/// Returns un-normalized weights (matches C++ behavior).
/// The caller should normalize for interior pixels or re-normalize for borders.
#[must_use]
pub fn compute_kernel(sigma: f32) -> Vec<f32> {
    const M: f32 = 2.25; // Accuracy increases when m is increased
    let scaler = -1.0 / (2.0 * sigma * sigma);
    let diff = (M * sigma.abs()).max(1.0) as i32;
    let size = (2 * diff + 1) as usize;
    let mut kernel = vec![0.0f32; size];

    for i in -diff..=diff {
        let weight = (scaler * (i * i) as f32).exp();
        kernel[(i + diff) as usize] = weight;
    }

    kernel
}

/// Computes a horizontal convolution with transpose (output is transposed).
///
/// This makes the subsequent vertical pass cache-friendly since it becomes
/// a horizontal pass on the transposed image.
///
/// The interior dispatch function `F` is provided by the caller, allowing
/// the dispatch decision to be hoisted to the outermost blur function.
/// When called from an `#[arcane]` context, `F` should be a `#[rite]` function
/// so LLVM can inline the SIMD kernel into the full blur pipeline.
#[allow(clippy::inline_always)]
#[inline(always)]
fn convolve_horizontal_transpose<F>(
    input: &ImageF,
    kernel: &[f32],
    border_ratio: f32,
    pool: &BufferPool,
    interior_fn: F,
) -> ImageF
where
    F: Fn(&ImageF, &[f32], usize, usize, usize, &mut ImageF),
{
    let width = input.width();
    let height = input.height();
    let half = kernel.len() / 2;

    // Output is transposed: height x width
    let mut output = ImageF::from_pool_dirty(height, width, pool);

    // Compute total weight for interior pixels (no border clipping)
    let weight_no_border: f32 = kernel.iter().sum();
    let scale_no_border = 1.0 / weight_no_border;

    // Pre-scale kernel for interior pixels
    let scaled_kernel: Vec<f32> = kernel.iter().map(|&k| k * scale_no_border).collect();

    let border1 = if width <= half { width } else { half };
    let border2 = if width > half { width - half } else { 0 };

    // Process left border (x < half)
    for x in 0..border1 {
        convolve_border_column_h(
            input,
            kernel,
            weight_no_border,
            border_ratio,
            x,
            &mut output,
        );
    }

    // Process interior (no bounds checking needed)
    if border2 > border1 {
        interior_fn(input, &scaled_kernel, border1, border2, half, &mut output);
    }

    // Process right border
    for x in border2..width {
        convolve_border_column_h(
            input,
            kernel,
            weight_no_border,
            border_ratio,
            x,
            &mut output,
        );
    }

    output
}

/// AVX-512 interior convolution with f32x16 (16 floats at a time).
#[cfg(target_arch = "x86_64")]
#[archmage::rite]
fn convolve_interior_v4(
    token: archmage::X64V4Token,
    input: &ImageF,
    scaled_kernel: &[f32],
    border1: usize,
    border2: usize,
    half: usize,
    output: &mut ImageF,
) {
    use magetypes::simd::f32x16;
    let height = input.height();
    let simd_chunks = (border2 - border1) / 16;

    for y in 0..height {
        let row_in = input.row(y);

        // SIMD path: process 16 pixels at a time
        for chunk_idx in 0..simd_chunks {
            let x = border1 + chunk_idx * 16;
            let d = x - half;
            let mut sum = f32x16::splat(token, 0.0);

            // For each kernel position, load 16 values and accumulate
            for (j, &k) in scaled_kernel.iter().enumerate() {
                let arr: [f32; 16] = row_in[d + j..d + j + 16].try_into().unwrap();
                sum += f32x16::from_array(token, arr) * f32x16::splat(token, k);
            }

            // Store results (transposed write)
            let results: [f32; 16] = sum.into();
            for (i, &val) in results.iter().enumerate() {
                output.set(y, x + i, val);
            }
        }

        // Scalar tail for remaining pixels
        let simd_end = border1 + simd_chunks * 16;
        for x in simd_end..border2 {
            let d = x - half;
            let sum: f32 = scaled_kernel
                .iter()
                .enumerate()
                .map(|(j, &k)| row_in[d + j] * k)
                .sum();
            output.set(y, x, sum);
        }
    }
}

/// AVX2 interior convolution with f32x8 (8 floats at a time).
#[cfg(target_arch = "x86_64")]
#[archmage::rite]
fn convolve_interior_v3(
    token: archmage::X64V3Token,
    input: &ImageF,
    scaled_kernel: &[f32],
    border1: usize,
    border2: usize,
    half: usize,
    output: &mut ImageF,
) {
    use magetypes::simd::f32x8;
    let height = input.height();
    let simd_chunks = (border2 - border1) / 8;

    for y in 0..height {
        let row_in = input.row(y);

        // SIMD path: process 8 pixels at a time
        for chunk_idx in 0..simd_chunks {
            let x = border1 + chunk_idx * 8;
            let d = x - half;
            let mut sum = f32x8::splat(token, 0.0);

            // For each kernel position, load 8 values and accumulate
            for (j, &k) in scaled_kernel.iter().enumerate() {
                let arr: [f32; 8] = row_in[d + j..d + j + 8].try_into().unwrap();
                sum += f32x8::from_array(token, arr) * f32x8::splat(token, k);
            }

            // Store results (transposed write)
            let results: [f32; 8] = sum.into();
            for (i, &val) in results.iter().enumerate() {
                output.set(y, x + i, val);
            }
        }

        // Scalar tail for remaining pixels
        let simd_end = border1 + simd_chunks * 8;
        for x in simd_end..border2 {
            let d = x - half;
            let sum: f32 = scaled_kernel
                .iter()
                .enumerate()
                .map(|(j, &k)| row_in[d + j] * k)
                .sum();
            output.set(y, x, sum);
        }
    }
}

/// Scalar fallback for interior convolution.
#[allow(clippy::inline_always)]
#[inline(always)]
fn convolve_interior_scalar(
    input: &ImageF,
    scaled_kernel: &[f32],
    border1: usize,
    border2: usize,
    half: usize,
    output: &mut ImageF,
) {
    let height = input.height();
    for y in 0..height {
        let row_in = input.row(y);
        for x in border1..border2 {
            let d = x - half;
            let sum: f32 = scaled_kernel
                .iter()
                .enumerate()
                .map(|(j, &k)| row_in[d + j] * k)
                .sum();
            output.set(y, x, sum);
        }
    }
}

/// Helper for border handling during horizontal convolution with transpose.
fn convolve_border_column_h(
    input: &ImageF,
    kernel: &[f32],
    weight_no_border: f32,
    border_ratio: f32,
    x: usize,
    output: &mut ImageF,
) {
    let width = input.width();
    let height = input.height();
    let half = kernel.len() / 2;

    let minx = x.saturating_sub(half);
    let maxx = (x + half).min(width - 1);

    // Compute actual weight for this column
    let mut weight = 0.0f32;
    for j in minx..=maxx {
        weight += kernel[j + half - x];
    }

    // Interpolate between no-border and border scaling
    let effective_weight = (1.0 - border_ratio) * weight + border_ratio * weight_no_border;
    let scale = 1.0 / effective_weight;

    for y in 0..height {
        let row_in = input.row(y);
        let mut sum = 0.0f32;
        for j in minx..=maxx {
            sum += row_in[j] * kernel[j + half - x];
        }
        // Write transposed
        output.set(y, x, sum * scale);
    }
}

/// Applies a 2D Gaussian blur to an image.
///
/// This is implemented as two separable 1D convolutions:
/// 1. Horizontal convolution with transpose
/// 2. Horizontal convolution on transposed result (effectively vertical) with transpose back
///
/// # Arguments
/// * `input` - Input image
/// * `sigma` - Standard deviation of the Gaussian
///
/// # Returns
/// Blurred image
pub fn gaussian_blur(input: &ImageF, sigma: f32, pool: &BufferPool) -> ImageF {
    if sigma <= 0.0 {
        return input.clone();
    }
    archmage::incant!(gaussian_blur_dispatch(input, sigma, pool))
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn gaussian_blur_dispatch_v4(
    token: archmage::X64V4Token,
    input: &ImageF,
    sigma: f32,
    pool: &BufferPool,
) -> ImageF {
    let kernel = compute_kernel(sigma);
    let interior = |inp: &ImageF, sk: &[f32], b1: usize, b2: usize, h: usize, out: &mut ImageF| {
        convolve_interior_v4(token, inp, sk, b1, b2, h, out);
    };
    let temp = convolve_horizontal_transpose(input, &kernel, 0.0, pool, interior);
    let result = convolve_horizontal_transpose(&temp, &kernel, 0.0, pool, interior);
    temp.recycle(pool);
    result
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn gaussian_blur_dispatch_v3(
    token: archmage::X64V3Token,
    input: &ImageF,
    sigma: f32,
    pool: &BufferPool,
) -> ImageF {
    let kernel = compute_kernel(sigma);
    let interior = |inp: &ImageF, sk: &[f32], b1: usize, b2: usize, h: usize, out: &mut ImageF| {
        convolve_interior_v3(token, inp, sk, b1, b2, h, out);
    };
    let temp = convolve_horizontal_transpose(input, &kernel, 0.0, pool, interior);
    let result = convolve_horizontal_transpose(&temp, &kernel, 0.0, pool, interior);
    temp.recycle(pool);
    result
}

fn gaussian_blur_dispatch_scalar(
    _token: archmage::ScalarToken,
    input: &ImageF,
    sigma: f32,
    pool: &BufferPool,
) -> ImageF {
    let kernel = compute_kernel(sigma);
    let temp = convolve_horizontal_transpose(input, &kernel, 0.0, pool, convolve_interior_scalar);
    let result =
        convolve_horizontal_transpose(&temp, &kernel, 0.0, pool, convolve_interior_scalar);
    temp.recycle(pool);
    result
}

/// Blur with border ratio parameter (matches C++ Blur signature).
pub fn blur_with_border(
    input: &ImageF,
    sigma: f32,
    border_ratio: f32,
    pool: &BufferPool,
) -> ImageF {
    if sigma <= 0.0 {
        return input.clone();
    }
    archmage::incant!(blur_with_border_dispatch(input, sigma, border_ratio, pool))
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn blur_with_border_dispatch_v4(
    token: archmage::X64V4Token,
    input: &ImageF,
    sigma: f32,
    border_ratio: f32,
    pool: &BufferPool,
) -> ImageF {
    let kernel = compute_kernel(sigma);
    let interior = |inp: &ImageF, sk: &[f32], b1: usize, b2: usize, h: usize, out: &mut ImageF| {
        convolve_interior_v4(token, inp, sk, b1, b2, h, out);
    };
    let temp = convolve_horizontal_transpose(input, &kernel, border_ratio, pool, interior);
    let result = convolve_horizontal_transpose(&temp, &kernel, border_ratio, pool, interior);
    temp.recycle(pool);
    result
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn blur_with_border_dispatch_v3(
    token: archmage::X64V3Token,
    input: &ImageF,
    sigma: f32,
    border_ratio: f32,
    pool: &BufferPool,
) -> ImageF {
    let kernel = compute_kernel(sigma);
    let interior = |inp: &ImageF, sk: &[f32], b1: usize, b2: usize, h: usize, out: &mut ImageF| {
        convolve_interior_v3(token, inp, sk, b1, b2, h, out);
    };
    let temp = convolve_horizontal_transpose(input, &kernel, border_ratio, pool, interior);
    let result = convolve_horizontal_transpose(&temp, &kernel, border_ratio, pool, interior);
    temp.recycle(pool);
    result
}

fn blur_with_border_dispatch_scalar(
    _token: archmage::ScalarToken,
    input: &ImageF,
    sigma: f32,
    border_ratio: f32,
    pool: &BufferPool,
) -> ImageF {
    let kernel = compute_kernel(sigma);
    let temp =
        convolve_horizontal_transpose(input, &kernel, border_ratio, pool, convolve_interior_scalar);
    let result = convolve_horizontal_transpose(
        &temp,
        &kernel,
        border_ratio,
        pool,
        convolve_interior_scalar,
    );
    temp.recycle(pool);
    result
}

/// Applies blur in-place (modifies the input image).
pub fn gaussian_blur_inplace(image: &mut ImageF, sigma: f32, pool: &BufferPool) {
    if sigma <= 0.0 {
        return;
    }

    let blurred = gaussian_blur(image, sigma, pool);
    image.copy_from(&blurred);
    blurred.recycle(pool);
}

/// Mirrors a coordinate outside image bounds.
///
/// This matches C++ libjxl's Mirror function - the mirror is placed
/// outside the last pixel (edge pixel is not repeated at mirror point).
///
/// For x < 0: x = -x - 1 (so -1 → 0, -2 → 1)
/// For x >= size: x = 2*size - 1 - x (so size → size-1, size+1 → size-2)
#[inline]
fn mirror(mut x: i32, size: i32) -> usize {
    while x < 0 || x >= size {
        if x < 0 {
            x = -x - 1;
        } else {
            x = 2 * size - 1 - x;
        }
    }
    x as usize
}

/// Blur with mirrored boundary handling for 5x5 kernel.
///
/// This matches C++ Separable5 which is used when kernel size == 5.
/// The key difference from clamp-and-renormalize is that mirrored values
/// are used at borders instead of clamping and adjusting weights.
/// Blur with mirrored boundary handling for 5x5 kernel.
///
/// This matches C++ Separable5 which is used when kernel size == 5.
/// SIMD-optimized for interior pixels.
pub fn blur_mirrored_5x5(input: &ImageF, weights: &[f32; 3], pool: &BufferPool) -> ImageF {
    archmage::incant!(blur_mirrored_5x5(input, weights, pool))
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn blur_mirrored_5x5_v4(
    token: archmage::X64V4Token,
    input: &ImageF,
    weights: &[f32; 3],
    pool: &BufferPool,
) -> ImageF {
    use magetypes::simd::f32x16;

    let width = input.width();
    let height = input.height();

    let w0 = weights[0];
    let w1 = weights[1];
    let w2 = weights[2];

    let w0_v = f32x16::splat(token, w0);
    let w1_v = f32x16::splat(token, w1);
    let w2_v = f32x16::splat(token, w2);

    let iwidth = width as i32;
    let iheight = height as i32;

    // Temporary for horizontal pass (NOT transposed for SIMD efficiency)
    let mut temp = ImageF::from_pool_dirty(width, height, pool);

    // Horizontal pass - SIMD for interior, scalar for borders
    let border = 2.min(width);
    let interior_end = if width > 4 { width - 2 } else { 0 };
    for y in 0..height {
        let row = input.row(y);
        let out_row = temp.row_mut(y);

        // Left border (scalar with mirror)
        for x in 0..border {
            let ix = x as i32;
            let v_m2 = row[mirror(ix - 2, iwidth)];
            let v_m1 = row[mirror(ix - 1, iwidth)];
            let v_0 = row[x];
            let v_p1 = row[mirror(ix + 1, iwidth)];
            let v_p2 = row[mirror(ix + 2, iwidth)];
            out_row[x] = v_0 * w0 + (v_m1 + v_p1) * w1 + (v_m2 + v_p2) * w2;
        }

        // Interior SIMD (16 at a time)
        let mut x = border;
        while x + 16 <= interior_end {
            let v_m2 = f32x16::load(token, (&row[x - 2..x + 14]).try_into().unwrap());
            let v_m1 = f32x16::load(token, (&row[x - 1..x + 15]).try_into().unwrap());
            let v_0 = f32x16::load(token, (&row[x..x + 16]).try_into().unwrap());
            let v_p1 = f32x16::load(token, (&row[x + 1..x + 17]).try_into().unwrap());
            let v_p2 = f32x16::load(token, (&row[x + 2..x + 18]).try_into().unwrap());

            let sum = v_0 * w0_v + (v_m1 + v_p1) * w1_v + (v_m2 + v_p2) * w2_v;
            sum.store((&mut out_row[x..x + 16]).try_into().unwrap());
            x += 16;
        }

        // Remaining interior (scalar)
        while x < interior_end {
            let v_m2 = row[x - 2];
            let v_m1 = row[x - 1];
            let v_0 = row[x];
            let v_p1 = row[x + 1];
            let v_p2 = row[x + 2];
            out_row[x] = v_0 * w0 + (v_m1 + v_p1) * w1 + (v_m2 + v_p2) * w2;
            x += 1;
        }

        // Right border (scalar with mirror)
        for x in interior_end..width {
            let ix = x as i32;
            let v_m2 = row[mirror(ix - 2, iwidth)];
            let v_m1 = row[mirror(ix - 1, iwidth)];
            let v_0 = row[x];
            let v_p1 = row[mirror(ix + 1, iwidth)];
            let v_p2 = row[mirror(ix + 2, iwidth)];
            out_row[x] = v_0 * w0 + (v_m1 + v_p1) * w1 + (v_m2 + v_p2) * w2;
        }
    }

    // Vertical pass - row-major with SIMD on x dimension (cache-friendly)
    let mut output = ImageF::from_pool_dirty(width, height, pool);
    let v_border = 2.min(height);
    let v_interior_end = if height > 4 { height - 2 } else { 0 };

    // Top border rows
    for y in 0..v_border {
        let iy = y as i32;
        let rm2 = temp.row(mirror(iy - 2, iheight));
        let rm1 = temp.row(mirror(iy - 1, iheight));
        let r0 = temp.row(y);
        let rp1 = temp.row(mirror(iy + 1, iheight));
        let rp2 = temp.row(mirror(iy + 2, iheight));
        let out = output.row_mut(y);
        let mut x = 0;
        while x + 16 <= width {
            let vm2 = f32x16::load(token, (&rm2[x..x + 16]).try_into().unwrap());
            let vm1 = f32x16::load(token, (&rm1[x..x + 16]).try_into().unwrap());
            let v0 = f32x16::load(token, (&r0[x..x + 16]).try_into().unwrap());
            let vp1 = f32x16::load(token, (&rp1[x..x + 16]).try_into().unwrap());
            let vp2 = f32x16::load(token, (&rp2[x..x + 16]).try_into().unwrap());
            let sum = v0 * w0_v + (vm1 + vp1) * w1_v + (vm2 + vp2) * w2_v;
            sum.store((&mut out[x..x + 16]).try_into().unwrap());
            x += 16;
        }
        while x < width {
            out[x] = r0[x] * w0 + (rm1[x] + rp1[x]) * w1 + (rm2[x] + rp2[x]) * w2;
            x += 1;
        }
    }

    // Interior rows (no mirror needed)
    for y in v_border..v_interior_end {
        let rm2 = temp.row(y - 2);
        let rm1 = temp.row(y - 1);
        let r0 = temp.row(y);
        let rp1 = temp.row(y + 1);
        let rp2 = temp.row(y + 2);
        let out = output.row_mut(y);
        let mut x = 0;
        while x + 16 <= width {
            let vm2 = f32x16::load(token, (&rm2[x..x + 16]).try_into().unwrap());
            let vm1 = f32x16::load(token, (&rm1[x..x + 16]).try_into().unwrap());
            let v0 = f32x16::load(token, (&r0[x..x + 16]).try_into().unwrap());
            let vp1 = f32x16::load(token, (&rp1[x..x + 16]).try_into().unwrap());
            let vp2 = f32x16::load(token, (&rp2[x..x + 16]).try_into().unwrap());
            let sum = v0 * w0_v + (vm1 + vp1) * w1_v + (vm2 + vp2) * w2_v;
            sum.store((&mut out[x..x + 16]).try_into().unwrap());
            x += 16;
        }
        while x < width {
            out[x] = r0[x] * w0 + (rm1[x] + rp1[x]) * w1 + (rm2[x] + rp2[x]) * w2;
            x += 1;
        }
    }

    // Bottom border rows
    for y in v_interior_end..height {
        let iy = y as i32;
        let rm2 = temp.row(mirror(iy - 2, iheight));
        let rm1 = temp.row(mirror(iy - 1, iheight));
        let r0 = temp.row(y);
        let rp1 = temp.row(mirror(iy + 1, iheight));
        let rp2 = temp.row(mirror(iy + 2, iheight));
        let out = output.row_mut(y);
        let mut x = 0;
        while x + 16 <= width {
            let vm2 = f32x16::load(token, (&rm2[x..x + 16]).try_into().unwrap());
            let vm1 = f32x16::load(token, (&rm1[x..x + 16]).try_into().unwrap());
            let v0 = f32x16::load(token, (&r0[x..x + 16]).try_into().unwrap());
            let vp1 = f32x16::load(token, (&rp1[x..x + 16]).try_into().unwrap());
            let vp2 = f32x16::load(token, (&rp2[x..x + 16]).try_into().unwrap());
            let sum = v0 * w0_v + (vm1 + vp1) * w1_v + (vm2 + vp2) * w2_v;
            sum.store((&mut out[x..x + 16]).try_into().unwrap());
            x += 16;
        }
        while x < width {
            out[x] = r0[x] * w0 + (rm1[x] + rp1[x]) * w1 + (rm2[x] + rp2[x]) * w2;
            x += 1;
        }
    }

    temp.recycle(pool);
    output
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn blur_mirrored_5x5_v3(
    token: archmage::X64V3Token,
    input: &ImageF,
    weights: &[f32; 3],
    pool: &BufferPool,
) -> ImageF {
    use magetypes::simd::f32x8;

    let width = input.width();
    let height = input.height();

    let w0 = weights[0];
    let w1 = weights[1];
    let w2 = weights[2];

    let w0_v = f32x8::splat(token, w0);
    let w1_v = f32x8::splat(token, w1);
    let w2_v = f32x8::splat(token, w2);

    let iwidth = width as i32;
    let iheight = height as i32;

    let mut temp = ImageF::from_pool_dirty(width, height, pool);

    let border = 2.min(width);
    let interior_end = if width > 4 { width - 2 } else { 0 };

    for y in 0..height {
        let row = input.row(y);
        let out_row = temp.row_mut(y);

        // Left border
        for x in 0..border {
            let ix = x as i32;
            let v_m2 = row[mirror(ix - 2, iwidth)];
            let v_m1 = row[mirror(ix - 1, iwidth)];
            let v_0 = row[x];
            let v_p1 = row[mirror(ix + 1, iwidth)];
            let v_p2 = row[mirror(ix + 2, iwidth)];
            out_row[x] = v_0 * w0 + (v_m1 + v_p1) * w1 + (v_m2 + v_p2) * w2;
        }

        // Interior SIMD (8 at a time)
        let mut x = border;
        while x + 8 <= interior_end {
            let v_m2 = f32x8::load(token, (&row[x - 2..x + 6]).try_into().unwrap());
            let v_m1 = f32x8::load(token, (&row[x - 1..x + 7]).try_into().unwrap());
            let v_0 = f32x8::load(token, (&row[x..x + 8]).try_into().unwrap());
            let v_p1 = f32x8::load(token, (&row[x + 1..x + 9]).try_into().unwrap());
            let v_p2 = f32x8::load(token, (&row[x + 2..x + 10]).try_into().unwrap());

            let sum = v_0 * w0_v + (v_m1 + v_p1) * w1_v + (v_m2 + v_p2) * w2_v;
            sum.store((&mut out_row[x..x + 8]).try_into().unwrap());
            x += 8;
        }

        // Remaining interior
        while x < interior_end {
            let v_m2 = row[x - 2];
            let v_m1 = row[x - 1];
            let v_0 = row[x];
            let v_p1 = row[x + 1];
            let v_p2 = row[x + 2];
            out_row[x] = v_0 * w0 + (v_m1 + v_p1) * w1 + (v_m2 + v_p2) * w2;
            x += 1;
        }

        // Right border
        for x in interior_end..width {
            let ix = x as i32;
            let v_m2 = row[mirror(ix - 2, iwidth)];
            let v_m1 = row[mirror(ix - 1, iwidth)];
            let v_0 = row[x];
            let v_p1 = row[mirror(ix + 1, iwidth)];
            let v_p2 = row[mirror(ix + 2, iwidth)];
            out_row[x] = v_0 * w0 + (v_m1 + v_p1) * w1 + (v_m2 + v_p2) * w2;
        }
    }

    // Vertical pass - row-major with SIMD on x dimension (cache-friendly)
    let mut output = ImageF::from_pool_dirty(width, height, pool);
    let v_border = 2.min(height);
    let v_interior_end = if height > 4 { height - 2 } else { 0 };

    for y in 0..v_border {
        let iy = y as i32;
        let rm2 = temp.row(mirror(iy - 2, iheight));
        let rm1 = temp.row(mirror(iy - 1, iheight));
        let r0 = temp.row(y);
        let rp1 = temp.row(mirror(iy + 1, iheight));
        let rp2 = temp.row(mirror(iy + 2, iheight));
        let out = output.row_mut(y);
        let mut x = 0;
        while x + 8 <= width {
            let vm2 = f32x8::load(token, (&rm2[x..x + 8]).try_into().unwrap());
            let vm1 = f32x8::load(token, (&rm1[x..x + 8]).try_into().unwrap());
            let v0 = f32x8::load(token, (&r0[x..x + 8]).try_into().unwrap());
            let vp1 = f32x8::load(token, (&rp1[x..x + 8]).try_into().unwrap());
            let vp2 = f32x8::load(token, (&rp2[x..x + 8]).try_into().unwrap());
            let sum = v0 * w0_v + (vm1 + vp1) * w1_v + (vm2 + vp2) * w2_v;
            sum.store((&mut out[x..x + 8]).try_into().unwrap());
            x += 8;
        }
        while x < width {
            out[x] = r0[x] * w0 + (rm1[x] + rp1[x]) * w1 + (rm2[x] + rp2[x]) * w2;
            x += 1;
        }
    }

    for y in v_border..v_interior_end {
        let rm2 = temp.row(y - 2);
        let rm1 = temp.row(y - 1);
        let r0 = temp.row(y);
        let rp1 = temp.row(y + 1);
        let rp2 = temp.row(y + 2);
        let out = output.row_mut(y);
        let mut x = 0;
        while x + 8 <= width {
            let vm2 = f32x8::load(token, (&rm2[x..x + 8]).try_into().unwrap());
            let vm1 = f32x8::load(token, (&rm1[x..x + 8]).try_into().unwrap());
            let v0 = f32x8::load(token, (&r0[x..x + 8]).try_into().unwrap());
            let vp1 = f32x8::load(token, (&rp1[x..x + 8]).try_into().unwrap());
            let vp2 = f32x8::load(token, (&rp2[x..x + 8]).try_into().unwrap());
            let sum = v0 * w0_v + (vm1 + vp1) * w1_v + (vm2 + vp2) * w2_v;
            sum.store((&mut out[x..x + 8]).try_into().unwrap());
            x += 8;
        }
        while x < width {
            out[x] = r0[x] * w0 + (rm1[x] + rp1[x]) * w1 + (rm2[x] + rp2[x]) * w2;
            x += 1;
        }
    }

    for y in v_interior_end..height {
        let iy = y as i32;
        let rm2 = temp.row(mirror(iy - 2, iheight));
        let rm1 = temp.row(mirror(iy - 1, iheight));
        let r0 = temp.row(y);
        let rp1 = temp.row(mirror(iy + 1, iheight));
        let rp2 = temp.row(mirror(iy + 2, iheight));
        let out = output.row_mut(y);
        let mut x = 0;
        while x + 8 <= width {
            let vm2 = f32x8::load(token, (&rm2[x..x + 8]).try_into().unwrap());
            let vm1 = f32x8::load(token, (&rm1[x..x + 8]).try_into().unwrap());
            let v0 = f32x8::load(token, (&r0[x..x + 8]).try_into().unwrap());
            let vp1 = f32x8::load(token, (&rp1[x..x + 8]).try_into().unwrap());
            let vp2 = f32x8::load(token, (&rp2[x..x + 8]).try_into().unwrap());
            let sum = v0 * w0_v + (vm1 + vp1) * w1_v + (vm2 + vp2) * w2_v;
            sum.store((&mut out[x..x + 8]).try_into().unwrap());
            x += 8;
        }
        while x < width {
            out[x] = r0[x] * w0 + (rm1[x] + rp1[x]) * w1 + (rm2[x] + rp2[x]) * w2;
            x += 1;
        }
    }

    temp.recycle(pool);
    output
}

fn blur_mirrored_5x5_scalar(_token: archmage::ScalarToken, input: &ImageF, weights: &[f32; 3], pool: &BufferPool) -> ImageF {
    let width = input.width();
    let height = input.height();

    let w0 = weights[0];
    let w1 = weights[1];
    let w2 = weights[2];

    let iwidth = width as i32;
    let iheight = height as i32;

    let mut temp = ImageF::from_pool_dirty(height, width, pool);

    for y in 0..height {
        let row = input.row(y);
        for x in 0..width {
            let ix = x as i32;
            let v_m2 = row[mirror(ix - 2, iwidth)];
            let v_m1 = row[mirror(ix - 1, iwidth)];
            let v_0 = row[x];
            let v_p1 = row[mirror(ix + 1, iwidth)];
            let v_p2 = row[mirror(ix + 2, iwidth)];
            let sum = v_0 * w0 + (v_m1 + v_p1) * w1 + (v_m2 + v_p2) * w2;
            temp.set(y, x, sum);
        }
    }

    let mut output = ImageF::from_pool_dirty(width, height, pool);
    for x in 0..width {
        let col = temp.row(x);
        for y in 0..height {
            let iy = y as i32;
            let v_m2 = col[mirror(iy - 2, iheight)];
            let v_m1 = col[mirror(iy - 1, iheight)];
            let v_0 = col[y];
            let v_p1 = col[mirror(iy + 1, iheight)];
            let v_p2 = col[mirror(iy + 2, iheight)];
            let sum = v_0 * w0 + (v_m1 + v_p1) * w1 + (v_m2 + v_p2) * w2;
            output.set(x, y, sum);
        }
    }

    temp.recycle(pool);
    output
}

/// Fast blur for small sigma values (optimized 5x5 kernel).
///
/// This is faster than the general blur for sigma ~= 1.0.
/// Uses clamp-and-renormalize boundary handling like the general blur.
pub fn blur_5x5(input: &ImageF, weights: &[f32; 3], pool: &BufferPool) -> ImageF {
    let width = input.width();
    let height = input.height();

    // Separable 5x5 kernel: [w2, w1, w0, w1, w2]
    let w0 = weights[0];
    let w1 = weights[1];
    let w2 = weights[2];
    let kernel = [w2, w1, w0, w1, w2];
    let weight_sum: f32 = kernel.iter().sum();
    let scale = 1.0 / weight_sum;
    let scaled_kernel: [f32; 5] = [
        kernel[0] * scale,
        kernel[1] * scale,
        kernel[2] * scale,
        kernel[3] * scale,
        kernel[4] * scale,
    ];

    // Temporary for horizontal pass (transposed)
    let mut temp = ImageF::from_pool_dirty(height, width, pool);

    // Horizontal pass with fast interior
    let border = 2.min(width);
    let interior_end = if width > 2 { width - 2 } else { 0 };

    // Left border
    for x in 0..border {
        for y in 0..height {
            let row = input.row(y);
            let minx = x.saturating_sub(2);
            let maxx = (x + 2).min(width - 1);

            let mut sum = 0.0f32;
            let mut wsum = 0.0f32;
            for j in minx..=maxx {
                let k_idx = j + 2 - x;
                let k_val = kernel[k_idx];
                wsum += k_val;
                sum += row[j] * k_val;
            }
            temp.set(y, x, if wsum > 0.0 { sum / wsum } else { 0.0 });
        }
    }

    // Interior (no bounds check)
    if interior_end > border {
        for y in 0..height {
            let row = input.row(y);
            for x in border..interior_end {
                let sum = row[x - 2] * scaled_kernel[0]
                    + row[x - 1] * scaled_kernel[1]
                    + row[x] * scaled_kernel[2]
                    + row[x + 1] * scaled_kernel[3]
                    + row[x + 2] * scaled_kernel[4];
                temp.set(y, x, sum);
            }
        }
    }

    // Right border
    for x in interior_end..width {
        for y in 0..height {
            let row = input.row(y);
            let minx = x.saturating_sub(2);
            let maxx = (x + 2).min(width - 1);

            let mut sum = 0.0f32;
            let mut wsum = 0.0f32;
            for j in minx..=maxx {
                let k_idx = j + 2 - x;
                let k_val = kernel[k_idx];
                wsum += k_val;
                sum += row[j] * k_val;
            }
            temp.set(y, x, if wsum > 0.0 { sum / wsum } else { 0.0 });
        }
    }

    // Vertical pass (on transposed data, so it's another horizontal pass)
    // Result goes back to original orientation
    let mut output = ImageF::from_pool_dirty(width, height, pool);

    let h_border = 2.min(height);
    let h_interior_end = if height > 2 { height - 2 } else { 0 };

    // Top border
    for y in 0..h_border {
        for x in 0..width {
            // temp is transposed, so temp.row(x) gives column x of original
            let col = temp.row(x);
            let miny = y.saturating_sub(2);
            let maxy = (y + 2).min(height - 1);

            let mut sum = 0.0f32;
            let mut wsum = 0.0f32;
            for j in miny..=maxy {
                let k_idx = j + 2 - y;
                let k_val = kernel[k_idx];
                wsum += k_val;
                sum += col[j] * k_val;
            }
            output.set(x, y, if wsum > 0.0 { sum / wsum } else { 0.0 });
        }
    }

    // Interior
    if h_interior_end > h_border {
        for x in 0..width {
            let col = temp.row(x);
            for y in h_border..h_interior_end {
                let sum = col[y - 2] * scaled_kernel[0]
                    + col[y - 1] * scaled_kernel[1]
                    + col[y] * scaled_kernel[2]
                    + col[y + 1] * scaled_kernel[3]
                    + col[y + 2] * scaled_kernel[4];
                output.set(x, y, sum);
            }
        }
    }

    // Bottom border
    for y in h_interior_end..height {
        for x in 0..width {
            let col = temp.row(x);
            let miny = y.saturating_sub(2);
            let maxy = (y + 2).min(height - 1);

            let mut sum = 0.0f32;
            let mut wsum = 0.0f32;
            for j in miny..=maxy {
                let k_idx = j + 2 - y;
                let k_val = kernel[k_idx];
                wsum += k_val;
                sum += col[j] * k_val;
            }
            output.set(x, y, if wsum > 0.0 { sum / wsum } else { 0.0 });
        }
    }

    temp.recycle(pool);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_generation() {
        let kernel = compute_kernel(1.0);
        assert!(!kernel.is_empty());
        assert_eq!(kernel.len() % 2, 1); // Should be odd

        // Center should be maximum
        let center = kernel.len() / 2;
        for (i, &v) in kernel.iter().enumerate() {
            if i != center {
                assert!(v <= kernel[center]);
            }
        }

        // Should sum to some positive value (un-normalized)
        let sum: f32 = kernel.iter().sum();
        assert!(sum > 0.0);
    }

    #[test]
    fn test_blur_constant_image() {
        // Blurring a constant image should give the same constant
        let pool = BufferPool::new();
        let img = ImageF::filled(32, 32, 0.5);
        let blurred = gaussian_blur(&img, 2.0, &pool);

        for y in 2..30 {
            for x in 2..30 {
                assert!(
                    (blurred.get(x, y) - 0.5).abs() < 0.01,
                    "Expected 0.5, got {} at ({}, {})",
                    blurred.get(x, y),
                    x,
                    y
                );
            }
        }
    }

    #[test]
    fn test_blur_reduces_delta() {
        // A single bright pixel should spread out
        let pool = BufferPool::new();
        let mut img = ImageF::new(32, 32);
        img.set(16, 16, 1.0);

        let blurred = gaussian_blur(&img, 2.0, &pool);

        // Center should be lower
        assert!(blurred.get(16, 16) < 1.0);
        // Neighbors should be non-zero
        assert!(blurred.get(15, 16) > 0.0);
        assert!(blurred.get(17, 16) > 0.0);
    }

    #[test]
    fn test_blur_5x5_constant() {
        let pool = BufferPool::new();
        let img = ImageF::filled(32, 32, 0.5);
        let weights = [1.0f32, 0.5, 0.25]; // Example weights
        let blurred = blur_5x5(&img, &weights, &pool);

        // Interior should stay constant
        for y in 4..28 {
            for x in 4..28 {
                assert!(
                    (blurred.get(x, y) - 0.5).abs() < 0.01,
                    "Expected 0.5, got {} at ({}, {})",
                    blurred.get(x, y),
                    x,
                    y
                );
            }
        }
    }
}
