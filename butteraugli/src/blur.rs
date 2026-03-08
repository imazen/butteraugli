//! Gaussian blur implementation for butteraugli.
//!
//! Butteraugli uses Gaussian blurs at various scales to separate
//! frequency bands. The blur is implemented as a separable convolution.
//!
//! The C++ butteraugli uses clamp-to-edge boundary handling with
//! re-normalization for border pixels. This module matches that behavior.
//!
//! Optimizations:
//! - Non-transposing H+V convolution for sequential writes (zero D1 write misses)
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

/// Non-transposing horizontal convolution for border pixels only.
///
/// Handles left (0..border1) and right (border2..width) border columns.
/// Interior pixels are handled by SIMD-specific functions.
/// When no SIMD is available (scalar path), also handles interior.
#[allow(clippy::inline_always)]
#[inline(always)]
fn convolve_horizontal_borders(
    input: &ImageF,
    kernel: &[f32],
    scaled_kernel: &[f32],
    border_ratio: f32,
    output: &mut ImageF,
    include_interior: bool,
) {
    let width = input.width();
    let height = input.height();
    let half = kernel.len() / 2;
    let weight_no_border: f32 = kernel.iter().sum();
    let border1 = half.min(width);
    let border2 = if width > half { width - half } else { 0 };

    for y in 0..height {
        let row_in = input.row(y);
        let row_out = output.row_mut(y);

        // Left border
        for x in 0..border1 {
            let minx = x.saturating_sub(half);
            let maxx = (x + half).min(width - 1);
            let k_start = minx + half - x;
            let k_end = maxx + half - x + 1;
            let ks = &kernel[k_start..k_end];
            let weight: f32 = ks.iter().sum();
            let effective = (1.0 - border_ratio) * weight + border_ratio * weight_no_border;
            let scale = 1.0 / effective;
            let sum: f32 = row_in[minx..minx + ks.len()]
                .iter()
                .zip(ks)
                .map(|(&r, &k)| r * k)
                .sum::<f32>()
                * scale;
            row_out[x] = sum;
        }

        // Interior (only for scalar fallback)
        if include_interior {
            for x in border1..border2 {
                let d = x - half;
                let base = &row_in[d..d + scaled_kernel.len()];
                let sum: f32 = base.iter().zip(scaled_kernel).map(|(&r, &k)| r * k).sum();
                row_out[x] = sum;
            }
        }

        // Right border
        for x in border2..width {
            let minx = x.saturating_sub(half);
            let maxx = (x + half).min(width - 1);
            let k_start = minx + half - x;
            let k_end = maxx + half - x + 1;
            let ks = &kernel[k_start..k_end];
            let weight: f32 = ks.iter().sum();
            let effective = (1.0 - border_ratio) * weight + border_ratio * weight_no_border;
            let scale = 1.0 / effective;
            let sum: f32 = row_in[minx..minx + ks.len()]
                .iter()
                .zip(ks)
                .map(|(&r, &k)| r * k)
                .sum::<f32>()
                * scale;
            row_out[x] = sum;
        }
    }
}

/// AVX2 non-transposing horizontal convolution interior.
#[cfg(target_arch = "x86_64")]
#[archmage::rite]
fn convolve_horizontal_interior_v3(
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
    let kernel_len = scaled_kernel.len();
    let simd_chunks = (border2 - border1) / 8;
    let simd_end = border1 + simd_chunks * 8;

    for y in 0..height {
        let row_in = input.row(y);
        let row_out = output.row_mut(y);

        // SIMD path: process 8 pixels at a time, write sequentially
        for chunk_idx in 0..simd_chunks {
            let x = border1 + chunk_idx * 8;
            let d = x - half;
            let base = &row_in[d..d + kernel_len + 7];
            let mut sum = f32x8::zero(token);

            for (j, &k) in scaled_kernel.iter().enumerate() {
                let loaded = f32x8::from_slice(token, &base[j..]);
                sum = loaded.mul_add(f32x8::splat(token, k), sum);
            }

            // Sequential write — no cache misses
            let results = sum.to_array();
            row_out[x..x + 8].copy_from_slice(&results);
        }

        // Scalar tail
        for x in simd_end..border2 {
            let d = x - half;
            let base = &row_in[d..d + kernel_len];
            let sum: f32 = base
                .iter()
                .zip(scaled_kernel)
                .fold(0.0f32, |acc, (&r, &k)| r.mul_add(k, acc));
            row_out[x] = sum;
        }
    }
}

/// AVX-512 non-transposing horizontal convolution interior.
#[cfg(target_arch = "x86_64")]
#[archmage::rite]
fn convolve_horizontal_interior_v4(
    token: archmage::X64V4Token,
    input: &ImageF,
    scaled_kernel: &[f32],
    border1: usize,
    border2: usize,
    half: usize,
    output: &mut ImageF,
) {
    use magetypes::simd::v4::f32x16;
    let height = input.height();
    let kernel_len = scaled_kernel.len();
    let simd_chunks = (border2 - border1) / 16;
    let simd_end = border1 + simd_chunks * 16;

    for y in 0..height {
        let row_in = input.row(y);
        let row_out = output.row_mut(y);

        for chunk_idx in 0..simd_chunks {
            let x = border1 + chunk_idx * 16;
            let d = x - half;
            let base = &row_in[d..d + kernel_len + 15];
            let mut sum = f32x16::zero(token);

            for (j, &k) in scaled_kernel.iter().enumerate() {
                let loaded = f32x16::from_slice(token, &base[j..]);
                sum = loaded.mul_add(f32x16::splat(token, k), sum);
            }

            let results = sum.to_array();
            row_out[x..x + 16].copy_from_slice(&results);
        }

        for x in simd_end..border2 {
            let d = x - half;
            let base = &row_in[d..d + kernel_len];
            let sum: f32 = base
                .iter()
                .zip(scaled_kernel)
                .fold(0.0f32, |acc, (&r, &k)| r.mul_add(k, acc));
            row_out[x] = sum;
        }
    }
}

/// NEON non-transposing horizontal convolution interior.
#[cfg(target_arch = "aarch64")]
#[archmage::rite]
fn convolve_horizontal_interior_neon(
    token: archmage::NeonToken,
    input: &ImageF,
    scaled_kernel: &[f32],
    border1: usize,
    border2: usize,
    half: usize,
    output: &mut ImageF,
) {
    use magetypes::simd::f32x8;
    let height = input.height();
    let kernel_len = scaled_kernel.len();
    let simd_chunks = (border2 - border1) / 8;
    let simd_end = border1 + simd_chunks * 8;

    for y in 0..height {
        let row_in = input.row(y);
        let row_out = output.row_mut(y);

        for chunk_idx in 0..simd_chunks {
            let x = border1 + chunk_idx * 8;
            let d = x - half;
            let base = &row_in[d..d + kernel_len + 7];
            let mut sum = f32x8::zero(token);

            for (j, &k) in scaled_kernel.iter().enumerate() {
                let loaded = f32x8::load(token, (&base[j..j + 8]).try_into().unwrap());
                sum = loaded.mul_add(f32x8::splat(token, k), sum);
            }

            sum.store((&mut row_out[x..x + 8]).try_into().unwrap());
        }

        for x in simd_end..border2 {
            let d = x - half;
            let base = &row_in[d..d + kernel_len];
            let sum: f32 = base
                .iter()
                .zip(scaled_kernel)
                .fold(0.0f32, |acc, (&r, &k)| r.mul_add(k, acc));
            row_out[x] = sum;
        }
    }
}

/// WASM SIMD128 non-transposing horizontal convolution interior.
#[cfg(target_arch = "wasm32")]
#[archmage::rite]
fn convolve_horizontal_interior_wasm128(
    token: archmage::Wasm128Token,
    input: &ImageF,
    scaled_kernel: &[f32],
    border1: usize,
    border2: usize,
    half: usize,
    output: &mut ImageF,
) {
    use magetypes::simd::f32x8;
    let height = input.height();
    let kernel_len = scaled_kernel.len();
    let simd_chunks = (border2 - border1) / 8;
    let simd_end = border1 + simd_chunks * 8;

    for y in 0..height {
        let row_in = input.row(y);
        let row_out = output.row_mut(y);

        for chunk_idx in 0..simd_chunks {
            let x = border1 + chunk_idx * 8;
            let d = x - half;
            let base = &row_in[d..d + kernel_len + 7];
            let mut sum = f32x8::zero(token);

            for (j, &k) in scaled_kernel.iter().enumerate() {
                let loaded = f32x8::load(token, (&base[j..j + 8]).try_into().unwrap());
                sum = loaded.mul_add(f32x8::splat(token, k), sum);
            }

            sum.store((&mut row_out[x..x + 8]).try_into().unwrap());
        }

        for x in simd_end..border2 {
            let d = x - half;
            let base = &row_in[d..d + kernel_len];
            let sum: f32 = base
                .iter()
                .zip(scaled_kernel)
                .fold(0.0f32, |acc, (&r, &k)| r.mul_add(k, acc));
            row_out[x] = sum;
        }
    }
}

/// Vertical convolution with SIMD across x dimension.
///
/// For each output row y, accumulates across kernel_len input rows.
/// All reads and writes are sequential — cache-friendly in both directions.
/// Border rows use clamp-to-edge with re-normalization.
#[cfg(target_arch = "x86_64")]
#[archmage::rite]
fn convolve_vertical_v3(
    token: archmage::X64V3Token,
    input: &ImageF,
    kernel: &[f32],
    scaled_kernel: &[f32],
    border_ratio: f32,
    output: &mut ImageF,
) {
    use magetypes::simd::f32x8;
    let width = input.width();
    let height = input.height();
    let half = kernel.len() / 2;
    let weight_no_border: f32 = kernel.iter().sum();

    let border_top = half.min(height);
    let border_bottom = if height > half { height - half } else { 0 };
    let simd_width = (width / 8) * 8;

    // Top border rows
    for y in 0..border_top {
        let miny = y.saturating_sub(half);
        let maxy = (y + half).min(height - 1);
        let k_start = miny + half - y;
        let k_end = maxy + half - y + 1;
        let ks = &kernel[k_start..k_end];
        let weight: f32 = ks.iter().sum();
        let effective = (1.0 - border_ratio) * weight + border_ratio * weight_no_border;
        let scale = 1.0 / effective;

        let row_out = output.row_mut(y);
        let mut x = 0;
        while x + 8 <= simd_width {
            let mut sum = f32x8::zero(token);
            for (ki, &kw) in ks.iter().enumerate() {
                let src_y = miny + ki;
                let row_in = input.row(src_y);
                let loaded = f32x8::load(token, (&row_in[x..x + 8]).try_into().unwrap());
                sum = loaded.mul_add(f32x8::splat(token, kw * scale), sum);
            }
            sum.store((&mut row_out[x..x + 8]).try_into().unwrap());
            x += 8;
        }
        while x < width {
            let mut sum = 0.0f32;
            for (ki, &kw) in ks.iter().enumerate() {
                sum += input.row(miny + ki)[x] * kw;
            }
            row_out[x] = sum * scale;
            x += 1;
        }
    }

    // Interior rows: row-major loop, first kernel row mul-stores (no zero-init needed).
    for y in border_top..border_bottom {
        let start_y = y - half;

        // First kernel row: multiply-store (replaces fill(0.0) + first FMA)
        {
            let row_in = input.row(start_y);
            let kv = f32x8::splat(token, scaled_kernel[0]);
            let row_out = output.row_mut(y);
            for (in_c, out_c) in row_in[..simd_width]
                .chunks_exact(8)
                .zip(row_out[..simd_width].chunks_exact_mut(8))
            {
                let result = f32x8::load(token, in_c.try_into().unwrap()) * kv;
                result.store(out_c.try_into().unwrap());
            }
        }

        // Remaining kernel rows: FMA into output
        for (ki, &kw) in scaled_kernel.iter().enumerate().skip(1) {
            let row_in = input.row(start_y + ki);
            let kv = f32x8::splat(token, kw);
            let row_out = output.row_mut(y);
            for (in_c, out_c) in row_in[..simd_width]
                .chunks_exact(8)
                .zip(row_out[..simd_width].chunks_exact_mut(8))
            {
                let loaded = f32x8::load(token, in_c.try_into().unwrap());
                let current = f32x8::load(token, (&*out_c).try_into().unwrap());
                loaded.mul_add(kv, current).store(out_c.try_into().unwrap());
            }
        }

        // Scalar tail
        let row_out = output.row_mut(y);
        for x in simd_width..width {
            let mut sum = input.row(start_y)[x] * scaled_kernel[0];
            for (ki, &kw) in scaled_kernel.iter().enumerate().skip(1) {
                sum += input.row(start_y + ki)[x] * kw;
            }
            row_out[x] = sum;
        }
    }

    // Bottom border rows
    for y in border_bottom..height {
        let miny = y.saturating_sub(half);
        let maxy = (y + half).min(height - 1);
        let k_start = miny + half - y;
        let k_end = maxy + half - y + 1;
        let ks = &kernel[k_start..k_end];
        let weight: f32 = ks.iter().sum();
        let effective = (1.0 - border_ratio) * weight + border_ratio * weight_no_border;
        let scale = 1.0 / effective;

        let row_out = output.row_mut(y);
        let mut x = 0;
        while x + 8 <= simd_width {
            let mut sum = f32x8::zero(token);
            for (ki, &kw) in ks.iter().enumerate() {
                let src_y = miny + ki;
                let row_in = input.row(src_y);
                let loaded = f32x8::load(token, (&row_in[x..x + 8]).try_into().unwrap());
                sum = loaded.mul_add(f32x8::splat(token, kw * scale), sum);
            }
            sum.store((&mut row_out[x..x + 8]).try_into().unwrap());
            x += 8;
        }
        while x < width {
            let mut sum = 0.0f32;
            for (ki, &kw) in ks.iter().enumerate() {
                sum += input.row(miny + ki)[x] * kw;
            }
            row_out[x] = sum * scale;
            x += 1;
        }
    }
}

/// AVX-512 vertical convolution.
#[cfg(target_arch = "x86_64")]
#[archmage::rite]
fn convolve_vertical_v4(
    token: archmage::X64V4Token,
    input: &ImageF,
    kernel: &[f32],
    scaled_kernel: &[f32],
    border_ratio: f32,
    output: &mut ImageF,
) {
    use magetypes::simd::v4::f32x16;
    let width = input.width();
    let height = input.height();
    let half = kernel.len() / 2;
    let weight_no_border: f32 = kernel.iter().sum();

    let border_top = half.min(height);
    let border_bottom = if height > half { height - half } else { 0 };
    let simd_width = (width / 16) * 16;

    // Helper: process one row with border handling
    let process_border_row =
        |y: usize, miny: usize, ks: &[f32], scale: f32, row_out: &mut [f32]| {
            let mut x = 0;
            while x + 16 <= simd_width {
                let mut sum = f32x16::zero(token);
                for (ki, &kw) in ks.iter().enumerate() {
                    let row_in = input.row(miny + ki);
                    let loaded = f32x16::from_slice(token, &row_in[x..]);
                    sum = loaded.mul_add(f32x16::splat(token, kw * scale), sum);
                }
                let results = sum.to_array();
                row_out[x..x + 16].copy_from_slice(&results);
                x += 16;
            }
            while x < width {
                let mut sum = 0.0f32;
                for (ki, &kw) in ks.iter().enumerate() {
                    sum += input.row(miny + ki)[x] * kw;
                }
                row_out[x] = sum * scale;
                x += 1;
            }
            let _ = y; // suppress warning
        };

    for y in 0..border_top {
        let miny = y.saturating_sub(half);
        let maxy = (y + half).min(height - 1);
        let k_start = miny + half - y;
        let k_end = maxy + half - y + 1;
        let ks = &kernel[k_start..k_end];
        let weight: f32 = ks.iter().sum();
        let effective = (1.0 - border_ratio) * weight + border_ratio * weight_no_border;
        let scale = 1.0 / effective;
        let row_out = output.row_mut(y);
        process_border_row(y, miny, ks, scale, row_out);
    }

    let simd16_width = (width / 16) * 16;
    for y in border_top..border_bottom {
        let start_y = y - half;

        // First kernel row: multiply-store (no zero-init needed)
        {
            let row_in = input.row(start_y);
            let kv = f32x16::splat(token, scaled_kernel[0]);
            let row_out = output.row_mut(y);
            for (in_c, out_c) in row_in[..simd16_width]
                .chunks_exact(16)
                .zip(row_out[..simd16_width].chunks_exact_mut(16))
            {
                let result = f32x16::from_slice(token, in_c) * kv;
                out_c.copy_from_slice(&result.to_array());
            }
        }

        // Remaining kernel rows: FMA into output
        for (ki, &kw) in scaled_kernel.iter().enumerate().skip(1) {
            let row_in = input.row(start_y + ki);
            let kv = f32x16::splat(token, kw);
            let row_out = output.row_mut(y);
            for (in_c, out_c) in row_in[..simd16_width]
                .chunks_exact(16)
                .zip(row_out[..simd16_width].chunks_exact_mut(16))
            {
                let loaded = f32x16::from_slice(token, in_c);
                let current = f32x16::from_slice(token, out_c);
                let result = loaded.mul_add(kv, current);
                out_c.copy_from_slice(&result.to_array());
            }
        }

        // Scalar tail
        let row_out = output.row_mut(y);
        for x in simd16_width..width {
            let mut sum = input.row(start_y)[x] * scaled_kernel[0];
            for (ki, &kw) in scaled_kernel.iter().enumerate().skip(1) {
                sum += input.row(start_y + ki)[x] * kw;
            }
            row_out[x] = sum;
        }
    }

    for y in border_bottom..height {
        let miny = y.saturating_sub(half);
        let maxy = (y + half).min(height - 1);
        let k_start = miny + half - y;
        let k_end = maxy + half - y + 1;
        let ks = &kernel[k_start..k_end];
        let weight: f32 = ks.iter().sum();
        let effective = (1.0 - border_ratio) * weight + border_ratio * weight_no_border;
        let scale = 1.0 / effective;
        let row_out = output.row_mut(y);
        process_border_row(y, miny, ks, scale, row_out);
    }
}

/// NEON vertical convolution.
#[cfg(target_arch = "aarch64")]
#[archmage::rite]
fn convolve_vertical_neon(
    token: archmage::NeonToken,
    input: &ImageF,
    kernel: &[f32],
    scaled_kernel: &[f32],
    border_ratio: f32,
    output: &mut ImageF,
) {
    use magetypes::simd::f32x8;
    let width = input.width();
    let height = input.height();
    let half = kernel.len() / 2;
    let weight_no_border: f32 = kernel.iter().sum();

    let border_top = half.min(height);
    let border_bottom = if height > half { height - half } else { 0 };
    let simd_width = (width / 8) * 8;

    // Border row helper
    let process_border_row = |miny: usize, ks: &[f32], scale: f32, row_out: &mut [f32]| {
        let mut x = 0;
        while x + 8 <= simd_width {
            let mut sum = f32x8::zero(token);
            for (ki, &kw) in ks.iter().enumerate() {
                let row_in = input.row(miny + ki);
                let loaded = f32x8::load(token, (&row_in[x..x + 8]).try_into().unwrap());
                sum = loaded.mul_add(f32x8::splat(token, kw * scale), sum);
            }
            sum.store((&mut row_out[x..x + 8]).try_into().unwrap());
            x += 8;
        }
        while x < width {
            let mut sum = 0.0f32;
            for (ki, &kw) in ks.iter().enumerate() {
                sum += input.row(miny + ki)[x] * kw;
            }
            row_out[x] = sum * scale;
            x += 1;
        }
    };

    for y in 0..border_top {
        let miny = y.saturating_sub(half);
        let maxy = (y + half).min(height - 1);
        let k_start = miny + half - y;
        let k_end = maxy + half - y + 1;
        let ks = &kernel[k_start..k_end];
        let weight: f32 = ks.iter().sum();
        let effective = (1.0 - border_ratio) * weight + border_ratio * weight_no_border;
        let scale = 1.0 / effective;
        process_border_row(miny, ks, scale, output.row_mut(y));
    }

    for y in border_top..border_bottom {
        let start_y = y - half;

        // First kernel row: multiply-store (no zero-init needed)
        {
            let row_in = input.row(start_y);
            let kv = f32x8::splat(token, scaled_kernel[0]);
            let row_out = output.row_mut(y);
            for (in_c, out_c) in row_in[..simd_width]
                .chunks_exact(8)
                .zip(row_out[..simd_width].chunks_exact_mut(8))
            {
                let result = f32x8::load(token, in_c.try_into().unwrap()) * kv;
                result.store(out_c.try_into().unwrap());
            }
        }

        // Remaining kernel rows: FMA into output
        for (ki, &kw) in scaled_kernel.iter().enumerate().skip(1) {
            let row_in = input.row(start_y + ki);
            let kv = f32x8::splat(token, kw);
            let row_out = output.row_mut(y);
            for (in_c, out_c) in row_in[..simd_width]
                .chunks_exact(8)
                .zip(row_out[..simd_width].chunks_exact_mut(8))
            {
                let loaded = f32x8::load(token, in_c.try_into().unwrap());
                let current = f32x8::load(token, (&*out_c).try_into().unwrap());
                loaded.mul_add(kv, current).store(out_c.try_into().unwrap());
            }
        }

        // Scalar tail
        let row_out = output.row_mut(y);
        for x in simd_width..width {
            let mut sum = input.row(start_y)[x] * scaled_kernel[0];
            for (ki, &kw) in scaled_kernel.iter().enumerate().skip(1) {
                sum += input.row(start_y + ki)[x] * kw;
            }
            row_out[x] = sum;
        }
    }

    for y in border_bottom..height {
        let miny = y.saturating_sub(half);
        let maxy = (y + half).min(height - 1);
        let k_start = miny + half - y;
        let k_end = maxy + half - y + 1;
        let ks = &kernel[k_start..k_end];
        let weight: f32 = ks.iter().sum();
        let effective = (1.0 - border_ratio) * weight + border_ratio * weight_no_border;
        let scale = 1.0 / effective;
        process_border_row(miny, ks, scale, output.row_mut(y));
    }
}

/// WASM SIMD128 vertical convolution.
#[cfg(target_arch = "wasm32")]
#[archmage::rite]
fn convolve_vertical_wasm128(
    token: archmage::Wasm128Token,
    input: &ImageF,
    kernel: &[f32],
    scaled_kernel: &[f32],
    border_ratio: f32,
    output: &mut ImageF,
) {
    use magetypes::simd::f32x8;
    let width = input.width();
    let height = input.height();
    let half = kernel.len() / 2;
    let weight_no_border: f32 = kernel.iter().sum();

    let border_top = half.min(height);
    let border_bottom = if height > half { height - half } else { 0 };
    let simd_width = (width / 8) * 8;

    let process_border_row = |miny: usize, ks: &[f32], scale: f32, row_out: &mut [f32]| {
        let mut x = 0;
        while x + 8 <= simd_width {
            let mut sum = f32x8::zero(token);
            for (ki, &kw) in ks.iter().enumerate() {
                let row_in = input.row(miny + ki);
                let loaded = f32x8::load(token, (&row_in[x..x + 8]).try_into().unwrap());
                sum = loaded.mul_add(f32x8::splat(token, kw * scale), sum);
            }
            sum.store((&mut row_out[x..x + 8]).try_into().unwrap());
            x += 8;
        }
        while x < width {
            let mut sum = 0.0f32;
            for (ki, &kw) in ks.iter().enumerate() {
                sum += input.row(miny + ki)[x] * kw;
            }
            row_out[x] = sum * scale;
            x += 1;
        }
    };

    for y in 0..border_top {
        let miny = y.saturating_sub(half);
        let maxy = (y + half).min(height - 1);
        let k_start = miny + half - y;
        let k_end = maxy + half - y + 1;
        let ks = &kernel[k_start..k_end];
        let weight: f32 = ks.iter().sum();
        let effective = (1.0 - border_ratio) * weight + border_ratio * weight_no_border;
        let scale = 1.0 / effective;
        process_border_row(miny, ks, scale, output.row_mut(y));
    }

    for y in border_top..border_bottom {
        let start_y = y - half;

        // First kernel row: multiply-store (no zero-init needed)
        {
            let row_in = input.row(start_y);
            let kv = f32x8::splat(token, scaled_kernel[0]);
            let row_out = output.row_mut(y);
            for (in_c, out_c) in row_in[..simd_width]
                .chunks_exact(8)
                .zip(row_out[..simd_width].chunks_exact_mut(8))
            {
                let result = f32x8::load(token, in_c.try_into().unwrap()) * kv;
                result.store(out_c.try_into().unwrap());
            }
        }

        // Remaining kernel rows: FMA into output
        for (ki, &kw) in scaled_kernel.iter().enumerate().skip(1) {
            let row_in = input.row(start_y + ki);
            let kv = f32x8::splat(token, kw);
            let row_out = output.row_mut(y);
            for (in_c, out_c) in row_in[..simd_width]
                .chunks_exact(8)
                .zip(row_out[..simd_width].chunks_exact_mut(8))
            {
                let loaded = f32x8::load(token, in_c.try_into().unwrap());
                let current = f32x8::load(token, (&*out_c).try_into().unwrap());
                loaded.mul_add(kv, current).store(out_c.try_into().unwrap());
            }
        }

        // Scalar tail
        let row_out = output.row_mut(y);
        for x in simd_width..width {
            let mut sum = input.row(start_y)[x] * scaled_kernel[0];
            for (ki, &kw) in scaled_kernel.iter().enumerate().skip(1) {
                sum += input.row(start_y + ki)[x] * kw;
            }
            row_out[x] = sum;
        }
    }

    for y in border_bottom..height {
        let miny = y.saturating_sub(half);
        let maxy = (y + half).min(height - 1);
        let k_start = miny + half - y;
        let k_end = maxy + half - y + 1;
        let ks = &kernel[k_start..k_end];
        let weight: f32 = ks.iter().sum();
        let effective = (1.0 - border_ratio) * weight + border_ratio * weight_no_border;
        let scale = 1.0 / effective;
        process_border_row(miny, ks, scale, output.row_mut(y));
    }
}

/// Scalar vertical convolution fallback.
fn convolve_vertical_scalar(
    input: &ImageF,
    kernel: &[f32],
    scaled_kernel: &[f32],
    border_ratio: f32,
    output: &mut ImageF,
) {
    let width = input.width();
    let height = input.height();
    let half = kernel.len() / 2;
    let weight_no_border: f32 = kernel.iter().sum();

    let border_top = half.min(height);
    let border_bottom = if height > half { height - half } else { 0 };

    // Border row helper
    let process_border_row = |miny: usize, ks: &[f32], scale: f32, row_out: &mut [f32]| {
        for x in 0..width {
            let mut sum = 0.0f32;
            for (ki, &kw) in ks.iter().enumerate() {
                sum += input.row(miny + ki)[x] * kw;
            }
            row_out[x] = sum * scale;
        }
    };

    for y in 0..border_top {
        let miny = y.saturating_sub(half);
        let maxy = (y + half).min(height - 1);
        let k_start = miny + half - y;
        let k_end = maxy + half - y + 1;
        let ks = &kernel[k_start..k_end];
        let weight: f32 = ks.iter().sum();
        let effective = (1.0 - border_ratio) * weight + border_ratio * weight_no_border;
        let scale = 1.0 / effective;
        process_border_row(miny, ks, scale, output.row_mut(y));
    }

    for y in border_top..border_bottom {
        let start_y = y - half;

        // First kernel row: multiply-store (no zero-init needed)
        {
            let row_in = input.row(start_y);
            let kw = scaled_kernel[0];
            let row_out = output.row_mut(y);
            for (out_v, &in_v) in row_out[..width].iter_mut().zip(row_in[..width].iter()) {
                *out_v = in_v * kw;
            }
        }

        // Remaining kernel rows: FMA into output
        for (ki, &kw) in scaled_kernel.iter().enumerate().skip(1) {
            let row_in = input.row(start_y + ki);
            let row_out = output.row_mut(y);
            for (out_v, &in_v) in row_out[..width].iter_mut().zip(row_in[..width].iter()) {
                *out_v = in_v.mul_add(kw, *out_v);
            }
        }
    }

    for y in border_bottom..height {
        let miny = y.saturating_sub(half);
        let maxy = (y + half).min(height - 1);
        let k_start = miny + half - y;
        let k_end = maxy + half - y + 1;
        let ks = &kernel[k_start..k_end];
        let weight: f32 = ks.iter().sum();
        let effective = (1.0 - border_ratio) * weight + border_ratio * weight_no_border;
        let scale = 1.0 / effective;
        process_border_row(miny, ks, scale, output.row_mut(y));
    }
}

/// Applies a 2D Gaussian blur to an image.
///
/// This is implemented as two separable 1D convolutions:
/// 1. Horizontal convolution (non-transposing, sequential writes)
/// 2. Vertical convolution with SIMD across x (sequential reads and writes)
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
    archmage::incant!(
        gaussian_blur_dispatch(input, sigma, pool),
        [v4, v3, neon, wasm128]
    )
}

/// Maximum possible kernel size: 2 * floor(M * sigma_max) + 1.
/// With M=2.25 and SIGMA_LF ≈ 7.156: 2 * 16 + 1 = 33.
const MAX_KERNEL_SIZE: usize = 64;

/// Computes normalized scaled kernel on the stack, avoiding Vec allocation.
/// Returns the used portion of the buffer.
#[inline]
fn compute_scaled_kernel<'a>(
    kernel: &[f32],
    buf: &'a mut [f32; MAX_KERNEL_SIZE],
) -> &'a [f32] {
    debug_assert!(kernel.len() <= MAX_KERNEL_SIZE);
    let weight: f32 = kernel.iter().sum();
    let inv_weight = 1.0 / weight;
    for (i, &k) in kernel.iter().enumerate() {
        buf[i] = k * inv_weight;
    }
    &buf[..kernel.len()]
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
    let half = kernel.len() / 2;
    let mut scaled_buf = [0.0f32; MAX_KERNEL_SIZE];
    let scaled = compute_scaled_kernel(&kernel, &mut scaled_buf);
    let width = input.width();
    let height = input.height();
    let border1 = half.min(width);
    let border2 = if width > half { width - half } else { 0 };

    // H-pass: non-transposing
    let mut temp = ImageF::from_pool_dirty(width, height, pool);
    convolve_horizontal_borders(input, &kernel, &scaled, 0.0, &mut temp, false);
    if border2 > border1 {
        convolve_horizontal_interior_v4(token, input, &scaled, border1, border2, half, &mut temp);
    }

    // V-pass: accumulate across rows
    let mut output = ImageF::from_pool_dirty(width, height, pool);
    convolve_vertical_v4(token, &temp, &kernel, &scaled, 0.0, &mut output);
    temp.recycle(pool);
    output
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
    let half = kernel.len() / 2;
    let mut scaled_buf = [0.0f32; MAX_KERNEL_SIZE];
    let scaled = compute_scaled_kernel(&kernel, &mut scaled_buf);
    let width = input.width();
    let height = input.height();
    let border1 = half.min(width);
    let border2 = if width > half { width - half } else { 0 };

    // H-pass: non-transposing
    let mut temp = ImageF::from_pool_dirty(width, height, pool);
    convolve_horizontal_borders(input, &kernel, &scaled, 0.0, &mut temp, false);
    if border2 > border1 {
        convolve_horizontal_interior_v3(token, input, &scaled, border1, border2, half, &mut temp);
    }

    // V-pass: accumulate across rows
    let mut output = ImageF::from_pool_dirty(width, height, pool);
    convolve_vertical_v3(token, &temp, &kernel, &scaled, 0.0, &mut output);
    temp.recycle(pool);
    output
}

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn gaussian_blur_dispatch_neon(
    token: archmage::NeonToken,
    input: &ImageF,
    sigma: f32,
    pool: &BufferPool,
) -> ImageF {
    let kernel = compute_kernel(sigma);
    let half = kernel.len() / 2;
    let mut scaled_buf = [0.0f32; MAX_KERNEL_SIZE];
    let scaled = compute_scaled_kernel(&kernel, &mut scaled_buf);
    let width = input.width();
    let height = input.height();
    let border1 = half.min(width);
    let border2 = if width > half { width - half } else { 0 };

    let mut temp = ImageF::from_pool_dirty(width, height, pool);
    convolve_horizontal_borders(input, &kernel, &scaled, 0.0, &mut temp, false);
    if border2 > border1 {
        convolve_horizontal_interior_neon(token, input, &scaled, border1, border2, half, &mut temp);
    }

    let mut output = ImageF::from_pool_dirty(width, height, pool);
    convolve_vertical_neon(token, &temp, &kernel, &scaled, 0.0, &mut output);
    temp.recycle(pool);
    output
}

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
fn gaussian_blur_dispatch_wasm128(
    token: archmage::Wasm128Token,
    input: &ImageF,
    sigma: f32,
    pool: &BufferPool,
) -> ImageF {
    let kernel = compute_kernel(sigma);
    let half = kernel.len() / 2;
    let mut scaled_buf = [0.0f32; MAX_KERNEL_SIZE];
    let scaled = compute_scaled_kernel(&kernel, &mut scaled_buf);
    let width = input.width();
    let height = input.height();
    let border1 = half.min(width);
    let border2 = if width > half { width - half } else { 0 };

    let mut temp = ImageF::from_pool_dirty(width, height, pool);
    convolve_horizontal_borders(input, &kernel, &scaled, 0.0, &mut temp, false);
    if border2 > border1 {
        convolve_horizontal_interior_wasm128(
            token, input, &scaled, border1, border2, half, &mut temp,
        );
    }

    let mut output = ImageF::from_pool_dirty(width, height, pool);
    convolve_vertical_wasm128(token, &temp, &kernel, &scaled, 0.0, &mut output);
    temp.recycle(pool);
    output
}

fn gaussian_blur_dispatch_scalar(
    _token: archmage::ScalarToken,
    input: &ImageF,
    sigma: f32,
    pool: &BufferPool,
) -> ImageF {
    let kernel = compute_kernel(sigma);
    let mut scaled_buf = [0.0f32; MAX_KERNEL_SIZE];
    let scaled = compute_scaled_kernel(&kernel, &mut scaled_buf);
    let width = input.width();
    let height = input.height();

    let mut temp = ImageF::from_pool_dirty(width, height, pool);
    convolve_horizontal_borders(input, &kernel, &scaled, 0.0, &mut temp, true);

    let mut output = ImageF::from_pool_dirty(width, height, pool);
    convolve_vertical_scalar(&temp, &kernel, &scaled, 0.0, &mut output);
    temp.recycle(pool);
    output
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
    archmage::incant!(
        blur_with_border_dispatch(input, sigma, border_ratio, pool),
        [v4, v3, neon, wasm128]
    )
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
    let half = kernel.len() / 2;
    let mut scaled_buf = [0.0f32; MAX_KERNEL_SIZE];
    let scaled = compute_scaled_kernel(&kernel, &mut scaled_buf);
    let width = input.width();
    let height = input.height();
    let border1 = half.min(width);
    let border2 = if width > half { width - half } else { 0 };

    let mut temp = ImageF::from_pool_dirty(width, height, pool);
    convolve_horizontal_borders(input, &kernel, &scaled, border_ratio, &mut temp, false);
    if border2 > border1 {
        convolve_horizontal_interior_v4(token, input, &scaled, border1, border2, half, &mut temp);
    }

    let mut output = ImageF::from_pool_dirty(width, height, pool);
    convolve_vertical_v4(token, &temp, &kernel, &scaled, border_ratio, &mut output);
    temp.recycle(pool);
    output
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
    let half = kernel.len() / 2;
    let mut scaled_buf = [0.0f32; MAX_KERNEL_SIZE];
    let scaled = compute_scaled_kernel(&kernel, &mut scaled_buf);
    let width = input.width();
    let height = input.height();
    let border1 = half.min(width);
    let border2 = if width > half { width - half } else { 0 };

    let mut temp = ImageF::from_pool_dirty(width, height, pool);
    convolve_horizontal_borders(input, &kernel, &scaled, border_ratio, &mut temp, false);
    if border2 > border1 {
        convolve_horizontal_interior_v3(token, input, &scaled, border1, border2, half, &mut temp);
    }

    let mut output = ImageF::from_pool_dirty(width, height, pool);
    convolve_vertical_v3(token, &temp, &kernel, &scaled, border_ratio, &mut output);
    temp.recycle(pool);
    output
}

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn blur_with_border_dispatch_neon(
    token: archmage::NeonToken,
    input: &ImageF,
    sigma: f32,
    border_ratio: f32,
    pool: &BufferPool,
) -> ImageF {
    let kernel = compute_kernel(sigma);
    let half = kernel.len() / 2;
    let mut scaled_buf = [0.0f32; MAX_KERNEL_SIZE];
    let scaled = compute_scaled_kernel(&kernel, &mut scaled_buf);
    let width = input.width();
    let height = input.height();
    let border1 = half.min(width);
    let border2 = if width > half { width - half } else { 0 };

    let mut temp = ImageF::from_pool_dirty(width, height, pool);
    convolve_horizontal_borders(input, &kernel, &scaled, border_ratio, &mut temp, false);
    if border2 > border1 {
        convolve_horizontal_interior_neon(token, input, &scaled, border1, border2, half, &mut temp);
    }

    let mut output = ImageF::from_pool_dirty(width, height, pool);
    convolve_vertical_neon(token, &temp, &kernel, &scaled, border_ratio, &mut output);
    temp.recycle(pool);
    output
}

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
fn blur_with_border_dispatch_wasm128(
    token: archmage::Wasm128Token,
    input: &ImageF,
    sigma: f32,
    border_ratio: f32,
    pool: &BufferPool,
) -> ImageF {
    let kernel = compute_kernel(sigma);
    let half = kernel.len() / 2;
    let mut scaled_buf = [0.0f32; MAX_KERNEL_SIZE];
    let scaled = compute_scaled_kernel(&kernel, &mut scaled_buf);
    let width = input.width();
    let height = input.height();
    let border1 = half.min(width);
    let border2 = if width > half { width - half } else { 0 };

    let mut temp = ImageF::from_pool_dirty(width, height, pool);
    convolve_horizontal_borders(input, &kernel, &scaled, border_ratio, &mut temp, false);
    if border2 > border1 {
        convolve_horizontal_interior_wasm128(
            token, input, &scaled, border1, border2, half, &mut temp,
        );
    }

    let mut output = ImageF::from_pool_dirty(width, height, pool);
    convolve_vertical_wasm128(token, &temp, &kernel, &scaled, border_ratio, &mut output);
    temp.recycle(pool);
    output
}

fn blur_with_border_dispatch_scalar(
    _token: archmage::ScalarToken,
    input: &ImageF,
    sigma: f32,
    border_ratio: f32,
    pool: &BufferPool,
) -> ImageF {
    let kernel = compute_kernel(sigma);
    let mut scaled_buf = [0.0f32; MAX_KERNEL_SIZE];
    let scaled = compute_scaled_kernel(&kernel, &mut scaled_buf);
    let width = input.width();
    let height = input.height();

    let mut temp = ImageF::from_pool_dirty(width, height, pool);
    convolve_horizontal_borders(input, &kernel, &scaled, border_ratio, &mut temp, true);

    let mut output = ImageF::from_pool_dirty(width, height, pool);
    convolve_vertical_scalar(&temp, &kernel, &scaled, border_ratio, &mut output);
    temp.recycle(pool);
    output
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
    archmage::incant!(
        blur_mirrored_5x5(input, weights, pool),
        [v4, v3, neon, wasm128]
    )
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn blur_mirrored_5x5_v4(
    token: archmage::X64V4Token,
    input: &ImageF,
    weights: &[f32; 3],
    pool: &BufferPool,
) -> ImageF {
    use magetypes::simd::v4::f32x16;

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

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn blur_mirrored_5x5_neon(
    token: archmage::NeonToken,
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

        // Interior SIMD (8 at a time, 2×NEON f32x4)
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

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
fn blur_mirrored_5x5_wasm128(
    token: archmage::Wasm128Token,
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

        for x in 0..border {
            let ix = x as i32;
            let v_m2 = row[mirror(ix - 2, iwidth)];
            let v_m1 = row[mirror(ix - 1, iwidth)];
            let v_0 = row[x];
            let v_p1 = row[mirror(ix + 1, iwidth)];
            let v_p2 = row[mirror(ix + 2, iwidth)];
            out_row[x] = v_0 * w0 + (v_m1 + v_p1) * w1 + (v_m2 + v_p2) * w2;
        }

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

        while x < interior_end {
            let v_m2 = row[x - 2];
            let v_m1 = row[x - 1];
            let v_0 = row[x];
            let v_p1 = row[x + 1];
            let v_p2 = row[x + 2];
            out_row[x] = v_0 * w0 + (v_m1 + v_p1) * w1 + (v_m2 + v_p2) * w2;
            x += 1;
        }

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

fn blur_mirrored_5x5_scalar(
    _token: archmage::ScalarToken,
    input: &ImageF,
    weights: &[f32; 3],
    pool: &BufferPool,
) -> ImageF {
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
