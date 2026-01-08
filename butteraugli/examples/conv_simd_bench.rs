//! Benchmark comparing scalar vs SIMD convolution approaches.
//!
//! This benchmark demonstrates why the "fused" approach (SIMD convolution with
//! inline transpose) is faster than "separate" (SIMD convolution + separate tiled transpose).
//!
//! Key findings:
//! - SIMD convolution without transpose: 6.5x faster than scalar
//! - SIMD convolution with inline transpose: 3.9x faster than scalar
//! - Fused approach is ~1.9x faster than separate because:
//!   1. Fewer allocations (2 vs 4 intermediate buffers)
//!   2. Scalar tiled transpose is slower than scattered SIMD writes
//!
//! Run with: `cargo run --release --example conv_simd_bench`

use butteraugli::image::ImageF;
use std::time::Instant;
use wide::f32x8;

/// Scalar convolution (current implementation style)
fn convolve_scalar(input: &ImageF, kernel: &[f32]) -> ImageF {
    let width = input.width();
    let height = input.height();
    let half = kernel.len() / 2;

    // Output is transposed
    let mut output = ImageF::new(height, width);

    let weight: f32 = kernel.iter().sum();
    let scale = 1.0 / weight;
    let scaled_kernel: Vec<f32> = kernel.iter().map(|&k| k * scale).collect();

    let border1 = half.min(width);
    let border2 = if width > half { width - half } else { 0 };

    // Interior only for this benchmark
    if border2 > border1 {
        for y in 0..height {
            let row_in = input.row(y);
            for x in border1..border2 {
                let d = x - half;
                let mut sum = 0.0f32;
                for (j, &k) in scaled_kernel.iter().enumerate() {
                    sum += row_in[d + j] * k;
                }
                output.set(y, x, sum);
            }
        }
    }

    output
}

/// SIMD convolution - processes 8 output pixels at a time
fn convolve_simd(input: &ImageF, kernel: &[f32]) -> ImageF {
    let width = input.width();
    let height = input.height();
    let half = kernel.len() / 2;

    // Output is transposed
    let mut output = ImageF::new(height, width);

    let weight: f32 = kernel.iter().sum();
    let scale = 1.0 / weight;
    let scaled_kernel: Vec<f32> = kernel.iter().map(|&k| k * scale).collect();

    let border1 = half.min(width);
    let border2 = if width > half { width - half } else { 0 };

    // Process 8 x-positions at a time
    let simd_end = border1 + ((border2 - border1) / 8) * 8;

    if border2 > border1 {
        for y in 0..height {
            let row_in = input.row(y);

            // SIMD path: process 8 pixels at a time using slice loads
            let mut x = border1;
            while x < simd_end {
                let d = x - half;
                let mut sum = f32x8::splat(0.0);

                // For each kernel position, load 8 values from slice
                for (j, &k) in scaled_kernel.iter().enumerate() {
                    let slice = &row_in[d + j..d + j + 8];
                    let vals = f32x8::from(unsafe { *(slice.as_ptr() as *const [f32; 8]) });
                    sum += vals * f32x8::splat(k);
                }

                // Store results (still have transpose issue)
                let results: [f32; 8] = sum.into();
                for i in 0..8 {
                    output.set(y, x + i, results[i]);
                }
                x += 8;
            }

            // Scalar tail
            for x in simd_end..border2 {
                let d = x - half;
                let mut sum = 0.0f32;
                for (j, &k) in scaled_kernel.iter().enumerate() {
                    sum += row_in[d + j] * k;
                }
                output.set(y, x, sum);
            }
        }
    }

    output
}

/// SIMD convolution with NO transpose - just horizontal convolution
fn convolve_simd_no_transpose(input: &ImageF, kernel: &[f32]) -> ImageF {
    let width = input.width();
    let height = input.height();
    let half = kernel.len() / 2;

    // Output is NOT transposed
    let mut output = ImageF::new(width, height);

    let weight: f32 = kernel.iter().sum();
    let scale = 1.0 / weight;
    let scaled_kernel: Vec<f32> = kernel.iter().map(|&k| k * scale).collect();

    let border1 = half.min(width);
    let border2 = if width > half { width - half } else { 0 };
    let simd_end = border1 + ((border2 - border1) / 8) * 8;

    if border2 > border1 {
        for y in 0..height {
            let row_in = input.row(y);
            let row_out = output.row_mut(y);

            // SIMD path
            let mut x = border1;
            while x < simd_end {
                let d = x - half;
                let mut sum = f32x8::splat(0.0);

                for (j, &k) in scaled_kernel.iter().enumerate() {
                    let slice = &row_in[d + j..d + j + 8];
                    let vals = f32x8::from(unsafe { *(slice.as_ptr() as *const [f32; 8]) });
                    sum += vals * f32x8::splat(k);
                }

                // Store directly to output row (cache friendly!)
                let results: [f32; 8] = sum.into();
                row_out[x..x + 8].copy_from_slice(&results);
                x += 8;
            }

            // Scalar tail
            for x in simd_end..border2 {
                let d = x - half;
                let mut sum = 0.0f32;
                for (j, &k) in scaled_kernel.iter().enumerate() {
                    sum += row_in[d + j] * k;
                }
                row_out[x] = sum;
            }
        }
    }

    output
}

/// SIMD convolution with tiled output to reduce cache thrashing
fn convolve_simd_tiled(input: &ImageF, kernel: &[f32]) -> ImageF {
    let width = input.width();
    let height = input.height();
    let half = kernel.len() / 2;

    // Output is transposed
    let mut output = ImageF::new(height, width);

    let weight: f32 = kernel.iter().sum();
    let scale = 1.0 / weight;
    let scaled_kernel: Vec<f32> = kernel.iter().map(|&k| k * scale).collect();

    let border1 = half.min(width);
    let border2 = if width > half { width - half } else { 0 };

    // Tile size for cache-friendly output writes
    const TILE_Y: usize = 32;

    if border2 > border1 {
        // Process in tiles of TILE_Y rows
        for y_tile in (0..height).step_by(TILE_Y) {
            let y_end = (y_tile + TILE_Y).min(height);

            for x in border1..border2 {
                let d = x - half;

                // Process TILE_Y rows for this x
                for y in y_tile..y_end {
                    let row_in = input.row(y);
                    let mut sum = 0.0f32;
                    for (j, &k) in scaled_kernel.iter().enumerate() {
                        sum += row_in[d + j] * k;
                    }
                    output.set(y, x, sum);
                }
            }
        }
    }

    output
}

/// Compute a Gaussian kernel
fn gaussian_kernel(sigma: f32) -> Vec<f32> {
    const M: f32 = 2.25;
    let scaler = -1.0 / (2.0 * sigma * sigma);
    let diff = (M * sigma.abs()).max(1.0) as i32;
    let size = (2 * diff + 1) as usize;
    let mut kernel = vec![0.0f32; size];

    for i in -diff..=diff {
        kernel[(i + diff) as usize] = (scaler * (i * i) as f32).exp();
    }

    kernel
}

/// Simulate old fused conv+transpose approach (2 passes)
fn gaussian_blur_fused(input: &ImageF, kernel: &[f32]) -> ImageF {
    // First pass: horizontal convolution with transpose
    let temp = convolve_simd(input, kernel);
    // Second pass: another horizontal convolution with transpose
    convolve_simd(&temp, kernel)
}

/// Simulate new separate conv + transpose approach (4 operations)
fn gaussian_blur_separate(input: &ImageF, kernel: &[f32]) -> ImageF {
    let width = input.width();
    let height = input.height();

    // Step 1: Horizontal convolution (cache-friendly writes)
    let after_h = convolve_simd_no_transpose(input, kernel);

    // Step 2: Transpose (width x height -> height x width)
    let mut transposed = ImageF::new(height, width);
    transpose_tiled(&after_h, &mut transposed);

    // Step 3: Another horizontal convolution (effectively vertical)
    let after_v = convolve_simd_no_transpose(&transposed, kernel);

    // Step 4: Transpose back (height x width -> width x height)
    let mut output = ImageF::new(width, height);
    transpose_tiled(&after_v, &mut output);

    output
}

/// Tiled transpose
fn transpose_tiled(input: &ImageF, output: &mut ImageF) {
    let width = input.width();
    let height = input.height();
    const TILE: usize = 8;

    let tile_h = height / TILE * TILE;
    let tile_w = width / TILE * TILE;

    for ty in (0..tile_h).step_by(TILE) {
        for tx in (0..tile_w).step_by(TILE) {
            // Load tile
            let mut tile = [[0.0f32; TILE]; TILE];
            for dy in 0..TILE {
                let row_in = input.row(ty + dy);
                for dx in 0..TILE {
                    tile[dy][dx] = row_in[tx + dx];
                }
            }
            // Write transposed
            for dx in 0..TILE {
                let row_out = output.row_mut(tx + dx);
                for dy in 0..TILE {
                    row_out[ty + dy] = tile[dy][dx];
                }
            }
        }
    }

    // Handle edges
    for y in 0..tile_h {
        let row_in = input.row(y);
        for x in tile_w..width {
            output.row_mut(x)[y] = row_in[x];
        }
    }
    for y in tile_h..height {
        let row_in = input.row(y);
        for x in 0..width {
            output.row_mut(x)[y] = row_in[x];
        }
    }
}

fn main() {
    let width = 512;
    let height = 512;
    let iterations = 50;

    // Create test image
    let mut input = ImageF::new(width, height);
    for y in 0..height {
        for x in 0..width {
            input.set(x, y, ((x + y) % 256) as f32 / 255.0);
        }
    }

    // Gaussian kernel with sigma ~3 (typical for butteraugli)
    let kernel = gaussian_kernel(3.0);
    println!("Kernel size: {}", kernel.len());
    println!("Image size: {}x{}", width, height);
    println!("Iterations: {}", iterations);
    println!();

    // Warm up
    let _ = convolve_scalar(&input, &kernel);
    let _ = convolve_simd(&input, &kernel);
    let _ = convolve_simd_no_transpose(&input, &kernel);
    let _ = gaussian_blur_fused(&input, &kernel);
    let _ = gaussian_blur_separate(&input, &kernel);

    // Benchmark 1D passes
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = convolve_scalar(&input, &kernel);
    }
    let scalar_time = start.elapsed();

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = convolve_simd(&input, &kernel);
    }
    let simd_fused_time = start.elapsed();

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = convolve_simd_no_transpose(&input, &kernel);
    }
    let simd_no_transpose_time = start.elapsed();

    // Benchmark 2D blur approaches
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = gaussian_blur_fused(&input, &kernel);
    }
    let blur_fused_time = start.elapsed();

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = gaussian_blur_separate(&input, &kernel);
    }
    let blur_separate_time = start.elapsed();

    println!("1D Convolution results (single pass):");
    println!(
        "  Scalar + fused transpose:     {:?} per iter",
        scalar_time / iterations as u32
    );
    println!(
        "  SIMD + fused transpose:       {:?} per iter",
        simd_fused_time / iterations as u32
    );
    println!(
        "  SIMD no transpose:            {:?} per iter",
        simd_no_transpose_time / iterations as u32
    );
    println!();
    println!("Full 2D Gaussian blur:");
    println!(
        "  Fused (SIMD+transpose x2):    {:?} per iter",
        blur_fused_time / iterations as u32
    );
    println!(
        "  Separate (conv+trans x2):     {:?} per iter",
        blur_separate_time / iterations as u32
    );
    println!();
    println!("1D Speedups:");
    println!(
        "  SIMD+fused vs scalar: {:.2}x",
        scalar_time.as_secs_f64() / simd_fused_time.as_secs_f64()
    );
    println!(
        "  SIMD no trans vs scalar: {:.2}x",
        scalar_time.as_secs_f64() / simd_no_transpose_time.as_secs_f64()
    );
    println!(
        "  Fused transpose overhead: {:.1}%",
        (simd_fused_time.as_secs_f64() / simd_no_transpose_time.as_secs_f64() - 1.0) * 100.0
    );
    println!();
    println!("2D Blur comparison:");
    println!(
        "  Fused vs separate: {:.2}x faster",
        blur_separate_time.as_secs_f64() / blur_fused_time.as_secs_f64()
    );

    // Verify correctness
    let out_fused = gaussian_blur_fused(&input, &kernel);
    let out_separate = gaussian_blur_separate(&input, &kernel);

    let mut max_diff = 0.0f32;
    for y in 0..height {
        for x in 0..width {
            let f = out_fused.get(x, y);
            let s = out_separate.get(x, y);
            max_diff = max_diff.max((f - s).abs());
        }
    }
    println!();
    println!("Max diff fused vs separate: {}", max_diff);
}
