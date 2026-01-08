//! Benchmark comparing scalar vs SIMD convolution.
//!
//! Run with: cargo run --release --example conv_simd_bench

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
                    let vals = f32x8::from(unsafe {
                        *(slice.as_ptr() as *const [f32; 8])
                    });
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
                    let vals = f32x8::from(unsafe {
                        *(slice.as_ptr() as *const [f32; 8])
                    });
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
    let _ = convolve_simd_tiled(&input, &kernel);
    let _ = convolve_simd_no_transpose(&input, &kernel);

    // Benchmark scalar
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = convolve_scalar(&input, &kernel);
    }
    let scalar_time = start.elapsed();

    // Benchmark SIMD (with transpose)
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = convolve_simd(&input, &kernel);
    }
    let simd_time = start.elapsed();

    // Benchmark SIMD no transpose
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = convolve_simd_no_transpose(&input, &kernel);
    }
    let simd_no_transpose_time = start.elapsed();

    // Benchmark tiled
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = convolve_simd_tiled(&input, &kernel);
    }
    let tiled_time = start.elapsed();

    println!("Convolution results:");
    println!("  Scalar + transpose:      {:?} per iter", scalar_time / iterations as u32);
    println!("  SIMD + transpose:        {:?} per iter", simd_time / iterations as u32);
    println!("  SIMD no transpose:       {:?} per iter", simd_no_transpose_time / iterations as u32);
    println!("  Tiled + transpose:       {:?} per iter", tiled_time / iterations as u32);
    println!();
    println!("Speedup SIMD+transpose vs scalar: {:.2}x", scalar_time.as_secs_f64() / simd_time.as_secs_f64());
    println!("Speedup SIMD no transpose vs scalar: {:.2}x", scalar_time.as_secs_f64() / simd_no_transpose_time.as_secs_f64());
    println!("Transpose overhead: {:.1}%", (simd_time.as_secs_f64() / simd_no_transpose_time.as_secs_f64() - 1.0) * 100.0);

    // Verify correctness
    let out_scalar = convolve_scalar(&input, &kernel);
    let out_simd = convolve_simd(&input, &kernel);
    let out_tiled = convolve_simd_tiled(&input, &kernel);

    let mut max_diff_simd = 0.0f32;
    let mut max_diff_tiled = 0.0f32;
    for y in 0..height {
        for x in 0..width {
            let s = out_scalar.get(x, y);
            let v = out_simd.get(x, y);
            let t = out_tiled.get(x, y);
            max_diff_simd = max_diff_simd.max((s - v).abs());
            max_diff_tiled = max_diff_tiled.max((s - t).abs());
        }
    }
    println!();
    println!("Max diff SIMD vs scalar: {}", max_diff_simd);
    println!("Max diff tiled vs scalar: {}", max_diff_tiled);
}
