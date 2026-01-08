//! Benchmark comparing aligned vs unaligned image storage.
//!
//! Run with: cargo run --release --example aligned_bench

use butteraugli::image::ImageF;
use butteraugli::image_aligned::AlignedImageF;
use std::time::Instant;
use wide::f32x8;

fn main() {
    let width = 512;
    let height = 512;
    let iterations = 100;

    println!("Image size: {}x{}", width, height);
    println!("Iterations: {}", iterations);
    println!();

    // Benchmark creation
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = ImageF::new(width, height);
    }
    let unaligned_create = start.elapsed();

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = AlignedImageF::new(width, height);
    }
    let aligned_create = start.elapsed();

    println!("Creation:");
    println!("  ImageF:        {:?}", unaligned_create / iterations as u32);
    println!("  AlignedImageF: {:?}", aligned_create / iterations as u32);
    println!();

    // Benchmark row-by-row fill
    let mut img_u = ImageF::new(width, height);
    let mut img_a = AlignedImageF::new(width, height);

    let start = Instant::now();
    for _ in 0..iterations {
        for y in 0..height {
            let row = img_u.row_mut(y);
            for x in 0..width {
                row[x] = 1.0;
            }
        }
    }
    let unaligned_fill = start.elapsed();

    let start = Instant::now();
    for _ in 0..iterations {
        for y in 0..height {
            let row = img_a.row_mut(y);
            for x in 0..width {
                row[x] = 1.0;
            }
        }
    }
    let aligned_fill_scalar = start.elapsed();

    // Benchmark SIMD fill (aligned only)
    let start = Instant::now();
    let val = f32x8::splat(1.0);
    for _ in 0..iterations {
        for y in 0..height {
            for chunk in img_a.row_simd_mut(y) {
                *chunk = val;
            }
        }
    }
    let aligned_fill_simd = start.elapsed();

    println!("Fill (scalar per-element):");
    println!("  ImageF:        {:?}", unaligned_fill / iterations as u32);
    println!("  AlignedImageF: {:?}", aligned_fill_scalar / iterations as u32);
    println!("  AlignedImageF (SIMD): {:?}", aligned_fill_simd / iterations as u32);
    println!();

    // Benchmark copy
    let src_u = ImageF::filled(width, height, 5.0);
    let src_a = AlignedImageF::filled(width, height, 5.0);
    let mut dst_u = ImageF::new(width, height);
    let mut dst_a = AlignedImageF::new(width, height);

    let start = Instant::now();
    for _ in 0..iterations {
        dst_u.copy_from(&src_u);
    }
    let unaligned_copy = start.elapsed();

    let start = Instant::now();
    for _ in 0..iterations {
        dst_a.copy_from(&src_a);
    }
    let aligned_copy_scalar = start.elapsed();

    let start = Instant::now();
    for _ in 0..iterations {
        dst_a.copy_from_simd(&src_a);
    }
    let aligned_copy_simd = start.elapsed();

    println!("Copy:");
    println!("  ImageF:        {:?}", unaligned_copy / iterations as u32);
    println!("  AlignedImageF: {:?}", aligned_copy_scalar / iterations as u32);
    println!("  AlignedImageF (SIMD): {:?}", aligned_copy_simd / iterations as u32);
    println!();

    // Benchmark dot product (sum of products)
    let a = ImageF::filled(width, height, 2.0);
    let b = ImageF::filled(width, height, 3.0);
    let aa = AlignedImageF::filled(width, height, 2.0);
    let bb = AlignedImageF::filled(width, height, 3.0);

    let start = Instant::now();
    let mut sum_u = 0.0f32;
    for _ in 0..iterations {
        sum_u = 0.0;
        for y in 0..height {
            let ra = a.row(y);
            let rb = b.row(y);
            for x in 0..width {
                sum_u += ra[x] * rb[x];
            }
        }
    }
    let unaligned_dot = start.elapsed();

    let start = Instant::now();
    let mut sum_a = 0.0f32;
    for _ in 0..iterations {
        sum_a = 0.0;
        for y in 0..height {
            let ra = aa.row(y);
            let rb = bb.row(y);
            for x in 0..width {
                sum_a += ra[x] * rb[x];
            }
        }
    }
    let aligned_dot_scalar = start.elapsed();

    let start = Instant::now();
    let mut sum_simd = 0.0f32;
    for _ in 0..iterations {
        let mut acc = f32x8::splat(0.0);
        for y in 0..height {
            let ra = aa.row_simd(y);
            let rb = bb.row_simd(y);
            for i in 0..ra.len() {
                acc += ra[i] * rb[i];
            }
        }
        // Horizontal sum
        let arr: [f32; 8] = acc.into();
        sum_simd = arr.iter().sum();
    }
    let aligned_dot_simd = start.elapsed();

    println!("Dot product:");
    println!("  ImageF:        {:?} (sum={})", unaligned_dot / iterations as u32, sum_u);
    println!("  AlignedImageF: {:?} (sum={})", aligned_dot_scalar / iterations as u32, sum_a);
    println!("  AlignedImageF (SIMD): {:?} (sum={})", aligned_dot_simd / iterations as u32, sum_simd);
    println!();

    // Check alignment
    println!("Alignment check:");
    let img = AlignedImageF::new(100, 50);
    let row = img.row_simd(0);
    let ptr = row.as_ptr() as usize;
    println!("  AlignedImageF row ptr mod 32: {}", ptr % 32);
    println!("  AlignedImageF row ptr mod 64: {}", ptr % 64);

    let img2 = ImageF::new(100, 50);
    let ptr2 = img2.row(0).as_ptr() as usize;
    println!("  ImageF row ptr mod 32: {}", ptr2 % 32);
    println!("  ImageF row ptr mod 64: {}", ptr2 % 64);
}
