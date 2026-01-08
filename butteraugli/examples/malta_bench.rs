//! Benchmark for Malta filter performance.
//!
//! Run with: `cargo run --release --example malta_bench`

use butteraugli::image::ImageF;
use butteraugli::malta::malta_diff_map;
use std::time::Instant;

fn main() {
    let width = 512;
    let height = 512;
    let iterations = 50;

    // Create test images with varied data
    let mut lum0 = ImageF::new(width, height);
    let mut lum1 = ImageF::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let base = ((x + y) % 256) as f32 / 255.0;
            lum0.set(x, y, base);
            // Add some difference in lum1
            lum1.set(x, y, base * 0.95 + 0.02);
        }
    }

    println!("Malta filter benchmark");
    println!("Image size: {}x{}", width, height);
    println!("Iterations: {}", iterations);
    println!();

    // Warm up
    let _ = malta_diff_map(&lum0, &lum1, 1.0, 1.0, 1.0, false);
    let _ = malta_diff_map(&lum0, &lum1, 1.0, 1.0, 1.0, true);

    // Benchmark HF (use_lf = false)
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = malta_diff_map(&lum0, &lum1, 1.0, 1.0, 1.0, false);
    }
    let hf_time = start.elapsed();

    // Benchmark LF (use_lf = true)
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = malta_diff_map(&lum0, &lum1, 1.0, 1.0, 1.0, true);
    }
    let lf_time = start.elapsed();

    println!(
        "Malta HF (9-sample): {:?} per iter",
        hf_time / iterations as u32
    );
    println!(
        "Malta LF (5-sample): {:?} per iter",
        lf_time / iterations as u32
    );
}
