//! Benchmark comparing precomputed reference vs full computation.
//!
//! Run with: cargo run --release --example precompute_bench

use butteraugli::{compute_butteraugli, ButteraugliParams, ButteraugliReference};
use std::time::Instant;

fn main() {
    let width = 512;
    let height = 512;

    // Create a reference image with some structure
    let reference: Vec<u8> = (0..width * height)
        .flat_map(|i| {
            let x = i % width;
            let y = i / width;
            // Gradient with some variation
            let r = ((x * 255) / width) as u8;
            let g = ((y * 255) / height) as u8;
            let b = (((x + y) * 127) / (width + height)) as u8;
            [r, g, b]
        })
        .collect();

    // Create multiple distorted versions
    let num_distortions = 20;
    let distortions: Vec<Vec<u8>> = (1..=num_distortions)
        .map(|offset| {
            reference
                .iter()
                .map(|&v| v.saturating_add(offset as u8))
                .collect()
        })
        .collect();

    let params = ButteraugliParams::default();

    // Warm up
    let _ = compute_butteraugli(&reference, &distortions[0], width, height, &params);

    // Benchmark full computation
    let start = Instant::now();
    let mut full_scores = Vec::with_capacity(num_distortions);
    for distorted in &distortions {
        let result = compute_butteraugli(&reference, distorted, width, height, &params)
            .expect("valid input");
        full_scores.push(result.score);
    }
    let full_time = start.elapsed();

    // Benchmark precomputed reference
    let precompute_start = Instant::now();
    let precomputed = ButteraugliReference::new(&reference, width, height, params.clone())
        .expect("valid reference");
    let precompute_time = precompute_start.elapsed();

    let compare_start = Instant::now();
    let mut precomputed_scores = Vec::with_capacity(num_distortions);
    for distorted in &distortions {
        let result = precomputed.compare(distorted).expect("valid input");
        precomputed_scores.push(result.score);
    }
    let compare_time = compare_start.elapsed();

    let precomputed_total = precompute_time + compare_time;

    // Verify parity
    for (i, (full, pre)) in full_scores
        .iter()
        .zip(precomputed_scores.iter())
        .enumerate()
    {
        assert!(
            (full - pre).abs() < 1e-10,
            "Score mismatch at {}: full={}, precomputed={}",
            i,
            full,
            pre
        );
    }

    println!("Image size: {}x{}", width, height);
    println!("Number of comparisons: {}", num_distortions);
    println!();
    println!("Full computation:");
    println!("  Total time: {:?}", full_time);
    println!("  Per comparison: {:?}", full_time / num_distortions as u32);
    println!();
    println!("Precomputed reference:");
    println!("  Precompute time: {:?}", precompute_time);
    println!("  Compare time: {:?}", compare_time);
    println!(
        "  Per comparison: {:?}",
        compare_time / num_distortions as u32
    );
    println!("  Total time: {:?}", precomputed_total);
    println!();

    let speedup = full_time.as_secs_f64() / precomputed_total.as_secs_f64();
    let compare_speedup = (full_time.as_secs_f64() / num_distortions as f64)
        / (compare_time.as_secs_f64() / num_distortions as f64);

    println!("Overall speedup: {:.2}x", speedup);
    println!("Per-comparison speedup: {:.2}x", compare_speedup);
    println!();
    println!(
        "Break-even point: {:.1} comparisons",
        precompute_time.as_secs_f64()
            / ((full_time.as_secs_f64() / num_distortions as f64)
                - (compare_time.as_secs_f64() / num_distortions as f64))
    );
}
