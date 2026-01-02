use butteraugli::{compute_butteraugli, ButteraugliParams};
use std::time::Instant;

fn main() {
    let width = 512;
    let height = 512;

    // Create gradient images with small differences
    let mut rgb1 = vec![0u8; width * height * 3];
    let mut rgb2 = vec![0u8; width * height * 3];

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let val = ((x as f32 / width as f32) * 200.0) as u8;
            rgb1[idx] = val;
            rgb1[idx + 1] = val;
            rgb1[idx + 2] = val;

            let val2 = val.saturating_add(((x * y) % 10) as u8);
            rgb2[idx] = val2;
            rgb2[idx + 1] = val2;
            rgb2[idx + 2] = val2;
        }
    }

    let params = ButteraugliParams::default();

    // Warmup
    let _ = compute_butteraugli(&rgb1, &rgb2, width, height, &params);

    // Benchmark
    let iterations = 10;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = compute_butteraugli(&rgb1, &rgb2, width, height, &params);
    }
    let elapsed = start.elapsed();

    println!(
        "512x512 image: {:.2}ms per iteration ({} iterations, total {:.2}s)",
        elapsed.as_secs_f64() * 1000.0 / iterations as f64,
        iterations,
        elapsed.as_secs_f64()
    );
}
