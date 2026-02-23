use butteraugli::{ButteraugliParams, Img, RGB8, butteraugli};
use std::time::Instant;

fn main() {
    let width: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    let height = width;

    // Create gradient images with small differences
    let pixels1: Vec<RGB8> = (0..width * height)
        .map(|i| {
            let x = i % width;
            let val = ((x as f32 / width as f32) * 200.0) as u8;
            RGB8::new(val, val, val)
        })
        .collect();

    let pixels2: Vec<RGB8> = (0..width * height)
        .map(|i| {
            let x = i % width;
            let y = i / width;
            let val = ((x as f32 / width as f32) * 200.0) as u8;
            let val2 = val.saturating_add(((x * y) % 10) as u8);
            RGB8::new(val2, val2, val2)
        })
        .collect();

    let img1 = Img::new(pixels1, width, height);
    let img2 = Img::new(pixels2, width, height);
    let params = ButteraugliParams::default();

    // Warmup
    let _ = butteraugli(img1.as_ref(), img2.as_ref(), &params);

    // Benchmark
    let iterations = 10;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = butteraugli(img1.as_ref(), img2.as_ref(), &params);
    }
    let elapsed = start.elapsed();

    println!(
        "{}x{} image: {:.2}ms per iteration ({} iterations, total {:.2}s)",
        width,
        height,
        elapsed.as_secs_f64() * 1000.0 / iterations as f64,
        iterations,
        elapsed.as_secs_f64()
    );
}
