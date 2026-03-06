use butteraugli::{butteraugli, ButteraugliParams, Img, RGB8};
use std::time::Instant;

fn main() {
    // Generate test images
    let width = 512;
    let height = 512;
    
    let img1: Vec<RGB8> = (0..width * height)
        .map(|i| {
            let x = (i % width) as u8;
            let y = (i / width) as u8;
            RGB8::new(x, y, ((x as u16 + y as u16) / 2) as u8)
        })
        .collect();
    
    let img2: Vec<RGB8> = (0..width * height)
        .map(|i| {
            let x = (i % width) as u8;
            let y = (i / width) as u8;
            RGB8::new(x.wrapping_add(5), y.wrapping_add(3), ((x as u16 + y as u16) / 2) as u8)
        })
        .collect();
    
    let i1 = Img::new(img1, width, height);
    let i2 = Img::new(img2, width, height);
    let params = ButteraugliParams::default();
    
    // Warmup
    for _ in 0..3 {
        let _ = butteraugli(i1.as_ref(), i2.as_ref(), &params);
    }
    
    // Benchmark
    let iterations = 20;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = butteraugli(i1.as_ref(), i2.as_ref(), &params);
    }
    let elapsed = start.elapsed();
    
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    println!("512x512 butteraugli: {:.2}ms avg over {} iterations", avg_ms, iterations);
}
