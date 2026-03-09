use butteraugli::{ButteraugliParams, ButteraugliReference, Img, RGB8, butteraugli};
use std::time::Instant;

fn srgb_to_linear(v: u8) -> f32 {
    let s = v as f32 / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

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

    // Benchmark one-shot butteraugli
    let iterations = 10;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = butteraugli(img1.as_ref(), img2.as_ref(), &params);
    }
    let elapsed = start.elapsed();

    println!(
        "butteraugli one-shot {}x{}: {:.2}ms per iteration ({} iters)",
        width,
        height,
        elapsed.as_secs_f64() * 1000.0 / iterations as f64,
        iterations,
    );

    // Benchmark ButteraugliReference path (jxl-encoder style)
    // Convert to planar linear RGB
    let mut r1 = vec![0.0f32; width * height];
    let mut g1 = vec![0.0f32; width * height];
    let mut b1 = vec![0.0f32; width * height];
    let mut r2 = vec![0.0f32; width * height];
    let mut g2 = vec![0.0f32; width * height];
    let mut b2 = vec![0.0f32; width * height];

    for (i, px) in img1.buf().iter().enumerate() {
        r1[i] = srgb_to_linear(px.r);
        g1[i] = srgb_to_linear(px.g);
        b1[i] = srgb_to_linear(px.b);
    }
    for (i, px) in img2.buf().iter().enumerate() {
        r2[i] = srgb_to_linear(px.r);
        g2[i] = srgb_to_linear(px.g);
        b2[i] = srgb_to_linear(px.b);
    }

    // Time reference creation
    let start = Instant::now();
    let reference =
        ButteraugliReference::new_linear_planar(&r1, &g1, &b1, width, height, width, params)
            .unwrap();
    let ref_time = start.elapsed();

    // Warmup
    let _ = reference
        .compare_linear_planar(&r2, &g2, &b2, width)
        .unwrap();

    // Benchmark compare_linear_planar
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = reference
            .compare_linear_planar(&r2, &g2, &b2, width)
            .unwrap();
    }
    let elapsed = start.elapsed();

    println!(
        "reference creation {}x{}: {:.2}ms",
        width,
        height,
        ref_time.as_secs_f64() * 1000.0,
    );
    println!(
        "compare_linear_planar {}x{}: {:.2}ms per iteration ({} iters)",
        width,
        height,
        elapsed.as_secs_f64() * 1000.0 / iterations as f64,
        iterations,
    );
}
