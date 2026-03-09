use butteraugli::{ButteraugliParams, ButteraugliReference, Img, RGB8};

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

    let mut r1 = vec![0.0f32; width * height];
    let mut g1 = vec![0.0f32; width * height];
    let mut b1 = vec![0.0f32; width * height];
    let mut r2 = vec![0.0f32; width * height];
    let mut g2 = vec![0.0f32; width * height];
    let mut b2 = vec![0.0f32; width * height];

    for (i, px) in Img::new(pixels1, width, height).buf().iter().enumerate() {
        r1[i] = srgb_to_linear(px.r);
        g1[i] = srgb_to_linear(px.g);
        b1[i] = srgb_to_linear(px.b);
    }
    for (i, px) in Img::new(pixels2, width, height).buf().iter().enumerate() {
        r2[i] = srgb_to_linear(px.r);
        g2[i] = srgb_to_linear(px.g);
        b2[i] = srgb_to_linear(px.b);
    }

    let params = ButteraugliParams::default();
    let reference =
        ButteraugliReference::new_linear_planar(&r1, &g1, &b1, width, height, width, params)
            .unwrap();

    // Single compare_linear_planar call for callgrind
    let result = reference
        .compare_linear_planar(&r2, &g2, &b2, width)
        .unwrap();
    println!("score: {:.4}", result.score);
}
