//! Shared image generation and distortion functions for butteraugli tests.
//!
//! These produce deterministic synthetic images using an LCG PRNG,
//! ensuring identical test inputs across all platforms.

use butteraugli::RGB8;

/// Convert RGB byte slice to Vec<RGB8>
pub fn rgb_bytes_to_pixels(rgb: &[u8]) -> Vec<RGB8> {
    rgb.chunks_exact(3)
        .map(|c| RGB8::new(c[0], c[1], c[2]))
        .collect()
}

/// Parse "WxH" dimensions from the last segment of a test case name.
///
/// E.g. "uniform_gray_128_shift_10_8x8" → Some((8, 8))
///      "gradient_diag_shift_20_23x31" → Some((23, 31))
pub fn parse_dimensions(name: &str) -> Option<(usize, usize)> {
    let last = name.rsplit('_').next()?;
    let (w_str, h_str) = last.split_once('x')?;
    let w: usize = w_str.parse().ok()?;
    let h: usize = h_str.parse().ok()?;
    Some((w, h))
}

// ============================================================================
// LCG PRNG
// ============================================================================

/// LCG pseudo-random number generator (deterministic)
pub struct Lcg {
    state: u64,
}

impl Lcg {
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u8(&mut self) -> u8 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.state >> 33) & 0xFF) as u8
    }

    pub fn next_u8_range(&mut self, min: u8, max: u8) -> u8 {
        let range = (max - min) as u64 + 1;
        let val = self.next_u8() as u64;
        (min as u64 + (val * range / 256)) as u8
    }
}

// ============================================================================
// Image Generation Functions
// ============================================================================

/// Generate uniform color image
pub fn gen_uniform(width: usize, height: usize, r: u8, g: u8, b: u8) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for _ in 0..width * height {
        data.push(r);
        data.push(g);
        data.push(b);
    }
    data
}

/// Generate horizontal gradient (grayscale)
pub fn gen_gradient_h(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for _y in 0..height {
        for x in 0..width {
            let val = if width > 1 {
                (x * 255 / (width - 1)) as u8
            } else {
                128
            };
            data.push(val);
            data.push(val);
            data.push(val);
        }
    }
    data
}

/// Generate vertical gradient (grayscale)
pub fn gen_gradient_v(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        let val = if height > 1 {
            (y * 255 / (height - 1)) as u8
        } else {
            128
        };
        for _x in 0..width {
            data.push(val);
            data.push(val);
            data.push(val);
        }
    }
    data
}

/// Generate diagonal gradient (grayscale)
pub fn gen_gradient_diag(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    let max_dist = width + height - 2;
    for y in 0..height {
        for x in 0..width {
            let val = if max_dist > 0 {
                ((x + y) * 255 / max_dist) as u8
            } else {
                128
            };
            data.push(val);
            data.push(val);
            data.push(val);
        }
    }
    data
}

/// Generate RGB color gradient
pub fn gen_color_gradient(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r = if width > 1 {
                (x * 255 / (width - 1)) as u8
            } else {
                128
            };
            let g = if height > 1 {
                (y * 255 / (height - 1)) as u8
            } else {
                128
            };
            let b = 128;
            data.push(r);
            data.push(g);
            data.push(b);
        }
    }
    data
}

/// Generate checkerboard pattern
pub fn gen_checkerboard(width: usize, height: usize, block_size: usize, lo: u8, hi: u8) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let checker = ((x / block_size) + (y / block_size)) % 2 == 0;
            let val = if checker { hi } else { lo };
            data.push(val);
            data.push(val);
            data.push(val);
        }
    }
    data
}

/// Generate inverse checkerboard
pub fn gen_checkerboard_inv(
    width: usize,
    height: usize,
    block_size: usize,
    lo: u8,
    hi: u8,
) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let checker = ((x / block_size) + (y / block_size)) % 2 == 1;
            let val = if checker { hi } else { lo };
            data.push(val);
            data.push(val);
            data.push(val);
        }
    }
    data
}

/// Generate horizontal stripes
pub fn gen_stripes_h(width: usize, height: usize, stripe_height: usize, lo: u8, hi: u8) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        let val = if (y / stripe_height) % 2 == 0 { hi } else { lo };
        for _x in 0..width {
            data.push(val);
            data.push(val);
            data.push(val);
        }
    }
    data
}

/// Generate vertical stripes
pub fn gen_stripes_v(width: usize, height: usize, stripe_width: usize, lo: u8, hi: u8) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for _y in 0..height {
        for x in 0..width {
            let val = if (x / stripe_width) % 2 == 0 { hi } else { lo };
            data.push(val);
            data.push(val);
            data.push(val);
        }
    }
    data
}

/// Generate seeded random image
pub fn gen_random(width: usize, height: usize, seed: u64) -> Vec<u8> {
    let mut rng = Lcg::new(seed);
    let mut data = Vec::with_capacity(width * height * 3);
    for _ in 0..width * height * 3 {
        data.push(rng.next_u8());
    }
    data
}

/// Generate seeded random image with limited range (avoids extremes)
pub fn gen_random_midrange(width: usize, height: usize, seed: u64) -> Vec<u8> {
    let mut rng = Lcg::new(seed);
    let mut data = Vec::with_capacity(width * height * 3);
    for _ in 0..width * height * 3 {
        data.push(rng.next_u8_range(32, 224));
    }
    data
}

/// Generate smooth sine wave pattern
pub fn gen_sine_wave(width: usize, height: usize, freq_x: f32, freq_y: f32) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let fx = (x as f32 * freq_x * std::f32::consts::TAU / width as f32).sin();
            let fy = (y as f32 * freq_y * std::f32::consts::TAU / height as f32).sin();
            let val = ((fx + fy + 2.0) / 4.0 * 255.0) as u8;
            data.push(val);
            data.push(val);
            data.push(val);
        }
    }
    data
}

/// Generate radial gradient (distance from center)
pub fn gen_radial(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    let cx = width as f32 / 2.0;
    let cy = height as f32 / 2.0;
    let max_dist = (cx * cx + cy * cy).sqrt();
    for y in 0..height {
        for x in 0..width {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            let val = ((1.0 - dist / max_dist) * 255.0).clamp(0.0, 255.0) as u8;
            data.push(val);
            data.push(val);
            data.push(val);
        }
    }
    data
}

/// Generate edge pattern (sharp transition at center, vertical)
pub fn gen_edge_v(width: usize, height: usize, lo: u8, hi: u8) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    let mid = width / 2;
    for _y in 0..height {
        for x in 0..width {
            let val = if x < mid { lo } else { hi };
            data.push(val);
            data.push(val);
            data.push(val);
        }
    }
    data
}

/// Generate edge pattern (horizontal)
pub fn gen_edge_h(width: usize, height: usize, lo: u8, hi: u8) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    let mid = height / 2;
    for y in 0..height {
        let val = if y < mid { lo } else { hi };
        for _x in 0..width {
            data.push(val);
            data.push(val);
            data.push(val);
        }
    }
    data
}

// ============================================================================
// Distortion Functions
// ============================================================================

/// Add uniform brightness shift
pub fn distort_brightness(img: &[u8], delta: i16) -> Vec<u8> {
    img.iter()
        .map(|&v| (v as i16 + delta).clamp(0, 255) as u8)
        .collect()
}

/// Add per-pixel noise with fixed seed
pub fn distort_noise(img: &[u8], seed: u64, amplitude: u8) -> Vec<u8> {
    let mut rng = Lcg::new(seed);
    img.iter()
        .map(|&v| {
            let noise = rng.next_u8() as i16 - 128;
            let scaled = noise * amplitude as i16 / 128;
            (v as i16 + scaled).clamp(0, 255) as u8
        })
        .collect()
}

/// Contrast adjustment
pub fn distort_contrast(img: &[u8], factor: f32) -> Vec<u8> {
    img.iter()
        .map(|&v| {
            let centered = v as f32 - 128.0;
            let adjusted = centered * factor + 128.0;
            adjusted.clamp(0.0, 255.0) as u8
        })
        .collect()
}

/// Gamma adjustment
pub fn distort_gamma(img: &[u8], gamma: f32) -> Vec<u8> {
    img.iter()
        .map(|&v| {
            let normalized = v as f32 / 255.0;
            let adjusted = normalized.powf(gamma);
            (adjusted * 255.0).clamp(0.0, 255.0) as u8
        })
        .collect()
}

/// Simple box blur (3x3)
pub fn distort_blur(img: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut out = vec![0u8; img.len()];
    for y in 0..height {
        for x in 0..width {
            for c in 0..3 {
                let mut sum = 0u32;
                let mut count = 0u32;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                            let idx = (ny as usize * width + nx as usize) * 3 + c;
                            sum += img[idx] as u32;
                            count += 1;
                        }
                    }
                }
                let idx = (y * width + x) * 3 + c;
                out[idx] = (sum / count) as u8;
            }
        }
    }
    out
}

/// Channel swap (R <-> B)
pub fn distort_channel_swap_rb(img: &[u8]) -> Vec<u8> {
    let mut out = img.to_vec();
    for chunk in out.chunks_mut(3) {
        chunk.swap(0, 2);
    }
    out
}

/// Hue shift (rotate RGB)
pub fn distort_hue_shift(img: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(img.len());
    for chunk in img.chunks(3) {
        out.push(chunk[1]); // G -> R
        out.push(chunk[2]); // B -> G
        out.push(chunk[0]); // R -> B
    }
    out
}

/// Quantize to fewer levels
pub fn distort_quantize(img: &[u8], levels: u8) -> Vec<u8> {
    let step = 256 / levels as u16;
    img.iter()
        .map(|&v| {
            let bucket = v as u16 / step;
            (bucket * step + step / 2).min(255) as u8
        })
        .collect()
}

// ============================================================================
// Image pair generation from test case name
// ============================================================================

/// Generate an image pair from a test case name.
///
/// Returns (image_a, image_b) or None if the name pattern isn't recognized.
pub fn generate_image_pair(name: &str, width: usize, height: usize) -> Option<(Vec<u8>, Vec<u8>)> {
    let parts: Vec<&str> = name.split('_').collect();

    // Uniform patterns
    if name.starts_with("uniform_gray_128_shift_") {
        let shift: i16 = parts
            .iter()
            .position(|&p| p == "shift")
            .and_then(|i| parts.get(i + 1))
            .and_then(|s| s.parse().ok())?;
        let a = gen_uniform(width, height, 128, 128, 128);
        let b = gen_uniform(
            width,
            height,
            (128 + shift) as u8,
            (128 + shift) as u8,
            (128 + shift) as u8,
        );
        return Some((a, b));
    }

    if name.starts_with("uniform_red_shift_") {
        let shift: i16 = parts
            .iter()
            .position(|&p| p == "shift")
            .and_then(|i| parts.get(i + 1))
            .and_then(|s| s.parse().ok())?;
        let a = gen_uniform(width, height, 128, 64, 64);
        let b = gen_uniform(width, height, (128 + shift) as u8, 64, 64);
        return Some((a, b));
    }

    if name.starts_with("uniform_green_shift_") {
        let shift: i16 = parts
            .iter()
            .position(|&p| p == "shift")
            .and_then(|i| parts.get(i + 1))
            .and_then(|s| s.parse().ok())?;
        let a = gen_uniform(width, height, 64, 128, 64);
        let b = gen_uniform(width, height, 64, (128 + shift) as u8, 64);
        return Some((a, b));
    }

    if name.starts_with("uniform_blue_shift_") {
        let shift: i16 = parts
            .iter()
            .position(|&p| p == "shift")
            .and_then(|i| parts.get(i + 1))
            .and_then(|s| s.parse().ok())?;
        let a = gen_uniform(width, height, 64, 64, 128);
        let b = gen_uniform(width, height, 64, 64, (128 + shift) as u8);
        return Some((a, b));
    }

    // Gradient patterns
    if name.starts_with("gradient_h_shift_") {
        let shift: i16 = parts
            .iter()
            .position(|&p| p == "shift")
            .and_then(|i| parts.get(i + 1))
            .and_then(|s| s.parse().ok())?;
        let a = gen_gradient_h(width, height);
        let b = distort_brightness(&a, shift);
        return Some((a, b));
    }

    if name.starts_with("gradient_v_shift_") {
        let shift: i16 = parts
            .iter()
            .position(|&p| p == "shift")
            .and_then(|i| parts.get(i + 1))
            .and_then(|s| s.parse().ok())?;
        let a = gen_gradient_v(width, height);
        let b = distort_brightness(&a, shift);
        return Some((a, b));
    }

    if name.starts_with("gradient_diag_shift_") {
        let shift: i16 = parts
            .iter()
            .position(|&p| p == "shift")
            .and_then(|i| parts.get(i + 1))
            .and_then(|s| s.parse().ok())?;
        let a = gen_gradient_diag(width, height);
        let b = distort_brightness(&a, shift);
        return Some((a, b));
    }

    if name.starts_with("color_gradient_shift_") {
        let shift: i16 = parts
            .iter()
            .position(|&p| p == "shift")
            .and_then(|i| parts.get(i + 1))
            .and_then(|s| s.parse().ok())?;
        let a = gen_color_gradient(width, height);
        let b = distort_brightness(&a, shift);
        return Some((a, b));
    }

    // Checkerboard patterns
    if name.starts_with("checkerboard_vs_inverse_") {
        let block_size: usize = parts
            .iter()
            .find(|p| p.ends_with("px"))
            .and_then(|p| p.strip_suffix("px"))
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let a = gen_checkerboard(width, height, block_size, 50, 200);
        let b = gen_checkerboard_inv(width, height, block_size, 50, 200);
        return Some((a, b));
    }

    if name.starts_with("checkerboard_shift_") {
        let shift: i16 = parts
            .iter()
            .position(|&p| p == "shift")
            .and_then(|i| parts.get(i + 1))
            .and_then(|s| s.parse().ok())?;
        let a = gen_checkerboard(width, height, 2, 50, 200);
        let b = distort_brightness(&a, shift);
        return Some((a, b));
    }

    // Stripe patterns
    if name.starts_with("stripes_h_") {
        let shift: i16 = parts
            .iter()
            .position(|&p| p == "shift")
            .and_then(|i| parts.get(i + 1))
            .and_then(|s| s.parse().ok())?;
        let a = gen_stripes_h(width, height, 2, 50, 200);
        let b = distort_brightness(&a, shift);
        return Some((a, b));
    }

    if name.starts_with("stripes_v_") {
        let shift: i16 = parts
            .iter()
            .position(|&p| p == "shift")
            .and_then(|i| parts.get(i + 1))
            .and_then(|s| s.parse().ok())?;
        let a = gen_stripes_v(width, height, 2, 50, 200);
        let b = distort_brightness(&a, shift);
        return Some((a, b));
    }

    // Sine wave patterns
    if name.starts_with("sine_") {
        let freq: f32 = if name.contains("1x1") {
            1.0
        } else if name.contains("2x2") {
            2.0
        } else if name.contains("4x4") {
            4.0
        } else {
            return None;
        };
        let a = gen_sine_wave(width, height, freq, freq);
        let b = distort_brightness(&a, 10);
        return Some((a, b));
    }

    // Radial patterns
    if name.starts_with("radial_shift_") {
        let shift: i16 = parts
            .iter()
            .position(|&p| p == "shift")
            .and_then(|i| parts.get(i + 1))
            .and_then(|s| s.parse().ok())?;
        let a = gen_radial(width, height);
        let b = distort_brightness(&a, shift);
        return Some((a, b));
    }

    // Edge patterns
    if name.starts_with("edge_v_shift_") {
        let shift: i16 = parts
            .iter()
            .position(|&p| p == "shift")
            .and_then(|i| parts.get(i + 1))
            .and_then(|s| s.parse().ok())?;
        let a = gen_edge_v(width, height, 50, 200);
        let b = distort_brightness(&a, shift);
        return Some((a, b));
    }

    if name.starts_with("edge_h_shift_") {
        let shift: i16 = parts
            .iter()
            .position(|&p| p == "shift")
            .and_then(|i| parts.get(i + 1))
            .and_then(|s| s.parse().ok())?;
        let a = gen_edge_h(width, height, 50, 200);
        let b = distort_brightness(&a, shift);
        return Some((a, b));
    }

    if name.starts_with("edge_v_vs_blur_") {
        let a = gen_edge_v(width, height, 50, 200);
        let b = distort_blur(&a, width, height);
        return Some((a, b));
    }

    // Random patterns with brightness shift
    if name.starts_with("random_seed") && name.contains("_shift_") {
        let seed_idx: usize = parts
            .iter()
            .find(|p| p.starts_with("seed"))
            .and_then(|p| p.strip_prefix("seed"))
            .and_then(|s| s.parse().ok())?;
        let seeds: &[u64] = &[
            0x12345678_9ABCDEF0,
            0xDEADBEEF_CAFEBABE,
            0x0BADC0DE_FEEDFACE,
            0x13371337_42424242,
            0xAAAAAAAA_55555555,
        ];
        let seed = seeds.get(seed_idx)?;
        let a = gen_random(width, height, *seed);
        let b = distort_brightness(&a, 10);
        return Some((a, b));
    }

    // Random patterns with noise
    if name.starts_with("random_seed") && name.contains("_noise_") {
        let seed_idx: usize = parts
            .iter()
            .find(|p| p.starts_with("seed"))
            .and_then(|p| p.strip_prefix("seed"))
            .and_then(|s| s.parse().ok())?;
        let seeds: &[u64] = &[
            0x12345678_9ABCDEF0,
            0xDEADBEEF_CAFEBABE,
            0x0BADC0DE_FEEDFACE,
            0x13371337_42424242,
            0xAAAAAAAA_55555555,
        ];
        let seed = seeds.get(seed_idx)?;
        let noise_seed = seed.wrapping_add(1);
        let a = gen_random(width, height, *seed);
        let b = distort_noise(&a, noise_seed, 20);
        return Some((a, b));
    }

    // Random midrange patterns
    let mid_seed = 0xFEDCBA98_76543210u64;

    if name.starts_with("random_mid_contrast_") {
        let a = gen_random_midrange(width, height, mid_seed);
        let b = distort_contrast(&a, 1.2);
        return Some((a, b));
    }

    if name.starts_with("random_mid_gamma_") {
        let a = gen_random_midrange(width, height, mid_seed);
        let b = distort_gamma(&a, 0.9);
        return Some((a, b));
    }

    if name.starts_with("random_mid_blur_") {
        let a = gen_random_midrange(width, height, mid_seed);
        let b = distort_blur(&a, width, height);
        return Some((a, b));
    }

    if name.starts_with("random_mid_quantize_") {
        let a = gen_random_midrange(width, height, mid_seed);
        let b = distort_quantize(&a, 32);
        return Some((a, b));
    }

    // Color distortions
    if name.starts_with("color_grad_channel_swap_") {
        let a = gen_color_gradient(width, height);
        let b = distort_channel_swap_rb(&a);
        return Some((a, b));
    }

    if name.starts_with("color_grad_hue_shift_") {
        let a = gen_color_gradient(width, height);
        let b = distort_hue_shift(&a);
        return Some((a, b));
    }

    let random_color_seed = 0x1234567890ABCDEFu64;

    if name.starts_with("random_color_channel_swap_") {
        let a = gen_random(width, height, random_color_seed);
        let b = distort_channel_swap_rb(&a);
        return Some((a, b));
    }

    if name.starts_with("random_color_hue_shift_") {
        let a = gen_random(width, height, random_color_seed);
        let b = distort_hue_shift(&a);
        return Some((a, b));
    }

    None
}
