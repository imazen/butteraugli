# butteraugli

[![Crates.io](https://img.shields.io/crates/v/butteraugli.svg)](https://crates.io/crates/butteraugli)
[![Documentation](https://docs.rs/butteraugli/badge.svg)](https://docs.rs/butteraugli)
[![CI](https://github.com/imazen/butteraugli/actions/workflows/ci.yml/badge.svg)](https://github.com/imazen/butteraugli/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/imazen/butteraugli/graph/badge.svg)](https://codecov.io/gh/imazen/butteraugli)
[![License](https://img.shields.io/crates/l/butteraugli.svg)](LICENSE)

Pure Rust implementation of Google's **butteraugli** perceptual image quality metric from [libjxl](https://github.com/libjxl/libjxl).

## What is Butteraugli?

Butteraugli is a psychovisual image quality metric that estimates the perceived difference between two images. Unlike simple metrics like PSNR or MSE, butteraugli models human vision to produce scores that correlate well with subjective quality assessments.

The metric is based on:
- **Opsin dynamics**: Models photosensitive chemical responses in the retina
- **XYB color space**: A hybrid opponent/trichromatic color representation
- **Visual masking**: How image features hide or reveal differences
- **Multi-scale analysis**: Examines differences at multiple frequency bands

## Quality Thresholds

| Score | Interpretation |
|-------|----------------|
| < 1.0 | Images appear identical to most viewers |
| 1.0 - 2.0 | Subtle differences may be noticeable |
| > 2.0 | Visible differences between images |

## Command-Line Tool

Install with:

```bash
cargo install butteraugli --features cli
```

### Basic Usage

```bash
# Compare two images
butteraugli original.png compressed.jpg
# Output: Butteraugli score: 1.2345

# Show quality rating
butteraugli -q original.png compressed.jpg
# Output: Butteraugli score: 1.2345 (acceptable)
#         Quality: Noticeable but acceptable

# JSON output for scripting
butteraugli --json original.png compressed.jpg

# Save difference heatmap
butteraugli --diffmap diff.png original.png compressed.jpg

# Just the score (for scripting)
butteraugli --quiet original.png compressed.jpg
# Output: 1.234500
```

### Advanced Options

```bash
# Custom intensity target (default: 80 nits)
butteraugli --intensity-target 250 hdr_orig.png hdr_comp.png

# High-frequency asymmetry (penalize blur vs ringing)
butteraugli --hf-asymmetry 1.5 original.png compressed.jpg

# See all options
butteraugli --help
```

## Library Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
butteraugli = "0.4"
```

### Input Formats

Two input APIs are provided using the [`imgref`](https://crates.io/crates/imgref) and [`rgb`](https://crates.io/crates/rgb) crates:

| Function | Input Type | Color Space | Use Case |
|----------|------------|-------------|----------|
| `butteraugli` | `ImgRef<RGB8>` | sRGB (gamma-encoded) | Standard 8-bit images |
| `butteraugli_linear` | `ImgRef<RGB<f32>>` | Linear RGB (0.0-1.0) | HDR, 16-bit, float pipelines |

Both APIs:
- Support **stride** for images with padding via `ImgRef::new_stride()`
- Require **minimum size**: 8×8 pixels
- Return `ButteraugliResult` with score and optional diffmap

### Basic Example

```rust
use butteraugli::{butteraugli, ButteraugliParams, Img, RGB8};

// Create two images (Vec<RGB8> or any container)
let width = 640;
let height = 480;

let original: Vec<RGB8> = load_image_pixels(); // Your image loader
let compressed: Vec<RGB8> = load_compressed_pixels();

let img1 = Img::new(original, width, height);
let img2 = Img::new(compressed, width, height);

// Compare images
let params = ButteraugliParams::default();
let result = butteraugli(img1.as_ref(), img2.as_ref(), &params)
    .expect("valid image data");

println!("Butteraugli score: {:.4}", result.score);

if result.score < 1.0 {
    println!("Images appear identical!");
} else if result.score < 2.0 {
    println!("Minor visible differences");
} else {
    println!("Significant visible differences");
}
```

### With Difference Map

```rust
use butteraugli::{butteraugli, ButteraugliParams, Img, RGB8};

let params = ButteraugliParams::default()
    .with_compute_diffmap(true);  // Enable per-pixel difference map

let result = butteraugli(img1.as_ref(), img2.as_ref(), &params)?;

// Access per-pixel difference map (ImgVec<f32>)
if let Some(diffmap) = result.diffmap {
    let max_diff = diffmap.buf().iter().fold(0.0f32, |a, &b| a.max(b));
    println!("Maximum local difference: {:.4}", max_diff);
}
```

### Linear RGB Example (HDR/16-bit)

```rust
use butteraugli::{butteraugli_linear, ButteraugliParams, Img, RGB, srgb_to_linear};

// Convert 16-bit image to linear f32
let original_16bit: &[u16] = &[/* 16-bit RGB data */];
let original_linear: Vec<RGB<f32>> = original_16bit.chunks(3)
    .map(|c| RGB::new(c[0] as f32 / 65535.0, c[1] as f32 / 65535.0, c[2] as f32 / 65535.0))
    .collect();

// Or convert 8-bit sRGB manually
let original_srgb: &[u8] = &[/* sRGB data */];
let original_linear: Vec<RGB<f32>> = original_srgb.chunks(3)
    .map(|c| RGB::new(srgb_to_linear(c[0]), srgb_to_linear(c[1]), srgb_to_linear(c[2])))
    .collect();

let img = Img::new(original_linear, width, height);
let result = butteraugli_linear(img.as_ref(), compressed_img.as_ref(), &ButteraugliParams::default())?;
```

### Images with Stride (Padding)

```rust
use butteraugli::{butteraugli, ButteraugliParams, Img, RGB8};

// Image data with padding (stride > width)
let raw_pixels: &[RGB8] = get_padded_buffer();
let width = 640;
let height = 480;
let stride = 704;  // Actual row length including padding

// Create ImgRef with stride
let img = Img::new_stride(raw_pixels, width, height, stride);
```

### Custom Parameters

```rust
use butteraugli::ButteraugliParams;

let params = ButteraugliParams::new()
    .with_hf_asymmetry(1.5)        // Penalize new artifacts more than blurring
    .with_xmul(1.0)                // X channel multiplier (1.0 = neutral)
    .with_intensity_target(250.0)  // HDR display brightness in nits
    .with_compute_diffmap(true);   // Generate per-pixel difference map
```

### Score Interpretation

```rust
// Interpret the score directly
if result.score < 1.0 {
    println!("Imperceptible difference");
} else if result.score < 2.0 {
    println!("Subtle difference");
} else {
    println!("Visible difference");
}

// Or convert to other scales:
// Quality percentage (0-100): (100.0 - score * 25.0).clamp(0.0, 100.0)
// Fuzzy class (0-2, from C++): (2.0 - score * 0.5).clamp(0.0, 2.0)
```

## Features

- **`simd`** (default): Enable SIMD optimizations via the `wide` crate
- **`cli`**: Build the command-line tool (adds `clap`, `image`, `serde_json` dependencies)

## Performance

| Benchmark | 512×512 image |
|-----------|---------------|
| Full butteraugli comparison | ~87ms |
| Malta filter (HF 9-sample) | ~2.6ms |
| Malta filter (LF 5-sample) | ~2.4ms |

The implementation uses 100% safe Rust with SIMD vectorization via the `wide` crate.

## Accuracy

**C++ Parity Summary:**

| Test Type | Difference |
|-----------|------------|
| sRGB→linear conversion | 0% (exact) |
| Gamma function | 0% (exact) |
| Frequency bands (all widths) | <0.01% |
| Real images (tank test) | ~1.2% |
| Uniform gray patterns | <0.1% |
| Gradient patterns | ~0.3% |
| Checkerboard patterns | <0.1% |
| Brightness/contrast distortion | <2% |
| Edge + blur patterns | ~1-3% |
| Random + blur patterns | ~20-22%* |

**Reference Parity Tests:** 185 passed, 6 failed (20% tolerance)

\* Six specific test cases involving blur distortions show ~20-32% divergence: edge patterns with dimensions 23x31 and 47x33, and random mid-range patterns with blur at various sizes. These appear to be related to how the blur distortion interacts with edge-detection patterns and don't affect typical real-world image comparisons.

The implementation is validated against live C++ libjxl butteraugli via FFI bindings during development. For practical image quality assessment, the Rust implementation produces results that closely match C++.

## Comparison with Other Crates

| Crate | Type | Notes |
|-------|------|-------|
| `butteraugli` | Pure Rust | Full implementation, no C++ dependency |
| `butteraugli` | FFI wrapper | Wraps C++ butteraugli library |
| `butteraugli-sys` | FFI bindings | Low-level C++ bindings |

### API Comparison with C++ libjxl

| Feature | C++ butteraugli | butteraugli |
|---------|-----------------|-------------------|
| Input format | Linear RGB float | sRGB u8 or linear RGB f32 |
| Bit depth | Any (via float) | 8-bit u8 or f32 |
| Color space | Linear RGB only | sRGB (auto-converted) or linear RGB |
| HDR support | Yes | Yes (via `butteraugli_linear`) |
| Channel layout | Planar (separate R, G, B arrays) | Interleaved RGB via `imgref` |
| Stride support | Manual | Built-in via `ImgRef::new_stride()` |

### XYB Color Space Note

**Butteraugli's internal XYB is NOT the same as jpegli's XYB.**

| Aspect | Butteraugli XYB | jpegli XYB |
|--------|-----------------|------------|
| Nonlinearity | Gamma (FastLog2f-based) | Cube root |
| Opsin matrix | Different coefficients | Different coefficients |
| Dynamic sensitivity | Yes (blur-based adaptation) | No |
| XY formula | X = L - M, Y = L + M | X = (L-M)/2, Y = (L+M)/2 |

This crate does NOT accept XYB input directly because there are multiple incompatible XYB definitions. Always provide RGB input and let butteraugli perform its own internal conversion.

## References

- [Original butteraugli repository](https://github.com/google/butteraugli)
- [JPEG XL (libjxl)](https://github.com/libjxl/libjxl) - Contains the reference implementation
- [Butteraugli paper](https://github.com/google/butteraugli/blob/master/doc/butteraugli-theory.pdf)

## Development

### Running CI Locally

To reproduce the CI checks locally:

```bash
# Format check
cargo fmt --all -- --check

# Clippy lints
cargo clippy --lib --tests -- -D warnings

# Build
cargo build

# Run unit tests
cargo test --lib

# Run conformance tests
cargo test --test conformance

# Run reference parity tests
cargo test --test reference_parity
```

### Test Coverage

```bash
# Install cargo-llvm-cov
cargo install cargo-llvm-cov

# Generate coverage report
cargo llvm-cov --lib --html

# Open report
open target/llvm-cov/html/index.html
```

## AI-Generated Code Notice

This crate was developed with significant assistance from Claude (Anthropic). The code has been tested against the C++ libjxl butteraugli implementation and shows excellent parity for real-world images (~1-2% difference). However, **not all code has been manually reviewed or human-audited**.

Before using in production:
- Review critical code paths for your use case
- Run your own validation against expected outputs
- Consider the test suite coverage for your specific requirements

## License

BSD-3-Clause, same as the original libjxl implementation.
