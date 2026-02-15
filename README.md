# butteraugli

[![Crates.io](https://img.shields.io/crates/v/butteraugli.svg)](https://crates.io/crates/butteraugli)
[![Documentation](https://docs.rs/butteraugli/badge.svg)](https://docs.rs/butteraugli)
[![CI](https://github.com/imazen/butteraugli/actions/workflows/ci.yml/badge.svg)](https://github.com/imazen/butteraugli/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/imazen/butteraugli/graph/badge.svg)](https://codecov.io/gh/imazen/butteraugli)
[![License](https://img.shields.io/crates/l/butteraugli.svg)](LICENSE)

Pure Rust implementation of Google's **butteraugli** perceptual image quality metric from [libjxl](https://github.com/libjxl/libjxl).

## What is Butteraugli?

Butteraugli estimates the perceived difference between two images using a model of human vision. Unlike simple pixel-wise metrics (PSNR, MSE), butteraugli accounts for:

- **Opsin dynamics**: Photosensitive chemical responses in the retina
- **XYB color space**: Hybrid opponent/trichromatic representation
- **Visual masking**: How image features hide or reveal differences
- **Multi-scale analysis**: UHF, HF, MF, LF frequency bands

## Quality Thresholds

| Score | Interpretation |
|-------|----------------|
| < 1.0 | Images appear identical to most viewers |
| 1.0 - 2.0 | Subtle differences may be noticeable |
| > 2.0 | Visible differences between images |

## Command-Line Tool

```bash
cargo install butteraugli-cli
```

### Usage

```bash
# Compare two images
butteraugli original.png compressed.jpg
# Output: Butteraugli score: 1.2345

# Quality rating
butteraugli -q original.png compressed.jpg

# JSON output
butteraugli --json original.png compressed.jpg

# Save difference heatmap
butteraugli --diffmap diff.png original.png compressed.jpg

# Just the score
butteraugli --quiet original.png compressed.jpg
```

### Options

```bash
butteraugli --intensity-target 250 hdr_orig.png hdr_comp.png  # HDR (250 nits)
butteraugli --hf-asymmetry 1.5 original.png compressed.jpg    # Penalize ringing > blur
butteraugli --help
```

## Library Usage

```toml
[dependencies]
butteraugli = "0.7"
```

### Input Formats

| Function | Input Type | Color Space | Use Case |
|----------|------------|-------------|----------|
| `butteraugli` | `ImgRef<RGB8>` | sRGB (gamma-encoded) | Standard 8-bit images |
| `butteraugli_linear` | `ImgRef<RGB<f32>>` | Linear RGB (0.0-1.0) | HDR, 16-bit, float pipelines |

Both APIs support stride (padding) and require minimum 8x8 images.

### Example

```rust
use butteraugli::{butteraugli, ButteraugliParams, Img, RGB8};

let original: Vec<RGB8> = load_image();
let compressed: Vec<RGB8> = load_compressed();

let img1 = Img::new(original, width, height);
let img2 = Img::new(compressed, width, height);

let result = butteraugli(img1.as_ref(), img2.as_ref(), &ButteraugliParams::default())?;

println!("Score: {:.4}", result.score);
```

### Difference Map

```rust
let params = ButteraugliParams::default().with_compute_diffmap(true);
let result = butteraugli(img1.as_ref(), img2.as_ref(), &params)?;

if let Some(diffmap) = result.diffmap {
    let max_diff = diffmap.buf().iter().fold(0.0f32, |a, &b| a.max(b));
    println!("Maximum local difference: {:.4}", max_diff);
}
```

### Custom Parameters

```rust
let params = ButteraugliParams::new()
    .with_hf_asymmetry(1.5)        // Penalize artifacts > blur
    .with_intensity_target(250.0)  // HDR display (nits)
    .with_compute_diffmap(true);
```

## Features

- **`internals`**: Expose internal modules for testing/benchmarking (unstable API)
- **`unsafe-performance`**: Unchecked indexing in hot loops (~6% fewer instructions, pre-validated ranges)

## Performance

SIMD-accelerated via [`archmage`](https://crates.io/crates/archmage) with runtime dispatch:

| Target | CPU Support |
|--------|-------------|
| x86-64-v4 | AVX-512 (Skylake-X, Zen 4+) |
| x86-64-v3 | AVX2/FMA (Haswell+, Zen 1+) |
| x86-64-v2 | SSE4.2 (Nehalem+) |
| ARM64 | NEON (Apple Silicon, Cortex-A75+) |

No C dependencies. Safe Rust by default (`unsafe-performance` is opt-in).

## Accuracy

Validated against libjxl's `butteraugli_main` on 21 real photograph pairs across multiple sizes and JPEG quality levels:

| Image | C++ libjxl | Rust | Relative Diff |
|-------|-----------|------|--------------|
| baby (576x576, Q75) | 3.0873 | 3.0873 | 0.0000% |
| bulb (576x576, Q75) | 2.3174 | 2.3174 | 0.0003% |
| city (576x576, Q75) | 3.8511 | 3.8511 | 0.0000% |
| guitar (576x576, Q75) | 6.5399 | 6.5399 | 0.0000% |
| ... (6 more at 576x576) | | | < 0.0001% |
| 3 images at Q50/Q90 | | | < 0.0001% |
| 5 images at 1024-2048px | | | < 0.0001% |

**All 21 test pairs: < 0.0003% relative difference vs libjxl.**

## API Comparison with C++ libjxl

| Feature | C++ butteraugli | This crate |
|---------|-----------------|------------|
| Input format | Linear RGB float | sRGB u8 or linear RGB f32 |
| Color space | Linear RGB only | sRGB (auto-converted) or linear |
| Channel layout | Planar | Interleaved RGB via `imgref` |
| Stride support | Manual | Built-in via `ImgRef::new_stride()` |

### XYB Note

Butteraugli's internal XYB differs from jpegli's XYB (different nonlinearity, matrix coefficients, and formulas). Always provide RGB input; butteraugli handles the conversion internally.

## References

- [Original butteraugli repository](https://github.com/google/butteraugli)
- [JPEG XL (libjxl)](https://github.com/libjxl/libjxl)
- [Butteraugli paper](https://github.com/google/butteraugli/blob/master/doc/butteraugli-theory.pdf)

## Development

```bash
cargo fmt --all -- --check
cargo clippy --lib --tests -- -D warnings
cargo test --lib
cargo test --test conformance
```

## AI-Generated Code Notice

Developed with Claude (Anthropic). Validated against C++ libjxl `butteraugli_main` with < 0.0003% difference on real photographs.

## License

BSD-3-Clause (same as libjxl)
