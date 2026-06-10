# butteraugli [![CI](https://img.shields.io/github/actions/workflow/status/imazen/butteraugli/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/butteraugli/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/butteraugli?style=flat-square)](https://crates.io/crates/butteraugli) [![lib.rs](https://img.shields.io/crates/v/butteraugli?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/butteraugli) [![docs.rs](https://img.shields.io/docsrs/butteraugli?style=flat-square)](https://docs.rs/butteraugli) [![codecov](https://img.shields.io/codecov/c/gh/imazen/butteraugli?style=flat-square)](https://codecov.io/gh/imazen/butteraugli) [![License](https://img.shields.io/crates/l/butteraugli?style=flat-square)](LICENSE)

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
butteraugli = "0.9"
```

### Input Formats

| Function | Input Type | Color Space | Use Case |
|----------|------------|-------------|----------|
| `butteraugli` | `ImgRef<RGB8>` | sRGB (gamma-encoded) | Standard 8-bit images |
| `butteraugli_linear` | `ImgRef<RGB<f32>>` | Linear RGB | HDR, 16-bit, float pipelines |

Both APIs support stride (padding). Images smaller than 8x8 (down to 1x1)
are reflect(mirror)-padded up to butteraugli's 8x8 floor and scored; the
diffmap is cropped back to the input size. The strip APIs below still
require at least 8x8.

**Scaling contract:** linear `1.0` maps to `intensity_target` nits
(default `80.0` — the SDR convention used by libjxl's `butteraugli_main`).
Values above `1.0` are accepted and map proportionally above
`intensity_target`. For HDR, scale your linear data so `1.0` is the
mastering/display peak and set `.with_intensity_target(peak_nits)`. The
sRGB u8 path decodes with the sRGB EOTF to linear before the same scaling.

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

### Aggregations

`result.score` is the **max-norm** distance — the historical butteraugli score
used for `<1.0 = good`, `>2.0 = bad` thresholds. `result.pnorm_3` is the
libjxl 3-norm aggregation reported by `butteraugli_main --pnorm` and used in
the Cloudinary CID22 paper — useful for codec rate-distortion sweeps where
averaging tail and bulk distortion is more informative than max alone. Both
are produced in a single fused reduction pass and available regardless of
whether `compute_diffmap` was enabled (no extra allocation).

```rust
println!("max-norm:  {:.4}", result.score);     // or result.max_norm()
println!("3-norm:    {:.4}", result.pnorm_3);
println!("p=4 norm:  {:?}", result.pnorm(4.0)); // requires compute_diffmap=true
```

The CLI exposes the 3-norm via `butteraugli --pnorm`; JSON output always
includes a `pnorm_3` field.

### Custom Parameters

```rust
let params = ButteraugliParams::new()
    .with_hf_asymmetry(1.5)        // Penalize artifacts > blur
    .with_intensity_target(250.0)  // HDR display (nits)
    .with_compute_diffmap(true);
```

### Repeated Comparisons and Bounded Memory

For comparing many distorted images against one reference, precompute the
reference once with `ButteraugliReference` (`new` / `new_linear` /
`new_linear_planar`), then call `compare` / `compare_linear` /
`compare_linear_planar` per candidate — the reference-side XYB pyramid and
masks are reused.

For very large images, the strip API bounds peak memory by walking the
image in horizontal strips (3.8x lower peak heap at 40 MP, equivalent wall
time): `butteraugli_strip` / `butteraugli_linear_strip` one-shot, or
`ButteraugliReference::compare_strip` and friends with a cached reference.
Strip scores match the full-image path to ~1e-2; see `ButteraugliStripConfig`.

## Features

- **`rayon`** *(default)*: Multi-threaded blur and Malta passes
- **`avx512`** *(default)*: AVX-512 runtime dispatch (only used when the CPU supports it)
- **`iir-blur`**: O(N) recursive Gaussian instead of FIR convolution; faster on non-AVX-512
  hardware but not score-parity with libjxl — off by default
- **`unsafe-performance`**: Unchecked indexing in hot loops (~6% fewer instructions, pre-validated ranges)
- **`internals`**: Expose internal modules for testing/benchmarking (unstable API)

## Performance

SIMD-accelerated via [`archmage`](https://crates.io/crates/archmage) with runtime dispatch:

| Target | CPU Support |
|--------|-------------|
| x86-64-v4 | AVX-512 (Skylake-X, Zen 4+) |
| x86-64-v3 | AVX2/FMA (Haswell+, Zen 1+) |
| x86-64-v2 | SSE4.2 (Nehalem+) |
| ARM64 | NEON (Apple Silicon, Cortex-A75+) |
| WASM SIMD128 | Browser/WASI runtimes with SIMD |

No C dependencies. Safe Rust by default (`unsafe-performance` is opt-in).

### vs libjxl C++ `butteraugli_main`

Benchmarked on AMD Ryzen 9 7950X (Zen 4), sRGB-only PNGs (no ICC profiles), mozjpeg q100 vs q50, 5 iterations per image:

| Image | Single-threaded | Multi-threaded (32T) |
|-------|:-:|:-:|
| 1022x818 | 1.2x faster | 1.5x faster |
| 1024x1024 (3 images) | 1.0-1.3x faster | 1.5-2.0x faster |
| 512x512 | 1.3x faster | 1.6x faster |

Multi-threaded gains are larger because Rust parallelizes blur and Malta filter passes via rayon, while `butteraugli_main` shows no effective threading benefit on these image sizes.

## Accuracy

Validated against libjxl's `butteraugli_main` on sRGB photographs (no ICC profiles or gAMA/cHRM chunks) across multiple sizes and JPEG quality levels:

| Image | Size | Quality | C++ libjxl | Rust | Relative Diff |
|-------|------|---------|-----------|------|--------------|
| baby | 576x576 | Q75 | 3.0873 | 3.0873 | 0.0000% |
| bulb | 576x576 | Q75 | 2.3174 | 2.3174 | 0.0003% |
| city | 576x576 | Q75 | 3.8511 | 3.8511 | 0.0000% |
| guitar | 576x576 | Q75 | 6.5399 | 6.5399 | 0.0000% |
| photo A | 1024x1024 | Q25 | 11.3686 | 11.3686 | 0.0000% |
| photo A | 1024x1024 | Q50 | 4.9663 | 4.9663 | 0.0000% |
| photo A | 1024x1024 | Q90 | 1.8161 | 1.8161 | 0.0007% |
| photo B | 1024x1024 | Q75 | 2.9628 | 2.9628 | 0.0003% |
| photo C | 1024x1024 | Q50 | 3.1502 | 3.1502 | 0.0000% |
| photo D | 1022x818 | Q50 | 5.4199 | 5.4199 | 0.0000% |

**All test pairs: < 0.001% relative difference vs libjxl `butteraugli_main`.** Residual differences are FMA rounding noise from hardware fused multiply-add instructions.

> **ICC profiles:** This crate assumes sRGB input. libjxl's `butteraugli_main` applies ICC profile and gAMA/cHRM transforms via its CMS before scoring. Images with non-sRGB ICC profiles (Adobe RGB, Display P3, ProPhoto RGB) will produce different scores between the two implementations. Strip ICC profiles or convert to sRGB before comparing.

## API Comparison with C++ libjxl

| Feature | C++ butteraugli | This crate |
|---------|-----------------|------------|
| Input format | Linear RGB float | sRGB u8 or linear RGB f32 |
| Color space | Linear RGB only | sRGB (auto-converted) or linear |
| ICC profiles | CMS transforms to linear sRGB | Assumes sRGB (profiles ignored) |
| Channel layout | Planar | Interleaved RGB via `imgref` |
| Stride support | Manual | Built-in via `ImgRef::new_stride()` |

### XYB Note

Butteraugli's internal "XYB" is **not** the same color space as JPEG XL / jpegli XYB. Key differences:

- **Matrix coefficients**: Different opsin absorbance weights (e.g., row 2 is `[0.02, 0.02, 0.205]` vs jpegli's `[0.243, 0.205, 0.552]`)
- **Nonlinearity**: Log-based Gamma function (FastLog2f), not cube root
- **Dynamic adaptation**: Blurs the input, computes per-pixel sensitivity ratios, and modulates the opsin-transformed signal — jpegli XYB has no equivalent step
- **Biases**: Large additive biases (`~1.76, ~1.76, ~12.23`) vs jpegli's small bias (`~0.0038`)

Always provide RGB input (sRGB u8 or linear f32); butteraugli handles the conversion internally. Pre-converted XYB cannot be used because the dynamic sensitivity adaptation requires raw linear RGB.

## References

- [JPEG XL reference implementation (libjxl)](https://github.com/libjxl/libjxl) — canonical source for butteraugli
- [Butteraugli paper](https://github.com/google/butteraugli/blob/master/doc/butteraugli-theory.pdf)

> **Note:** This crate ports from libjxl's `lib/extras/butteraugli.cc`, where butteraugli development continued after the original [google/butteraugli](https://github.com/google/butteraugli) repository was archived. Scores are validated against libjxl's `butteraugli_main`, not the archived standalone version.

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
