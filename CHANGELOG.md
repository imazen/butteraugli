# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.2] - 2026-02-15

### Fixed
- Eliminated all clippy warnings (removed unused macros, fixed manual slice copy, suppressed intentional unsafe-performance lints)

## [0.6.1] - 2026-02-15

### Changed
- Updated README with verified parity numbers (< 0.0003% vs libjxl on 21 real photographs)
- Fixed `cargo install` instructions (separate `butteraugli-cli` crate)

## [0.5.1] - 2026-02-14

### Changed
- Updated `archmage` and `magetypes` dependencies from 0.4 to 0.7

## [0.5.0] - 2026-02-14

### Changed
- **Breaking:** Removed `clear_buffer_pool()` from public API. Buffer pools are now
  instance-owned by `ButteraugliReference` and freed automatically on drop. Standalone
  API functions (`butteraugli()`, `butteraugli_linear()`) create a local pool per call.
- Planar linear RGB API (`ButteraugliReference::new_linear_planar`,
  `compare_linear_planar`) avoids interleave/de-interleave overhead when the caller
  already has planar channel data.
- `single_resolution` mode on `ButteraugliParams` skips the half-resolution pass
  for ~25% faster approximate results.

### Performance
- Eliminated ~400 MB temporary allocation in `xyb_low_freq_to_vals`
- Added ImageF buffer pool (reuses allocations within a comparison, avoids
  mmap/munmap churn)
- Fixed vertical-pass cache thrashing in `blur_mirrored_5x5`
- AVX-512 and AVX2 SIMD via `archmage`/`magetypes` for blur convolution
- ARM64 (NEON) autovectorization via `multiversed` presets

### Added
- `ButteraugliReference::new_linear_planar` and `compare_linear_planar` for
  stride-aware planar f32 input
- `ButteraugliParams::with_single_resolution` to skip the half-res pass
- 32-bit (i686) and WASM (wasm32-unknown-unknown, wasm32-wasip1) CI targets

## [0.4.0] - 2025-12-27

### Changed
- **Breaking:** Redesigned public API around `imgref::ImgRef` and `rgb::RGB8`/`RGB<f32>`
  types for ergonomic stride-aware image handling.
- New primary entry points: `butteraugli()` and `butteraugli_linear()`.
- Old `compute_butteraugli`/`compute_butteraugli_linear` deprecated.
- Added `ButteraugliReference` for precomputed reference data (~40-50% speedup
  when comparing multiple distorted images against the same reference).

## [0.3.2] - 2025-12-26

### Changed
- Removed `unsafe-perf` feature; all SIMD code is 100% safe Rust via the `wide` crate.
- Reference precomputation (`ButteraugliReference`) with multiversion SIMD dispatch.
- Aligned SIMD backing store (`simd_aligned`) for cache-friendly access.
- Mirrored boundary handling for kernel-size-5 blur (fixes accuracy).

## [0.3.0] - 2025-12-26

### Performance
- ~2.5x speedup matching C butteraugli performance.
- Safe SIMD via `wide` crate (always-on, no feature flag needed).
- Malta filter optimization.

## [0.2.1] - 2025-12-25

### Added
- Pixel layout diagram in README for clearer input format documentation

## [0.2.0] - 2025-12-25

### Added
- `compute_butteraugli_linear` function for linear RGB f32 input (HDR/16-bit workflows)
- `srgb_to_linear` helper function for manual color space conversion
- Comprehensive documentation of XYB color space differences vs jpegli
- Input requirements documentation (minimum size, channel layout)

### Changed
- Improved API idiomaticity
- Better documentation of C++ API differences

## [0.1.0] - 2025-12-24

### Added
- Initial release
- Pure Rust implementation of butteraugli perceptual quality metric
- `compute_butteraugli` function for sRGB u8 input
- Multi-resolution analysis (SubSample2x)
- Malta convolution filter
- MaskY, MaskDcY, MaskPsychoImage, CombineChannelsToDiffmap
- SIMD optimizations via `wide` crate
- 195 synthetic test cases validated against C++ libjxl
- `ButteraugliParams` for customization (hf_asymmetry, xmul, intensity_target)
- `score_to_quality` and `butteraugli_fuzzy_class` helper functions
- Optional diffmap output for per-pixel analysis

### Technical Details
- Accurate port of FastLog2f rational polynomial from C++
- Correct Y-channel MaximumClamp order in frequency separation
- Matched C++ blur and multiresolution behavior
