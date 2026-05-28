# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.3] - 2026-05-28

### Added
- `butteraugli_strip` and `butteraugli_linear_strip` — strip-wise butteraugli with bounded peak memory for very large images. Processes the image in horizontal strips (default halo of 64 rows for the chained FIR Gaussian + Malta + mask blur stack) and aggregates the max-norm + libjxl 3-norm reductions across strips. Scores match the full-image path to within `~1e-2` on identical and `~1e-3` on different inputs at 1024² (FIR finite-support guarantees bit-identical interior diffmaps; aggregation differences arise only from f64 sum associativity). At 40 MP (7680×5120 heaptrack), peak heap drops from 7.43 GB to 1.94 GB (3.8× reduction) at equivalent wall time.
- `ButteraugliReference::compare_strip` (sRGB u8 dist), `compare_linear_strip` (f32 linear dist), `compare_strip_srgb` (ImgRef<RGB8>), `compare_strip_linear_imgref` (ImgRef<RGB<f32>>) — cached-ref strip APIs. Strip-walks the dist side; the ref-side blurs are recomputed per strip so dist and ref share FIR boundary handling. Requires the reference to have been built via `new` or `new_linear` (the planar constructor doesn't retain interleaved source).
- `ButteraugliStripConfig` with `halo_rows` knob for callers that want to trade per-strip overhead for tighter parity.
- `HALO_ROWS_DEFAULT` (64) and `MIN_STRIP_HEIGHT` (8) public constants.
- Hidden `ButteraugliReference::source_linear_rgb` accessor returning the retained interleaved linear-RGB source as `Option<&[f32]>`; used by the strip walker. Marked `#[doc(hidden)]` because external callers should not depend on the representation.

### Changed
- `ButteraugliReference::new` and `new_linear` now retain a clone of the interleaved linear-RGB source so `compare_strip` can slice strip-shaped windows from it. Memory cost: `width * height * 3 * 4 B` per reference (~480 MB at 40 MP). The planar constructor (`new_linear_planar`) is unchanged and does NOT retain the source — `compare_strip` on a planar-constructed reference returns `InvalidParameter`.

## [0.9.2] - 2026-05-01

### Added
- `ButteraugliResult.pnorm_3` — libjxl 3-norm aggregation, matching
  `lib/extras/metrics.cc:ComputeDistanceP` at p=3 (the value reported by
  `butteraugli_main --pnorm` and used in the Cloudinary CID22 paper). The
  average of three p-norms at exponents 3, 6, 12. Precomputed in the same
  fused pass as the max-norm score; always populated regardless of
  `compute_diffmap`, with no extra allocation (the transient internal
  diffmap is freed before return when the user didn't request it). Cost:
  +2.1M callgrind instructions on 512×512 (~0.3% of total); scales linearly,
  so negligible at 4K/8K. Avoids materializing a 33 MB / 132 MB diffmap
  back to the caller just to derive a 3-norm. Closes #6.
- `ButteraugliResult::pnorm(p) -> Option<f64>` — short-circuits to
  `pnorm_3` for `p ≈ 3.0`; for other `p` values requires
  `compute_diffmap = true` (returns `None` otherwise).
- `ButteraugliResult::max_norm()` — explicit alias for the `score` field
  so call sites can be unambiguous when `pnorm_3` is in play.
- `butteraugli --pnorm` CLI flag — adds `3-norm: X.XXXXXX` line to text
  and quality output. JSON output always includes the `pnorm_3` field
  regardless of the flag.
- `iir-blur` cargo feature — Charalampidis 2016 recursive Gaussian as an O(N)
  per-pixel alternative to the FIR separable convolution (a62453a, ef750a3).
  Off by default. Real-photo parity vs FIR: 0.1–5% relative score deviation
  (mean ~2%, GB82 corpus Q75 576×576). Tiny synthetic images deviate widely
  because the IIR uses zero-padding while FIR uses clamp-to-edge — unsuitable
  for parity-sensitive workflows. Speed: matches FIR on Zen 4 with AVX-512;
  ~22% faster on hardware without v4 (most ARM, older x86). The CLI accepts
  the same `iir-blur` feature for consistent dispatch.

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
