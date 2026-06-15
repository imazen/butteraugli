# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

<!-- NOTE: the workspace version was bumped to 0.9.4 (3e41f32) but 0.9.4 was
     never published to crates.io — the latest published release is 0.9.3.
     Everything below ships with the next publish. -->

### Added
- Cooperative cancellation: `butteraugli_with_stop`, `butteraugli_linear_with_stop`,
  and `butteraugli_strip_with_stop` mirror `butteraugli` / `butteraugli_linear` /
  `butteraugli_strip` but take a trailing `stop: &dyn enough::Stop` token. The token
  is checked at the outermost per-scale boundary of the core compute (one-shot) and
  once per strip at the top of the strip loop — never inside the per-pixel
  psycho/blur/malta/mask/opsin kernels — so a cancellation is honoured at scale /
  strip granularity with zero hot-loop overhead. A new additive
  `ButteraugliError::Cancelled(enough::StopReason)` variant is returned on cancel
  (the enum is `#[non_exhaustive]`, so this is non-breaking). The non-`_with_stop`
  functions delegate to the `_with_stop` versions with `enough::Unstoppable` (zero
  cost). The `enough` crate is re-exported (`butteraugli::enough`) so callers can
  name `Stop` / `Unstoppable` / `StopReason` without a direct dependency.
  The warm-reference `ButteraugliReference` batch API (the codec-sweep hot path:
  precompute the reference once, compare many distorted images) is covered too —
  `compare_with_stop`, `compare_linear_with_stop`, `compare_srgb_with_stop`,
  `compare_linear_imgref_with_stop` check the token at the warm-ref core's outermost
  per-scale boundary (before the full-res + half-res `maybe_join` dispatch), and
  `compare_strip_with_stop`, `compare_linear_strip_with_stop`,
  `compare_strip_srgb_with_stop`, `compare_strip_linear_imgref_with_stop` thread it
  into the already-per-strip-checked strip walker. Same non-breaking delegation: the
  existing methods call the `_with_stop` versions with `enough::Unstoppable`.
- `ButteraugliReference::drop_strip_source(&mut self)` — drops the retained reference-side source data (sRGB u8 or linear f32) so subsequent `compare_strip` / `compare_linear_strip` / `compare_strip_srgb` / `compare_strip_linear_imgref` calls return `InvalidParameter`. The non-strip `compare` / `compare_linear` / `compare_linear_planar` paths are unaffected. Use when the caller has determined that no strip dispatch will follow on this reference and wants to reclaim the per-pixel retention. (3e41f32)
- `ButteraugliReference::shrink_to_fit(&mut self)` — drains the persistent `BufferPool`, releasing any cached transient buffers held between `compare` calls at the cost of one re-allocation on the next `compare` call. Cached XYB pyramid / mask / source data is retained (the warm-ref speedup over a cold `butteraugli()` call still applies). (3e41f32)
- Versioned public-API surface snapshot at `docs/public-api/butteraugli.txt`, regenerated on every `cargo test` by `butteraugli/tests/public_api_doc.rs` (`ZEN_API_DOC=check` verifies in the CI lint job, `=off` skips); `justfile` recipes `fmt` / `api-doc` / `api-doc-check`. Dev-only — not part of the published package.

### Changed
- Exclude `tests/` directories from published packages for both `butteraugli` and `butteraugli-cli`; local `cargo test` is unaffected (3b7afe7)
- `ButteraugliReference::source_linear_rgb` (`#[doc(hidden)]`) now returns `None` for `new()`-built references — they store sRGB u8 instead of linear f32 after the memory fix below. The strip walker uses the new `source_linear_rgb_owned` accessor (also `#[doc(hidden)]`) which materialises the linear bytes from whichever storage form was retained — clones when `new_linear()`-built, LUT-converts when `new()`-built. External callers should not depend on either accessor's signature. (3e41f32)
- Docs: crate-level "Input scaling and `intensity_target`" section documenting the exact contract (linear 1.0 → `intensity_target` nits, default 80 = SDR; >1.0 accepted for HDR; negatives clamped; NaN/Inf rejected); corrected stale "minimum 8x8" / phantom `cli`-feature claims; README badges, strip-API section, complete feature list. (5f1791b, 6afdf90, 9ea9684)
- Tests: the two corpus-dependent conformance tests are now gated behind a new test-only `corpus-tests` feature instead of silently skipping at runtime when `JPEGLI_TESTDATA` is absent; with the feature on, missing data fails loudly. CI compiles them (`--no-run`) so they can't rot. (63a13de, aba2e25)

### Removed
- Dead crate-private `src/xyb.rs` module deleted — it implemented the JPEG XL *codec's* cbrt-based XYB transform, not butteraugli's own opsin transform (`opsin.rs`), had zero callers anywhere in the repo, and its docs misleadingly described the cbrt transform as "used by butteraugli". Its three `XYB_*` consts in `consts.rs` (referenced only by it) went with it. No public API impact — the module was `pub(crate)` and never exported. (c645a39)
- `butteraugli::reference_data` (`#[doc(hidden)]`, 10.9k lines of auto-generated C++ reference scores) moved out of the library into `tests/common/` — it was consumed only by `tests/reference_parity.rs` and shipped as dead weight in the published crate (139 KB → 96 KB package). It was documented as internal test infrastructure; no known external consumer. (601bd9a)
- `test_cpp_butteraugli_comparison` stub removed from the conformance suite — it never performed the C++ comparison its name promised (TODO body, unused cjpegli handle, println-only). Real parity coverage: `tests/reference_parity.rs` (908 `butteraugli_main` cases) + the cpp-parity suites. (63a13de)

### Fixed
- **`iir-blur` stride fix restored after being lost in the 0.9.3 merge.** The W44-phase3-B8 fix (579e91a: stride-aware row addressing in the IIR passes so stale pooled padding can't leak into the visible region) was silently reverted by merge 6e3a7bd — 0.9.3 shipped with `--features iir-blur` nondeterministic across `compare_linear_planar` calls and failing its own regression tests. blur_iir.rs restored from the fix commit (including the `iir_stride_vs_width_repro` test that vanished with it); iir-blur lib tests 87 pass + 2 fail → 90 pass. Default FIR path: zero source change. (ae50608)
- clippy `-D warnings` now green across every feature combo (default, `iir-blur`, `internals`, `unsafe-performance`, all-combined) — CI only linted the default set, so the gated combos had rotted. Includes removing the never-called `ImageF::{get,set,row,row_mut}_unchecked` accessors (`unsafe-performance`-gated; Malta uses its own pre-validated slice access). (58c665d)
- CI: `cargo fmt` over the unformatted reflect-pad commit; `Self::`-qualified the two broken `compare_linear_planar` intra-doc links that failed the Documentation job; `strip_parity` suite (11 tests) added to the CI test matrix — it was never wired in when the 0.9.3 strip API shipped. (5f1791b, 6afdf90, aba2e25)
- **+18% warm-ref peak-heap regression at 16-40 MP** introduced in 0.9.3. (3e41f32) The 0.9.3 strip-API work added a 12 B/pixel linear-f32 source clone to every `new()`/`new_linear()`-built `ButteraugliReference`, on top of a `BufferPool` cap of 48 that allowed the persistent reference pool to retain up to ~3.2 GB of full-image planes at 16 MP. CPU sweep on 2026-05-28 (`benchmarks/heaptrack/summary*.tsv`) measured warm-ref peak heap 3.81 GB vs cold 3.26 GB at 16 MP (+16.9 %), 8.71 GB vs 7.34 GB at 40 MP (+18.7 %). 0.9.4 fixes this in two steps that together restore warm-ref ≤ cold-path peak heap at every measured size:
  - `new()`-built references now retain the original `Vec<u8>` sRGB bytes (3 B/pixel) instead of cloning the pre-converted linear `Vec<f32>` (12 B/pixel). `compare_strip` re-derives the linear bytes on demand via the same `SRGB_TO_LINEAR_LUT` it already applies to the distorted side. `new_linear()`-built references store the input as `Vec<f32>` as before (no compression opportunity beyond clone elision). At 16 MP this saves 192 MB of persistent footprint per reference; at 40 MP, 480 MB.
  - `BufferPool::put` cap reduced 48 → 8 buffers. The original 48-cap was sized for the parallel join's worst-case concurrent buffer count plus headroom, but in practice the persistent reference's pool fills to its cap between compares and holds those buffers through the next compare's peak. 8 is sufficient to skip the mmap/munmap churn within a single `compare` call (validated by three-trial heaptrack measurement) while preventing the persistent pool footprint from dominating heap.
- Post-fix three-trial median peak heap on the `cpu-profile` driver:

  | size | cold | warm-ref (0.9.3) | warm-ref (0.9.4) | Δ vs cold |
  | --- | --- | --- | --- | --- |
  | 4 MP   | 814.61 MB | 836.77 MB (+2.7 %)  | **799.02 MB** | **-1.9 %** |
  | 16 MP  | 3.26 GB   | 3.81 GB   (+16.9 %) | **3.23 GB**   | **-0.9 %** |
  | 40 MP  | 7.79 GB   | 8.71 GB   (+11.8 %) | **7.83 GB**   | **+0.5 %** |

  Wall time is unchanged from 0.9.3 (within run-to-run variance). The warm-ref path remains slower than cold-path on a single compare (the precompute cost is wasted on N=1) and faster per-amortized-call at N ≥ 2, as designed.

## [0.9.3] - 2026-05-28

### Added
- `butteraugli_strip` and `butteraugli_linear_strip` — strip-wise butteraugli with bounded peak memory for very large images. Processes the image in horizontal strips (default halo of 64 rows for the chained FIR Gaussian + Malta + mask blur stack) and aggregates the max-norm + libjxl 3-norm reductions across strips. Scores match the full-image path to within `~1e-2` on identical and `~1e-3` on different inputs at 1024² (FIR finite-support guarantees bit-identical interior diffmaps; aggregation differences arise only from f64 sum associativity). At 40 MP (7680×5120 heaptrack), peak heap drops from 7.43 GB to 1.94 GB (3.8× reduction) at equivalent wall time.
- `ButteraugliReference::compare_strip` (sRGB u8 dist), `compare_linear_strip` (f32 linear dist), `compare_strip_srgb` (ImgRef<RGB8>), `compare_strip_linear_imgref` (ImgRef<RGB<f32>>) — cached-ref strip APIs. Strip-walks the dist side; the ref-side blurs are recomputed per strip so dist and ref share FIR boundary handling. Requires the reference to have been built via `new` or `new_linear` (the planar constructor doesn't retain interleaved source).
- `ButteraugliStripConfig` with `halo_rows` knob for callers that want to trade per-strip overhead for tighter parity.
- `HALO_ROWS_DEFAULT` (64) and `MIN_STRIP_HEIGHT` (8) public constants.
- Hidden `ButteraugliReference::source_linear_rgb` accessor returning the retained interleaved linear-RGB source as `Option<&[f32]>`; used by the strip walker. Marked `#[doc(hidden)]` because external callers should not depend on the representation.

### Changed
- `ButteraugliReference::new` and `new_linear` now retain a clone of the interleaved linear-RGB source so `compare_strip` can slice strip-shaped windows from it. Memory cost: `width * height * 3 * 4 B` per reference (~480 MB at 40 MP). The planar constructor (`new_linear_planar`) is unchanged and does NOT retain the source — `compare_strip` on a planar-constructed reference returns `InvalidParameter`.

### Removed
- W44-PHASE3-B7d strip-tile kernel-level work (`blur_strip.rs`, `malta_strip.rs`, `psycho_strip.rs`, and the `ImageF::strip_view` primitive). The Day 6 honest-stop A/B confirmed the bottom-up kernel approach was slower at every size; the top-down strip walker above is the shipping replacement.

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
