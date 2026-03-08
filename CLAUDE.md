# butteraugli — Project Guide

Pure Rust port of libjxl's butteraugli perceptual image quality metric.

## Known Bugs

None currently known. Parity with libjxl `butteraugli_main` verified at <0.0003% on
21 real photograph pairs (GB82 576x576 + large images 1024-2048px, Q50/Q75/Q90).

## Parity Status (2026-02-14)

### FIXED: Subsampling in sRGB u8 instead of linear float

**Commit:** `0f396af`

The Rust port subsampled sRGB u8 pixel data (averaging gamma-compressed values and
truncating to u8). The C++ subsamples `Image3F` in linear float space. Fix: convert
sRGB → linear f32 before subsampling, subsample in linear space.

### RESOLVED: "Single-level multiresolution" was a misdiagnosis

The C++ `Make()` recurses, but `Diffmap()` only ever uses ONE sub-level via
`DiffmapOpsinDynamicsImage()`. The Rust single-level behavior is correct.

Exhaustive code tracing confirmed: `DiffmapOpsinDynamicsImage` calls
`SeparateFrequencies` + `DiffmapPsychoImage` — no recursion. The recursive
tree is built but only the immediate child is used during diffmap computation.

### RESOLVED: 5-25% gap was a test artifact, not an algorithm bug

The apparent 5-25% score gap on real photographs was caused by **PNG gamma metadata
mismatch** in the test images, not by any butteraugli algorithm difference.

**Root cause:** The Q75 JPEG → PNG conversion tool (ImageMagick) embedded `gAMA: 0.45455`
and `cHRM` chunks in the distorted PNGs. The reference lossless PNGs had no such
metadata. libjxl's CMS reads these chunks and applies a different color transfer
function to each image, inflating the perceived difference.

The Rust code's PNG decoder ignores gamma metadata and applies standard sRGB transfer
to both images — giving the **correct** butteraugli score.

**Proof:** Stripping gamma metadata from all Q75 PNGs made C++ scores match Rust
within 0.0003% on all 10 test images:

| Image    | C++ (gamma) | C++ (stripped) | Rust    | Old gap | New gap  |
|----------|-------------|----------------|---------|---------|----------|
| baby     | 3.3165      | 3.0873         | 3.0872  | 6.9%    | 0.0000%  |
| bulb     | 2.8715      | 2.3174         | 2.3174  | 19.3%   | 0.0003%  |
| city     | 4.0689      | 3.8511         | 3.8511  | 5.4%    | 0.0000%  |
| dog      | 3.1956      | 2.5060         | 2.5060  | 21.6%   | 0.0001%  |
| flowers  | 3.2353      | 2.4274         | 2.4274  | 25.0%   | 0.0001%  |
| girl     | 5.1367      | 4.8268         | 4.8268  | 6.0%    | 0.0000%  |
| grass    | 2.8591      | 2.4344         | 2.4344  | 14.9%   | 0.0001%  |
| guitar   | 6.8867      | 6.5399         | 6.5399  | 5.0%    | 0.0000%  |
| haze     | 2.3533      | 2.1042         | 2.1042  | 10.6%   | 0.0001%  |
| house    | 3.9174      | 3.0071         | 3.0071  | 23.2%   | 0.0000%  |

**Lesson for test methodology:** When comparing perceptual metrics, ensure BOTH images
have identical color metadata. PNG gAMA/cHRM chunks cause CMS-aware decoders (libjxl)
to apply different transfer functions. Strip metadata or use identical encoding pipelines.

## Parity Testing Post-Mortem

### How the sRGB subsampling bug survived the test suite

1. **Loose tolerances masked real bugs.** Tests used 20-30% relative tolerance with
   `|| diff < 0.5` escape hatches. Rule: max 2% tolerance on real images, no escape hatches.

2. **Small synthetic images hide architectural bugs.** Tests used 32x32 and 64x64 images.
   The sRGB subsampling error is smaller on synthetic images because they mostly live in
   the linear range of the sRGB curve. Rule: test with real photographs at 512x512+.

3. **Reference tests were #[ignore] with manual setup.** Rule: ship a small test image
   pair with baked-in C++ reference scores. No env vars, no external binaries.

4. **FFI tolerances absorbed the diff.** The FFI tests did call the correct C++ path,
   but 20-30% tolerance on tiny images meant the bug was "close enough" to pass.
   Rule: if a test needs 20% tolerance, the implementation has a bug — find it.

## Testing Standards

### Butteraugli parity requirements

1. **Real images, real sizes.** At least 5 photographs at 512x512+ in CI.
2. **Tight tolerances.** Max 2% relative difference vs libjxl `butteraugli_main`.
3. **No escape hatches.** No `|| diff < X` clauses. No "allow N% to fail."
4. **Test the right tool.** Reference is `butteraugli_main` from libjxl (recursive
   Comparator), NOT standalone `butteraugli-c` (different codebase).
5. **Match color metadata.** Both images must have identical PNG gAMA/cHRM chunks
   (or both have none) to avoid CMS-induced differences.

## Architecture Notes

### Multiresolution structure

C++ `Make()` builds a recursive tree but `Diffmap()` only uses one sub-level.
Rust correctly matches this single-sub-level behavior.

```
Diffmap(rgb1):
  1. OpsinDynamicsImage(rgb1) → xyb1
  2. DiffmapPsychoImage(xyb1) → result (full-res diffmap)
  3. If sub_:
     a. SubSample2x(rgb1) → rgb1_sub (linear float)
     b. DiffmapOpsinDynamicsImage(rgb1_sub) → subresult (NOT recursive)
     c. AddSupersampled2x(subresult, 0.5, result)
```

### AddSupersampled2x

```
dest[y][x] *= (1.0 - 0.3 * 0.5);  // kHeuristicMixingValue = 0.3, weight = 0.5
dest[y][x] += 0.5 * src[y/2][x/2];
```

### C++ entry points

| Entry point | Multiresolution | Used by |
|------------|----------------|---------|
| `ButteraugliComparator::Make` + `Diffmap` | Recursive tree, single sub-level used | `butteraugli_main`, jxl encoder |
| `ButteraugliInterfaceInPlace` | Single level | Legacy API |
| `ButteraugliDiffmap` | Recursive (via Comparator) | Most callers |

## Performance Notes

### Optimization session 7 (2026-03-08)

Autoversioned remaining non-SIMD functions: add_supersampled_2x, subsample_channel_2x,
compute_score_from_diffmap.

| Optimization | Instructions (single call V3) | Reduction |
|-------------|------------------------------|-----------|
| Baseline (post-session-6) | 648.1M | — |
| Autoversioned add_supersampled_2x (pair loop) | 643.9M | -0.65% (3.9M → 0.3M) |
| Multi-lane max reduction (compute_score) | 643.8M | score: 1.26M → 256K (-80%) |
| Autoversioned subsample_channel_2x_interior | 642.2M | subsample: 1.87M → 0.84M (-55%) |

Key techniques:
- add_supersampled_2x: Process dest pixels in pairs sharing same src pixel. LLVM generates
  vshufps/vpermpd for 2:1 deinterleave + vfmadd213ps for blend+weight FMA.
- compute_score: 8 independent max accumulators break loop-carried dependency.
  LLVM uses vmaxps (4-wide packed max) instead of scalar vmaxss + NaN handling.
- subsample_channel: zip+chunks_exact(2) for per-channel SIMD. LLVM uses vhaddps
  for horizontal pair sums + vshufps for row deinterleave.

### Optimization session 6 (2026-03-08)

Stack-allocated kernels, register-accumulate vertical blur, mask copy elimination,
zip patterns for bounds check elimination, pool reuse from construction.

| Optimization | Instructions (single call V3) | Reduction |
|-------------|------------------------------|-----------|
| Baseline (post-session-5) | ~649.5M | — |
| Stack-allocated kernel + scaled kernel | ~648.5M | minor (allocation elimination) |
| Register-accumulate vertical blur | 648.5M | 0% (LLVM already optimized) |
| Mask copy elimination (1MB) | 648.5M | minor (memory traffic only) |
| Zip patterns in l2_diff_asymmetric | 648.1M | l2_diff_asymmetric -21.4% |
| Pool reuse from construction | — | wall-clock first-call improvement |

Profiling methodology: Single-threaded callgrind_single example (1 reference + 1 compare).

Confirmed algorithmic optimization ceiling reached:
- blur v3: 297M (46.2%) — unrolled 4×, zero bounds checks in hot loop
- malta v3: 118M (18.3%) — 8-wide SIMD, zero-padded borders
- blur_5x5 v3: 18M (2.8%)
- opsin v3: 16M (2.5%) — fully vectorized (vdivps, vfmadd, vmaxps)
- All remaining functions < 1.3% each, all autoversioned

Assembly verification confirms optimal codegen:
- Blur vertical interior: LLVM unrolls kernel loop 4×, no bounds checks inside loop
- Opsin inner loop: 6× vdivps (3 log2f + 3 sensitivity) + FMA chains
- combine_channels_to_diffmap_fused: vdivps + vsqrtps (mask + DC diff fused)
- All process_uhf_hf_x/y: vmaxps/vminps/vandps vectorized

Further gains require: `unsafe-performance` (~6% memset elimination), IIR blur (breaks parity),
or major architectural changes (streaming/tiling).

### Optimization session 5 (2026-03-08)

Malta zero-padding to eliminate border handling, f32 mask computation.

| Optimization | Instructions (bench V3) | Reduction |
|-------------|----------------------|-----------|
| Baseline (post-session-4) | 9,269M | — |
| Malta zero-padding (eliminate extract_window) | 8,657M | -6.6% |
| f32 mask + border-only zeroing | 8,391M | -3.1% |

Total: 9,269M → 8,391M = 878M / 9.5% reduction.

Key techniques:
- Zero-pad Malta diff image with 4px borders so ALL pixels use fast SIMD interior path.
  Eliminates extract_window (369M) and scalar malta_unit border path (97M).
- Border-only zeroing: only zero the border strips (~2% of padded image) instead of
  the entire image. Interior gets overwritten by the copy.
- Convert mask_y/mask_dc_y from f64 to f32 inline in combine_channels_to_diffmap_fused.
  The f64 division blocked SIMD vectorization. f32 precision is sufficient (34M → 25M per call).

Remaining profile (V3 callgrind, bench 512x512, 21 iterations):
- blur v3: 3,298M (39.3%) — algorithmic minimum, V4 runs natively
- malta v3: 2,590M (30.9%) — algorithmic minimum, V4 runs natively
- memset: 521M (6.2%) — pool allocation zeroing, needs `unsafe-performance`
- blur_5x5 v3: 309M (3.7%) — opsin preprocessing
- opsin v3: 269M (3.2%) — gamma/log2
- srgb_to_linear: 196M (2.3%) — one-shot only
- malta_compute_scaled_diffs v3: 173M (2.1%)
- All other functions < 1.5% each

### Optimization session 4 (2026-03-08)

Reference mask precomputation, branch-free autoversioned per-pixel transforms,
fuzzy erosion SIMD, Malta first-pass extraction.

| Optimization | Instructions (one-shot V3) | Reduction |
|-------------|--------------------------|-----------|
| Baseline (post-session-3) | 2.123B | — |
| Mask precompute + fuzzy erosion split | ~2.07B | ~2.5% |
| DC diff fusion (combine_channels) | ~2.05B | ~1% |
| Psycho autoversion (subtract, range) | ~2.02B | ~1.5% |
| Fused UHF/HF single-pass (process_uhf_hf_x/y) | 1.587B | -21.7% |
| Fuzzy erosion SIMD (update_min3 branch-free) | 1.552B | -2.2% |
| Branch-free l2_diff_asymmetric | 1.531B | -1.4% |
| Malta first-pass autoversion | 1.513B | -1.2% |

Total: 2.123B → 1.513B = 28.7% reduction.

Key techniques:
- Branch-free patterns: copysign+max replaces if/else chains, enabling SIMD vectorization
- update_min3: f32::min/max sorting network replaces branchy 3-element insertion sort
- Malta diff copysign trick: `sv1 = v1 * copysign(1, v0)` unifies sign-dependent logic
- Fused single-pass: read data once, write multiple outputs (UHF + HF from same blur)
- Valgrind 3.18 doesn't support AVX-512; callgrind shows V3 but V4 runs natively

### Optimization session 3 (2026-03-08)

V-pass row-major restructure, mask copy elimination, psycho pass fusion, LF swap.

| Optimization | Instructions (512x512) | Reduction |
|-------------|----------------------|-----------|
| Baseline (post-session-2) | 13.36B | — |
| V-pass row-major + chunks_exact BCE | 10.68B | -20.1% (blur 5.69B→3.33B) |
| Mask copy elimination | ~10.6B | minor |
| Psycho pass fusion (HF/MF/UHF) | ~10.6B | minor |
| LF swap (mem::swap vs copy) | 10.96B* | -18% total |

*Final measurement slightly higher due to callgrind noise; real improvement ~18%.

Key technique: Restructured V-pass from column-major (x outer, kernel inner) to
row-major (kernel outer, x inner). This amortizes kernel weight splat across all
x positions and enables `chunks_exact(8)` for bounds-check-free SIMD. LLVM preserved
the row-major order, generating 2-way unrolled inner loops with memory-source FMA.

### Optimization session 2 (2026-03-08)

Non-transposing H+V blur, V-pass row pointer pre-collection, AVX-512 dispatch fix.

| Optimization | Instructions (flower_big 512x341) | Wall clock (512x512) |
|-------------|----------------------------------|---------------------|
| Baseline (post-session-1) | 758M | ~44ms |
| Non-transposing H+V blur | ~703M (-7.3%) | ~41ms |
| V-pass row pointer pre-collection | 522M (-25.8%) | ~39ms |
| AVX-512 dispatch fix | 522M (same, v4 at runtime) | 32ms (-17%) |

Key fix: butteraugli's Cargo.toml didn't forward `avx512` feature to archmage,
so `#[cfg(feature = "avx512")]` in the incant macro was always false. The v4
dispatch functions existed but were never compiled. Adding `avx512 = ["archmage/avx512",
"magetypes/avx512"]` as a default feature activated AVX-512 runtime dispatch.

Zen 4 note: AVX-512 uses 256-bit execution units (zmm split to 2×ymm μops),
so f32x16 doesn't improve throughput over 2×f32x8. But it helps through reduced
loop overhead (fewer iterations) and wider stores.

### Optimization session (2026-02-14)

Total instruction reduction: 14.16B → 8.60B (39.3%) on 512x512.

| Optimization | Before → After | Reduction |
|-------------|---------------|-----------|
| SIMD Malta (AVX2 f32x8) | 2.09B → 1.50B | -28% |
| Border convolution batch+zip | 1.64B → 0.58B | -64.6% |
| FMA + pre-sliced SIMD interior | 4.51B → 2.26B | -49.8% |
| Malta FMA + pattern dedup | 1.50B → 1.48B | -1.7% |
| Buffer pool (earlier commits) | 0.85B → 0.44B | -47.8% memset |
| blur_mirrored_5x5 (earlier) | — | -80.5% |

All optimizations are correctness-preserving: max 7e-6 difference vs unoptimized
(FMA rounding noise only).

### Key SIMD lessons

- `f32::mul_add` only compiles to FMA inside `#[rite]`/`#[arcane]` contexts. Outside
  them it calls `fmaf` library function — 12% slower than no FMA at all.
- Pre-slicing for bounds check elimination: create one validated slice covering all
  SIMD loads in a chunk, then subslice within it.
- `iter().zip()` eliminates per-element bounds checks vs indexed access.
- Malta HF patterns 13-16 have identical offsets to 8,7,6,5. Cache `sum*sum` and
  reuse at the original accumulation positions (not `2*sum*sum` — different FP result).
- **archmage feature gates**: `incant!` wraps v4 code in `#[cfg(feature = "avx512")]`
  which checks the CONSUMING crate's features, not archmage's. Must forward feature.
