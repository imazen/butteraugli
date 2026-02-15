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
