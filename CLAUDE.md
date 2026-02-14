# butteraugli — Project Guide

Pure Rust port of libjxl's butteraugli perceptual image quality metric.

## Known Bugs

### BUG: Subsampling in sRGB u8 instead of linear float (UNFIXED)

**Files:** `diff.rs:109` (`subsample_rgb_2x`), `precompute.rs:951` (same function duplicated)

The C++ `SubSample2x` operates on `Image3F` (linear float RGB), averaging with `0.25f * pixel`.
The Rust version averages sRGB u8 values as `u32` and truncates back to `u8`.

This is wrong in two ways:

1. **Nonlinear averaging.** sRGB is gamma-compressed (~2.2). Averaging compressed values
   produces darker results than averaging in linear space then compressing. For two pixels
   at sRGB 50 and 200: linear average → sRGB ≈ 159, but sRGB average = 125. That's a
   massive difference in dark regions where the gamma curve is steepest.

2. **Quantization loss.** Truncating averaged floats to u8 destroys precision. Linear float
   values 0.0039 and 0.0042 average to 0.00405, but in sRGB u8, values 16 and 17 average
   to 16 (truncated). This matters most at low luminance — exactly where butteraugli's
   masking model is most sensitive.

**Impact:** Systematic underestimation of distortion at sub-resolutions, especially in
dark regions and smooth gradients. Contributes to the 8-25% gap vs libjxl.

**Fix:** Convert sRGB → linear float before subsampling (or subsample the `Image3F` that
`srgb_to_xyb_butteraugli` already produces internally). The C++ never touches sRGB u8
for subsampling — it receives `Image3F` (linear) and subsamples that.

### BUG: Single-level multiresolution instead of recursive (UNFIXED)

**Files:** `precompute.rs:116-130`, `diff.rs:665-700`

The C++ `ButteraugliComparator::Make` recurses: it calls `Make(SubSample2x(rgb0))` until
dimensions drop below 8. For a 512x512 image, that's 6 levels (512→256→128→64→32→16→8).
Each sub-level's diffmap is added via `AddSupersampled2x(subresult, 0.5, result)`.

The Rust implementation only does **one level** of subsampling (`half: Option<ScaleData>`).

The `butteraugli_main` tool and `ButteraugliDiffmap` both use the recursive
`ButteraugliComparator` path. The `ButteraugliInterfaceInPlace` path also does one level,
but that's NOT what the standard tools or jxl encoder use.

**Impact:** Missing recursive sub-levels means low-frequency differences are underweighted.
Each deeper level adds `0.5 * sub_score` (scaled by `1 - 0.3*0.5`) to the result. The
cumulative effect is 8-25% underestimation on real images, growing with image size (more
recursion levels for larger images).

**Fix:** Make the multiresolution recursive. `ButteraugliReference` should contain a
`Box<Option<ButteraugliReference>>` for the sub-level, not `Option<ScaleData>`. The
standalone API in `diff.rs` needs the same recursive structure.

## Parity Testing Failures — Post-Mortem

### How two major algorithmic bugs survived the test suite

These bugs existed since the initial port and were never caught. Here's why, and what
to do differently.

### Problem 1: Loose tolerances masked real bugs

The parity tests used:
- `cpp_parity.rs`: 20-30% relative **OR** absolute diff < 0.5-1.5
- `reference_parity.rs`: 20% relative, with 10% of tests allowed to fail
- Edge+blur at 32% divergence was labeled "known divergence" and accepted

**Rule: No tolerance above 5% relative for real-image butteraugli parity.** If a test
needs 20% tolerance, the implementation has a bug — find it instead of widening the
tolerance. The `OR diff < 0.5` escape hatch is banned; it lets any score below 2.5
pass with 20% error.

### Problem 2: Small synthetic images hide size-dependent bugs

Tests used 32x32 and 64x64 images. At 32x32, the recursive multiresolution only adds
2 levels (32→16→8). At 512x512, it's 6 levels. The cumulative contribution from deeper
levels grows with image size, so the bug was invisible on tiny test images.

The sRGB subsampling error is also smaller on synthetic images (uniform, gradient,
checkerboard) because they mostly live in the linear range of the sRGB curve. Real
photographs have dark shadows where the gamma nonlinearity is steepest.

**Rule: Parity tests MUST include real photographs at realistic sizes (512x512+).**
Synthetic tests catch formula errors. Only real images at real sizes catch architectural
errors like missing recursion levels or wrong color space for subsampling.

### Problem 3: Reference tests were #[ignore] and required manual setup

The `test_real_image_score_parity` test was `#[ignore]` and required `JPEGLI_TESTDATA`
and `CJPEGLI_PATH` environment variables. Nobody ran it routinely.

**Rule: At least one real-image parity test must run in CI without manual setup.** Ship
a small (64KB) test image pair with known C++ reference scores baked into the test. No
env vars, no external binaries, no ignoring.

### Problem 4: FFI tests called the right C++ path but tolerances absorbed the diff

The `butteraugli_compare` FFI wrapper calls `ButteraugliDiffmap` → `ButteraugliComparator::Make`
(recursive). So the C++ side WAS using the correct recursive path. But the 20-30% tolerance
on 32x32-64x64 images meant the single-level Rust result was "close enough" to pass.

**Rule: When a parity test passes with 20% tolerance, that's not parity — that's a
coincidence detector.** Tighten tolerances until tests fail, then fix the bugs they reveal.

## Testing Standards

### Butteraugli parity requirements

1. **Real images, real sizes.** At least 5 photographs at 512x512+ in CI. Use GB82 corpus
   subset or ship compressed test pairs in the repo.

2. **Tight tolerances.** Max 2% relative difference on real images vs libjxl `butteraugli_main`.
   Max 0.001% between Rust optimization levels (FMA noise only).

3. **No escape hatches.** No `|| diff < X` clauses. No "allow N% of tests to fail."
   Every test case must pass individually.

4. **Test the tool people actually use.** The reference is `butteraugli_main` from libjxl,
   which uses `ButteraugliComparator` (recursive). NOT the standalone Google `butteraugli-c`
   (different codebase). NOT `ButteraugliInterfaceInPlace` (single-level).

5. **Test at multiple sizes.** Same image at 256, 512, 1024 to catch size-dependent bugs.
   The recursion depth scales with image size.

6. **Intermediate verification.** Don't just test the final score. Test blur output, mask
   output, frequency separation at intermediate stages with tight tolerances. A 2% error
   at each stage compounds to 20% at the end.

## Architecture Notes

### Multiresolution structure (C++ reference)

```
ButteraugliComparator::Make(rgb0, params):
  1. OpsinDynamicsImage(rgb0) → xyb0
  2. SeparateFrequencies(xyb0) → pi0 (PsychoImage)
  3. SubSample2x(rgb0) → rgb0_sub (linear float, NOT sRGB u8)
  4. Make(rgb0_sub, params) → sub_ (RECURSIVE)
  5. Base case: width < 8 || height < 8

ButteraugliComparator::Diffmap(rgb1):
  1. OpsinDynamicsImage(rgb1) → xyb1
  2. DiffmapPsychoImage(xyb1) → result (full-res diffmap)
  3. If sub_:
     a. SubSample2x(rgb1) → rgb1_sub
     b. OpsinDynamicsImage(rgb1_sub) → sub_xyb
     c. sub_.DiffmapOpsinDynamicsImage(sub_xyb) → subresult (RECURSIVE)
     d. AddSupersampled2x(subresult, 0.5, result)
```

### SubSample2x (C++ reference)

Operates on `Image3F` (linear float). Per-channel:
- Accumulates `0.25 * pixel` into output (2x2 box filter)
- If input has odd width: doubles the last output column
- If input has odd height: doubles the last output row

NOT: averaging sRGB u8 values. NOT: truncating to integers.

### AddSupersampled2x (C++ reference)

```
dest[y][x] *= (1.0 - 0.3 * 0.5);  // kHeuristicMixingValue = 0.3, weight = 0.5
dest[y][x] += 0.5 * src[y/2][x/2];
```

### C++ entry points (which path does what)

| Entry point | Multiresolution | Used by |
|------------|----------------|---------|
| `ButteraugliComparator::Make` + `Diffmap` | Recursive (N levels) | `butteraugli_main`, jxl encoder, FFI `butteraugli_compare` |
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
