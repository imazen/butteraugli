# Butteraugli Full C++ Parity Plan

## Current Status

- **185/191 tests pass** (97% pass rate at 20% tolerance)
- **6 tests fail** with ~20-32% divergence
- All failures involve blur distortion patterns
- Rust consistently scores **lower** than C++ (detects less difference)

### Failing Tests

| Test | Rust | C++ | Divergence |
|------|------|-----|------------|
| edge_v_vs_blur_23x31 | 3.79 | 5.58 | 32% |
| edge_v_vs_blur_47x33 | 3.82 | 5.69 | 33% |
| random_mid_contrast_1.2_32x32 | 2.55 | 3.26 | 22% |
| random_mid_blur_32x32 | 7.31 | 9.40 | 22% |
| random_mid_blur_47x47 | 7.48 | 9.47 | 21% |
| random_mid_blur_64x64 | 7.64 | 9.62 | 21% |

### What Already Matches

- sRGB→linear conversion: **exact**
- Gamma function (FastLog2f): **exact**
- Frequency bands (LF, MF, HF, UHF): **<0.01%**
- OpsinDynamicsImage blur (sigma=1.2): **exact** (mirrored boundaries)

## Investigation Roadmap

### Phase 1: Isolate Divergence Location

The divergence is downstream of frequency separation. Need to identify which stage:

```
Frequency Bands (MATCH) → Malta Filter → Mask → Combine → Score (DIVERGES)
```

#### 1.1 Add Malta Filter FFI Comparison

**File:** `butteraugli/tests/step_by_step_comparison.rs`

Create test that compares Malta filter output directly:

```rust
// Need C++ FFI wrapper for MaltaDiffMap
fn test_malta_diff_map_parity() {
    // Create identical frequency band inputs
    // Run both Rust and C++ Malta filter
    // Compare outputs pixel-by-pixel
}
```

**Requires:** New FFI function `butteraugli_malta_diff_map` in:
- `/home/lilith/work/jpegli-rs/internal/jpegli-cpp/lib/extras/butteraugli_c.cc`
- `/home/lilith/work/jpegli-rs/internal/jpegli-cpp/lib/extras/butteraugli_c.h`

#### 1.2 Add Mask Computation FFI Comparison

Compare intermediate mask values:
- After `DiffPrecompute`
- After blur (sigma=2.7)
- After `FuzzyErosion`
- Final mask values

**Requires:** New FFI functions:
- `butteraugli_diff_precompute`
- `butteraugli_fuzzy_erosion`
- `butteraugli_mask`

#### 1.3 Add CombineChannelsToDiffmap Comparison

Compare the final diffmap combination before score extraction.

### Phase 2: Potential Root Causes

Based on analysis, these are the most likely sources of divergence:

#### 2.1 Malta Filter Scaling Differences

**File:** `butteraugli/src/malta.rs:876-901`

The Malta diff map computation involves several scaling factors:

```rust
let w_pre0gt1 = mulli * (K_WEIGHT0 * w_0gt1).sqrt() / (LEN * 2.0 + 1.0);
let w_pre0lt1 = mulli * (K_WEIGHT1 * w_0lt1).sqrt() / (LEN * 2.0 + 1.0);
let norm2_0gt1 = (w_pre0gt1 * norm1) as f32;
let norm2_0lt1 = (w_pre0lt1 * norm1) as f32;
```

**Check:**
- f64 vs f32 precision in intermediate calculations
- Order of operations matching C++
- Constant values matching exactly

#### 2.2 Malta Filter Border Accumulation

C++ uses SIMD for interior pixels and `PaddedMaltaUnit` for borders.

**File:** `butteraugli/src/malta.rs:952-1038`

The Rust safe path processes all pixels uniformly. Verify:
- Border pixel handling matches C++ `PaddedMaltaUnit`
- The 12-column stride padding in C++ doesn't affect results
- SIMD vs scalar gives identical results

#### 2.3 Mask Blur Boundary Handling

**File:** `butteraugli/src/mask.rs:202-203`

```rust
let blurred0 = gaussian_blur(&diff0, MASK_RADIUS);  // MASK_RADIUS = 2.7
let blurred1 = gaussian_blur(&diff1, MASK_RADIUS);
```

For sigma=2.7, kernel size = 2*floor(2.25*2.7)+1 = 13.

C++ uses `ConvolutionWithTranspose` for kernel size != 5.
Rust uses `gaussian_blur` with clamp-and-renormalize.

**Check:** Does C++ ConvolutionWithTranspose use different boundary handling?

#### 2.4 FuzzyErosion Neighbor Order

**File:** `butteraugli/src/mask.rs:95-156`

The C++ FuzzyErosion checks neighbors in a specific order. Verify Rust matches exactly.

#### 2.5 MaskToErrorMul Accumulation

**File:** `butteraugli/src/mask.rs:215-219`

```rust
if let Some(ref mut ac) = diff_ac {
    let diff = blurred0.get(x, y) - blurred1.get(x, y);
    let prev = ac.get(x, y);
    ac.set(x, y, prev + MASK_TO_ERROR_MUL * diff * diff);
}
```

This accumulates into the Y channel of block_diff_ac. Verify this matches C++.

#### 2.6 Multiresolution Contribution

**File:** `butteraugli/src/diff.rs:577-579`

```rust
if let Some(sub) = sub_diffmap {
    add_supersampled_2x(&sub, 0.5, &mut diffmap);
}
```

The heuristic mixing might differ from C++.

### Phase 3: Specific Fixes

Based on Phase 1 findings, implement fixes:

#### 3.1 If Malta Filter Diverges

Options:
1. Match C++ precision (f64 intermediates where needed)
2. Match C++ accumulation order
3. Implement SIMD path matching C++ exactly

#### 3.2 If Mask Diverges

Options:
1. Implement mirrored boundary for mask blur (like sigma=1.2 fix)
2. Match C++ DiffPrecompute precision
3. Match FuzzyErosion neighbor sampling exactly

#### 3.3 If Multiresolution Diverges

Options:
1. Match C++ `AddSupersampled2x` exactly
2. Verify subsampling produces identical half-resolution images

### Phase 4: Validation

After each fix:

1. Run `cargo test --features cpp-parity --test reference_parity`
2. Verify no regressions in passing tests
3. Check if failing tests improve

Target: **191/191 tests pass at 20% tolerance**

## Implementation Order

1. **Week 1:** Add Malta FFI and comparison tests
2. **Week 2:** Add Mask FFI and comparison tests
3. **Week 3:** Identify root cause from test results
4. **Week 4:** Implement fix for identified cause
5. **Week 5:** Validation and documentation

## Files to Modify

### FFI Layer (jpegli-cpp)
- `lib/extras/butteraugli_c.h` - Add new function declarations
- `lib/extras/butteraugli_c.cc` - Implement FFI wrappers

### Rust Implementation (butteraugli)
- `src/malta.rs` - Likely fix location
- `src/mask.rs` - Possible fix location
- `src/diff.rs` - Possible fix location
- `tests/step_by_step_comparison.rs` - Add comparison tests

## Success Criteria

- [ ] All 191 reference parity tests pass at 20% tolerance
- [ ] No regression in real image tests (~1.2% divergence maintained)
- [ ] Documentation updated to reflect full parity
- [ ] Performance not significantly degraded

## Notes

The failing patterns all involve blur distortion, which reduces high-frequency content. This suggests the divergence is in how the Malta filter (edge detector) responds to smoothed regions, or how the masking attenuates differences in low-contrast areas.

The consistent pattern of Rust scoring **lower** than C++ indicates Rust is either:
1. Computing smaller Malta filter responses
2. Applying stronger masking (reducing perceived difference)
3. Missing some contribution in the final aggregation
