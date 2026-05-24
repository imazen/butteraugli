//! Strip-tiled Malta + L2-diff kernels — W44-PHASE3-B7d Day 3.
//!
//! Strip-tile variants of:
//! - [`crate::malta::malta_diff_map`]   (halo = 4 rows per side, see [`MALTA_DIFF_MAP_HALO`])
//! - [`crate::diff`]`::l2_diff`           (halo = 0, pure pointwise)
//! - [`crate::diff`]`::l2_diff_write`     (halo = 0, pure pointwise)
//! - [`crate::diff`]`::l2_diff_asymmetric` (halo = 0, pure pointwise)
//! - [`crate::diff`]`::accumulate_two`    (halo = 0, pure pointwise)
//!
//! All variants operate on row-windows ([`StripView`] / [`StripViewMut`]) of an
//! image while preserving the full-buffer arithmetic byte-for-byte.
//!
//! # Why strip-tile?
//!
//! See `~/work/zen/jxl-encoder/docs/RFC_W44_PHASE3_B7D_STRIP_TILE.md` for the
//! design rationale and Day 1 / Day 2 foundation memos.
//!
//! # Byte-identical guarantee
//!
//! These strip variants produce output BIT-IDENTICAL to the corresponding
//! full-buffer kernel on every supported SIMD tier (X64V4 / X64V3 / Neon /
//! Wasm128 / Scalar). Strip-tile is purely a memory-layout optimization; no
//! arithmetic order changes vs the existing kernels.
//!
//! Specifically:
//! - `malta_compute_scaled_diffs` per-row arithmetic is preserved verbatim
//!   (callable from a strip context because the inner loop is pointwise).
//! - The Malta interior reuse the SAME `malta_unit_interior` /
//!   `malta_unit_lf_interior` / `malta_unit_interior_8x_*` /
//!   `malta_unit_interior_16x_v4` functions as the full kernel (now
//!   `pub(crate)` for strip access).
//! - Border handling matches the full kernel's PAD=4 zero-pad behaviour:
//!   parent top/bottom edges contribute zeros for malta windows that hang
//!   off the image; horizontal padding is identical (4 zero cols each side).
//! - The pointwise diff/accum kernels use identical `mul_add` chains.
//!
//! # Halo summary (per kernel)
//!
//! | kernel                  | halo (rows / side) |
//! |---                      |---                 |
//! | `malta_diff_map`        | 4                  |
//! | `l2_diff`               | 0                  |
//! | `l2_diff_write`         | 0                  |
//! | `l2_diff_asymmetric`    | 0                  |
//! | `accumulate_two`        | 0                  |

// archmage::arcane generates wrapper functions that inherit the arg count
// but not the #[allow] attribute (same constraint as `crate::malta`).
#![allow(clippy::too_many_arguments)]

use crate::image::{BufferPool, ImageF, StripView, StripViewMut};
use crate::malta::{malta_unit_interior, malta_unit_lf_interior};

#[cfg(target_arch = "x86_64")]
use crate::malta::{
    malta_unit_interior_8x_v3, malta_unit_interior_16x_v4, malta_unit_lf_interior_8x_v3,
    malta_unit_lf_interior_16x_v4,
};

#[cfg(target_arch = "aarch64")]
use crate::malta::{malta_unit_interior_8x_neon, malta_unit_lf_interior_8x_neon};

#[cfg(target_arch = "wasm32")]
use crate::malta::{malta_unit_interior_8x_wasm128, malta_unit_lf_interior_8x_wasm128};

/// Halo (rows per side) required for the Malta diff map's interior filter.
/// The 9x9 Malta window reaches ±4 pixels both vertically and horizontally,
/// so the strip caller must provide 4 halo rows on each side (clamped to
/// parent edges).
pub const MALTA_DIFF_MAP_HALO: usize = 4;

/// Halo for `l2_diff` and friends — zero, purely pointwise.
pub const L2_DIFF_HALO: usize = 0;

// ============================================================================
// Pointwise l2_diff family — halo = 0
// ============================================================================

/// Strip-tiled L2 difference (symmetric, accumulating).
///
/// Equivalent to [`crate::diff`]`::l2_diff` applied to the strip — accumulates
/// `(i0 - i1)^2 * w` into `diffmap`.
///
/// # Panics
/// Mismatched strip dimensions panic.
#[archmage::autoversion]
pub fn l2_diff_strip(
    _token: archmage::SimdToken,
    i0: &StripView<'_>,
    i1: &StripView<'_>,
    w: f32,
    diffmap: &mut StripViewMut<'_>,
) {
    assert_eq!(
        i0.width(),
        i1.width(),
        "l2_diff_strip: i0/i1 width mismatch"
    );
    assert_eq!(
        i0.height(),
        i1.height(),
        "l2_diff_strip: i0/i1 height mismatch"
    );
    assert_eq!(
        i0.width(),
        diffmap.width(),
        "l2_diff_strip: i0/diffmap width mismatch"
    );
    assert_eq!(
        i0.height(),
        diffmap.height(),
        "l2_diff_strip: i0/diffmap height mismatch"
    );

    let height = i0.height();
    for y in 0..height {
        let row0 = i0.row(y);
        let row1 = i1.row(y);
        let row_diff = diffmap.row_mut(y);

        for ((d, &v0), &v1) in row_diff.iter_mut().zip(row0.iter()).zip(row1.iter()) {
            let diff = v0 - v1;
            *d = (diff * diff).mul_add(w, *d);
        }
    }
}

/// Strip-tiled L2 difference (symmetric, write-only).
///
/// Equivalent to [`crate::diff`]`::l2_diff_write` applied to the strip — writes
/// `(i0 - i1)^2 * w` to `diffmap` (overwrites).
#[archmage::autoversion]
pub fn l2_diff_write_strip(
    _token: archmage::SimdToken,
    i0: &StripView<'_>,
    i1: &StripView<'_>,
    w: f32,
    diffmap: &mut StripViewMut<'_>,
) {
    assert_eq!(
        i0.width(),
        i1.width(),
        "l2_diff_write_strip: i0/i1 width mismatch"
    );
    assert_eq!(
        i0.height(),
        i1.height(),
        "l2_diff_write_strip: i0/i1 height mismatch"
    );
    assert_eq!(
        i0.width(),
        diffmap.width(),
        "l2_diff_write_strip: i0/diffmap width mismatch"
    );
    assert_eq!(
        i0.height(),
        diffmap.height(),
        "l2_diff_write_strip: i0/diffmap height mismatch"
    );

    let height = i0.height();
    for y in 0..height {
        let row0 = i0.row(y);
        let row1 = i1.row(y);
        let row_diff = diffmap.row_mut(y);

        for ((d, &v0), &v1) in row_diff.iter_mut().zip(row0.iter()).zip(row1.iter()) {
            let diff = v0 - v1;
            *d = diff * diff * w;
        }
    }
}

/// Strip-tiled asymmetric L2 difference.
///
/// Equivalent to [`crate::diff`]`::l2_diff_asymmetric` applied to the strip —
/// accumulates an asymmetric penalty (artifact > blur) into `diffmap`.
///
/// Matches the full kernel's short-circuit on `w_0gt1 == 0.0 && w_0lt1 == 0.0`.
#[archmage::autoversion]
pub fn l2_diff_asymmetric_strip(
    _token: archmage::SimdToken,
    i0: &StripView<'_>,
    i1: &StripView<'_>,
    w_0gt1: f32,
    w_0lt1: f32,
    diffmap: &mut StripViewMut<'_>,
) {
    if w_0gt1 == 0.0 && w_0lt1 == 0.0 {
        return;
    }

    assert_eq!(
        i0.width(),
        i1.width(),
        "l2_diff_asymmetric_strip: i0/i1 width mismatch"
    );
    assert_eq!(
        i0.height(),
        i1.height(),
        "l2_diff_asymmetric_strip: i0/i1 height mismatch"
    );
    assert_eq!(
        i0.width(),
        diffmap.width(),
        "l2_diff_asymmetric_strip: i0/diffmap width mismatch"
    );
    assert_eq!(
        i0.height(),
        diffmap.height(),
        "l2_diff_asymmetric_strip: i0/diffmap height mismatch"
    );

    let width = i0.width();
    let height = i0.height();
    let vw_0gt1 = w_0gt1 * 0.8;
    let vw_0lt1 = w_0lt1 * 0.8;

    for y in 0..height {
        let row0 = i0.row(y);
        let row1 = i1.row(y);
        let row_diff = diffmap.row_mut(y);

        for x in 0..width {
            let val0 = row0[x];
            let val1 = row1[x];

            let diff = val0 - val1;
            let total = (diff * diff).mul_add(vw_0gt1, row_diff[x]);

            // Branch-free asymmetric penalty: flip val1 to match val0's sign
            // direction, then clamp. Matches `l2_diff_asymmetric` exactly.
            let fabs0 = val0.abs();
            let too_small = 0.4 * fabs0;
            let sign = 1.0f32.copysign(val0);
            let sv1 = val1 * sign;
            let v = (too_small - sv1).max(0.0) + (sv1 - fabs0).max(0.0);

            row_diff[x] = (v * v).mul_add(vw_0lt1, total);
        }
    }
}

/// Strip-tiled `accumulate_two`.
///
/// Equivalent to [`crate::diff`]`::accumulate_two` applied to the strip —
/// `dst[y][x] += a[y][x] + b[y][x]`.
#[archmage::autoversion]
pub fn accumulate_two_strip(
    _token: archmage::SimdToken,
    a: &StripView<'_>,
    b: &StripView<'_>,
    dst: &mut StripViewMut<'_>,
) {
    assert_eq!(
        a.width(),
        b.width(),
        "accumulate_two_strip: a/b width mismatch"
    );
    assert_eq!(
        a.height(),
        b.height(),
        "accumulate_two_strip: a/b height mismatch"
    );
    assert_eq!(
        a.width(),
        dst.width(),
        "accumulate_two_strip: a/dst width mismatch"
    );
    assert_eq!(
        a.height(),
        dst.height(),
        "accumulate_two_strip: a/dst height mismatch"
    );

    let height = a.height();
    for y in 0..height {
        let ra = a.row(y);
        let rb = b.row(y);
        let rd = dst.row_mut(y);
        for ((d, &va), &vb) in rd.iter_mut().zip(ra.iter()).zip(rb.iter()) {
            *d += va + vb;
        }
    }
}

// ============================================================================
// malta_diff_map_strip — halo = 4 rows per side
// ============================================================================

/// Strip-tiled `malta_diff_map`.
///
/// Computes the asymmetric Malta difference between two luminance images,
/// writing per-output-row results to `output_strip`.
///
/// # Strip semantics
///
/// `lum0_with_halo` and `lum1_with_halo` are halo-extended input strips
/// covering parent rows
/// `[output_strip.start_row_in_parent().saturating_sub(MALTA_DIFF_MAP_HALO),
///  (output_strip.start_row_in_parent() + output_strip.height() + MALTA_DIFF_MAP_HALO).min(parent_height))`.
///
/// `parent_height` is the height of the parent input image (used to compute
/// parent-edge zero-pad behaviour: Malta windows that hang off the parent
/// see zeros, matching `malta_diff_map`'s `PAD` zero-fill).
///
/// Calling `malta_diff_map_strip` with a single strip covering the full image
/// (output start = 0, output height = parent_height, halo strip = full image)
/// produces output BIT-IDENTICAL to [`crate::malta`]`::malta_diff_map`. Splitting
/// into N strips and concatenating outputs produces BIT-IDENTICAL results.
///
/// # Panics
/// - `lum0_with_halo` / `lum1_with_halo` width mismatch with `output_strip`.
/// - `output_strip.start_row_in_parent() + output_strip.height() > parent_height`.
/// - Input strip does not cover the required halo region.
#[allow(clippy::too_many_arguments)]
pub fn malta_diff_map_strip(
    lum0_with_halo: &StripView<'_>,
    lum1_with_halo: &StripView<'_>,
    output_strip: &mut StripViewMut<'_>,
    parent_height: usize,
    w_0gt1: f64,
    w_0lt1: f64,
    norm1: f64,
    use_lf: bool,
    pool: &BufferPool,
) {
    let width = output_strip.width();
    assert_eq!(
        lum0_with_halo.width(),
        width,
        "malta_diff_map_strip: lum0/output width mismatch ({} vs {})",
        lum0_with_halo.width(),
        width
    );
    assert_eq!(
        lum1_with_halo.width(),
        width,
        "malta_diff_map_strip: lum1/output width mismatch ({} vs {})",
        lum1_with_halo.width(),
        width
    );
    assert_eq!(
        lum0_with_halo.start_row_in_parent(),
        lum1_with_halo.start_row_in_parent(),
        "malta_diff_map_strip: lum0/lum1 strips must start at same parent row"
    );
    assert_eq!(
        lum0_with_halo.height(),
        lum1_with_halo.height(),
        "malta_diff_map_strip: lum0/lum1 strips must have same height"
    );

    let out_start = output_strip.start_row_in_parent();
    let out_height = output_strip.height();
    assert!(
        out_start + out_height <= parent_height,
        "malta_diff_map_strip: output strip [{}, {}) exceeds parent_height {}",
        out_start,
        out_start + out_height,
        parent_height
    );

    let in_start = lum0_with_halo.start_row_in_parent();
    let in_height = lum0_with_halo.height();
    let in_end = in_start + in_height;

    // The input strip MUST cover at minimum the halo-extended output range.
    let need_in_start = out_start.saturating_sub(MALTA_DIFF_MAP_HALO);
    let need_in_end = (out_start + out_height + MALTA_DIFF_MAP_HALO).min(parent_height);
    assert!(
        in_start <= need_in_start && in_end >= need_in_end,
        "malta_diff_map_strip: input strip [{in_start}, {in_end}) does not cover required halo \
         range [{need_in_start}, {need_in_end}) (output [{out_start}, {}), halo={MALTA_DIFF_MAP_HALO})",
        out_start + out_height,
    );

    archmage::incant!(
        malta_diff_map_strip_dispatch(
            lum0_with_halo,
            lum1_with_halo,
            output_strip,
            parent_height,
            w_0gt1,
            w_0lt1,
            norm1,
            use_lf,
            pool,
        ),
        [v4, v3, neon, wasm128]
    );
}

// ----------------------------------------------------------------------------
// Shared scaled-diff helper (matches malta::malta_compute_scaled_diffs per-row)
// ----------------------------------------------------------------------------

/// Per-row branch-free scaled diff (matches `malta_compute_scaled_diffs` exactly).
///
/// Writes `out[x] = scaled_diff + sign * impact` for each pixel in the row.
#[inline]
fn compute_scaled_diff_row(
    row0: &[f32],
    row1: &[f32],
    norm2_0gt1: f32,
    norm2_0lt1: f32,
    norm1_f32: f32,
    out: &mut [f32],
) {
    for (o, (&v0, &v1)) in out.iter_mut().zip(row0.iter().zip(row1.iter())) {
        let absval = 0.5 * (v0.abs() + v1.abs());
        let inv_norm = 1.0 / (norm1_f32 + absval);
        let diff = v0 - v1;
        let scaled_diff = norm2_0gt1 * inv_norm * diff;

        // Branch-free asymmetric penalty
        let fabs0 = v0.abs();
        let too_small = 0.55 * fabs0;
        let too_big = 1.05 * fabs0;
        let sign = 1.0f32.copysign(v0);
        let sv1 = v1 * sign;
        let below = (too_small - sv1).max(0.0);
        let above = (sv1 - too_big).max(0.0);
        let impact = norm2_0lt1 * inv_norm * (below - above);

        *o = scaled_diff + sign * impact;
    }
}

/// Builds a strip-local padded diffs buffer.
///
/// Mirrors `malta_diff_map_impl`'s PAD=4 zero-fill behaviour:
///   - parent-edge rows (above row 0 / below row parent_height-1) → zeros
///   - left/right column padding → zeros
///   - interior rows → scaled diff from `compute_scaled_diff_row`
///
/// The returned ImageF has:
///   - `width = parent_width + 2*PAD`
///   - `height = (out_height + 2*PAD)` if interior strip;
///     less if strip is at parent top/bottom (zero rows below the parent edge
///     replace the missing halo).
///
/// Returns the padded ImageF and the row offset within it where output row 0
/// of the strip lives. The Malta interior at output local_y reads rows
/// `[padded_y - PAD ..= padded_y + PAD]` where `padded_y = pad_row_offset + local_y`.
fn build_padded_strip_diffs(
    lum0_with_halo: &StripView<'_>,
    lum1_with_halo: &StripView<'_>,
    out_start: usize,
    out_height: usize,
    parent_height: usize,
    norm2_0gt1: f32,
    norm2_0lt1: f32,
    norm1_f32: f32,
    pool: &BufferPool,
) -> (ImageF, usize) {
    const PAD: usize = MALTA_DIFF_MAP_HALO;
    let width = lum0_with_halo.width();
    let in_start = lum0_with_halo.start_row_in_parent();

    // Halo rows actually available from the input strip (clamped at parent edges).
    // The Malta interior at output parent row P reads parent rows [P-4, P+4].
    // For each parent row in that range that exists, we fill the corresponding
    // padded buffer row. For parent rows outside [0, parent_height), we leave zeros.
    let pad_top = PAD; // PAD rows of (zeros or halo) above the output strip
    let pad_bottom = PAD; // PAD rows of (zeros or halo) below
    let pad_w = width + 2 * PAD;
    let pad_h = out_height + pad_top + pad_bottom;

    let mut padded = ImageF::from_pool_dirty(pad_w, pad_h, pool);
    let pad_stride = padded.stride();

    // Zero ALL rows first (cheap; bulk-fill). The interior rows will then be
    // overwritten with diffs; the leftover top/bottom border + left/right column
    // padding stays at zero, matching the full kernel's PAD zero-fill.
    for y in 0..pad_h {
        padded.row_full_mut(y)[..pad_stride].fill(0.0);
    }

    // Fill padded rows [PAD - top_halo_rows .. PAD + out_height + bottom_halo_rows)
    // with scaled diffs from the corresponding input strip rows.
    let halo_top_rows_avail = out_start.min(PAD); // rows above parent row out_start, capped at PAD
    let halo_bottom_rows_avail = (parent_height - (out_start + out_height)).min(PAD);

    // Iterate parent rows from [out_start - halo_top_rows_avail, out_start + out_height + halo_bottom_rows_avail)
    let pr_first = out_start - halo_top_rows_avail;
    let pr_last_excl = out_start + out_height + halo_bottom_rows_avail;

    for parent_y in pr_first..pr_last_excl {
        debug_assert!(
            parent_y >= in_start && parent_y < in_start + lum0_with_halo.height(),
            "build_padded_strip_diffs: parent_y {parent_y} outside input strip \
             [{in_start}, {})",
            in_start + lum0_with_halo.height()
        );
        let in_local_y = parent_y - in_start;
        let row0 = lum0_with_halo.row(in_local_y);
        let row1 = lum1_with_halo.row(in_local_y);

        // Map parent_y → padded buffer row:
        //   parent row out_start    → padded row PAD
        //   parent row out_start-1  → padded row PAD-1
        //   parent row out_start+H  → padded row PAD+H
        let padded_y = (parent_y as isize - out_start as isize + PAD as isize) as usize;
        let dst_full = padded.row_full_mut(padded_y);

        // Left zero padding [0..PAD) already 0; write interior; right padding already 0.
        compute_scaled_diff_row(
            row0,
            row1,
            norm2_0gt1,
            norm2_0lt1,
            norm1_f32,
            &mut dst_full[PAD..PAD + width],
        );
    }

    (padded, PAD)
}

/// Common impl shared across SIMD dispatches.
#[allow(clippy::too_many_arguments, clippy::inline_always)]
#[inline(always)]
fn malta_diff_map_strip_impl<F>(
    lum0_with_halo: &StripView<'_>,
    lum1_with_halo: &StripView<'_>,
    output_strip: &mut StripViewMut<'_>,
    parent_height: usize,
    w_0gt1: f64,
    w_0lt1: f64,
    norm1: f64,
    use_lf: bool,
    pool: &BufferPool,
    interior_row: F,
) where
    F: Fn(&[f32], usize, usize, usize, bool, &mut [f32]),
{
    // Mirror malta_diff_map_impl's constants exactly.
    const K_WEIGHT0: f64 = 0.5;
    const K_WEIGHT1: f64 = 0.33;
    const LEN: f64 = 3.75;
    let mulli = if use_lf {
        0.611612573796
    } else {
        0.39905817637
    };

    let w_pre0gt1 = mulli * (K_WEIGHT0 * w_0gt1).sqrt() / (LEN * 2.0 + 1.0);
    let w_pre0lt1 = mulli * (K_WEIGHT1 * w_0lt1).sqrt() / (LEN * 2.0 + 1.0);
    let norm2_0gt1 = (w_pre0gt1 * norm1) as f32;
    let norm2_0lt1 = (w_pre0lt1 * norm1) as f32;
    let norm1_f32 = norm1 as f32;

    let out_start = output_strip.start_row_in_parent();
    let out_height = output_strip.height();
    let width = output_strip.width();

    // Build strip-local padded diffs (mirrors malta_diff_map_impl's pad+copy).
    let (padded, pad_row_offset) = build_padded_strip_diffs(
        lum0_with_halo,
        lum1_with_halo,
        out_start,
        out_height,
        parent_height,
        norm2_0gt1,
        norm2_0lt1,
        norm1_f32,
        pool,
    );
    let pad_stride = padded.stride();
    let pad_data = padded.data();
    const PAD_X: usize = MALTA_DIFF_MAP_HALO;

    // Apply Malta interior filter for each output row.
    for local_y in 0..out_height {
        let out = output_strip.row_mut(local_y);
        // padded_y at the output strip's local row `local_y`:
        let padded_y = pad_row_offset + local_y;
        let center_base = padded_y * pad_stride + PAD_X;
        interior_row(pad_data, center_base, pad_stride, width, use_lf, out);
    }

    padded.recycle(pool);
}

#[allow(clippy::too_many_arguments)]
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn malta_diff_map_strip_dispatch_v3(
    token: archmage::X64V3Token,
    lum0_with_halo: &StripView<'_>,
    lum1_with_halo: &StripView<'_>,
    output_strip: &mut StripViewMut<'_>,
    parent_height: usize,
    w_0gt1: f64,
    w_0lt1: f64,
    norm1: f64,
    use_lf: bool,
    pool: &BufferPool,
) {
    let interior = |data: &[f32],
                    center_base: usize,
                    stride: usize,
                    count: usize,
                    use_lf: bool,
                    out: &mut [f32]| {
        let mut x = 0;
        while x + 8 <= count {
            let center = center_base + x;
            let results = if use_lf {
                malta_unit_lf_interior_8x_v3(token, data, center, stride)
            } else {
                malta_unit_interior_8x_v3(token, data, center, stride)
            };
            results.store((&mut out[x..x + 8]).try_into().unwrap());
            x += 8;
        }
        while x < count {
            let center = center_base + x;
            out[x] = if use_lf {
                malta_unit_lf_interior(data, center, stride)
            } else {
                malta_unit_interior(data, center, stride)
            };
            x += 1;
        }
    };

    malta_diff_map_strip_impl(
        lum0_with_halo,
        lum1_with_halo,
        output_strip,
        parent_height,
        w_0gt1,
        w_0lt1,
        norm1,
        use_lf,
        pool,
        interior,
    );
}

#[allow(clippy::too_many_arguments)]
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn malta_diff_map_strip_dispatch_v4(
    token: archmage::X64V4Token,
    lum0_with_halo: &StripView<'_>,
    lum1_with_halo: &StripView<'_>,
    output_strip: &mut StripViewMut<'_>,
    parent_height: usize,
    w_0gt1: f64,
    w_0lt1: f64,
    norm1: f64,
    use_lf: bool,
    pool: &BufferPool,
) {
    let interior = |data: &[f32],
                    center_base: usize,
                    stride: usize,
                    count: usize,
                    use_lf: bool,
                    out: &mut [f32]| {
        let mut x = 0;
        while x + 16 <= count {
            let center = center_base + x;
            let results = if use_lf {
                malta_unit_lf_interior_16x_v4(token, data, center, stride)
            } else {
                malta_unit_interior_16x_v4(token, data, center, stride)
            };
            results.store((&mut out[x..x + 16]).try_into().unwrap());
            x += 16;
        }
        while x < count {
            let center = center_base + x;
            out[x] = if use_lf {
                malta_unit_lf_interior(data, center, stride)
            } else {
                malta_unit_interior(data, center, stride)
            };
            x += 1;
        }
    };

    malta_diff_map_strip_impl(
        lum0_with_halo,
        lum1_with_halo,
        output_strip,
        parent_height,
        w_0gt1,
        w_0lt1,
        norm1,
        use_lf,
        pool,
        interior,
    );
}

#[allow(clippy::too_many_arguments)]
#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn malta_diff_map_strip_dispatch_neon(
    token: archmage::NeonToken,
    lum0_with_halo: &StripView<'_>,
    lum1_with_halo: &StripView<'_>,
    output_strip: &mut StripViewMut<'_>,
    parent_height: usize,
    w_0gt1: f64,
    w_0lt1: f64,
    norm1: f64,
    use_lf: bool,
    pool: &BufferPool,
) {
    let interior = |data: &[f32],
                    center_base: usize,
                    stride: usize,
                    count: usize,
                    use_lf: bool,
                    out: &mut [f32]| {
        let mut x = 0;
        while x + 8 <= count {
            let center = center_base + x;
            let results = if use_lf {
                malta_unit_lf_interior_8x_neon(token, data, center, stride)
            } else {
                malta_unit_interior_8x_neon(token, data, center, stride)
            };
            results.store((&mut out[x..x + 8]).try_into().unwrap());
            x += 8;
        }
        while x < count {
            let center = center_base + x;
            out[x] = if use_lf {
                malta_unit_lf_interior(data, center, stride)
            } else {
                malta_unit_interior(data, center, stride)
            };
            x += 1;
        }
    };

    malta_diff_map_strip_impl(
        lum0_with_halo,
        lum1_with_halo,
        output_strip,
        parent_height,
        w_0gt1,
        w_0lt1,
        norm1,
        use_lf,
        pool,
        interior,
    );
}

#[allow(clippy::too_many_arguments)]
#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
fn malta_diff_map_strip_dispatch_wasm128(
    token: archmage::Wasm128Token,
    lum0_with_halo: &StripView<'_>,
    lum1_with_halo: &StripView<'_>,
    output_strip: &mut StripViewMut<'_>,
    parent_height: usize,
    w_0gt1: f64,
    w_0lt1: f64,
    norm1: f64,
    use_lf: bool,
    pool: &BufferPool,
) {
    let interior = |data: &[f32],
                    center_base: usize,
                    stride: usize,
                    count: usize,
                    use_lf: bool,
                    out: &mut [f32]| {
        let mut x = 0;
        while x + 8 <= count {
            let center = center_base + x;
            let results = if use_lf {
                malta_unit_lf_interior_8x_wasm128(token, data, center, stride)
            } else {
                malta_unit_interior_8x_wasm128(token, data, center, stride)
            };
            results.store((&mut out[x..x + 8]).try_into().unwrap());
            x += 8;
        }
        while x < count {
            let center = center_base + x;
            out[x] = if use_lf {
                malta_unit_lf_interior(data, center, stride)
            } else {
                malta_unit_interior(data, center, stride)
            };
            x += 1;
        }
    };

    malta_diff_map_strip_impl(
        lum0_with_halo,
        lum1_with_halo,
        output_strip,
        parent_height,
        w_0gt1,
        w_0lt1,
        norm1,
        use_lf,
        pool,
        interior,
    );
}

#[allow(clippy::too_many_arguments)]
fn malta_diff_map_strip_dispatch_scalar(
    _token: archmage::ScalarToken,
    lum0_with_halo: &StripView<'_>,
    lum1_with_halo: &StripView<'_>,
    output_strip: &mut StripViewMut<'_>,
    parent_height: usize,
    w_0gt1: f64,
    w_0lt1: f64,
    norm1: f64,
    use_lf: bool,
    pool: &BufferPool,
) {
    let interior = |data: &[f32],
                    center_base: usize,
                    stride: usize,
                    count: usize,
                    use_lf: bool,
                    out: &mut [f32]| {
        for x in 0..count {
            let center = center_base + x;
            out[x] = if use_lf {
                malta_unit_lf_interior(data, center, stride)
            } else {
                malta_unit_interior(data, center, stride)
            };
        }
    };

    malta_diff_map_strip_impl(
        lum0_with_halo,
        lum1_with_halo,
        output_strip,
        parent_height,
        w_0gt1,
        w_0lt1,
        norm1,
        use_lf,
        pool,
        interior,
    );
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff::tests_support::{
        accumulate_two_full, l2_diff_asymmetric_full, l2_diff_full, l2_diff_write_full,
    };
    use crate::malta::malta_diff_map;

    /// Build a deterministic-content ImageF (pseudo-random based on indices).
    fn build_image(width: usize, height: usize, seed: u32) -> ImageF {
        let mut img = ImageF::new(width, height);
        for y in 0..height {
            for x in 0..width {
                // Hash-like values to exercise all sign + magnitude paths
                let v =
                    (((x as u32).wrapping_mul(2654435761) ^ (y as u32).wrapping_mul(40503) ^ seed)
                        & 0xFFFF) as f32
                        / 65535.0;
                // Center around 0 with some negative values for asymmetry tests
                let centered = v * 4.0 - 2.0;
                img.set(x, y, centered);
            }
        }
        img
    }

    fn assert_bits_eq_strip(actual: &ImageF, expected: &ImageF, what: &str) {
        assert_eq!(actual.width(), expected.width(), "{what}: width mismatch");
        assert_eq!(
            actual.height(),
            expected.height(),
            "{what}: height mismatch"
        );
        for y in 0..actual.height() {
            let a = actual.row(y);
            let e = expected.row(y);
            for x in 0..actual.width() {
                assert_eq!(
                    a[x].to_bits(),
                    e[x].to_bits(),
                    "{what}: pixel ({x},{y}) bits differ (actual {} vs expected {})",
                    a[x],
                    e[x]
                );
            }
        }
    }

    fn run_malta_full(lum0: &ImageF, lum1: &ImageF, use_lf: bool, pool: &BufferPool) -> ImageF {
        malta_diff_map(lum0, lum1, 1.0, 1.0, 1.0, use_lf, pool)
    }

    /// Run malta_diff_map_strip with a single strip covering the entire image.
    fn run_malta_single_strip(
        lum0: &ImageF,
        lum1: &ImageF,
        use_lf: bool,
        pool: &BufferPool,
    ) -> ImageF {
        let width = lum0.width();
        let height = lum0.height();
        let mut output = ImageF::new(width, height);
        {
            let in0 = lum0.strip_view(0, height);
            let in1 = lum1.strip_view(0, height);
            let mut out_strip = output.strip_view_mut(0, height);
            malta_diff_map_strip(
                &in0,
                &in1,
                &mut out_strip,
                height,
                1.0,
                1.0,
                1.0,
                use_lf,
                pool,
            );
        }
        output
    }

    /// Run malta_diff_map_strip in N row-strips, concatenating output.
    fn run_malta_n_strips(
        lum0: &ImageF,
        lum1: &ImageF,
        use_lf: bool,
        strip_rows: usize,
        pool: &BufferPool,
    ) -> ImageF {
        let width = lum0.width();
        let height = lum0.height();
        let mut output = ImageF::new(width, height);
        let mut out_start = 0;
        while out_start < height {
            let out_end = (out_start + strip_rows).min(height);
            let in_start = out_start.saturating_sub(MALTA_DIFF_MAP_HALO);
            let in_end = (out_end + MALTA_DIFF_MAP_HALO).min(height);

            let in0 = lum0.strip_view(in_start, in_end);
            let in1 = lum1.strip_view(in_start, in_end);
            let mut out_strip = output.strip_view_mut(out_start, out_end);
            malta_diff_map_strip(
                &in0,
                &in1,
                &mut out_strip,
                height,
                1.0,
                1.0,
                1.0,
                use_lf,
                pool,
            );

            out_start = out_end;
        }
        output
    }

    // ---- Test 1: identity (single full strip = full kernel) ----

    #[test]
    fn test_malta_diff_map_strip_identity_full_hf() {
        let lum0 = build_image(64, 64, 0xAA);
        let lum1 = build_image(64, 64, 0x55);
        let pool = BufferPool::new();
        let expected = run_malta_full(&lum0, &lum1, false, &pool);
        let actual = run_malta_single_strip(&lum0, &lum1, false, &pool);
        assert_bits_eq_strip(&actual, &expected, "malta HF single-strip identity");
    }

    #[test]
    fn test_malta_diff_map_strip_identity_full_lf() {
        let lum0 = build_image(64, 64, 0xCAFE);
        let lum1 = build_image(64, 64, 0xBEEF);
        let pool = BufferPool::new();
        let expected = run_malta_full(&lum0, &lum1, true, &pool);
        let actual = run_malta_single_strip(&lum0, &lum1, true, &pool);
        assert_bits_eq_strip(&actual, &expected, "malta LF single-strip identity");
    }

    // ---- Test 2: 2-strip split matches full ----

    #[test]
    fn test_malta_diff_map_strip_two_split_hf() {
        let lum0 = build_image(128, 128, 0x1234);
        let lum1 = build_image(128, 128, 0x9876);
        let pool = BufferPool::new();
        let expected = run_malta_full(&lum0, &lum1, false, &pool);
        let actual = run_malta_n_strips(&lum0, &lum1, false, 64, &pool);
        assert_bits_eq_strip(&actual, &expected, "malta HF 2-strip split");
    }

    #[test]
    fn test_malta_diff_map_strip_two_split_lf() {
        let lum0 = build_image(128, 128, 0xACE0);
        let lum1 = build_image(128, 128, 0x0F0F);
        let pool = BufferPool::new();
        let expected = run_malta_full(&lum0, &lum1, true, &pool);
        let actual = run_malta_n_strips(&lum0, &lum1, true, 64, &pool);
        assert_bits_eq_strip(&actual, &expected, "malta LF 2-strip split");
    }

    // ---- Test 3: many-strip (16-row strips) matches full ----

    #[test]
    fn test_malta_diff_map_strip_many_strip_hf() {
        let lum0 = build_image(64, 256, 0xDEAD);
        let lum1 = build_image(64, 256, 0xBEEF);
        let pool = BufferPool::new();
        let expected = run_malta_full(&lum0, &lum1, false, &pool);
        let actual = run_malta_n_strips(&lum0, &lum1, false, 16, &pool);
        assert_bits_eq_strip(&actual, &expected, "malta HF 16-strip many-split");
    }

    #[test]
    fn test_malta_diff_map_strip_many_strip_lf() {
        let lum0 = build_image(64, 256, 0xFADE);
        let lum1 = build_image(64, 256, 0xC0DE);
        let pool = BufferPool::new();
        let expected = run_malta_full(&lum0, &lum1, true, &pool);
        let actual = run_malta_n_strips(&lum0, &lum1, true, 16, &pool);
        assert_bits_eq_strip(&actual, &expected, "malta LF 16-strip many-split");
    }

    // ---- Test 4: edge strip (top + bottom strip together) matches full ----

    #[test]
    fn test_malta_diff_map_strip_edge_strips() {
        // 32-row image split into two strips at row 16. Top strip (0..16) has no top halo
        // (clamped to parent edge with zeros); bottom strip (16..32) has no bottom halo.
        let lum0 = build_image(48, 32, 0x12345678);
        let lum1 = build_image(48, 32, 0x87654321);
        let pool = BufferPool::new();
        let expected = run_malta_full(&lum0, &lum1, false, &pool);
        let actual = run_malta_n_strips(&lum0, &lum1, false, 16, &pool);
        assert_bits_eq_strip(&actual, &expected, "malta edge-strip split at row 16");
    }

    // ---- Test 5: stride preservation (image with stride > width) ----

    #[test]
    fn test_malta_diff_map_strip_stride_preserved() {
        // ImageF::new pads stride to a multiple of 16 for SIMD; width=37 produces
        // stride=48 (37 padded up to next 16-multiple).
        let lum0 = build_image(37, 33, 0xABCD);
        let lum1 = build_image(37, 33, 0x4321);
        assert!(
            lum0.stride() >= lum0.width(),
            "test setup: stride must be ≥ width"
        );
        let pool = BufferPool::new();
        let expected = run_malta_full(&lum0, &lum1, false, &pool);
        let actual = run_malta_n_strips(&lum0, &lum1, false, 11, &pool);
        assert_bits_eq_strip(&actual, &expected, "malta strided-input split");

        // Also assert output stride is preserved (output ImageF::new for both paths).
        assert_eq!(actual.stride(), expected.stride(), "output stride mismatch");
    }

    // ---- l2_diff family tests ----

    fn run_l2_full<F>(i0: &ImageF, i1: &ImageF, init_val: f32, full: F) -> ImageF
    where
        F: Fn(&ImageF, &ImageF, &mut ImageF),
    {
        let mut diffmap = ImageF::filled(i0.width(), i0.height(), init_val);
        full(i0, i1, &mut diffmap);
        diffmap
    }

    fn run_l2_n_strips<F>(
        i0: &ImageF,
        i1: &ImageF,
        init_val: f32,
        strip_rows: usize,
        per_strip: F,
    ) -> ImageF
    where
        F: Fn(&StripView<'_>, &StripView<'_>, &mut StripViewMut<'_>),
    {
        let width = i0.width();
        let height = i0.height();
        let mut diffmap = ImageF::filled(width, height, init_val);

        let mut s = 0;
        while s < height {
            let e = (s + strip_rows).min(height);
            let in0 = i0.strip_view(s, e);
            let in1 = i1.strip_view(s, e);
            let mut out = diffmap.strip_view_mut(s, e);
            per_strip(&in0, &in1, &mut out);
            s = e;
        }
        diffmap
    }

    #[test]
    fn test_l2_diff_strip_identity_full() {
        let i0 = build_image(32, 32, 0xAB);
        let i1 = build_image(32, 32, 0xCD);
        let expected = run_l2_full(&i0, &i1, 0.25, |a, b, d| l2_diff_full(a, b, 2.5, d));
        let actual = run_l2_n_strips(&i0, &i1, 0.25, 32, |a, b, d| l2_diff_strip(a, b, 2.5, d));
        assert_bits_eq_strip(&actual, &expected, "l2_diff single-strip identity");
    }

    #[test]
    fn test_l2_diff_strip_n_split() {
        let i0 = build_image(48, 64, 0x10);
        let i1 = build_image(48, 64, 0x20);
        let expected = run_l2_full(&i0, &i1, 0.0, |a, b, d| l2_diff_full(a, b, 1.7, d));
        let actual = run_l2_n_strips(&i0, &i1, 0.0, 8, |a, b, d| l2_diff_strip(a, b, 1.7, d));
        assert_bits_eq_strip(&actual, &expected, "l2_diff 8-row strip split");
    }

    #[test]
    fn test_l2_diff_write_strip_n_split() {
        let i0 = build_image(40, 40, 0xFE);
        let i1 = build_image(40, 40, 0xED);
        let expected = run_l2_full(&i0, &i1, 99.0, |a, b, d| l2_diff_write_full(a, b, 0.5, d));
        let actual = run_l2_n_strips(&i0, &i1, 99.0, 10, |a, b, d| {
            l2_diff_write_strip(a, b, 0.5, d);
        });
        assert_bits_eq_strip(&actual, &expected, "l2_diff_write 10-row strip split");
    }

    #[test]
    fn test_l2_diff_asymmetric_strip_n_split() {
        let i0 = build_image(32, 48, 0x77);
        let i1 = build_image(32, 48, 0x88);
        let expected = run_l2_full(&i0, &i1, 0.125, |a, b, d| {
            l2_diff_asymmetric_full(a, b, 1.3, 0.6, d);
        });
        let actual = run_l2_n_strips(&i0, &i1, 0.125, 12, |a, b, d| {
            l2_diff_asymmetric_strip(a, b, 1.3, 0.6, d);
        });
        assert_bits_eq_strip(&actual, &expected, "l2_diff_asymmetric 12-row strip split");
    }

    #[test]
    fn test_l2_diff_asymmetric_strip_zero_short_circuit() {
        let i0 = build_image(16, 16, 0x01);
        let i1 = build_image(16, 16, 0x02);
        // w_0gt1 == 0.0 && w_0lt1 == 0.0 → short-circuits without touching diffmap
        let init: f32 = core::f32::consts::PI; // arbitrary nonzero init
        let expected = run_l2_full(&i0, &i1, init, |a, b, d| {
            l2_diff_asymmetric_full(a, b, 0.0, 0.0, d);
        });
        let actual = run_l2_n_strips(&i0, &i1, init, 4, |a, b, d| {
            l2_diff_asymmetric_strip(a, b, 0.0, 0.0, d);
        });
        assert_bits_eq_strip(&actual, &expected, "l2_diff_asymmetric zero short-circuit");
        // Sanity: result is just the init value (no writes).
        for y in 0..actual.height() {
            for x in 0..actual.width() {
                assert_eq!(actual.row(y)[x].to_bits(), init.to_bits());
            }
        }
    }

    #[test]
    fn test_accumulate_two_strip_n_split() {
        let a = build_image(32, 32, 0x11);
        let b = build_image(32, 32, 0x22);
        let expected = run_l2_full(&a, &b, 0.5, accumulate_two_full);
        let actual = run_l2_n_strips(&a, &b, 0.5, 4, accumulate_two_strip);
        assert_bits_eq_strip(&actual, &expected, "accumulate_two 4-row strip split");
    }

    // ---- Test: stride preservation for pointwise kernels ----

    #[test]
    fn test_l2_diff_strip_stride_preserved() {
        let i0 = build_image(37, 31, 0xAA);
        let i1 = build_image(37, 31, 0x55);
        assert!(i0.stride() >= i0.width());
        let expected = run_l2_full(&i0, &i1, 0.0, |a, b, d| l2_diff_full(a, b, 1.0, d));
        let actual = run_l2_n_strips(&i0, &i1, 0.0, 5, |a, b, d| l2_diff_strip(a, b, 1.0, d));
        assert_bits_eq_strip(&actual, &expected, "l2_diff strided-input split");
        assert_eq!(actual.stride(), expected.stride(), "stride mismatch");
    }
}
