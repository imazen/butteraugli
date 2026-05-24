//! Strip-tiled Gaussian blur — W44-PHASE3-B7d Day 2.
//!
//! Strip-tile variants of [`blur::gaussian_blur`] and [`blur::blur_mirrored_5x5`]
//! that operate on row-windows of an image while preserving the full-buffer
//! arithmetic byte-for-byte.
//!
//! # Why strip-tile?
//!
//! See `~/work/zen/jxl-encoder/docs/RFC_W44_PHASE3_B7D_STRIP_TILE.md` for the
//! design rationale. In short: the full pipeline allocates ~12 MB plane buffers
//! per stage at 1024², blowing through Zen 4's 32-MB-per-CCD L3. Strip-tiling
//! decomposes each pass into N-row strips whose per-stage working set fits L2.
//!
//! # API
//!
//! Each kernel is split into separable horizontal + vertical phases (matching
//! the existing [`blur::gaussian_blur`] separable structure). The caller is
//! responsible for sizing the input strip to include the kernel halo for the
//! vertical phase; the horizontal phase has no vertical halo and accepts the
//! same input/output strip dimensions.
//!
//! # Byte-identical guarantee
//!
//! These strip variants produce output BIT-IDENTICAL to the corresponding
//! full-buffer kernel on every supported SIMD tier (X64V4 / X64V3 / Neon /
//! Wasm128 / Scalar). Strip-tile is purely a memory-layout optimization; no
//! arithmetic order changes.
//!
//! Specifically:
//! - The horizontal SIMD-FMA chain over `scaled_kernel` is preserved.
//! - The vertical register-accumulate chain (mul, then mul_add for each
//!   subsequent kernel row) is preserved.
//! - Border re-normalization (clamp-to-edge) is computed against the PARENT
//!   image height, not the strip height.
//! - For [`blur_mirrored_5x5_v_strip`], mirroring is at parent edges.

use crate::image::{StripView, StripViewMut};

/// Maximum possible kernel size: 2 * floor(M * sigma_max) + 1 with M=2.25 and
/// SIGMA_LF ≈ 7.156: 2 * 16 + 1 = 33. Keep matched to [`blur::MAX_KERNEL_SIZE`].
const MAX_KERNEL_SIZE: usize = 64;

/// Computes a 1D Gaussian kernel into a stack-allocated buffer.
/// Returns the used portion. Identical to `blur::compute_kernel_stack`.
#[inline]
fn compute_kernel_stack(sigma: f32, buf: &mut [f32; MAX_KERNEL_SIZE]) -> &[f32] {
    const M: f32 = 2.25;
    let scaler = -1.0 / (2.0 * sigma * sigma);
    let diff = (M * sigma.abs()).max(1.0) as i32;
    let size = (2 * diff + 1) as usize;
    debug_assert!(size <= MAX_KERNEL_SIZE);

    for i in -diff..=diff {
        let weight = (scaler * (i * i) as f32).exp();
        buf[(i + diff) as usize] = weight;
    }

    &buf[..size]
}

/// Normalize kernel so weights sum to 1. Identical to `blur::compute_scaled_kernel`.
#[inline]
fn compute_scaled_kernel<'a>(kernel: &[f32], buf: &'a mut [f32; MAX_KERNEL_SIZE]) -> &'a [f32] {
    debug_assert!(kernel.len() <= MAX_KERNEL_SIZE);
    let weight: f32 = kernel.iter().sum();
    let inv_weight = 1.0 / weight;
    for (i, &k) in kernel.iter().enumerate() {
        buf[i] = k * inv_weight;
    }
    &buf[..kernel.len()]
}

/// Halo (rows on each side) required for a Gaussian blur at the given sigma.
///
/// For sigma ≤ 0 the halo is 0 (blur is a no-op). For positive sigma, halo
/// equals `diff = (M * sigma).max(1.0) as i32` (truncating cast, matching
/// libjxl's `BlurAttribs::diff` computation in [`compute_kernel_stack`]).
/// Kernel size = `2 * halo + 1`; kernel half-width = halo. SIGMA_LF = 7.156 →
/// halo = 16.
#[must_use]
pub fn gaussian_blur_halo(sigma: f32) -> usize {
    if sigma <= 0.0 {
        return 0;
    }
    const M: f32 = 2.25;
    let diff = (M * sigma.abs()).max(1.0) as i32;
    diff as usize
}

/// Halo for the 5x5 mirrored blur — fixed at 2 rows per side.
pub const BLUR_MIRRORED_5X5_HALO: usize = 2;

// ============================================================================
// Helpers — horizontal pass (per-row arithmetic)
// ============================================================================

/// Per-row horizontal border + interior fallback (matches
/// `blur::convolve_horizontal_borders` row inner loop arithmetic).
///
/// Operates on a single row pair `(row_in, row_out)`. The interior loop runs
/// when `include_interior` is true (scalar fallback path).
#[allow(clippy::too_many_arguments)]
#[inline]
fn convolve_horizontal_row_borders(
    row_in: &[f32],
    row_out: &mut [f32],
    kernel: &[f32],
    scaled_kernel: &[f32],
    border_ratio: f32,
    width: usize,
    half: usize,
    include_interior: bool,
) {
    let weight_no_border: f32 = kernel.iter().sum();
    let border1 = half.min(width);
    let border2 = if width > half { width - half } else { 0 };

    // Left border
    for x in 0..border1 {
        let minx = x.saturating_sub(half);
        let maxx = (x + half).min(width - 1);
        let k_start = minx + half - x;
        let k_end = maxx + half - x + 1;
        let ks = &kernel[k_start..k_end];
        let weight: f32 = ks.iter().sum();
        let effective = (1.0 - border_ratio) * weight + border_ratio * weight_no_border;
        let scale = 1.0 / effective;
        let sum: f32 = row_in[minx..minx + ks.len()]
            .iter()
            .zip(ks)
            .map(|(&r, &k)| r * k)
            .sum::<f32>()
            * scale;
        row_out[x] = sum;
    }

    if include_interior {
        for x in border1..border2 {
            let d = x - half;
            let base = &row_in[d..d + scaled_kernel.len()];
            let sum: f32 = base.iter().zip(scaled_kernel).map(|(&r, &k)| r * k).sum();
            row_out[x] = sum;
        }
    }

    // Right border
    for x in border2..width {
        let minx = x.saturating_sub(half);
        let maxx = (x + half).min(width - 1);
        let k_start = minx + half - x;
        let k_end = maxx + half - x + 1;
        let ks = &kernel[k_start..k_end];
        let weight: f32 = ks.iter().sum();
        let effective = (1.0 - border_ratio) * weight + border_ratio * weight_no_border;
        let scale = 1.0 / effective;
        let sum: f32 = row_in[minx..minx + ks.len()]
            .iter()
            .zip(ks)
            .map(|(&r, &k)| r * k)
            .sum::<f32>()
            * scale;
        row_out[x] = sum;
    }
}

/// Per-row AVX2 interior horizontal SIMD (matches `convolve_horizontal_interior_v3` row loop).
#[cfg(target_arch = "x86_64")]
#[archmage::rite]
fn convolve_horizontal_row_interior_v3(
    token: archmage::X64V3Token,
    row_in: &[f32],
    row_out: &mut [f32],
    scaled_kernel: &[f32],
    border1: usize,
    border2: usize,
    half: usize,
) {
    use magetypes::simd::f32x8;
    let kernel_len = scaled_kernel.len();
    let simd_chunks = (border2 - border1) / 8;
    let simd_end = border1 + simd_chunks * 8;

    for chunk_idx in 0..simd_chunks {
        let x = border1 + chunk_idx * 8;
        let d = x - half;
        let base = &row_in[d..d + kernel_len + 7];
        let mut sum = f32x8::zero(token);

        for (j, &k) in scaled_kernel.iter().enumerate() {
            let loaded = f32x8::from_slice(token, &base[j..]);
            sum = loaded.mul_add(f32x8::splat(token, k), sum);
        }

        let results = sum.to_array();
        row_out[x..x + 8].copy_from_slice(&results);
    }

    for x in simd_end..border2 {
        let d = x - half;
        let base = &row_in[d..d + kernel_len];
        let sum: f32 = base
            .iter()
            .zip(scaled_kernel)
            .fold(0.0f32, |acc, (&r, &k)| r.mul_add(k, acc));
        row_out[x] = sum;
    }
}

/// Per-row AVX-512 interior horizontal SIMD.
#[cfg(target_arch = "x86_64")]
#[archmage::rite]
fn convolve_horizontal_row_interior_v4(
    token: archmage::X64V4Token,
    row_in: &[f32],
    row_out: &mut [f32],
    scaled_kernel: &[f32],
    border1: usize,
    border2: usize,
    half: usize,
) {
    use magetypes::simd::v4::f32x16;
    let kernel_len = scaled_kernel.len();
    let simd_chunks = (border2 - border1) / 16;
    let simd_end = border1 + simd_chunks * 16;

    for chunk_idx in 0..simd_chunks {
        let x = border1 + chunk_idx * 16;
        let d = x - half;
        let base = &row_in[d..d + kernel_len + 15];
        let mut sum = f32x16::zero(token);

        for (j, &k) in scaled_kernel.iter().enumerate() {
            let loaded = f32x16::from_slice(token, &base[j..]);
            sum = loaded.mul_add(f32x16::splat(token, k), sum);
        }

        let results = sum.to_array();
        row_out[x..x + 16].copy_from_slice(&results);
    }

    for x in simd_end..border2 {
        let d = x - half;
        let base = &row_in[d..d + kernel_len];
        let sum: f32 = base
            .iter()
            .zip(scaled_kernel)
            .fold(0.0f32, |acc, (&r, &k)| r.mul_add(k, acc));
        row_out[x] = sum;
    }
}

/// Per-row NEON interior horizontal SIMD.
#[cfg(target_arch = "aarch64")]
#[archmage::rite]
fn convolve_horizontal_row_interior_neon(
    token: archmage::NeonToken,
    row_in: &[f32],
    row_out: &mut [f32],
    scaled_kernel: &[f32],
    border1: usize,
    border2: usize,
    half: usize,
) {
    use magetypes::simd::f32x8;
    let kernel_len = scaled_kernel.len();
    let simd_chunks = (border2 - border1) / 8;
    let simd_end = border1 + simd_chunks * 8;

    for chunk_idx in 0..simd_chunks {
        let x = border1 + chunk_idx * 8;
        let d = x - half;
        let base = &row_in[d..d + kernel_len + 7];
        let mut sum = f32x8::zero(token);

        for (j, &k) in scaled_kernel.iter().enumerate() {
            let loaded = f32x8::load(token, (&base[j..j + 8]).try_into().unwrap());
            sum = loaded.mul_add(f32x8::splat(token, k), sum);
        }

        sum.store((&mut row_out[x..x + 8]).try_into().unwrap());
    }

    for x in simd_end..border2 {
        let d = x - half;
        let base = &row_in[d..d + kernel_len];
        let sum: f32 = base
            .iter()
            .zip(scaled_kernel)
            .fold(0.0f32, |acc, (&r, &k)| r.mul_add(k, acc));
        row_out[x] = sum;
    }
}

/// Per-row WASM SIMD128 interior horizontal SIMD.
#[cfg(target_arch = "wasm32")]
#[archmage::rite]
fn convolve_horizontal_row_interior_wasm128(
    token: archmage::Wasm128Token,
    row_in: &[f32],
    row_out: &mut [f32],
    scaled_kernel: &[f32],
    border1: usize,
    border2: usize,
    half: usize,
) {
    use magetypes::simd::f32x8;
    let kernel_len = scaled_kernel.len();
    let simd_chunks = (border2 - border1) / 8;
    let simd_end = border1 + simd_chunks * 8;

    for chunk_idx in 0..simd_chunks {
        let x = border1 + chunk_idx * 8;
        let d = x - half;
        let base = &row_in[d..d + kernel_len + 7];
        let mut sum = f32x8::zero(token);

        for (j, &k) in scaled_kernel.iter().enumerate() {
            let loaded = f32x8::load(token, (&base[j..j + 8]).try_into().unwrap());
            sum = loaded.mul_add(f32x8::splat(token, k), sum);
        }

        sum.store((&mut row_out[x..x + 8]).try_into().unwrap());
    }

    for x in simd_end..border2 {
        let d = x - half;
        let base = &row_in[d..d + kernel_len];
        let sum: f32 = base
            .iter()
            .zip(scaled_kernel)
            .fold(0.0f32, |acc, (&r, &k)| r.mul_add(k, acc));
        row_out[x] = sum;
    }
}

// ============================================================================
// gaussian_blur_h_strip — horizontal phase
// ============================================================================

/// Strip-tiled horizontal Gaussian blur (separable phase 1 of 2).
///
/// Operates row-by-row on the input strip, writing to the output strip with
/// identical bounds. No vertical halo required — H is purely per-row.
///
/// The output's logical region is the full `output.height()` rows;
/// `input.height()` must equal `output.height()`.
///
/// # Panics
/// - `input.width() != output.width()` or `input.height() != output.height()`.
///
/// # Byte-identical with full-buffer
///
/// Calling `gaussian_blur_h_strip` on the entire image (strip = full image)
/// produces output BIT-IDENTICAL to the H phase of [`blur::gaussian_blur`].
/// Splitting the image into N strips and concatenating outputs produces
/// BIT-IDENTICAL results to the single-strip call.
pub fn gaussian_blur_h_strip(input: &StripView<'_>, output: &mut StripViewMut<'_>, sigma: f32) {
    assert_eq!(
        input.width(),
        output.width(),
        "gaussian_blur_h_strip: width mismatch (in={}, out={})",
        input.width(),
        output.width()
    );
    assert_eq!(
        input.height(),
        output.height(),
        "gaussian_blur_h_strip: height mismatch (in={}, out={}) — H is per-row, dims must match",
        input.height(),
        output.height()
    );
    if sigma <= 0.0 {
        // Copy input → output verbatim
        for y in 0..output.height() {
            let src = input.row(y);
            output.row_mut(y).copy_from_slice(src);
        }
        return;
    }

    archmage::incant!(
        gaussian_blur_h_strip_dispatch(input, output, sigma),
        [v4, v3, neon, wasm128]
    );
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn gaussian_blur_h_strip_dispatch_v4(
    token: archmage::X64V4Token,
    input: &StripView<'_>,
    output: &mut StripViewMut<'_>,
    sigma: f32,
) {
    let mut kernel_buf = [0.0f32; MAX_KERNEL_SIZE];
    let kernel = compute_kernel_stack(sigma, &mut kernel_buf);
    let half = kernel.len() / 2;
    let mut scaled_buf = [0.0f32; MAX_KERNEL_SIZE];
    let scaled = compute_scaled_kernel(kernel, &mut scaled_buf);
    let width = input.width();
    let border1 = half.min(width);
    let border2 = if width > half { width - half } else { 0 };

    for y in 0..output.height() {
        let row_in = input.row(y);
        let row_out = output.row_mut(y);
        convolve_horizontal_row_borders(row_in, row_out, kernel, scaled, 0.0, width, half, false);
        if border2 > border1 {
            convolve_horizontal_row_interior_v4(
                token, row_in, row_out, scaled, border1, border2, half,
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn gaussian_blur_h_strip_dispatch_v3(
    token: archmage::X64V3Token,
    input: &StripView<'_>,
    output: &mut StripViewMut<'_>,
    sigma: f32,
) {
    let mut kernel_buf = [0.0f32; MAX_KERNEL_SIZE];
    let kernel = compute_kernel_stack(sigma, &mut kernel_buf);
    let half = kernel.len() / 2;
    let mut scaled_buf = [0.0f32; MAX_KERNEL_SIZE];
    let scaled = compute_scaled_kernel(kernel, &mut scaled_buf);
    let width = input.width();
    let border1 = half.min(width);
    let border2 = if width > half { width - half } else { 0 };

    for y in 0..output.height() {
        let row_in = input.row(y);
        let row_out = output.row_mut(y);
        convolve_horizontal_row_borders(row_in, row_out, kernel, scaled, 0.0, width, half, false);
        if border2 > border1 {
            convolve_horizontal_row_interior_v3(
                token, row_in, row_out, scaled, border1, border2, half,
            );
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn gaussian_blur_h_strip_dispatch_neon(
    token: archmage::NeonToken,
    input: &StripView<'_>,
    output: &mut StripViewMut<'_>,
    sigma: f32,
) {
    let mut kernel_buf = [0.0f32; MAX_KERNEL_SIZE];
    let kernel = compute_kernel_stack(sigma, &mut kernel_buf);
    let half = kernel.len() / 2;
    let mut scaled_buf = [0.0f32; MAX_KERNEL_SIZE];
    let scaled = compute_scaled_kernel(kernel, &mut scaled_buf);
    let width = input.width();
    let border1 = half.min(width);
    let border2 = if width > half { width - half } else { 0 };

    for y in 0..output.height() {
        let row_in = input.row(y);
        let row_out = output.row_mut(y);
        convolve_horizontal_row_borders(row_in, row_out, kernel, scaled, 0.0, width, half, false);
        if border2 > border1 {
            convolve_horizontal_row_interior_neon(
                token, row_in, row_out, scaled, border1, border2, half,
            );
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
fn gaussian_blur_h_strip_dispatch_wasm128(
    token: archmage::Wasm128Token,
    input: &StripView<'_>,
    output: &mut StripViewMut<'_>,
    sigma: f32,
) {
    let mut kernel_buf = [0.0f32; MAX_KERNEL_SIZE];
    let kernel = compute_kernel_stack(sigma, &mut kernel_buf);
    let half = kernel.len() / 2;
    let mut scaled_buf = [0.0f32; MAX_KERNEL_SIZE];
    let scaled = compute_scaled_kernel(kernel, &mut scaled_buf);
    let width = input.width();
    let border1 = half.min(width);
    let border2 = if width > half { width - half } else { 0 };

    for y in 0..output.height() {
        let row_in = input.row(y);
        let row_out = output.row_mut(y);
        convolve_horizontal_row_borders(row_in, row_out, kernel, scaled, 0.0, width, half, false);
        if border2 > border1 {
            convolve_horizontal_row_interior_wasm128(
                token, row_in, row_out, scaled, border1, border2, half,
            );
        }
    }
}

fn gaussian_blur_h_strip_dispatch_scalar(
    _token: archmage::ScalarToken,
    input: &StripView<'_>,
    output: &mut StripViewMut<'_>,
    sigma: f32,
) {
    let mut kernel_buf = [0.0f32; MAX_KERNEL_SIZE];
    let kernel = compute_kernel_stack(sigma, &mut kernel_buf);
    let half = kernel.len() / 2;
    let mut scaled_buf = [0.0f32; MAX_KERNEL_SIZE];
    let scaled = compute_scaled_kernel(kernel, &mut scaled_buf);
    let width = input.width();

    for y in 0..output.height() {
        let row_in = input.row(y);
        let row_out = output.row_mut(y);
        convolve_horizontal_row_borders(row_in, row_out, kernel, scaled, 0.0, width, half, true);
    }
}

// ============================================================================
// gaussian_blur_v_strip — vertical phase
// ============================================================================

/// Strip-tiled vertical Gaussian blur (separable phase 2 of 2).
///
/// Operates on the rows of `input_with_halo` corresponding to output rows
/// in `output`, plus halo rows on each side (clamped to parent edges).
///
/// `parent_height` MUST be the height of the original ImageF that the
/// `output` strip is a view into. This is required for clamp-to-edge
/// border handling at parent top/bottom — the strip's local indexing
/// does not preserve where the parent edges live.
///
/// The caller must arrange that `input_with_halo` covers, at minimum:
///
/// ```text
///   input_top    = output.start_row_in_parent().saturating_sub(halo)
///   input_bottom = (output.start_row_in_parent() + output.height() + halo).min(parent_height)
/// ```
///
/// where `halo = gaussian_blur_halo(sigma)`. The strip may carry MORE rows
/// than needed (extra are simply unused).
///
/// # Panics
/// - `input_with_halo.width() != output.width()`.
/// - `output.start_row_in_parent() + output.height() > parent_height`.
/// - Input strip does not cover the required halo region.
///
/// # Byte-identical with full-buffer
///
/// Splitting an image into N strips and chaining
/// `gaussian_blur_v_strip(input_with_halo, output_strip_n, parent_height, sigma)`
/// for each output strip produces output BIT-IDENTICAL to the V phase of
/// [`blur::gaussian_blur`] called on the full image.
pub fn gaussian_blur_v_strip(
    input_with_halo: &StripView<'_>,
    output: &mut StripViewMut<'_>,
    parent_height: usize,
    sigma: f32,
) {
    assert_eq!(
        input_with_halo.width(),
        output.width(),
        "gaussian_blur_v_strip: width mismatch (in={}, out={})",
        input_with_halo.width(),
        output.width()
    );
    let out_start = output.start_row_in_parent();
    let out_height = output.height();
    assert!(
        out_start + out_height <= parent_height,
        "gaussian_blur_v_strip: output strip [{}, {}) exceeds parent_height {}",
        out_start,
        out_start + out_height,
        parent_height
    );

    if sigma <= 0.0 {
        // Pass-through (matches gaussian_blur identity at sigma=0)
        let in_start = input_with_halo.start_row_in_parent();
        let offset = out_start - in_start;
        for y in 0..out_height {
            let src = input_with_halo.row(offset + y);
            output.row_mut(y).copy_from_slice(src);
        }
        return;
    }

    archmage::incant!(
        gaussian_blur_v_strip_dispatch(input_with_halo, output, parent_height, sigma),
        [v4, v3, neon, wasm128]
    );
}

/// Per-output-row vertical border handler (matches `blur::convolve_vertical_*`
/// border-row arithmetic). Used at parent top/bottom where the kernel hangs
/// off the edge and re-normalization applies.
#[allow(clippy::too_many_arguments)]
#[inline]
fn convolve_vertical_border_row_scalar(
    input_with_halo: &StripView<'_>,
    output_row: &mut [f32],
    kernel: &[f32],
    weight_no_border: f32,
    border_ratio: f32,
    parent_y: usize,
    parent_height: usize,
    input_start_in_parent: usize,
    width: usize,
    half: usize,
) {
    let miny_parent = parent_y.saturating_sub(half);
    let maxy_parent = (parent_y + half).min(parent_height - 1);
    let k_start = miny_parent + half - parent_y;
    let k_end = maxy_parent + half - parent_y + 1;
    let ks = &kernel[k_start..k_end];
    let weight: f32 = ks.iter().sum();
    let effective = (1.0 - border_ratio) * weight + border_ratio * weight_no_border;
    let scale = 1.0 / effective;

    for x in 0..width {
        let mut sum = 0.0f32;
        for (ki, &kw) in ks.iter().enumerate() {
            let src_parent_y = miny_parent + ki;
            let src_local_y = src_parent_y - input_start_in_parent;
            sum += input_with_halo.row(src_local_y)[x] * kw;
        }
        output_row[x] = sum * scale;
    }
}

#[cfg(target_arch = "x86_64")]
#[archmage::rite]
#[allow(clippy::too_many_arguments)]
fn convolve_vertical_border_row_v3(
    token: archmage::X64V3Token,
    input_with_halo: &StripView<'_>,
    output_row: &mut [f32],
    kernel: &[f32],
    weight_no_border: f32,
    border_ratio: f32,
    parent_y: usize,
    parent_height: usize,
    input_start_in_parent: usize,
    width: usize,
    half: usize,
) {
    use magetypes::simd::f32x8;
    let simd_width = (width / 8) * 8;
    let miny_parent = parent_y.saturating_sub(half);
    let maxy_parent = (parent_y + half).min(parent_height - 1);
    let k_start = miny_parent + half - parent_y;
    let k_end = maxy_parent + half - parent_y + 1;
    let ks = &kernel[k_start..k_end];
    let weight: f32 = ks.iter().sum();
    let effective = (1.0 - border_ratio) * weight + border_ratio * weight_no_border;
    let scale = 1.0 / effective;

    let mut x = 0;
    while x + 8 <= simd_width {
        let mut sum = f32x8::zero(token);
        for (ki, &kw) in ks.iter().enumerate() {
            let src_parent_y = miny_parent + ki;
            let src_local_y = src_parent_y - input_start_in_parent;
            let row_in = input_with_halo.row(src_local_y);
            let loaded = f32x8::load(token, (&row_in[x..x + 8]).try_into().unwrap());
            sum = loaded.mul_add(f32x8::splat(token, kw * scale), sum);
        }
        sum.store((&mut output_row[x..x + 8]).try_into().unwrap());
        x += 8;
    }
    while x < width {
        let mut sum = 0.0f32;
        for (ki, &kw) in ks.iter().enumerate() {
            let src_parent_y = miny_parent + ki;
            let src_local_y = src_parent_y - input_start_in_parent;
            sum += input_with_halo.row(src_local_y)[x] * kw;
        }
        output_row[x] = sum * scale;
        x += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[archmage::rite]
#[allow(clippy::too_many_arguments)]
fn convolve_vertical_border_row_v4(
    token: archmage::X64V4Token,
    input_with_halo: &StripView<'_>,
    output_row: &mut [f32],
    kernel: &[f32],
    weight_no_border: f32,
    border_ratio: f32,
    parent_y: usize,
    parent_height: usize,
    input_start_in_parent: usize,
    width: usize,
    half: usize,
) {
    use magetypes::simd::v4::f32x16;
    let simd_width = (width / 16) * 16;
    let miny_parent = parent_y.saturating_sub(half);
    let maxy_parent = (parent_y + half).min(parent_height - 1);
    let k_start = miny_parent + half - parent_y;
    let k_end = maxy_parent + half - parent_y + 1;
    let ks = &kernel[k_start..k_end];
    let weight: f32 = ks.iter().sum();
    let effective = (1.0 - border_ratio) * weight + border_ratio * weight_no_border;
    let scale = 1.0 / effective;

    let mut x = 0;
    while x + 16 <= simd_width {
        let mut sum = f32x16::zero(token);
        for (ki, &kw) in ks.iter().enumerate() {
            let src_parent_y = miny_parent + ki;
            let src_local_y = src_parent_y - input_start_in_parent;
            let row_in = input_with_halo.row(src_local_y);
            let loaded = f32x16::from_slice(token, &row_in[x..]);
            sum = loaded.mul_add(f32x16::splat(token, kw * scale), sum);
        }
        let results = sum.to_array();
        output_row[x..x + 16].copy_from_slice(&results);
        x += 16;
    }
    while x < width {
        let mut sum = 0.0f32;
        for (ki, &kw) in ks.iter().enumerate() {
            let src_parent_y = miny_parent + ki;
            let src_local_y = src_parent_y - input_start_in_parent;
            sum += input_with_halo.row(src_local_y)[x] * kw;
        }
        output_row[x] = sum * scale;
        x += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[archmage::rite]
#[allow(clippy::too_many_arguments)]
fn convolve_vertical_border_row_neon(
    token: archmage::NeonToken,
    input_with_halo: &StripView<'_>,
    output_row: &mut [f32],
    kernel: &[f32],
    weight_no_border: f32,
    border_ratio: f32,
    parent_y: usize,
    parent_height: usize,
    input_start_in_parent: usize,
    width: usize,
    half: usize,
) {
    use magetypes::simd::f32x8;
    let simd_width = (width / 8) * 8;
    let miny_parent = parent_y.saturating_sub(half);
    let maxy_parent = (parent_y + half).min(parent_height - 1);
    let k_start = miny_parent + half - parent_y;
    let k_end = maxy_parent + half - parent_y + 1;
    let ks = &kernel[k_start..k_end];
    let weight: f32 = ks.iter().sum();
    let effective = (1.0 - border_ratio) * weight + border_ratio * weight_no_border;
    let scale = 1.0 / effective;

    let mut x = 0;
    while x + 8 <= simd_width {
        let mut sum = f32x8::zero(token);
        for (ki, &kw) in ks.iter().enumerate() {
            let src_parent_y = miny_parent + ki;
            let src_local_y = src_parent_y - input_start_in_parent;
            let row_in = input_with_halo.row(src_local_y);
            let loaded = f32x8::load(token, (&row_in[x..x + 8]).try_into().unwrap());
            sum = loaded.mul_add(f32x8::splat(token, kw * scale), sum);
        }
        sum.store((&mut output_row[x..x + 8]).try_into().unwrap());
        x += 8;
    }
    while x < width {
        let mut sum = 0.0f32;
        for (ki, &kw) in ks.iter().enumerate() {
            let src_parent_y = miny_parent + ki;
            let src_local_y = src_parent_y - input_start_in_parent;
            sum += input_with_halo.row(src_local_y)[x] * kw;
        }
        output_row[x] = sum * scale;
        x += 1;
    }
}

#[cfg(target_arch = "wasm32")]
#[archmage::rite]
#[allow(clippy::too_many_arguments)]
fn convolve_vertical_border_row_wasm128(
    token: archmage::Wasm128Token,
    input_with_halo: &StripView<'_>,
    output_row: &mut [f32],
    kernel: &[f32],
    weight_no_border: f32,
    border_ratio: f32,
    parent_y: usize,
    parent_height: usize,
    input_start_in_parent: usize,
    width: usize,
    half: usize,
) {
    use magetypes::simd::f32x8;
    let simd_width = (width / 8) * 8;
    let miny_parent = parent_y.saturating_sub(half);
    let maxy_parent = (parent_y + half).min(parent_height - 1);
    let k_start = miny_parent + half - parent_y;
    let k_end = maxy_parent + half - parent_y + 1;
    let ks = &kernel[k_start..k_end];
    let weight: f32 = ks.iter().sum();
    let effective = (1.0 - border_ratio) * weight + border_ratio * weight_no_border;
    let scale = 1.0 / effective;

    let mut x = 0;
    while x + 8 <= simd_width {
        let mut sum = f32x8::zero(token);
        for (ki, &kw) in ks.iter().enumerate() {
            let src_parent_y = miny_parent + ki;
            let src_local_y = src_parent_y - input_start_in_parent;
            let row_in = input_with_halo.row(src_local_y);
            let loaded = f32x8::load(token, (&row_in[x..x + 8]).try_into().unwrap());
            sum = loaded.mul_add(f32x8::splat(token, kw * scale), sum);
        }
        sum.store((&mut output_row[x..x + 8]).try_into().unwrap());
        x += 8;
    }
    while x < width {
        let mut sum = 0.0f32;
        for (ki, &kw) in ks.iter().enumerate() {
            let src_parent_y = miny_parent + ki;
            let src_local_y = src_parent_y - input_start_in_parent;
            sum += input_with_halo.row(src_local_y)[x] * kw;
        }
        output_row[x] = sum * scale;
        x += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn gaussian_blur_v_strip_dispatch_v4(
    token: archmage::X64V4Token,
    input_with_halo: &StripView<'_>,
    output: &mut StripViewMut<'_>,
    parent_height: usize,
    sigma: f32,
) {
    use magetypes::simd::v4::f32x16;
    let mut kernel_buf = [0.0f32; MAX_KERNEL_SIZE];
    let kernel = compute_kernel_stack(sigma, &mut kernel_buf);
    let half = kernel.len() / 2;
    let mut scaled_buf = [0.0f32; MAX_KERNEL_SIZE];
    let scaled = compute_scaled_kernel(kernel, &mut scaled_buf);
    let width = input_with_halo.width();
    let in_start = input_with_halo.start_row_in_parent();
    let out_start = output.start_row_in_parent();
    let out_height = output.height();
    let weight_no_border: f32 = kernel.iter().sum();

    let border_top = half.min(parent_height);
    let border_bottom = if parent_height > half {
        parent_height - half
    } else {
        0
    };
    let simd_width = (width / 16) * 16;

    for local_y in 0..out_height {
        let parent_y = out_start + local_y;
        let row_out = output.row_mut(local_y);
        if parent_y < border_top || parent_y >= border_bottom {
            convolve_vertical_border_row_v4(
                token,
                input_with_halo,
                row_out,
                kernel,
                weight_no_border,
                0.0,
                parent_y,
                parent_height,
                in_start,
                width,
                half,
            );
        } else {
            // Interior: register-accumulate using scaled kernel
            let start_y_parent = parent_y - half;
            let start_y_local = start_y_parent - in_start;
            let mut x = 0;
            while x + 16 <= simd_width {
                let row0 = input_with_halo.row(start_y_local);
                let mut sum =
                    f32x16::from_slice(token, &row0[x..]) * f32x16::splat(token, scaled[0]);
                for (ki, &kw) in scaled.iter().enumerate().skip(1) {
                    let row = input_with_halo.row(start_y_local + ki);
                    let loaded = f32x16::from_slice(token, &row[x..]);
                    sum = loaded.mul_add(f32x16::splat(token, kw), sum);
                }
                row_out[x..x + 16].copy_from_slice(&sum.to_array());
                x += 16;
            }
            // Scalar tail
            for x in simd_width..width {
                let mut sum = input_with_halo.row(start_y_local)[x] * scaled[0];
                for (ki, &kw) in scaled.iter().enumerate().skip(1) {
                    sum = input_with_halo.row(start_y_local + ki)[x].mul_add(kw, sum);
                }
                row_out[x] = sum;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn gaussian_blur_v_strip_dispatch_v3(
    token: archmage::X64V3Token,
    input_with_halo: &StripView<'_>,
    output: &mut StripViewMut<'_>,
    parent_height: usize,
    sigma: f32,
) {
    use magetypes::simd::f32x8;
    let mut kernel_buf = [0.0f32; MAX_KERNEL_SIZE];
    let kernel = compute_kernel_stack(sigma, &mut kernel_buf);
    let half = kernel.len() / 2;
    let mut scaled_buf = [0.0f32; MAX_KERNEL_SIZE];
    let scaled = compute_scaled_kernel(kernel, &mut scaled_buf);
    let width = input_with_halo.width();
    let in_start = input_with_halo.start_row_in_parent();
    let out_start = output.start_row_in_parent();
    let out_height = output.height();
    let weight_no_border: f32 = kernel.iter().sum();

    let border_top = half.min(parent_height);
    let border_bottom = if parent_height > half {
        parent_height - half
    } else {
        0
    };
    let simd_width = (width / 8) * 8;

    for local_y in 0..out_height {
        let parent_y = out_start + local_y;
        let row_out = output.row_mut(local_y);
        if parent_y < border_top || parent_y >= border_bottom {
            convolve_vertical_border_row_v3(
                token,
                input_with_halo,
                row_out,
                kernel,
                weight_no_border,
                0.0,
                parent_y,
                parent_height,
                in_start,
                width,
                half,
            );
        } else {
            let start_y_parent = parent_y - half;
            let start_y_local = start_y_parent - in_start;
            let mut x = 0;
            while x + 8 <= simd_width {
                let row0 = input_with_halo.row(start_y_local);
                let mut sum = f32x8::load(token, (&row0[x..x + 8]).try_into().unwrap())
                    * f32x8::splat(token, scaled[0]);
                for (ki, &kw) in scaled.iter().enumerate().skip(1) {
                    let row = input_with_halo.row(start_y_local + ki);
                    let loaded = f32x8::load(token, (&row[x..x + 8]).try_into().unwrap());
                    sum = loaded.mul_add(f32x8::splat(token, kw), sum);
                }
                sum.store((&mut row_out[x..x + 8]).try_into().unwrap());
                x += 8;
            }
            for x in simd_width..width {
                let mut sum = input_with_halo.row(start_y_local)[x] * scaled[0];
                for (ki, &kw) in scaled.iter().enumerate().skip(1) {
                    sum = input_with_halo.row(start_y_local + ki)[x].mul_add(kw, sum);
                }
                row_out[x] = sum;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn gaussian_blur_v_strip_dispatch_neon(
    token: archmage::NeonToken,
    input_with_halo: &StripView<'_>,
    output: &mut StripViewMut<'_>,
    parent_height: usize,
    sigma: f32,
) {
    use magetypes::simd::f32x8;
    let mut kernel_buf = [0.0f32; MAX_KERNEL_SIZE];
    let kernel = compute_kernel_stack(sigma, &mut kernel_buf);
    let half = kernel.len() / 2;
    let mut scaled_buf = [0.0f32; MAX_KERNEL_SIZE];
    let scaled = compute_scaled_kernel(kernel, &mut scaled_buf);
    let width = input_with_halo.width();
    let in_start = input_with_halo.start_row_in_parent();
    let out_start = output.start_row_in_parent();
    let out_height = output.height();
    let weight_no_border: f32 = kernel.iter().sum();

    let border_top = half.min(parent_height);
    let border_bottom = if parent_height > half {
        parent_height - half
    } else {
        0
    };
    let simd_width = (width / 8) * 8;

    for local_y in 0..out_height {
        let parent_y = out_start + local_y;
        let row_out = output.row_mut(local_y);
        if parent_y < border_top || parent_y >= border_bottom {
            convolve_vertical_border_row_neon(
                token,
                input_with_halo,
                row_out,
                kernel,
                weight_no_border,
                0.0,
                parent_y,
                parent_height,
                in_start,
                width,
                half,
            );
        } else {
            let start_y_parent = parent_y - half;
            let start_y_local = start_y_parent - in_start;
            let mut x = 0;
            while x + 8 <= simd_width {
                let row0 = input_with_halo.row(start_y_local);
                let mut sum = f32x8::load(token, (&row0[x..x + 8]).try_into().unwrap())
                    * f32x8::splat(token, scaled[0]);
                for (ki, &kw) in scaled.iter().enumerate().skip(1) {
                    let row = input_with_halo.row(start_y_local + ki);
                    let loaded = f32x8::load(token, (&row[x..x + 8]).try_into().unwrap());
                    sum = loaded.mul_add(f32x8::splat(token, kw), sum);
                }
                sum.store((&mut row_out[x..x + 8]).try_into().unwrap());
                x += 8;
            }
            for x in simd_width..width {
                let mut sum = input_with_halo.row(start_y_local)[x] * scaled[0];
                for (ki, &kw) in scaled.iter().enumerate().skip(1) {
                    sum = input_with_halo.row(start_y_local + ki)[x].mul_add(kw, sum);
                }
                row_out[x] = sum;
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
fn gaussian_blur_v_strip_dispatch_wasm128(
    token: archmage::Wasm128Token,
    input_with_halo: &StripView<'_>,
    output: &mut StripViewMut<'_>,
    parent_height: usize,
    sigma: f32,
) {
    use magetypes::simd::f32x8;
    let mut kernel_buf = [0.0f32; MAX_KERNEL_SIZE];
    let kernel = compute_kernel_stack(sigma, &mut kernel_buf);
    let half = kernel.len() / 2;
    let mut scaled_buf = [0.0f32; MAX_KERNEL_SIZE];
    let scaled = compute_scaled_kernel(kernel, &mut scaled_buf);
    let width = input_with_halo.width();
    let in_start = input_with_halo.start_row_in_parent();
    let out_start = output.start_row_in_parent();
    let out_height = output.height();
    let weight_no_border: f32 = kernel.iter().sum();

    let border_top = half.min(parent_height);
    let border_bottom = if parent_height > half {
        parent_height - half
    } else {
        0
    };
    let simd_width = (width / 8) * 8;

    for local_y in 0..out_height {
        let parent_y = out_start + local_y;
        let row_out = output.row_mut(local_y);
        if parent_y < border_top || parent_y >= border_bottom {
            convolve_vertical_border_row_wasm128(
                token,
                input_with_halo,
                row_out,
                kernel,
                weight_no_border,
                0.0,
                parent_y,
                parent_height,
                in_start,
                width,
                half,
            );
        } else {
            let start_y_parent = parent_y - half;
            let start_y_local = start_y_parent - in_start;
            let mut x = 0;
            while x + 8 <= simd_width {
                let row0 = input_with_halo.row(start_y_local);
                let mut sum = f32x8::load(token, (&row0[x..x + 8]).try_into().unwrap())
                    * f32x8::splat(token, scaled[0]);
                for (ki, &kw) in scaled.iter().enumerate().skip(1) {
                    let row = input_with_halo.row(start_y_local + ki);
                    let loaded = f32x8::load(token, (&row[x..x + 8]).try_into().unwrap());
                    sum = loaded.mul_add(f32x8::splat(token, kw), sum);
                }
                sum.store((&mut row_out[x..x + 8]).try_into().unwrap());
                x += 8;
            }
            for x in simd_width..width {
                let mut sum = input_with_halo.row(start_y_local)[x] * scaled[0];
                for (ki, &kw) in scaled.iter().enumerate().skip(1) {
                    sum = input_with_halo.row(start_y_local + ki)[x].mul_add(kw, sum);
                }
                row_out[x] = sum;
            }
        }
    }
}

fn gaussian_blur_v_strip_dispatch_scalar(
    _token: archmage::ScalarToken,
    input_with_halo: &StripView<'_>,
    output: &mut StripViewMut<'_>,
    parent_height: usize,
    sigma: f32,
) {
    let mut kernel_buf = [0.0f32; MAX_KERNEL_SIZE];
    let kernel = compute_kernel_stack(sigma, &mut kernel_buf);
    let half = kernel.len() / 2;
    let mut scaled_buf = [0.0f32; MAX_KERNEL_SIZE];
    let scaled = compute_scaled_kernel(kernel, &mut scaled_buf);
    let width = input_with_halo.width();
    let in_start = input_with_halo.start_row_in_parent();
    let out_start = output.start_row_in_parent();
    let out_height = output.height();
    let weight_no_border: f32 = kernel.iter().sum();

    let border_top = half.min(parent_height);
    let border_bottom = if parent_height > half {
        parent_height - half
    } else {
        0
    };

    for local_y in 0..out_height {
        let parent_y = out_start + local_y;
        let row_out = output.row_mut(local_y);
        if parent_y < border_top || parent_y >= border_bottom {
            convolve_vertical_border_row_scalar(
                input_with_halo,
                row_out,
                kernel,
                weight_no_border,
                0.0,
                parent_y,
                parent_height,
                in_start,
                width,
                half,
            );
        } else {
            let start_y_parent = parent_y - half;
            let start_y_local = start_y_parent - in_start;
            for x in 0..width {
                let mut sum = input_with_halo.row(start_y_local)[x] * scaled[0];
                for (ki, &kw) in scaled.iter().enumerate().skip(1) {
                    sum = input_with_halo.row(start_y_local + ki)[x].mul_add(kw, sum);
                }
                row_out[x] = sum;
            }
        }
    }
}

// ============================================================================
// blur_mirrored_5x5 — strip-tiled (mirrored boundary, 5x5 kernel)
// ============================================================================

/// Mirror a coordinate at edges. Matches `blur::mirror` exactly.
#[inline]
fn mirror(mut x: i32, size: i32) -> usize {
    while x < 0 || x >= size {
        if x < 0 {
            x = -x - 1;
        } else {
            x = 2 * size - 1 - x;
        }
    }
    x as usize
}

/// Strip-tiled horizontal phase of [`blur::blur_mirrored_5x5`].
///
/// Per-row, mirrored at left/right edges. No vertical halo needed.
///
/// # Panics
/// - `input.width() != output.width()` or `input.height() != output.height()`.
pub fn blur_mirrored_5x5_h_strip(
    input: &StripView<'_>,
    output: &mut StripViewMut<'_>,
    weights: &[f32; 3],
) {
    assert_eq!(
        input.width(),
        output.width(),
        "blur_mirrored_5x5_h_strip: width mismatch"
    );
    assert_eq!(
        input.height(),
        output.height(),
        "blur_mirrored_5x5_h_strip: height mismatch (H is per-row)"
    );
    archmage::incant!(
        blur_mirrored_5x5_h_strip_dispatch(input, output, weights),
        [v4, v3, neon, wasm128]
    );
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn blur_mirrored_5x5_h_strip_dispatch_v4(
    token: archmage::X64V4Token,
    input: &StripView<'_>,
    output: &mut StripViewMut<'_>,
    weights: &[f32; 3],
) {
    use magetypes::simd::v4::f32x16;
    let width = input.width();
    let w0 = weights[0];
    let w1 = weights[1];
    let w2 = weights[2];
    let w0_v = f32x16::splat(token, w0);
    let w1_v = f32x16::splat(token, w1);
    let w2_v = f32x16::splat(token, w2);
    let iwidth = width as i32;
    let border = 2.min(width);
    let interior_end = if width > 4 { width - 2 } else { 0 };

    for y in 0..output.height() {
        let row = input.row(y);
        let out_row = output.row_mut(y);

        for x in 0..border {
            let ix = x as i32;
            let v_m2 = row[mirror(ix - 2, iwidth)];
            let v_m1 = row[mirror(ix - 1, iwidth)];
            let v_0 = row[x];
            let v_p1 = row[mirror(ix + 1, iwidth)];
            let v_p2 = row[mirror(ix + 2, iwidth)];
            out_row[x] = v_0 * w0 + (v_m1 + v_p1) * w1 + (v_m2 + v_p2) * w2;
        }

        let mut x = border;
        while x + 16 <= interior_end {
            let v_m2 = f32x16::load(token, (&row[x - 2..x + 14]).try_into().unwrap());
            let v_m1 = f32x16::load(token, (&row[x - 1..x + 15]).try_into().unwrap());
            let v_0 = f32x16::load(token, (&row[x..x + 16]).try_into().unwrap());
            let v_p1 = f32x16::load(token, (&row[x + 1..x + 17]).try_into().unwrap());
            let v_p2 = f32x16::load(token, (&row[x + 2..x + 18]).try_into().unwrap());
            let sum = v_0 * w0_v + (v_m1 + v_p1) * w1_v + (v_m2 + v_p2) * w2_v;
            sum.store((&mut out_row[x..x + 16]).try_into().unwrap());
            x += 16;
        }
        while x < interior_end {
            let v_m2 = row[x - 2];
            let v_m1 = row[x - 1];
            let v_0 = row[x];
            let v_p1 = row[x + 1];
            let v_p2 = row[x + 2];
            out_row[x] = v_0 * w0 + (v_m1 + v_p1) * w1 + (v_m2 + v_p2) * w2;
            x += 1;
        }

        for x in interior_end..width {
            let ix = x as i32;
            let v_m2 = row[mirror(ix - 2, iwidth)];
            let v_m1 = row[mirror(ix - 1, iwidth)];
            let v_0 = row[x];
            let v_p1 = row[mirror(ix + 1, iwidth)];
            let v_p2 = row[mirror(ix + 2, iwidth)];
            out_row[x] = v_0 * w0 + (v_m1 + v_p1) * w1 + (v_m2 + v_p2) * w2;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn blur_mirrored_5x5_h_strip_dispatch_v3(
    token: archmage::X64V3Token,
    input: &StripView<'_>,
    output: &mut StripViewMut<'_>,
    weights: &[f32; 3],
) {
    use magetypes::simd::f32x8;
    let width = input.width();
    let w0 = weights[0];
    let w1 = weights[1];
    let w2 = weights[2];
    let w0_v = f32x8::splat(token, w0);
    let w1_v = f32x8::splat(token, w1);
    let w2_v = f32x8::splat(token, w2);
    let iwidth = width as i32;
    let border = 2.min(width);
    let interior_end = if width > 4 { width - 2 } else { 0 };

    for y in 0..output.height() {
        let row = input.row(y);
        let out_row = output.row_mut(y);

        for x in 0..border {
            let ix = x as i32;
            let v_m2 = row[mirror(ix - 2, iwidth)];
            let v_m1 = row[mirror(ix - 1, iwidth)];
            let v_0 = row[x];
            let v_p1 = row[mirror(ix + 1, iwidth)];
            let v_p2 = row[mirror(ix + 2, iwidth)];
            out_row[x] = v_0 * w0 + (v_m1 + v_p1) * w1 + (v_m2 + v_p2) * w2;
        }

        let mut x = border;
        while x + 8 <= interior_end {
            let v_m2 = f32x8::load(token, (&row[x - 2..x + 6]).try_into().unwrap());
            let v_m1 = f32x8::load(token, (&row[x - 1..x + 7]).try_into().unwrap());
            let v_0 = f32x8::load(token, (&row[x..x + 8]).try_into().unwrap());
            let v_p1 = f32x8::load(token, (&row[x + 1..x + 9]).try_into().unwrap());
            let v_p2 = f32x8::load(token, (&row[x + 2..x + 10]).try_into().unwrap());
            let sum = v_0 * w0_v + (v_m1 + v_p1) * w1_v + (v_m2 + v_p2) * w2_v;
            sum.store((&mut out_row[x..x + 8]).try_into().unwrap());
            x += 8;
        }
        while x < interior_end {
            let v_m2 = row[x - 2];
            let v_m1 = row[x - 1];
            let v_0 = row[x];
            let v_p1 = row[x + 1];
            let v_p2 = row[x + 2];
            out_row[x] = v_0 * w0 + (v_m1 + v_p1) * w1 + (v_m2 + v_p2) * w2;
            x += 1;
        }

        for x in interior_end..width {
            let ix = x as i32;
            let v_m2 = row[mirror(ix - 2, iwidth)];
            let v_m1 = row[mirror(ix - 1, iwidth)];
            let v_0 = row[x];
            let v_p1 = row[mirror(ix + 1, iwidth)];
            let v_p2 = row[mirror(ix + 2, iwidth)];
            out_row[x] = v_0 * w0 + (v_m1 + v_p1) * w1 + (v_m2 + v_p2) * w2;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn blur_mirrored_5x5_h_strip_dispatch_neon(
    token: archmage::NeonToken,
    input: &StripView<'_>,
    output: &mut StripViewMut<'_>,
    weights: &[f32; 3],
) {
    use magetypes::simd::f32x8;
    let width = input.width();
    let w0 = weights[0];
    let w1 = weights[1];
    let w2 = weights[2];
    let w0_v = f32x8::splat(token, w0);
    let w1_v = f32x8::splat(token, w1);
    let w2_v = f32x8::splat(token, w2);
    let iwidth = width as i32;
    let border = 2.min(width);
    let interior_end = if width > 4 { width - 2 } else { 0 };

    for y in 0..output.height() {
        let row = input.row(y);
        let out_row = output.row_mut(y);

        for x in 0..border {
            let ix = x as i32;
            let v_m2 = row[mirror(ix - 2, iwidth)];
            let v_m1 = row[mirror(ix - 1, iwidth)];
            let v_0 = row[x];
            let v_p1 = row[mirror(ix + 1, iwidth)];
            let v_p2 = row[mirror(ix + 2, iwidth)];
            out_row[x] = v_0 * w0 + (v_m1 + v_p1) * w1 + (v_m2 + v_p2) * w2;
        }

        let mut x = border;
        while x + 8 <= interior_end {
            let v_m2 = f32x8::load(token, (&row[x - 2..x + 6]).try_into().unwrap());
            let v_m1 = f32x8::load(token, (&row[x - 1..x + 7]).try_into().unwrap());
            let v_0 = f32x8::load(token, (&row[x..x + 8]).try_into().unwrap());
            let v_p1 = f32x8::load(token, (&row[x + 1..x + 9]).try_into().unwrap());
            let v_p2 = f32x8::load(token, (&row[x + 2..x + 10]).try_into().unwrap());
            let sum = v_0 * w0_v + (v_m1 + v_p1) * w1_v + (v_m2 + v_p2) * w2_v;
            sum.store((&mut out_row[x..x + 8]).try_into().unwrap());
            x += 8;
        }
        while x < interior_end {
            let v_m2 = row[x - 2];
            let v_m1 = row[x - 1];
            let v_0 = row[x];
            let v_p1 = row[x + 1];
            let v_p2 = row[x + 2];
            out_row[x] = v_0 * w0 + (v_m1 + v_p1) * w1 + (v_m2 + v_p2) * w2;
            x += 1;
        }

        for x in interior_end..width {
            let ix = x as i32;
            let v_m2 = row[mirror(ix - 2, iwidth)];
            let v_m1 = row[mirror(ix - 1, iwidth)];
            let v_0 = row[x];
            let v_p1 = row[mirror(ix + 1, iwidth)];
            let v_p2 = row[mirror(ix + 2, iwidth)];
            out_row[x] = v_0 * w0 + (v_m1 + v_p1) * w1 + (v_m2 + v_p2) * w2;
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
fn blur_mirrored_5x5_h_strip_dispatch_wasm128(
    token: archmage::Wasm128Token,
    input: &StripView<'_>,
    output: &mut StripViewMut<'_>,
    weights: &[f32; 3],
) {
    use magetypes::simd::f32x8;
    let width = input.width();
    let w0 = weights[0];
    let w1 = weights[1];
    let w2 = weights[2];
    let w0_v = f32x8::splat(token, w0);
    let w1_v = f32x8::splat(token, w1);
    let w2_v = f32x8::splat(token, w2);
    let iwidth = width as i32;
    let border = 2.min(width);
    let interior_end = if width > 4 { width - 2 } else { 0 };

    for y in 0..output.height() {
        let row = input.row(y);
        let out_row = output.row_mut(y);

        for x in 0..border {
            let ix = x as i32;
            let v_m2 = row[mirror(ix - 2, iwidth)];
            let v_m1 = row[mirror(ix - 1, iwidth)];
            let v_0 = row[x];
            let v_p1 = row[mirror(ix + 1, iwidth)];
            let v_p2 = row[mirror(ix + 2, iwidth)];
            out_row[x] = v_0 * w0 + (v_m1 + v_p1) * w1 + (v_m2 + v_p2) * w2;
        }

        let mut x = border;
        while x + 8 <= interior_end {
            let v_m2 = f32x8::load(token, (&row[x - 2..x + 6]).try_into().unwrap());
            let v_m1 = f32x8::load(token, (&row[x - 1..x + 7]).try_into().unwrap());
            let v_0 = f32x8::load(token, (&row[x..x + 8]).try_into().unwrap());
            let v_p1 = f32x8::load(token, (&row[x + 1..x + 9]).try_into().unwrap());
            let v_p2 = f32x8::load(token, (&row[x + 2..x + 10]).try_into().unwrap());
            let sum = v_0 * w0_v + (v_m1 + v_p1) * w1_v + (v_m2 + v_p2) * w2_v;
            sum.store((&mut out_row[x..x + 8]).try_into().unwrap());
            x += 8;
        }
        while x < interior_end {
            let v_m2 = row[x - 2];
            let v_m1 = row[x - 1];
            let v_0 = row[x];
            let v_p1 = row[x + 1];
            let v_p2 = row[x + 2];
            out_row[x] = v_0 * w0 + (v_m1 + v_p1) * w1 + (v_m2 + v_p2) * w2;
            x += 1;
        }

        for x in interior_end..width {
            let ix = x as i32;
            let v_m2 = row[mirror(ix - 2, iwidth)];
            let v_m1 = row[mirror(ix - 1, iwidth)];
            let v_0 = row[x];
            let v_p1 = row[mirror(ix + 1, iwidth)];
            let v_p2 = row[mirror(ix + 2, iwidth)];
            out_row[x] = v_0 * w0 + (v_m1 + v_p1) * w1 + (v_m2 + v_p2) * w2;
        }
    }
}

fn blur_mirrored_5x5_h_strip_dispatch_scalar(
    _token: archmage::ScalarToken,
    input: &StripView<'_>,
    output: &mut StripViewMut<'_>,
    weights: &[f32; 3],
) {
    // NB: scalar full-buffer blur_mirrored_5x5 uses transposed temp +
    // per-column scan. Per-row scalar produces identical bits on each
    // row independently — the transpose was a cache trick, not arithmetic.
    let width = input.width();
    let w0 = weights[0];
    let w1 = weights[1];
    let w2 = weights[2];
    let iwidth = width as i32;
    for y in 0..output.height() {
        let row = input.row(y);
        let out_row = output.row_mut(y);
        for x in 0..width {
            let ix = x as i32;
            let v_m2 = row[mirror(ix - 2, iwidth)];
            let v_m1 = row[mirror(ix - 1, iwidth)];
            let v_0 = row[x];
            let v_p1 = row[mirror(ix + 1, iwidth)];
            let v_p2 = row[mirror(ix + 2, iwidth)];
            out_row[x] = v_0 * w0 + (v_m1 + v_p1) * w1 + (v_m2 + v_p2) * w2;
        }
    }
}

/// Strip-tiled vertical phase of [`blur::blur_mirrored_5x5`].
///
/// Mirrored boundary at parent top/bottom. Halo is fixed at
/// [`BLUR_MIRRORED_5X5_HALO`] (= 2) rows per side.
///
/// `parent_height` MUST be the height of the original ImageF that the output
/// strip is a view into — used for parent-edge mirror semantics.
///
/// The caller must arrange that `input_with_halo` covers, at minimum:
///
/// ```text
///   input_top    = output.start_row_in_parent().saturating_sub(2)
///   input_bottom = (output.start_row_in_parent() + output.height() + 2).min(parent_height)
/// ```
pub fn blur_mirrored_5x5_v_strip(
    input_with_halo: &StripView<'_>,
    output: &mut StripViewMut<'_>,
    parent_height: usize,
    weights: &[f32; 3],
) {
    assert_eq!(
        input_with_halo.width(),
        output.width(),
        "blur_mirrored_5x5_v_strip: width mismatch"
    );
    let out_start = output.start_row_in_parent();
    let out_height = output.height();
    assert!(
        out_start + out_height <= parent_height,
        "blur_mirrored_5x5_v_strip: output strip exceeds parent_height"
    );
    archmage::incant!(
        blur_mirrored_5x5_v_strip_dispatch(input_with_halo, output, parent_height, weights),
        [v4, v3, neon, wasm128]
    );
}

/// Get a row from the input strip given a parent-y coordinate (mirrored if
/// out of parent bounds). Centralized so SIMD impls share semantics.
#[inline]
fn mirror_parent_y_to_local<'a>(
    input_with_halo: &'a StripView<'_>,
    parent_y: i32,
    parent_height: usize,
    input_start_in_parent: usize,
) -> &'a [f32] {
    let mirrored_parent_y = mirror(parent_y, parent_height as i32);
    let local_y = mirrored_parent_y - input_start_in_parent;
    input_with_halo.row(local_y)
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn blur_mirrored_5x5_v_strip_dispatch_v4(
    token: archmage::X64V4Token,
    input_with_halo: &StripView<'_>,
    output: &mut StripViewMut<'_>,
    parent_height: usize,
    weights: &[f32; 3],
) {
    use magetypes::simd::v4::f32x16;
    let width = input_with_halo.width();
    let in_start = input_with_halo.start_row_in_parent();
    let out_start = output.start_row_in_parent();
    let out_height = output.height();
    let w0 = weights[0];
    let w1 = weights[1];
    let w2 = weights[2];
    let w0_v = f32x16::splat(token, w0);
    let w1_v = f32x16::splat(token, w1);
    let w2_v = f32x16::splat(token, w2);

    let v_border = 2.min(parent_height);
    let v_interior_end = if parent_height > 4 {
        parent_height - 2
    } else {
        0
    };

    for local_y in 0..out_height {
        let parent_y = out_start + local_y;
        let py = parent_y as i32;
        let out = output.row_mut(local_y);

        let rm2 = if parent_y < v_border || parent_y >= v_interior_end {
            mirror_parent_y_to_local(input_with_halo, py - 2, parent_height, in_start)
        } else {
            input_with_halo.row(parent_y - 2 - in_start)
        };
        let rm1 = if parent_y < v_border || parent_y >= v_interior_end {
            mirror_parent_y_to_local(input_with_halo, py - 1, parent_height, in_start)
        } else {
            input_with_halo.row(parent_y - 1 - in_start)
        };
        let r0 = input_with_halo.row(parent_y - in_start);
        let rp1 = if parent_y < v_border || parent_y >= v_interior_end {
            mirror_parent_y_to_local(input_with_halo, py + 1, parent_height, in_start)
        } else {
            input_with_halo.row(parent_y + 1 - in_start)
        };
        let rp2 = if parent_y < v_border || parent_y >= v_interior_end {
            mirror_parent_y_to_local(input_with_halo, py + 2, parent_height, in_start)
        } else {
            input_with_halo.row(parent_y + 2 - in_start)
        };

        let mut x = 0;
        while x + 16 <= width {
            let vm2 = f32x16::load(token, (&rm2[x..x + 16]).try_into().unwrap());
            let vm1 = f32x16::load(token, (&rm1[x..x + 16]).try_into().unwrap());
            let v0 = f32x16::load(token, (&r0[x..x + 16]).try_into().unwrap());
            let vp1 = f32x16::load(token, (&rp1[x..x + 16]).try_into().unwrap());
            let vp2 = f32x16::load(token, (&rp2[x..x + 16]).try_into().unwrap());
            let sum = v0 * w0_v + (vm1 + vp1) * w1_v + (vm2 + vp2) * w2_v;
            sum.store((&mut out[x..x + 16]).try_into().unwrap());
            x += 16;
        }
        while x < width {
            out[x] = r0[x] * w0 + (rm1[x] + rp1[x]) * w1 + (rm2[x] + rp2[x]) * w2;
            x += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn blur_mirrored_5x5_v_strip_dispatch_v3(
    token: archmage::X64V3Token,
    input_with_halo: &StripView<'_>,
    output: &mut StripViewMut<'_>,
    parent_height: usize,
    weights: &[f32; 3],
) {
    use magetypes::simd::f32x8;
    let width = input_with_halo.width();
    let in_start = input_with_halo.start_row_in_parent();
    let out_start = output.start_row_in_parent();
    let out_height = output.height();
    let w0 = weights[0];
    let w1 = weights[1];
    let w2 = weights[2];
    let w0_v = f32x8::splat(token, w0);
    let w1_v = f32x8::splat(token, w1);
    let w2_v = f32x8::splat(token, w2);

    let v_border = 2.min(parent_height);
    let v_interior_end = if parent_height > 4 {
        parent_height - 2
    } else {
        0
    };

    for local_y in 0..out_height {
        let parent_y = out_start + local_y;
        let py = parent_y as i32;
        let out = output.row_mut(local_y);

        let rm2 = if parent_y < v_border || parent_y >= v_interior_end {
            mirror_parent_y_to_local(input_with_halo, py - 2, parent_height, in_start)
        } else {
            input_with_halo.row(parent_y - 2 - in_start)
        };
        let rm1 = if parent_y < v_border || parent_y >= v_interior_end {
            mirror_parent_y_to_local(input_with_halo, py - 1, parent_height, in_start)
        } else {
            input_with_halo.row(parent_y - 1 - in_start)
        };
        let r0 = input_with_halo.row(parent_y - in_start);
        let rp1 = if parent_y < v_border || parent_y >= v_interior_end {
            mirror_parent_y_to_local(input_with_halo, py + 1, parent_height, in_start)
        } else {
            input_with_halo.row(parent_y + 1 - in_start)
        };
        let rp2 = if parent_y < v_border || parent_y >= v_interior_end {
            mirror_parent_y_to_local(input_with_halo, py + 2, parent_height, in_start)
        } else {
            input_with_halo.row(parent_y + 2 - in_start)
        };

        let mut x = 0;
        while x + 8 <= width {
            let vm2 = f32x8::load(token, (&rm2[x..x + 8]).try_into().unwrap());
            let vm1 = f32x8::load(token, (&rm1[x..x + 8]).try_into().unwrap());
            let v0 = f32x8::load(token, (&r0[x..x + 8]).try_into().unwrap());
            let vp1 = f32x8::load(token, (&rp1[x..x + 8]).try_into().unwrap());
            let vp2 = f32x8::load(token, (&rp2[x..x + 8]).try_into().unwrap());
            let sum = v0 * w0_v + (vm1 + vp1) * w1_v + (vm2 + vp2) * w2_v;
            sum.store((&mut out[x..x + 8]).try_into().unwrap());
            x += 8;
        }
        while x < width {
            out[x] = r0[x] * w0 + (rm1[x] + rp1[x]) * w1 + (rm2[x] + rp2[x]) * w2;
            x += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn blur_mirrored_5x5_v_strip_dispatch_neon(
    token: archmage::NeonToken,
    input_with_halo: &StripView<'_>,
    output: &mut StripViewMut<'_>,
    parent_height: usize,
    weights: &[f32; 3],
) {
    use magetypes::simd::f32x8;
    let width = input_with_halo.width();
    let in_start = input_with_halo.start_row_in_parent();
    let out_start = output.start_row_in_parent();
    let out_height = output.height();
    let w0 = weights[0];
    let w1 = weights[1];
    let w2 = weights[2];
    let w0_v = f32x8::splat(token, w0);
    let w1_v = f32x8::splat(token, w1);
    let w2_v = f32x8::splat(token, w2);

    let v_border = 2.min(parent_height);
    let v_interior_end = if parent_height > 4 {
        parent_height - 2
    } else {
        0
    };

    for local_y in 0..out_height {
        let parent_y = out_start + local_y;
        let py = parent_y as i32;
        let out = output.row_mut(local_y);

        let rm2 = if parent_y < v_border || parent_y >= v_interior_end {
            mirror_parent_y_to_local(input_with_halo, py - 2, parent_height, in_start)
        } else {
            input_with_halo.row(parent_y - 2 - in_start)
        };
        let rm1 = if parent_y < v_border || parent_y >= v_interior_end {
            mirror_parent_y_to_local(input_with_halo, py - 1, parent_height, in_start)
        } else {
            input_with_halo.row(parent_y - 1 - in_start)
        };
        let r0 = input_with_halo.row(parent_y - in_start);
        let rp1 = if parent_y < v_border || parent_y >= v_interior_end {
            mirror_parent_y_to_local(input_with_halo, py + 1, parent_height, in_start)
        } else {
            input_with_halo.row(parent_y + 1 - in_start)
        };
        let rp2 = if parent_y < v_border || parent_y >= v_interior_end {
            mirror_parent_y_to_local(input_with_halo, py + 2, parent_height, in_start)
        } else {
            input_with_halo.row(parent_y + 2 - in_start)
        };

        let mut x = 0;
        while x + 8 <= width {
            let vm2 = f32x8::load(token, (&rm2[x..x + 8]).try_into().unwrap());
            let vm1 = f32x8::load(token, (&rm1[x..x + 8]).try_into().unwrap());
            let v0 = f32x8::load(token, (&r0[x..x + 8]).try_into().unwrap());
            let vp1 = f32x8::load(token, (&rp1[x..x + 8]).try_into().unwrap());
            let vp2 = f32x8::load(token, (&rp2[x..x + 8]).try_into().unwrap());
            let sum = v0 * w0_v + (vm1 + vp1) * w1_v + (vm2 + vp2) * w2_v;
            sum.store((&mut out[x..x + 8]).try_into().unwrap());
            x += 8;
        }
        while x < width {
            out[x] = r0[x] * w0 + (rm1[x] + rp1[x]) * w1 + (rm2[x] + rp2[x]) * w2;
            x += 1;
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
fn blur_mirrored_5x5_v_strip_dispatch_wasm128(
    token: archmage::Wasm128Token,
    input_with_halo: &StripView<'_>,
    output: &mut StripViewMut<'_>,
    parent_height: usize,
    weights: &[f32; 3],
) {
    use magetypes::simd::f32x8;
    let width = input_with_halo.width();
    let in_start = input_with_halo.start_row_in_parent();
    let out_start = output.start_row_in_parent();
    let out_height = output.height();
    let w0 = weights[0];
    let w1 = weights[1];
    let w2 = weights[2];
    let w0_v = f32x8::splat(token, w0);
    let w1_v = f32x8::splat(token, w1);
    let w2_v = f32x8::splat(token, w2);

    let v_border = 2.min(parent_height);
    let v_interior_end = if parent_height > 4 {
        parent_height - 2
    } else {
        0
    };

    for local_y in 0..out_height {
        let parent_y = out_start + local_y;
        let py = parent_y as i32;
        let out = output.row_mut(local_y);

        let rm2 = if parent_y < v_border || parent_y >= v_interior_end {
            mirror_parent_y_to_local(input_with_halo, py - 2, parent_height, in_start)
        } else {
            input_with_halo.row(parent_y - 2 - in_start)
        };
        let rm1 = if parent_y < v_border || parent_y >= v_interior_end {
            mirror_parent_y_to_local(input_with_halo, py - 1, parent_height, in_start)
        } else {
            input_with_halo.row(parent_y - 1 - in_start)
        };
        let r0 = input_with_halo.row(parent_y - in_start);
        let rp1 = if parent_y < v_border || parent_y >= v_interior_end {
            mirror_parent_y_to_local(input_with_halo, py + 1, parent_height, in_start)
        } else {
            input_with_halo.row(parent_y + 1 - in_start)
        };
        let rp2 = if parent_y < v_border || parent_y >= v_interior_end {
            mirror_parent_y_to_local(input_with_halo, py + 2, parent_height, in_start)
        } else {
            input_with_halo.row(parent_y + 2 - in_start)
        };

        let mut x = 0;
        while x + 8 <= width {
            let vm2 = f32x8::load(token, (&rm2[x..x + 8]).try_into().unwrap());
            let vm1 = f32x8::load(token, (&rm1[x..x + 8]).try_into().unwrap());
            let v0 = f32x8::load(token, (&r0[x..x + 8]).try_into().unwrap());
            let vp1 = f32x8::load(token, (&rp1[x..x + 8]).try_into().unwrap());
            let vp2 = f32x8::load(token, (&rp2[x..x + 8]).try_into().unwrap());
            let sum = v0 * w0_v + (vm1 + vp1) * w1_v + (vm2 + vp2) * w2_v;
            sum.store((&mut out[x..x + 8]).try_into().unwrap());
            x += 8;
        }
        while x < width {
            out[x] = r0[x] * w0 + (rm1[x] + rp1[x]) * w1 + (rm2[x] + rp2[x]) * w2;
            x += 1;
        }
    }
}

fn blur_mirrored_5x5_v_strip_dispatch_scalar(
    _token: archmage::ScalarToken,
    input_with_halo: &StripView<'_>,
    output: &mut StripViewMut<'_>,
    parent_height: usize,
    weights: &[f32; 3],
) {
    // NB: scalar full-buffer blur_mirrored_5x5_scalar uses a transposed
    // temp buffer + per-column scan. Per-row scalar produces identical
    // bits — transpose was a cache optimization, not arithmetic.
    let width = input_with_halo.width();
    let in_start = input_with_halo.start_row_in_parent();
    let out_start = output.start_row_in_parent();
    let out_height = output.height();
    let w0 = weights[0];
    let w1 = weights[1];
    let w2 = weights[2];

    for local_y in 0..out_height {
        let parent_y = out_start + local_y;
        let py = parent_y as i32;
        let out = output.row_mut(local_y);

        let rm2 = mirror_parent_y_to_local(input_with_halo, py - 2, parent_height, in_start);
        let rm1 = mirror_parent_y_to_local(input_with_halo, py - 1, parent_height, in_start);
        let r0 = input_with_halo.row(parent_y - in_start);
        let rp1 = mirror_parent_y_to_local(input_with_halo, py + 1, parent_height, in_start);
        let rp2 = mirror_parent_y_to_local(input_with_halo, py + 2, parent_height, in_start);

        for x in 0..width {
            out[x] = r0[x] * w0 + (rm1[x] + rp1[x]) * w1 + (rm2[x] + rp2[x]) * w2;
        }
    }
}

// ============================================================================
// Tests — W44-PHASE3-B7d Day 2
// ============================================================================
//
// Validate strip-tiled blur output is BYTE-IDENTICAL to the full-buffer blur
// it replaces. Tests are gated `#[cfg(not(feature = "iir-blur"))]` because
// when the IIR backend is selected, `crate::blur::gaussian_blur` dispatches
// to the recursive Charalampidis path that intentionally differs from the
// FIR convolution our strip variants implement.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{BufferPool, ImageF};

    /// Reproducible synthetic image fill. Per-pixel value =
    /// `(x.wrapping_mul(2654435761) ^ y.wrapping_mul(40503))` reinterpreted
    /// as f32 with the upper bits masked to keep magnitudes finite.
    fn fill_random(img: &mut ImageF, seed: u32) {
        let width = img.width();
        let height = img.height();
        for y in 0..height {
            let row = img.row_mut(y);
            for x in 0..width {
                let k = (x as u32)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add((y as u32).wrapping_mul(40_503))
                    .wrapping_add(seed);
                // Map to [0, 1] with reasonable distribution
                row[x] = (k as f32 / u32::MAX as f32).clamp(0.0, 1.0);
            }
        }
    }

    /// Full-buffer reference: apply the existing H phase then V phase via
    /// the strip API with the whole image as a single strip. Used to verify
    /// the strip API matches the legacy [`crate::blur::gaussian_blur`]
    /// byte-for-byte when invoked as a degenerate single strip.
    #[cfg(not(feature = "iir-blur"))]
    fn reference_full_via_strip(input: &ImageF, sigma: f32, pool: &BufferPool) -> ImageF {
        let width = input.width();
        let height = input.height();
        let strip_in = input.strip_view(0, height);
        let mut temp = ImageF::from_pool_dirty(width, height, pool);
        {
            let mut temp_strip = temp.strip_view_mut(0, height);
            gaussian_blur_h_strip(&strip_in, &mut temp_strip, sigma);
        }

        let temp_strip = temp.strip_view(0, height);
        let mut output = ImageF::from_pool_dirty(width, height, pool);
        {
            let mut out_strip = output.strip_view_mut(0, height);
            gaussian_blur_v_strip(&temp_strip, &mut out_strip, height, sigma);
        }
        temp.recycle(pool);
        output
    }

    /// Assert two ImageF buffers are byte-for-byte identical on every pixel.
    fn assert_image_eq_bits(actual: &ImageF, expected: &ImageF, label: &str) {
        assert_eq!(actual.width(), expected.width(), "{label}: width");
        assert_eq!(actual.height(), expected.height(), "{label}: height");
        for y in 0..actual.height() {
            let a = actual.row(y);
            let e = expected.row(y);
            for x in 0..a.len() {
                let ab = a[x].to_bits();
                let eb = e[x].to_bits();
                assert_eq!(
                    ab, eb,
                    "{label}: pixel ({x}, {y}) bits differ: actual={ab:#010x} ({}) expected={eb:#010x} ({})",
                    a[x], e[x]
                );
            }
        }
    }

    // ---- Test #1 — identity test: full-image strip = full-buffer blur ----

    #[cfg(not(feature = "iir-blur"))]
    #[test]
    fn test_gaussian_blur_h_v_strip_identity_matches_full() {
        // Full-image single strip applied via strip API must match
        // crate::blur::gaussian_blur byte-for-byte.
        let pool = BufferPool::new();
        for &sigma in &[0.5f32, 1.0, 3.0, 7.156] {
            for &(w, h) in &[(64usize, 64usize), (128, 96), (33, 17)] {
                let mut img = ImageF::new(w, h);
                fill_random(&mut img, 0xCAFE);
                let full = crate::blur::gaussian_blur(&img, sigma, &pool);
                let strip = reference_full_via_strip(&img, sigma, &pool);
                assert_image_eq_bits(
                    &strip,
                    &full,
                    &format!("identity sigma={sigma} dims={w}x{h}"),
                );
            }
        }
    }

    // ---- Test #2 — two-strip split, output concatenation matches full ----

    #[cfg(not(feature = "iir-blur"))]
    #[test]
    fn test_gaussian_blur_two_strip_split_matches_full() {
        // 256x256 image, split into two 128-row strips with 17-row halo on
        // each side (gaussian_blur_halo(7.156) = 17). Output strips
        // concatenate to byte-identical with the full-buffer blur.
        let pool = BufferPool::new();
        let sigma = 7.156f32;
        let halo = gaussian_blur_halo(sigma);
        // diff = ceil(2.25 * 7.156).max(1.0) as i32 = 16 (truncation, not ceil)
        // kernel size = 2 * 16 + 1 = 33; half = 16 = halo per side.
        assert_eq!(halo, 16, "halo for sigma=7.156 must be 16");

        let width = 256usize;
        let height = 256usize;
        let mut img = ImageF::new(width, height);
        fill_random(&mut img, 0xDEAD);

        // Reference: full-buffer blur (H + V via existing code path).
        let full = crate::blur::gaussian_blur(&img, sigma, &pool);

        // H phase strip-tile (per-row, no vertical halo): can be done per-strip
        // with logical row ranges, OR full-image (no difference). To exercise
        // strips, run H per-strip then V per-strip with V halo.
        let mut temp = ImageF::from_pool_dirty(width, height, &pool);
        // We can't construct a strip from an empty ImageF and write incrementally
        // easily — simplest is to do H on full image into temp, then V per-strip.
        // That's a fair Day-2 split test of V; H is per-row trivially equal.
        {
            let in_strip = img.strip_view(0, height);
            let mut out_strip = temp.strip_view_mut(0, height);
            gaussian_blur_h_strip(&in_strip, &mut out_strip, sigma);
        }

        // V phase per-strip: two output strips of 128 rows each.
        let mut output = ImageF::from_pool_dirty(width, height, &pool);
        let mid = 128usize;
        // First strip: output rows [0, 128). Input halo: [0, 128 + 17) = [0, 145).
        {
            let in_top = 0usize;
            let in_bot = (mid + halo).min(height);
            let in_strip = temp.strip_view(in_top, in_bot);
            let mut out_strip = output.strip_view_mut(0, mid);
            gaussian_blur_v_strip(&in_strip, &mut out_strip, height, sigma);
        }
        // Second strip: output rows [128, 256). Input halo: [128 - 17, 256) = [111, 256).
        {
            let in_top = mid.saturating_sub(halo);
            let in_bot = height;
            let in_strip = temp.strip_view(in_top, in_bot);
            let mut out_strip = output.strip_view_mut(mid, height);
            gaussian_blur_v_strip(&in_strip, &mut out_strip, height, sigma);
        }

        assert_image_eq_bits(&output, &full, "two-strip 256x256 sigma=7.156");
        temp.recycle(&pool);
    }

    // ---- Test #3 — many-strip (16-row) byte-identical ----

    #[cfg(not(feature = "iir-blur"))]
    #[test]
    fn test_gaussian_blur_many_strip_byte_identical() {
        let pool = BufferPool::new();
        let sigma = 7.156f32;
        let halo = gaussian_blur_halo(sigma);
        let width = 128usize;
        let height = 128usize;
        let strip_rows = 16usize;

        let mut img = ImageF::new(width, height);
        fill_random(&mut img, 0xBEEF);
        let full = crate::blur::gaussian_blur(&img, sigma, &pool);

        // H pass: full image (could be per-strip too; H is per-row trivial)
        let mut temp = ImageF::from_pool_dirty(width, height, &pool);
        {
            let in_strip = img.strip_view(0, height);
            let mut out_strip = temp.strip_view_mut(0, height);
            gaussian_blur_h_strip(&in_strip, &mut out_strip, sigma);
        }

        // V pass: 8 strips of 16 rows each, each with halo on both sides.
        let mut output = ImageF::from_pool_dirty(width, height, &pool);
        let strip_count = height / strip_rows;
        assert_eq!(strip_count * strip_rows, height);
        for s in 0..strip_count {
            let out_top = s * strip_rows;
            let out_bot = out_top + strip_rows;
            let in_top = out_top.saturating_sub(halo);
            let in_bot = (out_bot + halo).min(height);
            let in_strip = temp.strip_view(in_top, in_bot);
            let mut out_strip = output.strip_view_mut(out_top, out_bot);
            gaussian_blur_v_strip(&in_strip, &mut out_strip, height, sigma);
        }

        assert_image_eq_bits(
            &output,
            &full,
            "many-strip 128x128 sigma=7.156 strip_rows=16",
        );
        temp.recycle(&pool);
    }

    // ---- Test #4 — edge strips (top has no top-halo, bottom has no bottom-halo) ----

    #[cfg(not(feature = "iir-blur"))]
    #[test]
    fn test_gaussian_blur_edge_strips_byte_identical() {
        let pool = BufferPool::new();
        let sigma = 3.0f32; // halo = 6 (diff = (2.25*3.0) as i32 = 6)
        let halo = gaussian_blur_halo(sigma);
        assert_eq!(halo, 6);
        let width = 96usize;
        let height = 96usize;
        let strip_rows = 16usize;

        let mut img = ImageF::new(width, height);
        fill_random(&mut img, 0x1234);
        let full = crate::blur::gaussian_blur(&img, sigma, &pool);

        let mut temp = ImageF::from_pool_dirty(width, height, &pool);
        {
            let in_strip = img.strip_view(0, height);
            let mut out_strip = temp.strip_view_mut(0, height);
            gaussian_blur_h_strip(&in_strip, &mut out_strip, sigma);
        }

        // Top strip [0, 16): in_top=0 (clamped), so NO top halo at all.
        let mut output = ImageF::from_pool_dirty(width, height, &pool);
        {
            let in_top = 0;
            let in_bot = (strip_rows + halo).min(height);
            let in_strip = temp.strip_view(in_top, in_bot);
            let mut out_strip = output.strip_view_mut(0, strip_rows);
            gaussian_blur_v_strip(&in_strip, &mut out_strip, height, sigma);
        }
        // Bottom strip [80, 96): in_bot=96 (clamped), so NO bottom halo.
        {
            let out_top = height - strip_rows;
            let in_top = out_top.saturating_sub(halo);
            let in_bot = height;
            let in_strip = temp.strip_view(in_top, in_bot);
            let mut out_strip = output.strip_view_mut(out_top, height);
            gaussian_blur_v_strip(&in_strip, &mut out_strip, height, sigma);
        }
        // Middle strips [16, 80) chunked into 16-row pieces with full halos.
        for out_top in (strip_rows..(height - strip_rows)).step_by(strip_rows) {
            let out_bot = out_top + strip_rows;
            let in_top = out_top - halo;
            let in_bot = out_bot + halo;
            let in_strip = temp.strip_view(in_top, in_bot);
            let mut out_strip = output.strip_view_mut(out_top, out_bot);
            gaussian_blur_v_strip(&in_strip, &mut out_strip, height, sigma);
        }
        assert_image_eq_bits(&output, &full, "edge-strips 96x96 sigma=3.0");

        // Verify the per-strip top rows match the full path exactly. Top
        // strip rows 0..7 are in the parent's border region, exercising
        // the border-handling branch in the V-pass.
        for y in 0..(halo + 1).min(height) {
            assert_eq!(
                output
                    .row(y)
                    .iter()
                    .map(|f| f.to_bits())
                    .collect::<Vec<_>>(),
                full.row(y).iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                "top edge row {y}",
            );
        }
        // Bottom strip rows height-halo-1..height are in the bottom border.
        for y in (height - halo - 1)..height {
            assert_eq!(
                output
                    .row(y)
                    .iter()
                    .map(|f| f.to_bits())
                    .collect::<Vec<_>>(),
                full.row(y).iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                "bottom edge row {y}",
            );
        }

        temp.recycle(&pool);
    }

    // ---- Test #5 — stride preservation ----

    #[cfg(not(feature = "iir-blur"))]
    #[test]
    fn test_gaussian_blur_strip_preserves_stride() {
        // ImageF stride is computed as (width + 15) & !15. Pick a width that
        // forces stride > width (e.g. width=17 → stride=32).
        let pool = BufferPool::new();
        let sigma = 1.0f32;
        let width = 17usize;
        let height = 24usize;

        let mut img = ImageF::new(width, height);
        assert!(
            img.stride() >= width,
            "stride={} should be >= width={}",
            img.stride(),
            width
        );
        // Sentinel-fill padding bytes; verify they remain untouched after blur.
        let sentinel = f32::from_bits(0xDEADBEEF);
        // Initialize padding via row_full
        for y in 0..height {
            // Fill rows[0..stride], then overwrite [0..width] with content.
            for v in img.row_full_mut(y).iter_mut() {
                *v = sentinel;
            }
        }
        // Now overwrite the in-width portion with random data
        fill_random(&mut img, 0x42);

        let full = crate::blur::gaussian_blur(&img, sigma, &pool);
        let strip = reference_full_via_strip(&img, sigma, &pool);
        assert_image_eq_bits(&strip, &full, "stride-preservation 17x24 sigma=1.0");

        // Strides must be equal (both pool-allocated buffers; pool clusters
        // by capacity, but cap recovers stride deterministically).
        assert_eq!(strip.stride(), full.stride(), "result strides differ");
    }

    // ---- Test #6 — sigma sweep ----

    #[cfg(not(feature = "iir-blur"))]
    #[test]
    fn test_gaussian_blur_strip_sigma_sweep() {
        let pool = BufferPool::new();
        let width = 64usize;
        let height = 64usize;
        let mut img = ImageF::new(width, height);
        fill_random(&mut img, 0x99);

        for &sigma in &[0.5f32, 1.0, 3.0, 7.156] {
            let full = crate::blur::gaussian_blur(&img, sigma, &pool);
            // Run strip API two ways:
            // (a) single-strip
            let single = reference_full_via_strip(&img, sigma, &pool);
            assert_image_eq_bits(&single, &full, &format!("sigma={sigma} single-strip"));

            // (b) 8-row strips
            let halo = gaussian_blur_halo(sigma);
            let mut temp = ImageF::from_pool_dirty(width, height, &pool);
            {
                let in_s = img.strip_view(0, height);
                let mut out_s = temp.strip_view_mut(0, height);
                gaussian_blur_h_strip(&in_s, &mut out_s, sigma);
            }
            let mut out = ImageF::from_pool_dirty(width, height, &pool);
            for out_top in (0..height).step_by(8) {
                let out_bot = (out_top + 8).min(height);
                let in_top = out_top.saturating_sub(halo);
                let in_bot = (out_bot + halo).min(height);
                let in_s = temp.strip_view(in_top, in_bot);
                let mut out_s = out.strip_view_mut(out_top, out_bot);
                gaussian_blur_v_strip(&in_s, &mut out_s, height, sigma);
            }
            assert_image_eq_bits(&out, &full, &format!("sigma={sigma} 8-row strips"));
            temp.recycle(&pool);
        }
    }

    // ---- Test #7 — blur_mirrored_5x5 strip identity + split ----

    #[test]
    fn test_blur_mirrored_5x5_strip_identity_and_split() {
        let pool = BufferPool::new();
        let weights: [f32; 3] = [0.5f32, 0.25, 0.05];
        let width = 64usize;
        let height = 64usize;
        let mut img = ImageF::new(width, height);
        fill_random(&mut img, 0xACED);

        let full = crate::blur::blur_mirrored_5x5(&img, &weights, &pool);

        // Identity: single-strip H + V via strip API
        let mut temp1 = ImageF::from_pool_dirty(width, height, &pool);
        {
            let in_s = img.strip_view(0, height);
            let mut out_s = temp1.strip_view_mut(0, height);
            blur_mirrored_5x5_h_strip(&in_s, &mut out_s, &weights);
        }
        let mut single = ImageF::from_pool_dirty(width, height, &pool);
        {
            let in_s = temp1.strip_view(0, height);
            let mut out_s = single.strip_view_mut(0, height);
            blur_mirrored_5x5_v_strip(&in_s, &mut out_s, height, &weights);
        }
        assert_image_eq_bits(&single, &full, "5x5 single-strip identity");

        // Two-strip split
        let halo = BLUR_MIRRORED_5X5_HALO;
        let mid = 32usize;
        let mut temp2 = ImageF::from_pool_dirty(width, height, &pool);
        {
            let in_s = img.strip_view(0, height);
            let mut out_s = temp2.strip_view_mut(0, height);
            blur_mirrored_5x5_h_strip(&in_s, &mut out_s, &weights);
        }
        let mut split = ImageF::from_pool_dirty(width, height, &pool);
        {
            let in_top = 0;
            let in_bot = (mid + halo).min(height);
            let in_s = temp2.strip_view(in_top, in_bot);
            let mut out_s = split.strip_view_mut(0, mid);
            blur_mirrored_5x5_v_strip(&in_s, &mut out_s, height, &weights);
        }
        {
            let in_top = mid.saturating_sub(halo);
            let in_bot = height;
            let in_s = temp2.strip_view(in_top, in_bot);
            let mut out_s = split.strip_view_mut(mid, height);
            blur_mirrored_5x5_v_strip(&in_s, &mut out_s, height, &weights);
        }
        assert_image_eq_bits(&split, &full, "5x5 two-strip split");

        // Many strips of 8 rows
        let mut temp3 = ImageF::from_pool_dirty(width, height, &pool);
        {
            let in_s = img.strip_view(0, height);
            let mut out_s = temp3.strip_view_mut(0, height);
            blur_mirrored_5x5_h_strip(&in_s, &mut out_s, &weights);
        }
        let mut many = ImageF::from_pool_dirty(width, height, &pool);
        for out_top in (0..height).step_by(8) {
            let out_bot = (out_top + 8).min(height);
            let in_top = out_top.saturating_sub(halo);
            let in_bot = (out_bot + halo).min(height);
            let in_s = temp3.strip_view(in_top, in_bot);
            let mut out_s = many.strip_view_mut(out_top, out_bot);
            blur_mirrored_5x5_v_strip(&in_s, &mut out_s, height, &weights);
        }
        assert_image_eq_bits(&many, &full, "5x5 many-strip (8 rows each)");

        temp1.recycle(&pool);
        temp2.recycle(&pool);
        temp3.recycle(&pool);
    }
}
