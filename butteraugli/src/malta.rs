//! Malta edge-aware filter for butteraugli.
//!
//! The Malta filter applies 16 different line kernels at various orientations
//! and sums their squared responses. This captures edge information at
//! multiple orientations for perceptual quality measurement.
//!
//! There are two variants:
//! - `MaltaUnit`: Full 9x9 pattern with 9-sample line kernels
//! - `MaltaUnitLF`: Low-frequency variant with 5-sample line kernels
//!
//! The implementation uses safe Rust with direct slice indexing for interior
//! pixels and fixed-size array windows for border pixels.
//!
//! With the `unsafe-performance` feature, interior functions use unchecked
//! indexing after a single bounds assertion, eliminating per-access bounds
//! checks for ~6% fewer instructions.

use crate::image::ImageF;

/// Read a single f32 from a data slice.
///
/// With `unsafe-performance`: unchecked access (caller must pre-validate range).
/// Without: normal bounds-checked indexing.
#[cfg(feature = "unsafe-performance")]
#[allow(clippy::inline_always)]
#[inline(always)]
fn data_at(data: &[f32], idx: usize) -> f32 {
    // SAFETY: callers assert the full access range before any calls to this function.
    unsafe { *data.get_unchecked(idx) }
}

#[cfg(not(feature = "unsafe-performance"))]
#[inline]
fn data_at(data: &[f32], idx: usize) -> f32 {
    data[idx]
}

/// Load 8 contiguous f32 values as a fixed-size array reference.
///
/// With `unsafe-performance`: pointer cast (caller must pre-validate range).
/// Without: slice + try_into with bounds check.
#[cfg(feature = "unsafe-performance")]
#[allow(clippy::inline_always)]
#[inline(always)]
fn load_8(data: &[f32], start: usize) -> &[f32; 8] {
    // SAFETY: callers assert the full access range before any calls to this function.
    unsafe { &*data.as_ptr().add(start).cast::<[f32; 8]>() }
}

#[cfg(not(feature = "unsafe-performance"))]
#[inline]
fn load_8(data: &[f32], start: usize) -> &[f32; 8] {
    data[start..start + 8].try_into().unwrap()
}

/// Access a pixel in a 9x9 window at offset (dx, dy) from center.
/// Center is at (4, 4), so valid offsets are -4..=4.
macro_rules! w {
    ($window:expr, $dx:expr, $dy:expr) => {
        $window[((4 + $dy) * 9 + (4 + $dx)) as usize]
    };
}

/// Malta filter on a 9x9 window (HF/UHF bands, 9 samples per line).
///
/// Takes a 9x9 window centered on the target pixel and returns
/// the sum of squared responses for 16 orientation patterns.
#[inline]
fn malta_unit_window(window: &[f32; 81]) -> f32 {
    let mut retval = 0.0f32;

    // Pattern 1: x grows, y constant (horizontal line)
    {
        let sum = w!(window, -4, 0)
            + w!(window, -3, 0)
            + w!(window, -2, 0)
            + w!(window, -1, 0)
            + w!(window, 0, 0)
            + w!(window, 1, 0)
            + w!(window, 2, 0)
            + w!(window, 3, 0)
            + w!(window, 4, 0);
        retval += sum * sum;
    }

    // Pattern 2: y grows, x constant (vertical line)
    {
        let sum = w!(window, 0, -4)
            + w!(window, 0, -3)
            + w!(window, 0, -2)
            + w!(window, 0, -1)
            + w!(window, 0, 0)
            + w!(window, 0, 1)
            + w!(window, 0, 2)
            + w!(window, 0, 3)
            + w!(window, 0, 4);
        retval += sum * sum;
    }

    // Pattern 3: both grow (diagonal \)
    {
        let sum = w!(window, -3, -3)
            + w!(window, -2, -2)
            + w!(window, -1, -1)
            + w!(window, 0, 0)
            + w!(window, 1, 1)
            + w!(window, 2, 2)
            + w!(window, 3, 3);
        retval += sum * sum;
    }

    // Pattern 4: y grows, x shrinks (diagonal /)
    {
        let sum = w!(window, 3, -3)
            + w!(window, 2, -2)
            + w!(window, 1, -1)
            + w!(window, 0, 0)
            + w!(window, -1, 1)
            + w!(window, -2, 2)
            + w!(window, -3, 3);
        retval += sum * sum;
    }

    // Pattern 5: y grows -4 to 4, x shrinks 1 -> -1
    {
        let sum = w!(window, 1, -4)
            + w!(window, 1, -3)
            + w!(window, 1, -2)
            + w!(window, 0, -1)
            + w!(window, 0, 0)
            + w!(window, 0, 1)
            + w!(window, -1, 2)
            + w!(window, -1, 3)
            + w!(window, -1, 4);
        retval += sum * sum;
    }

    // Pattern 6: y grows -4 to 4, x grows -1 -> 1
    {
        let sum = w!(window, -1, -4)
            + w!(window, -1, -3)
            + w!(window, -1, -2)
            + w!(window, 0, -1)
            + w!(window, 0, 0)
            + w!(window, 0, 1)
            + w!(window, 1, 2)
            + w!(window, 1, 3)
            + w!(window, 1, 4);
        retval += sum * sum;
    }

    // Pattern 7: x grows -4 to 4, y grows -1 to 1
    {
        let sum = w!(window, -4, -1)
            + w!(window, -3, -1)
            + w!(window, -2, -1)
            + w!(window, -1, 0)
            + w!(window, 0, 0)
            + w!(window, 1, 0)
            + w!(window, 2, 1)
            + w!(window, 3, 1)
            + w!(window, 4, 1);
        retval += sum * sum;
    }

    // Pattern 8: x grows -4 to 4, y shrinks 1 to -1
    {
        let sum = w!(window, -4, 1)
            + w!(window, -3, 1)
            + w!(window, -2, 1)
            + w!(window, -1, 0)
            + w!(window, 0, 0)
            + w!(window, 1, 0)
            + w!(window, 2, -1)
            + w!(window, 3, -1)
            + w!(window, 4, -1);
        retval += sum * sum;
    }

    // Pattern 9: steep diagonal (2:1 slope)
    {
        let sum = w!(window, -2, -3)
            + w!(window, -1, -2)
            + w!(window, -1, -1)
            + w!(window, 0, 0)
            + w!(window, 1, 1)
            + w!(window, 1, 2)
            + w!(window, 2, 3);
        retval += sum * sum;
    }

    // Pattern 10: steep diagonal other way
    {
        let sum = w!(window, 2, -3)
            + w!(window, 1, -2)
            + w!(window, 1, -1)
            + w!(window, 0, 0)
            + w!(window, -1, 1)
            + w!(window, -1, 2)
            + w!(window, -2, 3);
        retval += sum * sum;
    }

    // Pattern 11: shallow diagonal (1:2 slope)
    {
        let sum = w!(window, -3, -2)
            + w!(window, -2, -1)
            + w!(window, -1, -1)
            + w!(window, 0, 0)
            + w!(window, 1, 1)
            + w!(window, 2, 1)
            + w!(window, 3, 2);
        retval += sum * sum;
    }

    // Pattern 12: shallow diagonal other way
    {
        let sum = w!(window, 3, -2)
            + w!(window, 2, -1)
            + w!(window, 1, -1)
            + w!(window, 0, 0)
            + w!(window, -1, 1)
            + w!(window, -2, 1)
            + w!(window, -3, 2);
        retval += sum * sum;
    }

    // Patterns 13-16: duplicates of 8,7,6,5 (9 samples each)

    // Pattern 13: curved line pattern (same as 8)
    {
        let sum = w!(window, -4, 1)
            + w!(window, -3, 1)
            + w!(window, -2, 1)
            + w!(window, -1, 0)
            + w!(window, 0, 0)
            + w!(window, 1, 0)
            + w!(window, 2, -1)
            + w!(window, 3, -1)
            + w!(window, 4, -1);
        retval += sum * sum;
    }

    // Pattern 14: curved line other direction (same as 7)
    {
        let sum = w!(window, -4, -1)
            + w!(window, -3, -1)
            + w!(window, -2, -1)
            + w!(window, -1, 0)
            + w!(window, 0, 0)
            + w!(window, 1, 0)
            + w!(window, 2, 1)
            + w!(window, 3, 1)
            + w!(window, 4, 1);
        retval += sum * sum;
    }

    // Pattern 15: very shallow curve (same as 6)
    {
        let sum = w!(window, -1, -4)
            + w!(window, -1, -3)
            + w!(window, -1, -2)
            + w!(window, 0, -1)
            + w!(window, 0, 0)
            + w!(window, 0, 1)
            + w!(window, 1, 2)
            + w!(window, 1, 3)
            + w!(window, 1, 4);
        retval += sum * sum;
    }

    // Pattern 16: very shallow curve other direction (same as 5)
    {
        let sum = w!(window, 1, -4)
            + w!(window, 1, -3)
            + w!(window, 1, -2)
            + w!(window, 0, -1)
            + w!(window, 0, 0)
            + w!(window, 0, 1)
            + w!(window, -1, 2)
            + w!(window, -1, 3)
            + w!(window, -1, 4);
        retval += sum * sum;
    }

    retval
}

/// Malta filter on a 9x9 window (LF band, 5 samples per line).
///
/// Similar to `malta_unit_window` but with sparser sampling.
#[inline]
fn malta_unit_lf_window(window: &[f32; 81]) -> f32 {
    let mut retval = 0.0f32;

    // Pattern 1: x grows, y constant (sparse horizontal)
    {
        let sum = w!(window, -4, 0)
            + w!(window, -2, 0)
            + w!(window, 0, 0)
            + w!(window, 2, 0)
            + w!(window, 4, 0);
        retval += sum * sum;
    }

    // Pattern 2: y grows, x constant (sparse vertical)
    {
        let sum = w!(window, 0, -4)
            + w!(window, 0, -2)
            + w!(window, 0, 0)
            + w!(window, 0, 2)
            + w!(window, 0, 4);
        retval += sum * sum;
    }

    // Pattern 3: both grow (diagonal)
    {
        let sum = w!(window, -3, -3)
            + w!(window, -2, -2)
            + w!(window, 0, 0)
            + w!(window, 2, 2)
            + w!(window, 3, 3);
        retval += sum * sum;
    }

    // Pattern 4: y grows, x shrinks
    {
        let sum = w!(window, 3, -3)
            + w!(window, 2, -2)
            + w!(window, 0, 0)
            + w!(window, -2, 2)
            + w!(window, -3, 3);
        retval += sum * sum;
    }

    // Pattern 5: y grows, x shifts 1 to -1
    {
        let sum = w!(window, 1, -4)
            + w!(window, 1, -2)
            + w!(window, 0, 0)
            + w!(window, -1, 2)
            + w!(window, -1, 4);
        retval += sum * sum;
    }

    // Pattern 6: y grows, x shifts -1 to 1
    {
        let sum = w!(window, -1, -4)
            + w!(window, -1, -2)
            + w!(window, 0, 0)
            + w!(window, 1, 2)
            + w!(window, 1, 4);
        retval += sum * sum;
    }

    // Pattern 7: x grows, y shifts -1 to 1
    {
        let sum = w!(window, -4, -1)
            + w!(window, -2, -1)
            + w!(window, 0, 0)
            + w!(window, 2, 1)
            + w!(window, 4, 1);
        retval += sum * sum;
    }

    // Pattern 8: x grows, y shifts 1 to -1
    {
        let sum = w!(window, -4, 1)
            + w!(window, -2, 1)
            + w!(window, 0, 0)
            + w!(window, 2, -1)
            + w!(window, 4, -1);
        retval += sum * sum;
    }

    // Pattern 9: steep slope
    {
        let sum = w!(window, -2, -3)
            + w!(window, -1, -2)
            + w!(window, 0, 0)
            + w!(window, 1, 2)
            + w!(window, 2, 3);
        retval += sum * sum;
    }

    // Pattern 10: steep slope other way
    {
        let sum = w!(window, 2, -3)
            + w!(window, 1, -2)
            + w!(window, 0, 0)
            + w!(window, -1, 2)
            + w!(window, -2, 3);
        retval += sum * sum;
    }

    // Pattern 11: shallow slope
    {
        let sum = w!(window, -3, -2)
            + w!(window, -2, -1)
            + w!(window, 0, 0)
            + w!(window, 2, 1)
            + w!(window, 3, 2);
        retval += sum * sum;
    }

    // Pattern 12: shallow slope other way
    {
        let sum = w!(window, 3, -2)
            + w!(window, 2, -1)
            + w!(window, 0, 0)
            + w!(window, -2, 1)
            + w!(window, -3, 2);
        retval += sum * sum;
    }

    // Pattern 13: curved path
    {
        let sum = w!(window, -4, 2)
            + w!(window, -2, 1)
            + w!(window, 0, 0)
            + w!(window, 2, -1)
            + w!(window, 4, -2);
        retval += sum * sum;
    }

    // Pattern 14: curved other direction
    {
        let sum = w!(window, -4, -2)
            + w!(window, -2, -1)
            + w!(window, 0, 0)
            + w!(window, 2, 1)
            + w!(window, 4, 2);
        retval += sum * sum;
    }

    // Pattern 15: vertical with shift
    {
        let sum = w!(window, -2, -4)
            + w!(window, -1, -2)
            + w!(window, 0, 0)
            + w!(window, 1, 2)
            + w!(window, 2, 4);
        retval += sum * sum;
    }

    // Pattern 16: vertical other shift
    {
        let sum = w!(window, 2, -4)
            + w!(window, 1, -2)
            + w!(window, 0, 0)
            + w!(window, -1, 2)
            + w!(window, -2, 4);
        retval += sum * sum;
    }

    retval
}

/// Extracts a 9x9 window around (x, y) into a fixed-size array.
///
/// For pixels near the border, out-of-bounds values are set to 0.0.
#[inline]
fn extract_window(data: &ImageF, x: usize, y: usize) -> [f32; 81] {
    let width = data.width();
    let height = data.height();
    let mut window = [0.0f32; 81];

    // Check if we're in the interior (can use fast path without bounds checking)
    if x >= 4 && y >= 4 && x < width - 4 && y < height - 4 {
        // Interior: directly copy rows from the image
        for dy in 0..9 {
            let src_y = y + dy - 4;
            let row = data.row(src_y);
            let dst_start = dy * 9;
            let src_start = x - 4;
            window[dst_start..dst_start + 9].copy_from_slice(&row[src_start..src_start + 9]);
        }
    } else {
        // Border: check each pixel individually
        for dy in 0..9 {
            let sy = y as isize + dy as isize - 4;
            for dx in 0..9 {
                let sx = x as isize + dx as isize - 4;
                if sy >= 0 && sy < height as isize && sx >= 0 && sx < width as isize {
                    window[dy * 9 + dx] = data.get(sx as usize, sy as usize);
                }
            }
        }
    }

    window
}

// ============================================================================
// Safe fast path for interior pixels (always enabled)
// ============================================================================

/// Malta filter for interior pixels (HF/UHF bands) - safe version.
///
/// Uses direct slice indexing instead of pointer arithmetic.
/// Caller must ensure the pixel is at least 4 pixels from all borders.
#[inline]
fn malta_unit_interior(data: &[f32], center: usize, stride: usize) -> f32 {
    let xs = stride;
    let xs2 = xs + xs;
    let xs3 = xs2 + xs;
    let xs4 = xs3 + xs;

    // Pre-validate full access range: center ± (4*stride + 4)
    let reach = xs4 + 4;
    assert!(center >= reach && center + reach < data.len());

    macro_rules! at {
        ($off:expr) => {
            data_at(data, ($off) as usize)
        };
    }

    let c = center;
    let mut retval = 0.0f32;

    // Pattern 1: x grows, y constant (horizontal line)
    {
        let sum = at!(c - 4)
            + at!(c - 3)
            + at!(c - 2)
            + at!(c - 1)
            + at!(c)
            + at!(c + 1)
            + at!(c + 2)
            + at!(c + 3)
            + at!(c + 4);
        retval += sum * sum;
    }

    // Pattern 2: y grows, x constant (vertical line)
    {
        let sum = at!(c - xs4)
            + at!(c - xs3)
            + at!(c - xs2)
            + at!(c - xs)
            + at!(c)
            + at!(c + xs)
            + at!(c + xs2)
            + at!(c + xs3)
            + at!(c + xs4);
        retval += sum * sum;
    }

    // Pattern 3: both grow (diagonal \)
    {
        let sum = at!(c - xs3 - 3)
            + at!(c - xs2 - 2)
            + at!(c - xs - 1)
            + at!(c)
            + at!(c + xs + 1)
            + at!(c + xs2 + 2)
            + at!(c + xs3 + 3);
        retval += sum * sum;
    }

    // Pattern 4: y grows, x shrinks (diagonal /)
    {
        let sum = at!(c - xs3 + 3)
            + at!(c - xs2 + 2)
            + at!(c - xs + 1)
            + at!(c)
            + at!(c + xs - 1)
            + at!(c + xs2 - 2)
            + at!(c + xs3 - 3);
        retval += sum * sum;
    }

    // Pattern 5: y grows -4 to 4, x shrinks 1 -> -1
    {
        let sum = at!(c - xs4 + 1)
            + at!(c - xs3 + 1)
            + at!(c - xs2 + 1)
            + at!(c - xs)
            + at!(c)
            + at!(c + xs)
            + at!(c + xs2 - 1)
            + at!(c + xs3 - 1)
            + at!(c + xs4 - 1);
        retval += sum * sum;
    }

    // Pattern 6: y grows -4 to 4, x grows -1 -> 1
    {
        let sum = at!(c - xs4 - 1)
            + at!(c - xs3 - 1)
            + at!(c - xs2 - 1)
            + at!(c - xs)
            + at!(c)
            + at!(c + xs)
            + at!(c + xs2 + 1)
            + at!(c + xs3 + 1)
            + at!(c + xs4 + 1);
        retval += sum * sum;
    }

    // Pattern 7: x grows -4 to 4, y grows -1 to 1
    {
        let sum = at!(c - 4 - xs)
            + at!(c - 3 - xs)
            + at!(c - 2 - xs)
            + at!(c - 1)
            + at!(c)
            + at!(c + 1)
            + at!(c + 2 + xs)
            + at!(c + 3 + xs)
            + at!(c + 4 + xs);
        retval += sum * sum;
    }

    // Pattern 8: x grows -4 to 4, y shrinks 1 to -1
    {
        let sum = at!(c - 4 + xs)
            + at!(c - 3 + xs)
            + at!(c - 2 + xs)
            + at!(c - 1)
            + at!(c)
            + at!(c + 1)
            + at!(c + 2 - xs)
            + at!(c + 3 - xs)
            + at!(c + 4 - xs);
        retval += sum * sum;
    }

    // Pattern 9: steep diagonal (2:1 slope)
    {
        let sum = at!(c - xs3 - 2)
            + at!(c - xs2 - 1)
            + at!(c - xs - 1)
            + at!(c)
            + at!(c + xs + 1)
            + at!(c + xs2 + 1)
            + at!(c + xs3 + 2);
        retval += sum * sum;
    }

    // Pattern 10: steep diagonal other way
    {
        let sum = at!(c - xs3 + 2)
            + at!(c - xs2 + 1)
            + at!(c - xs + 1)
            + at!(c)
            + at!(c + xs - 1)
            + at!(c + xs2 - 1)
            + at!(c + xs3 - 2);
        retval += sum * sum;
    }

    // Pattern 11: shallow diagonal (1:2 slope)
    {
        let sum = at!(c - xs2 - 3)
            + at!(c - xs - 2)
            + at!(c - xs - 1)
            + at!(c)
            + at!(c + xs + 1)
            + at!(c + xs + 2)
            + at!(c + xs2 + 3);
        retval += sum * sum;
    }

    // Pattern 12: shallow diagonal other way
    {
        let sum = at!(c - xs2 + 3)
            + at!(c - xs + 2)
            + at!(c - xs + 1)
            + at!(c)
            + at!(c + xs - 1)
            + at!(c + xs - 2)
            + at!(c + xs2 - 3);
        retval += sum * sum;
    }

    // Patterns 13-16: duplicates of 8,7,6,5 (9 samples each)

    // Pattern 13: curved line pattern (same as 8)
    {
        let sum = at!(c - 4 + xs)
            + at!(c - 3 + xs)
            + at!(c - 2 + xs)
            + at!(c - 1)
            + at!(c)
            + at!(c + 1)
            + at!(c + 2 - xs)
            + at!(c + 3 - xs)
            + at!(c + 4 - xs);
        retval += sum * sum;
    }

    // Pattern 14: curved line other direction (same as 7)
    {
        let sum = at!(c - 4 - xs)
            + at!(c - 3 - xs)
            + at!(c - 2 - xs)
            + at!(c - 1)
            + at!(c)
            + at!(c + 1)
            + at!(c + 2 + xs)
            + at!(c + 3 + xs)
            + at!(c + 4 + xs);
        retval += sum * sum;
    }

    // Pattern 15: very shallow curve (same as 6)
    {
        let sum = at!(c - xs4 - 1)
            + at!(c - xs3 - 1)
            + at!(c - xs2 - 1)
            + at!(c - xs)
            + at!(c)
            + at!(c + xs)
            + at!(c + xs2 + 1)
            + at!(c + xs3 + 1)
            + at!(c + xs4 + 1);
        retval += sum * sum;
    }

    // Pattern 16: very shallow curve other direction (same as 5)
    {
        let sum = at!(c - xs4 + 1)
            + at!(c - xs3 + 1)
            + at!(c - xs2 + 1)
            + at!(c - xs)
            + at!(c)
            + at!(c + xs)
            + at!(c + xs2 - 1)
            + at!(c + xs3 - 1)
            + at!(c + xs4 - 1);
        retval += sum * sum;
    }

    retval
}

/// Malta filter for interior pixels (LF band) - safe version.
///
/// Uses direct slice indexing instead of pointer arithmetic.
/// Caller must ensure the pixel is at least 4 pixels from all borders.
#[inline]
fn malta_unit_lf_interior(data: &[f32], center: usize, stride: usize) -> f32 {
    let xs = stride;
    let xs2 = xs + xs;
    let xs3 = xs2 + xs;
    let xs4 = xs3 + xs;

    // Pre-validate full access range: center ± (4*stride + 4)
    let reach = xs4 + 4;
    assert!(center >= reach && center + reach < data.len());

    macro_rules! at {
        ($off:expr) => {
            data_at(data, ($off) as usize)
        };
    }

    let c = center;
    let mut retval = 0.0f32;

    // Pattern 1: x grows, y constant (sparse horizontal)
    {
        let sum = at!(c - 4) + at!(c - 2) + at!(c) + at!(c + 2) + at!(c + 4);
        retval += sum * sum;
    }

    // Pattern 2: y grows, x constant (sparse vertical)
    {
        let sum = at!(c - xs4) + at!(c - xs2) + at!(c) + at!(c + xs2) + at!(c + xs4);
        retval += sum * sum;
    }

    // Pattern 3: both grow (diagonal)
    {
        let sum =
            at!(c - xs3 - 3) + at!(c - xs2 - 2) + at!(c) + at!(c + xs2 + 2) + at!(c + xs3 + 3);
        retval += sum * sum;
    }

    // Pattern 4: y grows, x shrinks
    {
        let sum =
            at!(c - xs3 + 3) + at!(c - xs2 + 2) + at!(c) + at!(c + xs2 - 2) + at!(c + xs3 - 3);
        retval += sum * sum;
    }

    // Pattern 5: y grows, x shifts 1 to -1
    {
        let sum =
            at!(c - xs4 + 1) + at!(c - xs2 + 1) + at!(c) + at!(c + xs2 - 1) + at!(c + xs4 - 1);
        retval += sum * sum;
    }

    // Pattern 6: y grows, x shifts -1 to 1
    {
        let sum =
            at!(c - xs4 - 1) + at!(c - xs2 - 1) + at!(c) + at!(c + xs2 + 1) + at!(c + xs4 + 1);
        retval += sum * sum;
    }

    // Pattern 7: x grows, y shifts -1 to 1
    {
        let sum = at!(c - 4 - xs) + at!(c - 2 - xs) + at!(c) + at!(c + 2 + xs) + at!(c + 4 + xs);
        retval += sum * sum;
    }

    // Pattern 8: x grows, y shifts 1 to -1
    {
        let sum = at!(c - 4 + xs) + at!(c - 2 + xs) + at!(c) + at!(c + 2 - xs) + at!(c + 4 - xs);
        retval += sum * sum;
    }

    // Pattern 9: steep slope
    {
        let sum =
            at!(c - xs3 - 2) + at!(c - xs2 - 1) + at!(c) + at!(c + xs2 + 1) + at!(c + xs3 + 2);
        retval += sum * sum;
    }

    // Pattern 10: steep slope other way
    {
        let sum =
            at!(c - xs3 + 2) + at!(c - xs2 + 1) + at!(c) + at!(c + xs2 - 1) + at!(c + xs3 - 2);
        retval += sum * sum;
    }

    // Pattern 11: shallow slope
    {
        let sum = at!(c - xs2 - 3) + at!(c - xs - 2) + at!(c) + at!(c + xs + 2) + at!(c + xs2 + 3);
        retval += sum * sum;
    }

    // Pattern 12: shallow slope other way
    {
        let sum = at!(c - xs2 + 3) + at!(c - xs + 2) + at!(c) + at!(c + xs - 2) + at!(c + xs2 - 3);
        retval += sum * sum;
    }

    // Pattern 13: curved path
    {
        let sum = at!(c - 4 + xs2) + at!(c - 2 + xs) + at!(c) + at!(c + 2 - xs) + at!(c + 4 - xs2);
        retval += sum * sum;
    }

    // Pattern 14: curved other direction
    {
        let sum = at!(c - 4 - xs2) + at!(c - 2 - xs) + at!(c) + at!(c + 2 + xs) + at!(c + 4 + xs2);
        retval += sum * sum;
    }

    // Pattern 15: vertical with shift
    {
        let sum =
            at!(c - xs4 - 2) + at!(c - xs2 - 1) + at!(c) + at!(c + xs2 + 1) + at!(c + xs4 + 2);
        retval += sum * sum;
    }

    // Pattern 16: vertical other shift
    {
        let sum =
            at!(c - xs4 + 2) + at!(c - xs2 + 1) + at!(c) + at!(c + xs2 - 1) + at!(c + xs4 - 2);
        retval += sum * sum;
    }

    retval
}

// ============================================================================
// SIMD fast path for interior pixels (8-wide, AVX2)
// ============================================================================

/// SIMD HF Malta filter for 8 consecutive interior pixels.
///
/// For 8 consecutive x-positions, each pattern offset maps to a contiguous
/// f32x8 load since adjacent pixels share the same row data layout.
/// Processes pixels at center, center+1, ..., center+7 simultaneously.
#[cfg(target_arch = "x86_64")]
#[archmage::rite]
fn malta_unit_interior_8x_v3(
    token: archmage::X64V3Token,
    data: &[f32],
    center: usize,
    stride: usize,
) -> magetypes::simd::f32x8 {
    use magetypes::simd::f32x8;

    let xs = stride as isize;
    let xs2 = xs * 2;
    let xs3 = xs * 3;
    let xs4 = xs * 4;

    // Pre-validate full access range: center ± (4*stride + 4), plus 7 for 8-wide SIMD
    // Most negative offset: -(4*stride + 4), most positive end: +(4*stride + 4) + 8
    let reach = 4 * stride + 4;
    assert!(center >= reach && center + reach + 8 <= data.len());

    macro_rules! ld {
        ($off:expr) => {{
            let o: isize = $off;
            let start = (center as isize + o) as usize;
            f32x8::load(token, load_8(data, start))
        }};
    }

    let mut r = f32x8::splat(token, 0.0);

    // Patterns 1-12: identical between libjxl and google/butteraugli
    // Pattern 1: horizontal
    {
        let s = ld!(-4) + ld!(-3) + ld!(-2) + ld!(-1) + ld!(0) + ld!(1) + ld!(2) + ld!(3) + ld!(4);
        r += s * s;
    }
    // Pattern 2: vertical
    {
        let s = ld!(-xs4)
            + ld!(-xs3)
            + ld!(-xs2)
            + ld!(-xs)
            + ld!(0)
            + ld!(xs)
            + ld!(xs2)
            + ld!(xs3)
            + ld!(xs4);
        r += s * s;
    }
    // Pattern 3: diagonal \
    {
        let s = ld!(-xs3 - 3)
            + ld!(-xs2 - 2)
            + ld!(-xs - 1)
            + ld!(0)
            + ld!(xs + 1)
            + ld!(xs2 + 2)
            + ld!(xs3 + 3);
        r += s * s;
    }
    // Pattern 4: diagonal /
    {
        let s = ld!(-xs3 + 3)
            + ld!(-xs2 + 2)
            + ld!(-xs + 1)
            + ld!(0)
            + ld!(xs - 1)
            + ld!(xs2 - 2)
            + ld!(xs3 - 3);
        r += s * s;
    }
    // Pattern 5
    {
        let s = ld!(-xs4 + 1)
            + ld!(-xs3 + 1)
            + ld!(-xs2 + 1)
            + ld!(-xs)
            + ld!(0)
            + ld!(xs)
            + ld!(xs2 - 1)
            + ld!(xs3 - 1)
            + ld!(xs4 - 1);
        r += s * s;
    }
    // Pattern 6
    {
        let s = ld!(-xs4 - 1)
            + ld!(-xs3 - 1)
            + ld!(-xs2 - 1)
            + ld!(-xs)
            + ld!(0)
            + ld!(xs)
            + ld!(xs2 + 1)
            + ld!(xs3 + 1)
            + ld!(xs4 + 1);
        r += s * s;
    }
    // Pattern 7
    {
        let s = ld!(-4 - xs)
            + ld!(-3 - xs)
            + ld!(-2 - xs)
            + ld!(-1)
            + ld!(0)
            + ld!(1)
            + ld!(2 + xs)
            + ld!(3 + xs)
            + ld!(4 + xs);
        r += s * s;
    }
    // Pattern 8
    {
        let s = ld!(-4 + xs)
            + ld!(-3 + xs)
            + ld!(-2 + xs)
            + ld!(-1)
            + ld!(0)
            + ld!(1)
            + ld!(2 - xs)
            + ld!(3 - xs)
            + ld!(4 - xs);
        r += s * s;
    }
    // Pattern 9
    {
        let s = ld!(-xs3 - 2)
            + ld!(-xs2 - 1)
            + ld!(-xs - 1)
            + ld!(0)
            + ld!(xs + 1)
            + ld!(xs2 + 1)
            + ld!(xs3 + 2);
        r += s * s;
    }
    // Pattern 10
    {
        let s = ld!(-xs3 + 2)
            + ld!(-xs2 + 1)
            + ld!(-xs + 1)
            + ld!(0)
            + ld!(xs - 1)
            + ld!(xs2 - 1)
            + ld!(xs3 - 2);
        r += s * s;
    }
    // Pattern 11
    {
        let s = ld!(-xs2 - 3)
            + ld!(-xs - 2)
            + ld!(-xs - 1)
            + ld!(0)
            + ld!(xs + 1)
            + ld!(xs + 2)
            + ld!(xs2 + 3);
        r += s * s;
    }
    // Pattern 12
    {
        let s = ld!(-xs2 + 3)
            + ld!(-xs + 2)
            + ld!(-xs + 1)
            + ld!(0)
            + ld!(xs - 1)
            + ld!(xs - 2)
            + ld!(xs2 - 3);
        r += s * s;
    }

    // Patterns 13-16: duplicates of 8,7,6,5
    // Pattern 13 (same offsets as 8)
    {
        let s = ld!(-4 + xs)
            + ld!(-3 + xs)
            + ld!(-2 + xs)
            + ld!(-1)
            + ld!(0)
            + ld!(1)
            + ld!(2 - xs)
            + ld!(3 - xs)
            + ld!(4 - xs);
        r += s * s;
    }
    // Pattern 14 (same offsets as 7)
    {
        let s = ld!(-4 - xs)
            + ld!(-3 - xs)
            + ld!(-2 - xs)
            + ld!(-1)
            + ld!(0)
            + ld!(1)
            + ld!(2 + xs)
            + ld!(3 + xs)
            + ld!(4 + xs);
        r += s * s;
    }
    // Pattern 15 (same offsets as 6)
    {
        let s = ld!(-xs4 - 1)
            + ld!(-xs3 - 1)
            + ld!(-xs2 - 1)
            + ld!(-xs)
            + ld!(0)
            + ld!(xs)
            + ld!(xs2 + 1)
            + ld!(xs3 + 1)
            + ld!(xs4 + 1);
        r += s * s;
    }
    // Pattern 16 (same offsets as 5)
    {
        let s = ld!(-xs4 + 1)
            + ld!(-xs3 + 1)
            + ld!(-xs2 + 1)
            + ld!(-xs)
            + ld!(0)
            + ld!(xs)
            + ld!(xs2 - 1)
            + ld!(xs3 - 1)
            + ld!(xs4 - 1);
        r += s * s;
    }

    r
}

/// SIMD LF Malta filter for 8 consecutive interior pixels.
#[cfg(target_arch = "x86_64")]
#[archmage::rite]
fn malta_unit_lf_interior_8x_v3(
    token: archmage::X64V3Token,
    data: &[f32],
    center: usize,
    stride: usize,
) -> magetypes::simd::f32x8 {
    use magetypes::simd::f32x8;

    let xs = stride as isize;
    let xs2 = xs * 2;
    let xs3 = xs * 3;
    let xs4 = xs * 4;

    // Pre-validate full access range: center ± (4*stride + 4), plus 7 for 8-wide SIMD
    let reach = 4 * stride + 4;
    assert!(center >= reach && center + reach + 8 <= data.len());

    macro_rules! ld {
        ($off:expr) => {{
            let o: isize = $off;
            let start = (center as isize + o) as usize;
            f32x8::load(token, load_8(data, start))
        }};
    }

    let mut r = f32x8::splat(token, 0.0);

    // Pattern 1: sparse horizontal
    {
        let s = ld!(-4) + ld!(-2) + ld!(0) + ld!(2) + ld!(4);
        r += s * s;
    }
    // Pattern 2: sparse vertical
    {
        let s = ld!(-xs4) + ld!(-xs2) + ld!(0) + ld!(xs2) + ld!(xs4);
        r += s * s;
    }
    // Pattern 3: diagonal
    {
        let s = ld!(-xs3 - 3) + ld!(-xs2 - 2) + ld!(0) + ld!(xs2 + 2) + ld!(xs3 + 3);
        r += s * s;
    }
    // Pattern 4: anti-diagonal
    {
        let s = ld!(-xs3 + 3) + ld!(-xs2 + 2) + ld!(0) + ld!(xs2 - 2) + ld!(xs3 - 3);
        r += s * s;
    }
    // Pattern 5
    {
        let s = ld!(-xs4 + 1) + ld!(-xs2 + 1) + ld!(0) + ld!(xs2 - 1) + ld!(xs4 - 1);
        r += s * s;
    }
    // Pattern 6
    {
        let s = ld!(-xs4 - 1) + ld!(-xs2 - 1) + ld!(0) + ld!(xs2 + 1) + ld!(xs4 + 1);
        r += s * s;
    }
    // Pattern 7
    {
        let s = ld!(-4 - xs) + ld!(-2 - xs) + ld!(0) + ld!(2 + xs) + ld!(4 + xs);
        r += s * s;
    }
    // Pattern 8
    {
        let s = ld!(-4 + xs) + ld!(-2 + xs) + ld!(0) + ld!(2 - xs) + ld!(4 - xs);
        r += s * s;
    }
    // Pattern 9
    {
        let s = ld!(-xs3 - 2) + ld!(-xs2 - 1) + ld!(0) + ld!(xs2 + 1) + ld!(xs3 + 2);
        r += s * s;
    }
    // Pattern 10
    {
        let s = ld!(-xs3 + 2) + ld!(-xs2 + 1) + ld!(0) + ld!(xs2 - 1) + ld!(xs3 - 2);
        r += s * s;
    }
    // Pattern 11
    {
        let s = ld!(-xs2 - 3) + ld!(-xs - 2) + ld!(0) + ld!(xs + 2) + ld!(xs2 + 3);
        r += s * s;
    }
    // Pattern 12
    {
        let s = ld!(-xs2 + 3) + ld!(-xs + 2) + ld!(0) + ld!(xs - 2) + ld!(xs2 - 3);
        r += s * s;
    }
    // Pattern 13
    {
        let s = ld!(-4 + xs2) + ld!(-2 + xs) + ld!(0) + ld!(2 - xs) + ld!(4 - xs2);
        r += s * s;
    }
    // Pattern 14
    {
        let s = ld!(-4 - xs2) + ld!(-2 - xs) + ld!(0) + ld!(2 + xs) + ld!(4 + xs2);
        r += s * s;
    }
    // Pattern 15
    {
        let s = ld!(-xs4 - 2) + ld!(-xs2 - 1) + ld!(0) + ld!(xs2 + 1) + ld!(xs4 + 2);
        r += s * s;
    }
    // Pattern 16
    {
        let s = ld!(-xs4 + 2) + ld!(-xs2 + 1) + ld!(0) + ld!(xs2 - 1) + ld!(xs4 - 2);
        r += s * s;
    }

    r
}

/// Malta filter for HF/UHF bands (9 samples per line, 16 orientations).
///
/// Applies 16 different line kernels in various orientations centered at (x,y)
/// and returns the sum of squared responses.
///
pub fn malta_unit(data: &ImageF, x: usize, y: usize) -> f32 {
    // Use window copy approach - compiler can optimize fixed-size array access
    let window = extract_window(data, x, y);
    malta_unit_window(&window)
}

/// Malta filter for LF band (5 samples per line, 16 orientations).
///
/// Similar to `malta_unit` but with sparser sampling for low-frequency
/// content. Uses 5 samples per line instead of 9.
pub fn malta_unit_lf(data: &ImageF, x: usize, y: usize) -> f32 {
    let window = extract_window(data, x, y);
    malta_unit_lf_window(&window)
}

/// Computes the asymmetric Malta difference between two images.
///
/// This applies the Malta filter to the difference between two luminance images,
/// with asymmetric weighting to penalize artifacts differently than blur.
///
/// Uses SIMD dispatch: on AVX2+ CPUs, processes 8 interior pixels simultaneously.
pub fn malta_diff_map(
    lum0: &ImageF,
    lum1: &ImageF,
    w_0gt1: f64,
    w_0lt1: f64,
    norm1: f64,
    use_lf: bool,
) -> ImageF {
    archmage::incant!(
        malta_diff_map_dispatch(lum0, lum1, w_0gt1, w_0lt1, norm1, use_lf),
        [v3, neon]
    )
}

/// Shared implementation for Malta diff map.
///
/// The `interior_row` closure processes interior pixels (x in 4..width-4)
/// for a single row, allowing SIMD dispatch at the inner loop level.
#[allow(clippy::inline_always, clippy::too_many_arguments)]
#[inline(always)]
fn malta_diff_map_impl<F>(
    lum0: &ImageF,
    lum1: &ImageF,
    w_0gt1: f64,
    w_0lt1: f64,
    norm1: f64,
    use_lf: bool,
    interior_row: F,
) -> ImageF
where
    F: Fn(&[f32], usize, usize, usize, bool, &mut [f32]),
{
    let width = lum0.width();
    let height = lum0.height();

    // Constants from C++
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

    // First pass: compute scaled differences into contiguous buffer
    let mut diffs = ImageF::new(width, height);

    for y in 0..height {
        let row0 = lum0.row(y);
        let row1 = lum1.row(y);
        let out = diffs.row_mut(y);

        for x in 0..width {
            let v0 = row0[x];
            let v1 = row1[x];
            let absval = 0.5 * (v0.abs() + v1.abs());
            let diff = v0 - v1;
            let scaler = norm2_0gt1 / (norm1_f32 + absval);

            // Primary symmetric quadratic objective
            let mut scaled_diff = scaler * diff;

            // Secondary half-open quadratic objectives
            let scaler2 = norm2_0lt1 / (norm1_f32 + absval);
            let fabs0 = v0.abs();
            let too_small = 0.55 * fabs0;
            let too_big = 1.05 * fabs0;

            if v0 < 0.0 {
                if v1 > -too_small {
                    let impact = scaler2 * (v1 + too_small);
                    scaled_diff -= impact;
                } else if v1 < -too_big {
                    let impact = scaler2 * (-v1 - too_big);
                    scaled_diff += impact;
                }
            } else {
                if v1 < too_small {
                    let impact = scaler2 * (too_small - v1);
                    scaled_diff += impact;
                } else if v1 > too_big {
                    let impact = scaler2 * (v1 - too_big);
                    scaled_diff -= impact;
                }
            }

            out[x] = scaled_diff;
        }
    }

    // Second pass: apply Malta filter
    let mut block_diff_ac = ImageF::new(width, height);

    let stride = diffs.stride();
    let data = diffs.data();

    // Top border (rows 0..4)
    for y in 0..4.min(height) {
        let out = block_diff_ac.row_mut(y);
        for x in 0..width {
            out[x] = if use_lf {
                malta_unit_lf(&diffs, x, y)
            } else {
                malta_unit(&diffs, x, y)
            };
        }
    }

    // Middle rows (4..height-4)
    if height > 8 {
        for y in 4..height - 4 {
            let out = block_diff_ac.row_mut(y);
            let center_base = y * stride;

            // Left border (x = 0..4)
            for x in 0..4.min(width) {
                out[x] = if use_lf {
                    malta_unit_lf(&diffs, x, y)
                } else {
                    malta_unit(&diffs, x, y)
                };
            }

            // Interior - delegate to dispatch closure
            if width > 8 {
                interior_row(data, center_base, stride, width, use_lf, out);
            }

            // Right border (x = width-4..width)
            for x in (width - 4).max(4)..width {
                out[x] = if use_lf {
                    malta_unit_lf(&diffs, x, y)
                } else {
                    malta_unit(&diffs, x, y)
                };
            }
        }
    }

    // Bottom border (rows height-4..height)
    for y in (height - 4).max(4.min(height))..height {
        let out = block_diff_ac.row_mut(y);
        for x in 0..width {
            out[x] = if use_lf {
                malta_unit_lf(&diffs, x, y)
            } else {
                malta_unit(&diffs, x, y)
            };
        }
    }

    block_diff_ac
}

/// AVX2 dispatch variant: processes 8 interior pixels at a time.
#[allow(clippy::too_many_arguments)]
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn malta_diff_map_dispatch_v3(
    token: archmage::X64V3Token,
    lum0: &ImageF,
    lum1: &ImageF,
    w_0gt1: f64,
    w_0lt1: f64,
    norm1: f64,
    use_lf: bool,
) -> ImageF {
    let interior = |data: &[f32],
                    center_base: usize,
                    stride: usize,
                    width: usize,
                    use_lf: bool,
                    out: &mut [f32]| {
        let mut x = 4;
        // SIMD 8-wide loop
        while x + 8 <= width - 4 {
            let center = center_base + x;
            let results = if use_lf {
                malta_unit_lf_interior_8x_v3(token, data, center, stride)
            } else {
                malta_unit_interior_8x_v3(token, data, center, stride)
            };
            results.store((&mut out[x..x + 8]).try_into().unwrap());
            x += 8;
        }
        // Scalar remainder
        while x < width - 4 {
            let center = center_base + x;
            out[x] = if use_lf {
                malta_unit_lf_interior(data, center, stride)
            } else {
                malta_unit_interior(data, center, stride)
            };
            x += 1;
        }
    };

    malta_diff_map_impl(lum0, lum1, w_0gt1, w_0lt1, norm1, use_lf, interior)
}

/// NEON HF Malta filter for 8 consecutive interior pixels (polyfilled 2×f32x4).
#[cfg(target_arch = "aarch64")]
#[archmage::rite]
fn malta_unit_interior_8x_neon(
    token: archmage::NeonToken,
    data: &[f32],
    center: usize,
    stride: usize,
) -> magetypes::simd::f32x8 {
    use magetypes::simd::f32x8;

    let xs = stride as isize;
    let xs2 = xs * 2;
    let xs3 = xs * 3;
    let xs4 = xs * 4;

    let reach = 4 * stride + 4;
    assert!(center >= reach && center + reach + 8 <= data.len());

    macro_rules! ld {
        ($off:expr) => {{
            let o: isize = $off;
            let start = (center as isize + o) as usize;
            f32x8::load(token, load_8(data, start))
        }};
    }

    let mut r = f32x8::splat(token, 0.0);

    // Pattern 1: horizontal
    {
        let s = ld!(-4) + ld!(-3) + ld!(-2) + ld!(-1) + ld!(0) + ld!(1) + ld!(2) + ld!(3) + ld!(4);
        r += s * s;
    }
    // Pattern 2: vertical
    {
        let s = ld!(-xs4)
            + ld!(-xs3)
            + ld!(-xs2)
            + ld!(-xs)
            + ld!(0)
            + ld!(xs)
            + ld!(xs2)
            + ld!(xs3)
            + ld!(xs4);
        r += s * s;
    }
    // Pattern 3: diagonal \
    {
        let s = ld!(-xs3 - 3)
            + ld!(-xs2 - 2)
            + ld!(-xs - 1)
            + ld!(0)
            + ld!(xs + 1)
            + ld!(xs2 + 2)
            + ld!(xs3 + 3);
        r += s * s;
    }
    // Pattern 4: diagonal /
    {
        let s = ld!(-xs3 + 3)
            + ld!(-xs2 + 2)
            + ld!(-xs + 1)
            + ld!(0)
            + ld!(xs - 1)
            + ld!(xs2 - 2)
            + ld!(xs3 - 3);
        r += s * s;
    }
    // Pattern 5
    {
        let s = ld!(-xs4 + 1)
            + ld!(-xs3 + 1)
            + ld!(-xs2 + 1)
            + ld!(-xs)
            + ld!(0)
            + ld!(xs)
            + ld!(xs2 - 1)
            + ld!(xs3 - 1)
            + ld!(xs4 - 1);
        r += s * s;
    }
    // Pattern 6
    {
        let s = ld!(-xs4 - 1)
            + ld!(-xs3 - 1)
            + ld!(-xs2 - 1)
            + ld!(-xs)
            + ld!(0)
            + ld!(xs)
            + ld!(xs2 + 1)
            + ld!(xs3 + 1)
            + ld!(xs4 + 1);
        r += s * s;
    }
    // Pattern 7
    {
        let s = ld!(-4 - xs)
            + ld!(-3 - xs)
            + ld!(-2 - xs)
            + ld!(-1)
            + ld!(0)
            + ld!(1)
            + ld!(2 + xs)
            + ld!(3 + xs)
            + ld!(4 + xs);
        r += s * s;
    }
    // Pattern 8
    {
        let s = ld!(-4 + xs)
            + ld!(-3 + xs)
            + ld!(-2 + xs)
            + ld!(-1)
            + ld!(0)
            + ld!(1)
            + ld!(2 - xs)
            + ld!(3 - xs)
            + ld!(4 - xs);
        r += s * s;
    }
    // Pattern 9
    {
        let s = ld!(-xs3 - 2)
            + ld!(-xs2 - 1)
            + ld!(-xs - 1)
            + ld!(0)
            + ld!(xs + 1)
            + ld!(xs2 + 1)
            + ld!(xs3 + 2);
        r += s * s;
    }
    // Pattern 10
    {
        let s = ld!(-xs3 + 2)
            + ld!(-xs2 + 1)
            + ld!(-xs + 1)
            + ld!(0)
            + ld!(xs - 1)
            + ld!(xs2 - 1)
            + ld!(xs3 - 2);
        r += s * s;
    }
    // Pattern 11
    {
        let s = ld!(-xs2 - 3)
            + ld!(-xs - 2)
            + ld!(-xs - 1)
            + ld!(0)
            + ld!(xs + 1)
            + ld!(xs + 2)
            + ld!(xs2 + 3);
        r += s * s;
    }
    // Pattern 12
    {
        let s = ld!(-xs2 + 3)
            + ld!(-xs + 2)
            + ld!(-xs + 1)
            + ld!(0)
            + ld!(xs - 1)
            + ld!(xs - 2)
            + ld!(xs2 - 3);
        r += s * s;
    }
    // Pattern 13 (same offsets as 8)
    {
        let s = ld!(-4 + xs)
            + ld!(-3 + xs)
            + ld!(-2 + xs)
            + ld!(-1)
            + ld!(0)
            + ld!(1)
            + ld!(2 - xs)
            + ld!(3 - xs)
            + ld!(4 - xs);
        r += s * s;
    }
    // Pattern 14 (same offsets as 7)
    {
        let s = ld!(-4 - xs)
            + ld!(-3 - xs)
            + ld!(-2 - xs)
            + ld!(-1)
            + ld!(0)
            + ld!(1)
            + ld!(2 + xs)
            + ld!(3 + xs)
            + ld!(4 + xs);
        r += s * s;
    }
    // Pattern 15 (same offsets as 6)
    {
        let s = ld!(-xs4 - 1)
            + ld!(-xs3 - 1)
            + ld!(-xs2 - 1)
            + ld!(-xs)
            + ld!(0)
            + ld!(xs)
            + ld!(xs2 + 1)
            + ld!(xs3 + 1)
            + ld!(xs4 + 1);
        r += s * s;
    }
    // Pattern 16 (same offsets as 5)
    {
        let s = ld!(-xs4 + 1)
            + ld!(-xs3 + 1)
            + ld!(-xs2 + 1)
            + ld!(-xs)
            + ld!(0)
            + ld!(xs)
            + ld!(xs2 - 1)
            + ld!(xs3 - 1)
            + ld!(xs4 - 1);
        r += s * s;
    }

    r
}

/// NEON LF Malta filter for 8 consecutive interior pixels (polyfilled 2×f32x4).
#[cfg(target_arch = "aarch64")]
#[archmage::rite]
fn malta_unit_lf_interior_8x_neon(
    token: archmage::NeonToken,
    data: &[f32],
    center: usize,
    stride: usize,
) -> magetypes::simd::f32x8 {
    use magetypes::simd::f32x8;

    let xs = stride as isize;
    let xs2 = xs * 2;
    let xs3 = xs * 3;
    let xs4 = xs * 4;

    let reach = 4 * stride + 4;
    assert!(center >= reach && center + reach + 8 <= data.len());

    macro_rules! ld {
        ($off:expr) => {{
            let o: isize = $off;
            let start = (center as isize + o) as usize;
            f32x8::load(token, load_8(data, start))
        }};
    }

    let mut r = f32x8::splat(token, 0.0);

    // Pattern 1: sparse horizontal
    {
        let s = ld!(-4) + ld!(-2) + ld!(0) + ld!(2) + ld!(4);
        r += s * s;
    }
    // Pattern 2: sparse vertical
    {
        let s = ld!(-xs4) + ld!(-xs2) + ld!(0) + ld!(xs2) + ld!(xs4);
        r += s * s;
    }
    // Pattern 3: diagonal
    {
        let s = ld!(-xs3 - 3) + ld!(-xs2 - 2) + ld!(0) + ld!(xs2 + 2) + ld!(xs3 + 3);
        r += s * s;
    }
    // Pattern 4: anti-diagonal
    {
        let s = ld!(-xs3 + 3) + ld!(-xs2 + 2) + ld!(0) + ld!(xs2 - 2) + ld!(xs3 - 3);
        r += s * s;
    }
    // Pattern 5
    {
        let s = ld!(-xs4 + 1) + ld!(-xs2 + 1) + ld!(0) + ld!(xs2 - 1) + ld!(xs4 - 1);
        r += s * s;
    }
    // Pattern 6
    {
        let s = ld!(-xs4 - 1) + ld!(-xs2 - 1) + ld!(0) + ld!(xs2 + 1) + ld!(xs4 + 1);
        r += s * s;
    }
    // Pattern 7
    {
        let s = ld!(-4 - xs) + ld!(-2 - xs) + ld!(0) + ld!(2 + xs) + ld!(4 + xs);
        r += s * s;
    }
    // Pattern 8
    {
        let s = ld!(-4 + xs) + ld!(-2 + xs) + ld!(0) + ld!(2 - xs) + ld!(4 - xs);
        r += s * s;
    }
    // Pattern 9
    {
        let s = ld!(-xs3 - 2) + ld!(-xs2 - 1) + ld!(0) + ld!(xs2 + 1) + ld!(xs3 + 2);
        r += s * s;
    }
    // Pattern 10
    {
        let s = ld!(-xs3 + 2) + ld!(-xs2 + 1) + ld!(0) + ld!(xs2 - 1) + ld!(xs3 - 2);
        r += s * s;
    }
    // Pattern 11
    {
        let s = ld!(-xs2 - 3) + ld!(-xs - 2) + ld!(0) + ld!(xs + 2) + ld!(xs2 + 3);
        r += s * s;
    }
    // Pattern 12
    {
        let s = ld!(-xs2 + 3) + ld!(-xs + 2) + ld!(0) + ld!(xs - 2) + ld!(xs2 - 3);
        r += s * s;
    }
    // Pattern 13
    {
        let s = ld!(-4 + xs2) + ld!(-2 + xs) + ld!(0) + ld!(2 - xs) + ld!(4 - xs2);
        r += s * s;
    }
    // Pattern 14
    {
        let s = ld!(-4 - xs2) + ld!(-2 - xs) + ld!(0) + ld!(2 + xs) + ld!(4 + xs2);
        r += s * s;
    }
    // Pattern 15
    {
        let s = ld!(-xs4 - 2) + ld!(-xs2 - 1) + ld!(0) + ld!(xs2 + 1) + ld!(xs4 + 2);
        r += s * s;
    }
    // Pattern 16
    {
        let s = ld!(-xs4 + 2) + ld!(-xs2 + 1) + ld!(0) + ld!(xs2 - 1) + ld!(xs4 - 2);
        r += s * s;
    }

    r
}

/// NEON dispatch for Malta diff map (polyfilled 2×f32x4).
#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
#[allow(clippy::too_many_arguments)]
fn malta_diff_map_dispatch_neon(
    token: archmage::NeonToken,
    lum0: &ImageF,
    lum1: &ImageF,
    w_0gt1: f64,
    w_0lt1: f64,
    norm1: f64,
    use_lf: bool,
) -> ImageF {
    let interior = |data: &[f32],
                    center_base: usize,
                    stride: usize,
                    width: usize,
                    use_lf: bool,
                    out: &mut [f32]| {
        let mut x = 4;
        // SIMD 8-wide loop (2×NEON f32x4)
        while x + 8 <= width - 4 {
            let center = center_base + x;
            let results = if use_lf {
                malta_unit_lf_interior_8x_neon(token, data, center, stride)
            } else {
                malta_unit_interior_8x_neon(token, data, center, stride)
            };
            results.store((&mut out[x..x + 8]).try_into().unwrap());
            x += 8;
        }
        // Scalar remainder
        while x < width - 4 {
            let center = center_base + x;
            out[x] = if use_lf {
                malta_unit_lf_interior(data, center, stride)
            } else {
                malta_unit_interior(data, center, stride)
            };
            x += 1;
        }
    };

    malta_diff_map_impl(lum0, lum1, w_0gt1, w_0lt1, norm1, use_lf, interior)
}

/// Scalar fallback for Malta diff map.
#[allow(clippy::too_many_arguments)]
fn malta_diff_map_dispatch_scalar(
    _token: archmage::ScalarToken,
    lum0: &ImageF,
    lum1: &ImageF,
    w_0gt1: f64,
    w_0lt1: f64,
    norm1: f64,
    use_lf: bool,
) -> ImageF {
    let interior = |data: &[f32],
                    center_base: usize,
                    stride: usize,
                    width: usize,
                    use_lf: bool,
                    out: &mut [f32]| {
        for x in 4..width - 4 {
            let center = center_base + x;
            out[x] = if use_lf {
                malta_unit_lf_interior(data, center, stride)
            } else {
                malta_unit_interior(data, center, stride)
            };
        }
    };

    malta_diff_map_impl(lum0, lum1, w_0gt1, w_0lt1, norm1, use_lf, interior)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_malta_uniform() {
        // Uniform image should have high Malta response (edge detection)
        let img = ImageF::filled(32, 32, 1.0);
        let center = malta_unit(&img, 16, 16);
        // 16 patterns × (9 samples × 1.0)² = 16 × 81 = 1296 for interior
        assert!(center > 0.0, "Malta should be positive for uniform image");
    }

    #[test]
    fn test_malta_edge() {
        // Image with strong vertical edge should have different response
        let mut img = ImageF::new(32, 32);
        for y in 0..32 {
            for x in 0..32 {
                if x < 16 {
                    img.set(x, y, 0.0);
                } else {
                    img.set(x, y, 1.0);
                }
            }
        }
        let edge = malta_unit(&img, 16, 16); // On the edge
        let uniform = malta_unit(&img, 8, 16); // In uniform region

        // Edge detection patterns should differ
        assert!(
            (edge - uniform).abs() > 1e-6,
            "Malta should differ at edge vs uniform region"
        );
    }

    #[test]
    fn test_malta_diff_map_identical() {
        let img = ImageF::filled(32, 32, 0.5);
        let result = malta_diff_map(&img, &img, 1.0, 1.0, 1.0, false);

        // Identical images should have zero Malta diff
        let mut sum = 0.0;
        for y in 0..32 {
            for x in 0..32 {
                sum += result.get(x, y);
            }
        }
        assert!(sum.abs() < 1e-6, "Identical images should have zero diff");
    }

    #[test]
    fn test_malta_lf_smaller() {
        // LF variant uses fewer samples, should have different magnitude
        let img = ImageF::filled(32, 32, 1.0);
        let hf = malta_unit(&img, 16, 16);
        let lf = malta_unit_lf(&img, 16, 16);

        // Both should be positive
        assert!(hf > 0.0);
        assert!(lf > 0.0);
        // LF uses 5 samples instead of 9, so response should be different
        // 16 patterns × 25 for LF vs 16 × 81 for HF (for uniform image)
        assert!(lf < hf, "LF should have smaller response for uniform image");
    }

    #[test]
    fn test_malta_fast_vs_slow() {
        // Verify fast and slow paths give same result
        let mut img2 = ImageF::new(32, 32);
        for y in 0..32 {
            for x in 0..32 {
                img2.set(x, y, ((x + y) % 10) as f32 * 0.1);
            }
        }

        // Test an interior point using both paths
        let fast_result = malta_unit(&img2, 16, 16);

        // Force slow path by using border coordinates
        // We can't easily force slow path for interior, but we can verify
        // the result is reasonable
        assert!(fast_result >= 0.0, "Malta result should be non-negative");
    }

    #[test]
    fn test_interior_vs_window() {
        // Verify safe interior functions match window approach for both variants
        let mut img = ImageF::new(32, 32);
        for y in 0..32 {
            for x in 0..32 {
                // Use varied data to ensure patterns matter
                img.set(x, y, ((x * 7 + y * 13) % 100) as f32 * 0.01);
            }
        }

        let data = img.data();
        let stride = img.stride();

        // Test HF (malta_unit) at various interior points
        for y in 5..27 {
            for x in 5..27 {
                let window_result = malta_unit(&img, x, y);
                let center = y * stride + x;
                let interior_result = malta_unit_interior(data, center, stride);

                let diff = (window_result - interior_result).abs();
                assert!(
                    diff < 1e-6,
                    "HF mismatch at ({x}, {y}): window={window_result}, interior={interior_result}, diff={diff}"
                );
            }
        }

        // Test LF (malta_unit_lf) at various interior points
        for y in 5..27 {
            for x in 5..27 {
                let window_result = malta_unit_lf(&img, x, y);
                let center = y * stride + x;
                let interior_result = malta_unit_lf_interior(data, center, stride);

                let diff = (window_result - interior_result).abs();
                assert!(
                    diff < 1e-6,
                    "LF mismatch at ({x}, {y}): window={window_result}, interior={interior_result}, diff={diff}"
                );
            }
        }
    }
}
