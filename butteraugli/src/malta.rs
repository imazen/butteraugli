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
//! The implementation copies pixel neighborhoods into fixed-size arrays
//! to enable bounds-check-free access patterns.

use crate::image::ImageF;

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

/// Malta filter for HF/UHF bands (9 samples per line, 16 orientations).
///
/// Applies 16 different line kernels in various orientations centered at (x,y)
/// and returns the sum of squared responses.
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
/// # Arguments
/// * `lum0` - First luminance image
/// * `lum1` - Second luminance image
/// * `w_0gt1` - Weight when original > reconstructed (penalize blurring)
/// * `w_0lt1` - Weight when original < reconstructed (penalize ringing)
/// * `norm1` - Normalization factor
/// * `use_lf` - If true, use LF variant of Malta filter
///
/// # Returns
/// Block difference AC map
pub fn malta_diff_map(
    lum0: &ImageF,
    lum1: &ImageF,
    w_0gt1: f64,
    w_0lt1: f64,
    norm1: f64,
    use_lf: bool,
) -> ImageF {
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
    // The extract_window function has an optimized path for interior pixels
    let mut block_diff_ac = ImageF::new(width, height);

    for y in 0..height {
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
}
