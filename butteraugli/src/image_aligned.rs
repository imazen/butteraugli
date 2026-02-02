//! SIMD-aligned image buffer types for butteraugli.
//!
//! These types use the `simd_aligned` crate to guarantee proper memory
//! alignment for SIMD operations. This eliminates unaligned loads/stores
//! and enables more efficient vectorization.
//!
//! The backing store is aligned to 32 bytes (AVX2) with rows padded to
//! multiples of 8 floats (f32x8).

use simd_aligned::{MatSimd, Rows};
use wide::f32x8;

/// Single-channel floating point image with SIMD-aligned storage.
///
/// Uses `MatSimd<f32x8, Rows>` as backing store, ensuring:
/// - 32-byte aligned rows (for AVX2 loads/stores)
/// - Row stride is multiple of 8 floats
/// - Can access as `&[f32]` (flat) or `&[f32x8]` (SIMD)
#[derive(Debug, Clone)]
pub struct AlignedImageF {
    data: MatSimd<f32x8, Rows>,
    width: usize,
    height: usize,
}

impl AlignedImageF {
    /// Creates a new image filled with zeros.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        // MatSimd::with_dimension(width, height) - rows x cols for Rows strategy
        let data = MatSimd::with_dimension(height, width);
        Self {
            data,
            width,
            height,
        }
    }

    /// Creates an image filled with a constant value.
    #[must_use]
    pub fn filled(width: usize, height: usize, value: f32) -> Self {
        let mut img = Self::new(width, height);
        img.fill(value);
        img
    }

    /// Image width in pixels.
    #[inline]
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Image height in pixels.
    #[inline]
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Number of f32x8 SIMD vectors per row (including padding).
    #[inline]
    #[must_use]
    pub fn simd_width(&self) -> usize {
        self.data.row(0).len()
    }

    /// Returns a flat view of a row (only valid pixels, no padding).
    #[inline]
    #[must_use]
    pub fn row(&self, y: usize) -> &[f32] {
        &self.data.row_as_flat(y)[..self.width]
    }

    /// Returns a mutable flat view of a row.
    #[inline]
    pub fn row_mut(&mut self, y: usize) -> &mut [f32] {
        &mut self.data.row_as_flat_mut(y)[..self.width]
    }

    /// Returns a SIMD view of a row (f32x8 slices, includes padding).
    #[inline]
    #[must_use]
    pub fn row_simd(&self, y: usize) -> &[f32x8] {
        self.data.row(y)
    }

    /// Returns a mutable SIMD view of a row.
    #[inline]
    pub fn row_simd_mut(&mut self, y: usize) -> &mut [f32x8] {
        self.data.row_mut(y)
    }

    /// Gets a pixel value.
    #[inline]
    #[must_use]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data.flat()[(y, x)]
    }

    /// Sets a pixel value.
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, value: f32) {
        self.data.flat_mut()[(y, x)] = value;
    }

    /// Returns the underlying MatSimd for direct SIMD access.
    #[inline]
    #[must_use]
    pub fn as_simd(&self) -> &MatSimd<f32x8, Rows> {
        &self.data
    }

    /// Returns the underlying MatSimd for direct mutable SIMD access.
    #[inline]
    pub fn as_simd_mut(&mut self) -> &mut MatSimd<f32x8, Rows> {
        &mut self.data
    }

    /// Checks if two images have the same dimensions.
    #[must_use]
    pub fn same_size(&self, other: &Self) -> bool {
        self.width == other.width && self.height == other.height
    }

    /// Copies data from another image row-by-row.
    ///
    /// # Panics
    /// Panics if dimensions don't match.
    pub fn copy_from(&mut self, other: &Self) {
        assert!(self.same_size(other));
        for y in 0..self.height {
            self.row_mut(y).copy_from_slice(other.row(y));
        }
    }

    /// Fills the image with a constant value.
    pub fn fill(&mut self, value: f32) {
        let simd_val = f32x8::splat(value);
        for y in 0..self.height {
            for chunk in self.row_simd_mut(y) {
                *chunk = simd_val;
            }
        }
    }

    /// Copies data from an AlignedImageF using SIMD.
    ///
    /// This is faster than copy_from for aligned data.
    pub fn copy_from_simd(&mut self, other: &Self) {
        assert!(self.same_size(other));
        for y in 0..self.height {
            let src = other.row_simd(y);
            let dst = self.row_simd_mut(y);
            dst.copy_from_slice(src);
        }
    }
}

/// Three-channel floating point image with SIMD-aligned storage.
#[derive(Debug, Clone)]
pub struct AlignedImage3F {
    planes: [AlignedImageF; 3],
}

impl AlignedImage3F {
    /// Creates a new 3-channel image.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            planes: [
                AlignedImageF::new(width, height),
                AlignedImageF::new(width, height),
                AlignedImageF::new(width, height),
            ],
        }
    }

    /// Image width.
    #[inline]
    #[must_use]
    pub fn width(&self) -> usize {
        self.planes[0].width()
    }

    /// Image height.
    #[inline]
    #[must_use]
    pub fn height(&self) -> usize {
        self.planes[0].height()
    }

    /// Returns a reference to a specific plane.
    #[inline]
    #[must_use]
    pub fn plane(&self, index: usize) -> &AlignedImageF {
        &self.planes[index]
    }

    /// Returns a mutable reference to a specific plane.
    #[inline]
    pub fn plane_mut(&mut self, index: usize) -> &mut AlignedImageF {
        &mut self.planes[index]
    }

    /// Returns mutable references to all three planes simultaneously.
    #[inline]
    pub fn planes_mut(&mut self) -> (&mut AlignedImageF, &mut AlignedImageF, &mut AlignedImageF) {
        let [p0, p1, p2] = &mut self.planes;
        (p0, p1, p2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_image_creation() {
        let img = AlignedImageF::new(100, 50);
        assert_eq!(img.width(), 100);
        assert_eq!(img.height(), 50);
    }

    #[test]
    fn test_pixel_access() {
        let mut img = AlignedImageF::new(10, 10);
        img.set(5, 3, 42.0);
        assert!((img.get(5, 3) - 42.0).abs() < 0.001);
    }

    #[test]
    fn test_row_access() {
        let mut img = AlignedImageF::new(10, 10);
        img.row_mut(5)[3] = 99.0;
        assert!((img.row(5)[3] - 99.0).abs() < 0.001);
    }

    #[test]
    fn test_simd_access() {
        let mut img = AlignedImageF::new(16, 4);

        // Fill using SIMD
        let val = f32x8::splat(7.0);
        for chunk in img.row_simd_mut(0) {
            *chunk = val;
        }

        // Verify via scalar access
        assert!((img.get(0, 0) - 7.0).abs() < 0.001);
        assert!((img.get(7, 0) - 7.0).abs() < 0.001);
    }

    #[test]
    fn test_copy_from_simd() {
        let src = AlignedImageF::filled(32, 32, 5.0);
        let mut dst = AlignedImageF::new(32, 32);

        dst.copy_from_simd(&src);

        for y in 0..32 {
            for x in 0..32 {
                assert!((dst.get(x, y) - 5.0).abs() < 0.001);
            }
        }
    }

    #[test]
    fn test_aligned_image3f() {
        let img = AlignedImage3F::new(100, 50);
        assert_eq!(img.width(), 100);
        assert_eq!(img.height(), 50);
    }

    #[test]
    fn test_alignment() {
        let img = AlignedImageF::new(100, 50);
        // Check that SIMD row access works
        let row = img.row_simd(0);
        assert!(row.len() >= 100_usize.div_ceil(8));
    }
}
