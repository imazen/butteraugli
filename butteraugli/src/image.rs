//! Image buffer types for butteraugli.
//!
//! These types provide efficient storage for floating-point image data
//! with row-stride support for cache-friendly access patterns.

use imgref::ImgVec;
use std::ops::{Index, IndexMut};
use std::sync::Mutex;

/// Reusable buffer pool for `ImageF` allocations.
///
/// Avoids repeated mmap/munmap for large temporary buffers. Thread-safe via
/// `Mutex`, so it can be shared across rayon tasks. Owned by the caller
/// (typically `ButteraugliReference` or a local in standalone API functions).
/// When the pool is dropped, all cached buffers are freed.
pub struct BufferPool {
    buffers: Mutex<Vec<Vec<f32>>>,
}

impl core::fmt::Debug for BufferPool {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let count = self.buffers.lock().unwrap().len();
        f.debug_struct("BufferPool")
            .field("cached_buffers", &count)
            .finish()
    }
}

impl Default for BufferPool {
    fn default() -> Self {
        Self {
            buffers: Mutex::new(Vec::new()),
        }
    }
}

impl BufferPool {
    /// Creates a new empty buffer pool.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Takes a buffer of at least `needed` elements from the pool (best-fit).
    /// Returns stale data — caller must zero-fill if needed.
    ///
    /// With `unsafe-performance`, new allocations skip zero-fill entirely.
    pub(crate) fn take(&self, needed: usize) -> Vec<f32> {
        let mut pool = self.buffers.lock().unwrap();
        let mut best_idx = None;
        let mut best_excess = usize::MAX;
        for (i, buf) in pool.iter().enumerate() {
            let cap = buf.len();
            if cap >= needed && cap - needed < best_excess {
                best_idx = Some(i);
                best_excess = cap - needed;
            }
        }
        if let Some(idx) = best_idx {
            let mut buf = pool.swap_remove(idx);
            drop(pool); // release lock before potential realloc
            buf.truncate(needed);
            if buf.len() < needed {
                #[cfg(feature = "unsafe-performance")]
                #[allow(clippy::uninit_vec)]
                {
                    buf.reserve(needed - buf.len());
                    // SAFETY: f32 has no validity invariant beyond being initialized.
                    // Callers (from_pool_dirty) overwrite all data before reading.
                    unsafe { buf.set_len(needed) };
                }
                #[cfg(not(feature = "unsafe-performance"))]
                buf.resize(needed, 0.0);
            }
            buf
        } else {
            drop(pool); // release lock before allocation
            #[cfg(feature = "unsafe-performance")]
            #[allow(clippy::uninit_vec)]
            {
                let mut buf = Vec::with_capacity(needed);
                // SAFETY: f32 has no validity invariant beyond being initialized.
                // Callers (from_pool_dirty) overwrite all data before reading.
                unsafe { buf.set_len(needed) };
                buf
            }
            #[cfg(not(feature = "unsafe-performance"))]
            {
                vec![0.0; needed]
            }
        }
    }

    /// Returns a buffer to the pool. Dropped silently if pool is full (48 buffers).
    pub(crate) fn put(&self, buf: Vec<f32>) {
        let mut pool = self.buffers.lock().unwrap();
        if pool.len() < 48 {
            pool.push(buf);
        }
    }
}

impl Clone for BufferPool {
    /// Clone creates a fresh empty pool — the clone does not share buffers.
    fn clone(&self) -> Self {
        Self::new()
    }
}

/// Single-channel floating point image.
///
/// This is the primary image type used throughout butteraugli for
/// intermediate computations. It stores pixel values as f32 with
/// optional row padding for alignment.
#[derive(Debug, Clone)]
pub struct ImageF {
    data: Vec<f32>,
    width: usize,
    height: usize,
    stride: usize, // pixels per row (may be > width for alignment)
}

impl ImageF {
    /// Creates a new image filled with zeros.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        // Align stride to 16 floats (64 bytes) for SIMD
        let stride = (width + 15) & !15;
        Self {
            data: vec![0.0; stride * height],
            width,
            height,
            stride,
        }
    }

    /// Creates a new image WITHOUT zero-filling (when `unsafe-performance` is enabled).
    ///
    /// Without the feature, this falls back to zero-filled allocation (same as `new`).
    /// Use this only when the allocation will be fully populated before reading
    /// (e.g., blur output, copy targets, format conversion outputs).
    #[must_use]
    pub(crate) fn new_uninit(width: usize, height: usize) -> Self {
        let stride = (width + 15) & !15;
        let needed = stride * height;
        #[cfg(feature = "unsafe-performance")]
        let data = {
            let mut buf = Vec::with_capacity(needed);
            // SAFETY: f32 has no validity invariant beyond being initialized memory.
            // Any bit pattern is a valid f32 (including NaN/Inf). Callers must
            // overwrite all data before reading.
            #[allow(clippy::uninit_vec)]
            unsafe {
                buf.set_len(needed);
            }
            buf
        };
        #[cfg(not(feature = "unsafe-performance"))]
        let data = vec![0.0; needed];
        Self {
            data,
            width,
            height,
            stride,
        }
    }

    /// Creates an image from existing data.
    ///
    /// # Panics
    /// Panics if data length doesn't match width * height.
    #[must_use]
    pub fn from_vec(data: Vec<f32>, width: usize, height: usize) -> Self {
        assert_eq!(data.len(), width * height);
        // For imported data, don't add padding
        Self {
            data,
            width,
            height,
            stride: width,
        }
    }

    /// Creates an image with padding for aligned rows.
    #[must_use]
    pub fn from_vec_padded(data: Vec<f32>, width: usize, height: usize, stride: usize) -> Self {
        assert!(stride >= width);
        assert_eq!(data.len(), stride * height);
        Self {
            data,
            width,
            height,
            stride,
        }
    }

    /// Creates an image filled with a constant value.
    #[must_use]
    pub fn filled(width: usize, height: usize, value: f32) -> Self {
        let stride = (width + 15) & !15;
        Self {
            data: vec![value; stride * height],
            width,
            height,
            stride,
        }
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

    /// Number of pixels per row (may include padding).
    #[inline]
    #[must_use]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Returns a reference to a row.
    #[inline]
    #[must_use]
    pub fn row(&self, y: usize) -> &[f32] {
        let start = y * self.stride;
        &self.data[start..start + self.width]
    }

    /// Returns a mutable reference to a row.
    #[inline]
    pub fn row_mut(&mut self, y: usize) -> &mut [f32] {
        let start = y * self.stride;
        &mut self.data[start..start + self.width]
    }

    /// Returns a reference to a full row including padding.
    #[inline]
    #[must_use]
    pub fn row_full(&self, y: usize) -> &[f32] {
        let start = y * self.stride;
        &self.data[start..start + self.stride]
    }

    /// Returns a mutable reference to a full row including padding.
    #[inline]
    pub fn row_full_mut(&mut self, y: usize) -> &mut [f32] {
        let start = y * self.stride;
        &mut self.data[start..start + self.stride]
    }

    /// Gets a pixel value.
    #[inline]
    #[must_use]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.stride + x]
    }

    /// Sets a pixel value.
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, value: f32) {
        self.data[y * self.stride + x] = value;
    }

    /// Gets a pixel value without bounds checking.
    ///
    /// # Safety
    /// Caller must ensure `y * stride + x < data.len()`.
    #[cfg(feature = "unsafe-performance")]
    #[allow(clippy::inline_always)]
    #[inline(always)]
    #[must_use]
    pub(crate) unsafe fn get_unchecked(&self, x: usize, y: usize) -> f32 {
        // SAFETY: caller asserts y * stride + x < data.len()
        unsafe { *self.data.get_unchecked(y * self.stride + x) }
    }

    /// Sets a pixel value without bounds checking.
    ///
    /// # Safety
    /// Caller must ensure `y * stride + x < data.len()`.
    #[cfg(feature = "unsafe-performance")]
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub(crate) unsafe fn set_unchecked(&mut self, x: usize, y: usize, value: f32) {
        // SAFETY: caller asserts y * stride + x < data.len()
        unsafe { *self.data.get_unchecked_mut(y * self.stride + x) = value };
    }

    /// Returns a row slice without bounds checking.
    ///
    /// # Safety
    /// Caller must ensure `y < height`.
    #[cfg(feature = "unsafe-performance")]
    #[allow(clippy::inline_always)]
    #[inline(always)]
    #[must_use]
    pub(crate) unsafe fn row_unchecked(&self, y: usize) -> &[f32] {
        let start = y * self.stride;
        // SAFETY: caller asserts y < height
        unsafe { self.data.get_unchecked(start..start + self.width) }
    }

    /// Returns a mutable row slice without bounds checking.
    ///
    /// # Safety
    /// Caller must ensure `y < height`.
    #[cfg(feature = "unsafe-performance")]
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub(crate) unsafe fn row_mut_unchecked(&mut self, y: usize) -> &mut [f32] {
        let start = y * self.stride;
        // SAFETY: caller asserts y < height
        unsafe { self.data.get_unchecked_mut(start..start + self.width) }
    }

    /// Returns a raw pointer to the data for SIMD operations.
    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }

    /// Returns a mutable raw pointer to the data for SIMD operations.
    #[inline]
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.data.as_mut_ptr()
    }

    /// Returns the raw data as a slice.
    #[inline]
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Returns the raw data as a mutable slice.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Creates a new image using a pooled buffer if available, WITHOUT zero-filling.
    ///
    /// The buffer may contain stale data from a previous use. Only use this when
    /// the caller will overwrite every pixel (e.g., blur output buffers).
    #[must_use]
    pub fn from_pool_dirty(width: usize, height: usize, pool: &BufferPool) -> Self {
        let stride = (width + 15) & !15;
        let needed = stride * height;
        let data = pool.take(needed);

        Self {
            data,
            width,
            height,
            stride,
        }
    }

    /// Creates a new zero-filled image using a pooled buffer if available.
    ///
    /// Use for accumulator images that need zeroing before `+=` operations.
    /// Saves mmap/munmap syscalls on repeated calls when the pool is warm.
    #[must_use]
    pub fn from_pool_zeroed(width: usize, height: usize, pool: &BufferPool) -> Self {
        let mut img = Self::from_pool_dirty(width, height, pool);
        img.data.fill(0.0);
        img
    }

    /// Returns the internal buffer to the pool for reuse.
    ///
    /// If the pool is full (32 buffers), the buffer is dropped normally.
    /// Forgetting to call this is not a bug — the buffer is freed as usual.
    pub fn recycle(self, pool: &BufferPool) {
        pool.put(self.data);
    }

    /// Checks if two images have the same dimensions.
    #[must_use]
    pub fn same_size(&self, other: &Self) -> bool {
        self.width == other.width && self.height == other.height
    }

    /// Copies data from another image.
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
        self.data.fill(value);
    }

    /// Borrows a zero-allocation row-slice view of this image.
    ///
    /// The returned `StripView<'_>` exposes a subset of rows `[top_row,
    /// bottom_row)` (half-open) of the underlying image as a read-only
    /// view that mirrors the `ImageF` row-access API. Local index `0`
    /// in the strip view maps to row `top_row` in the parent.
    ///
    /// Stride and width are preserved verbatim from the parent — kernels
    /// operating on the strip see the same row padding the parent has.
    ///
    /// Cost: O(1) — borrows a slice and copies width/stride scalars.
    /// No allocation, no copy.
    ///
    /// # Halo bookkeeping
    ///
    /// The view records `start_row_in_parent = top_row` so subsequent
    /// strip-tile passes can compute "is this strip near the top/bottom
    /// image edge?" for mirror-edge halo handling.
    ///
    /// # Panics
    ///
    /// Panics if `top_row > bottom_row` or `bottom_row > self.height()`.
    /// An empty strip (`top_row == bottom_row`) is permitted and returns
    /// a view of height 0.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // (requires the `internals` cargo feature to expose `ImageF`)
    /// use butteraugli::image::ImageF;
    /// let mut img = ImageF::new(8, 8);
    /// img.set(2, 5, 42.0);
    /// // Borrow rows 4..6 — strip-local row 1 == parent row 5.
    /// let strip = img.strip_view(4, 6);
    /// assert_eq!(strip.height(), 2);
    /// assert_eq!(strip.start_row_in_parent(), 4);
    /// assert_eq!(strip.row(1)[2], 42.0);
    /// ```
    #[must_use]
    pub fn strip_view(&self, top_row: usize, bottom_row: usize) -> StripView<'_> {
        assert!(
            top_row <= bottom_row,
            "strip_view: top_row ({top_row}) > bottom_row ({bottom_row})"
        );
        assert!(
            bottom_row <= self.height,
            "strip_view: bottom_row ({bottom_row}) > height ({})",
            self.height
        );
        let height = bottom_row - top_row;
        let start = top_row * self.stride;
        let end = start + height * self.stride;
        // For an empty strip (height == 0), `start == end` and the slice
        // is empty — still a valid &[f32].
        let data = &self.data[start..end];
        StripView {
            data,
            width: self.width,
            height,
            stride: self.stride,
            start_row_in_parent: top_row,
        }
    }

    /// Borrows a zero-allocation mutable row-slice view of this image.
    ///
    /// Mutable counterpart to [`ImageF::strip_view`]. The returned
    /// `StripViewMut<'_>` permits writes to the strip's rows; writes
    /// land directly in the parent's backing buffer.
    ///
    /// # Panics
    ///
    /// Same conditions as [`ImageF::strip_view`].
    pub fn strip_view_mut(&mut self, top_row: usize, bottom_row: usize) -> StripViewMut<'_> {
        assert!(
            top_row <= bottom_row,
            "strip_view_mut: top_row ({top_row}) > bottom_row ({bottom_row})"
        );
        assert!(
            bottom_row <= self.height,
            "strip_view_mut: bottom_row ({bottom_row}) > height ({})",
            self.height
        );
        let height = bottom_row - top_row;
        let start = top_row * self.stride;
        let end = start + height * self.stride;
        let width = self.width;
        let stride = self.stride;
        let data = &mut self.data[start..end];
        StripViewMut {
            data,
            width,
            height,
            stride,
            start_row_in_parent: top_row,
        }
    }

    /// Converts this image to an `ImgVec<f32>` for return to users.
    ///
    /// If the image has padding (stride > width), the data is copied
    /// to remove the padding. Otherwise, ownership is transferred directly.
    #[must_use]
    pub(crate) fn into_imgvec(self) -> ImgVec<f32> {
        if self.stride == self.width {
            ImgVec::new(self.data, self.width, self.height)
        } else {
            // Copy to remove padding
            let mut out = Vec::with_capacity(self.width * self.height);
            for y in 0..self.height {
                let start = y * self.stride;
                out.extend_from_slice(&self.data[start..start + self.width]);
            }
            ImgVec::new(out, self.width, self.height)
        }
    }
}

impl Index<(usize, usize)> for ImageF {
    type Output = f32;

    #[inline]
    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        &self.data[y * self.stride + x]
    }
}

impl IndexMut<(usize, usize)> for ImageF {
    #[inline]
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut Self::Output {
        &mut self.data[y * self.stride + x]
    }
}

/// Borrowed read-only view of a row-range of an [`ImageF`].
///
/// Returned from [`ImageF::strip_view`]. Exposes the same row-access API
/// as [`ImageF`] but bounded to the strip's local `0..height` row range.
/// All row accesses index into the parent [`ImageF`]'s backing buffer
/// without copying.
///
/// The `start_row_in_parent` field lets halo arithmetic compute whether
/// this strip touches the top/bottom edge of the parent image (which
/// callers need for mirror-edge halo handling in blur kernels).
///
/// # Stride
///
/// The view preserves the parent's `stride` verbatim. SIMD kernels that
/// stride-walk row pointers (e.g. by `stride` bytes per step) see the
/// same padding behavior they'd see on the parent — strip_view does NOT
/// rewrite or pack rows.
///
/// # Safety
///
/// All accessors bounds-check against the strip's local `height`. Indexing
/// outside `0..height` is a hard panic, NOT undefined behaviour.
#[derive(Debug)]
pub struct StripView<'a> {
    /// Borrowed slice covering exactly `height * stride` f32s of the parent.
    /// Length is `0` for an empty strip.
    data: &'a [f32],
    /// Width in pixels, copied from the parent.
    width: usize,
    /// Strip height = `bottom_row - top_row` from the borrowing call.
    height: usize,
    /// Stride in pixels, copied from the parent (may exceed `width` for SIMD padding).
    stride: usize,
    /// Parent row index that strip-local row `0` corresponds to.
    /// Used by callers for halo-mirror edge-detection.
    start_row_in_parent: usize,
}

impl StripView<'_> {
    /// Strip width in pixels (matches parent's `width`).
    #[inline]
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Strip height in rows. Equal to `bottom_row - top_row` from the
    /// originating [`ImageF::strip_view`] call.
    #[inline]
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Strip stride in pixels (matches parent's `stride`, preserving any padding).
    #[inline]
    #[must_use]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Parent row index corresponding to strip-local row `0`.
    ///
    /// Use this to detect strip-at-top-edge (`start_row_in_parent == 0`)
    /// and strip-at-bottom-edge (`start_row_in_parent + height == parent.height()`)
    /// when implementing halo handling.
    #[inline]
    #[must_use]
    pub fn start_row_in_parent(&self) -> usize {
        self.start_row_in_parent
    }

    /// Returns a slice of the strip-local row `y`, length = `width`.
    ///
    /// # Panics
    ///
    /// Panics if `y >= self.height()`. Empty-strip views panic for
    /// any `y` since `height` is `0`.
    #[inline]
    #[must_use]
    pub fn row(&self, y: usize) -> &[f32] {
        assert!(
            y < self.height,
            "StripView::row: y ({y}) out of bounds (height = {})",
            self.height
        );
        let start = y * self.stride;
        &self.data[start..start + self.width]
    }

    /// Returns a slice of strip-local row `y` including any stride padding,
    /// length = `stride`.
    #[inline]
    #[must_use]
    pub fn row_full(&self, y: usize) -> &[f32] {
        assert!(
            y < self.height,
            "StripView::row_full: y ({y}) out of bounds (height = {})",
            self.height
        );
        let start = y * self.stride;
        &self.data[start..start + self.stride]
    }

    /// Gets pixel `(x, y)` from the strip.
    ///
    /// # Panics
    ///
    /// Panics if `x >= self.width()` or `y >= self.height()`.
    #[inline]
    #[must_use]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        assert!(
            x < self.width,
            "StripView::get: x ({x}) out of bounds (width = {})",
            self.width
        );
        assert!(
            y < self.height,
            "StripView::get: y ({y}) out of bounds (height = {})",
            self.height
        );
        self.data[y * self.stride + x]
    }

    /// Returns a raw pointer to the strip's start (row `0`, column `0`).
    ///
    /// Useful for SIMD kernels that stride-walk by `stride * 4` bytes.
    /// Pointer is valid for `(self.height() - 1) * self.stride() + self.width()`
    /// f32 elements ahead.
    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }
}

/// Borrowed mutable view of a row-range of an [`ImageF`].
///
/// Returned from [`ImageF::strip_view_mut`]. Mirrors [`StripView`] but
/// permits writes; writes land directly in the parent's backing buffer.
///
/// See [`StripView`] for stride/halo/bounds semantics.
#[derive(Debug)]
pub struct StripViewMut<'a> {
    data: &'a mut [f32],
    width: usize,
    height: usize,
    stride: usize,
    start_row_in_parent: usize,
}

impl StripViewMut<'_> {
    /// Strip width in pixels.
    #[inline]
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Strip height in rows.
    #[inline]
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Strip stride in pixels (matches parent's `stride`).
    #[inline]
    #[must_use]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Parent row index corresponding to strip-local row `0`.
    #[inline]
    #[must_use]
    pub fn start_row_in_parent(&self) -> usize {
        self.start_row_in_parent
    }

    /// Returns an immutable slice of strip-local row `y`, length = `width`.
    ///
    /// # Panics
    ///
    /// Panics if `y >= self.height()`.
    #[inline]
    #[must_use]
    pub fn row(&self, y: usize) -> &[f32] {
        assert!(
            y < self.height,
            "StripViewMut::row: y ({y}) out of bounds (height = {})",
            self.height
        );
        let start = y * self.stride;
        &self.data[start..start + self.width]
    }

    /// Returns a mutable slice of strip-local row `y`, length = `width`.
    ///
    /// # Panics
    ///
    /// Panics if `y >= self.height()`.
    #[inline]
    pub fn row_mut(&mut self, y: usize) -> &mut [f32] {
        assert!(
            y < self.height,
            "StripViewMut::row_mut: y ({y}) out of bounds (height = {})",
            self.height
        );
        let start = y * self.stride;
        &mut self.data[start..start + self.width]
    }

    /// Returns an immutable slice of strip-local row `y` including stride padding.
    #[inline]
    #[must_use]
    pub fn row_full(&self, y: usize) -> &[f32] {
        assert!(
            y < self.height,
            "StripViewMut::row_full: y ({y}) out of bounds (height = {})",
            self.height
        );
        let start = y * self.stride;
        &self.data[start..start + self.stride]
    }

    /// Returns a mutable slice of strip-local row `y` including stride padding.
    #[inline]
    pub fn row_full_mut(&mut self, y: usize) -> &mut [f32] {
        assert!(
            y < self.height,
            "StripViewMut::row_full_mut: y ({y}) out of bounds (height = {})",
            self.height
        );
        let start = y * self.stride;
        &mut self.data[start..start + self.stride]
    }

    /// Gets pixel `(x, y)`.
    ///
    /// # Panics
    ///
    /// Panics if `x >= self.width()` or `y >= self.height()`.
    #[inline]
    #[must_use]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        assert!(
            x < self.width,
            "StripViewMut::get: x ({x}) out of bounds (width = {})",
            self.width
        );
        assert!(
            y < self.height,
            "StripViewMut::get: y ({y}) out of bounds (height = {})",
            self.height
        );
        self.data[y * self.stride + x]
    }

    /// Sets pixel `(x, y)`.
    ///
    /// # Panics
    ///
    /// Panics if `x >= self.width()` or `y >= self.height()`.
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, value: f32) {
        assert!(
            x < self.width,
            "StripViewMut::set: x ({x}) out of bounds (width = {})",
            self.width
        );
        assert!(
            y < self.height,
            "StripViewMut::set: y ({y}) out of bounds (height = {})",
            self.height
        );
        self.data[y * self.stride + x] = value;
    }

    /// Returns a raw mutable pointer to the strip's start.
    #[inline]
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.data.as_mut_ptr()
    }

    /// Returns a raw pointer to the strip's start.
    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }
}

/// Three-channel floating point image (for XYB data).
#[derive(Debug, Clone)]
pub struct Image3F {
    planes: [ImageF; 3],
}

impl Image3F {
    /// Creates a new 3-channel image filled with zeros.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            planes: [
                ImageF::new(width, height),
                ImageF::new(width, height),
                ImageF::new(width, height),
            ],
        }
    }

    /// Creates a new 3-channel image WITHOUT zero-filling.
    ///
    /// Callers MUST overwrite every pixel before reading.
    #[must_use]
    pub(crate) fn new_uninit(width: usize, height: usize) -> Self {
        Self {
            planes: [
                ImageF::new_uninit(width, height),
                ImageF::new_uninit(width, height),
                ImageF::new_uninit(width, height),
            ],
        }
    }

    /// Creates a new 3-channel image using pooled buffers (dirty, caller must overwrite).
    #[must_use]
    pub(crate) fn from_pool_dirty(width: usize, height: usize, pool: &BufferPool) -> Self {
        Self {
            planes: [
                ImageF::from_pool_dirty(width, height, pool),
                ImageF::from_pool_dirty(width, height, pool),
                ImageF::from_pool_dirty(width, height, pool),
            ],
        }
    }

    /// Recycles all three planes back to the pool.
    pub(crate) fn recycle(self, pool: &BufferPool) {
        let [p0, p1, p2] = self.planes;
        p0.recycle(pool);
        p1.recycle(pool);
        p2.recycle(pool);
    }

    /// Creates from three separate planes.
    #[must_use]
    pub fn from_planes(plane0: ImageF, plane1: ImageF, plane2: ImageF) -> Self {
        assert!(plane0.same_size(&plane1));
        assert!(plane0.same_size(&plane2));
        Self {
            planes: [plane0, plane1, plane2],
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
    pub fn plane(&self, index: usize) -> &ImageF {
        &self.planes[index]
    }

    /// Returns a mutable reference to a specific plane.
    #[inline]
    pub fn plane_mut(&mut self, index: usize) -> &mut ImageF {
        &mut self.planes[index]
    }

    /// Returns a row from a specific plane.
    #[inline]
    #[must_use]
    pub fn plane_row(&self, plane: usize, y: usize) -> &[f32] {
        self.planes[plane].row(y)
    }

    /// Returns a mutable row from a specific plane.
    #[inline]
    pub fn plane_row_mut(&mut self, plane: usize, y: usize) -> &mut [f32] {
        self.planes[plane].row_mut(y)
    }

    /// Returns mutable references to all three planes simultaneously.
    ///
    /// This uses array destructuring to allow safe split borrows,
    /// avoiding the need for unsafe code when writing to multiple planes.
    #[inline]
    pub fn planes_mut(&mut self) -> (&mut ImageF, &mut ImageF, &mut ImageF) {
        let [p0, p1, p2] = &mut self.planes;
        (p0, p1, p2)
    }
}

impl Index<usize> for Image3F {
    type Output = ImageF;

    fn index(&self, index: usize) -> &Self::Output {
        &self.planes[index]
    }
}

impl IndexMut<usize> for Image3F {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.planes[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_creation() {
        let img = ImageF::new(100, 50);
        assert_eq!(img.width(), 100);
        assert_eq!(img.height(), 50);
        assert!(img.stride() >= 100);
        assert_eq!(img.stride() % 16, 0); // Aligned
    }

    #[test]
    fn test_pixel_access() {
        let mut img = ImageF::new(10, 10);
        img.set(5, 3, 42.0);
        assert!((img.get(5, 3) - 42.0).abs() < 0.001);
        assert!((img[(5, 3)] - 42.0).abs() < 0.001);
    }

    #[test]
    fn test_row_access() {
        let mut img = ImageF::new(10, 10);
        img.row_mut(5)[3] = 99.0;
        assert!((img.row(5)[3] - 99.0).abs() < 0.001);
    }

    #[test]
    fn test_image3f() {
        let img = Image3F::new(100, 50);
        assert_eq!(img.width(), 100);
        assert_eq!(img.height(), 50);
    }

    // ------- W44-PHASE3-B7d Day 1: strip_view primitive parity tests -------
    //
    // These 7 tests prove the strip_view + strip_view_mut borrow-window API
    // is byte-identical to direct ImageF access on every dimension permutation
    // that matters for the Day 2-7 strip-tile arc (per the RFC at
    // `docs/RFC_W44_PHASE3_B7D_STRIP_TILE.md`):
    //
    //   #1 identity (strip = full image)              — sanity
    //   #2 partial strip in the middle                — main use case
    //   #3 edges (top-aligned + bottom-aligned)       — halo-edge detection
    //   #4 iteration (split + concat)                 — Day 2/3 kernel use
    //   #5 mutable writes propagate to parent         — output-strip mechanism
    //   #6 bounds (inverted + over + empty)           — defensive contracts
    //   #7 stride preservation (padded image)         — subtle SIMD-padding bug
    //                                                   (the one the RFC §9.R3
    //                                                   risk callout warns about)
    //
    // The test fixtures use ImageF::filled(...) + per-pixel set() so each
    // pixel is unique and any mis-indexing surfaces as a value mismatch.

    /// Helper: fill image with a deterministic per-pixel pattern.
    /// Returns y*1000 + x for each pixel (>=1 for nonzero discriminator).
    fn fill_unique(img: &mut ImageF) {
        for y in 0..img.height() {
            for x in 0..img.width() {
                img.set(x, y, (y * 1000 + x + 1) as f32);
            }
        }
    }

    /// Test #1 — identity: `strip_view(0, height())` exposes byte-identical
    /// row data to the parent ImageF across {32, 256, 1024}² fixtures.
    #[test]
    fn test_strip_view_identity_matches_parent() {
        for size in [32usize, 256, 1024] {
            let mut img = ImageF::new(size, size);
            fill_unique(&mut img);
            let strip = img.strip_view(0, size);
            assert_eq!(strip.width(), size, "width mismatch at size {size}");
            assert_eq!(strip.height(), size, "height mismatch at size {size}");
            assert_eq!(strip.stride(), img.stride(), "stride mismatch at size {size}");
            assert_eq!(strip.start_row_in_parent(), 0);
            for y in 0..size {
                assert_eq!(
                    strip.row(y),
                    img.row(y),
                    "row {y} mismatch at size {size}"
                );
            }
        }
    }

    /// Test #2 — partial strip in the middle. `strip_view(8, 24)` exposes
    /// rows 8..24 of the parent; local index 0 maps to parent row 8.
    #[test]
    fn test_strip_view_partial_middle() {
        let mut img = ImageF::new(64, 64);
        fill_unique(&mut img);
        let strip = img.strip_view(8, 24);
        assert_eq!(strip.height(), 16);
        assert_eq!(strip.width(), 64);
        assert_eq!(strip.start_row_in_parent(), 8);
        for local_y in 0..16 {
            let parent_y = 8 + local_y;
            assert_eq!(
                strip.row(local_y),
                img.row(parent_y),
                "local row {local_y} should match parent row {parent_y}"
            );
            // Spot-check via get()
            assert_eq!(strip.get(5, local_y), img.get(5, parent_y));
        }
    }

    /// Test #3 — edge strips: top-aligned (rows 0..16) and bottom-aligned
    /// (rows N-16..N). Verifies `start_row_in_parent` matches and rows
    /// are byte-identical.
    #[test]
    fn test_strip_view_edges() {
        let mut img = ImageF::new(128, 256);
        fill_unique(&mut img);

        // Top-edge strip: rows 0..16
        let top = img.strip_view(0, 16);
        assert_eq!(top.start_row_in_parent(), 0);
        assert_eq!(top.height(), 16);
        for y in 0..16 {
            assert_eq!(top.row(y), img.row(y), "top strip row {y}");
        }

        // Bottom-edge strip: rows (h-16)..h
        let h = img.height();
        let bottom = img.strip_view(h - 16, h);
        assert_eq!(bottom.start_row_in_parent(), h - 16);
        assert_eq!(bottom.height(), 16);
        for y in 0..16 {
            assert_eq!(
                bottom.row(y),
                img.row(h - 16 + y),
                "bottom strip row {y}"
            );
        }
    }

    /// Test #4 — split a 256² image into 16 strips of 16 rows, iterate,
    /// concatenate the row data back, and verify it byte-identically
    /// equals the original image.
    #[test]
    fn test_strip_view_iteration_concat() {
        let mut img = ImageF::new(256, 256);
        fill_unique(&mut img);

        const STRIP_ROWS: usize = 16;
        let strip_count = 256 / STRIP_ROWS;
        assert_eq!(strip_count, 16);

        let mut reconstructed: Vec<f32> = Vec::with_capacity(256 * 256);
        for s in 0..strip_count {
            let top = s * STRIP_ROWS;
            let bottom = top + STRIP_ROWS;
            let strip = img.strip_view(top, bottom);
            assert_eq!(strip.start_row_in_parent(), top);
            assert_eq!(strip.height(), STRIP_ROWS);
            for y in 0..STRIP_ROWS {
                reconstructed.extend_from_slice(strip.row(y));
            }
        }

        // Concat reference: row-by-row read of parent.
        let mut reference: Vec<f32> = Vec::with_capacity(256 * 256);
        for y in 0..256 {
            reference.extend_from_slice(img.row(y));
        }
        assert_eq!(
            reconstructed, reference,
            "concatenated strip rows must equal full image rows"
        );
    }

    /// Test #5 — `strip_view_mut` writes propagate to parent. Write a
    /// distinct sentinel value through a strip view; verify the parent
    /// ImageF sees it byte-identically afterwards.
    #[test]
    fn test_strip_view_mut_writes_propagate_to_parent() {
        let mut img = ImageF::new(64, 64);
        fill_unique(&mut img);

        // Snapshot rows OUTSIDE the strip range — they must stay untouched.
        let above = img.row(7).to_vec();
        let below = img.row(40).to_vec();

        {
            let mut strip = img.strip_view_mut(8, 40);
            assert_eq!(strip.height(), 32);
            assert_eq!(strip.start_row_in_parent(), 8);
            // Overwrite every pixel with a distinct sentinel.
            for local_y in 0..32 {
                for x in 0..64 {
                    strip.set(x, local_y, (-(local_y as f32) - 100.0) - x as f32 / 10.0);
                }
            }
            // Spot-check via the mutable view's read accessors.
            assert!((strip.get(0, 0) - (-100.0)).abs() < 1e-6);
            assert!(strip.row(5)[10] < 0.0);
        }

        // Parent now sees the writes at parent rows 8..40.
        for local_y in 0..32 {
            let parent_y = 8 + local_y;
            for x in 0..64 {
                let expected = (-(local_y as f32) - 100.0) - x as f32 / 10.0;
                assert!(
                    (img.get(x, parent_y) - expected).abs() < 1e-6,
                    "parent ({x}, {parent_y}) should reflect strip write"
                );
            }
        }
        // Parent untouched outside strip range.
        assert_eq!(img.row(7), above.as_slice());
        assert_eq!(img.row(40), below.as_slice());
    }

    /// Test #6 — bounds contracts: inverted range panics, over-extent
    /// panics, empty strip (top == bottom) returns height=0 view.
    #[test]
    #[should_panic(expected = "top_row")]
    fn test_strip_view_inverted_range_panics() {
        let img = ImageF::new(16, 16);
        let _ = img.strip_view(10, 5);
    }

    #[test]
    #[should_panic(expected = "bottom_row")]
    fn test_strip_view_over_height_panics() {
        let img = ImageF::new(16, 16);
        let _ = img.strip_view(0, 17);
    }

    #[test]
    fn test_strip_view_empty_is_valid() {
        let img = ImageF::new(16, 16);
        let empty = img.strip_view(8, 8);
        assert_eq!(empty.height(), 0);
        assert_eq!(empty.width(), 16);
        assert_eq!(empty.start_row_in_parent(), 8);
        // No row(0) call — would panic on empty (correct: height==0).
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_strip_view_row_out_of_bounds_panics() {
        let img = ImageF::new(16, 16);
        let strip = img.strip_view(0, 4);
        // Asking for strip-local row 4 (== bottom) is out-of-bounds.
        let _ = strip.row(4);
    }

    /// Test #7 — stride preservation. Build an ImageF whose stride > width
    /// (padding columns present per the SIMD 16-pixel alignment). Verify
    /// strip_view exposes the same padded stride and the row data lives
    /// at the same memory addresses as the parent (zero-copy proof).
    #[test]
    fn test_strip_view_preserves_stride() {
        // width=17 forces stride to round up to 32 (next multiple of 16).
        let mut img = ImageF::new(17, 24);
        fill_unique(&mut img);
        assert_eq!(img.width(), 17);
        assert_eq!(img.stride(), 32);
        assert!(img.stride() > img.width(), "fixture must have padding");

        let strip = img.strip_view(4, 20);
        // Stride preserved verbatim.
        assert_eq!(strip.stride(), 32);
        assert_eq!(strip.width(), 17);
        assert_eq!(strip.height(), 16);

        // Row data byte-identical to parent.
        for local_y in 0..16 {
            let parent_y = 4 + local_y;
            assert_eq!(strip.row(local_y), img.row(parent_y));
            // Verify row_full picks up the padded portion verbatim.
            assert_eq!(strip.row_full(local_y), img.row_full(parent_y));
        }

        // Zero-copy proof: the strip's row(0) slice must point to the same
        // memory address as the parent's row(top_row) slice.
        let strip_row0_ptr = strip.row(0).as_ptr();
        let parent_row4_ptr = img.row(4).as_ptr();
        assert_eq!(
            strip_row0_ptr, parent_row4_ptr,
            "strip row 0 must alias parent row 4 (zero-copy borrow)"
        );

        // Same proof via as_ptr on the strip itself.
        assert_eq!(strip.as_ptr(), parent_row4_ptr);
    }
}
