//! Image buffer types for butteraugli.
//!
//! These types provide efficient storage for floating-point image data
//! with row-stride support for cache-friendly access patterns.

use imgref::ImgVec;
use std::cell::RefCell;
use std::ops::{Index, IndexMut};
use std::sync::Mutex;
use thread_local::ThreadLocal;

/// TLS cap per thread per pool instance.
///
/// The full-res + half-res buttloop allocates ~51 pool ops per `compare`
/// call (B6 audit). With 8 worker threads in `maybe_join`, each thread
/// owns a slice of those ops — 16 per thread is plenty to keep the TLS
/// fast path warm across multiple iters without spilling to overflow.
const TLS_BUFFERS_PER_THREAD: usize = 16;

/// Overflow cap (shared across all threads for one pool instance).
///
/// Mirrors the original global pool cap so storage bound is unchanged.
/// The overflow is only touched when a thread's TLS is empty AND we
/// need to put a buffer back — almost never on the steady-state warm
/// path. Best-fit search inside `take_overflow` is O(N_overflow).
const OVERFLOW_BUFFERS: usize = 48;

/// Reusable buffer pool for `ImageF` allocations.
///
/// **W44-phase3-B7c (2026-05-23)**: TLS-cached fast path with shared
/// `Mutex` overflow. Most `take` / `put` calls hit the per-thread cache
/// (lock-free), eliminating the Mutex contention that ate the B7a+b
/// alloc-count savings on parallel `maybe_join` branches.
///
/// Per-instance, per-thread cache via `thread_local` crate. Each
/// `BufferPool` instance has its own `ThreadLocal<RefCell<Vec<Vec<f32>>>>`
/// so two different pools sharing a thread don't mix buffers.
///
/// `Sync` + `Send`: TLS slots use `RefCell` (only touched by their owning
/// thread); overflow uses `Mutex`.
///
/// Avoids repeated mmap/munmap for large temporary buffers. Owned by
/// the caller (typically `ButteraugliReference` or a local in
/// standalone API functions). When the pool is dropped, all cached
/// buffers are freed.
pub struct BufferPool {
    /// Per-(thread, instance) buffer cache. Fast path: no lock.
    tls: ThreadLocal<RefCell<Vec<Vec<f32>>>>,
    /// Shared overflow when a thread's TLS is full. Slow path: locked.
    overflow: Mutex<Vec<Vec<f32>>>,
}

impl core::fmt::Debug for BufferPool {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let overflow_count = self.overflow.lock().unwrap().len();
        // We can't iterate `tls` from &self because `ThreadLocal::iter`
        // requires `T: Sync` and `RefCell` is not `Sync`. The per-thread
        // counts are only visible via `iter_mut(&mut self)`. Debug
        // reports overflow only — TLS counts are best surfaced via
        // dedicated stats hooks if needed.
        f.debug_struct("BufferPool")
            .field("overflow_buffers", &overflow_count)
            .field("tls", &"<per-thread slots; see iter_mut for counts>")
            .finish()
    }
}

impl Default for BufferPool {
    fn default() -> Self {
        Self {
            tls: ThreadLocal::new(),
            overflow: Mutex::new(Vec::new()),
        }
    }
}

impl BufferPool {
    /// Creates a new empty buffer pool.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Best-fit selection from a slice of buffers (returns the index with
    /// the smallest excess capacity ≥ `needed`, if any).
    #[inline]
    fn best_fit_index(buffers: &[Vec<f32>], needed: usize) -> Option<usize> {
        let mut best_idx = None;
        let mut best_excess = usize::MAX;
        for (i, buf) in buffers.iter().enumerate() {
            let cap = buf.len();
            if cap >= needed && cap - needed < best_excess {
                best_idx = Some(i);
                best_excess = cap - needed;
                if best_excess == 0 {
                    // Perfect fit; stop early.
                    break;
                }
            }
        }
        best_idx
    }

    /// Adjusts an in-hand buffer to `needed` elements without changing
    /// existing capacity if the buffer is already large enough.
    /// Returns stale data — caller must zero-fill if needed.
    #[inline]
    fn finalize_buffer(mut buf: Vec<f32>, needed: usize) -> Vec<f32> {
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
    }

    /// Allocate a fresh buffer of `needed` elements (fallback when both
    /// TLS and overflow are empty/no-fit).
    #[inline]
    fn fresh_buffer(needed: usize) -> Vec<f32> {
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

    /// Takes a buffer of at least `needed` elements from the pool (best-fit).
    /// Returns stale data — caller must zero-fill if needed.
    ///
    /// Fast path: per-thread TLS cache (no lock).
    /// Slow path: shared overflow `Mutex` (only when TLS missed).
    /// Fallback: fresh allocation.
    ///
    /// With `unsafe-performance`, new allocations skip zero-fill entirely.
    pub(crate) fn take(&self, needed: usize) -> Vec<f32> {
        // Fast path: lock-free TLS lookup. `get_or` initialises an empty
        // slot the first time this thread touches the pool.
        let cell = self.tls.get_or(|| RefCell::new(Vec::new()));
        if let Some(buf) = {
            let mut local = cell.borrow_mut();
            Self::best_fit_index(&local, needed).map(|idx| local.swap_remove(idx))
        } {
            return Self::finalize_buffer(buf, needed);
        }

        // Slow path: shared overflow. Best-fit + swap_remove under the lock,
        // then release the lock before any resize.
        let from_overflow = {
            let mut overflow = self.overflow.lock().unwrap();
            Self::best_fit_index(&overflow, needed).map(|idx| overflow.swap_remove(idx))
        };
        if let Some(buf) = from_overflow {
            return Self::finalize_buffer(buf, needed);
        }

        // Fallback: fresh allocation.
        Self::fresh_buffer(needed)
    }

    /// Returns a buffer to the pool.
    ///
    /// Fast path: per-thread TLS (up to [`TLS_BUFFERS_PER_THREAD`]).
    /// Slow path: shared overflow (up to [`OVERFLOW_BUFFERS`]).
    /// Beyond both caps: buffer dropped silently.
    pub(crate) fn put(&self, buf: Vec<f32>) {
        let cell = self.tls.get_or(|| RefCell::new(Vec::new()));
        {
            let mut local = cell.borrow_mut();
            if local.len() < TLS_BUFFERS_PER_THREAD {
                local.push(buf);
                return;
            }
        }
        // TLS full for this thread — spill to overflow.
        let mut overflow = self.overflow.lock().unwrap();
        if overflow.len() < OVERFLOW_BUFFERS {
            overflow.push(buf);
        }
        // else: drop silently (pool at capacity)
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

    // ===========================================================
    // B7c (2026-05-23): TLS pool tests
    // ===========================================================

    /// take → put → take must hit the TLS fast path on the same thread
    /// and return the same underlying allocation (no fresh allocation).
    #[test]
    fn test_pool_tls_take_put_take_reuse() {
        let pool = BufferPool::new();
        let buf = pool.take(1024);
        let ptr_before = buf.as_ptr();
        pool.put(buf);
        let buf2 = pool.take(1024);
        let ptr_after = buf2.as_ptr();
        assert_eq!(
            ptr_before, ptr_after,
            "TLS fast path must reuse the same allocation",
        );
        assert_eq!(buf2.len(), 1024);
    }

    /// Best-fit selection must work within the TLS cache: returning
    /// buffers of multiple sizes, then taking should pick the smallest
    /// excess.
    #[test]
    fn test_pool_tls_best_fit() {
        let pool = BufferPool::new();
        // Seed TLS with three different sizes via put.
        pool.put(vec![0.0; 1024]);
        pool.put(vec![0.0; 4096]);
        pool.put(vec![0.0; 2048]);
        // Asking for 1500 should pick the 2048 (smallest excess ≥ 1500).
        let buf = pool.take(1500);
        assert_eq!(buf.len(), 1500);
        // Capacity should reflect that we picked the 2048 buffer (not 4096).
        assert!(
            buf.capacity() >= 1500 && buf.capacity() < 4000,
            "expected best-fit 2048-cap buffer, got cap={}",
            buf.capacity(),
        );
    }

    /// Beyond `TLS_BUFFERS_PER_THREAD` puts, the pool must spill to the
    /// shared overflow rather than dropping or panicking.
    #[test]
    fn test_pool_tls_overflow_spill() {
        let pool = BufferPool::new();
        // Fill the TLS cache.
        for _ in 0..TLS_BUFFERS_PER_THREAD {
            pool.put(vec![0.0; 64]);
        }
        // The next put should spill into overflow.
        pool.put(vec![0.0; 128]);
        // Take the spilled buffer back out — TLS still has 16 entries
        // of size 64, none of which fits ≥ 100, so the take MUST hit
        // overflow.
        let buf = pool.take(100);
        assert!(buf.len() == 100);
        assert!(
            buf.capacity() >= 128,
            "expected overflow buffer (cap≥128), got cap={}",
            buf.capacity(),
        );
    }

    /// Two pool instances on the same thread MUST NOT share buffers
    /// via TLS — each `BufferPool` has its own per-thread slot.
    #[test]
    fn test_pool_instances_isolated() {
        let pool_a = BufferPool::new();
        let pool_b = BufferPool::new();
        let buf_a = pool_a.take(512);
        let ptr_a = buf_a.as_ptr();
        pool_a.put(buf_a);
        // Take from a different pool — must NOT return pool_a's buffer.
        let buf_b = pool_b.take(512);
        let ptr_b = buf_b.as_ptr();
        assert_ne!(
            ptr_a, ptr_b,
            "TLS slots must be per-instance, not shared across pools",
        );
    }

    /// Concurrent take/put across rayon threads must not deadlock or
    /// produce garbage. Stress-tests both TLS fast path and overflow
    /// spill under contention.
    #[test]
    #[cfg(feature = "rayon")]
    fn test_pool_parallel_take_put() {
        use rayon::prelude::*;

        let pool = BufferPool::new();
        let n_ops = 256;
        // All threads hammer the same pool.
        (0..n_ops).into_par_iter().for_each(|i| {
            let needed = 64 + (i % 32) * 16;
            let mut buf = pool.take(needed);
            // Touch the buffer to prove it's writable.
            buf[0] = i as f32;
            buf[needed - 1] = i as f32;
            pool.put(buf);
        });
        // No panic, no deadlock. Verify the pool still works after the storm.
        let buf = pool.take(1024);
        assert_eq!(buf.len(), 1024);
    }

    /// Pool is Sync + Send — required so &BufferPool can cross
    /// rayon::join branches.
    #[test]
    fn test_pool_sync_send_static() {
        fn assert_sync_send<T: Sync + Send>() {}
        assert_sync_send::<BufferPool>();
    }
}
