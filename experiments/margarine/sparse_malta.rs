//! Evaluate the complete Malta bank at every second output coordinate.
//! All native input samples and sixteen orientations remain in each response.
//! Four phase planes turn the strided gathers into contiguous SIMD loads.
use crate::image::{BufferPool, ImageF};

#[allow(clippy::too_many_arguments)]
pub(crate) fn malta_diff_map(
    a: &ImageF,
    b: &ImageF,
    greater: f64,
    smaller: f64,
    norm: f64,
    lf: bool,
    pool: &BufferPool,
) -> ImageF {
    let padded =
        crate::shared_malta::malta_scaled_differences(a, b, greater, smaller, norm, lf, pool);
    let (w, h) = (a.width(), a.height());
    let phases = phase_planes(&padded, pool);
    padded.recycle(pool);
    let mut coarse = ImageF::from_pool_dirty(w.div_ceil(2), h.div_ceil(2), pool);
    evaluate(&phases, lf, &mut coarse);
    for plane in phases {
        plane.recycle(pool);
    }
    let mut result = ImageF::from_pool_dirty(w, h, pool);
    reconstruct(&coarse, &mut result);
    coarse.recycle(pool);
    result
}

fn phase_planes(input: &ImageF, pool: &BufferPool) -> [ImageF; 4] {
    // Eight guard columns make the last SIMD group safe even at odd widths.
    let pw = input.width().div_ceil(2) + 8;
    let ph = input.height().div_ceil(2);
    let mut planes = std::array::from_fn(|_| ImageF::from_pool_dirty(pw, ph, pool));
    for (phase, plane) in planes.iter_mut().enumerate() {
        for y in 0..ph {
            let dst = plane.row_mut(y);
            dst.fill(0.0);
            let iy = y * 2 + phase / 2;
            if iy < input.height() {
                for (d, &s) in dst
                    .iter_mut()
                    .zip(input.row(iy).iter().skip(phase % 2).step_by(2))
                {
                    *d = s;
                }
            }
        }
    }
    planes
}

#[derive(Clone, Copy)]
struct V([f32; 8]);
impl std::ops::Add for V {
    type Output = Self;
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i] + other.0[i]))
    }
}
impl std::ops::Mul for V {
    type Output = Self;
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i] * other.0[i]))
    }
}
impl std::ops::AddAssign for V {
    #[inline(always)]
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

struct Window<'a> {
    planes: &'a [ImageF; 4],
    center: usize,
    stride: usize,
}
#[inline(always)]
fn load(window: &Window<'_>, dx: isize, dy: isize) -> V {
    let phase = (dy.rem_euclid(2) * 2 + dx.rem_euclid(2)) as usize;
    let offset = dy.div_euclid(2) * window.stride as isize + dx.div_euclid(2);
    let start = (window.center as isize + offset) as usize;
    let values: &[f32; 8] = window.planes[phase].data()[start..start + 8]
        .try_into()
        .unwrap();
    V(*values)
}
macro_rules! w {
    ($window:expr,$dx:expr,$dy:expr) => {
        load($window, $dx, $dy)
    };
}

#[archmage::autoversion]
fn evaluate(_token: archmage::SimdToken, planes: &[ImageF; 4], lf: bool, out: &mut ImageF) {
    let stride = planes[0].stride();
    for y in 0..out.height() {
        for (block, dst) in out.row_mut(y).chunks_mut(8).enumerate() {
            let window = Window {
                planes,
                center: (y + 2) * stride + 2 + block * 8,
                stride,
            };
            let value = if lf {
                lf_bank(&window)
            } else {
                hf_bank(&window)
            };
            dst.copy_from_slice(&value.0[..dst.len()]);
        }
    }
}

#[archmage::autoversion]
fn reconstruct(_token: archmage::SimdToken, input: &ImageF, out: &mut ImageF) {
    for y in 0..out.height() {
        let a = input.row(y / 2);
        let b = input.row((y / 2 + 1).min(input.height() - 1));
        let fy = (y % 2) as f32 * 0.5;
        for (x, dst) in out.row_mut(y).iter_mut().enumerate() {
            let x0 = x / 2;
            let x1 = (x0 + 1).min(input.width() - 1);
            let fx = (x % 2) as f32 * 0.5;
            let top = a[x0] + fx * (a[x1] - a[x0]);
            let bottom = b[x0] + fx * (b[x1] - b[x0]);
            *dst = top + fy * (bottom - top);
        }
    }
}

// The two banks below preserve the tap order and all sixteen patterns from
// butteraugli/src/malta.rs::{malta_unit_window,malta_unit_lf_window}.

#[inline(always)]
fn hf_bank(window: &Window<'_>) -> V {
    let mut retval = V([0.0; 8]);

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

#[inline(always)]
fn lf_bank(window: &Window<'_>) -> V {
    let mut retval = V([0.0; 8]);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampled_nodes_match_complete_bank_with_odd_sizes_and_padding() {
        let pool = BufferPool::new();
        for (w, h) in [(1, 1), (3, 5), (31, 27), (65, 39)] {
            let stride = w + 7;
            let mut a = vec![f32::NAN; stride * h];
            let mut b = a.clone();
            for y in 0..h {
                for x in 0..w {
                    a[y * stride + x] = ((x * 31 + y * 97 + x * y) % 997) as f32 * 0.013 - 4.0;
                    b[y * stride + x] = a[y * stride + x] * 0.87 + ((x + y) % 3) as f32 * 0.23;
                }
            }
            let a = ImageF::from_vec_padded(a, w, h, stride);
            let b = ImageF::from_vec_padded(b, w, h, stride);
            for lf in [false, true] {
                let full = crate::shared_malta::malta_diff_map(&a, &b, 1.8, 0.7, 10.0, lf, &pool);
                let sparse = malta_diff_map(&a, &b, 1.8, 0.7, 10.0, lf, &pool);
                for y in (0..h).step_by(2) {
                    for x in (0..w).step_by(2) {
                        assert_eq!(
                            sparse.row(y)[x],
                            full.row(y)[x],
                            "{w}x{h} ({x},{y}) lf={lf}"
                        );
                    }
                }
            }
        }
    }
}
