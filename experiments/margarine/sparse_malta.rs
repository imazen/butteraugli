//! Evaluate the complete Malta bank at every second output coordinate.
//! All native input samples and sixteen orientations remain in each response.
//! Four phase planes turn the strided gathers into contiguous SIMD loads.
use crate::image::{BufferPool, ImageF};
use crate::malta_bank::{V, Window as BankWindow, hf_bank, lf_bank};

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
    if cfg!(feature = "lattice") {
        let mut coarse = ImageF::from_pool_dirty(w.div_ceil(2), h.div_ceil(2), pool);
        native_evaluate(&padded, lf, &mut coarse);
        padded.recycle(pool);
        let mut result = ImageF::from_pool_dirty(w, h, pool);
        reconstruct(&coarse, &mut result);
        coarse.recycle(pool);
        return result;
    }
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

struct NativeWindow<'a> {
    rows: [&'a [f32; 23]; 9],
}
impl BankWindow for NativeWindow<'_> {
    #[inline(always)]
    fn load(&self, dx: isize, dy: isize) -> V {
        let row = self.rows[(dy + 4) as usize];
        let start = (dx + 4) as usize;
        V(std::array::from_fn(|lane| row[start + lane * 2]))
    }
}

#[archmage::autoversion]
fn native_evaluate(_token: archmage::SimdToken, padded: &ImageF, lf: bool, out: &mut ImageF) {
    for y in 0..out.height() {
        let full = out.width() / 8;
        let row = out.row_mut(y);
        for (block, dst) in row[..full * 8]
            .as_chunks_mut::<8>()
            .0
            .iter_mut()
            .enumerate()
        {
            let rows = std::array::from_fn(|r| {
                padded.row(y * 2 + r)[block * 16..block * 16 + 23]
                    .try_into()
                    .unwrap()
            });
            let window = NativeWindow { rows };
            *dst = if lf {
                lf_bank(&window).0
            } else {
                hf_bank(&window).0
            };
        }
        if row.len() != full * 8 {
            let mut tail = [[0.0; 23]; 9];
            for (r, dst) in tail.iter_mut().enumerate() {
                let src = &padded.row(y * 2 + r)[full * 16..];
                let n = src.len().min(dst.len());
                dst[..n].copy_from_slice(&src[..n]);
            }
            let window = NativeWindow {
                rows: std::array::from_fn(|r| &tail[r]),
            };
            let values = if lf {
                lf_bank(&window)
            } else {
                hf_bank(&window)
            };
            let dst = &mut row[full * 8..];
            dst.copy_from_slice(&values.0[..dst.len()]);
        }
    }
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

struct Window<'a> {
    // Nine native rows, split into two column phases. Each twelve-value row
    // covers every eight-lane load at the five possible coarse x offsets.
    rows: [[&'a [f32; 12]; 2]; 9],
}
impl BankWindow for Window<'_> {
    #[inline(always)]
    fn load(&self, dx: isize, dy: isize) -> V {
        let row = self.rows[(dy + 4) as usize][dx.rem_euclid(2) as usize];
        let start = (dx.div_euclid(2) + 2) as usize;
        let values: &[f32; 8] = row[start..start + 8].try_into().unwrap();
        V(*values)
    }
}

#[inline(always)]
fn phases_data(planes: &[ImageF; 4], phase: usize, start: usize) -> &[f32; 12] {
    planes[phase].data()[start..start + 12].try_into().unwrap()
}

#[archmage::autoversion]
fn evaluate(_token: archmage::SimdToken, planes: &[ImageF; 4], lf: bool, out: &mut ImageF) {
    let stride = planes[0].stride();
    for y in 0..out.height() {
        for (block, dst) in out.row_mut(y).chunks_mut(8).enumerate() {
            let rows = std::array::from_fn(|row| {
                let dy = row as isize - 4;
                std::array::from_fn(|phase_x| {
                    let phase = dy.rem_euclid(2) as usize * 2 + phase_x;
                    let iy = (y as isize + 2 + dy.div_euclid(2)) as usize;
                    let start = iy * stride + block * 8;
                    phases_data(planes, phase, start)
                })
            });
            let window = Window { rows };
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
        let pairs = (input.width() - 1).min(out.width() / 2);
        let blocks = pairs / 8;
        let row = out.row_mut(y);
        for (block, dst) in row[..blocks * 16]
            .as_chunks_mut::<16>()
            .0
            .iter_mut()
            .enumerate()
        {
            let a: &[f32; 9] = a[block * 8..block * 8 + 9].try_into().unwrap();
            let b: &[f32; 9] = b[block * 8..block * 8 + 9].try_into().unwrap();
            for i in 0..8 {
                let top = a[i] + 0.5 * (a[i + 1] - a[i]);
                let bottom = b[i] + 0.5 * (b[i + 1] - b[i]);
                dst[2 * i] = a[i] + fy * (b[i] - a[i]);
                dst[2 * i + 1] = top + fy * (bottom - top);
            }
        }
        for (pair, dst) in row[blocks * 16..pairs * 2]
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .enumerate()
        {
            let i = blocks * 8 + pair;
            let top = a[i] + 0.5 * (a[i + 1] - a[i]);
            let bottom = b[i] + 0.5 * (b[i + 1] - b[i]);
            dst[0] = a[i] + fy * (b[i] - a[i]);
            dst[1] = top + fy * (bottom - top);
        }
        for (x, dst) in row.iter_mut().enumerate().skip(pairs * 2) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reconstruction_preserves_scalar_formula_at_every_phase_and_tail() {
        for w in [1usize, 2, 3, 15, 16, 17, 31, 32, 33, 65, 128, 129] {
            for h in [1usize, 2, 3, 17] {
                let mut input = ImageF::new(w.div_ceil(2), h.div_ceil(2));
                for y in 0..input.height() {
                    for x in 0..input.width() {
                        input.row_mut(y)[x] = ((x * 313 + y * 997) % 4093) as f32 * 0.017;
                    }
                }
                let mut output = ImageF::new(w, h);
                reconstruct(&input, &mut output);
                for y in 0..h {
                    for x in 0..w {
                        let a = input.row(y / 2);
                        let b = input.row((y / 2 + 1).min(input.height() - 1));
                        let (x0, x1) = (x / 2, (x / 2 + 1).min(input.width() - 1));
                        let (fx, fy) = ((x % 2) as f32 * 0.5, (y % 2) as f32 * 0.5);
                        let top = a[x0] + fx * (a[x1] - a[x0]);
                        let bottom = b[x0] + fx * (b[x1] - b[x0]);
                        assert_eq!(
                            output.row(y)[x].to_bits(),
                            (top + fy * (bottom - top)).to_bits()
                        );
                    }
                }
            }
        }
    }

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
