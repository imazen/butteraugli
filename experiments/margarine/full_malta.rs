//! Complete native Malta responses with each row's stencil range checked once.
//! The shared nonlinear difference transform and all sixteen lines are retained.
use crate::image::{BufferPool, ImageF};
use crate::malta_bank::{V, Window, hf_bank, lf_bank};

const LANES: usize = if cfg!(feature = "wide-malta") { 16 } else { 8 };
const WINDOW: usize = LANES + 8;
struct NativeWindow<'a> {
    rows: [&'a [f32; WINDOW]; 9],
}
impl Window for NativeWindow<'_> {
    type Vector = V<LANES>;
    #[inline(always)]
    fn zero(&self) -> Self::Vector {
        V::splat(0.0)
    }
    #[inline(always)]
    fn load(&self, dx: isize, dy: isize) -> V<LANES> {
        let start = (dx + 4) as usize;
        let values: &[f32; LANES] = self.rows[(dy + 4) as usize][start..start + LANES]
            .try_into()
            .unwrap();
        V(*values)
    }
}

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
    if a.width() < LANES {
        return crate::shared_malta::malta_diff_map(a, b, greater, smaller, norm, lf, pool);
    }
    let padded =
        crate::shared_malta::malta_scaled_differences(a, b, greater, smaller, norm, lf, pool);
    let mut out = ImageF::from_pool_dirty(a.width(), a.height(), pool);
    evaluate(&padded, lf, 1, &mut out);
    padded.recycle(pool);
    out
}

#[cfg(not(feature = "simd-malta"))]
#[archmage::autoversion]
fn evaluate(_token: archmage::SimdToken, padded: &ImageF, lf: bool, step: usize, out: &mut ImageF) {
    let width = out.width();
    for y in (0..out.height()).step_by(step) {
        let row = out.row_mut(y);
        for block in 0..width.div_ceil(LANES) {
            // The last block overlaps when width is not divisible by eight.
            let start = (block * LANES).min(width - LANES);
            let window = NativeWindow {
                rows: std::array::from_fn(|r| {
                    padded.row(y + r)[start..start + WINDOW].try_into().unwrap()
                }),
            };
            let values = if lf {
                lf_bank(&window)
            } else {
                hf_bank(&window)
            };
            row[start..start + LANES].copy_from_slice(&values.0);
        }
    }
}

#[cfg(feature = "simd-malta")]
fn evaluate(padded: &ImageF, lf: bool, step: usize, out: &mut ImageF) {
    archmage::incant!(
        evaluate_simd(padded, lf, step, out),
        [v4, v3, neon, wasm128, scalar]
    )
}

#[cfg(feature = "simd-malta")]
struct LoadWindow<F, T> {
    load: F,
    zero: T,
}
#[cfg(feature = "simd-malta")]
impl<F, T> Window for LoadWindow<F, T>
where
    F: Fn(isize, isize) -> T,
    T: Copy + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + std::ops::AddAssign,
{
    type Vector = T;
    #[inline(always)]
    fn zero(&self) -> T {
        self.zero
    }
    #[inline(always)]
    fn load(&self, dx: isize, dy: isize) -> T {
        (self.load)(dx, dy)
    }
}

#[cfg(feature = "simd-malta")]
#[archmage::magetypes(v4, v3, neon, wasm128, scalar)]
fn evaluate_simd(token: Token, padded: &ImageF, lf: bool, step: usize, out: &mut ImageF) {
    #[cfg(not(feature = "wide-malta"))]
    type Lanes = magetypes::simd::generic::f32x8<Token>;
    #[cfg(feature = "wide-malta")]
    type Lanes = magetypes::simd::generic::f32x16<Token>;
    let width = out.width();
    for y in (0..out.height()).step_by(step) {
        let row = out.row_mut(y);
        for block in 0..width.div_ceil(LANES) {
            let start = (block * LANES).min(width - LANES);
            let rows: [&[f32; WINDOW]; 9] = std::array::from_fn(|r| {
                padded.row(y + r)[start..start + WINDOW].try_into().unwrap()
            });
            let window = LoadWindow {
                zero: Lanes::splat(token, 0.0),
                load: |dx: isize, dy: isize| {
                    let x = (dx + 4) as usize;
                    Lanes::load(
                        token,
                        rows[(dy + 4) as usize][x..x + LANES].try_into().unwrap(),
                    )
                },
            };
            let values = if lf {
                lf_bank(&window)
            } else {
                hf_bank(&window)
            };
            values.store((&mut row[start..start + LANES]).try_into().unwrap());
        }
    }
}

/// Sample every second output row, retaining every native input and column.
#[cfg(feature = "row-malta")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sampled_rows_diff_map(
    a: &ImageF,
    b: &ImageF,
    greater: f64,
    smaller: f64,
    norm: f64,
    lf: bool,
    pool: &BufferPool,
) -> ImageF {
    let mut out = if a.width() < LANES {
        malta_diff_map(a, b, greater, smaller, norm, lf, pool)
    } else {
        let padded =
            crate::shared_malta::malta_scaled_differences(a, b, greater, smaller, norm, lf, pool);
        let mut out = ImageF::from_pool_dirty(a.width(), a.height(), pool);
        evaluate(&padded, lf, 2, &mut out);
        padded.recycle(pool);
        out
    };
    interpolate_rows(&mut out);
    out
}

#[cfg(feature = "row-malta")]
#[archmage::autoversion]
fn interpolate_rows(_token: archmage::SimdToken, out: &mut ImageF) {
    let (width, height, stride) = (out.width(), out.height(), out.stride());
    for y in (1..height).step_by(2) {
        let (before, rest) = out.data_mut().split_at_mut(y * stride);
        let a = &before[(y - 1) * stride..(y - 1) * stride + width];
        let (row, after) = rest.split_at_mut(stride);
        let b = if y + 1 < height { &after[..width] } else { a };
        for ((dst, &a), &b) in row[..width].iter_mut().zip(a).zip(b) {
            *dst = a + 0.5 * (b - a);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "row-malta")]
    #[test]
    fn sampled_rows_keep_native_nodes_and_interpolate_only_missing_rows() {
        let pool = BufferPool::new();
        for (w, h) in [(1, 1), (3, 5), (8, 10), (17, 18), (33, 37), (634, 17)] {
            let mut a = ImageF::new(w, h);
            let mut b = ImageF::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    a.row_mut(y)[x] = ((x * 31 + y * 97 + x * y) % 101) as f32 * 0.13 - 5.0;
                    b.row_mut(y)[x] = a.row(y)[x] * 0.91 + ((x + y) % 3) as f32 * 0.1;
                }
            }
            for lf in [false, true] {
                let full = crate::shared_malta::malta_diff_map(&a, &b, 0.5, 1.7, 1.2, lf, &pool);
                let sampled = sampled_rows_diff_map(&a, &b, 0.5, 1.7, 1.2, lf, &pool);
                for y in 0..h {
                    for x in 0..w {
                        let expected = if y % 2 == 0 {
                            full.row(y)[x]
                        } else {
                            let a = full.row(y - 1)[x];
                            let b = full.row(if y + 1 < h { y + 1 } else { y - 1 })[x];
                            a + 0.5 * (b - a)
                        };
                        assert_eq!(sampled.row(y)[x], expected, "{w}x{h}, ({x},{y}), lf={lf}");
                    }
                }
            }
        }
    }
    #[test]
    fn every_native_response_matches_shared_bank_at_borders_and_vector_tails() {
        let pool = BufferPool::new();
        for (w, h) in (1..34)
            .map(|w| (w, w + 2))
            .chain([(128, 129), (154, 151), (317, 129)])
        {
            let mut a = ImageF::new(w, h);
            let mut b = ImageF::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    a.row_mut(y)[x] = ((x * 31 + y * 97 + x * y) % 101) as f32 * 0.13 - 5.0;
                    b.row_mut(y)[x] = a.row(y)[x] * 0.91 + ((x + y) % 3) as f32 * 0.1;
                }
            }
            for lf in [false, true] {
                for (greater, smaller, norm) in [(1.0, 1.0, 0.5), (0.5, 1.7, 1.2), (1.7, 0.5, 2.0)]
                {
                    let expected = crate::shared_malta::malta_diff_map(
                        &a, &b, greater, smaller, norm, lf, &pool,
                    );
                    let actual = malta_diff_map(&a, &b, greater, smaller, norm, lf, &pool);
                    for y in 0..h {
                        if actual.row(y) != expected.row(y) {
                            let padded = crate::shared_malta::malta_scaled_differences(
                                &a, &b, greater, smaller, norm, lf, &pool,
                            );
                            for x in 0..w {
                                if actual.row(y)[x] != expected.row(y)[x] {
                                    let direct = if lf {
                                        crate::shared_malta::malta_unit_lf(&padded, x + 4, y + 4)
                                    } else {
                                        crate::shared_malta::malta_unit(&padded, x + 4, y + 4)
                                    };
                                    eprintln!(
                                        "x={x} weights={greater},{smaller},{norm} direct={direct} actual={} expected={}",
                                        actual.row(y)[x],
                                        expected.row(y)[x]
                                    );
                                }
                            }
                        }
                        assert_eq!(actual.row(y), expected.row(y), "{w}x{h}, lf={lf}, row {y}");
                    }
                }
            }
        }
    }
}
