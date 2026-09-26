//! Pull frequency rows through bounded caches, retaining overlap across scoring
//! strips. Shared Gaussian coefficients, nonlinearities and Malta scoring remain
//! the contract; this is an experimental execution schedule.
#[cfg(any(not(feature = "simd-opsin"), test))]
use crate::opsin;
use crate::{blur, consts::*, diff, image, ingress, psycho, stream_blur, strips};
use butteraugli::ButteraugliParams;
use std::error::Error;

type Row = Vec<f32>;
#[derive(Clone, Copy)]
enum Op {
    Input,
    MirrorH(usize, [f32; 3]),
    MirrorV(usize, [f32; 3]),
    Opsin(usize, usize),
    Reduce(usize, usize),
    GaussianH(usize, usize),
    GaussianV(usize, usize),
    Expand(usize, usize),
    Subtract(usize, usize),
    High(usize, usize),
    Finish([usize; 4]),
}
struct Node {
    op: Op,
    width: usize,
    height: usize,
    channels: usize,
    step: usize,
    latency: usize,
    cache: Vec<Option<(usize, Row)>>,
    generated: usize,
}
struct Graph<'a, 'b> {
    input: &'a ingress::EncodedRows<'b>,
    factor: usize,
    intensity: f32,
    nodes: Vec<Node>,
    kernels: Vec<(Vec<f32>, Vec<f32>)>,
    output: usize,
}
impl<'a, 'b> Graph<'a, 'b> {
    fn new(input: &'a ingress::EncodedRows<'b>, factor: usize, intensity: f32) -> Self {
        let mut g = Self {
            input,
            factor,
            intensity,
            nodes: Vec::new(),
            kernels: Vec::new(),
            output: 0,
        };
        let w = input.width.div_ceil(factor);
        let h = input.height.div_ceil(factor);
        let rgb = g.add(Op::Input, [w, h, 3, 1, 0]);
        let weights = blur::compute_separable5_weights(1.2);
        let horizontal = g.add(Op::MirrorH(rgb, weights), [w, h, 3, 1, 0]);
        let vertical = g.add(Op::MirrorV(horizontal, weights), [w, h, 3, 1, 2]);
        let xyb = g.add(Op::Opsin(rgb, vertical), [w, h, 3, 1, 2]);
        let lf = g.gaussian(xyb, SIGMA_LF as f32);
        let mf = g.same(Op::Subtract(xyb, lf), lf, 3);
        let mf_blur = g.gaussian(mf, SIGMA_HF as f32);
        let hf = g.same(Op::High(mf, mf_blur), mf_blur, 2);
        let hf_blur = g.gaussian(hf, SIGMA_UHF as f32);
        g.output = g.same(Op::Finish([lf, mf_blur, hf, hf_blur]), hf_blur, 10);
        let mut lookahead = vec![0; g.nodes.len()];
        for node in &g.nodes {
            let sources: Vec<usize> = match &node.op {
                Op::Input => vec![],
                Op::MirrorH(s, _)
                | Op::MirrorV(s, _)
                | Op::Reduce(s, _)
                | Op::GaussianH(s, _)
                | Op::GaussianV(s, _)
                | Op::Expand(s, _) => vec![*s],
                Op::Opsin(a, b) | Op::Subtract(a, b) | Op::High(a, b) => vec![*a, *b],
                Op::Finish(s) => s.to_vec(),
            };
            for source in sources {
                lookahead[source] = lookahead[source].max(node.latency - g.nodes[source].latency);
            }
        }
        for (id, node) in g.nodes.iter_mut().enumerate() {
            // A stage retains its direct consumers' lag. Deeper consumers
            // revisit their own cached intermediates, not every ancestor.
            // Only the final rows are revisited by overlapping scoring strips.
            let support = if id == g.output {
                2 * local_halo() + 9
            } else {
                2 * lookahead[id] + 9
            };
            // Rows are visited through a circular cache. A power-of-two capacity
            // replaces division on every recursive producer/consumer lookup.
            let capacity = support
                .div_ceil(node.step)
                .min(node.height)
                .next_power_of_two();
            node.cache = vec![None; capacity];
        }
        g
    }
    fn add(&mut self, op: Op, shape: [usize; 5]) -> usize {
        let [width, height, channels, step, latency] = shape;
        let index = self.nodes.len();
        self.nodes.push(Node {
            op,
            width,
            height,
            channels,
            step,
            latency,
            cache: Vec::new(),
            generated: 0,
        });
        index
    }
    fn same(&mut self, op: Op, source: usize, channels: usize) -> usize {
        let n = &self.nodes[source];
        self.add(op, [n.width, n.height, channels, n.step, n.latency])
    }
    fn gaussian(&mut self, source: usize, sigma: f32) -> usize {
        let (factor, sigma) = blur::geometry(sigma);
        let n = &self.nodes[source];
        let [w, h, c, step, latency] = [n.width, n.height, n.channels, n.step, n.latency];
        let reduced = if factor == 1 {
            source
        } else {
            self.add(
                Op::Reduce(source, factor),
                [
                    w.div_ceil(factor),
                    h.div_ceil(factor),
                    c,
                    step * factor,
                    latency + (factor - 1) * step,
                ],
            )
        };
        let kernel = crate::exact_blur::compute_kernel(sigma);
        let radius = kernel.len() / 2;
        let inverse = 1.0 / kernel.iter().sum::<f32>();
        let scaled = kernel.iter().map(|v| v * inverse).collect();
        let kernel_id = self.kernels.len();
        self.kernels.push((kernel, scaled));
        let horizontal = self.same(Op::GaussianH(reduced, kernel_id), reduced, c);
        let n = &self.nodes[horizontal];
        let vertical = self.add(
            Op::GaussianV(horizontal, kernel_id),
            [n.width, n.height, c, n.step, n.latency + radius * n.step],
        );
        if factor == 1 {
            vertical
        } else {
            let latency = self.nodes[vertical].latency + factor * step;
            self.add(Op::Expand(vertical, factor), [w, h, c, step, latency])
        }
    }
    fn row(&mut self, id: usize, y: usize) -> &[f32] {
        self.ensure(id, y);
        self.cached(id, y)
    }
    fn cached(&self, id: usize, y: usize) -> &[f32] {
        let node = &self.nodes[id];
        let (stored, row) = node.cache[y & (node.cache.len() - 1)].as_ref().unwrap();
        assert_eq!(
            *stored, y,
            "row-cache support must retain all consumer inputs"
        );
        row
    }
    fn ensure(&mut self, id: usize, y: usize) {
        let node = &mut self.nodes[id];
        let slot = y & (node.cache.len() - 1);
        if node.cache[slot]
            .as_ref()
            .is_some_and(|(stored, _)| *stored == y)
        {
            return;
        }
        let mut out = node.cache[slot]
            .take()
            .map(|(_, r)| r)
            .unwrap_or_else(|| vec![0.0; node.width * node.channels]);
        let (op, w, c) = (node.op, node.width, node.channels);
        match op {
            Op::Input => {
                let (r, gb) = out.split_at_mut(w);
                let (g, b) = gb.split_at_mut(w);
                self.input
                    .linear_planar_row(y * self.factor, self.factor, [r, g, b]);
            }
            Op::MirrorH(source, weights) => {
                self.ensure(source, y);
                let row = self.cached(source, y);
                for channel in 0..c {
                    mirror_horizontal(
                        &row[channel * w..(channel + 1) * w],
                        weights,
                        &mut out[channel * w..(channel + 1) * w],
                    );
                }
            }
            Op::MirrorV(source, weights) => {
                let h = self.nodes[source].height;
                let ys: [usize; 5] =
                    std::array::from_fn(|i| mirror(y as isize + i as isize - 2, h));
                for yy in ys {
                    self.ensure(source, yy);
                }
                mirror_vertical(ys.map(|yy| self.cached(source, yy)), weights, &mut out);
            }
            Op::Opsin(a, b) => {
                self.ensure(a, y);
                self.ensure(b, y);
                let (a, b) = (self.cached(a, y), self.cached(b, y));
                opsin_row(a, b, self.intensity, w, &mut out);
            }
            Op::Reduce(source, factor) => {
                let sw = self.nodes[source].width;
                let end = ((y + 1) * factor).min(self.nodes[source].height);
                let count = end - y * factor;
                for yy in y * factor..end {
                    self.ensure(source, yy);
                }
                let views: [&[f32]; 4] = std::array::from_fn(|i| {
                    if i < count {
                        self.cached(source, y * factor + i)
                    } else {
                        &[]
                    }
                });
                reduce_row(&views[..count], sw, w, factor, &mut out);
            }
            Op::GaussianH(source, kernel_id) => {
                self.ensure(source, y);
                let row = self.cached(source, y);
                let (kernel, scaled) = &self.kernels[kernel_id];
                for channel in 0..c {
                    stream_blur::horizontal(
                        &row[channel * w..(channel + 1) * w],
                        kernel,
                        scaled,
                        &mut out[channel * w..(channel + 1) * w],
                    );
                }
            }
            Op::GaussianV(source, kernel_id) => {
                let radius = self.kernels[kernel_id].0.len() / 2;
                let start = y.saturating_sub(radius);
                let end = (y + radius + 1).min(self.nodes[source].height);
                let count = end - start;
                for yy in start..end {
                    self.ensure(source, yy);
                }
                let (kernel, scaled) = &self.kernels[kernel_id];
                let raw = &kernel[start + radius - y..end + radius - y];
                let mut border = [0.0; 64];
                let weights = if raw.len() == kernel.len() {
                    &scaled[..]
                } else {
                    let inv = 1.0 / raw.iter().sum::<f32>();
                    for (out, value) in border.iter_mut().zip(raw) {
                        *out = value * inv;
                    }
                    &border[..count]
                };
                let views: [&[f32]; 64] = std::array::from_fn(|i| {
                    if i < count {
                        self.cached(source, start + i)
                    } else {
                        &[]
                    }
                });
                stream_blur::vertical(&views[..count], weights, &mut out);
            }
            Op::Expand(source, factor) => {
                let sw = self.nodes[source].width;
                let sh = self.nodes[source].height;
                let (a, b, fy) = coordinate(y, factor, sh);
                self.ensure(source, a);
                self.ensure(source, b);
                let (a, b) = (self.cached(source, a), self.cached(source, b));
                expand_row(a, b, [sw, w, factor], fy, &mut out);
            }
            Op::Subtract(a, b) => {
                self.ensure(a, y);
                self.ensure(b, y);
                let (a, b) = (self.cached(a, y), self.cached(b, y));
                subtract_row(a, b, &mut out);
            }
            Op::High(a, b) => {
                self.ensure(a, y);
                self.ensure(b, y);
                let (a, b) = (self.cached(a, y), self.cached(b, y));
                high_row(a, b, w, &mut out);
            }
            Op::Finish(sources) => {
                for id in sources {
                    self.ensure(id, y);
                }
                finish_row(sources.map(|id| self.cached(id, y)), w, &mut out);
            }
        }
        self.nodes[id].cache[slot] = Some((y, out));
        self.nodes[id].generated += 1;
    }
    fn prepare(&mut self, y0: usize, y1: usize, pool: &image::BufferPool) -> psycho::PsychoImage {
        let w = self.nodes[self.output].width;
        let mut result = psycho::PsychoImage::from_pool(w, y1 - y0, pool);
        for y in y0..y1 {
            let row = self.row(self.output, y);
            for c in 0..10 {
                let plane = match c {
                    0..=1 => &mut result.uhf[c],
                    2..=3 => &mut result.hf[c - 2],
                    4..=6 => result.mf.plane_mut(c - 4),
                    _ => result.lf.plane_mut(c - 7),
                };
                plane
                    .row_mut(y - y0)
                    .copy_from_slice(&row[c * w..(c + 1) * w]);
            }
        }
        result
    }
}
fn mirror(mut x: isize, size: usize) -> usize {
    while x < 0 || x >= size as isize {
        x = if x < 0 {
            -x - 1
        } else {
            2 * size as isize - 1 - x
        };
    }
    x as usize
}
fn coordinate(pixel: usize, factor: usize, length: usize) -> (usize, usize, f32) {
    let p = ((pixel as f32 + 0.5) / factor as f32 - 0.5).max(0.0);
    let a = (p as usize).min(length - 1);
    (a, (a + 1).min(length - 1), p - a as f32)
}
fn local_halo() -> usize {
    4.max(blur::support(MASK_RADIUS) + 3)
}

#[archmage::autoversion]
fn mirror_horizontal(
    _token: archmage::SimdToken,
    input: &[f32],
    weights: [f32; 3],
    out: &mut [f32],
) {
    let [a, b, c] = weights;
    let begin = 2.min(input.len());
    let end = input.len().saturating_sub(2).max(begin);
    for x in (0..begin).chain(end..input.len()) {
        let at = |dx| input[mirror(x as isize + dx, input.len())];
        out[x] = at(0) * a + (at(-1) + at(1)) * b + (at(-2) + at(2)) * c;
    }
    let full = (end - begin) / 8 * 8;
    for (block, dst) in out[begin..begin + full]
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .enumerate()
    {
        let start = begin + block * 8 - 2;
        let values: &[f32; 12] = input[start..start + 12].try_into().unwrap();
        for i in 0..8 {
            dst[i] = values[i + 2] * a
                + (values[i + 1] + values[i + 3]) * b
                + (values[i] + values[i + 4]) * c;
        }
    }
    for x in begin + full..end {
        out[x] =
            input[x] * a + (input[x - 1] + input[x + 1]) * b + (input[x - 2] + input[x + 2]) * c;
    }
}
#[archmage::autoversion]
fn mirror_vertical(
    _token: archmage::SimdToken,
    rows: [&[f32]; 5],
    weights: [f32; 3],
    out: &mut [f32],
) {
    let [a, b, c] = weights;
    for (i, v) in out.iter_mut().enumerate() {
        *v = rows[2][i] * a + (rows[1][i] + rows[3][i]) * b + (rows[0][i] + rows[4][i]) * c;
    }
}
#[cfg(feature = "simd-opsin")]
use crate::opsin_rows::convert as opsin_row;

#[cfg(not(feature = "simd-opsin"))]
#[archmage::autoversion]
fn opsin_row(
    _token: archmage::SimdToken,
    a: &[f32],
    b: &[f32],
    intensity: f32,
    w: usize,
    out: &mut [f32],
) {
    let (min0, min1, min2) = opsin::opsin_absorbance(0.0, 0.0, 0.0, false);
    for x in 0..w {
        let (p0, p1, p2) = opsin::opsin_absorbance(
            b[x] * intensity,
            b[w + x] * intensity,
            b[2 * w + x] * intensity,
            true,
        );
        let [p0, p1, p2] = [p0, p1, p2].map(|p| p.max(1e-4));
        let [s0, s1, s2] = [p0, p1, p2].map(|p| (opsin::gamma(p) / p).max(1e-4));
        let (v0, v1, v2) = opsin::opsin_absorbance(
            a[x] * intensity,
            a[w + x] * intensity,
            a[2 * w + x] * intensity,
            false,
        );
        let (v0, v1, v2) = (
            (v0 * s0).max(min0),
            (v1 * s1).max(min1),
            (v2 * s2).max(min2),
        );
        out[x] = v0 - v1;
        out[w + x] = v0 + v1;
        out[2 * w + x] = v2;
    }
}
#[archmage::autoversion]
fn reduce_row(
    _token: archmage::SimdToken,
    rows: &[&[f32]],
    sw: usize,
    w: usize,
    factor: usize,
    out: &mut [f32],
) {
    match factor {
        2 => reduce_fixed::<2, 16>(rows, sw, w, out),
        4 => reduce_fixed::<4, 32>(rows, sw, w, out),
        _ => unreachable!(),
    }
}

#[inline(always)]
fn reduce_fixed<const F: usize, const N: usize>(
    rows: &[&[f32]],
    sw: usize,
    w: usize,
    out: &mut [f32],
) {
    for (channel, dst) in out.chunks_exact_mut(w).enumerate() {
        let full = if rows.len() == F { sw / F / 8 * 8 } else { 0 };
        for (block, target) in dst[..full].as_chunks_mut::<8>().0.iter_mut().enumerate() {
            let mut sums = [0.0; 8];
            for row in rows {
                let start = channel * sw + block * 8 * F;
                let values: &[f32; N] = row[start..start + N].try_into().unwrap();
                for offset in 0..F {
                    for lane in 0..8 {
                        sums[lane] += values[lane * F + offset];
                    }
                }
            }
            for lane in 0..8 {
                target[lane] = sums[lane] / (F * F) as f32;
            }
        }
        for (x, value) in dst.iter_mut().enumerate().skip(full) {
            let x0 = x * F;
            let x1 = (x0 + F).min(sw);
            let mut sum = 0.0;
            for row in rows {
                for &v in &row[channel * sw + x0..channel * sw + x1] {
                    sum += v;
                }
            }
            *value = sum / ((x1 - x0) * rows.len()) as f32;
        }
    }
}

#[archmage::autoversion]
#[allow(clippy::too_many_arguments)]
fn expand_row(
    _token: archmage::SimdToken,
    a: &[f32],
    b: &[f32],
    shape: [usize; 3],
    fy: f32,
    out: &mut [f32],
) {
    let [sw, w, factor] = shape;
    for (channel, dst) in out.chunks_exact_mut(w).enumerate() {
        let (a, b) = (
            &a[channel * sw..(channel + 1) * sw],
            &b[channel * sw..(channel + 1) * sw],
        );
        match factor {
            2 => blur::expand_row::<2>(a, b, fy, dst),
            4 => blur::expand_row::<4>(a, b, fy, dst),
            _ => unreachable!(),
        }
    }
}
#[archmage::autoversion]
fn subtract_row(_token: archmage::SimdToken, a: &[f32], b: &[f32], out: &mut [f32]) {
    for ((v, a), b) in out.iter_mut().zip(a).zip(b) {
        *v = a - b;
    }
}
#[archmage::autoversion]
fn high_row(_token: archmage::SimdToken, a: &[f32], b: &[f32], w: usize, out: &mut [f32]) {
    let s = SUPPRESS_S as f32;
    let yw = SUPPRESS_XY as f32;
    for x in 0..w {
        let y = a[w + x] - b[w + x];
        out[x] = (a[x] - b[x]) * (yw / y.mul_add(y, yw)).mul_add(1.0 - s, s);
        out[w + x] = y;
    }
}
#[archmage::autoversion]
fn finish_row(_token: archmage::SimdToken, rows: [&[f32]; 4], w: usize, out: &mut [f32]) {
    let [lf, mf, hf, blurred] = rows;
    let remove = |v: f32, range: f32| (v.abs() - range).max(0.0).copysign(v);
    let amplify = |v: f32, range: f32| v + v.abs().min(range).copysign(v);
    let clamp = |v: f32, limit: f32| {
        let c = v.min(limit).max(-limit);
        (v - c).mul_add(0.724_216_146_f64 as f32, c)
    };
    for x in 0..w {
        let hc = clamp(blurred[w + x], MAXCLAMP_HF as f32);
        out[x] = remove(hf[x] - blurred[x], REMOVE_UHF_RANGE as f32);
        out[w + x] = clamp(hf[w + x] - hc, MAXCLAMP_UHF as f32) * MUL_Y_UHF as f32;
        out[2 * w + x] = remove(blurred[x], REMOVE_HF_RANGE as f32);
        out[3 * w + x] = amplify(hc * MUL_Y_HF as f32, ADD_HF_RANGE as f32);
        out[4 * w + x] = remove(mf[x], REMOVE_MF_RANGE as f32);
        out[5 * w + x] = amplify(mf[w + x], ADD_MF_RANGE as f32);
        out[6 * w + x] = mf[2 * w + x];
        out[7 * w + x] = lf[x] * XMUL_LF_TO_VALS as f32;
        out[8 * w + x] = lf[w + x] * YMUL_LF_TO_VALS as f32;
        out[9 * w + x] = (Y_TO_B_MUL_LF_TO_VALS as f32).mul_add(lf[w + x], lf[2 * w + x])
            * BMUL_LF_TO_VALS as f32;
    }
}

pub(super) fn compute(
    a: &ingress::EncodedRows<'_>,
    b: &ingress::EncodedRows<'_>,
    rows: usize,
    params: &ButteraugliParams,
) -> Result<diff::InternalResult, Box<dyn Error>> {
    let columns = match std::env::var("MARGARINE_TILE_COLUMNS") {
        Ok(value) => value.parse::<usize>()?,
        Err(std::env::VarError::NotPresent) => 512,
        Err(error) => return Err(error.into()),
    };
    compute_geometry(a, b, rows, columns, params)
}

fn compute_geometry(
    a: &ingress::EncodedRows<'_>,
    b: &ingress::EncodedRows<'_>,
    rows: usize,
    columns: usize,
    params: &ButteraugliParams,
) -> Result<diff::InternalResult, Box<dyn Error>> {
    if columns == 0 || !columns.is_multiple_of(4) {
        return Err("tile columns must be a positive multiple of four".into());
    }
    let (w, h) = (a.width, a.height);
    let scale = |factor| {
        let (sw, sh) = (w.div_ceil(factor), h.div_ceil(factor));
        let mut result = image::ImageF::new(sw, sh);
        let columns = if cfg!(feature = "row-tiles") {
            columns
        } else {
            sw
        };
        for left in (0..sw).step_by(columns) {
            let right = (left + columns).min(sw);
            let x0 = left.saturating_sub(strips::halo()) / 4 * 4;
            let x1 = (right + strips::halo())
                .div_ceil(4)
                .saturating_mul(4)
                .min(sw);
            let a = a.columns(x0 * factor, (x1 * factor).min(w));
            let b = b.columns(x0 * factor, (x1 * factor).min(w));
            let pool = image::BufferPool::with_capacity(if sh > rows { 32 } else { 0 });
            // One strip has no vertical overlap to reuse.
            if sh <= rows {
                let map = strips::single_scale_encoded(
                    &a,
                    &b,
                    factor,
                    [0, 0, x1 - x0, sh],
                    params,
                    &pool,
                );
                for y in 0..sh {
                    result.row_mut(y)[left..right]
                        .copy_from_slice(&map.row(y)[left - x0..right - x0]);
                }
                continue;
            }
            let mut a = Graph::new(&a, factor, params.intensity_target());
            let mut b = Graph::new(&b, factor, params.intensity_target());
            let mut previous_height = 0;
            for start in (0..sh).step_by(rows) {
                let end = (start + rows).min(sh);
                let y0 = start.saturating_sub(local_halo()) / 4 * 4;
                let y1 = (end + local_halo()).div_ceil(4).saturating_mul(4).min(sh);
                if y1 - y0 != previous_height {
                    pool.clear();
                    previous_height = y1 - y0;
                }
                let (pa, pb) =
                    diff::maybe_join(|| a.prepare(y0, y1, &pool), || b.prepare(y0, y1, &pool));
                let map = strips::finish_scale(pa, pb, params, &pool);
                for y in start..end {
                    result.row_mut(y)[left..right]
                        .copy_from_slice(&map.row(y - y0)[left - x0..right - x0]);
                }
                map.recycle(&pool);
            }
        }
        result
    };
    let sub = (!params.single_resolution() && w >= 15 && h >= 15).then(|| scale(2));
    let mut map = scale(1);
    if let Some(sub) = sub {
        diff::add_supersampled_2x(&sub, 0.5, &mut map);
    }
    let (score, pnorm_3) = diff::compute_score_from_diffmap(&map);
    Ok(diff::InternalResult {
        score,
        pnorm_3,
        diffmap: Some(map),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn column_tiles_preserve_full_map_with_rgb16_stride_and_odd_edges() {
        use ingress::{EncodedRows, Samples};
        let (w, h, stride) = (1031, 137, 1031 * 3 + 7);
        let mut a = vec![0; stride * h];
        let mut b = a.clone();
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let i = y * stride + x * 3 + c;
                    a[i] = ((x * 113 + y * 331 + c * 19937 + x * y * 7) % 65536) as u16;
                    b[i] = a[i].saturating_add(((x + y + c) % 101) as u16);
                }
            }
        }
        let a = EncodedRows::new(Samples::U16(&a), w, h, stride, 3).unwrap();
        let b = EncodedRows::new(Samples::U16(&b), w, h, stride, 3).unwrap();
        let params = ButteraugliParams::default();
        let expected = strips::compute(
            &a.linear_strip(0, h),
            &b.linear_strip(0, h),
            w,
            h,
            w * 3,
            64,
            &params,
        )
        .unwrap();
        for columns in [256, 512, 768, 1024] {
            let actual = compute_geometry(&a, &b, 64, columns, &params).unwrap();
            for y in 0..h {
                assert_eq!(
                    actual.diffmap.as_ref().unwrap().row(y),
                    expected.diffmap.as_ref().unwrap().row(y),
                    "columns {columns}, row {y}"
                );
            }
        }
        for columns in [0, 1, 3, 513] {
            assert!(compute_geometry(&a, &b, 64, columns, &params).is_err());
        }
    }
    #[test]
    fn every_frequency_row_matches_shared_pipeline_without_recomputation() {
        use ingress::{EncodedRows, Samples};
        for (w, h) in [(1, 1), (3, 5), (31, 73), (129, 277)] {
            let stride = w * 3 + 7;
            let mut data = vec![u16::MAX; stride * h];
            for y in 0..h {
                for x in 0..w {
                    for c in 0..3 {
                        data[y * stride + x * 3 + c] =
                            ((x * 217 + y * 1597 + c * 21739 + x * y * 17) % 65536) as u16;
                    }
                }
            }
            let input = EncodedRows::new(Samples::U16(&data), w, h, stride, 3).unwrap();
            for factor in [1, 2] {
                let (sw, sh) = (w.div_ceil(factor), h.div_ceil(factor));
                let pool = image::BufferPool::with_capacity(0);
                let mut linear = image::Image3F::new(sw, sh);
                for y in 0..sh {
                    let (r, g, b) = linear.planes_mut();
                    input.linear_planar_row(
                        y * factor,
                        factor,
                        [r.row_mut(y), g.row_mut(y), b.row_mut(y)],
                    );
                }
                let xyb = opsin::opsin_dynamics_image(&linear, 80.0, &pool);
                let expected = psycho::separate_frequencies_owned(xyb, &pool);
                let mut graph = Graph::new(&input, factor, 80.0);
                for start in (0..sh).step_by(64) {
                    let y0 = start.saturating_sub(local_halo()) / 4 * 4;
                    let y1 = (start + 64 + local_halo())
                        .div_ceil(4)
                        .saturating_mul(4)
                        .min(sh);
                    let actual = graph.prepare(y0, y1, &pool);
                    let planes = |p: &psycho::PsychoImage, y: usize| {
                        (0..10)
                            .map(|c| {
                                match c {
                                    0..=1 => p.uhf[c].row(y),
                                    2..=3 => p.hf[c - 2].row(y),
                                    4..=6 => p.mf.plane(c - 4).row(y),
                                    _ => p.lf.plane(c - 7).row(y),
                                }
                                .to_vec()
                            })
                            .collect::<Vec<_>>()
                    };
                    for y in y0..y1 {
                        assert_eq!(
                            planes(&actual, y - y0),
                            planes(&expected, y),
                            "{w}x{h}, factor {factor}, row {y}"
                        );
                    }
                }
                for (id, node) in graph.nodes.iter().enumerate() {
                    assert_eq!(
                        node.generated, node.height,
                        "node {id} recomputed rows on {w}x{h}, factor {factor}"
                    );
                }
            }
        }
    }
}
