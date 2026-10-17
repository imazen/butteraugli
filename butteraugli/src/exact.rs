//! libjxl v0.12 arithmetic, isolated from the default optimized kernels.
//!
//! Operation order follows a7a9c787's butteraugli.cc, fast_math-inl.h and
//! cms/transfer_functions-inl.h. Explicit mul_add gives fused rounding on
//! machines with and without hardware FMA.
#![allow(clippy::excessive_precision)]
// Explicit FMA must inline into the archmage target-feature body.
#![allow(clippy::inline_always)]
use crate::image::{BufferPool, Image3F, ImageF};
use crate::psycho::PsychoImage;
use crate::{ButteraugliError, ButteraugliParams};
use enough::Stop;

pub(crate) fn blur_numerics<const EXACT: bool>(
    input: &ImageF,
    sigma: f32,
    pool: &BufferPool,
    stop: &dyn Stop,
) -> Result<ImageF, enough::StopReason> {
    if EXACT {
        blur(input, sigma, pool, stop)
    } else {
        crate::blur::gaussian_blur_stop(input, sigma, pool, stop)
    }
}

fn kernel(sigma: f32) -> Vec<f32> {
    let scale = -1.0 / (2.0 * f64::from(sigma) * f64::from(sigma));
    let radius = (2.25_f32 * sigma.abs()).max(1.0) as i32;
    (-radius..=radius)
        .map(|i| (scale * f64::from(i) * f64::from(i)).exp() as f32)
        .collect()
}

fn mirror(mut p: isize, n: usize) -> usize {
    let n = n as isize;
    while p < 0 || p >= n {
        p = if p < 0 { -p - 1 } else { 2 * n - 1 - p };
    }
    p as usize
}

fn blur5(
    input: &ImageF,
    k: &[f32],
    pool: &BufferPool,
    stop: &dyn Stop,
) -> Result<ImageF, enough::StopReason> {
    let scale = 1.0 / k.iter().sum::<f32>();
    let w = [k[2] * scale, k[1] * scale, k[0] * scale];
    let width = input.width();
    let height = input.height();
    // Pin the AVX2 reference's small-width fallback, independently of the
    // Rust dispatch tier: libjxl Separable5 uses SlowSeparable5 below 8+2.
    if width < 10 {
        let mut out = ImageF::from_pool_dirty(width, height, pool);
        for y in 0..height {
            stop.check()?;
            for x in 0..width {
                let mut sum = 0.0_f32;
                for dy in -2_isize..=2 {
                    for dx in -2_isize..=2 {
                        sum += input.row(mirror(y as isize + dy, height))
                            [mirror(x as isize + dx, width)]
                            * w[dx.unsigned_abs()]
                            * w[dy.unsigned_abs()];
                    }
                }
                out.row_mut(y)[x] = sum;
            }
        }
        return Ok(out);
    }
    let mut temp = ImageF::from_pool_dirty(width, height, pool);
    blur5_pass::<false>(archmage::ScalarToken, input, &mut temp, w, stop)?;
    let mut out = ImageF::from_pool_dirty(width, height, pool);
    blur5_pass::<true>(archmage::ScalarToken, &temp, &mut out, w, stop)?;
    temp.recycle(pool);
    Ok(out)
}

#[archmage::attune(make(dispatch, +v4(avx512)), define(f32x8))]
fn blur5_pass<const VERTICAL: bool>(
    token: Token,
    input: &ImageF,
    out: &mut ImageF,
    w: [f32; 3],
    stop: &dyn Stop,
) -> Result<(), enough::StopReason> {
    let width = input.width();
    let weights = w.map(|v| f32x8::splat_t(token, v));
    for y in 0..input.height() {
        if y % 64 == 0 {
            stop.check()?;
        }
        let rows = std::array::from_fn::<_, 5, _>(|i| {
            input.row(if VERTICAL {
                mirror(y as isize + i as isize - 2, input.height())
            } else {
                y
            })
        });
        let scalar = |x: usize| {
            let at = |i: usize| {
                rows[i][if VERTICAL {
                    x
                } else {
                    mirror(x as isize + i as isize - 2, width)
                }]
            };
            (at(0) + at(4)).mul_add(w[2], (at(1) + at(3)).mul_add(w[1], at(2) * w[0]))
        };
        let (begin, end) = if VERTICAL { (0, width) } else { (2, width - 2) };
        let dst = out.row_mut(y);
        for (x, v) in dst[..begin].iter_mut().enumerate() {
            *v = scalar(x);
        }
        let (chunks, tail) = f32x8::partition_slice_mut_t(token, &mut dst[begin..end]);
        let tail_start = begin + chunks.len() * f32x8::LANES;
        for (i, chunk) in chunks.iter_mut().enumerate() {
            let x = begin + i * f32x8::LANES;
            let at = |j: usize| {
                f32x8::from_slice_t(token, &rows[j][if VERTICAL { x } else { x + j - 2 }..])
            };
            (at(0) + at(4))
                .mul_add_portable(
                    weights[2],
                    (at(1) + at(3)).mul_add_portable(weights[1], at(2) * weights[0]),
                )
                .store(chunk);
        }
        for (i, v) in tail.iter_mut().enumerate() {
            *v = scalar(tail_start + i);
        }
        for (i, v) in dst[end..].iter_mut().enumerate() {
            *v = scalar(end + i);
        }
    }
    Ok(())
}

// Keep each pixel's libjxl summation order; vector lanes are independent pixels.
#[inline(always)]
fn fir(k: &[f32], at: impl Fn(usize) -> f32) -> f32 {
    let radius = k.len() / 2;
    if k.len() == 7 {
        at(3) * k[3] + (at(0) + at(6)) * k[0] + (at(1) + at(5)) * k[1] + (at(2) + at(4)) * k[2]
    } else {
        let mut sums = [0.0; 4];
        for i in 0..radius {
            let t = (at(i) + at(k.len() - 1 - i)) * k[i];
            if i < 4 {
                sums[i] = t;
            } else {
                sums[i % 4] += t;
            }
        }
        at(radius) * k[radius] + sums[0] + sums[1] + sums[2] + sums[3]
    }
}

#[archmage::attune(make(dispatch, +v4(avx512)), define(f32x8))]
fn blur_pass<const VERTICAL: bool>(
    token: Token,
    input: &ImageF,
    out: &mut ImageF,
    k: &[f32],
    sk: &[f32],
    stop: &dyn Stop,
) -> Result<(), enough::StopReason> {
    let width = input.width();
    let n = k.len();
    let radius = n / 2;
    for y in 0..input.height() {
        if y % 64 == 0 {
            stop.check()?;
        }
        let border = |x: usize| {
            let pos = if VERTICAL { y } else { x };
            let length = if VERTICAL { input.height() } else { width };
            let lo = pos.saturating_sub(radius);
            let hi = (pos + radius).min(length - 1);
            let mut z = 0.0;
            let mut v = 0.0;
            for j in lo..=hi {
                let weight = k[j + radius - pos];
                z += weight;
                v += if VERTICAL {
                    input.row(j)[x]
                } else {
                    input.row(y)[j]
                } * weight;
            }
            v * (1.0 / z)
        };
        let dst = out.row_mut(y);
        if VERTICAL && (y < radius || y + radius >= input.height()) {
            // Clipped vertical borders share their normalization across all columns.
            let lo = y.saturating_sub(radius);
            let hi = (y + radius).min(input.height() - 1);
            let mut z = 0.0;
            for j in lo..=hi {
                z += k[j + radius - y];
            }
            let scale = f32x8::splat_t(token, 1.0 / z);
            let (chunks, tail) = f32x8::partition_slice_mut_t(token, dst);
            let tail_start = chunks.len() * f32x8::LANES;
            for (i, chunk) in chunks.iter_mut().enumerate() {
                let x = i * f32x8::LANES;
                let mut v = f32x8::zero_t(token);
                for j in lo..=hi {
                    v += f32x8::from_slice_t(token, &input.row(j)[x..])
                        * f32x8::splat_t(token, k[j + radius - y]);
                }
                (v * scale).store(chunk);
            }
            for (i, v) in tail.iter_mut().enumerate() {
                *v = border(tail_start + i);
            }
            continue;
        }
        let (begin, end) = if VERTICAL {
            (0, width)
        } else {
            (
                radius.min(width),
                width.saturating_sub(radius).max(radius.min(width)),
            )
        };
        for (x, v) in dst[..begin].iter_mut().enumerate() {
            *v = border(x);
        }
        let (chunks, tail) = f32x8::partition_slice_mut_t(token, &mut dst[begin..end]);
        let tail_start = begin + chunks.len() * f32x8::LANES;
        for (i, chunk) in chunks.iter_mut().enumerate() {
            let x = begin + i * f32x8::LANES;
            let at = |j: usize| {
                f32x8::from_slice_t(
                    token,
                    if VERTICAL {
                        &input.row(y + j - radius)[x..]
                    } else {
                        &input.row(y)[x + j - radius..]
                    },
                )
            };
            let weight = |j| f32x8::splat_t(token, sk[j]);
            let value = if n == 7 {
                at(3) * weight(3)
                    + (at(0) + at(6)) * weight(0)
                    + (at(1) + at(5)) * weight(1)
                    + (at(2) + at(4)) * weight(2)
            } else {
                let mut sums = [f32x8::zero_t(token); 4];
                for j in 0..radius {
                    let t = (at(j) + at(n - 1 - j)) * weight(j);
                    if j < 4 {
                        sums[j] = t;
                    } else {
                        sums[j % 4] += t;
                    }
                }
                at(radius) * weight(radius) + sums[0] + sums[1] + sums[2] + sums[3]
            };
            value.store(chunk);
        }
        for (i, v) in tail.iter_mut().enumerate() {
            let x = tail_start + i;
            *v = fir(sk, |j| {
                if VERTICAL {
                    input.row(y + j - radius)[x]
                } else {
                    input.row(y)[x + j - radius]
                }
            });
        }
        for (i, v) in dst[end..].iter_mut().enumerate() {
            *v = border(end + i);
        }
    }
    Ok(())
}

pub(crate) fn blur(
    input: &ImageF,
    sigma: f32,
    pool: &BufferPool,
    stop: &dyn Stop,
) -> Result<ImageF, enough::StopReason> {
    let k = kernel(sigma);
    if k.len() == 5 {
        return blur5(input, &k, pool, stop);
    }
    let scale = 1.0 / k.iter().sum::<f32>();
    let sk: Vec<_> = k.iter().map(|&v| v * scale).collect();
    let mut temp = ImageF::from_pool_dirty(input.width(), input.height(), pool);
    blur_pass::<false>(archmage::ScalarToken, input, &mut temp, &k, &sk, stop)?;
    let mut out = ImageF::from_pool_dirty(input.width(), input.height(), pool);
    blur_pass::<true>(archmage::ScalarToken, &temp, &mut out, &k, &sk, stop)?;
    temp.recycle(pool);
    Ok(out)
}

pub(crate) fn srgb_lut(numerics: crate::Numerics) -> &'static [f32; 256] {
    static EXACT: std::sync::LazyLock<[f32; 256]> =
        std::sync::LazyLock::new(|| std::array::from_fn(|i| srgb(i as u8)));
    match numerics {
        crate::Numerics::Fast => &crate::opsin::SRGB_TO_LINEAR_LUT,
        crate::Numerics::LibjxlExact => &EXACT,
    }
}

pub(crate) fn srgb_bytes(rgb: &[u8], numerics: crate::Numerics) -> Vec<f32> {
    let lut = srgb_lut(numerics);
    rgb.iter().map(|&v| lut[v as usize]).collect()
}
#[inline(always)]
fn rational(x: f32, p: &[f32], q: &[f32]) -> f32 {
    #[inline(always)]
    fn horner(x: f32, c: &[f32]) -> f32 {
        let mut y = c[c.len() - 1];
        for &v in c[..c.len() - 1].iter().rev() {
            y = y.mul_add(x, v);
        }
        y
    }
    horner(x, p) / horner(x, q)
}

fn srgb(v: u8) -> f32 {
    let x = f32::from(v) * (1.0_f32 / 255.0);
    if x <= 0.04045 {
        x * (1.0_f32 / 12.92)
    } else {
        rational(
            x,
            &[
                2.200248328e-4,
                1.043637593e-2,
                1.624820318e-1,
                7.961564959e-1,
                8.210152774e-1,
            ],
            &[
                2.631846970e-1,
                1.076976492,
                4.987528350e-1,
                -5.512498495e-2,
                6.521209011e-3,
            ],
        )
    }
}
#[inline(always)]
fn gamma(v: f32) -> f32 {
    let x = v.max(0.0) + 9.9710635769299145_f32;
    let bits = x.to_bits() as i32;
    let exp = bits.wrapping_sub(0x3f2aaaab) >> 23;
    let m = f32::from_bits(bits.wrapping_sub(exp << 23) as u32) - 1.0;
    let log = rational(
        m,
        &[
            -1.8503833400518310e-6,
            1.4287160470083755,
            7.4245873327820566e-1,
        ],
        &[
            9.9032814277590719e-1,
            1.0096718572241148,
            1.7409343003366853e-1,
        ],
    ) + exp as f32;
    (19.245013259874995_f32 * std::f32::consts::LN_2).mul_add(log, -23.16046239805755)
}
#[inline(always)]
fn absorb(v: [f32; 3]) -> [f32; 3] {
    let [r, g, b] = v;
    [
        0.29956550340058319_f32.mul_add(
            r,
            0.63373087833825936_f32
                .mul_add(g, 0.077705617820981968_f32.mul_add(b, 1.7557483643287353)),
        ),
        0.22158691104574774_f32.mul_add(
            r,
            0.69391388044116142_f32.mul_add(g, 0.0987313588422_f32.mul_add(b, 1.7557483643287353)),
        ),
        0.02_f32.mul_add(
            r,
            0.02_f32.mul_add(g, 0.20480129041026129_f32.mul_add(b, 12.226454707163354)),
        ),
    ]
}

/// Borrowed linear input; both layouts use the same Exact pipeline.
#[derive(Clone, Copy)]
pub(crate) enum Input<'a> {
    Interleaved(&'a [f32]),
    Planar([&'a [f32]; 3], usize),
}

impl Input<'_> {
    pub(crate) fn subsample(self, w: usize, h: usize) -> Vec<f32> {
        match self {
            Self::Interleaved(rgb) => crate::diff::subsample_linear_rgb_2x(rgb, w, h).0,
            Self::Planar(planes, stride) => subsample_planar(planes, w, h, stride),
        }
    }
}

pub(crate) fn prepare(
    input: Input<'_>,
    w: usize,
    h: usize,
    params: &ButteraugliParams,
    pool: &BufferPool,
    stop: &dyn Stop,
) -> Result<PsychoImage, enough::StopReason> {
    let mut image = Image3F::from_pool_dirty(w, h, pool);
    match input {
        Input::Interleaved(rgb) => {
            for y in 0..h {
                for x in 0..w {
                    for c in 0..3 {
                        image.plane_mut(c).row_mut(y)[x] = rgb[(y * w + x) * 3 + c];
                    }
                }
            }
        }
        Input::Planar(planes, stride) => {
            for (c, plane) in planes.iter().enumerate() {
                for y in 0..h {
                    image
                        .plane_mut(c)
                        .row_mut(y)
                        .copy_from_slice(&plane[y * stride..y * stride + w]);
                }
            }
        }
    }
    prepare_image(image, params, pool, stop)
}

fn prepare_image(
    image: Image3F,
    params: &ButteraugliParams,
    pool: &BufferPool,
    stop: &dyn Stop,
) -> Result<PsychoImage, enough::StopReason> {
    let (w, h) = (image.width(), image.height());
    let blurred = [
        blur(image.plane(0), 1.2, pool, stop)?,
        blur(image.plane(1), 1.2, pool, stop)?,
        blur(image.plane(2), 1.2, pool, stop)?,
    ];
    let mut xyb = Image3F::from_pool_dirty(w, h, pool);
    opsin_loop(&image, &blurred, &mut xyb, params.intensity_target(), stop)?;
    image.recycle(pool);
    for b in blurred {
        b.recycle(pool);
    }
    crate::psycho::separate_frequencies_owned_numerics::<true>(xyb, pool, stop)
}

// Autoversion generates archmage capability-token entry points. Helpers inline
// into each tier so explicit mul_add uses native FMA where available.
#[archmage::autoversion]
fn opsin_loop(
    image: &Image3F,
    blurred: &[ImageF; 3],
    xyb: &mut Image3F,
    intensity: f32,
    stop: &dyn Stop,
) -> Result<(), enough::StopReason> {
    let (w, h) = (image.width(), image.height());

    let minima = [
        1.7557483643287353_f32,
        1.7557483643287353,
        12.226454707163354,
    ];
    for y in 0..h {
        if y % 64 == 0 {
            stop.check()?;
        }
        for x in 0..w {
            let pre = absorb(std::array::from_fn(|c| blurred[c].row(y)[x] * intensity));
            let cur = absorb(std::array::from_fn(|c| {
                image.plane(c).row(y)[x] * intensity
            }));
            let mixed: [f32; 3] = std::array::from_fn(|c| {
                let p = pre[c].max(minima[c]).max(1e-4);
                let sensitivity = (gamma(p) / p).max(1e-4);
                (cur[c] * sensitivity).max(minima[c])
            });
            xyb.plane_mut(0).row_mut(y)[x] = mixed[0] - mixed[1];
            xyb.plane_mut(1).row_mut(y)[x] = mixed[0] + mixed[1];
            xyb.plane_mut(2).row_mut(y)[x] = mixed[2];
        }
    }
    Ok(())
}

pub(crate) fn compare_psycho(
    a: &PsychoImage,
    b: &PsychoImage,
    precomputed: Option<&crate::mask::PrecomputedMask>,
    params: &ButteraugliParams,
    pool: &BufferPool,
    stop: &dyn Stop,
) -> Result<ImageF, enough::StopReason> {
    let mut ac = crate::diff::compute_psycho_diff_malta(
        a,
        b,
        params.hf_asymmetry(),
        params.xmul(),
        pool,
        stop,
    )?;
    let mut owned_mask = None;
    let mask = if let Some(precomputed) = precomputed {
        crate::mask::apply_mask_correction_precomputed_stop_numerics::<true>(
            precomputed,
            &b.hf,
            &b.uhf,
            Some(ac.plane_mut(1)),
            pool,
            stop,
        )?;
        &precomputed.mask
    } else {
        owned_mask = Some(crate::mask::compute_mask_from_hf_uhf_stop_numerics::<true>(
            &a.hf,
            &a.uhf,
            &b.hf,
            &b.uhf,
            Some(ac.plane_mut(1)),
            pool,
            stop,
        )?);
        owned_mask.as_ref().unwrap()
    };
    let mut out = ImageF::from_pool_dirty(a.width(), a.height(), pool);
    for y in 0..a.height() {
        if y % 64 == 0 {
            stop.check()?;
        }
        for x in 0..a.width() {
            let mv = crate::mask::mask_y(f64::from(mask.row(y)[x])) as f32;
            let dc_mv = crate::mask::mask_dc_y(f64::from(mask.row(y)[x])) as f32;
            let dc: [f32; 3] = std::array::from_fn(|c| {
                let dc_sum = a.lf.plane(c).row(y)[x] - b.lf.plane(c).row(y)[x];
                dc_sum * dc_sum * crate::consts::WMUL[6 + c] as f32
            });
            let xmul = params.xmul();
            let dc_sum = dc[0] * xmul * dc_mv + dc[1] * dc_mv + dc[2] * dc_mv;
            let ac_sum = ac.plane(0).row(y)[x] * xmul * mv
                + ac.plane(1).row(y)[x] * mv
                + ac.plane(2).row(y)[x] * mv;
            out.row_mut(y)[x] = (dc_sum + ac_sum).sqrt();
        }
    }
    if let Some(mask) = owned_mask {
        mask.recycle(pool);
    }
    ac.recycle(pool);
    Ok(out)
}

pub(crate) fn single(
    a: &[f32],
    b: &[f32],
    w: usize,
    h: usize,
    params: &ButteraugliParams,
    stop: &dyn Stop,
) -> Result<ImageF, enough::StopReason> {
    let pool = BufferPool::new();
    let (a, b) = crate::diff::maybe_join(
        || prepare(Input::Interleaved(a), w, h, params, &pool, stop),
        || prepare(Input::Interleaved(b), w, h, params, &pool, stop),
    );
    compare_psycho(&a?, &b?, None, params, &pool, stop)
}

pub(crate) fn merge(sub: &ImageF, full: &mut ImageF) {
    for y in 0..full.height() {
        for x in 0..full.width() {
            let d = (f64::from(full.row(y)[x]) * 0.85) as f32;
            full.row_mut(y)[x] = d + 0.5 * sub.row(y / 2)[x / 2];
        }
    }
}

pub(crate) fn multires(
    a: &[f32],
    b: &[f32],
    w: usize,
    h: usize,
    params: &ButteraugliParams,
    stop: &dyn Stop,
) -> Result<ImageF, enough::StopReason> {
    if params.single_resolution()
        || w < crate::diff::MIN_SIZE_FOR_SUBSAMPLE
        || h < crate::diff::MIN_SIZE_FOR_SUBSAMPLE
    {
        return single(a, b, w, h, params, stop);
    }
    let (a2, sw, sh) = crate::diff::subsample_linear_rgb_2x(a, w, h);
    let (b2, _, _) = crate::diff::subsample_linear_rgb_2x(b, w, h);
    let (full, sub) = crate::diff::maybe_join(
        || single(a, b, w, h, params, stop),
        || single(&a2, &b2, sw, sh, params, stop),
    );
    let mut full = full?;
    merge(&sub?, &mut full);
    Ok(full)
}

pub(crate) fn linear(
    a: &[f32],
    b: &[f32],
    w: usize,
    h: usize,
    params: &ButteraugliParams,
    stop: &dyn Stop,
) -> Result<crate::diff::InternalResult, ButteraugliError> {
    stop.check().map_err(ButteraugliError::Cancelled)?;
    let dm = if a == b {
        ImageF::new(w, h)
    } else {
        multires(a, b, w, h, params, stop).map_err(ButteraugliError::Cancelled)?
    };
    let (score, pnorm_3) = crate::diff::compute_score_from_diffmap(&dm);
    Ok(crate::diff::InternalResult {
        score,
        pnorm_3,
        diffmap: Some(dm),
    })
}

/// Only half-resolution input is interleaved (12 bytes per half-scale pixel).
/// Sum order matches diff::subsample_linear_rgb_2x, including odd edges.
pub(crate) fn subsample_planar(planes: [&[f32]; 3], w: usize, h: usize, stride: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(w.div_ceil(2) * h.div_ceil(2) * 3);
    for y in (0..h).step_by(2) {
        for x in (0..w).step_by(2) {
            let x1 = (x + 1).min(w - 1);
            let y1 = (y + 1).min(h - 1);
            for p in planes {
                out.push(if x == x1 && y == y1 {
                    p[y * stride + x]
                } else if x == x1 {
                    (p[y * stride + x] + p[y1 * stride + x]) * 0.5
                } else if y == y1 {
                    (p[y * stride + x] + p[y * stride + x1]) * 0.5
                } else {
                    (p[y * stride + x]
                        + p[y * stride + x1]
                        + p[y1 * stride + x]
                        + p[y1 * stride + x1])
                        * 0.25
                });
            }
        }
    }
    out
}

pub(crate) fn compare_reference(
    reference: &PsychoImage,
    reference_mask: &crate::mask::PrecomputedMask,
    half: Option<(&PsychoImage, &crate::mask::PrecomputedMask)>,
    input: Input<'_>,
    params: &ButteraugliParams,
    pool: &BufferPool,
    stop: &dyn Stop,
) -> Result<crate::ButteraugliResult, ButteraugliError> {
    let w = reference.width();
    let h = reference.height();
    let candidate =
        prepare(input, w, h, params, pool, stop).map_err(ButteraugliError::Cancelled)?;
    let mut dm = compare_psycho(
        reference,
        &candidate,
        Some(reference_mask),
        params,
        pool,
        stop,
    )
    .map_err(ButteraugliError::Cancelled)?;
    candidate.recycle(pool);
    if let Some((half, mask)) = half {
        let sub = input.subsample(w, h);
        let candidate = prepare(
            Input::Interleaved(&sub),
            w.div_ceil(2),
            h.div_ceil(2),
            params,
            pool,
            stop,
        )
        .map_err(ButteraugliError::Cancelled)?;
        let sub = compare_psycho(half, &candidate, Some(mask), params, pool, stop)
            .map_err(ButteraugliError::Cancelled)?;
        merge(&sub, &mut dm);
        sub.recycle(pool);
        candidate.recycle(pool);
    }
    let (score, pnorm_3) = crate::diff::compute_score_from_diffmap(&dm);
    Ok(crate::ButteraugliResult {
        score,
        pnorm_3,
        diffmap: Some(dm.into_imgvec()),
    })
}

pub(crate) fn srgb_imgref(img: imgref::ImgRef<rgb::RGB8>, numerics: crate::Numerics) -> Vec<f32> {
    if numerics == crate::Numerics::Fast {
        return crate::diff::imgref_srgb_to_linear_f32(img);
    }
    let lut = srgb_lut(numerics);
    img.rows()
        .flat_map(|row| {
            row.iter()
                .flat_map(|p| [lut[p.r as usize], lut[p.g as usize], lut[p.b as usize]])
        })
        .collect()
}

#[cfg(test)]
mod tests;
