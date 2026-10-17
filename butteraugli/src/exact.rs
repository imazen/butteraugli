//! libjxl v0.12 arithmetic, isolated from the default optimized kernels.
//!
//! Operation order follows a7a9c787's butteraugli.cc, fast_math-inl.h and
//! cms/transfer_functions-inl.h. Explicit mul_add gives fused rounding on
//! machines with and without hardware FMA.
#![allow(clippy::excessive_precision)]
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
    for y in 0..height {
        if y % 64 == 0 {
            stop.check()?;
        }
        let row = input.row(y);
        for x in 0..width {
            let at = |dx| row[mirror(x as isize + dx, width)];
            temp.row_mut(y)[x] =
                (at(-2) + at(2)).mul_add(w[2], (at(-1) + at(1)).mul_add(w[1], at(0) * w[0]));
        }
    }
    let mut out = ImageF::from_pool_dirty(width, height, pool);
    for y in 0..height {
        if y % 64 == 0 {
            stop.check()?;
        }
        for x in 0..width {
            let at = |dy| temp.row(mirror(y as isize + dy, height))[x];
            out.row_mut(y)[x] =
                (at(-2) + at(2)).mul_add(w[2], (at(-1) + at(1)).mul_add(w[1], at(0) * w[0]));
        }
    }
    temp.recycle(pool);
    Ok(out)
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
    fn pass(
        input: &ImageF,
        k: &[f32],
        pool: &BufferPool,
        stop: &dyn Stop,
    ) -> Result<ImageF, enough::StopReason> {
        let n = k.len();
        let radius = n / 2;
        let scale = 1.0 / k.iter().sum::<f32>();
        let sk: Vec<_> = k.iter().map(|&v| v * scale).collect();
        let mut out = ImageF::from_pool_dirty(input.height(), input.width(), pool);
        for y in 0..input.height() {
            if y % 64 == 0 {
                stop.check()?;
            }
            let row = input.row(y);
            for x in 0..input.width() {
                let v = if x < radius || x + radius >= input.width() {
                    let lo = x.saturating_sub(radius);
                    let hi = (x + radius).min(input.width() - 1);
                    let mut z = 0.0_f32;
                    let mut v = 0.0_f32;
                    for j in lo..=hi {
                        z += k[j + radius - x];
                        v += row[j] * k[j + radius - x];
                    }
                    v * (1.0 / z)
                } else {
                    let r = &row[x - radius..=x + radius];
                    if n == 7 {
                        r[3] * sk[3]
                            + (r[0] + r[6]) * sk[0]
                            + (r[1] + r[5]) * sk[1]
                            + (r[2] + r[4]) * sk[2]
                    } else {
                        let mut sums = [0.0_f32; 4];
                        for i in 0..radius {
                            let t = (r[i] + r[n - 1 - i]) * sk[i];
                            if i < 4 {
                                sums[i] = t;
                            } else {
                                sums[i % 4] += t;
                            }
                        }
                        r[radius] * sk[radius] + sums[0] + sums[1] + sums[2] + sums[3]
                    }
                };
                out.row_mut(x)[y] = v;
            }
        }
        Ok(out)
    }
    let temp = pass(input, &k, pool, stop)?;
    let out = pass(&temp, &k, pool, stop)?;
    temp.recycle(pool);
    Ok(out)
}

fn rational(x: f32, p: &[f32], q: &[f32]) -> f32 {
    fn horner(x: f32, c: &[f32]) -> f32 {
        let mut y = c[c.len() - 1];
        for &v in c[..c.len() - 1].iter().rev() {
            y = y.mul_add(x, v);
        }
        y
    }
    horner(x, p) / horner(x, q)
}

pub(crate) fn srgb(v: u8) -> f32 {
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

pub(crate) fn prepare(
    rgb: &[f32],
    w: usize,
    h: usize,
    params: &ButteraugliParams,
    pool: &BufferPool,
    stop: &dyn Stop,
) -> Result<PsychoImage, enough::StopReason> {
    let mut image = Image3F::from_pool_dirty(w, h, pool);
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                image.plane_mut(c).row_mut(y)[x] = rgb[(y * w + x) * 3 + c];
            }
        }
    }
    let blurred = [
        blur(image.plane(0), 1.2, pool, stop)?,
        blur(image.plane(1), 1.2, pool, stop)?,
        blur(image.plane(2), 1.2, pool, stop)?,
    ];
    let mut xyb = Image3F::from_pool_dirty(w, h, pool);
    let intensity = params.intensity_target();
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
    image.recycle(pool);
    for b in blurred {
        b.recycle(pool);
    }
    crate::psycho::separate_frequencies_owned_numerics::<true>(xyb, pool, stop)
}

pub(crate) fn compare_psycho(
    a: &PsychoImage,
    b: &PsychoImage,
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
    let mask = crate::mask::compute_mask_from_hf_uhf_stop_numerics::<true>(
        &a.hf,
        &a.uhf,
        &b.hf,
        &b.uhf,
        Some(ac.plane_mut(1)),
        pool,
        stop,
    )?;
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
    mask.recycle(pool);
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
        || prepare(a, w, h, params, &pool, stop),
        || prepare(b, w, h, params, &pool, stop),
    );
    compare_psycho(&a?, &b?, params, &pool, stop)
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
    if params.single_resolution() || w < 15 || h < 15 {
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

pub(crate) fn interleave(
    red: &[f32],
    green: &[f32],
    blue: &[f32],
    width: usize,
    height: usize,
    stride: usize,
) -> Vec<f32> {
    let mut rgb = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let index = y * stride + x;
            rgb.extend_from_slice(&[red[index], green[index], blue[index]]);
        }
    }
    rgb
}

pub(crate) fn compare_reference(
    reference: &PsychoImage,
    half: Option<&PsychoImage>,
    rgb: &[f32],
    params: &ButteraugliParams,
    pool: &BufferPool,
    stop: &dyn Stop,
) -> Result<crate::ButteraugliResult, ButteraugliError> {
    let w = reference.width();
    let h = reference.height();
    let candidate = prepare(rgb, w, h, params, pool, stop).map_err(ButteraugliError::Cancelled)?;
    let mut dm = compare_psycho(reference, &candidate, params, pool, stop)
        .map_err(ButteraugliError::Cancelled)?;
    candidate.recycle(pool);
    if let Some(half) = half {
        let (sub, sw, sh) = crate::diff::subsample_linear_rgb_2x(rgb, w, h);
        let candidate =
            prepare(&sub, sw, sh, params, pool, stop).map_err(ButteraugliError::Cancelled)?;
        let sub = compare_psycho(half, &candidate, params, pool, stop)
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

pub(crate) fn srgb_imgref(img: imgref::ImgRef<rgb::RGB8>) -> Vec<f32> {
    let lut: [f32; 256] = std::array::from_fn(|i| srgb(i as u8));
    img.rows()
        .flat_map(|row| {
            row.iter()
                .flat_map(|p| [lut[p.r as usize], lut[p.g as usize], lut[p.b as usize]])
        })
        .collect()
}
