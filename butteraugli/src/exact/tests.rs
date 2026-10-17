use super::*;
use crate::ButteraugliReference;

#[test]
fn exact_planar_ingresses_match_srgb_and_account_retained_storage() {
    let (w, h, stride) = (33, 35, 38);
    let a: Vec<_> = (0..w * h * 3)
        .map(|i| ((i * 71 + i / 13) % 256) as u8)
        .collect();
    let b: Vec<_> = a.iter().map(|v| v / 8 * 8).collect();
    let params = ButteraugliParams::new()
        .with_numerics(crate::Numerics::LibjxlExact)
        .with_compute_diffmap(true);
    let reference = ButteraugliReference::new(&a, w, h, params.clone()).unwrap();
    let mut ap = std::array::from_fn::<_, 3, _>(|_| vec![0.0; stride * h]);
    let mut bp = ap.clone();
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                ap[c][y * stride + x] =
                    srgb_lut(crate::Numerics::LibjxlExact)[a[(y * w + x) * 3 + c] as usize];
                bp[c][y * stride + x] =
                    srgb_lut(crate::Numerics::LibjxlExact)[b[(y * w + x) * 3 + c] as usize];
            }
        }
    }
    let planar = ButteraugliReference::new_linear_planar(
        &ap[0],
        &ap[1],
        &ap[2],
        w,
        h,
        stride,
        params.clone(),
    )
    .unwrap();
    assert_eq!(
        planar.precompute_bytes(),
        ButteraugliReference::estimated_reference_bytes(w, h, &params)
    );
    assert!(planar.memory_bytes() >= planar.precompute_bytes());
    let expected = reference.compare(&b).unwrap();
    let mut map = Vec::new();
    let (max, p3) = planar
        .compare_linear_planar_into(&bp[0], &bp[1], &bp[2], stride, &mut map)
        .unwrap();
    assert_eq!(
        (max.to_bits(), p3.to_bits()),
        (expected.score.to_bits(), expected.pnorm_3.to_bits())
    );
    assert_eq!(&map, expected.diffmap.as_ref().unwrap().buf());
    let (max, p3) = planar
        .compare_linear_planar_into(&ap[0], &ap[1], &ap[2], stride, &mut map)
        .unwrap();
    assert_eq!((max, p3), (0.0, 0.0));
    assert!(map.iter().all(|&v| v == 0.0));
    #[cfg(feature = "linear-planes")]
    {
        use crate::linear_planes::{LinearPlanes, Scorer};
        let a = LinearPlanes::with_stride(&ap[0], &ap[1], &ap[2], w, h, stride).unwrap();
        let b = LinearPlanes::with_stride(&bp[0], &bp[1], &bp[2], w, h, stride).unwrap();
        let scorer = Scorer::builder()
            .with_numerics(crate::Numerics::LibjxlExact)
            .with_diffmap(true)
            .build(&a)
            .unwrap();
        let scores = scorer.score_into(&b, &mut map).unwrap();
        assert_eq!(
            (scores.max_norm.to_bits(), scores.pnorm_3.to_bits()),
            (expected.score.to_bits(), expected.pnorm_3.to_bits())
        );
        assert_eq!(&map, expected.diffmap.as_ref().unwrap().buf());
        let zero = scorer.score_into(&a, &mut map).unwrap();
        assert_eq!((zero.max_norm, zero.pnorm_3), (0.0, 0.0));
        assert!(map.iter().all(|&v| v == 0.0));
    }
}

// Retain the pre-SIMD, transposing FIR as an independent layout/order oracle.
fn transposed_reference(
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

#[test]
fn blur_chunks_and_remainders_match_scalar_order() {
    let pool = BufferPool::new();
    for width in 1..=65 {
        for height in [3, 19] {
            let input = ImageF::from_vec(
                (0..width * height)
                    .map(|i| ((i * 71 + i / width * 19) % 257) as f32 / 257.0)
                    .collect(),
                width,
                height,
            );
            for sigma in [1.5, 2.7, 3.2, 7.2] {
                let k = kernel(sigma);
                let temp = transposed_reference(&input, &k, &pool, &enough::Unstoppable).unwrap();
                let expected =
                    transposed_reference(&temp, &k, &pool, &enough::Unstoppable).unwrap();
                let actual = blur(&input, sigma, &pool, &enough::Unstoppable).unwrap();
                for y in 0..height {
                    assert_eq!(
                        actual
                            .row(y)
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        expected
                            .row(y)
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        "{width}x{height}, sigma={sigma}, row={y}"
                    );
                }
            }
            if width < 10 {
                continue;
            }
            let k = kernel(1.2);
            let scale = 1.0 / k.iter().sum::<f32>();
            let w = [k[2] * scale, k[1] * scale, k[0] * scale];
            for vertical in [false, true] {
                let mut actual = ImageF::new(width, height);
                if vertical {
                    blur5_pass::<true>(
                        archmage::ScalarToken,
                        &input,
                        &mut actual,
                        w,
                        &enough::Unstoppable,
                    )
                    .unwrap();
                } else {
                    blur5_pass::<false>(
                        archmage::ScalarToken,
                        &input,
                        &mut actual,
                        w,
                        &enough::Unstoppable,
                    )
                    .unwrap();
                }
                for y in 0..height {
                    for x in 0..width {
                        let at = |d| {
                            if vertical {
                                input.row(mirror(y as isize + d, height))[x]
                            } else {
                                input.row(y)[mirror(x as isize + d, width)]
                            }
                        };
                        let expected = (at(-2) + at(2))
                            .mul_add(w[2], (at(-1) + at(1)).mul_add(w[1], at(0) * w[0]));
                        assert_eq!(
                            actual.row(y)[x].to_bits(),
                            expected.to_bits(),
                            "five-tap {width}x{height}, vertical={vertical}, ({x},{y})"
                        );
                    }
                }
            }
        }
    }
}
