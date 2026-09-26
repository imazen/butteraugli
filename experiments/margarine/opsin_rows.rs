//! Portable SIMD expression of the shared Butteraugli opsin row arithmetic.
//! Coefficients and operation order follow butteraugli/src/opsin.rs.
#![allow(clippy::excessive_precision)] // Preserve the shared source coefficient literals.
use crate::opsin;

// The pinned backend supplies eight-lane integer bitcasts at v3, not v4.
// AVX-512 hosts use that same AVX2 expression for this eight-lane kernel.
pub(crate) fn convert(a: &[f32], b: &[f32], intensity: f32, w: usize, out: &mut [f32]) {
    if w < 8 {
        reference(a, b, intensity, w, out);
    } else {
        archmage::incant!(
            convert_vector(a, b, intensity, w, out),
            [v3, neon, wasm128, scalar]
        );
    }
}

#[archmage::autoversion]
fn reference(
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
        let [s0, s1, s2] = [p0, p1, p2].map(|p| {
            let p = p.max(1e-4);
            (opsin::gamma(p) / p).max(1e-4)
        });
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

#[archmage::magetypes(define(f32x8, i32x8), v3, neon, wasm128, scalar)]
fn convert_vector(token: Token, a: &[f32], b: &[f32], intensity: f32, w: usize, out: &mut [f32]) {
    let splat = |v| f32x8::splat(token, v);
    let absorb = |r: f32x8, g: f32x8, b: f32x8| {
        [
            splat(0.299_565_503_400_583_19_f64 as f32) * r
                + splat(0.633_730_878_338_259_36_f64 as f32) * g
                + splat(0.077_705_617_820_981_968_f64 as f32) * b
                + splat(1.755_748_364_328_735_3_f64 as f32),
            splat(0.221_586_911_045_747_74_f64 as f32) * r
                + splat(0.693_913_880_441_161_42_f64 as f32) * g
                + splat(0.098_731_358_842_2_f64 as f32) * b
                + splat(1.755_748_364_328_735_3_f64 as f32),
            splat(0.02) * r
                + splat(0.02) * g
                + splat(0.204_801_290_410_261_29_f64 as f32) * b
                + splat(12.226_454_707_163_354_f64 as f32),
        ]
    };
    let gamma = |v: f32x8| {
        let biased = v.max(splat(0.0)) + splat(9.971_063_576_929_914_5);
        let bits = biased.bitcast_to_i32();
        let exponent = (bits - i32x8::splat(token, 0x3f2aaaab)).shr_arithmetic_const::<23>();
        let m = (bits - exponent.shl_const::<23>()).bitcast_to_f32() - splat(1.0);
        let numerator = (splat(7.4245873327820566E-01) * m + splat(1.4287160470083755)) * m
            + splat(-1.8503833400518310E-06);
        let denominator = (splat(1.7409343003366853E-01) * m + splat(1.0096718572241148)) * m
            + splat(9.9032814277590719E-01);
        let log = numerator / denominator + exponent.to_f32();
        splat(19.245_013_259_874_995 * (1.0 / std::f32::consts::LOG2_E)) * log
            + splat(-23.160_462_398_057_55)
    };
    let minimum = [
        1.755_748_364_328_735_3,
        1.755_748_364_328_735_3,
        12.226_454_707_163_354,
    ]
    .map(splat);
    for block in 0..w.div_ceil(8) {
        let x = (block * 8).min(w - 8);
        let load = |data: &[f32], channel| {
            f32x8::load(
                token,
                data[channel * w + x..channel * w + x + 8]
                    .try_into()
                    .unwrap(),
            ) * splat(intensity)
        };
        let p = absorb(load(b, 0), load(b, 1), load(b, 2));
        let sensitivity: [f32x8; 3] = std::array::from_fn(|c| {
            let p = p[c].max(minimum[c]).max(splat(1e-4));
            (gamma(p) / p).max(splat(1e-4))
        });
        let v = absorb(load(a, 0), load(a, 1), load(a, 2));
        let [v0, v1, v2] = std::array::from_fn(|c| (v[c] * sensitivity[c]).max(minimum[c]));
        for (c, value) in [v0 - v1, v0 + v1, v2].into_iter().enumerate() {
            value.store((&mut out[c * w + x..c * w + x + 8]).try_into().unwrap());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn vector_rows_preserve_shared_opsin_arithmetic_at_tails_and_intensities() {
        for w in (1..34).chain([129, 512, 634]) {
            let a: Vec<_> = (0..3 * w)
                .map(|i| ((i * 137 + i * i * 19) % 65536) as f32 / 65535.0)
                .collect();
            let b: Vec<_> = a
                .iter()
                .enumerate()
                .map(|(i, &v)| v * 0.91 + (i % 3) as f32 * 0.017)
                .collect();
            for intensity in [0.0, 1.0, 80.0, 1000.0] {
                let mut expected = vec![0.0; 3 * w];
                let mut actual = expected.clone();
                reference(&a, &b, intensity, w, &mut expected);
                convert(&a, &b, intensity, w, &mut actual);
                assert_eq!(actual, expected, "width={w}, intensity={intensity}");
            }
        }
    }
}
