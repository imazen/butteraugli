//! Shared encoded-sRGB ingress for the experiment, not a general CMS.
//! Metadata policy is audited by the corpus runner. Preserve RGB16 samples;
//! opaque alpha is accepted, non-opaque alpha requires an explicit background.
use image_io::{DynamicImage, ImageReader};
use std::{error::Error, path::Path};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub(crate) enum Samples<'a> {
    U8(&'a [u8]),
    U16(&'a [u16]),
}

/// Borrowed encoded-sRGB rows; stride is measured in channel samples.
pub(crate) struct EncodedRows<'a> {
    samples: Samples<'a>,
    pub(crate) width: usize,
    pub(crate) height: usize,
    stride: usize,
    channels: usize,
}

impl<'a> EncodedRows<'a> {
    pub(crate) fn new(
        samples: Samples<'a>,
        width: usize,
        height: usize,
        stride: usize,
        channels: usize,
    ) -> Result<Self> {
        if width == 0 || height == 0 || !matches!(channels, 3 | 4) {
            return Err("invalid encoded row geometry".into());
        }
        let row = width.checked_mul(channels).ok_or("encoded row overflow")?;
        let needed = (height - 1)
            .checked_mul(stride)
            .and_then(|n| n.checked_add(row))
            .ok_or("encoded rows overflow")?;
        let length = match samples {
            Samples::U8(v) => v.len(),
            Samples::U16(v) => v.len(),
        };
        if stride < row || length < needed {
            return Err("encoded rows too short".into());
        }
        if channels == 4 {
            for y in 0..height {
                for x in 0..width {
                    let i = y * stride + 4 * x + 3;
                    let opaque = match samples {
                        Samples::U8(v) => v[i] == 255,
                        Samples::U16(v) => v[i] == 65535,
                    };
                    if !opaque {
                        return Err("non-opaque alpha requires a background".into());
                    }
                }
            }
        }
        Ok(Self {
            samples,
            width,
            height,
            stride,
            channels,
        })
    }

    /// Borrow a column interval without changing row stride or sample precision.
    /// The parent has already validated geometry and opaque alpha.
    #[allow(
        dead_code,
        reason = "shared ingress also compiles in the teacher binary"
    )]
    pub(crate) fn columns(&self, start: usize, end: usize) -> EncodedRows<'_> {
        assert!(start < end && end <= self.width);
        let offset = start * self.channels;
        let samples = match self.samples {
            Samples::U8(v) => Samples::U8(&v[offset..]),
            Samples::U16(v) => Samples::U16(&v[offset..]),
        };
        EncodedRows {
            samples,
            width: end - start,
            height: self.height,
            stride: self.stride,
            channels: self.channels,
        }
    }

    pub(crate) fn from_image(image: &'a DynamicImage) -> Result<Self> {
        let (w, h) = (image.width() as usize, image.height() as usize);
        match image {
            DynamicImage::ImageRgb8(v) => Self::new(Samples::U8(v.as_raw()), w, h, w * 3, 3),
            DynamicImage::ImageRgba8(v) => Self::new(Samples::U8(v.as_raw()), w, h, w * 4, 4),
            DynamicImage::ImageRgb16(v) => Self::new(Samples::U16(v.as_raw()), w, h, w * 3, 3),
            DynamicImage::ImageRgba16(v) => Self::new(Samples::U16(v.as_raw()), w, h, w * 4, 4),
            _ => Err("encoded-sRGB ingress requires RGB/RGBA 8 or 16-bit".into()),
        }
    }

    /// Encoded RGB sample for reference-only region selection. No quantization.
    #[allow(
        dead_code,
        reason = "shared ingress also compiles in the teacher binary"
    )]
    pub(crate) fn encoded_rgb(&self, x: usize, y: usize) -> [f32; 3] {
        assert!(x < self.width && y < self.height);
        let i = y * self.stride + x * self.channels;
        match self.samples {
            Samples::U8(v) => std::array::from_fn(|c| f32::from(v[i + c]) / 255.0),
            Samples::U16(v) => std::array::from_fn(|c| f32::from(v[i + c]) / 65535.0),
        }
    }

    pub(crate) fn linear_strip(&self, start: usize, end: usize) -> Vec<f32> {
        self.linear_region(0, self.width, start, end)
    }

    /// Native-precision teacher input without an intermediate full-image copy.
    pub(crate) fn linear_rgb(&self) -> Vec<butteraugli::RGB<f32>> {
        let mut out = Vec::with_capacity(self.width * self.height);
        for y in 0..self.height {
            let range = y * self.stride..y * self.stride + self.width * self.channels;
            match self.samples {
                Samples::U8(v) => {
                    let lut = linear8_table();
                    out.extend(v[range].chunks_exact(self.channels).map(|p| {
                        butteraugli::RGB::new(
                            lut[p[0] as usize],
                            lut[p[1] as usize],
                            lut[p[2] as usize],
                        )
                    }));
                }
                Samples::U16(v) => {
                    out.extend(v[range].chunks_exact(self.channels).map(|p| {
                        butteraugli::RGB::new(linear16(p[0]), linear16(p[1]), linear16(p[2]))
                    }));
                }
            }
        }
        out
    }

    pub(crate) fn linear_region(&self, x0: usize, x1: usize, start: usize, end: usize) -> Vec<f32> {
        assert!(start <= end && end <= self.height && x0 < x1 && x1 <= self.width);
        let mut result = vec![0.0; (end - start) * (x1 - x0) * 3];
        for (y, out) in (start..end).zip(result.chunks_exact_mut((x1 - x0) * 3)) {
            let range = y * self.stride + x0 * self.channels..y * self.stride + x1 * self.channels;
            match self.samples {
                Samples::U8(v) => {
                    let lut = linear8_table();
                    for (p, dst) in v[range]
                        .chunks_exact(self.channels)
                        .zip(out.as_chunks_mut::<3>().0.iter_mut())
                    {
                        dst.copy_from_slice(&[
                            lut[p[0] as usize],
                            lut[p[1] as usize],
                            lut[p[2] as usize],
                        ]);
                    }
                }
                Samples::U16(v) => {
                    for (p, dst) in v[range]
                        .chunks_exact(self.channels)
                        .zip(out.as_chunks_mut::<3>().0.iter_mut())
                    {
                        dst.copy_from_slice(&[linear16(p[0]), linear16(p[1]), linear16(p[2])]);
                    }
                }
            }
        }
        result
    }

    /// Write one full-width planar row, optionally averaging a native 2x2 cell.
    /// The destination slices are logical rows; caller-owned row padding is untouched.
    #[cfg(feature = "planar")]
    #[allow(dead_code, reason = "shared with the teacher binary")]
    pub(crate) fn linear_planar_row(&self, y: usize, factor: usize, out: [&mut [f32]; 3]) {
        self.linear_planar_region_row(0, self.width, y, factor, out);
    }

    #[cfg(feature = "planar")]
    pub(crate) fn linear_planar_region_row(
        &self,
        x0: usize,
        x1: usize,
        y: usize,
        factor: usize,
        out: [&mut [f32]; 3],
    ) {
        assert!(matches!(factor, 1 | 2) && y < self.height);
        assert!(x0 < x1 && x1 <= self.width);
        let width = x1 - x0;
        assert!(out.iter().all(|row| row.len() == width.div_ceil(factor)));
        let lut = linear8_table();
        match (&self.samples, self.channels) {
            (Samples::U8(v), 3) => planar_row::<_, 3>(
                &v[x0 * self.channels..],
                width,
                self.height,
                self.stride,
                y,
                factor,
                out,
                |v| lut[v as usize],
            ),
            (Samples::U8(v), 4) => planar_row::<_, 4>(
                &v[x0 * self.channels..],
                width,
                self.height,
                self.stride,
                y,
                factor,
                out,
                |v| lut[v as usize],
            ),
            (Samples::U16(v), 3) => planar_row::<_, 3>(
                &v[x0 * self.channels..],
                width,
                self.height,
                self.stride,
                y,
                factor,
                out,
                linear16,
            ),
            (Samples::U16(v), 4) => planar_row::<_, 4>(
                &v[x0 * self.channels..],
                width,
                self.height,
                self.stride,
                y,
                factor,
                out,
                linear16,
            ),
            _ => unreachable!(),
        }
    }
}

#[cfg(feature = "planar")]
#[allow(clippy::too_many_arguments)]
fn planar_row<T: Copy, const C: usize>(
    input: &[T],
    width: usize,
    height: usize,
    stride: usize,
    y: usize,
    factor: usize,
    out: [&mut [f32]; 3],
    linear: impl Fn(T) -> f32,
) {
    let [r, g, b] = out;
    let a = &input[y * stride..y * stride + width * C];
    if factor == 1 {
        for (((p, r), g), b) in a.as_chunks::<C>().0.iter().zip(r).zip(g).zip(b) {
            *r = linear(p[0]);
            *g = linear(p[1]);
            *b = linear(p[2]);
        }
        return;
    }
    let second = (y + 1 < height).then(|| &input[(y + 1) * stride..(y + 1) * stride + width * C]);
    for (x, ((r, g), b)) in r.iter_mut().zip(g).zip(b).enumerate() {
        let x0 = x * 2 * C;
        let p0: &[T; C] = a[x0..x0 + C].try_into().unwrap();
        let p1 = (x * 2 + 1 < width).then(|| <&[T; C]>::try_from(&a[x0 + C..x0 + 2 * C]).unwrap());
        let (p2, p3) = if let Some(second) = second {
            let p2: &[T; C] = second[x0..x0 + C].try_into().unwrap();
            let p3 = p1.map(|_| <&[T; C]>::try_from(&second[x0 + C..x0 + 2 * C]).unwrap());
            (Some(p2), p3)
        } else {
            (None, None)
        };
        for (c, dst) in [r, g, b].into_iter().enumerate() {
            // Match subsample_linear_rgb_2x's exact addition order and edge count.
            *dst = match (p1, p2, p3) {
                (Some(p1), Some(p2), Some(p3)) => {
                    (linear(p0[c]) + linear(p1[c]) + linear(p2[c]) + linear(p3[c])) * 0.25
                }
                (Some(p1), None, None) => (linear(p0[c]) + linear(p1[c])) * 0.5,
                (None, Some(p2), None) => (linear(p0[c]) + linear(p2[c])) * 0.5,
                (None, None, None) => linear(p0[c]),
                _ => unreachable!(),
            };
        }
    }
}

fn linear8_table() -> &'static [f32; 256] {
    static LUT: std::sync::LazyLock<[f32; 256]> = std::sync::LazyLock::new(|| {
        std::array::from_fn(|v| butteraugli::opsin::srgb_to_linear(v as u8))
    });
    &LUT
}

pub(crate) fn decode(path: impl AsRef<Path>) -> Result<DynamicImage> {
    Ok(ImageReader::open(path)?.with_guessed_format()?.decode()?)
}

fn linear16(value: u16) -> f32 {
    let s = f64::from(value) / 65535.0;
    (if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }) as f32
}

pub(crate) fn convert(input: DynamicImage) -> Result<(Vec<f32>, usize, usize)> {
    let view = EncodedRows::from_image(&input)?;
    Ok((view.linear_strip(0, view.height), view.width, view.height))
}

pub(crate) fn load(path: impl AsRef<Path>) -> Result<(Vec<f32>, usize, usize)> {
    convert(decode(path)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image_io::{ImageBuffer, Rgb, Rgba};

    #[cfg(feature = "planar")]
    #[test]
    fn planar_rows_preserve_native_samples_and_odd_cell_arithmetic() {
        for (w, h) in [(1, 1), (2, 2), (3, 7), (31, 6), (32, 7), (33, 7)] {
            for channels in [3, 4] {
                let stride = w * channels + 5;
                let mut a = vec![0u8; stride * h];
                let mut b = vec![0u16; stride * h];
                for y in 0..h {
                    for x in 0..w {
                        for c in 0..channels {
                            let i = y * stride + x * channels + c;
                            a[i] = if c == 3 { 255 } else { ((i * 79) % 256) as u8 };
                            b[i] = if c == 3 {
                                65535
                            } else {
                                ((i * 313) % 65536) as u16
                            };
                        }
                    }
                }
                for samples in [Samples::U8(&a), Samples::U16(&b)] {
                    let view = EncodedRows::new(samples, w, h, stride, channels).unwrap();
                    let packed = view.linear_strip(0, h);
                    let rgb = view.linear_rgb();
                    assert_eq!(rgb.len(), w * h);
                    for (p, v) in rgb.iter().zip(packed.chunks_exact(3)) {
                        assert_eq!([p.r, p.g, p.b], v);
                    }
                    for factor in [1, 2] {
                        for y in (0..h).step_by(factor) {
                            let width = w.div_ceil(factor);
                            let mut planes =
                                std::array::from_fn::<_, 3, _>(|_| vec![f32::NAN; width + 3]);
                            let [r, g, b] = &mut planes;
                            view.linear_planar_row(
                                y,
                                factor,
                                [&mut r[..width], &mut g[..width], &mut b[..width]],
                            );
                            for (c, plane) in planes.iter().enumerate() {
                                for (x, &actual) in plane[..width].iter().enumerate() {
                                    let mut sum = 0.0;
                                    let mut count = 0;
                                    for yy in y..(y + factor).min(h) {
                                        for xx in x * factor..((x + 1) * factor).min(w) {
                                            sum += packed[(yy * w + xx) * 3 + c];
                                            count += 1;
                                        }
                                    }
                                    assert_eq!(actual.to_bits(), (sum / count as f32).to_bits());
                                }
                                assert!(plane[width..].iter().all(|v| v.is_nan()));
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn bmp_ingress_preserves_bgr_channels_row_padding_and_orientation() {
        let mut bytes = vec![0u8; 70];
        bytes[0..2].copy_from_slice(b"BM");
        bytes[2..6].copy_from_slice(&70u32.to_le_bytes());
        bytes[10..14].copy_from_slice(&54u32.to_le_bytes());
        bytes[14..18].copy_from_slice(&40u32.to_le_bytes());
        bytes[18..22].copy_from_slice(&2i32.to_le_bytes());
        bytes[22..26].copy_from_slice(&2i32.to_le_bytes());
        bytes[26..28].copy_from_slice(&1u16.to_le_bytes());
        bytes[28..30].copy_from_slice(&24u16.to_le_bytes());
        bytes[54..70].copy_from_slice(&[
            255, 0, 0, 255, 255, 255, 99, 99, 0, 0, 255, 0, 255, 0, 99, 99,
        ]);
        let decoded = ImageReader::new(std::io::Cursor::new(bytes))
            .with_guessed_format()
            .unwrap()
            .decode()
            .unwrap();
        let DynamicImage::ImageRgb8(ref rgb) = decoded else {
            panic!("expected RGB8 BMP")
        };
        assert_eq!(
            rgb.as_raw(),
            &[255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255]
        );
        let view = EncodedRows::from_image(&decoded).unwrap();
        assert_eq!(
            view.linear_strip(0, 2),
            [255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255]
                .map(butteraugli::opsin::srgb_to_linear)
        );
    }

    #[test]
    fn strided_rgb8_excludes_padding_and_rejects_bad_geometry() {
        let data = [10, 20, 30, 1, 2, 3, 99, 99, 40, 50, 60, 4, 5, 6];
        let rows = EncodedRows::new(Samples::U8(&data), 2, 2, 8, 3).unwrap();
        let expected: Vec<_> = [10, 20, 30, 1, 2, 3, 40, 50, 60, 4, 5, 6]
            .into_iter()
            .map(butteraugli::opsin::srgb_to_linear)
            .collect();
        assert_eq!(rows.linear_strip(0, 2), expected);
        assert!(EncodedRows::new(Samples::U8(&data), usize::MAX, 2, 8, 3).is_err());
        assert!(EncodedRows::new(Samples::U8(&data[..13]), 2, 2, 8, 3).is_err());
        assert!(EncodedRows::new(Samples::U8(&[1, 2, 3, 254]), 1, 1, 4, 4).is_err());
    }

    #[test]
    fn rgb16_retains_low_bits() {
        let image =
            ImageBuffer::<Rgb<u16>, _>::from_raw(2, 1, vec![32768, 32769, 32770, 0, 65535, 1])
                .unwrap();
        let (pixels, w, h) = convert(DynamicImage::ImageRgb16(image)).unwrap();
        assert_eq!((w, h), (2, 1));
        assert!(pixels[0] < pixels[1] && pixels[1] < pixels[2]);
        assert_eq!(pixels[3], 0.0);
        assert_eq!(pixels[4], 1.0);
        assert!(pixels[5] > 0.0);
    }

    #[test]
    fn opaque_alpha_matches_rgb_and_transparency_fails() {
        let rgb = ImageBuffer::<Rgb<u8>, _>::from_raw(1, 1, vec![12, 128, 255]).unwrap();
        let rgba = ImageBuffer::<Rgba<u8>, _>::from_raw(1, 1, vec![12, 128, 255, 255]).unwrap();
        assert_eq!(
            convert(DynamicImage::ImageRgb8(rgb)).unwrap(),
            convert(DynamicImage::ImageRgba8(rgba.clone())).unwrap()
        );
        let mut transparent = rgba;
        transparent[(0, 0)][3] = 254;
        assert!(convert(DynamicImage::ImageRgba8(transparent)).is_err());
    }
}
