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

    pub(crate) fn linear_strip(&self, start: usize, end: usize) -> Vec<f32> {
        assert!(start <= end && end <= self.height);
        let mut result = Vec::with_capacity((end - start) * self.width * 3);
        for y in start..end {
            let range = y * self.stride..y * self.stride + self.width * self.channels;
            match self.samples {
                Samples::U8(v) => {
                    for p in v[range].chunks_exact(self.channels) {
                        result.extend(
                            p[..3]
                                .iter()
                                .copied()
                                .map(butteraugli::opsin::srgb_to_linear),
                        );
                    }
                }
                Samples::U16(v) => {
                    for p in v[range].chunks_exact(self.channels) {
                        result.extend(p[..3].iter().copied().map(linear16));
                    }
                }
            }
        }
        result
    }
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
