//! Shared encoded-sRGB ingress for the experiment, not a general CMS.
//! Metadata policy is audited by the corpus runner. Preserve RGB16 samples;
//! opaque alpha is accepted, non-opaque alpha requires an explicit background.
use image_io::{DynamicImage, ImageReader};
use std::{error::Error, path::Path};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn linear16(value: u16) -> f32 {
    let s = f64::from(value) / 65535.0;
    (if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }) as f32
}

pub(crate) fn convert(input: DynamicImage) -> Result<(Vec<f32>, usize, usize)> {
    let (w, h) = (input.width() as usize, input.height() as usize);
    let linear = match input {
        DynamicImage::ImageRgb8(rgb) => rgb
            .as_raw()
            .iter()
            .map(|&v| butteraugli::opsin::srgb_to_linear(v))
            .collect(),
        DynamicImage::ImageRgba8(rgba) => {
            if rgba.pixels().any(|p| p[3] != 255) {
                return Err("non-opaque alpha requires a background".into());
            }
            rgba.pixels()
                .flat_map(|p| [p[0], p[1], p[2]].map(butteraugli::opsin::srgb_to_linear))
                .collect()
        }
        DynamicImage::ImageRgb16(rgb) => rgb.as_raw().iter().copied().map(linear16).collect(),
        DynamicImage::ImageRgba16(rgba) => {
            if rgba.pixels().any(|p| p[3] != 65535) {
                return Err("non-opaque alpha requires a background".into());
            }
            rgba.pixels()
                .flat_map(|p| [linear16(p[0]), linear16(p[1]), linear16(p[2])])
                .collect()
        }
        _ => return Err("encoded-sRGB ingress requires RGB/RGBA 8 or 16-bit".into()),
    };
    Ok((linear, w, h))
}

pub(crate) fn load(path: impl AsRef<Path>) -> Result<(Vec<f32>, usize, usize)> {
    convert(ImageReader::open(path)?.decode()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image_io::{ImageBuffer, Rgb, Rgba};

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
