# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2025-12-25

### Added
- Pixel layout diagram in README for clearer input format documentation

## [0.2.0] - 2025-12-25

### Added
- `compute_butteraugli_linear` function for linear RGB f32 input (HDR/16-bit workflows)
- `srgb_to_linear` helper function for manual color space conversion
- Comprehensive documentation of XYB color space differences vs jpegli
- Input requirements documentation (minimum size, channel layout)

### Changed
- Improved API idiomaticity
- Better documentation of C++ API differences

## [0.1.0] - 2025-12-24

### Added
- Initial release
- Pure Rust implementation of butteraugli perceptual quality metric
- `compute_butteraugli` function for sRGB u8 input
- Multi-resolution analysis (SubSample2x)
- Malta convolution filter
- MaskY, MaskDcY, MaskPsychoImage, CombineChannelsToDiffmap
- SIMD optimizations via `wide` crate
- 195 synthetic test cases validated against C++ libjxl
- `ButteraugliParams` for customization (hf_asymmetry, xmul, intensity_target)
- `score_to_quality` and `butteraugli_fuzzy_class` helper functions
- Optional diffmap output for per-pixel analysis

### Technical Details
- Accurate port of FastLog2f rational polynomial from C++
- Correct Y-channel MaximumClamp order in frequency separation
- Matched C++ blur and multiresolution behavior
