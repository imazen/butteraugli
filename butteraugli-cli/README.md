# butteraugli-cli

Command-line tool for computing butteraugli perceptual image quality scores.

## Installation

```bash
cargo install butteraugli-cli
```

Or download pre-built binaries from [GitHub Releases](https://github.com/imazen/butteraugli/releases).

## Usage

**Argument order is `REFERENCE` then `DISTORTED`** — the first path is the
original/source image, the second is the compressed/modified one. The metric is
not symmetric (`hf_asymmetry` penalizes the distorted side differently), so
swapping them changes the score.

Both images must have **identical width and height**; mismatched dimensions are
a hard error (exit code `2`). Single-comparison inputs are decoded by the
[`image`](https://crates.io/crates/image) crate with the **PNG and JPEG**
features enabled, so `.png`, `.jpg`, and `.jpeg` are accepted. (Batch mode's
`--extensions` default also lists `webp,gif,bmp` for filename matching, but only
formats the build can actually decode will load.)

```bash
# Compare two images: REFERENCE first, DISTORTED second
butteraugli original.png compressed.jpg

# Show quality rating with colors
butteraugli -q original.png compressed.jpg

# CI mode - exit code 1 if score exceeds threshold
butteraugli --max-score 1.5 original.png compressed.jpg

# Compare all images in two directories
butteraugli --batch dir1/ dir2/

# Output JSON for scripting
butteraugli --json original.png compressed.jpg

# Save difference heatmap
butteraugli --diffmap diff.png original.png compressed.jpg
```

## Score Interpretation

The number printed as `Butteraugli score:` is a **perceptual distance**:

- **`0.0` means the images are identical.** Lower is better.
- The score is **unbounded above** — there is no maximum. `3.0+` in the table
  below is just where differences become "large and obvious"; pathological
  inputs can score far higher.
- It is the **max-norm (p = ∞) aggregation** of butteraugli's per-pixel
  difference map — i.e. the score is driven by the *single worst* region, not
  the average. This is the historical butteraugli "Distance" / libjxl's
  `ButteraugliScoreFromDiffmap`. For an average-style aggregation that is often
  more useful for rate-distortion sweeps, pass `--pnorm` to additionally print
  the libjxl **3-norm** (the average of p-norms at p = 3, 6, 12).

The quality bands are rules of thumb, not hard cutoffs:

| Score | Quality | Description |
|-------|---------|-------------|
| 0.0 - 0.5 | Excellent | Imperceptible difference |
| 0.5 - 1.0 | Good | Barely noticeable |
| 1.0 - 2.0 | Acceptable | Noticeable but acceptable |
| 2.0 - 3.0 | Poor | Clearly visible difference |
| 3.0+ | Bad | Large, obvious difference (no upper bound) |

### Viewing conditions and comparability across resolutions

Butteraugli models **absolute luminance** under an assumed viewing condition,
so the score depends on how bright the display is assumed to be. The
`--intensity-target` flag sets that luminance in nits (cd/m², default `80`, the
SDR convention); lower values make the metric more sensitive to differences in
dark regions, and HDR content should use `250` or higher with input scaled so
`1.0` maps to the mastering/display peak.

Because butteraugli's frequency model assumes a fixed pixels-per-degree, **the
same compression artifact scores differently at different image sizes**, and
scores are **not directly comparable across images of different resolutions**.
Compare like-for-like: same dimensions, same `--intensity-target`. (There is no
pixels-per-degree flag — only `--intensity-target`.)

## Exit Codes

- `0` - Success (score within threshold if `--max-score` specified)
- `1` - Score exceeded threshold
- `2` - Error (file not found, invalid image, etc.)

## Library

For programmatic use, see the [butteraugli](https://crates.io/crates/butteraugli) library crate.

## License

BSD-3-Clause
