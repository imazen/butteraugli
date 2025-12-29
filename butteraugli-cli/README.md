# butteraugli-cli

Command-line tool for computing butteraugli perceptual image quality scores.

## Installation

```bash
cargo install butteraugli-cli
```

Or download pre-built binaries from [GitHub Releases](https://github.com/imazen/butteraugli/releases).

## Usage

```bash
# Compare two images
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

| Score | Quality | Description |
|-------|---------|-------------|
| 0.0 - 0.5 | Excellent | Imperceptible difference |
| 0.5 - 1.0 | Good | Barely noticeable |
| 1.0 - 2.0 | Acceptable | Noticeable but acceptable |
| 2.0 - 3.0 | Poor | Clearly visible difference |
| 3.0+ | Bad | Large, obvious difference |

## Exit Codes

- `0` - Success (score within threshold if `--max-score` specified)
- `1` - Score exceeded threshold
- `2` - Error (file not found, invalid image, etc.)

## Library

For programmatic use, see the [butteraugli](https://crates.io/crates/butteraugli) library crate.

## License

BSD-3-Clause
