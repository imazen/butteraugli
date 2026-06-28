<!-- GENERATED FROM README.md by zenutils gen-readme-crates.sh — DO NOT EDIT. -->

# butteraugli

A pure-Rust port of **butteraugli**, the perceptual image-difference metric from
Google's [libjxl](https://github.com/libjxl/libjxl). It models human vision —
opsin dynamics, an opponent XYB color space, visual masking, and multi-scale
frequency analysis — to estimate how different two images *look*, where
pixel-wise metrics like PSNR and MSE do not. Scores are validated against
libjxl's `butteraugli_main` to within FMA rounding noise. No C dependencies;
runtime SIMD dispatch (AVX-512 / AVX2 / SSE4.2 / NEON / WASM) via
[`archmage`](https://crates.io/crates/archmage); safe Rust by default
(`unsafe-performance` is opt-in).

## Quick start

```toml
[dependencies]
butteraugli = "0.9.4"
```

```rust
use butteraugli::{butteraugli, ButteraugliParams, Img, RGB8};

// Two equally-sized images as packed RGB8 pixels (sRGB, gamma-encoded).
let reference: Vec<RGB8> = decode_reference();
let distorted: Vec<RGB8> = decode_distorted();

let r = Img::new(reference, width, height);
let d = Img::new(distorted, width, height);

let result = butteraugli(r.as_ref(), d.as_ref(), &ButteraugliParams::default())?;

println!("butteraugli score: {:.4}", result.score); // < 1.0 good, > 2.0 visible
```

`Img`, `ImgRef`, `ImgVec`, `RGB`, and `RGB8` are re-exported from the crate
root, so you don't need to add `imgref` or `rgb` to your own `Cargo.toml`. If
your decoder hands you a flat `Vec<u8>`, see
[From a flat `Vec<u8>`](#from-a-flat-vecu8-of-rgb-bytes) below.

## What is butteraugli?

Butteraugli estimates the perceived difference between two images using a model
of human vision. Unlike simple pixel-wise metrics (PSNR, MSE), it accounts for:

- **Opsin dynamics**: photosensitive chemical responses in the retina
- **XYB color space**: a hybrid opponent / trichromatic representation
- **Visual masking**: how image features hide or reveal differences
- **Multi-scale analysis**: UHF, HF, MF, and LF frequency bands

### Quality thresholds

| Score | Interpretation |
|-------|----------------|
| < 1.0 | Images appear identical to most viewers |
| 1.0 – 2.0 | Subtle differences may be noticeable |
| > 2.0 | Visible differences between images |

The score is the **max-norm** (worst-region, p = ∞) distance: `0.0` for
identical input, lower is better, unbounded above. Because it models absolute
luminance and a fixed pixels-per-degree, scores are **not** comparable across
different image resolutions. The public constants `BUTTERAUGLI_GOOD` (1.0) and
`BUTTERAUGLI_BAD` (2.0) name the two thresholds above.

## Command-line tool

The [`butteraugli-cli`](https://github.com/imazen/butteraugli/tree/main/butteraugli-cli)
crate installs a `butteraugli` binary:

```bash
cargo install butteraugli-cli
```

```bash
# Compare two images (REFERENCE first, then DISTORTED; same dimensions required)
butteraugli original.png compressed.jpg
# Butteraugli score: 1.2345

butteraugli -q original.png compressed.jpg          # add a quality rating
butteraugli --json original.png compressed.jpg      # machine-readable output
butteraugli --pnorm original.png compressed.jpg     # also print the libjxl 3-norm
butteraugli --diffmap diff.png original.png compressed.jpg   # save a heatmap
butteraugli --quiet original.png compressed.jpg     # just the number

butteraugli --intensity-target 250 hdr_a.png hdr_b.png   # HDR (display nits)
butteraugli --hf-asymmetry 1.5 original.png compressed.jpg  # penalize ringing > blur
butteraugli --help
```

## Library usage

```toml
[dependencies]
butteraugli = "0.9.4"
```

### Input formats

| Function | Input type | Color space | Use case |
|----------|------------|-------------|----------|
| `butteraugli` | `ImgRef<RGB8>` | sRGB (gamma-encoded) | Standard 8-bit images |
| `butteraugli_linear` | `ImgRef<RGB<f32>>` | Linear RGB | HDR, 16-bit, float pipelines |

Both APIs support stride (padding). Images smaller than 8×8 (down to 1×1) are
reflect (mirror)-padded up to butteraugli's 8×8 floor and scored; the diffmap is
cropped back to the input size. The strip APIs below still require at least 8×8.

**Scaling contract:** linear `1.0` maps to `intensity_target` nits (default
`80.0` — the SDR convention used by libjxl's `butteraugli_main`). Values above
`1.0` are accepted and map proportionally above `intensity_target`. For HDR,
scale your linear data so `1.0` is the mastering/display peak and set
`.with_intensity_target(peak_nits)`. The sRGB u8 path decodes with the sRGB EOTF
to linear before the same scaling. Negative values are clamped to `0.0` inside
the opsin stage; NaN/Inf inputs are rejected with
`ButteraugliError::NonFiniteResult`.

### Example

```rust
use butteraugli::{butteraugli, ButteraugliParams, Img, RGB8};

let original: Vec<RGB8> = load_image();
let compressed: Vec<RGB8> = load_compressed();

let img1 = Img::new(original, width, height);
let img2 = Img::new(compressed, width, height);

let result = butteraugli(img1.as_ref(), img2.as_ref(), &ButteraugliParams::default())?;

println!("Score: {:.4}", result.score);
```

### From a flat `Vec<u8>` of RGB bytes

Most decoders hand you a packed row-major `Vec<u8>` (3 bytes per pixel), not a
`Vec<RGB8>`. Turn the bytes into `RGB8` pixels with `RGB8::new` and wrap them
with `Img::new`:

```rust
use butteraugli::{butteraugli, ButteraugliParams, Img, RGB8};

// `orig` and `dist` are `&[u8]`, row-major, exactly `width * height * 3` bytes.
fn score(orig: &[u8], dist: &[u8], width: usize, height: usize)
    -> Result<f64, butteraugli::ButteraugliError>
{
    let to_pixels = |bytes: &[u8]| -> Vec<RGB8> {
        bytes
            .chunks_exact(3)
            .map(|px| RGB8::new(px[0], px[1], px[2]))
            .collect()
    };

    let img1 = Img::new(to_pixels(orig), width, height);
    let img2 = Img::new(to_pixels(dist), width, height);

    let result = butteraugli(img1.as_ref(), img2.as_ref(), &ButteraugliParams::default())?;
    Ok(result.score)
}
```

If your rows carry padding (a row stride larger than `width * 3`), keep the flat
buffer and build a borrowed view with `ImgRef::new_stride(buf, width, height,
stride_in_pixels)` instead of copying into a packed `Vec` — the linear path has a
matching `ImgRef<RGB<f32>>` form. Both the sRGB and linear APIs walk stride
natively at no extra cost on the tightly-packed path.

#### RGBA input

There is no RGBA entry point — butteraugli compares three color channels and
ignores alpha. If your buffer is 4 bytes per pixel, drop the alpha while building
the pixels with `chunks_exact(4)` and keep the first three lanes:

```rust
use butteraugli::RGB8;
let to_pixels = |bytes: &[u8]| -> Vec<RGB8> {
    bytes
        .chunks_exact(4) // RGBA in, RGB8 out
        .map(|px| RGB8::new(px[0], px[1], px[2]))
        .collect()
};
```

Using `chunks_exact(3)` on RGBA bytes does *not* fail — it silently misaligns
every pixel after the first and produces a meaningless score, so match the chunk
size to your actual layout. (Premultiplied alpha is not unpremultiplied for you;
convert to straight RGB first if your pipeline premultiplies.)

### Difference map

```rust
let params = ButteraugliParams::default().with_compute_diffmap(true);
let result = butteraugli(img1.as_ref(), img2.as_ref(), &params)?;

if let Some(diffmap) = result.diffmap {
    let max_diff = diffmap.buf().iter().fold(0.0f32, |a, &b| a.max(b));
    println!("Maximum local difference: {:.4}", max_diff);
}
```

### Aggregations

`result.score` is the **max-norm** distance — the historical butteraugli score
behind the `<1.0 = good`, `>2.0 = bad` thresholds. `result.pnorm_3` is the libjxl
3-norm aggregation reported by `butteraugli_main --pnorm` and used in the
Cloudinary CID22 paper — useful for codec rate-distortion sweeps where averaging
tail and bulk distortion is more informative than max alone. Both are produced in
a single fused reduction pass and available regardless of whether
`compute_diffmap` was enabled (no extra allocation).

```rust
println!("max-norm:  {:.4}", result.score);     // or result.max_norm()
println!("3-norm:    {:.4}", result.pnorm_3);
println!("p=4 norm:  {:?}", result.pnorm(4.0)); // requires compute_diffmap = true
```

The CLI exposes the 3-norm via `butteraugli --pnorm`; JSON output always includes
a `pnorm_3` field.

### Custom parameters

```rust
let params = ButteraugliParams::new()
    .with_hf_asymmetry(1.5)        // penalize artifacts > blur
    .with_intensity_target(250.0)  // HDR display (nits)
    .with_compute_diffmap(true);
```

### Repeated comparisons and bounded memory

For comparing many distorted images against one reference, precompute the
reference once with `ButteraugliReference`, then call a `compare` method per
candidate — the reference-side XYB pyramid and masks are reused.

| Build the reference | Compare a candidate |
|---------------------|---------------------|
| `new(&[u8], w, h, params)` (sRGB) | `compare(&[u8])` |
| `new_linear(&[f32], w, h, params)` | `compare_linear(&[f32])` |
| `new_linear_planar(r, g, b, w, h, stride, params)` | `compare_linear_planar(r, g, b, stride)` |
| `from_srgb(ImgRef<RGB8>, params)` | `compare_srgb(ImgRef<RGB8>)` |
| `from_linear(ImgRef<RGB<f32>>, params)` | `compare_linear_imgref(ImgRef<RGB<f32>>)` |

```rust
use butteraugli::{ButteraugliParams, ButteraugliReference};

let reference = ButteraugliReference::new(&ref_rgb, width, height, ButteraugliParams::default())?;
for candidate in candidates {
    let result = reference.compare(&candidate)?;
    println!("{:.4}", result.score);
}
```

**Budgeting memory.** `ButteraugliReference::estimated_reference_bytes(width,
height, &params)` returns the a-priori heap cost of a reference's persistent
precompute *before* you build it — so a caller on a memory budget (e.g. an
encoder's quantization loop) can reserve or reject up front instead of risking an
OOM. On a live reference, `precompute_bytes()` reports the persistent precompute
footprint and `memory_bytes()` the full retained footprint (precompute + strip
source + idle buffer pool). `width()`, `height()`, and `params()` read back the
reference's configuration.

**Reclaiming memory.** `shrink_to_fit()` drains the cached buffer pool (one
re-allocation on the next `compare`, no loss of the warm-reference speedup);
`drop_strip_source()` releases the retained reference-side source when you know no
strip comparison will follow (subsequent `compare_strip*` calls then return
`InvalidParameter`).

**Strip mode for very large images.** The strip API bounds peak memory by walking
the image in horizontal strips, at equivalent wall time:

- One-shot: `butteraugli_strip` / `butteraugli_linear_strip` (plus
  `*_with_config` and `*_with_stop` variants).
- Cached reference: `ButteraugliReference::compare_strip` /
  `compare_linear_strip` / `compare_strip_srgb` / `compare_strip_linear_imgref`
  (and their `*_with_config` / `*_with_stop` twins). These require a reference
  built via `new` / `new_linear` / `from_srgb` / `from_linear` — the planar
  constructor doesn't retain interleaved source.

Strip scores match the full-image path to within `~1e-2` (the `strip_parity` test
tolerance; the FIR finite support makes interior diffmaps bit-identical, so the
small residual is only f64 sum-associativity in the cross-strip reduction). At
40 MP a heaptrack run measured peak heap dropping from 7.43 GB to 1.94 GB (3.8×;
butteraugli 0.9.3 notes), and a committed max-RSS A/B
([`benchmarks/strip_vs_full_mem_2026-06-23.tsv`](https://github.com/imazen/butteraugli/blob/main/benchmarks/strip_vs_full_mem_2026-06-23.tsv))
shows ~2.8–3.0× lower peak RSS across 16–36 MP with bit-identical scores. Tune
the halo overlap via `ButteraugliStripConfig::with_halo_rows`
(`HALO_ROWS_DEFAULT` = 64; `MIN_STRIP_HEIGHT` = 8).

### Cooperative cancellation

Every slow entry point has a `*_with_stop` twin that takes a trailing `stop:
&dyn enough::Stop` token. The token is polled at the outermost per-scale boundary
(one-shot) and once per strip at the top of the strip loop — never inside the
per-pixel opsin / blur / Malta / masking kernels — so a cancellation is honored
at scale / strip granularity with zero hot-loop overhead. When the token fires,
the call returns `Err(ButteraugliError::Cancelled(reason))`.

| Non-cancellable | Cancellable twin |
|-----------------|------------------|
| `butteraugli` | `butteraugli_with_stop` |
| `butteraugli_linear` | `butteraugli_linear_with_stop` |
| `butteraugli_strip` | `butteraugli_strip_with_stop` |
| `butteraugli_linear_strip` | `butteraugli_linear_strip_with_stop` |
| `ButteraugliReference::compare` | `compare_with_stop` |
| `ButteraugliReference::compare_linear` | `compare_linear_with_stop` |
| `ButteraugliReference::compare_srgb` | `compare_srgb_with_stop` |
| `ButteraugliReference::compare_linear_imgref` | `compare_linear_imgref_with_stop` |
| `ButteraugliReference::compare_strip` | `compare_strip_with_stop` |
| `ButteraugliReference::compare_linear_strip` | `compare_linear_strip_with_stop` |
| `ButteraugliReference::compare_strip_srgb` | `compare_strip_srgb_with_stop` |
| `ButteraugliReference::compare_strip_linear_imgref` | `compare_strip_linear_imgref_with_stop` |

The non-`_with_stop` functions are exactly their `_with_stop` twin called with
`enough::Unstoppable`, the no-op token — so there is no behavioral or perf
difference between `butteraugli(a, b, &p)` and `butteraugli_with_stop(a, b, &p,
&enough::Unstoppable)`. The [`enough`](https://crates.io/crates/enough) crate is
re-exported as `butteraugli::enough`, so you can name `Stop` / `Unstoppable` /
`StopReason` without adding it to your own `Cargo.toml`.

For a token you can actually trip from another thread (a deadline, a "newer
request arrived" signal, a Ctrl-C handler), pull in
[`almost-enough`](https://crates.io/crates/almost-enough), which provides
`Stopper` — an `Arc`-backed, `Clone`-able, `Send + Sync` cancellation flag. Hand a
clone to the worker, keep one for yourself, and call `.cancel()` when you want the
comparison to bail:

```toml
[dependencies]
butteraugli = "0.9.4"
enough = "0.4.4"          # only if you name the Stop trait directly
almost-enough = "0.4.4"   # the Stopper / SyncStopper cancellation tokens
```

```rust
use almost_enough::Stopper;
use butteraugli::{butteraugli_with_stop, ButteraugliParams, ButteraugliError};

let stop = Stopper::new();
let worker_stop = stop.clone(); // clones share one flag; any clone can cancel

let handle = std::thread::spawn(move || {
    butteraugli_with_stop(img1.as_ref(), img2.as_ref(),
        &ButteraugliParams::default(), &worker_stop)
});

// ... later, from elsewhere — e.g. a newer request landed, or a deadline hit:
stop.cancel();

match handle.join().unwrap() {
    Ok(result)                            => println!("score {:.4}", result.score),
    Err(ButteraugliError::Cancelled(why)) => eprintln!("cancelled: {why:?}"),
    Err(e)                                => eprintln!("error: {e}"),
}
```

`Stopper` uses Relaxed atomics; reach for `almost_enough::SyncStopper` (same
shape) if you need Release/Acquire ordering to publish other writes alongside the
cancel. Both satisfy `&dyn enough::Stop`. A pre-cancelled token
(`Stopper::cancelled()`) makes the very first per-scale check bail before any
per-pixel work runs.

## Errors and result types

Every fallible entry point returns `Result<ButteraugliResult, ButteraugliError>`.

`ButteraugliError` is `#[non_exhaustive]` and implements both `Display` and
`std::error::Error`, so it slots into `?`, `anyhow`, `thiserror`, and friends. Its
variants cover dimension mismatch / too-small / overflow inputs, a
`NonFiniteResult` for NaN/Inf pixels, an `InvalidParameter` for out-of-range
params or strip misuse, and `Cancelled(enough::StopReason)` from the cancellation
tokens above. New variants may be added in future releases — match with a `_ =>
…` arm.

`ButteraugliResult` carries:

| Field / method | Type | Meaning |
|----------------|------|---------|
| `score` | `f64` | Max-norm distance (the historical score; `<1.0` good, `>2.0` bad). |
| `max_norm()` | `f64` | Same value as `score`, named for unambiguous call sites. |
| `pnorm_3` | `f64` | libjxl 3-norm aggregation (`butteraugli_main --pnorm`). Always populated. |
| `pnorm(p)` | `f64 -> Option<f64>` | libjxl p-norm of the diffmap. `p == 3.0` reuses `pnorm_3` for free; other `p` returns `None` unless `compute_diffmap` was enabled. |
| `diffmap` | `Option<ImgVec<f32>>` | Per-pixel difference map; `Some` only when `compute_diffmap` was set. |

The struct is also `#[non_exhaustive]`; construct it only via the comparison
functions, and read it by field/method.

## Features

- **`rayon`** *(default)*: multi-threaded blur and Malta passes.
- **`avx512`** *(default)*: AVX-512 runtime dispatch (used only when the CPU supports it).
- **`iir-blur`**: O(N) recursive Gaussian instead of FIR convolution; faster on
  non-AVX-512 hardware but not score-parity with libjxl — off by default.
- **`unsafe-performance`**: unchecked indexing in hot loops (~6% fewer
  instructions; each function pre-validates the full access range).
- **`internals`**: expose internal modules for testing/benchmarking (unstable API).

## Performance

SIMD-accelerated via [`archmage`](https://crates.io/crates/archmage) with runtime
dispatch — the binary picks the widest ISA the CPU supports at run time, so the
same build is fast everywhere:

| Target | CPU support |
|--------|-------------|
| x86-64-v4 | AVX-512 (Skylake-X, Zen 4+) |
| x86-64-v3 | AVX2/FMA (Haswell+, Zen 1+) |
| x86-64-v2 | SSE4.2 (Nehalem+) |
| ARM64 | NEON (Apple Silicon, Cortex-A75+) |
| WASM SIMD128 | Browser / WASI runtimes with SIMD |

No C dependencies. Safe Rust by default (`unsafe-performance` is opt-in).


> **ICC profiles:** this crate assumes sRGB input. libjxl's `butteraugli_main`
> applies ICC profile and gAMA/cHRM transforms via its CMS before scoring. Images
> with non-sRGB ICC profiles (Adobe RGB, Display P3, ProPhoto RGB) will produce
> different scores between the two implementations. Strip ICC profiles or convert
> to sRGB before comparing.

## API comparison with C++ libjxl

| Feature | C++ butteraugli | This crate |
|---------|-----------------|------------|
| Input format | Linear RGB float | sRGB u8 or linear RGB f32 |
| Color space | Linear RGB only | sRGB (auto-converted) or linear |
| ICC profiles | CMS transforms to linear sRGB | Assumes sRGB (profiles ignored) |
| Channel layout | Planar | Interleaved RGB via `imgref` (planar also supported) |
| Stride support | Manual | Built-in via `ImgRef::new_stride()` |

### XYB note

Butteraugli's internal "XYB" is **not** the same color space as JPEG XL / jpegli
XYB. Key differences:

- **Matrix coefficients**: different opsin absorbance weights (e.g. row 2 is
  `[0.02, 0.02, 0.205]` vs jpegli's `[0.243, 0.205, 0.552]`).
- **Nonlinearity**: log-based Gamma function (FastLog2f), not cube root.
- **Dynamic adaptation**: blurs the input, computes per-pixel sensitivity ratios,
  and modulates the opsin-transformed signal — jpegli XYB has no equivalent step.
- **Biases**: large additive biases (`~1.76, ~1.76, ~12.23`) vs jpegli's small
  bias (`~0.0038`).

Always provide RGB input (sRGB u8 or linear f32); butteraugli handles the
conversion internally. Pre-converted XYB cannot be used because the dynamic
sensitivity adaptation requires raw linear RGB.

## References

- [JPEG XL reference implementation (libjxl)](https://github.com/libjxl/libjxl) — canonical source for butteraugli
- [Butteraugli theory paper](https://github.com/google/butteraugli/blob/master/doc/butteraugli-theory.pdf)

This crate ports from libjxl's `lib/extras/butteraugli.cc`, where butteraugli
development continued after the original
[google/butteraugli](https://github.com/google/butteraugli) repository was
archived. Scores are validated against libjxl's `butteraugli_main`, not the
archived standalone version. With deep thanks to Jyrki Alakuijala and the JPEG XL
/ libjxl authors at Google, and to Cloudinary, whose CID22 work popularized the
3-norm aggregation.

## Development

```bash
cargo fmt --all -- --check
cargo clippy --lib --tests -- -D warnings
cargo test --lib
cargo test --test conformance
```

See [`CHANGELOG.md`](https://github.com/imazen/butteraugli/blob/main/CHANGELOG.md)
for release history and [`benchmarks/README.md`](https://github.com/imazen/butteraugli/blob/main/benchmarks/README.md)
for benchmark methodology.

## AI-generated code notice

Developed with Claude (Anthropic) and validated against C++ libjxl
`butteraugli_main` to within FMA rounding noise on real photographs.

## License

[BSD-3-Clause](https://github.com/imazen/butteraugli/blob/main/LICENSE) — the same
license as the upstream [libjxl](https://github.com/libjxl/libjxl) butteraugli
implementation this port is derived from. Copyright the JPEG XL Project Authors;
see [`LICENSE`](https://github.com/imazen/butteraugli/blob/main/LICENSE) for the
full text.

## Image tech I maintain

| | |
|:--|:--|
| **Codecs** ¹ | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] · [zenjxl] · [zenbitmaps] · [heic] · [zentiff] · [zenpdf] · [zensvg] · [zenjp2] · [zenraw] · [ultrahdr] |
| Codec internals | [zenjxl-decoder] · [jxl-encoder] · [zenrav1e] · [rav1d-safe] · [zenavif-parse] · [zenavif-serialize] |
| Compression | [zenflate] · [zenzop] · [zenzstd] |
| Processing | [zenresize] · [zenquant] · [zenblend] · [zenfilters] · [zensally] · [zentone] |
| Pixels & color | [zenpixels] · [zenpixels-convert] · [linear-srgb] · [garb] |
| Pipeline & framework | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] · [zenwasm] · [zentract] |
| Metrics | [zensim] · [fast-ssim2] · **butteraugli** · [zenmetrics] · [resamplescope-rs] |
| Pickers & ML | [zenanalyze] · [zenpredict] · [zenpicker] |
| Products | [Imageflow] image engine ([.NET][imageflow-dotnet] · [Node][imageflow-node] · [Go][imageflow-go]) · [Imageflow Server] · [ImageResizer] (C#) |

<sub>¹ pure-Rust, `#![forbid(unsafe_code)]` codecs, as of 2026</sub>

### General Rust awesomeness

[zenbench] · [archmage] · [magetypes] · [enough] · [whereat] · [cargo-copter]

[Open source](https://www.imazen.io/open-source) · [@imazen](https://github.com/imazen) · [@lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith)

[zenjpeg]: https://github.com/imazen/zenjpeg
[zenpng]: https://github.com/imazen/zenpng
[zenwebp]: https://github.com/imazen/zenwebp
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic
[zentiff]: https://github.com/imazen/zentiff
[zenpdf]: https://github.com/imazen/zenpdf
[zensvg]: https://github.com/imazen/zenextras
[zenjp2]: https://github.com/imazen/zenextras
[zenraw]: https://github.com/imazen/zenraw
[ultrahdr]: https://github.com/imazen/ultrahdr
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[jxl-encoder]: https://github.com/imazen/jxl-encoder
[zenrav1e]: https://github.com/imazen/zenrav1e
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[zenavif-parse]: https://github.com/imazen/zenavif-parse
[zenavif-serialize]: https://github.com/imazen/zenavif-serialize
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenzstd]: https://github.com/imazen/zenzstd
[zenresize]: https://github.com/imazen/zenresize
[zenquant]: https://github.com/imazen/zenquant
[zenblend]: https://github.com/imazen/zenblend
[zenfilters]: https://github.com/imazen/zenfilters
[zensally]: https://github.com/imazen/zensally
[zentone]: https://github.com/imazen/zentone
[zenpixels]: https://github.com/imazen/zenpixels
[zenpixels-convert]: https://github.com/imazen/zenpixels
[linear-srgb]: https://github.com/imazen/linear-srgb
[garb]: https://github.com/imazen/garb
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zencodecs
[zenlayout]: https://github.com/imazen/zenlayout
[zennode]: https://github.com/imazen/zennode
[zenwasm]: https://github.com/imazen/zenwasm
[zentract]: https://github.com/imazen/zentract
[zensim]: https://github.com/imazen/zensim
[fast-ssim2]: https://github.com/imazen/fast-ssim2
[zenmetrics]: https://github.com/imazen/zenmetrics
[resamplescope-rs]: https://github.com/imazen/resamplescope-rs
[zenanalyze]: https://github.com/imazen/zenanalyze
[zenpredict]: https://github.com/imazen/zenanalyze
[zenpicker]: https://github.com/imazen/zenanalyze
[zenbench]: https://github.com/imazen/zenbench
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[cargo-copter]: https://github.com/imazen/cargo-copter
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-dotnet-server
[ImageResizer]: https://github.com/imazen/resizer
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
