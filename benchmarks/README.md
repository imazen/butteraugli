# butteraugli benchmarks

Methodology and reproduction for the numbers quoted in the top-level
[`README.md`](../README.md). Two harnesses live here:

- **Speed A/B** — this crate vs `libjxl::ButteraugliDiffmap` (C++), via FFI.
- **Memory A/B** — strip-wise vs full-image peak memory at large sizes.

Raw results are committed alongside this file as `*.txt` (run output) + `*.meta`
(provenance) sidecars. All numbers here come from those committed files — nothing
is estimated or extrapolated. Re-runs should record a fresh `rustc -V` and host
line into the `.meta` sidecar.

> Built **without** `-C target-cpu=native`. The Rust side uses
> [`archmage`](https://github.com/imazen/archmage) runtime SIMD dispatch (the
> dispatch real users get); the C++ side pins its compile baseline to
> `x86-64-v2` and uses highway runtime dispatch. Pinning to a single ISA at
> compile time would give misleading numbers for either.

## 1. Speed: butteraugli (Rust) vs libjxl `ButteraugliDiffmap` (C++)

Harness: [`butteraugli-bench/benches/bench_compare.rs`](../butteraugli-bench/benches/bench_compare.rs),
driven by [zenbench](https://github.com/imazen/zenbench) (interleaved/randomized
round-robin execution + paired statistics, which removes the thermal/turbo bias
that back-to-back per-function timing bakes in).

### What is measured

Both implementations receive **identical pre-linearized planar f32 buffers**.
The only work in the timed region is the butteraugli pipeline itself (XYB
conversion, blur, masks, Malta, psycho, diffmap, score reduction). sRGB→linear
conversion and all buffer setup happen **once, outside** the measured loop;
`black_box` guards the inputs and the returned score so nothing is optimized
away. No file I/O occurs inside any timed closure.

Three variants per size:

| Variant | Threading | Notes |
|---------|-----------|-------|
| `rust_butteraugli` | rayon multi-threaded | out-of-the-box defaults (`rayon` + `avx512` dispatch) |
| `rust_butteraugli_st` | 1-thread rayon pool | apples-to-apples vs the single-threaded C++ side |
| `cpp_libjxl` | single-threaded | `libjxl::ButteraugliDiffmap` via the FFI shim |

Input: a synthetic gradient + saturating-add perturbation (a throughput probe —
this measures pipeline *speed*, not rate-distortion quality, so synthetic input
is appropriate here). Same image, dimensions, and pixel format across all three
contenders.

### Results (committed: `cpp_vs_rust_baseline_2026-04-27.{txt,meta}`)

- **Host:** AMD Ryzen 9 7950X (Zen 4), 128 GB RAM, WSL2
- **butteraugli commit:** `af27826`
- **Profile:** `--release`; Rust default features (`rayon` + `avx512`)
- **libjxl:** clone of `github.com/libjxl/libjxl` (vendored, unmodified upstream),
  `-march=x86-64-v2 -O3`, highway runtime dispatch (`SSE4 | AVX2 | AVX3 | AVX3_ZEN4`)

| Image | `rust_butteraugli` (32T) | `rust_butteraugli_st` (1T) | `cpp_libjxl` (1T) |
|-------|:-:|:-:|:-:|
| 512×512 | 17.8 ms | 23.4 ms | 46.3 ms |
| 1280×720 | 66.9 ms | 84.6 ms | 255.8 ms |
| 1920×1080 | 197.1 ms | 228.9 ms | 594.0 ms |
| 3840×2160 | 998.0 ms | 845.4 ms | 2629.2 ms |

Single-threaded, this crate is ~2–3× faster than the C++ pipeline on this
hardware; rayon adds further headroom at HD and below (at 4 K the rayon split
stops helping for this synthetic input — `st` is actually faster there, with a
later-rounds drift caveat noted in the raw output). The right chart for "which is
fastest" is a sorted throughput bar — zenbench prints exactly that in the `.txt`.

The other `cpp_libjxl_*_2026-04-27.*` files in this directory record A/Bs of
candidate kernel changes (h/v blur, UHF/HF fusion, Malta padding) explored as
potential upstream contributions; the corresponding `libjxl_*.patch` files are the
diffs those runs measured.

### Reproduce

```sh
git clone https://github.com/imazen/butteraugli && cd butteraugli
git checkout af27826        # the commit these numbers came from

# Rust-only (no C++ build needed):
BUTTERAUGLI_BENCH_NO_CPP=1 cargo bench -p butteraugli-bench --bench bench_compare

# With the C++ comparison. By default build.rs shallow-clones libjxl into
# experiments/libjxl-vendor/ and cmake-builds it (~600 MB, needs cmake + a C++
# toolchain). For a *pinned, reproducible* C++ baseline, point LIBJXL_DIR at your
# own libjxl checkout at a known commit instead of the auto-cloned HEAD:
git clone https://github.com/libjxl/libjxl /path/to/libjxl
git -C /path/to/libjxl checkout <LIBJXL_SHA>
git -C /path/to/libjxl submodule update --init --recursive
LIBJXL_DIR=/path/to/libjxl cargo bench -p butteraugli-bench --bench bench_compare
```

Note: the default vendored path clones libjxl at `HEAD`, so it is **not** pinned
across time — use `LIBJXL_DIR` with an explicit `git checkout` for a baseline you
can reproduce later. The `af27826` numbers above were taken against libjxl HEAD as
of 2026-04-27.

## 2. Memory: strip-wise vs full-image

Harness: [`butteraugli-bench/examples/strip_vs_full_mem.rs`](../butteraugli-bench/examples/strip_vs_full_mem.rs).
Peak memory is measured with `/usr/bin/time -v` (max RSS) around a **one-shot**
compare (not the warm-reference path), `strip_height = 256`, `halo = 64`,
multithreaded default.

### Results (committed: `strip_vs_full_mem_2026-06-23.{tsv,meta}`)

- **Host:** lilith (Ryzen 9 7950X), butteraugli commit `74fb6e0`
- **Input:** procedural high-entropy noise (worst case for compressibility)

| Size | full max-RSS | strip max-RSS | ratio | score (full == strip) |
|------|:-:|:-:|:-:|:-:|
| 16 MP | 3.10 GiB | 1.09 GiB | 2.84× | bit-identical |
| 24 MP | 4.66 GiB | 1.57 GiB | 2.97× | bit-identical |
| 36 MP | 6.47 GiB | 2.26 GiB | 2.86× | bit-identical |

The strip path is a correct drop-in: scores are bit-identical to the full-image
path at every measured size (the `strip_parity` test guarantees agreement to
within `0.01` in general; this high-entropy harness hit exact equality). Wall time
is equal-or-slightly-faster for strips (better cache locality offsets the ~50%
halo recompute). The ratio grows with image size because full scales `O(h·w)`
while strip scales `O(strip_height·w)` plus an `O(MP)` input/output floor.

A complementary heaptrack run (pure heap, not RSS) at 40 MP measured peak heap
dropping from 7.43 GB to 1.94 GB (3.8×) at equivalent wall time — recorded in the
butteraugli 0.9.3 changelog entry. RSS ratios are smaller than heap ratios because
RSS includes the fixed input/output buffers, stack, and code that don't shrink
with strip processing.

### Reproduce

```sh
git clone https://github.com/imazen/butteraugli && cd butteraugli
git checkout 74fb6e0
cargo build --release -p butteraugli-bench --example strip_vs_full_mem
for mp in 16 24 36; do
  /usr/bin/time -v ./target/release/examples/strip_vs_full_mem full  "$mp" 256
  /usr/bin/time -v ./target/release/examples/strip_vs_full_mem strip "$mp" 256
done
```

## Integrity notes

- No `-C target-cpu=native` anywhere (runtime dispatch is what ships).
- I/O and one-time setup are outside every timed region; outputs are consumed
  (`black_box` / asserted) so they can't be optimized away.
- Threading mode is stated per row; never single-thread A vs multi-thread B
  without labeling both.
- New benches use [zenbench](https://github.com/imazen/zenbench), not criterion.
- Numbers are copied verbatim from the committed `.txt` / `.tsv` / `.meta`
  sidecars; none are extrapolated across size or hardware.
