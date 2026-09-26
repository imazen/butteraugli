# Local Butteraugli versus libjxl (2026-09-24)

On the AMD Ryzen 9 7900X, this Rust crate's one-shot Butteraugli path was
1.51–2.16× faster than the vendored libjxl C++ implementation with both sides
running single-threaded. Their p-norm-3 scores agreed within 0.00021% on all
four tested image pairs. Whole-process peak RSS was lower for Rust too.

The same real PNG images and JPEG quality-75 distortions from the
[fmetrics comparison](fmetrics-head-to-head-2026-09-24.md) were used. PNG
decode, JPEG generation/decode, sRGB-to-linear conversion, and planar plus
interleaved input preparation happened before the timer. Both implementations
received identical linear-light planar `f32` pixels, intensity target 203 nits,
and the same p-norm-3 reduction (the mean of normalized p=3, 6, and 12 norms).
Rust timed `ButteraugliReference::new_linear_planar` plus
`compare_linear_planar`; libjxl timed the FFI shim's input copy to `Image3F`,
`ButteraugliDiffmap`, and p-norm calculation. The input copy is required by
this FFI adapter but is a small part of its total time. The FFI shim
was extended to set the explicit intensity target and compute p-norm-3, while
its original default-80-nit/max-score entry point remains available for old
benchmarks.

Each row is the median of alternating, reversed-order process pairs after two
warmups per backend. `RAYON_NUM_THREADS=1` and `MALLOC_ARENA_MAX=1` were set.
Peak RSS is the median of three fresh processes per backend using
`/usr/bin/time -f %M`. RSS includes input preparation and both input layouts,
so it is a whole-process comparison rather than metric-only heap usage.

| Image | Pairs | Rust metric | libjxl metric | libjxl / Rust | Rust RSS | libjxl RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GB82 baby photo, 576² | 20 | 40.115 ms | 86.754 ms | 2.16× | 63,260 KiB | 95,136 KiB |
| CLIC 2025 photo, 2048×1360 | 10 | 484.403 ms | 730.243 ms | 1.51× | 489,416 KiB | 731,244 KiB |
| GB82 screenshot, 2940×1912 | 8 | 700.268 ms | 1352.992 ms | 1.93× | 999,800 KiB | 1,464,208 KiB |
| Screenshot resized to 3840×2160 | 4 | 1016.223 ms | 1996.722 ms | 1.96× | 1,447,176 KiB | 2,151,420 KiB |

| Image | Rust p-norm-3 | libjxl p-norm-3 |
| --- | ---: | ---: |
| Baby | 1.07670513465694206 | 1.07670503449725885 |
| CLIC photo | 1.40043743194229209 | 1.40043736688917519 |
| Screenshot | 1.50013775861826737 | 1.50013759886955622 |
| Resized screenshot | 1.16084141705875177 | 1.16083901839081771 |

An indicative 12-thread Rust run on the 4K pair measured 605–657 ms over
three calls in one process, versus libjxl's single-thread 1997 ms median above;
one fresh 12-thread process peaked at 1,803,316 KiB RSS. The thread counts
are different, so the single-thread table is the algorithmic comparison.

The libjxl checkout was revision `b87738951c1254cd8cccaa6d47712ba735da56d8`
(2026-09-20), unmodified, built Release with `-march=x86-64-v2 -O3` and
Highway runtime dispatch including AVX3 and AVX3_ZEN4. The Rust tree was
based on `aac925f6e8aaad7e9ddbceecaa514b23f7b419a4` plus the Malta and XYB
memory changes recorded in commits `010dd97` and `12e7264`, built with Cargo's
release profile and rustc 1.98.1. These results apply to this CPU, four image
pairs, and the fresh one-shot path; persistent-reference throughput and other
CPUs were not measured here.

To reproduce after building the vendored libjxl library:

```sh
cargo build --release -p butteraugli-bench --example fmetrics_compare
python3 butteraugli-bench/scripts/run_fmetrics_compare.py \
  target/release/examples/fmetrics_compare \
  /home/lilith/work/codec-corpus/gb82/baby-lossless.png \
  --other libjxl --pairs 20
```
