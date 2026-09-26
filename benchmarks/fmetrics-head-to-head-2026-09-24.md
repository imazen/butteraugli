# Local Butteraugli versus fmetrics (2026-09-24)

On the same AMD Ryzen 9 7900X, this Rust crate was faster and used less peak
process RSS than [fmetrics](https://github.com/halidecx/fmetrics) on all four
tested inputs. This is a single-thread, fresh one-shot comparison; it is not
a claim that the implementations produce identical scores.

Both sides received the same linear-light RGB `f32` values. Source PNGs were
decoded and JPEG quality-75 distortions were generated before timing. The
Rust API received planar data and called `new_linear_planar` followed by
`compare_linear_planar`; fmetrics received interleaved data and called
`fmetrics_butteraugli_cmp` with a fresh workspace. Both used intensity 203
and reported p-norm 3. Both representations were kept in memory for both
processes. `RAYON_NUM_THREADS=1` and `MALLOC_ARENA_MAX=1` were set.

The timer surrounds only the metric API calls. Each timing is the median of
alternating process pairs, reversing order each pair, after two warmups per
backend. fmetrics was built with Zig 0.16.0 using `zig build --release=fast`;
the optional `-Dcpu=native` produced the same library archive. Rust used
`cargo build --release` with rustc 1.98.1. fmetrics revision was
`e8bf3cfe06fb78f51864804b2f7003667d4c5a71`; Rust was based on
`aac925f6e8aaad7e9ddbceecaa514b23f7b419a4` plus the Malta and early-XYB
changes now recorded in commits `010dd97` and `12e7264`.

| Source and size | Pairs | Rust metric | fmetrics metric | fmetrics / Rust |
| --- | ---: | ---: | ---: | ---: |
| GB82 baby photo, 576² | 20 | 40.6 ms | 105.4 ms | 2.60× |
| CLIC 2025 photo, 2048×1360 | 10 | 468.8 ms | 808.2 ms | 1.72× |
| GB82 screenshot, 2940×1912 | 8 | 704.5 ms | 1411.3 ms | 2.00× |
| Same screenshot resized to 3840×2160 | 4 | 1019.1 ms | 2089.2 ms | 2.05× |

The 4K standard-build result was replicated with fmetrics' optional LTO
build: six alternating pairs measured 1021.1 ms Rust versus 2101.3 ms
fmetrics. LTO was not faster on the 576² photo either (10 alternating pairs:
105.8 ms standard versus 107.8 ms LTO for fmetrics).

Peak RSS below is from `/usr/bin/time -f %M`, three one-shot processes per
backend. It includes image decoding, distortion generation, and both input
layouts, so it is a whole-process measure rather than isolated metric heap.
The 4K row uses the LTO fmetrics build; other rows use its standard build.

| Source and size | Rust peak RSS | fmetrics peak RSS |
| --- | ---: | ---: |
| Baby, 576² | 61,712 KiB | 76,152 KiB |
| CLIC photo, 2048×1360 | 487,716 KiB | 612,976 KiB |
| Screenshot, 2940×1912 | 998,372 KiB | 1,233,860 KiB |
| Resized screenshot, 3840×2160 | 1,445,308 KiB | 1,819,300 KiB |

The p-norm-3 scores differed despite matched input values and settings:

| Source | Rust | fmetrics | fmetrics relative to Rust |
| --- | ---: | ---: | ---: |
| Baby | 1.07670513 | 1.13181751 | +5.1% |
| CLIC photo | 1.40043743 | 1.52034070 | +8.6% |
| Screenshot | 1.50013776 | 1.57149636 | +4.8% |
| Resized screenshot | 1.16084142 | 1.17247241 | +1.0% |

The score differences mean fmetrics cannot be treated as a score-exact
replacement for this crate. Its authors' [published 4K result](https://halide.cx/blog/fmetrics/)
compares fmetrics against libjxl on another CPU and image pair, not against
this Rust crate. This local comparison covers a small set of inputs and only
the fresh one-shot API path; persistent-reference throughput and other CPUs
remain unmeasured.

To reproduce, build fmetrics and the optional benchmark example, then run
the paired driver on a source image:

```sh
cd /tmp/butteraugli-fmetrics
zig build --release=fast
cd /home/lilith/work/zen/butteraugli
FMETRICS_LIB_DIR=/tmp/butteraugli-fmetrics/zig-out/lib \
  BUTTERAUGLI_BENCH_NO_CPP=1 \
  cargo build --release -p butteraugli-bench --example fmetrics_compare \
  --features fmetrics-ffi
python3 butteraugli-bench/scripts/run_fmetrics_compare.py \
  target/release/examples/fmetrics_compare \
  /home/lilith/work/codec-corpus/gb82/baby-lossless.png --pairs 20
```
