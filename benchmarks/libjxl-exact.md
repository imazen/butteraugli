# libjxl numerical parity

`Numerics::LibjxlExact` follows libjxl v0.12's AVX2 arithmetic policy
(`a7a9c787341cf703dede03c2009fa460cae5e5df`, 80 nits). It changes transfer,
opsin, Gaussian/FIR ordering and final combination rounding. It preserves
strided input, reference reuse and consistent results across Rust SIMD tiers.

On 19 pairs (CID22 JPEG distortions, odd dimensions, and 2048x2048), Fast's
max-norm, 3-norm and map bits match main `0e397e5e`. Exact's largest absolute
C++ differences are **4.76837158203125e-7** (max) and **5.475178621594523e-8**
(3-norm). Frozen CI oracles require both errors <=1e-6, with Fast as a negative
control. Tests also pin Fast bits and Exact cross-tier fingerprints.

## Cost

Measured on r5900xt, nice/ionice, 24 GiB cap, no `target-cpu=native`.
Zenbench medians of seven rounds after warmup; resident RGB8 CID22 inputs,
decode excluded, result destruction included. Default threads run within a
28-CPU affinity mask. These are descriptive shared-host measurements.

| Size | Threads | Fast ms | Exact ms |
|---|---|---:|---:|
| 512x512 | 1 | 44.644016 | 135.591693 |
| 512x512 | default | 20.116496 | 64.669450 |
| 2048x2048 | 1 | 1078.813457 | 2372.012462 |
| 2048x2048 | default | 536.002881 | 907.986304 |

GNU time -v peak RSS, separate 2048x2048 planar processes, one reference and
three comparisons, including caller storage and allocator retention:

| Threads | Fast KiB | Exact KiB |
|---|---:|---:|
| 1 | 691680 | 740980 |
| default | 929680 | 892824 |

The reference precompute estimate equals retained psycho/mask storage in both
modes. The planar peak estimate covers extra Exact FIR/subsample buffers and
excludes caller storage and allocator overhead; it is not an RSS ceiling.

Measurement receipt: `rc=0 230s | peak-RSS 1.29GiB | min-avail 51072MiB | peak-load 6.21`.
Measured implementation: `a9063bf7`; zenbench harness: `7b70399f`.
Full evidence and source are preserved at commit
[`4b8e72de`](https://github.com/imazen/butteraugli/tree/4b8e72de6a127bf83ff79b1d6509b8c167ca24a5/benchmarks/libjxl_exact_review_2026-10-17).

## Generic blur chunks (2026-10-17)

`dd68df77` replaces Exact's transposed scalar FIR passes with contiguous
horizontal/vertical SIMD passes, using `attune` and generic eight-lane chunks
with scalar remainders. Kernel normalization is reused across the passes;
clipped vertical borders share normalization across columns. The five-tap
kernel uses `mul_add_portable` to preserve fused rounding on every backend.
The archmage/magetypes preview is `0.9.31-beta.1`, pinned to its release-source
commit `da0ed5de`; a git source lets stable benchmark dependencies coexist.

Same zenbench method as above; before is `3bf7d0cb`, after is `dd68df77`.
Medians in milliseconds:

| Size | Threads | Fast before | Fast after | Exact before | Exact after |
|---|---|---:|---:|---:|---:|
| 512x512 | 1 | 44.869157 | 45.119599 | 134.561461 | 57.868382 |
| 512x512 | default | 25.833142 | 20.585760 | 65.237968 | 34.202302 |
| 2048x2048 | 1 | 1116.941699 | 1101.002107 | 2421.189764 | 1279.596919 |
| 2048x2048 | default | 582.130545 | 550.116370 | 943.957154 | 661.031346 |

Both modes preserve both norm bits and every diffmap byte on all 19 pairs.
The scalar-reference blur test covers widths 1–65, short images, clipped borders
and all four general FIR lengths; the existing tier fingerprints also pass.
Default-thread timings have more shared-host variation than single-thread runs.

`perf record -m 128 -g` identified the scalar FIR pass at 55.32% of sampled
cycles in the earlier fixed-count planar runner (`7b70399f`). Raw zenbench JSON, profiler
traces, executable hashes, commands and provenance are on r5900xt under
`/mnt/v/output/butteraugli/libjxl-parity-2026-10-17/data/blur-attune/`.
Baseline profiler receipt: `rc=0 6s | peak-RSS 0.71GiB | min-avail 52578MiB | peak-load 3.21`.
Benchmark/parity/profile receipt:
`rc=0 84s | peak-RSS 1.26GiB | min-avail 48493MiB | peak-load 2.53`.

## Reproduce

`just exact-test` runs the regression suite. `just exact-bench` accepts
`BU_BENCH_REFERENCE`, `BU_BENCH_DISTORTED` and `BU_BENCH_OUTPUT` paths; set
`RAYON_NUM_THREADS=1` for the single-thread case. Run heavy commands under the
host's resource supervisor with TMPDIR on disk.

Regenerate the six procedural C++ scores with
`python3 butteraugli/tests/data/generate_libjxl_exact.py LIBJXL_ROOT OUTPUT_TSV`
(Pillow required). Build the pinned `butteraugli_main` with its p-norm printf
changed from `%f` to `%.17g`; the generator rejects missing instrumentation.
