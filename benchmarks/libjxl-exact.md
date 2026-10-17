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
[`4b8e72de`](https://github.com/imazen/butteraugli/tree/4b8e72de6a127bf83ff79b1d6509b8c167ca24a5/benchmarks/libjxl_exact_review_2026-10-17),
and on r5900xt at
`/mnt/v/output/butteraugli/libjxl-parity-2026-10-17/pr21-pre-cleanup-4b8e72de/`.

## Reproduce

`just exact-test` runs the regression suite. `just exact-bench` accepts
`BU_BENCH_REFERENCE`, `BU_BENCH_DISTORTED` and `BU_BENCH_OUTPUT` paths; set
`RAYON_NUM_THREADS=1` for the single-thread case. Run heavy commands under the
host's resource supervisor with TMPDIR on disk.

Regenerate the six procedural C++ scores with
`python3 butteraugli/tests/data/generate_libjxl_exact.py LIBJXL_ROOT OUTPUT_TSV`
(Pillow required). Build the pinned `butteraugli_main` with its p-norm printf
changed from `%f` to `%.17g`; the generator rejects missing instrumentation.
