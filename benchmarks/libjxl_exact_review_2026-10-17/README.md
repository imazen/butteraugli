# Exact numerics review qualification — 2026-10-17

The review fixes unify sRGB transfer selection, consume the retained reference
mask, preserve Exact strip reduction lanes, dispatch opsin FMA through archmage,
and copy strided planar input directly into image planes.

Before: `d29e066830ee41c6b4492274747aebb0310d459f`.
After implementation: `a9063bf7a7b49ac31f4451d965cb4eaae5629fbb`;
measurement harness: `7b70399fe9eda9a9764a608bc49a003ae671405c`.
Frozen Fast baseline: `0e397e5e716cd70ca40605ad93d68df590869dda`.
Oracle: libjxl v0.12 `a7a9c787341cf703dede03c2009fa460cae5e5df`, 80 nits.
All executions were on r5900xt, without `target-cpu=native`, under nice/ionice
and a 24 GiB run-heavy limit. Main was not changed.

## Output qualification

[All 19 pairs](parity.tsv), including 2048x2048: Fast max-norm, 3-norm and every
f32 map bit match frozen main. Exact's two norms and every map bit match the
pre-change Exact runner. Maximum deltas against C++:

| Norm | Maximum absolute delta | Maximum relative delta |
|---|---:|---:|
| max | 4.76837158203125e-7 | 1.5128295510073177e-7 |
| 3 | 5.475178621594523e-8 | 3.6318890071157644e-8 |

The frozen C++ CI bound stays 1e-6. Its Fast negative control fails that bound:
the 16x16 fixture has max-norm error 1.0490424279696597e-5. No real-image
fixture was needed. Frozen Fast output bits cover 16x16, 17x19 and 65x49 on
x86-64 AVX2/FMA; Exact score/map fingerprints cover 7x9, 16x16 and 65x49 at
all available tiers and in native/cross CI. Native x86 exercised seven token
permutations and ARM64/QEMU exercised 25. All sRGB reference ingresses,
strided planar `*_into`, Scorer and strip tests compare both norms and map bits,
including zero scores for identical images. One-shot recomputation and cached
Exact reference masks agree bit-for-bit.

## Speed

[zenbench results](speed.tsv): seven measured rounds per arm, 500 ms warmup,
one call per round. Resident CID22 RGB8 inputs; decoding excluded, result
destruction included. Default means Rayon defaults within 28 CPUs of affinity.
The before/after builds have separate Cargo targets and executable hashes in
[provenance](provenance.json). Raw JSON's automatic git field identifies the
launching checkout; the explicit before/after source pins above and in provenance
identify the implementations. These are descriptive shared-host medians;
zenbench reported noisy rounds but no run marked unreliable. Fast's timing
variation is reported without attributing an improvement to it.

| Size | Threads | Before Fast ms | After Fast ms | Before Exact ms | After Exact ms |
|---|---|---:|---:|---:|---:|
| 512x512 | 1 | 44.372272 | 44.644016 | 165.020543 | 135.591693 |
| 512x512 | default | 26.213091 | 20.116496 | 79.101873 | 64.669450 |
| 2048x2048 | 1 | 1083.718437 | 1078.813457 | 2868.079856 | 2372.012462 |
| 2048x2048 | default | 527.684582 | 536.002881 | 1093.887431 | 907.986304 |

Exact's median improves in all four measured cases; the optimization is retained.
Speed is reported through zenbench. No ad-hoc stage timers were added.
[Fixed-count perf counters](perf.tsv) measure one Exact planar reference plus
three comparisons at 2048x2048/one thread: instructions 106,147,638,044 →
92,981,768,191; cycles 29,683,581,450 → 23,312,119,204. These runs are separate
from the speed samples and include input setup.

## Memory

[GNU time -v measurements](memory.tsv), each in its own process: 2048x2048
synthetic planar input, one reference and three sequential comparisons. RSS
includes caller planes, output and allocator retention. Both score norms match
the before implementation bit-for-bit in all four workloads.

| Mode | Threads | Before peak RSS KiB | After peak RSS KiB |
|---|---|---:|---:|
| Fast | 1 | 692000 | 691680 |
| Fast | default | 917768 | 929680 |
| Exact | 1 | 781716 | 740980 |
| Exact | default | 925848 | 892824 |

Both modes retain 251,658,240 bytes of precomputed psycho/mask planes at this
size, exactly matching `estimated_reference_bytes`. `memory_bytes` additionally
counts the source and idle pool actually retained. Exact's conservative planar
peak estimate changes from 1,055,333,632 to 1,119,136,000 bytes to cover additional
transposed FIR/subsample scratch; Fast's estimate stays 1,055,333,632. This is an
image-buffer admission estimate excluding caller storage and allocator overhead,
not a process RSS ceiling. Planar Exact no longer allocates a full-size 12 B/px
interleave; its half-scale input is still interleaved.

## Reproduction and gates

From the workspace root, with TMPDIR on disk, all heavy steps under run-heavy:

- `just libjxl-exact-round3-check`: workspace tests/clippy/fmt, all six CI feature
  combinations with tests/clippy/docs, no-default-features and ARM64/QEMU Exact.
- `just libjxl-exact-round3-measure`: 19-pair strict bit comparison, isolated
  before/after zenbench builds, separate GNU time/perf runs, fresh C++ oracle capture.
  It expects a fresh `round3-before/zen-source` directory and preserves existing
  captures rather than replacing them.
- `just libjxl-exact-round3-summary`: small TSVs from the retained raw artifacts.

The oracle generator now rejects missing source instrumentation and low-precision
binary output. Its missing-source negative probe passed; the instrumented capture
exactly reproduced all six committed rows. Generated inputs use unique directories.
The earlier measurement logs use `$HOME` for the home-directory prefix.

Full local gate:
`run-heavy: done rc=0 1285s | peak-RSS 0.42GiB | min-avail 51587MiB | peak-load 6.78`.
Full measurement:
`run-heavy: done rc=0 230s | peak-RSS 1.29GiB | min-avail 51072MiB | peak-load 6.21`.
The measurement harness example also passes clippy with warnings denied.
Raw logs, JSON, RSS reports, counters, maps and executables remain off git;
[artifact pointers and hashes](artifacts.json) retain their locations on r5900xt.
