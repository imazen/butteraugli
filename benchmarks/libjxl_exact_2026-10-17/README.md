# Opt-in libjxl v0.12 numerics — 2026-10-17

`Numerics::LibjxlExact` selects libjxl-order transfer, opsin and FIR arithmetic.
`Numerics::Fast` remains the default. Set it with `ButteraugliParams::with_numerics`
or `linear_planes::ScorerBuilder::with_numerics`; the params getter is `numerics()`.
The exact mode always uses FIR, including with `iir-blur` enabled. Its small-width
5x5 fallback follows the pinned AVX2 oracle independently of the Rust SIMD tier.
No existing public signatures are changed.

Work branches from investigation commit `779c971a4f280a1eae9807409d32fa412b9f031e`;
default comparisons use main `0e397e5e716cd70ca40605ad93d68df590869dda`.
The independent oracle is libjxl v0.12.0 `a7a9c787341cf703dede03c2009fa460cae5e5df`.
Both implementations use 80 nits. [Input provenance and generation](../libjxl_parity_2026-10-17/README.md)
are retained from Round 1: three CID22 references, JPEG q20/q50/q80, odd sizes,
and a cropped/tiled 2048x2048 boundary probe, plus the original benchmark pair.

Default outputs unchanged (bit-identical on 19 pairs): max-norm, 3-norm and every
logical diffmap pixel match the frozen main runner, checked using raw score bits
and little-endian f32 map bytes. The [complete table](parity.tsv) records every pair.

| Opt-in versus libjxl, 19 pairs | Largest absolute delta | Largest relative delta |
|---|---:|---:|
| Max-norm | 4.76837158203125e-7 | 1.5128295510073177e-7 |
| 3-norm | 5.475178621594523e-8 | 3.6318890071157644e-8 |

The original benchmark pair now matches max-norm exactly (`1.2085671424865723`);
its 3-norm delta is `3.338262111185486e-9`. The whole ladder's maxima are larger
than that pair's error. The mode's validation bound is 1e-6 absolute; the name
identifies an arithmetic policy, not a promise of bitwise identity to every C++
runtime-dispatch target. Small C++ residuals remain in downstream arithmetic.

Timings below are milliseconds, median of seven calls after one warm call, with
RGB8 already decoded and diffmap enabled; map-file writing is excluded. Calls
include allocation, sRGB conversion and both scales. Measurements ran on r5900xt
under nice/ionice and a 24 GiB resource scope, without target-cpu=native. Default
Rayon scheduling runs with affinity limited to 28 allowed CPUs; the one-thread
rows explicitly set RAYON_NUM_THREADS=1. [Raw samples](speed.tsv) are retained.

| Size | Threads | Main baseline | New default Fast | LibjxlExact |
|---|---|---:|---:|---:|
| 512x512 | 1 | 44.495210 | 44.586579 | 171.918979 |
| 512x512 | default | 26.210952 | 21.065985 | 82.465373 |
| 2048x2048 | 1 | 1084.293777 | 1088.238124 | 2935.289233 |
| 2048x2048 | default | 532.603708 | 539.458091 | 1095.225875 |

These descriptive samples do not establish a general throughput change for the
default mode. The small multithreaded case varies more than the single-thread
case. Exact mode costs more CPU time; it has no throughput target.

The isolated measurement's resource receipt is: `run-heavy: done rc=0 112s |
peak-RSS 1.15GiB | min-avail 51295MiB | peak-load 3.47`. Both benchmark executables
are built into distinct Cargo target directories and hashed in
[measurement.meta.json](measurement.meta.json). A shared-target attempt is marked
invalid and retained off git; its rows are excluded. Initial valid measurements
remain in `*-initial.tsv` and `measure.log`.

CI oracle fixtures are generated procedurally by `oracles.py`; their C++ scores
are stored in `butteraugli/tests/data/libjxl_v012_exact.tsv`. Tests keep a 1e-6
absolute bound for both norms. The tier test also fixes score, 3-norm and map
fingerprints. It passed seven x86 permutations and 25 ARM64 permutations under
QEMU on r5900xt, with the same fingerprint. AVX-512 execution is not available on
this host; CI runs the gate wherever that tier is available. Strided RGB,
strided planar input and reusable references are covered as well.

Reproduce with the `libjxl-exact-*` just recipes under the host's resource
supervisor. `build_runners.py` expects an archived main tree at
`~/tmp/bu-libjxl-r2-baseline` with the same timing example minus the new numerics
builder call. The measurement script rejects executable collisions and scores
outside its oracle gate before recording timings. Raw inputs, maps, runner
binaries and logs remain on r5900xt at
`/mnt/v/output/butteraugli/libjxl-parity-2026-10-17/data/round2-final`.

Local validation passed: full workspace `cargo test` (including doctests),
`cargo clippy --all-targets -- -D warnings`, `cargo fmt --check`, exact-mode
integration tests with `linear-planes` and `iir-blur`, and the public-API
snapshot generator. The full gate receipt is `run-heavy: done rc=0 192s |
peak-RSS 0.39GiB | min-avail 51910MiB | peak-load 5.36`.
Complete log: r5900xt `data/round2/logs/bu-libjxl-r2-final-gates.log` under the
artifact root above (sibling of `round2-final`). No test thresholds were relaxed.
