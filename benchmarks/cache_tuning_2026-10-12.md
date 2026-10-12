# Configurable scratch cache (2026-10-12)

Performance tuning is **deferred at the user's request because the host is
busy**. The paired results below are exploratory records, not recommended
budgets or proof that a throughput regression is resolved. The measured default
was eight buffers; the subsequent automatic policy is described below. No
per-budget standalone RSS measurement was completed.

## Configuration contract

`ButteraugliParams::with_scratch_cache_bytes(bytes)` and the corresponding
`linear_planes::ScorerBuilder` method set retained float-buffer capacity for
whole-image reused references. Zero disables idle scratch retention. An explicit
budget permits up to 48 buffers within the byte limit, allocated as needed.
The automatic default budgets 24 padded planes, capped at 256 MiB per reference.
It is a bounded starting policy, not an optimum validated on the busy host.
An explicit budget overrides automatic sizing, including zero.
It excludes persistent reference data, active working buffers, input/output
pixels, allocator overhead and descriptor bookkeeping. It therefore does **not**
cap peak RSS. One-shot and strip calls ignore the setting.

Reference construction keeps exact-shape reuse and its original eight-buffer
bound. After construction the comparison cache may reuse nearby padded shapes
within a scale, but never a full-resolution plane for a half-resolution request.
Clones preserve the budget with an empty cache; `shrink_to_fit()` drains it.
`estimated_planar_peak_bytes` includes configured storage and construction's
default cache, covering sequential comparisons rather than arbitrary concurrent
calls. Size budgets per image and per concurrently retained reference.

## Exploratory records and provenance

[Paired budget CSV](cache_tuning_2026-10-12_paired.csv) measures review source
`8bd04e2faf773c748b4da5d560659e82959e27ef` against published 0.9.3, source
`6e3a7bd27ef247e6bc285d9473a691facdb984f8`. This is the source in
[PR 17](https://github.com/imazen/butteraugli/pull/17), not remote main.
Same Core Ultra 7 265K, Rust 1.99.0, runtime AVX2, thin LTO, no native-CPU
flags, and [photograph/hash/distortion preparation](version_audit_2026-10-12.md#scope-and-provenance)
as the original audit. Each budget runs the six eight-thread warm/warm-map
cases at 512×512, 1920×1080 and 3840×2160. Scores and p-norm-3 agree exactly
with 0.9.3. Paired zenbench: 15 rounds, one call per sample, 100ms warmup,
no early stopping. **The resource gate was disabled.** Background work and
changing host load mean the confidence intervals do not establish reliable
production tuning; no automatic budget policy was selected.

The [older 24-buffer diagnostic CSV](cache_tuning_2026-10-12_cap24.csv) records
an earlier count-based experiment, not the public byte-budget API. Its log
reports parent `dfd29d26ca53efa4071a67da9e7eae8a2f7ff17d` while code was dirty.
The preserved subsequent diagnostic is `34433419b892831adfc28c23668ca3311b2fd647`;
it is not an immutable source identification for every timed row. Keep this
artifact as an exploratory record, not a release/performance gate. Command:
`just audit-perf IMAGE 15`. Wrapper: `run-heavy: done rc=0 155s | peak-RSS
5.62GiB | min-avail 20629MiB | peak-load 1.89`.

The `default` rows below measured the former eight-buffer policy, not the
subsequent adaptive default. Whole-process records include **both** Rust versions, setup and all three
image sizes. These are not per-version or per-size peak RSS values:

- default MiB: `run-heavy: done rc=0 39s | peak-RSS 3.78GiB | min-avail 20505MiB | peak-load 3.42`
- 0 MiB: `run-heavy: done rc=0 36s | peak-RSS 3.52GiB | min-avail 22103MiB | peak-load 4.92`
- 32 MiB: `run-heavy: done rc=0 36s | peak-RSS 3.68GiB | min-avail 22353MiB | peak-load 4.10`
- 64 MiB: `run-heavy: done rc=0 40s | peak-RSS 3.66GiB | min-avail 20285MiB | peak-load 4.99`
- 128 MiB: `run-heavy: done rc=0 32s | peak-RSS 3.79GiB | min-avail 22207MiB | peak-load 4.71`
- 256 MiB: `run-heavy: done rc=0 39s | peak-RSS 3.90GiB | min-avail 20433MiB | peak-load 7.65`
- 512 MiB: `run-heavy: done rc=0 36s | peak-RSS 4.17GiB | min-avail 21418MiB | peak-load 7.25`
- 1024 MiB: `run-heavy: done rc=0 34s | peak-RSS 4.07GiB | min-avail 21857MiB | peak-load 6.82`

## Reproduction after the host is quiet

Build with `BUTTERAUGLI_BENCH_NO_CPP=1 just audit-build`, under `run-heavy`.
The current harness uses `GateConfig::strict()` by default, prints gate waits,
and refuses CSV output when zenbench flags a run unreliable. `--ungated` is an
explicit diagnostic opt-out. No further optimization runs were started after
the user's deferral.

Run `just audit-perf IMAGE 15 --group=_8t_warm --cache-bytes=BYTES` for each
budget. Measure standalone memory in separate processes with
`/usr/bin/time -v just audit-profile IMAGE current SIZE 8 20 --cache-bytes=BYTES`,
and repeat with `published` for the baseline. `RETAINED` separates persistent
precompute from current reference storage; `/usr/bin/time` reports process
maximum RSS. Keep all original stdout/stderr. Aggregate complete captures with
`just audit-cache-summary LOG_DIRECTORY OUTPUT_PREFIX`; `--paired-only`
archives paired records when independent profiles have not been measured.

Correctness gates compare exact max-norm, p-norm-3 and diffmap pixels across
budgets, strided inputs, repeated distortions, cloned references and thread
counts; they also check retained float-buffer bytes and reclamation. Performance
and memory recommendations still need quiet-host measurements.

## Local validation

The full correctness run passed: library/CLI all targets, doctests,
`linear-planes`, IIR, unsafe cache behavior, no-default-feature unit/cache tests,
strict workspace Clippy, Rust 1.89 locked library/CLI checks, API snapshot checks
and all 223 default-feature semver checks against 0.9.3. Captured C++ parity
covered 908 distorted pairs with exactly zero max-norm delta from 0.9.3.
The existing external-corpus tests were not run. Wrapper:
`run-heavy: done rc=0 243s | peak-RSS 0.35GiB | min-avail 22136MiB | peak-load 7.09`.

Clone descriptor preallocation passed separate safe and unsafe cache tests,
strict Clippy, locked MSRV and API snapshot checks. It keeps 48 descriptors
from growing to 64 and preserves the admission metadata bound. Wrapper:
`run-heavy: done rc=0 70s | peak-RSS 0.30GiB | min-avail 23922MiB | peak-load 2.99`.
The review PR's CI is the source of current platform-validation status.

## Adaptive-default follow-up

At the user's request, the automatic cache now budgets 24 padded planes, capped
at 256 MiB per reference. Padded plane bytes use the maximum of normal and
transposed layouts, four-pixel Malta borders, and 16-float row alignment. The
same calculation feeds reference construction and admission accounting. This
reserves no full budget up front: float buffers enter the cache only on use.
For an image whose scratch plane exceeds the cap, that plane is not retained.
Explicit budgets, including zero, retain their existing meaning. Construction
still uses eight exact-shape buffers, and each active scale keeps its own size.

The original eight-plane retention test was updated with explicit user approval
to enforce the adaptive byte bound. Exact score/pixel assertions stay unchanged.
No new timing measurements were run for this policy on the busy host.

Adaptive-default validation passed safe/unsafe exact cache parity, the approved
retention bound, memory admission tests, locked Rust 1.89, full library/CLI
all-target tests, doctests, planar/IIR/no-default tests, 908 captured-score
comparisons and 223 default-feature semver checks. The complete command ended
at the generated API inventory check, which omitted the unchanged `RGB` export:
`run-heavy: done rc=101 315s | peak-RSS 0.46GiB | min-avail 20186MiB | peak-load 33.88`.
Targeted cache/MSRV run: `run-heavy: done rc=0 101s | peak-RSS 0.25GiB | min-avail 20640MiB | peak-load 26.42`.
A direct generic `RGB` API usage test passes independently of the inventory
exporter. All-feature semver checking still reports only the three pre-existing
removed `internals` constants described in the release audit; this remains a
release decision, not a new cache/default break.
