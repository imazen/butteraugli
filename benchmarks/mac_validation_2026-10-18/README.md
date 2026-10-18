# Native ARM64 validation — 2026-10-18

MISSING: Fast frozen-main bit validation on ARM64. `fast_outputs_match_frozen_main_bits`
is x86-64-only in `butteraugli/tests/exact_numerics.rs`; it is absent on this host.
The whole workspace test command failed because the CLI harness ignores `CARGO_TARGET_DIR`.
No implementation or test expectations were changed.

Host: mac, Apple M4 Pro, aarch64-apple-darwin, macOS 27.0 (26A428).
Source: `97fc0aec` (PR [21](https://github.com/imazen/butteraugli/pull/21)).
Rust/cargo: 1.99.0. Default native SIMD dispatch is NEON (the ARM64 path in
`blur.rs`/`malta.rs`); Exact's token-permutation test also exercises scalar fallback.
The test reports 25 permutations at each of 7x9, 16x16 and 65x49.
Concurrent host load precludes speed comparisons.

Commands used one owned Cargo target, `CARGO_INCREMENTAL=0`,
`CARGO_PROFILE_DEV_DEBUG=0`, `CARGO_PROFILE_TEST_DEBUG=0`, and
`BUTTERAUGLI_BENCH_NO_CPP=1`. Each ran under the macOS resource wrapper
with a 12 GiB monitored RSS guard and two build/Rayon jobs, at nice 19.
The wrapper is a monitored guard, not a cgroup hard cap.

| command | result |
| --- | --- |
| `cargo test` | exit 101: library and numerical integration tests pass; CLI 0/11 pass because its hard-coded `<repo>/target/debug/butteraugli` does not exist with an external target directory |
| `cargo test -p butteraugli --test exact_numerics -- --nocapture --test-threads=1` | exit 0, 7/7 pass, including pinned C++ oracle, exact pixel/score/pnorm fingerprints, reference/strided/strip ingress parity, Fast negative control |

Exact fingerprints checked on every permutation:

| size | score bits | pnorm-3 bits | diffmap hash |
| --- | --- | --- | --- |
| 7x9 | 3ff918c740000000 | 3ff7ee5c8a0169f1 | 5d82edc20f83be27 |
| 16x16 | 3fff1e3aa0000000 | 3ffe78618ead8a9b | 9c7dabcec8620c81 |
| 65x49 | 4000312080000000 | 3ffe007c790e0907 | bd68558e8f5e1dce |

The full run's `cross_arch_parity` is 2 pass / 1 existing capture-only ignore;
`reference_parity` is 4/4 pass. This does not establish ARM64 Fast frozen bits.
The first workspace attempt was stopped during its optional C++ benchmark adapter
build, then restarted using the repository's documented Rust-only switch.

[Full workspace output](workspace.log), [Exact output](exact.log).
The workspace run's wrapper reports peak-RSS 0.32 GiB; Exact reports 0.35 GiB.
These are BSD time child-command maxima, not aggregate concurrent machine memory.
