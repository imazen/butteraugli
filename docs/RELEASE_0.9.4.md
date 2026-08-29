# Release 0.9.4 — checklist and maintainer steps

`0.9.4` has been prepared but **not** released. Everything a non-maintainer can
do is done and verified; the four steps that need the maintainer (tag, GitHub
release, publish, consumer unpinning) are in §3.

Prepared 2026-08-28 against `main`. Latest published version: **0.9.3**
(crates.io + GitHub release `v0.9.3`). The in-tree workspace version has been
`0.9.4` since 3e41f32 and has never been published.

---

## 1. What ships in 0.9.4

See `CHANGELOG.md` `## [0.9.4] - unreleased` for the full list. The
release-shaped summary:

- **New:** the `linear-planes` cargo feature — `butteraugli::linear_planes`, a
  documented supported API for planar linear-light `f32` input. Off by default.
- **New:** `ButteraugliReference::compare_linear_planar_with_stop` (additive to
  the default surface).
- **Fixed:** the 0.9.3 `iir-blur` stride regression (ae50608), a +18% warm-ref
  peak-heap regression at 16–40 MP (3e41f32), and clippy/rustdoc on current
  stable across architectures.
- **Behaviour change to the `internals` surface only:** three `consts::XYB_*`
  items removed in c645a39. See §2.3.

## 2. Verification — done

Every item below was run on `main` at commit `717c911c` or later, on
aarch64-apple-darwin unless stated.

### 2.1 Tests

| what | command | result |
|---|---|---|
| default suite | `cargo test -p butteraugli` | 129 tests, 0 failures |
| suite with `linear-planes` | `cargo test -p butteraugli --features linear-planes` | 162 tests, 0 failures (debug and `--release`) |
| every gated combination | `cargo test -p butteraugli --features <internals \| unsafe-performance \| linear-planes,internals,unsafe-performance>` | 0 failures in each |
| doctests | `cargo test -p butteraugli --doc --features linear-planes` | 6 passed |
| `linear-planes` unit tests | `cargo test -p butteraugli --lib --features linear-planes` | 98 passed (12 new) |
| `linear-planes` parity | `cargo test -p butteraugli --features linear-planes --test linear_planes_parity` | 18 passed |

The parity suite is the one that matters for this release: it asserts, with
`assert_eq!` on `f64` and no tolerance, that every `linear_planes` mode
reproduces the corresponding default-API call bit for bit — max-norm, 3-norm,
and the diffmap pixel for pixel — across default params, padded stride,
single-scale, HDR intensity target, and both strip variants. It holds in
`--release` as well as debug, so the bit-identity is not an artifact of
unoptimised codegen.

One assertion in that file carries a tolerance, and only one:
`strip_walk_agrees_with_whole_image_walk` compares strip-mode against
whole-image mode *within* this API. The max-norm is still asserted with `==`
(max is exact and order-independent); the 3-norm gets `< 1e-9` relative,
because the strip walker's per-strip reduction associates the f64 sums
differently. Measured worst case across 64x128/128x256/256x512 at
16/32/64-row strips: `6e-12`.

Throughput: `Scorer` is within noise of the equivalent default-API calls
(-0.7% at 1024x1024, 8 iterations, release build, aarch64-apple-darwin).

### 2.2 Lint / docs

| what | command | result |
|---|---|---|
| clippy, all targets, aarch64 | `cargo clippy --workspace --all-targets` | clean |
| clippy, lib, x86_64 cross-check | `cargo clippy -p butteraugli --lib --features internals,linear-planes,unsafe-performance --target x86_64-unknown-linux-gnu` | clean |
| rustdoc, default | `RUSTDOCFLAGS=-D warnings cargo doc --workspace --no-deps` | clean |
| rustdoc, gated | `RUSTDOCFLAGS=-D warnings cargo doc -p butteraugli --no-deps --features linear-planes,internals,unsafe-performance` | clean |
| API snapshots | `just api-doc` (`cargo test --manifest-path apidoc/Cargo.toml`) | regenerated, committed |
| API snapshots current | `just api-doc-check` (`ZEN_API_DOC=check`) | pass |
| MSRV (README badge claims 1.89) | `cargo +1.89 build -p butteraugli --features linear-planes` | builds |

CI has no MSRV job — it was disabled because the `image` crate's dev-dependency
chain carries broken `rust-version` metadata (`zune-jpeg 0.5.8` claims 1.87).
The library itself, including the new `linear-planes` module, does build on
1.89, so the badge is accurate for consumers; only the dev/test graph is what
blocks a CI lane.

The snapshots in `docs/public-api/` were stale before this pass — they predated
4e78d6d and were missing `butteraugli_linear_strip_with_stop`. They are current
now. `just api-doc-check` verifies.

### 2.3 `cargo semver-checks` vs 0.9.3

Two runs, and the difference between them is the thing to read carefully:

```
$ cargo semver-checks check-release -p butteraugli --baseline-version 0.9.3 --default-features
  Checked 196 checks: 196 pass, 57 skip
  Summary no semver update required
```

```
$ cargo semver-checks check-release -p butteraugli --baseline-version 0.9.3
  Checked 196 checks: 195 pass, 1 fail, 0 warn, 57 skip
  --- failure pub_module_level_const_missing ---
    XYB_NEG_OPSIN_ABSORBANCE_BIAS_CBRT   consts.rs:28
    XYB_OPSIN_ABSORBANCE_BIAS            consts.rs:24
    XYB_OPSIN_ABSORBANCE_MATRIX          consts.rs:11
  Summary semver requires new major version: 1 major and 0 minor checks failed
```

**The default surface is fully compatible with 0.9.3.** The single failure is
in the `internals` feature surface, from c645a39 (deletion of the dead
codec-XYB module, which took three `pub` consts in `consts.rs` with it) — work
that predates this release-prep pass. The CHANGELOG entry for c645a39 used to
claim "no public API impact"; that claim was wrong about the consts and has
been corrected.

**Maintainer decision required.** Three options, in order of what this prep
recommends:

1. **Ship as-is.** `internals` is documented as unstable with no compatibility
   promise, and the 2026-08-28 audit of every repo that path-patches this crate
   (`jxl-encoder`, `zenmetrics`, `zenpipe`) found **zero** external consumers of
   it — the only `features = ["internals"]` anywhere is this repo's own
   `butteraugli-bench` (`docs/MIGRATION_0.9.4.md` §1.1). Cost: a caller who
   *did* pin `internals` and use those consts gets a compile error on a patch
   bump. Nobody is known to be that caller.
2. **Restore the three consts under `internals`, `#[deprecated]`.** Makes
   semver-checks green with all features. Cost: three dead consts describing a
   transform this crate does not implement (they belong to the JPEG XL codec's
   cbrt XYB, not butteraugli's opsin transform), kept alive for no known user.
3. **Release as 0.10.0 instead of 0.9.4.** Honest to the letter of semver.
   Cost: bumps the leading non-zero component for a break in an explicitly
   unstable surface, and churns every consumer's dependency graph —
   which the global "never increase the leading non-zero digit unless the
   break is both real and unavoidable" rule exists to prevent. `zenmetrics`
   already pins `butteraugli = "0.9.4"`, so 0.10.0 also means editing that
   manifest.

### 2.4 Packaging

```
$ cargo publish -p butteraugli --dry-run
  Packaged 18 files, 560.1KiB (114.4KiB compressed)
  Verifying butteraugli v0.9.4 ... Finished
```

`tests/` and `examples/` are excluded from the package by design (3b7afe7), so
`linear_planes_parity.rs` does not ship — the parity guarantee is enforced in
CI, not at the consumer's build.

`docs.rs` metadata is set to build with `features = ["linear-planes"]` so the
new module appears in published docs. `internals` is deliberately left off
there.

### 2.5 CI

`.github/workflows/ci.yml` covers, per the global platform rule:

- `ubuntu-latest`, `ubuntu-22.04`, `macos-latest` (ARM), **`macos-26-intel`**,
  `windows-latest`, **`windows-11-arm`** — full test matrix.
- **`i686-unknown-linux-gnu`**, `armv7-unknown-linux-gnueabihf`,
  `aarch64-unknown-linux-gnu` — build + `--lib` + `cross_arch_parity` +
  (new) `linear_planes_parity` under QEMU.
- `wasm32-unknown-unknown` / `wasm32-wasip1` compile checks, simd128 and scalar.
- Lint (`cargo fmt --check`, `cargo clippy --workspace --all-targets -D
  warnings`), Documentation (`-D warnings`), Code Coverage, Publish Check
  (`cargo publish --dry-run` for both packages).
- **New `Features` matrix**: clippy `-D warnings` + test + `cargo doc -D
  warnings` for `linear-planes`, `internals`, `unsafe-performance`,
  `iir-blur`, and the three-way combination. The old matrix only ever built the
  default feature set, which is how the 0.9.3 `iir-blur` regression shipped
  undetected (CLAUDE.md records the incident).
- **New `linear-planes parity` matrix**: the parity suite on all five OS
  runners.

Confirm green on the release commit before tagging — `gh run list --repo
imazen/butteraugli --limit 1`, then `gh run view <id>` and check every job,
including `windows-11-arm` and `macos-26-intel`, which are the slowest and
therefore the easiest to tag ahead of.

---

## 3. Maintainer steps — NOT done here

These were deliberately not performed. The global rules forbid publishing, and
tagging/releasing is a maintainer decision.

1. **Decide the `internals` semver question** (§2.3). If option 2 or 3, do that
   first; the rest of this list assumes option 1 and version `0.9.4`.

2. **Re-read the READMEs before publishing.** `README.crates.md` is what
   crates.io shows (`readme = "../README.crates.md"`). Confirm the new
   "Planar linear-light input" section reads the way you want it to. The global
   rule is explicit: no publish without the author verifying the README.

3. **Confirm CI is green on the exact commit you will tag**, on every platform
   including `windows-11-arm`, `macos-26-intel` and `i686-unknown-linux-gnu`.
   Do not tag ahead of a running job.

4. **Set the release date in `CHANGELOG.md`.** Change
   `## [0.9.4] - unreleased` to `## [0.9.4] - YYYY-MM-DD`, leave
   `## [Unreleased]` with its `QUEUED BREAKING CHANGES` section in place
   (that section persists across patch releases and only clears when the
   breaking release ships). Commit and push; re-check CI.

5. **Tag, release, then publish — in that order, stopping at any failure:**

   ```bash
   git tag v0.9.4                       # must match the version being published
   git push origin v0.9.4
   gh release create v0.9.4 --title "v0.9.4" --generate-notes
   cargo publish -p butteraugli
   ```

   A crate on crates.io without a GitHub release page is not acceptable; a bare
   tag is not a release. If any step fails, stop and fix before the next.

6. **`butteraugli-cli`** is versioned in lockstep (workspace version `0.9.4`)
   and depends on `butteraugli = "0.9.4"`, so it can only be published *after*
   the library lands on crates.io. CI's "Check CLI packaging" step is
   `continue-on-error: true` for exactly this reason. Decide whether to publish
   it; if so, repeat step 5 for `butteraugli-cli` after the library is live.

7. **Unpin the consumers.** Once 0.9.4 is on crates.io:
   - `zenmetrics/Cargo.toml` — drop the `[patch.crates-io] butteraugli = { path = ... }`
     entry; the workspace dep already says `0.9.4`.
   - `zenmetrics/crates/zenmetrics-cli/Cargo.toml` — `butteraugli = { version = "0.9.2" }`
     → `{ workspace = true }`.
   - `jxl-encoder/Cargo.toml` — drop the `butteraugli = { path = ... }` patch;
     `jxl-encoder/jxl-encoder/Cargo.toml` says `version = "0.9"`, which resolves.
   - `jxl-encoder/zenjxl-tuning-runner/Cargo.toml` — `0.9.2` → `0.9.4`.

   These are edits in *other* repositories and were not made here. The exact
   diffs are in `docs/MIGRATION_0.9.4.md` §4.

8. **Optional, separate from the release:** the `linear_planes` migration for
   `jxl-encoder`'s `CpuButteraugliBackend`, diffed out in
   `docs/MIGRATION_0.9.4.md` §3. Purely optional — 0.9.4 is additive and the
   existing call sites keep working unchanged.

---

## 4. Open item for the maintainer, unrelated to the release gate

`ButteraugliResult::diffmap` is documented as "only present if
`compute_diffmap` was true". That holds for the free functions but **not** for
`ButteraugliReference::compare*`, which returns `Some(diffmap)` unconditionally
(`precompute.rs:1062`, `precompute.rs:1263`). Callers passing
`compute_diffmap = false` to a warm reference pay for a `width * height * 4 B`
diffmap they did not request.

Left untouched: changing the behaviour would silently break any caller relying
on the diffmap being there, and rewording the doc is a maintainer call. Whether
0.9.4 ships with it is independent — it has been true since the warm-reference
API was introduced. `linear_planes::Scorer` honours the documented contract on
its own surface either way.
