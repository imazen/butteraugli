# Butteraugli release audit (2026-10-12)

Published library and CLI are **0.9.3**. Remote main prepares **0.9.4**, which
has not been published. This audit does not authorize a release.

The initial clean checkout was behind remote commit
`795aa3727252dc56840501864aefc4a8a537c8ea` and was synced through jj.
Other checkouts were inspected for ongoing work; two had unrelated uncommitted
changes, which were left untouched. The quiet benchmark checkout was clean
before its owned diagnostic work began.

## API and defaults

`cargo semver-checks` 0.49.0, using Rust 1.98.1, checks published 0.9.3:

- Default features: **223 checks pass**, no version break required.
- All features: one failing check identifies three removed `internals` constants:
  `consts::XYB_OPSIN_ABSORBANCE_MATRIX`, `consts::XYB_OPSIN_ABSORBANCE_BIAS`,
  `consts::XYB_NEG_OPSIN_ABSORBANCE_BIAS_CBRT`.
  That exposed surface is documented as unstable, but consumers using the
  constants still fail to compile. The compatibility checker asks for a
  breaking release. Restoring deprecated constants would preserve compatibility;
  doing so needs public-API approval and a review PR. Accepting the unstable
  feature break in 0.9.4 also needs a deliberate release decision.
- `source_linear_rgb()` is a public, doc-hidden behavior change: sRGB-built
  references now return `None` where 0.9.3 returned `Some`. The replacement
  owned helper materializes linear RGB on demand. This saves retained memory
  but is not visible to a signature-only compatibility check.
- Additions include cancellation entry points, reference-memory introspection
  and the opt-in `linear-planes` API. Existing supported signatures remain.

Both versions' parameter defaults are hf_asymmetry=1, xmul=1, intensity=80 nits,
multiresolution enabled, and diffmap output disabled. Both enable Rayon and
AVX-512 runtime dispatch by default. `iir-blur` stays off; enabling it changes
metric scores and is not a like-for-like C++ configuration.

The legacy `ButteraugliReference::compare_*` still returns a diffmap even with
`compute_diffmap=false`, in both measured versions. The typed `linear-planes`
scorer honors its output flag. That distinction matters when migrating a
reused-reference caller.

## Scoring and performance

[Full measurements and reproduction](../benchmarks/version_audit_2026-10-12.md).
Both Rust versions return exactly identical max-norm scores on **908 distorted
synthetic pairs** captured from libjxl. They also agree exactly on max-norm and
p-norm-3 for the three resized photographic JPEG pairs measured here. Current
code is not measurably closer to C++ on these cases; its agreement is unchanged.
Live libjxl p-norm-3 differences stay below 0.000000289 on the three pairs.
This is one photograph at three sizes, not a whole-corpus claim.

Performance **did regress for reused references**. The original eight-buffer
pool measured +40.6–44.8% at 1080p/eight threads versus 0.9.3. Restoring 48
buffers in a diagnostic removes that gap. The landed fix retains larger
planes within the existing eight-buffer bound and reduces the regression to
+18.9–20.3% at 1080p and +10.5–13.1% at 4K. This is a partial fix; the default
memory-retention limit still trades throughput for a bounded cache. Fresh
large-image comparisons show much smaller differences, and the bounded
policy improves the eight-thread large-image fresh path.

The bounded fix preserves the eight-buffer count and existing padded memory
bound. Its regression test compares exact scores and diffmap pixels over
repeated distortions with one and four threads. The user approved correcting
the new test's budget to include existing Malta borders and transposed blur;
the production memory bound did not change.

## Validation and remaining release work

With the bounded fix, default library/CLI all-target tests, doctests,
`linear-planes` tests, IIR and no-default-feature unit tests, strict workspace
Clippy, formatting and locked Rust 1.89 library and CLI checks passed. The
opt-in external-corpus suite was not run. Captured C++ parity and live photographic probes passed. The complete
benchmark includes paired confidence intervals and whole-process run-heavy
peak-RSS records; it does not measure each backend's standalone memory peak.

Release automation previously swallowed both crate publication failures;
commit `8ff0f95c` makes those failures stop the workflow. Commit `9a26cdfe` also requires green CI on the exact release source commit,
checks out the requested tag when manually dispatched, restores library/CLI
MSRV CI and native Linux ARM coverage, and updates runner/action versions
verified against current releases. Unique-key workflow YAML, release-guard
shell syntax and the live CI-selection query were validated. Locked library
and CLI checks passed with Rust 1.89; the new remote matrix must still finish.

Before publishing:

1. Resolve the three-constant compatibility decision and hidden-helper behavior.
2. Decide whether the remaining reused-reference regression is acceptable.
3. Complete library/CLI MSRV and all feature/platform CI on the final commit,
   including Windows ARM, Intel macOS and i686. No Butteraugli Intel waiver
   was requested in this audit.
4. Review [README](../README.md), [crate README](../README.crates.md), and
   [changelog](../CHANGELOG.md); consolidate unreleased entries under the final
   version/date. Obtain explicit release approval.
5. After exact-commit CI is green, tag, create the GitHub release, publish the
   library, wait for registry availability, then publish the CLI and verify
   both registry source revisions.

Neither a release tag nor a publication was created by this audit.

## Configurable cache follow-up

[PR 17](https://github.com/imazen/butteraugli/pull/17) proposes an explicit idle
scratch byte budget for reused references and the typed whole-image scorer.
The follow-up automatic default budgets 24 padded planes, capped at 256 MiB
per reference; the byte limit is not a peak-RSS cap.
[Configuration contract and exploratory data](../benchmarks/cache_tuning_2026-10-12.md).
The user deferred performance tuning because the host is busy. This does not
resolve the remaining throughput release decision or authorize publication.
