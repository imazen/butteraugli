# Published-version and cache audit (2026-10-12)

Butteraugli scores did not change from published 0.9.3 in the measured cases.
Reused-reference throughput did regress. The retained scratch-cache limit
changed from 48 to 8; a bounded policy that retains larger planes reduces
that regression but does not remove it.

## Scope and provenance

- Published library: crates.io 0.9.3, source `6e3a7bd27ef247e6bc285d9473a691facdb984f8`.
- Original current metric: `795aa3727252dc56840501864aefc4a8a537c8ea`;
  benchmark harness commit `feb9f5851fb7f77c8ed019e8aff5b55b07ed14a6`.
- Follow-up harness: `9aae614d0ab530eb60eb5b40f794e2e858fb046a`.
  Bounded-policy diagnostic changes only `butteraugli/src/image.rs`;
  SHA-256 `c5044075c6e34cd2d9522eb9717a97e5d8cc62432905a4e25b08cbd5a7fd49b6`.
- CPU: Core Ultra 7 265K (20 logical CPUs); rustc 1.99.0.
- Both versions enable default Rayon and AVX-512 runtime dispatch; this CPU
  executes AVX2. Each case uses the same explicit 1-thread or 8-thread pool.
  No `target-cpu=native`; workspace release profile, thin LTO.
- Same dependency graph: archmage 0.9.28, magetypes 0.9.26, zenbench 0.1.7.
- Input: GB82 reference photograph, PNG SHA-256
  `9d41d594b2e4dcc6de58f56816aa737cf84109444725f5c36b84e6987d2f016c`.
  Resize with Triangle to each reported size, JPEG quality 75 roundtrip.
  The three sizes reuse one photograph; they are not three independent photos.
- Both implementations receive the same prelinearized planar f32 samples,
  80-nit intensity, hf_asymmetry/xmul 1, multiresolution enabled.
  Decode, distortion generation, linearization and persistent-reference setup
  are outside the timer. `cold` times construction plus comparison; `warm`
  times repeated comparison; `warm-map` also requests output diffmap.
- Paired zenbench rounds, 15 per group, exactly one call per sample, 100ms
  warmup, automatic early stopping disabled. Resource gating is explicitly
  disabled; the quiet host is monitored separately and run-heavy bounds jobs/RAM.
  Percentage effects and 95% intervals are zenbench's paired estimates;
  they need not equal the ratio of rounded unpaired means.

## Original current code versus 0.9.3

Data: [original CSV](version_audit_2026-10-12_i265.csv).

| Case | 0.9.3 ms | Current ms | Change | Paired 95% CI |
| --- | ---: | ---: | ---: | --- |
| 512x512_1t_cold | 32.99 | 33.76 | +2.33% | [+1.50%, +3.21%] |
| 512x512_1t_warm | 13.65 | 14.72 | +8.36% | [+7.24%, +9.38%] |
| 512x512_1t_warm-map | 13.62 | 14.76 | +8.32% | [+6.58%, +9.89%] |
| 512x512_8t_cold | 24.69 | 26.12 | +5.78% | [+4.13%, +7.43%] |
| 512x512_8t_warm | 9.31 | 11.24 | +20.70% | [+17.62%, +23.49%] |
| 512x512_8t_warm-map | 9.19 | 11.02 | +19.85% | [+17.37%, +22.16%] |
| 1920x1080_1t_cold | 235.87 | 235.21 | +0.44% | [-0.15%, +1.13%] |
| 1920x1080_1t_warm | 123.66 | 132.09 | +6.82% | [+5.19%, +8.60%] |
| 1920x1080_1t_warm-map | 122.88 | 130.94 | +6.56% | [+5.29%, +7.83%] |
| 1920x1080_8t_cold | 158.33 | 158.65 | +0.95% | [+0.58%, +1.28%] |
| 1920x1080_8t_warm | 65.12 | 91.55 | +40.57% | [+38.08%, +43.07%] |
| 1920x1080_8t_warm-map | 63.74 | 92.26 | +44.76% | [+38.32%, +51.40%] |
| 3840x2160_1t_cold | 893.39 | 912.73 | +2.16% | [+1.64%, +2.77%] |
| 3840x2160_1t_warm | 564.22 | 587.76 | +4.73% | [+3.43%, +5.98%] |
| 3840x2160_1t_warm-map | 521.24 | 550.52 | +5.62% | [+4.46%, +6.79%] |
| 3840x2160_8t_cold | 686.03 | 691.30 | +0.77% | [+0.17%, +1.37%] |
| 3840x2160_8t_warm | 291.25 | 358.79 | +23.19% | [+21.12%, +25.27%] |
| 3840x2160_8t_warm-map | 291.32 | 376.24 | +29.48% | [+28.66%, +30.30%] |

## Cache-cap diagnostic

At 1080p/8 threads, changing only the current internal cap from 8 back to 48
makes its reused-reference results indistinguishable from 0.9.3 within the
paired intervals. This is a diagnostic, not a proposed release default:
raising the cap would reverse the existing bounded-retention choice.

Data: [48-buffer CSV](version_audit_2026-10-12_cap48.csv).
The isolated eight-buffer/larger-plane probe is retained in
[this CSV](version_audit_2026-10-12_largest8_probe.csv).

## Bounded larger-plane policy versus 0.9.3

Keep the eight-buffer cap and exact-capacity reuse. When the pool is full,
replace a smaller cached plane with a larger incoming plane. This prevents
half-resolution planes from occupying every slot after that scale finishes
first. The maximum buffer count and maximum possible retained bytes stay
within the existing bound; actual retained memory can increase toward that bound.

Data: [full bounded-policy CSV](version_audit_2026-10-12_largest8.csv).

| Case | 0.9.3 ms | Current ms | Change | Paired 95% CI |
| --- | ---: | ---: | ---: | --- |
| 512x512_1t_cold | 33.83 | 34.34 | +1.93% | [+1.11%, +2.83%] |
| 512x512_1t_warm | 14.81 | 15.73 | +6.19% | [+4.68%, +7.75%] |
| 512x512_1t_warm-map | 14.76 | 15.63 | +5.87% | [+4.62%, +7.15%] |
| 512x512_8t_cold | 25.35 | 26.25 | +3.54% | [+2.12%, +4.96%] |
| 512x512_8t_warm | 9.62 | 11.26 | +17.08% | [+14.41%, +19.55%] |
| 512x512_8t_warm-map | 9.49 | 11.17 | +17.69% | [+14.47%, +20.85%] |
| 1920x1080_1t_cold | 271.18 | 270.68 | -0.19% | [-1.27%, +1.02%] |
| 1920x1080_1t_warm | 138.97 | 146.84 | +6.70% | [+6.06%, +7.34%] |
| 1920x1080_1t_warm-map | 128.10 | 133.82 | +4.47% | [+2.88%, +5.99%] |
| 1920x1080_8t_cold | 160.88 | 154.10 | -4.21% | [-5.00%, -3.49%] |
| 1920x1080_8t_warm | 64.07 | 77.09 | +20.33% | [+18.80%, +21.78%] |
| 1920x1080_8t_warm-map | 64.89 | 77.17 | +18.92% | [+17.60%, +20.31%] |
| 3840x2160_1t_cold | 897.57 | 900.29 | +0.30% | [-0.26%, +0.91%] |
| 3840x2160_1t_warm | 527.15 | 546.35 | +3.35% | [+2.69%, +4.06%] |
| 3840x2160_1t_warm-map | 570.31 | 590.63 | +4.12% | [+3.20%, +5.07%] |
| 3840x2160_8t_cold | 666.35 | 626.65 | -5.57% | [-6.47%, -4.79%] |
| 3840x2160_8t_warm | 292.94 | 331.17 | +13.05% | [+11.62%, +14.58%] |
| 3840x2160_8t_warm-map | 296.32 | 327.55 | +10.54% | [+9.58%, +11.53%] |

The remaining eight-thread reused-reference slowdown is 17–20% at 512²/1080p
and 10–13% at 4K on these inputs. The bounded change is a partial fix. At 1080p,
the isolated diffmap-enabled probe fell from the original 92.26ms to 77.77ms;
these are separate runs, each paired against 0.9.3, not a direct paired before/
after comparison. The full follow-up confirms the direction. Large-image
fresh-reference cases improve with eight threads; tiny fresh-reference and
single-thread reused-reference cases still show regressions.

## Scores and C++ fidelity

The frozen `butteraugli_main` table contains 908 distorted synthetic pairs;
all generators resolve, all dimensions are at least 8, and no identity pairs
were counted. 0.9.3 and current return **exactly identical max-norm scores**
on all 908. Mean absolute difference from the captured C++ scores is
0.00006927838760216 for both; maximum relative difference is
0.00006327171671572 (0.006327%); this is not an improvement in C++ agreement.
The larger-plane candidate preserves those exact results.

A separate live C++ probe uses libjxl
`b87738951c1254cd8cccaa6d47712ba735da56d8`, unmodified, with its prebuilt FFI
adapter on Ryzen 9 7900X. Both Rust versions return the same max-norm and
p-norm-3 for all three resized photographic JPEG pairs, in all tested
thread/output settings. Rust/C++ p-norm-3 values are:

| Size | Both Rust versions | libjxl |
| --- | ---: | ---: |
| 512×512 | 0.82708053328770637 | 0.82708049228664615 |
| 1920×1080 | 0.70697447334492425 | 0.70697455254298813 |
| 3840×2160 | 0.72239417935857020 | 0.72239446833749088 |

The cached legacy API returns a diffmap even when `compute_diffmap=false`,
in both versions; the output-presence columns in the score probes confirm it.
The opt-in `linear-planes` scorer follows its explicit diffmap-output setting.
IIR blur is a different metric configuration and was not enabled in these comparisons.
No cross-CPU performance conclusion or whole-corpus quality claim follows
from this experiment.

## Resource records and reproduction

Whole-process RSS includes both versions' persistent references, inputs,
allocator retention and benchmark machinery. It does not measure each
implementation's standalone peak.

- Original full run: `run-heavy: done rc=0 180s | peak-RSS 4.02GiB | min-avail 21899MiB | peak-load 3.05`.
- Cap-48 probe: `run-heavy: done rc=0 9s | peak-RSS 1.03GiB | min-avail 24995MiB | peak-load 1.06`.
- Larger-plane probe: `run-heavy: done rc=0 9s | peak-RSS 0.94GiB | min-avail 24878MiB | peak-load 0.69`.
- Larger-plane full run: `run-heavy: done rc=0 162s | peak-RSS 4.31GiB | min-avail 21420MiB | peak-load 3.09`.
- Live C++ scoring/MSRV build probe: `run-heavy: done rc=0 21s | peak-RSS 4.25GiB | min-avail 18380MiB | peak-load 4.76`.

```sh
# The Rust-only performance build explicitly disables C++ FFI.
BUTTERAUGLI_BENCH_NO_CPP=1 just audit-build
run-heavy --mem 16G --jobs 8 -- just audit-perf /path/to/reference.png 15
run-heavy --mem 16G --jobs 8 -- just audit-perf /path/to/reference.png 15 --group=1920x1080_8t_warm
run-heavy --mem 16G --jobs 8 -- just audit-golden
just audit-extract complete-run.log results.csv
just audit-extract complete-run.log results.md

# Requires a frozen libjxl build, explicitly enabled for live C++ scores.
LIBJXL_DIR=/path/to/libjxl LIBJXL_BUILD_DIR=/path/to/build just audit-build
run-heavy --mem 16G --jobs 8 -- just audit-scores /path/to/reference.png
```
