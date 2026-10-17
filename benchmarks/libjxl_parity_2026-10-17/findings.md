# Findings — 2026-10-17

Latest Rust main is `0e397e5e716cd70ca40605ad93d68df590869dda`; the compared pin is `791d08d725f27b5331ef7b96e73aad89ed895f45`. Reference: libjxl v0.12.0, `a7a9c787341cf703dede03c2009fa460cae5e5df`. The reference is statically built from that tag, not the installed libjxl 0.11 library. SDR intensity is explicitly 80 nits in both implementations. No production arithmetic or default outputs change in this branch.

The complete [TSV](parity.tsv) has 38 rows: 19 pairs × two Rust revisions. Latest main and pin return identical max-norm and 3-norm scores on every pair. The 18 generated JPEG pairs stay within the previously observed 7e-6 max-norm drift; the additional original benchmark pair exceeds it. This is an existing numerical parity gap, with no regression between the two checked revisions established.

| Original benchmark pair | Rust, both revisions | libjxl v0.12 | Absolute delta | Relative delta |
|---|---:|---:|---:|---:|
| Max-norm | 1.2085170745849609 | 1.2085671424865723 | 5.0067901611328125e-5 | 4.14274886774728e-5 |
| 3-norm | 0.49497793018194791 | 0.49498148913202938 | 3.558950081472556e-6 | 7.190067021927069e-6 |

Both implementations select max pixel `(511,4)`, near the right/top boundary. Largest full diffmap pixel discrepancy is `1.9383430480957031e-4`. C++ 3-norm is its actual `ComputeDistanceP(p=3)`, printed with 17 significant digits after a printf-only patch. An independent double-precision reduction agrees within 1e-12 on every tested map. The initial 18-pair `latest.tsv` retains the original six-decimal CLI print; use `parity.tsv` for full-precision comparisons.

The [stage table](stages.tsv) traces both full and half scales, both images, linear RGB, opsin/XYB, LF/MF/HF/UHF frequency separation, Malta accumulation, AC differences before/after mask correction, masking and the single-scale diffmap. At full scale, input linear values differ by at most `1.1920928955078125e-7`; XYB differences reach `7.62939453125e-5`; converted LF plane differences reach `0.0018310546875`. Those LF magnitudes are intermediate units, not final score units.

Controlled C++-intermediate substitution localizes the score gap:

| Diagnostic | Max-norm absolute delta | 3-norm absolute delta |
|---|---:|---:|
| Unchanged Rust | 5.0067901611328125e-5 | 3.558950081472556e-6 |
| C++ LF planes in Rust | 1.430511474609375e-6 | 4.58776732048527e-7 |
| Double-precision Gaussian kernel only | 4.684925079345703e-5 | 4.611634514872787e-6 |
| C++ Gaussian kernel and FIR sum order | 2.2530555725097656e-5 | 3.537778450857232e-6 |
| C++ kernel/FIR order plus C++ XYB | 1.1920928955078125e-7 | 1.9016464514187703e-8 |

C++ AC-plane substitution alone leaves almost the entire max-norm gap. XYB substitution alone makes max error larger (`8.022785186767578e-5`), so the effects interact; they cannot be assigned independent percentages. Matching XYB and the FIR arithmetic together leaves one float ULP in max-norm and a 1.90e-8 3-norm residual. This supports preprocessing/frequency arithmetic as the cause of the larger drift. It does not identify one isolated FMA instruction as the sole cause.

The source differences are concrete: [Rust Gaussian kernels](https://github.com/imazen/butteraugli/blob/0e397e5e716cd70ca40605ad93d68df590869dda/butteraugli/src/blur.rs) use f32 scaler/exp calculations and sequential FMA FIR accumulation; [C++ kernels and convolution](https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/butteraugli/butteraugli.cc) use a double scaler/exp and symmetric, four-accumulator summation for the 33-tap LF kernel. Input transfer/opsin arithmetic also differs, as the traced values show. The diagnostic implements the C++ kernel and summation tree in an archived Rust source copy; it is a parity experiment, not a performance-qualified replacement.

[Reduction checks](reductions.tsv) independently reconstruct each implementation's multiresolution merge exactly (zero pixel error), using its own single-scale maps. Rust uses fused f32 blending, while C++ scales by the double expression 0.85 and rounds before adding. That difference is secondary here. Independently reduced max/3-norm values agree with each implementation's reported values. Neither the norm formula nor a missing resolution level explains the 5e-5 gap.

Proposed fix: add an oracle-backed numerical-parity implementation for kernel generation and the 7/13/15/33-tap FIR summation tree in `butteraugli/src/blur.rs`, and align transfer/opsin operations in `butteraugli/src/opsin.rs`. Validate with identical linear inputs and frozen stage maps before optimizing it. Then assess the smaller f32-mask/combination and multiresolution-rounding residual in `diff.rs`. Any adoption in the default path changes existing scores and diffmap pixels; it needs a broader corpus, SIMD-tier checks and performance measurements. No such default change is shipped here, and no test tolerances were relaxed.

[Google-old checks](google-old.tsv) compare zenmetrics master `65f536433214b104536b469eeb33a1f16781529b`'s private metric module against Google C++ `71b18b636b9c7d1ae0c1d3730b85b3c127eb4511`, compiled with FP contraction disabled. Max scores are exactly equal: `1.066243290901184` on the original benchmark pair and `4.843130111694336` on CID22 1025469 q50. This verifies two pairs, not a corpus-wide bound. Google-old has no 3-norm comparison in this investigation. Source URLs are pinned in `google_old.py`.

Validation: scoped rustfmt and clippy for the new Rust example pass; default library tests pass (90 passed, no ignored tests); Python harness syntax checks pass. A local-disk test attempt failed with ENOSPC, then passed with Cargo output on the data volume. Full commands/logs are retained with the artifacts; [resource receipts](resources.txt) include both successful runs and the disk failure. No throughput claims are made.
