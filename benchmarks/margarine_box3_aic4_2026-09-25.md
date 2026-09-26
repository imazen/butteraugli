# Margarine box3: first AIC4_sample evaluation

This is an evaluation of an analytical box-filter control, not a Margarine
acceptance result. CID22 and the other required corpora have not run. Resource
targets, matched-budget encoder regret, clustered uncertainty and corruption
panels remain unverified.

Code: candidate `fac7b76606f8`, serial evaluator `63cfecfd2145`.
Host: ARM Mac, `RAYON_NUM_THREADS=2`, release build, runtime SIMD dispatch,
default FIR teacher with full and half resolution. Both arms use the same
untagged RGB8 PNGs interpreted as sRGB. No fitted coefficients. The candidate
keeps all scoring stages and replaces general Gaussians with three boxes.

All 300 published pairs were evaluated against original reconstructed JND
labels: five sources, six codecs, ten distortion levels. All 305 distinct
images were verified as 620×800 RGB8 with no color-profile/transfer tags.
Original labels and image hashes are preserved by the manifest tools.

| Pooling | Teacher SROCC | Box3 SROCC | Strict within-source reversals / 8,850 |
|---|---:|---:|---:|
| max | 0.865182 | 0.868484 | 111 |
| p1 | 0.893733 | 0.896122 | 17 |
| p2 | 0.898222 | 0.900860 | 28 |
| p3 | 0.896924 | 0.899443 | 42 |
| p6 | 0.889277 | 0.892481 | 49 |

The [full six-stat panel](margarine_box3_aic4_2026-09-25.tsv) contains corpus,
codec and source rows. These small pooled differences are not a statistical
non-inferiority claim. All pair orderings above use zero teacher tie epsilon;
they are not material-choice failure rates. There were no new ties.

The largest reversed teacher gap was 0.286935 in max pooling, between
`PTC_00007_JPEG-XL_08.png` and `PTC_00007_JPEG-XL_09.png`. At p3 it was
0.0627192 between `PTC_00006_AVIF_10.png` and `PTC_00006_JPEG-AI_10.png`.
No bitrate data enters these comparisons, so they cannot establish matched-byte
encoder-choice preservation.

## Artifacts and reproduction

Mac artifact root:
`/Users/lilith/work/codec-artifacts/margarine/box3-aic4-2026-09-25/`.
`cells.jsonl` SHA-256:
`3fcc5e2e763ca8a34404457b1d7da576e16c82a99d1e2ceb0f99c546cc860d4e`.
The JSON manifest records input and binary hashes. Every pair retains all five
norms and pointers to both native f32 diffmaps, named by map SHA-256.
Tower mirror: `/mnt/user/coefficient/output/margarine/box3-aic4-2026-09-25/`.
The cell ledger hash matched after copying, and three deterministically
sampled map hashes matched their content-addressed names (600 maps present).
No cloud mirror was made.

```sh
RAYON_NUM_THREADS=2 TMPDIR="$HOME/tmp" nice -n 19 python3 \
  experiments/margarine/score_manifest.py \
  /Users/lilith/work/codec-artifacts/margarine/aic4-sample-audit-2026-09-25/pairs.tsv \
  experiments/margarine/target/release NEW_OUTPUT --build-commit 63cfecfd2145
```

The first Mac timing attempt was terminated when zenbench detected another
agent's decoder benchmark. After that process finished, a fresh run completed
30 rounds: teacher 32.79 ms, box3 47.84 ms (paired 95% slowdown interval
44.1%–47.8%). Box3 fails the 4× speed target on this measured pair.
JSON and logs: `../box3-resources-2026-09-25/aic4-native-quiet.{json,log}`
relative to the artifact root above. Timing harness: `c11d287423e3`.

Fresh-process `/usr/bin/time -v` on r5900xt (Zen 3, two Rayon threads) measured
102,068 KiB peak RSS for teacher and 84,012 KiB for box3 on the same pair.
This includes decoding and caller buffers; neither figure is metric-only
working memory. Box3 fails the quarter-RAM target. The resource wrapper used
a 12 GiB cgroup cap and recorded minimum available system RAM of 58,446 MiB
and 58,375 MiB respectively. The Zen 3 timing attempts collected only four
rounds due to benchmark-process gating and are not accepted speed evidence.
