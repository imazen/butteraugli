# Margarine research

Margarine's target is Butteraugli-like quality rankings and encoder choices at
no more than one quarter of Butteraugli's runtime and peak memory. No candidate
has met those requirements yet. This directory is an unpublished experiment;
it does not change Butteraugli's API or arithmetic.

The first control scores linear-light 2×2 averages with Butteraugli. This is
deliberately a control, not the proposed finished metric: averaging can erase
fine artifacts. Its tests include such a counterexample. A useful candidate
must retain a full-resolution fine-detail signal while reducing storage and
work in the coarse frequency paths. Streaming intermediates and a smaller
directional-filter bank are hypotheses to test, not measured wins.

## Evaluation contract

Compare against this repository's FIR, multiresolution Butteraugli with the
same decoded pixels, intensity target, thread budget, and runtime dispatch.
Evaluate max and libjxl's mixed p/2p/4p pooling separately (p=1,2,3,6). Plain
Lp means are not substitutes. Persist raw pair scores and diffmaps before
aggregation; preserve codec, source, dimensions, configuration, and label
provenance. Never fit coefficients or select candidates using CID22 validation
or AIC holdouts. Split development data by source, including crop/resize and
cross-corpus duplicates.

Use the full six-stat `zenstats::compute_panel`: SROCC, PLCC, KROCC, OR, PWRC,
and Z-RMSE. Retain signed rank correlations beside its polarity-tolerant panel.
Report each dataset, codec, source, and quality band, including sample counts
and unavailable/degenerate cells. Compare candidate and teacher on identical
rows. Do not mistake a confidence interval containing zero for evidence of
non-inferiority: acceptable accuracy margins remain to be fixed before a
ship decision. Source-cluster resampling is needed for uncertainty across
independent contents; zenstats' existing bootstrap resamples rows instead.

The second panel checks encoder behavior: within-source candidate ordering,
material inversions, ties/dead zones, and teacher regret at matched byte budgets
or target quality. The third checks raw tails, structural corruptions, and
local-edit/diffmap agreement with actual score changes. A high pooled SROCC
does not pass either panel. Zensim's 0–100 dial thresholds do not automatically
apply to a Butteraugli-shaped distance.

Benchmark cold pairs and precomputed-reference comparisons separately, at
64², 256², 1024², and 4096² plus real native dimensions. Use zenbench for
interleaved timing and a fresh process per arm under heaptrack or platform
`time` peak-RSS measurement. Report full process peak and metric working memory
separately, including input/output policy; retained-buffer accounting is not a
peak measurement. A quarter-sized image is not evidence of either 4× target.

Any fitted approximation needs the four-axis sweep in the project instructions:
size, quality (including aggressive compression), modes, and content. Extend
through the existing zenfleet/zenmetrics job system for distributed work.

## Dataset identities

Inventory the available dataset versions before claiming “all datasets.” The
initial required set is CID22 validation, AIC-3, the older AIC-4 sample, full
AIC2026, KonJND, LIVE, CSIQ, KADID, TID, and non-photo/codec-ladder/corruption
instruments. Record which corpora were used to develop each candidate.

The current zenmetrics guide distinguishes AIC2026 curve-reconstructed human
targets from metric-derived `JND_*` columns. It also distinguishes the submitted
`proposal-Butteraugli` column from libjxl Butteraugli. Recompute the latter as
the teacher; do not use a metric-derived JND column as human ground truth.
Verify the raw dataset and split metadata before evaluating. The older 300-pair
AIC-4 sample and its JPEG-AI-SDR25 subset are not independent datasets.

## Sources inspected

- [zenstats README](https://github.com/imazen/zenmetrics/blob/0e44a266addae8aecfb02297214a8c491cce4967/crates/zenstats/README.md)
  and [panel implementation](https://github.com/imazen/zenmetrics/blob/0e44a266addae8aecfb02297214a8c491cce4967/crates/zenstats/src/panel.rs).
- [AIC2026 metric and fitting guide](https://github.com/imazen/zenmetrics/blob/0e44a266addae8aecfb02297214a8c491cce4967/docs/AIC2026_METRICS_AND_FITTING.md).
- [Zensim evaluation requirements](https://github.com/imazen/zensim/blob/9c0635f1/docs/EVAL_PANEL_REQUIREMENT.md),
  [full evaluation](https://github.com/imazen/zensim/blob/9c0635f1/docs/FULL_EVAL.md),
  and [data splits](https://github.com/imazen/zensim/blob/9c0635f1/docs/DATA_SPLITS.md).
- Butteraugli [frequency separation](../../butteraugli/src/psycho.rs),
  [six Malta calls](../../butteraugli/src/diff.rs), and
  [pooling](../../butteraugli/src/lib.rs).

The pinned zenstats dependency avoids consuming concurrent local zensim work.
This lab's PNG/JPEG reader assumes sRGB content; it is not a color-managed
corpus ingress. Scoring official corpora requires verified common ingress,
including bit depth and profiles, before calling the metric.

## Running the initial instruments

`just margarine-check` runs the control and evaluation-integrity tests and
clippy. On the Mac, the Linux `run-heavy` wrapper requires unavailable
`/proc`/systemd/ionice facilities; the recipe uses nice and two build workers.
Set `TMPDIR` to a directory under `~/tmp` and `RAYON_NUM_THREADS` explicitly.
On Linux run heavy commands through the shared `run-heavy` wrapper.

```sh
cargo run --release --manifest-path experiments/margarine/Cargo.toml \
  --bin margarine-score -- teacher ref.png dist.png teacher.f32le
cargo run --release --manifest-path experiments/margarine/Cargo.toml \
  --bin margarine-score -- half-control ref.png dist.png half.f32le
```

Both write raw little-endian f32 diffmaps (dimensions are in stdout) and print
max/p1/p2/p3/p6 scores. Each arm runs separately; native diffmap dimensions
differ. Never relabel the coarse map as a full-resolution localization map.
Files are created exclusively, so reruns cannot overwrite earlier results.
These invocations are scoring tools, not a timing benchmark.

`margarine-eval` reads aligned, tab-separated rows with this exact header:

```text
dataset\tsource\tcodec\tpair\ttarget\tdirection\tteacher\tcandidate
```

Use real tabs. `direction` is `quality` or `distortion` for the human target;
teacher and candidate are nonnegative distances. Use a separate file for each
pooling variant. Do not combine max and p3 as if they were one metric.

```sh
cargo run --release --manifest-path experiments/margarine/Cargo.toml \
  --bin margarine-eval -- scores.tsv panel.tsv 0
```

The last argument declares the teacher's tie epsilon; zero counts strict
disagreements. The initial report contains corpus/codec/source panels and
within-source cross-codec order counts. It explicitly marks the other contract
panels and uncertainty as not measured. Progress goes to stderr: capture both
stdout and stderr in a persistent log. No ship verdict is produced.
