# Margarine research

Margarine's target is Butteraugli-like quality rankings and encoder choices at
no more than one quarter of Butteraugli's runtime and peak memory. No candidate
has met those requirements yet. This directory is an unpublished experiment;
it does not change Butteraugli's API or arithmetic.

The memory acceptance gate is **total process peak, including decoding and
caller-owned inputs**, as specified by the user on 2026-09-26. Heap profiles
diagnose allocation costs; a metric-only heap reduction cannot pass this gate.
The user subsequently allowed the resource ratios to taper at smaller sizes,
provided both time and total-process RAM stay below Butteraugli. Report the
measured size curve; retain the 4×/quarter-RAM targets for larger images.

`--bench-rgb8 REF DIST NEW.json` and `--memory-rgb8
teacher|features228|features228-strips REF DIST` compare native RGB8 ingress.
They reject other sample formats. Timing excludes decoding and includes each
metric's sRGB conversion; fresh-process memory includes decoding and inputs.
These remain extractor cost probes with no trained score or quality claim.
The `features168` native probe removes SSIM features and their moment blurs,
retaining the 168 edge, high-frequency energy, XYB MSE and non-SSIM peak
features from the pinned 228 layout. Its active values are tested against the
full extractor, including strided input. This is a feature-cost ablation, not
a fitted model or an accuracy result. The RGB8 benchmark includes it.
`features168-strips64` uses 256-row interiors and a 64-row halo, with outer
strip parallelism. Radius 5 over four scales needs 40 source rows, but the
pinned reference builder pads sub-64-row tails while its distorted strip stays
unpadded. A 40-row halo panicked on height 769; the 64-row halo covers that
minimum as well as the blur support. Tests retain the tight feature agreement
check on seams and odd bottom tails. This geometry is specific to this profile.

`--export-edges REF DIST NEW.tsv` persists the measured 168-feature strip
extractor output, named by the original 228-layout indices. RGB8 uses the native
path; RGB16 and opaque RGBA retain precision through the shared linear ingress.
It refuses to overwrite a result and does not assign a quality score.

`score_manifest.py --teacher-features` runs serial local extraction without
human labels or fitting. Its input columns are `dataset source codec pair
reference distorted encoded bpp setting` (tab-separated); `encoded` retains
the original bitstream even when `distorted` names a decoded PNG. Teacher norms
and map hashes go into `cells.jsonl`; the separate `features.jsonl` sidecar joins
by reference/encoded SHA-256. Both input artifacts remain in their original
stores. The scorer and exporter detect file formats by signature, so encoded
blobs need no extension. Existing quality-evaluation mode remains available.
`--features-only` refreshes just the feature sidecar after an extractor change;
both training modes require `--feature-source COMMIT` to identify the dependency.
`fit_probe.py --features REFRESHED_DIRECTORY` joins the refreshed values to the
unchanged teacher ledger by exact reference/encoded hashes, requires equal key
sets, and records both manifests. Frozen teacher maps are retained in place.

`prepare_fit_input.py STAGE NEW_OUTPUT --dataset NAME --build-commit COMMIT`
validates Zenfleet's `pairs.tsv` against the staged references and encoded blobs,
checks the declared quality grid, and writes the extraction manifest. It freezes
source-level fit/tune/test partitions before extraction, with at least 20% each
for tuning and testing within four declared broad content groups. Original
content classes remain in the split file. These broad groups include generated
photo-like products; they do not certify the final per-class sample quotas.
Encoding/reconciliation stays in Zenfleet; this adapter only translates files.

`fit_probe.py EXTRACTION SPLITS NEW_OUTPUT --build-commit COMMIT` is a private
feasibility fitter using the separate teacher/feature sidecars. It consumes no
human labels, requires explicit `fit`/`tune`/`test` source partitions and rejects
identical reference bytes across those partitions. It fits nonnegative weights
for each teacher norm in log space, preserves zero for identical features, and
chooses regularization using tuning sources only. Final test sources do not
change the model. Its reports describe teacher agreement, not human-quality
acceptance or full corpus coverage. Weights remain experiment artifacts.

`--student MODEL.tsv REF DIST` evaluates the private fitted model and emits
all five norm predictions. It validates the 168-row layout, scales and weights;
it does not produce a spatial map. Formula tests and a real-image runtime smoke
check are separate from human-quality qualification. The runtime now pins
zensim `ad18b444`; the older frozen extraction must have its features refreshed
before fitting a compatible model. `compare_features.py` records paired exports
and exact differences for explicit manifests without applying a tolerance.

`score_manifest.py PAIRS BINARIES NEW_OUTPUT --model FIT_DIRECTORY --teacher
FROZEN_EVALUATION --build-commit COMMIT` evaluates a frozen student against the
existing human labels and teacher scores. It checks the model hash, requires the
same extractor binary used for fitting, verifies exact pair/label/input-hash
alignment, and writes all five zenstats panels. Teacher diffmaps remain in their
original store; this scalar student produces no spatial map. Its output supports
the existing clustered bootstrap and `choice_eval.py --candidate student`.

`resource_sweep.py --model MODEL.tsv` measures the fitted predictor at the
declared crop sizes. It retains fresh-process peak RSS including model loading,
decoding and inputs. `--bench-student` runs interleaved metric-only and
decode-included timing arms, with the model preloaded and a warm OS file cache.
The report fits fixed overhead and per-pixel cost separately for both timing
scopes. Noisy runs remain flagged; a synthetic-model smoke is not performance
qualification for a fitted candidate.

Install `requirements-training.txt` in a virtual environment, then run
`just margarine-fit-check /absolute/path/to/venv/bin/python`. The fitter uses
[SciPy NNLS](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html)
with source-balanced rows and an explicit ridge penalty. These additional
checks require that environment; the existing manifest checks use only stdlib.

`prepare_dense.py` prepares references from every full-image representative in
the existing train-only K500 selection, resolving URLs through the canonical
training catalog and verifying each rendered source against its pinned LFS
object. The renderer makes twenty log-spaced sizes from 32 to
`min(native_max,4096)` with Lanczos3 in encoded sRGB, without upscaling. It
records a uniform q0–100 step-2 plan. This is pilot data preparation: class
quotas, cross-corpus duplicate checks and validation representatives must be
resolved before treating it as a complete training corpus. Generated experiment
artifacts stay under Margarine with source provenance; they do not belong in
the canonical image store.
No model is fitted by this tool. Each completed source updates the manifest.
`--resource-crops REF DIST NEW_DIRECTORY` persists exact 64²/256²/1024² center
crops plus the native pair, recording crop coordinates. It does not resample
or upscale. These crops diagnose resource scaling on actual distorted pixels;
they are not independent content samples or a training corpus.

The first control scores linear-light 2×2 averages with Butteraugli. This is
deliberately a control, not the proposed finished metric: averaging can erase
fine artifacts. Its tests include such a counterexample. A useful candidate
must retain a full-resolution fine-detail signal while reducing storage and
work in the coarse frequency paths. Streaming intermediates and a smaller
directional-filter bank are hypotheses to test, not measured wins.

The `margarine-box3` control retains the full-resolution signal and all scoring
stages, replacing general Gaussian filters with three running-sum box filters.
Box widths are derived from the Gaussian variance, without dataset fitting.
The 5×5 opsin preprocessing filter remains exact. Scoring modules are included
from the production source so this experiment does not fork their arithmetic.
Clipped, renormalized box boundaries differ from the teacher's Gaussian filter.
The experiment retains the teacher's full-image intermediate storage; this is
an accuracy/cost control, not an implementation of the memory target.

`margarine-box3 --strip REF DIST MAP` processes 32-row interiors with a halo
derived from the complete filter support, including the half-resolution path.
It accepts strided linear input internally and aligns each slice to the global
2×2 sampling lattice. The whole output map is retained. This bounds scratch
height, at the cost of repeated halo computation; it is a memory control, not
a speed optimization. `--memory box3-strip REF DIST` measures this arm.

On the first 620×800 AIC pair, the strip map's SHA-256 matched whole-image
box3 exactly. Mac fresh-process peak RSS was 43,712,512 bytes for strips and
103,530,496 bytes for teacher (platform `time -l`, two Rayon threads, including
decode and caller buffers). This measured pair still fails the quarter-RAM
target. Artifacts: `/Users/lilith/work/codec-artifacts/margarine/box3-strip-smoke-2026-09-25/`.

## Evaluation contract

`score_manifest.py --defer-panels` records all scalar scores and native maps,
then marks the run `scores-complete`. `just margarine-panels SCORED EVALUATOR
NEW_OUTPUT CANDIDATE COMMIT` evaluates that ledger without decoding or scoring
again. It verifies the ledger hash, pair count, identities and scoring arms,
and records the separate evaluator hash and build commit. Panel replay accepts
completed older runs too; their original files remain unchanged.
`just margarine-ordinary-panels` takes the same arguments and explicitly
requests only the ordinary panels. It retains every row, including rows with
zero or missing published dispersion, and marks supplied-sigma panels as not
requested in the replay manifest. This does not qualify those separate panels.

`just margarine-participant-pairs SCORED OPINIONS EVALUATOR NEW_OUTPUT
CANDIDATE COMMIT DRAWS SEED` joins the full KADID ledger to the reconciled
sanitized opinions. It resamples workers with replacement, retaining each
worker's observations across images and repeated judgments. This follows the
[cluster-bootstrap approach](https://doi.org/10.1111/j.1467-9868.2007.00593.x);
the image set stays fixed. It writes central 95% percentile intervals for each
strict metric-order reversal, plus a simultaneous interval radius from the
maximum centered pair error across every within-source pair. The latter
comparison family is fixed before observing human losses. Both harmful and
beneficial disagreements are retained. Native double-precision bootstrap means
are persisted with image ordering, hashes, seed and evaluator provenance.
These are approximate bootstrap intervals, not exact finite-sample coverage.
KADID distortion levels do not provide encoded byte budgets, so these pairwise
diagnostics cannot establish the matched-rate encoder-choice acceptance gate.

`just margarine-aic-intervals SCORED ORIGINAL_LABELS NEW_OUTPUT CANDIDATE COMMIT`
uses AIC4's pinned original `CI_min`/`CI_max` columns. It records strict
within-source metric reversals, point-label loss, and non-overlap of the
published marginal 95% JND intervals, with separate cross-codec counts. The
source/target identities must match all 300 scored pairs. Marginal intervals
do not supply a paired-difference interval or simultaneous coverage; the tool
does not claim either. The image-only sample also lacks encoded byte budgets.

Compare against this repository's FIR, multiresolution Butteraugli with the
same decoded pixels, intensity target, thread budget, and runtime dispatch.
Evaluate max and libjxl's mixed p/2p/4p pooling separately (p=1,2,3,6). Plain
Lp means are not substitutes. Persist raw pair scores and diffmaps before
aggregation; preserve codec, source, dimensions, configuration, and label
provenance. Never fit coefficients or select candidates using CID22 validation
or AIC holdouts. Split development data by source, including crop/resize and
cross-corpus duplicates.

Use the full six-stat `zenstats::compute_panel`: SROCC, PLCC, KROCC, OR, PWRC,
and Z-RMSE. Report zenstats' `geomean3`, `harmean3`, and `min3` composites
of SROCC, PLCC, and PWRC beside those individual statistics. These are the
repository's panel aggregates, not additional statistics attributed to the
Mohammadi paper. Z-RMSE and OR remain separate lower-is-better criteria;
a composite improvement does not excuse a regression in either. Retain signed
rank correlations beside the polarity-tolerant panel.
Report each dataset, codec, source, and quality band, including sample counts
and unavailable/degenerate cells. Compare candidate and teacher on identical
rows. The user accepts up to 0.01 loss in rank correlation for Margarine’s
primary scalar against Butteraugli’s default maximum score, and 1% material
encoder-choice reversals. Report every max/p1/p2/p3/p6 variant; matching-norm
comparisons remain diagnostics, not five separate acceptance gates. Do not mistake a confidence interval containing zero
for evidence of non-inferiority against those margins. The material-choice
definition must be explicit in each decision panel; zero-epsilon pair counts
are diagnostic counts, not that acceptance gate.
`margarine-eval --bootstrap SCORES.tsv OUT.tsv DRAWS SEED` resamples whole
source groups with replacement, pairing candidate and teacher on every draw.
It refits each logistic mapping within each draw and reports central 95%
percentile intervals for candidate-minus-teacher deltas on the individual
statistics and composites. Any undefined draw makes that interval unavailable.
The seed and draw count are explicit. These intervals describe the sampled
source population; five AIC4_sample sources provide limited population coverage.
This differs from zenstats' existing row bootstrap.

`just margarine-quality-bands EVALUATOR SCORES.tsv NEW_OUTPUT.tsv BANDS`
reports the same statistics separately across human-quality rank bands, from
worst to best, within each dataset. The band count is explicit (recipe default
five). Equal human targets stay together, assigned by their first rank, so band
sizes can differ. Every row is retained exactly once for each scoring arm;
empty or degenerate bands remain explicitly unavailable. Band assignment uses
no metric predictions. These are conditional diagnostics with narrower label
ranges, not substitutes for pooled panels or matched-rate choice evaluation.

The second panel checks encoder behavior: within-source candidate ordering,
material inversions, ties/dead zones, and teacher regret at matched byte budgets
or target quality. The third checks raw tails, structural corruptions, and
local-edit/diffmap agreement with actual score changes. A high pooled SROCC
does not pass either panel. Zensim's 0–100 dial thresholds do not automatically
apply to a Butteraugli-shaped distance.

`choice_eval.py CELLS.jsonl NEW_OUTPUT --build-commit COMMIT` measures observed
byte-budget choices when the ledger contains bpp. At each distinct observed bpp
within a source, each arm selects its lowest score among eligible encodes. Ties
prefer fewer bytes, then pair ID; the candidate never uses teacher scores to
break a tie. Reports contain absolute and relative teacher regret, pooled and
equal-source exceedance rates at several diagnostic thresholds. These thresholds
are not an agreed definition of materiality. Observed budgets reflect each
dataset's ladder density; they are not a uniformly sampled production workload.

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
initial required set is CID22 validation, AIC-3, AIC4_sample, KonJND, LIVE,
CSIQ, KADID, TID, and non-photo/codec-ladder/corruption
instruments. Record which corpora were used to develop each candidate.
The user clarified that `AIC4_sample` is the intended AIC human-quality set.
Full AIC2026 is separate, additional coverage for teacher agreement.

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
The shared experimental reader uses encoded-sRGB input, preserving RGB16
samples through the sRGB transfer function in f64 before storing f32. RGB8
retains the teacher's lookup conversion. RGBA is accepted only when every
alpha sample is opaque. This is an explicit SDR convention, not a general CMS;
the corpus runner must audit metadata. Its original AIC mode still requires
untagged RGB8 PNGs. CID22 inventory found two distinct PNG ICC profiles, both
identifying sRGB; a broader audited corpus runner is still needed.

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
cargo run --release --manifest-path experiments/margarine/Cargo.toml \
  --bin margarine-box3 -- ref.png dist.png box3.f32le
```

Both write raw little-endian f32 diffmaps (dimensions are in stdout) and print
max/p1/p2/p3/p6 scores. Each arm runs separately; native diffmap dimensions
differ. Never relabel the coarse map as a full-resolution localization map.
Files are created exclusively, so reruns cannot overwrite earlier results.
These invocations are scoring tools, not a timing benchmark.

For interleaved cold-pair kernel timing, use `margarine-box3 --bench REF DIST
NEW_RESULTS.json` under the same nice/thread budget. It predecodes both inputs,
runs zenbench for 20–40 rounds with runtime SIMD dispatch, and includes scoring,
diffmap allocation and destruction in both arms. It does not time decoding,
disk output or auxiliary p1/p2/p6 reductions, and does not measure warm-reference
comparisons. Set `ZENBENCH_NO_SAVE=1` to disable its separate scratch autosave.
For fresh-process peak RSS, use platform `time` around `margarine-box3 --memory
teacher|box3 REF DIST`. This includes decode and input buffers, and computes
max/p3 plus a retained native diffmap. The input pair must be nonidentical;
identity shortcuts are not resource measurements of the scoring pipeline.

`score_manifest.py PAIRS.tsv BIN_DIRECTORY NEW_OUTPUT --build-commit COMMIT`
runs teacher and box3 serially on a manifest from `aic4_sample_manifest.py`.
It requires untagged RGB8 PNGs and interprets both arms as sRGB, hashes inputs
and binaries, persists all five score variants and content-addressed native
diffmaps, then runs the six-stat panel for each pooling variant. Its
`progress.log` and cell logs preserve progress and failures. It does not
dispatch distributed jobs or benchmark execution time.

The local AIC4 sample audit found all 305 PNGs untagged (no gAMA, cHRM, sRGB,
iCCP or cICP). The 300-pair manifest uses the original JPEG label CSV pinned
by SHA-256 in [aic4_sample_manifest.py](aic4_sample_manifest.py), with source
URLs and original dataset README retained. The first real-pair smoke check
persisted both 620×800 maps and all norms successfully; it is not a dataset
accuracy result.

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

## Initial verification, 2026-09-25

At `aef8e034`, five lab tests, clippy with `-D warnings`, and a release build
passed on the ARM Mac. The checkerboard test proves a failure of the
half-resolution control: the reduced pair is identical and scores zero while
the full-resolution Butteraugli score is positive. A 64×64 RGB8 photograph
identity smoke run wrote 64×64 teacher and 32×32 control maps; all stored f32
values and all five pooled scores were zero. This is I/O and correctness
evidence only, not perceptual-accuracy or performance evidence.

CI runs the lab on the existing native matrix (including Windows ARM and
macOS Intel), plus i686/armv7 cross jobs. The workflow passed local actionlint;
remote test results must be checked separately.

Data availability checked over SSH on `lilith`:

- CID22 validation is present at
  `/mnt/v/dataset/cid22/CID22_validation_set/`, with its CSV and pairs TSV.
- The older sample is present at
  `/mnt/v/dataset/aic4_sample/JPEG_AIC-4_Sample_Dataset/`.
- Full AIC2026 is on Tower at
  `root@tower:/mnt/user/coefficient/input/datasets/aic2026/`, reachable from
  the Mac through `ssh dev 'ssh root@192.168.50.170 ...'`. It is also mounted
  on `lilith` at `/mnt/tower/input/datasets/aic2026/`. Direct Mac SSH to Tower
  failed authentication; no credentials or server configuration were changed.
  The full-image CSV and ZIP align on all 9,618 pairs, 70 sources, and 17
  codec configurations. This checks archive membership, not pixel decoding
  or whole-archive checksums. See the
  [inventory](../../benchmarks/margarine_aic2026_inventory_2026-09-25.json).

The strict `aic2026_manifest.py ROOT OUTPUT --build-commit COMMIT` tool preserves
the metadata, attribution and README bytes (including hyperlinks), writes
archive-member identities, and rejects duplicates or missing pair members.
It does not substitute metric-derived JND values for human targets.
The generated manifest/provenance copy lives on `lilith` at
`/mnt/v/output/margarine/aic2026-audit-2026-09-25/`, mirrored to Tower at
`/mnt/user/coefficient/output/margarine/aic2026-audit-2026-09-25/`.
The pair TSV SHA-256 is
`1fcb931ebf008f5434ac5d4527b39e5941d221e8f9911d3c9a429898087f0cbf`.
Pair TSV, JSON manifest and copied metric CSV hashes match the Tower mirror.
No AIC2026 perceptual evaluation has run.

Still unimplemented: a Margarine candidate meeting both resource targets,
source-cluster uncertainty, quality bands, matched-budget encoder regret,
corruption/local-edit panels, and measured runtime/peak-memory gates. The
contract above records the intended evaluation, not completed coverage.

## Student feature cost probe

`margarine-box3 --bench-features REF DIST NEW.json` adds a pinned Zensim
372-feature extraction arm to the paired cold benchmark. `--memory features372
REF DIST` measures that extractor in a fresh process. This arm has no trained
Margarine score: its uniform weights force feature computation only. It must
not be reported as a completed approximation or evaluated as a human-quality
metric. Both arms receive the same decoded linear samples; conversion to the
extractor's linear RGBA byte view is outside timing and included in process
memory. A strided-input test checks feature identity with the packed view.

CID22 manifests can now be audited with `cid22_manifest.py ROOT NEW_OUTPUT
--build-commit SHA`, then scored with `score_manifest.py ... --ingress
cid22-srgb`. The policy preserves 16-bit samples, verifies PNG chunk CRCs,
allows only the three audited sRGB ICC identities (two PNG, one JPEG), and
rejects other color tags. It records bpp and encoder settings, and explicitly
excludes the 49 identity rows with zero opinions from the 4,292 rated pairs.
Opaque alpha is checked during decoding. This convention does not apply the
small transfer-table differences between the embedded sRGB profiles.

The cost probe also includes the base 228-feature set and its existing
Zensim strip entry (256-row interiors, 128-row margins, serial strips).
`--memory features228-strips REF DIST` selects that memory probe. A 65×801
seam/tail test compares all 228 strip features to the whole-image extractor.
These remain feature-cost probes, not trained Margarine implementations.

The 2026-09-26 box-blur optimization separates clipped borders from the
constant-width interior, enables runtime SIMD dispatch, and omits identity
box passes. On r5900xt, all 300 AIC4 maps and five scores are byte-/value-identical
to the frozen preceding binary. A 620×800 callgrind comparison drops total
instructions from 2,140,821,372 to 1,027,380,192, including decoding. This does
not establish a wall-clock speedup or either resource gate. The
[verification pointer](../../benchmarks/margarine_box3_optimization_2026-09-26.pointer.json)
records the binaries, source hash and persisted comparison artifacts.

## Multirate Butteraugli-lineage experiment

Build with `--features multirate` to replace broad Gaussian calls with area
reduction, a Gaussian on the reduced lattice, and bilinear reconstruction.
Sigma ≥6 uses a four-pixel lattice, sigma ≥2 uses two pixels, and narrower
filters retain the exact FIR. The residual Gaussian variance subtracts the
analytical reduction/reconstruction variance; these are experimental choices,
not fitted dataset constants. Opsin conversion, full-resolution residuals,
Malta filters, masking and multiscale combination retain their shared source.
Strip origins align to eight input rows to preserve both blur and half-scale
lattices. Existing seam and fine-artifact tests pass without relaxed tolerances.
Human-quality and resource acceptance remain unmeasured for this candidate.

`score_manifest.py --candidate multirate --teacher FROZEN_EVALUATION`
reuses verified teacher scores while persisting every candidate map and norm.
The same pair, label, source and pixel-hash checks used for the scalar probe
apply. This avoids recomputing teacher maps for each direct approximation.

`--native-strip ROWS REF DIST MAP` converts encoded RGB/RGBA 8- or 16-bit
samples only for each strip and its halo. `--memory-native ROWS REF DIST`
measures the same path without map serialization. Inputs retain their native
precision; non-opaque alpha is rejected. The borrowed row view supports sample
strides, and tests compare native strips to linear strips exactly, including
RGB16 one-bit differences at seams. The output map remains full resolution.

`--bench-direct ROWS REF DIST NEW.json` measures teacher and native-strip
candidate with interleaved metric-only and decode-included arms. The resource
sweep accepts `--direct multirate --strip-rows ROWS` and marks runs with fewer
than twenty samples unreliable. Zenbench is pinned to its committed Linux
self-thread detection fix (`1bf8a650`); no gate thresholds are relaxed.

`--features compact` selects an experimental LF/MF decomposition on a 2×
coarser lattice. HF/UHF residuals still include every native XYB sample,
and Butteraugli's nonlinear transforms, directional scoring and masking remain
shared. This first version reconstructs LF/MF before scoring; it does not yet
reduce their retained plane storage. Its constants are analytical, with no
human-label fitting. Quality and resource qualification are separate gates.

`--features compact4` additionally replaces the sixteen-direction Malta bank
with its axial and diagonal lines. The constant-response multiplier follows
the original tap counts (1104/260 for UHF, 4 for HF/MF). It shares the
asymmetric difference normalization and zero-border behavior. This is an
approximation experiment; angular-detail fidelity requires human evaluation.

`--features reuse` keeps a bounded 32-buffer scratch pool across same-height
strips of the compact candidate. It clears the pool when strip geometry
changes and between scales. Production Butteraugli retains its eight-buffer
default. This changes allocation scheduling only; total-process memory must
be measured before selecting a strip height or claiming a resource pass.

`--features sparse` retains all sixteen Malta patterns and native scaled
differences, evaluating responses on the even-coordinate lattice and linearly
reconstructing the intervening map values. Four phase planes make the sampled
stencil loads contiguous. Tests require exact agreement with the full bank
at sampled nodes, including odd dimensions and poisoned input padding.
This is a spatial approximation and requires its own human-quality panel.

`--features pooled` selects one native reference/distorted coordinate jointly
per 2×2 block, using the largest squared linear RGB difference. It scores
the selected pairs with the compact Butteraugli pipeline and repeats the
coarse map over the original geometry. Odd edge samples are included once
during selection. Tests cover every position of a one-bit RGB16 change,
checkerboard distortion, strided inputs and odd strip boundaries. This is
an uncalibrated pair-dependent approximation, not an image resampler.

`--features perceptual` runs native-resolution Butteraugli opsin adaptation
in bounded-height strips before jointly selecting a sample from each 2×2
cell. Selection uses the existing MF channel weights. The resulting XYB
planes feed the shared decomposition, complete Malta bank and masking;
the additional scale averages these XYB planes. This changes scale semantics
and is a separate approximation, with no inherited quality qualification.

`choice_eval.py --human-loss-threshold VALUE` additionally compares the native
human labels of the teacher-selected and candidate-selected encodes. Repeat
the flag to record an explicit diagnostic curve. Positive loss means worse
human quality, accounting for each dataset's declared label direction. Labels
must be complete and finite. These are observed-label differences, without
a confidence/significance claim or an agreed material-reversal threshold.

The `bounded` feature retains the multirate approximation's frequency bands,
all six Malta filters, constants, and accumulation order. It schedules Malta
by channel and recycles HF/MF response maps immediately, then reuses the
completed psycho planes across strips. Exact plane comparisons cover odd
dimensions and three asymmetry settings. This is a storage/scheduling
experiment; its speed, process RSS and real-corpus map equivalence require
separate measurements. It does not use sampled region correction.

The scorer accepts the raw-label adapter's `sigma` and `label_method` columns
and preserves them in its ledger. `--input-audit PREPARED/_MANIFEST.json`
requires the full source audit, identical pair-manifest bytes, and an exact
match of staged image hashes, sizes and dimensions before scoring. This
connects `prepare_human.py` to evaluation; label-only manifests cannot pass it.
The original zenstats panel uses corpus-level Z-RMSE. The separate
`--published-sigma` panel below consumes retained per-stimulus dispersion.

The `lattice` feature adds regular 2×2 Malta output sampling to `bounded`,
retaining its native frequency decomposition and all sixteen directional
stencils. It loads interleaved native samples directly into fixed arrays,
avoiding four materialized phase planes. Bilinear reconstruction fills the
other response locations. Sampled-node tests require bit identity with the
complete bank on tiny, odd, and strided inputs. This is an approximation
between nodes; corpus and resource acceptance are separate.

The `tiles` feature bounds lattice working buffers in both dimensions, using
256-column interiors and the requested row count independently at each scale.
Crop origins preserve the blur, Malta, and SIMD phases; finite halos are
discarded from the output. Bitwise map comparisons cover both tile axes, odd
RGB16 dimensions and strided rows. The tile geometry is experimental and has
no resource qualification until measured across the complete size workload.

The `planar` feature retains lattice scoring and converts encoded rows directly
into its pooled linear RGB planes. Half-scale rows average native linear samples
during conversion, preserving the existing arithmetic order and odd-edge counts.
This removes the interleaved linear strip and subsampling intermediates. Bitwise
tests cover RGB/RGBA 8/16-bit samples, input strides, output padding, odd dimensions
and complete strip maps. Default, lattice and planar tests pass on the ARM Mac;
corpus-wide equivalence and resource improvements require separate measurements.

`planar-tiles` combines that direct conversion with the existing 256-column
tile walker. Each tile forms native or half-scale planar rows directly from
its strided input region; global sampling phases and halos are unchanged.
The existing bitwise tile/strip comparison passes across both axes and odd
RGB16 edges. This is a storage experiment without a resource acceptance claim.

`stream-blur` retains planar lattice scoring and computes separable Gaussian
passes through a rolling horizontal-result buffer. Scratch height is bounded
by the kernel support. Coefficients and clipped-edge normalization come from
the original Gaussian; normalized border accumulation can differ by rounding
from its scalar tails. Strided/tiny/odd tests bound the difference at 1e-6 on
unit-range inputs, and the unchanged full-pipeline tests pass. This is an
experimental scheduling change; corpus and resource measurements are required.

The strip/tile walkers disable buffer retention when a scale has only one
interior region. Such a scale has no later strip to reuse the retained buffers.
Multi-region scales keep the existing bounded pool. The unchanged streaming
candidate tests pass after this allocation-lifetime change; process RSS and
time remain measurement requirements, including on tiny inputs.

### Reference-only region selection

The `reference-regions` experiment uses the same paired-RGB proxy and original
Butteraugli regional correction as `refined2`. It partitions reference tiles
into two texture strata using encoded-RGB squared neighbor differences, then
samples the tile nearest each stratum's area-weighted mean. Area weights enter
the correction. Distorted pixels cannot change region selection. This tests
selection stability; it does not establish quality or resource acceptance.
The selector supports RGB8/RGB16 and strided rows, with no input quantization.

### Published label dispersion

`margarine-eval --published-sigma SCORES_WITH_SIGMA.tsv NEW_OUTPUT.tsv` accepts
the ordinary score schema with a final `sigma` column in native label units.
It reports per-sample OR and Z-RMSE through zenstats, after logistic rescaling,
in separate columns from the original corpus-standardized panel. Missing sigma
is explicit `unavailable`; negative or nonfinite supplied values fail. Zero
is valid dispersion data and makes the full supplied-sigma panel unavailable
because sigma-normalized residuals would divide by zero. Ordinary panels keep
every row, including zero-dispersion observations.
The manifest scorer writes these additional files when raw labels supply sigma.
These statistics use the published dispersion as declared by `label_method`;
they do not reinterpret observer dispersion as a standard error or establish
statistically distinguishable encoder-choice harm.

### Encoder-choice disagreement diagnosis

`disagreements.py` collapses repeated observed budgets selecting the same two
encodes, while preserving their count and budget range. It reports both harmful
and beneficial point-label tails. Optional hash-verified native-map probes
compare the candidate at Butteraugli's peak pixel, before and after normalizing
by the p1 score ratio. This distinguishes a global scale shift from a spatial
peak discrepancy. It does not infer participant significance from point labels.

The `stable-peak` experiment separates reference-only global calibration from
peak localization. It uses 96-pixel interiors around both reference-stratum
representatives and a 32-pixel interior around the proxy's maximum. All three
regions use original Butteraugli with its finite halo. Only reference-stable
regions determine the global cubic correction; the peak patch is inserted
afterwards. These are experimental sampling dimensions, not calibrated defaults.

The `anchored-pool` experiment retains stratified regional correction but changes
the reduced reference representative to the linear RGB block average. It adds
the selected native distortion delta to that fixed reference value. This tests
whether moving the selected error between textured pixels destabilizes scores
by also changing the reference content. No clamping or learned coefficients are
introduced. Tests cover moving errors, odd cells, identity, checkerboards and
isolated RGB16 low-bit changes. Quality and resource qualification are pending.

### Raw participant opinions

`prepare_kadid_opinions.py` audits KADID crowd ratings against the published
rounded DMOS and variance. It separates TID controls and gold/tainted rows,
retains original image links, and replaces worker identifiers with stable
pseudonyms. Location and IP fields are not copied. Its manifest reports counts,
duplicate worker/image observations, and exact agreement within the published
decimal rounding intervals. A `requires-reconciliation` result must not be
treated as the participant sample behind the published labels.

`wide-malta` evaluates the complete native Malta bank in sixteen array lanes
instead of eight. It preserves the uniform overlapping-block tail traversal.
Debug and release exact-response tests pass on aarch64 and x86_64; these checks
do not establish a speed improvement. `margarine-direct-perf` records CPU cycles
for the interleaved timing workload. Its percentages include teacher, candidate
and decoding arms, and instrumented timings do not qualify resource targets.

`simd-malta` uses explicit safe vector loads for the same native Malta stencils.
The stencil interface accepts either array or SIMD vectors without changing
addition order. Runtime dispatch generates x86, NEON, WASM and scalar variants
from one kernel; `wide-malta` selects sixteen lanes, and `row-malta` independently
selects alternate-row sampling. Exact border/tail tests pass in debug and release
on ARM and x86 for the eight-lane full bank. No speed gain is implied by those
tests; corpus replay and resource measurements remain separate.

The row-tile experiments accept `MARGARINE_TILE_COLUMNS` (default 512) as a
positive multiple of four. The scoring and resource wrappers set it explicitly
from `--tile-columns` and record the value in their manifests. The resource
recipe accepts columns after rows. The geometry test compares 256, 512, 768 and
1024 columns against the same full-map reference, including strided RGB16 and
odd edges. This exposes a scheduling experiment; it does not select a new
default or change the quality/resource gates.

The row graph owns Gaussian kernels in an indexed table and copies lightweight
operation descriptors. Area reduction borrows up to four cached row views from
a stack array. This removes per-row kernel reference counting and temporary
reduction vectors while preserving filter coefficients and arithmetic; the
unchanged row/strip equivalence tests cover the resulting maps.

Row-cache consumers now ensure their input rows first, then borrow immutable
slices from the graph's owned buffers. An explicit stored-row assertion checks
that the derived cache support retains every requested input. This removes
per-row `Arc` allocation and cloning; it does not change cache geometry or
filter arithmetic. Exact row/strip tests pass on ARM and x86, with ARM checks
also covering phase rows, native Gaussian and alternate-row Malta.

Row-cache capacities round up to powers of two, so recursive lookups use an
index mask instead of division. The stored-row assertion remains active and
exact row/strip tests retain their original expectations. Additional retained
rows can increase memory, so this scheduling change requires a fresh process
RSS measurement before acceptance.

Corpus scoring now explicitly passes `--native-strip` to the candidate, using
`--strip-rows` (default 128), and validates the corresponding reported mode.
Earlier scorer invocations used full-height execution, even when the matching
resource run used 128 rows. Those ledgers remain valid full-height quality
results, but do not independently validate the streamed row cache. New runs
record rows and columns in the manifest and must match the resource geometry.

Coarse-row reconstruction now evaluates eight independent interpolation cells
with explicit portable SIMD and packs the resulting phases into contiguous
output arrays. The multiply/add order, edge handling and native output samples
are unchanged. Exact expansion and complete row/strip tests pass in debug and
release on ARM and x86; performance still requires the measured resource curve.
