# Margarine data index

The 168-feature fitted candidate fails the human-quality rank-loss limit on
AIC4_sample and CID22. It has no resource qualification as a fitted score. The analytic box-filter
control and feature-extraction cost probes remain separate experiments.
Human-quality coverage measured so far is AIC4_sample and CID22 validation, not
every required dataset. The total-process memory gate includes decoding and inputs.
The user permits smaller-image resource ratios to taper, provided they remain
below Butteraugli; larger-image targets remain 4× speed and quarter process RAM.

## Evaluation inputs

| Corpus | Canonical input | Audited subset |
|---|---|---|
| AIC4_sample | `lilith:/mnt/v/dataset/aic4_sample/JPEG_AIC-4_Sample_Dataset/` | 300 human-rated pairs, five sources; untagged RGB8 PNG |
| CID22 validation | `lilith:/mnt/v/dataset/cid22/CID22_validation_set/` | 4,292 rated pairs, 49 sources; 49 unrated identities excluded |
| Full AIC2026 | `tower:/mnt/user/coefficient/input/datasets/aic2026/` | Separate from AIC4_sample; human targets not verified |

CID22 preserves RGB16 low bits and accepts only opaque alpha. Its metadata audit
allows the three pinned sRGB profiles in
[cid22_manifest.py](experiments/margarine/cid22_manifest.py). Both scoring arms
use a common encoded-sRGB convention, not a general ICC transform. Manifests retain
human labels, bpp/settings where supplied, source hashes, and build commits.

Mac artifact root: `/Users/lilith/work/codec-artifacts/margarine/`.
Linux artifact root: `/home/lilith/work/codec-artifacts/margarine/`.
Remote `r5900xt` is reached from `dev` at `192.168.50.250`.

## Frozen scores and diffmaps

| Run | Full artifact location | Persistent Tower mirror |
|---|---|---|
| `box3-aic4-2026-09-25` | Mac artifact root | `/mnt/user/coefficient/output/margarine/box3-aic4-2026-09-25/` |
| `box3-cid22-2026-09-26` | r5900xt Linux artifact root | `/mnt/user/coefficient/output/margarine/box3-cid22-2026-09-26/` |

The AIC run contains 600 native diffmaps; CID22 contains 8,584. Both retain all
five pooling variants (max/p1/p2/p3/p6), content-addressed maps, binary hashes,
source hashes, and per-cell logs. The Mac CID22 copy contains score/manifest
metadata only, not the maps. Tower verification checked the ledger and three
deterministically selected maps for each run.

Ledger SHA-256 (`cells.jsonl`):

- AIC: `3fcc5e2e763ca8a34404457b1d7da576e16c82a99d1e2ceb0f99c546cc860d4e`
- CID22: `9d3a64cf8024ea67b87a4993d4ded4edd1f68e703e14e78009aec1c705023590`

Audits: Mac `aic4-sample-audit-2026-09-25/`; r5900xt
`cid22-audit-2026-09-26/`. The latter used build `f1c52826dc24`.
Source-cluster AIC uncertainty lives in Mac `cluster-aic4-2026-09-26/`;
its compact results and provenance are checked into `benchmarks/`.
The CID22 clustered run is complete in `cluster-cid22-2026-09-26/` on
r5900xt and Mac: all five norms × 2,000 source-cluster draws, with no undefined
draws across the ten reported statistics. The combined table and hashes
are tracked under `benchmarks/margarine_cluster_cid22_2026-09-26*`. These are
per-statistic intervals for box3, not simultaneous bounds or learned-model results.

## Evaluation and resource results

[The evaluation contract](experiments/margarine/README.md#evaluation-contract)
defines the full zenstats panel, separate composite aggregates, source-held-out
evaluation, and encoder-choice checks. Matched-byte material-choice gates and
the complete corruption/quality-band instruments remain unmeasured.

CID22 observed-budget diagnostics are stored in Mac
`choices-cid22-2026-09-26/`, with compact summaries under `benchmarks/`.
There are 4,285 distinct observed budgets across 49 sources. At a diagnostic
relative teacher-regret threshold of 1%, max exceeds it at 252 budgets, p3 at
12, and p1/p2 at zero. This is not the material-reversal acceptance gate;
its materiality threshold and workload weighting have not been agreed.

Tracked results are under `benchmarks/margarine_*`: composite corpus and worst
subgroup tables, AIC source bootstrap, raw zenbench JSON, and resource metadata.
Every benchmark's adjacent `.meta.json` identifies its measured build and input.
The single 620×800 RGB8 cost probe is not a size sweep. No 4×/quarter-RAM claim
has passed across sizes or datasets.

`resource-scaling-2026-09-26/` on the Mac measures exact center crops of AIC2026
S67 reference/AVIF level 4 at 64², 256², 1024² and native 3355×2516. The original
pair and attribution CSV are in `aic2026-resource-pair/`; crop coordinates are in
`resource-crops-2026-09-26/crops.tsv`. Summary, hashes and descriptive time fits
are tracked under `benchmarks/margarine_resource_scaling_mac_2026-09-26*`.
The 228-feature probe passes the quarter-process-memory comparison at 1024²
and native size, but its mean speedups there are 3.72× and 3.63×. Neither the
whole-image nor strip probe satisfies both resource targets across these sizes.

The 168-feature ablation (`resource-edges-2026-09-26/`) reaches measured mean
speedups of 4.79× at 1024² and 4.87× at native size, with total-process RSS
fractions of 20.1% and 16.2%. At 64² and 256² its RSS fractions are 94.1% and
55.6%. These are extractor costs only; no fitted predictor, model overhead,
quality result or confidence-bound pass is implied. Tracked summaries and
artifact hashes use the `margarine_resource_edges_mac_2026-09-26` prefix.

`resource-edge-strips64-2026-09-26/` measures parallel 256-row edge-feature
strips with the corrected 64-row halo. Native-size mean speedup is 4.40× and
RSS is 10.1% of the matched teacher; at 1024² these are 4.67× and 18.4%.
At 64² and 256², runtime and process RAM both remain below Butteraugli,
meeting the user’s relaxed small-image comparison on this pair. The extractor has no
trained score, so these are resource findings, not Margarine acceptance.

Heaptrack captures and reports live at `heap-2026-09-26/` on Mac and r5900xt;
[their pointer](benchmarks/margarine_heap_2026-09-26.pointer.json) carries hashes.
Heaptrack RSS includes instrumentation overhead. Use fresh-process platform
`time` peak RSS for the uninstrumented process gate. These smaller resource
artifacts have not yet been mirrored to Tower; none have been deleted.

## Potential teacher-training data: audited metadata only

The broader canonical index is on **lilith** at
`/home/lilith/work/zen/DATA_PROVENANCE.md`. That index is absent from the Mac
workspace root. Its path and data claims must be checked on the source host.

Verified Parquet schemas on lilith:

- `/mnt/v/datasets/kadis700k/canonical/kadis700k_canonical_gpu_2026-07-01.parquet`:
  700,000 rows, 372 feature columns, source IDs, synthetic distortion metadata,
  Butteraugli GPU max/p3. Whole-file hash and CPU-teacher parity remain unverified.
- `/mnt/v/datasets/fill4-6codec-2026-07-01/fill4metrics_sidecar_2026-07-02.parquet`:
  4,214,382 rows; encoded filename/hash and four metric columns, including one
  Butteraugli scalar. Its exact teacher/pooling semantics still need verification.
- `/mnt/v/output/canonical-picker-2026-07-01-zensimA/zenjpeg_lossy/train.parquet`:
  761,310 rows, 212 origins, 469 feature columns, no Butteraugli column. Joining
  teacher scores requires validating the encoded-filename/hash relationship.

The measured JPEG training inventory has only seven quality values
(5,15,30,50,70,85,95), at most 11 geometries per source, and every content class
marked unknown. It does **not** satisfy the required dense size/quality/class
coverage. See [the recorded inventory](benchmarks/margarine_teacher_training_inventory_2026-09-26.json).
No weights have been fitted from it. These tables do not supply all five
Butteraugli pooling targets or verified diffmap persistence.

The AVIF table `/mnt/v/output/avifgen-2026-08-06/unified/scores.parquet` was also
inventoried: 564,300 rows, 1,082 source IDs derived from explicit
`ID.scaleWxH.png` filenames, and 30 quality values. Of those sources, 858 have
only one represented size; none has more than nine. Low-quality sampling is
step 5 while high quality is step 2. Content classes are unavailable. This
table therefore also falls short of the requested dense training axes; its
GPU max/p3 labels still need CPU-teacher verification. See
[the metadata summary](benchmarks/margarine_avif_training_inventory_2026-09-26.json).

Do not train or select weights using AIC4_sample or CID22 validation. Before
training, verify source-family splits, duplicate/crop lineage, teacher arithmetic,
feature ordering, encoded-artifact availability, and the missing coverage axes.

## Dense reference pilot

Mac `dense-references-pilot-2026-09-26/` contains 1,000 PNG references: twenty
log-spaced dimensions for each of fifty existing train-only, full-image K500
representatives. All output hashes and dimensions were rechecked after rendering.
The [manifest pointer](benchmarks/margarine_dense_references_pilot_2026-09-26.pointer.json)
pins source/render commits and records class counts. Original URLs, source hashes,
cluster IDs/sizes and per-render hashes remain in `_MANIFEST.json`. Resampling is
Lanczos3 in encoded sRGB, with no upscaling and a 4096-pixel maximum dimension.

The reference preparation feeds the JPEG feasibility run below. Per-class
coverage and cross-corpus near-duplicate checks remain outstanding; the
feasibility fit uses the frozen source partitions described below. These are project-local Margarine
artifacts; the image store remains unchanged. The original manifest’s pending
registration note is superseded by this disposition. The pilot
is not a complete training corpus. Its Tower mirror is
`/mnt/user/coefficient/output/margarine/dense-references-pilot-2026-09-26/`;
the manifest and three deterministic output hashes match the Mac copy.

The three-cell `encode-smoke-2026-09-26/` on dev and Mac exercises Zenfleet
encode-only jobs (JPEG q0/50/100, one 21×32 reference). The Mac
`teacher-feature-smoke-2026-09-26/` persists all five teacher norms, native maps
and the separate 168-feature sidecar. Every encoded/map hash and sidecar join
was checked; this is pipeline validation, not fitting or quality evidence.
[The pointer](benchmarks/margarine_teacher_feature_smoke_2026-09-26.pointer.json)
pins the image digest, controller hash and extraction build. Local worker output
is `ledger.chunk-*.parquet`; the controller's local `pairs` command still
requires an endpoint argument (an unused localhost URL sufficed).

## Dense JPEG feasibility run

`dense-jpeg-pilot-2026-09-26/` on dev and Mac contains 51,000 completed Zenfleet
encode jobs: 1,000 references × q0–100 step2, JPEG mode `jp3_t0_small_420`.
Every encoded blob and staged reference hash was verified by the input adapter.
`dense-jpeg-fit-input-2026-09-26/` freezes 28 fit / 11 tune / 11 test sources.
The broad class counts are 25 photo-like (including generated products),
10 screen, 7 line-art and 8 mixed; these do not meet final per-class quotas.

The Mac `dense-jpeg-teacher-features-2026-09-26/` extraction completed all 51,000
pairs with
copied, hash-pinned binaries from `teacher-feature-binaries-8fe68ff7/`. Their
Rust entry sources and the runner match commit `8fe68ff79563`. Its native maps
require exactly 95,322,589,380 bytes for the declared cells; disk capacity was
checked with a separate 24 GiB reserve. This is storage admission, not an RSS
measurement. The [run pointer](benchmarks/margarine_dense_jpeg_2026-09-26.pointer.json)
pins inputs, partitions and encoder image. Encodes/maps remain Margarine data.

The local worker supports `ZEN_PERSISTENT_EXEC=1` with
`ZEN_PERSISTENT_KINDS=encode`; the JPEG run used a 1 GiB child recycle watermark
and 100-job recycle limit inside a 12 GiB container. A three-cell smoke produced
byte-identical encoded outputs with warm and fresh executors. It auto-folds
existing sidecars beside `--ledger-out`, so isolated comparison runs need
separate ledger directories. Completed cells were reused when enabling warmth.

The encoded stage is mirrored at Tower
`/mnt/user/coefficient/output/margarine/dense-jpeg-pilot-2026-09-26/`. Its stage
manifest, complete pairs export and three deterministic encoded-blob hashes
match the Mac copy. `refs/` was excluded from this mirror because those bytes
already live in `dense-references-pilot-2026-09-26/`; the stage manifest maps the
flat source/size names to the same source IDs, dimensions and hashes.

## Extractor compatibility and refreshed features

Zensim supplies feature extraction for a teacher-distillation feasibility probe;
its quality score is not a Margarine target or input. The representation still
needs held-out teacher and human-quality validation. It is not a selected final
architecture merely because its measured resource cost was promising.

The isolated x86 tail repair landed in zensim `7d6d7451`; helper feature gating
landed in `ad18b444`. The sibling checkout was synced and its duplicate local
changes were proved already landed; historical experiments remain remotely
tagged. Margarine's runtime pins `ad18b444` in commit `caccee81`.

The broader sync changes features on 94/158 real comparison pairs across the
50 pilot sources. The [compatibility record](benchmarks/margarine_x86_edge_tail_2026-09-26.json)
pins both binaries and the measured changes. Do not fit the old feature sidecar
and silently run the resulting model with the newer extractor.

`dense-jpeg-features-ad18b444-2026-09-26/` on dev is the completed 51,000-pair feature-only
refresh, using frozen `feature-binaries-caccee81/` and runner `710d9f19`.
The original Mac teacher run is complete; its maps and five scalar
targets are retained. `fit_probe.py --features` joined by
reference/encoded SHA-256 with equal key sets. The original 28/11/11
source partitions remain fixed. This refresh generates no new encodes or maps.

The private scalar runtime and cached-teacher evaluator are implemented. The
updated model has now been evaluated against human labels, as recorded below. The four-pair CID22 adapter smoke on r5900xt
uses synthetic coefficients solely to test alignment and panel generation;
its artifacts are `student-eval-smoke-2026-09-26/`, not quality evidence.

The original-extractor control is fitted in Mac
`fit-old-extractor-control-2026-09-26/`. On its fixed 11-source / 11,220-pair
test split, teacher SROCC is max 0.940031, p1 0.952762, p2 0.953241,
p3 0.956060 and p6 0.956691. Every norm selected ridge penalty 0.01 using
tuning sources only. These are teacher-agreement results, not human SROCC loss
or an encoder-choice gate. The [fit pointer](benchmarks/margarine_old_extractor_fit_2026-09-26.pointer.json)
records all splits, metrics and artifact hashes. This model uses the original
Mac extractor and must not be loaded with the updated extractor.

`fit-ad18b444-2026-09-26/` is fitted on the refreshed features with the same
28/11/11 source split. The [fit pointer](benchmarks/margarine_refreshed_fit_2026-09-26.pointer.json)
records its metrics and model hash. Its frozen scalar runtime was evaluated on
r5900xt in `student-aic4-2026-09-26/` and `student-cid22-2026-09-26/`; compact
ledgers and panels are copied to the Mac. Teacher scores were reused only after
label, pair and input-hash verification.

The candidate is rejected: all five AIC4 SROCC losses exceed 0.01 (p1 loses
0.014194; p3 loses 0.056157). CID22 p1/p2/p3 also exceed the loss limit (p3
loses 0.023411). The [full corpus panel](benchmarks/margarine_student_human_2026-09-26.tsv)
and [provenance](benchmarks/margarine_student_human_2026-09-26.meta.json) retain
all ten reported statistics for all five norms. These are point estimates;
clustered intervals and fitted-model resource qualification were not run for
this rejected candidate. The failure does not prove every learned model or
every use of zensim features impossible. The next direct-approximation experiment
profiles and reduces box3 blur costs while retaining Butteraugli's stages.

## Direct box3 profiling and completed teacher backup

The full 51,000-pair teacher extraction and original-extractor control fit are
now mirrored to Tower. The [verification record](benchmarks/margarine_teacher_tower_2026-09-26.pointer.json)
checks both complete sidecars, the manifest and three deterministic maps against
the Mac. There are 50,948 distinct content-addressed maps for 51,000 pairs.
Local copies remain intact.

The optimized direct box3 blur preserved every scalar and map byte on all 300
AIC4 pairs. Its [instruction profile](benchmarks/margarine_box3_optimization_2026-09-26.pointer.json)
records the reduction, but its [native timing](benchmarks/margarine_box3_timing_r5900_2026-09-26.meta.json)
is not sufficient for resource qualification: only four rounds completed, with
120 seconds spent waiting at the benchmark gate. Those samples put box3 at
76.79 ms versus 68.64 ms for Butteraugli on one 620×800 pair. A TTY diagnostic
reports a competing benchmark while the process inventory shows zenbench's own
heartbeat thread matching its benchmark-name filter. Resolve this measurement
issue before drawing conclusions from a larger timing sweep; thresholds remain
unchanged.

## Multirate direct approximation

Candidate `e7781180` retains Butteraugli's perceptual scoring stages and full
resolution residuals, evaluating broad Gaussian filters on reduced lattices.
All 300 AIC4 pairs completed in `multirate-aic4-2026-09-26/` on r5900xt;
the Mac copy contains score/panel metadata. Native maps remain on r5900xt.
The [corpus panel](benchmarks/margarine_multirate_aic4_2026-09-26.tsv) records
all five norms: the largest pooled SROCC loss is 0.00217025 (p1), with
p3 loss 0.00190935. These are point estimates, not clustered non-inferiority
or complete acceptance. CID22 and the direct native-strip resource sweep are
separate runs; resource measurements use the fixed zenbench Linux gate.

The first native-strip resource sweep (`34f01bdd`, 256-row interiors) fails
the target. The [size table](benchmarks/margarine_multirate_resources_2026-09-26.tsv)
records fresh-process decode-inclusive RSS and interleaved timings. At 1 MP,
RSS is 55.79% of teacher and metric speedup is 0.982×. At native 8.44 MP,
RSS reaches 24.86%, but speedup is only 0.811×; that timing hit the 120-second
wall limit after 19 rounds and is flagged unreliable. Tiny/small RSS was
slightly above teacher, so neither size passes its resource condition.
The full logs are retained in `multirate-resources-2026-09-26/` on r5900xt
and the Mac. No resource acceptance is claimed.

The broader zenmetrics evaluation guide (`docs/rev2_lan_stage.pointer.md`,
read 2026-09-26) identifies seven additional human-rated pixel corpora beyond
CID22: KADID, TID, CSIQ, LIVE, AIC3, KonJND and PIPAL. Its documented LAN
location is `s3://codec-corpus/eval372-rev2-2026-09-06/<corpus>/`. Those objects
and labels have not yet been independently reverified for Margarine. The
PIPAL guide records a 23,200-pair full set versus a 21,800-row historical
subset with unexplained exclusions; do not silently reuse that subset.

The multirate CID22 run is complete for all 4,292 pairs. Its
[corpus panel](benchmarks/margarine_multirate_cid22_2026-09-26.tsv) records
maximum SROCC loss 0.000113683 (max pooling); p3 changes from 0.792990393
to 0.793557406, and Z-RMSE from 0.620496825 to 0.619965005.
Maps and logs remain on r5900xt in `multirate-cid22-2026-09-26/`; the Mac
copy contains ledgers and panels only. Clustered uncertainty, encoder-choice
qualification and the seven additional human corpora remain outstanding.

The resampling loops now use fixed factors and process eight reduction outputs
together. The [native-strip verification](benchmarks/margarine_multirate_native_parity_2026-09-26.pointer.json)
found every map byte and all five norms identical to the frozen whole-image
multirate candidate on all 300 AIC4 pairs. The
[1 MP timing repeat](benchmarks/margarine_multirate_optimized_1mp_2026-09-26.json)
completed twenty rounds: teacher/candidate metric means 139.28/102.67 ms,
and decode-included means 157.60/115.23 ms. The harness reported nineteen
noisy rounds; these are optimization diagnostics, not resource qualification.
The candidate still misses 4× speed.

The experimental native-strip scheduler now walks the two scales separately,
using a scale-local halo and retaining only the completed half-resolution map
while processing the full scale. The [AIC4 comparison](benchmarks/margarine_independent_scales_2026-09-26.pointer.json)
found zero changed scalar values or map bytes on all 300 pairs with 256-row
interiors. This is arithmetic preservation, not a resource acceptance result.
The direct timing harness now allows 300 seconds to reach its unchanged
minimum of twenty rounds on larger images.

The independent-scale [four-size resource sweep](benchmarks/margarine_independent_scales_resources_2026-09-26.tsv)
uses 128-row interiors and completed at least twenty interleaved rounds at
every size. At 1 MP, mean speedup is 1.614× and total-process RSS fraction
31.75%; at 8.44 MP, these are 1.071× and 15.66%. Both smaller sizes are
below teacher time and RSS. Larger-image speed and 1 MP memory still fail.
Raw timings and descriptive fixed/per-pixel fits are tracked alongside the table.

The coarse-band candidate `88289438` completed 300 AIC4 pairs. Its
[panel](benchmarks/margarine_compact_aic4_2026-09-26.tsv) has maximum pooled SROCC
loss 0.00197247 (p1); p3 loses 0.00135690. Its 1 MP interleaved metric
means are teacher 144.858 ms and candidate 93.476 ms over twenty rounds.
It remains unqualified for speed; CID22 and other corpora are unmeasured for
this candidate. All native maps remain on r5900xt under
`compact-aic4-2026-09-26/`, with compact ledgers/panels copied to the Mac.

The four-direction Malta candidate `e725f169` is rejected: all five pooled
AIC4 SROCC losses exceed 0.01 (p1 loses 0.01004989; max loses 0.03633374).
The [full corpus panel](benchmarks/margarine_compact4_rejected_2026-09-26.tsv) records
the other statistics. Its 1 MP means are teacher 137.480 ms and candidate
91.666 ms over 21 rounds, so the approximation also fails the speed target.
The full directional bank remains the retained approach. Native maps and
logs are preserved on r5900xt under `compact4-aic4-2026-09-26/`.

The [compact instruction profile](benchmarks/margarine_compact_profile_2026-09-26.txt)
records 2,567,261,040 instructions for the native 1 MP/128-row path,
including decoding. Malta accounts for 29.71%, buffer clearing 19.37%,
and exact Gaussian filtering 10.83%. Callgrind timing/RSS are instrumented
and do not qualify native resource use. The capture remains on r5900xt
under `compact-profile-2026-09-26/`.

The [bounded-reuse timing](benchmarks/margarine_reuse_1mp_2026-09-26.json)
records teacher/candidate means 143.855/87.633 ms over 21 rounds. The
[AIC4 map comparison](benchmarks/margarine_reuse_parity_2026-09-26.pointer.json)
found no changed map bytes or scalar values on 300 pairs. This candidate
has not had a fresh total-process RSS sweep. The capture and full map
comparison remain in r5900xt `reuse-2026-09-26/`.

The full-direction spatially sparse candidate `32db0d72` completed AIC4.
Its [panel](benchmarks/margarine_sparse_aic4_2026-09-26.tsv) has largest pooled SROCC loss
0.00251958 (p3). The initial 1 MP means are 136.723/86.116 ms
(teacher/candidate, 21 rounds). Validated fixed row windows in `56414d83`
produce 142.668/86.866 ms in another 21-round run; that is no demonstrated
speed improvement. Neither reaches the resource goal. Native maps remain
in r5900xt `sparse-aic4-2026-09-26/`; compact metadata is copied to the Mac.

The joint linear-RGB sample candidate `1ae8eb96` reaches the measured
resource means on the four-size crop workload, but fails human quality.
The [resource sweep](benchmarks/margarine_pooled_resources_2026-09-26.tsv) records 6.172×/5.231×
metric speed at 1 MP/8.44 MP, 4.480×/4.099× with decoding, and total-process
RSS fractions 19.79%/12.12%. Smaller dimensions are below teacher time and
RAM. These resource observations do not qualify a failed-quality candidate.
The [AIC4 panel](benchmarks/margarine_pooled_rejected_2026-09-26.tsv)
exceeds the rank-loss limit for max, p2, p3 and p6; max loses 0.06333048
and p3 loses 0.01967489. p1 loses 0.00703697. The candidate is rejected.
Its generated maps remain in r5900xt `pooled-aic4-2026-09-26/`.
The original invocation used a preparation label as build provenance; the
manifest was corrected to the verified landed source commit, with the original
manifest preserved and hashed.

Native-XYB pair selection (`9ca083ff`) and corrected physical radii (`c428c4a7`)
are both rejected on all five AIC4 rank-loss comparisons. Their
[perceptual panel](benchmarks/margarine_perceptual_rejected_2026-09-26.tsv) and
[physical-radius panel](benchmarks/margarine_physical_rejected_2026-09-26.tsv)
record p3 SROCC 0.78912477 and 0.72752453, respectively, versus teacher
0.89692419. At 1 MP the interleaved metric means are 32.281/146.034 ms
(perceptual/teacher, 26 rounds) and 41.199/148.660 ms
(physical/teacher, 25 rounds). Decode-inclusive means are 45.665/162.451 ms
and 54.469/167.132 ms. Neither candidate qualifies the joint goal.
Their maps remain on r5900xt in `perceptual-aic4-2026-09-26/` and
`physical-aic4-2026-09-26/`; the Mac has compact metadata and panels.
The perceptual timing JSON records the checkout's older runtime Git parent;
the frozen binary and source provenance are pinned by the adjacent evaluation
manifest, not that runtime field.

The original-FIR region correction (`c6f2e82c`) passes the AIC4 pooled point
screen: largest SROCC loss 0.00259870 (max), with p1/p2/p3/p6 improving.
The [panel](benchmarks/margarine_refined_aic4_2026-09-26.tsv) records p3
0.90504428 and Z-RMSE 0.44917830. Three 128-pixel tiles are selected by
proxy cubic error mass; original Butteraugli corrects the map using those
regions. Crop origins align to 32 pixels to preserve SIMD/FMA grouping;
the finite-halo test is bit-identical at interior and image boundaries.
The [1 MP timing](benchmarks/margarine_refined_aic4_2026-09-26_1mp.json)
records 24 rounds: teacher/candidate 153.574/43.017 ms for scoring and
171.395/55.350 ms including decoding. Speed still fails. Native maps remain
on r5900xt in `refined-aic4-2026-09-26/`; other corpora and fresh RSS are
unmeasured for this candidate. Repeated AIC4 architecture screening is not
independent final validation.

The one-region variant (`2bab7498`) fails AIC4 p1 by 0.01252103; the
[panel](benchmarks/margarine_refined1_aic4_2026-09-26.tsv) records the other
four norms within the 0.01 limit. Its [resource sweep](benchmarks/margarine_refined1_resources_2026-09-26.tsv)
measures metric speedups 4.777× at 1 MP and 4.696× at 8.44 MP, with
RSS fractions 20.63% and 12.23%. Decode-inclusive speedups are 4.177×
and 3.884×. Tiny-image RSS exceeds teacher (6,377,472 versus 5,918,720 bytes),
so it also fails the small-image memory condition. All sizes reached at least
20 rounds with no unreliable flag. The run-heavy guard reports peak RSS
1.57 GiB, minimum available 55,676 MiB and peak load 3.25. This is a rejected
candidate, not resource or quality acceptance. The full records remain in
r5900xt `refined1-aic4-2026-09-26/` and `refined1-resources-2026-09-26/`.

Two-region correction (`19c719f5`) has largest AIC4 pooled SROCC loss
0.00070090 (max); p1/p2/p3/p6 improve in the
[panel](benchmarks/margarine_refined2_aic4_2026-09-26.tsv). These are screening
point estimates after repeated architecture comparisons, not independent
validation. The [resource table](benchmarks/margarine_refined2_resources_2026-09-26.tsv)
records 4.040×/4.670× metric speed at 1 MP/8.44 MP, total-process RSS
fractions 19.87%/12.61%, and decode-inclusive speedups 3.410×/3.727×.
At 64², RSS still exceeds teacher (6,262,784 versus 6,098,944 bytes).
All four sizes reached at least twenty rounds without an unreliable flag.
The guard measured peak RSS 1.58 GiB, minimum available 56,547 MiB and
peak load 2.66. Raw logs/maps remain in r5900xt `refined2-aic4-2026-09-26/`
and `refined2-resources-2026-09-26/`. The subsequent optimization removes
an unused proxy reduction and uses the paired-RGB path at dimensions up to
64; it requires its own quality/size verification before any resource claim.

The optimized two-region candidate (`01445505`) is rejected by CID22 p1:
SROCC falls from 0.82905345 to 0.81487377 (loss 0.01417968). The other four
norms stay within the rank-loss limit; the
[full panel](benchmarks/margarine_refined2_cid22_2026-09-26.tsv) retains the
Z-RMSE and composite changes. All 300 AIC4 candidate maps and five norms
remain [bit-identical](benchmarks/margarine_refined2_opt_parity_2026-09-26.json)
after removing the unused proxy reduction. The
[updated size sweep](benchmarks/margarine_refined2_opt_resources_2026-09-26.tsv)
now meets the measured metric-time and RAM means at every required crop size:
1 MP/8.44 MP speedups 4.169×/4.719× and RSS fractions 19.73%/12.11%; both
smaller sizes stay below teacher time and RAM. Decode-inclusive speedups
remain 3.469×/3.838× at the larger sizes. Guard peak RSS 1.58 GiB,
minimum available 56,548 MiB, peak load 2.27. Resource success does not
qualify a candidate that failed quality. Full maps and logs remain in
r5900xt `refined2-cid22-2026-09-26/` and `refined2-opt-aic4-2026-09-26/`.

A fresh broader-corpus audit found the documented
`/mnt/v/zen/zensim-training/rev2-lan-stage-2026-09-06/` directory absent on
lilith. The original local corpus directories and pair tables remain present.
Do not treat the staging guide as evidence that its manifests currently exist.
All 10,125 KADID legacy pair targets exactly equal `(5-dmos)/4`, the inverse
of canonical quality `(dmos-1)/4`; raw `dmos.csv` has mean values declining
from 4.078528 at level 1 to 2.006726 at level 5. New Margarine adapters must
read raw labels and preserve declared orientation rather than reuse that
legacy target column. Existing dataset files were left unchanged.
