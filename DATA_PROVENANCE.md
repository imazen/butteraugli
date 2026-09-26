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

## Stratified region correction: rank screen passes, choices diverge

Candidate `76728638` selects a peak-error tile and a representative tile for
the remainder, weighting their original-FIR corrections by represented area.
The [AIC4 panel](benchmarks/margarine_stratified_aic4_2026-09-26.tsv) and
[CID22 panel](benchmarks/margarine_stratified_cid22_2026-09-26.tsv) put all five
pooled SROCC/KROCC losses below 0.01. CID22 p1 loses 0.00818298 SROCC and
its Z-RMSE worsens by 0.016718; the aggregate panels are not universal wins.
Repeated AIC/CID screening is development evidence, not independent validation.

The [resource means](benchmarks/margarine_stratified_resources_2026-09-26.tsv)
meet the metric-time and RSS targets on the four-size crop workload: 4.082×
and 4.453× scoring speed at 1 MP and 8.44 MP, RSS fractions 19.86% and
12.31%. Decode-inclusive speedups are 3.402× and 4.016×. Both smaller sizes
stay below teacher time and RSS. The guard reports peak RSS 1.57 GiB, minimum
available 55,168 MiB, peak load 8.73. This remains one photo/encode crop family.

The [encoder-choice diagnostic](benchmarks/margarine_stratified_choices_cid22_2026-09-26.tsv)
exposes a large fidelity problem: at 1% relative teacher regret, 1,826 of
4,285 max-norm choices and 1,006 p3 choices exceed the diagnostic threshold.
That threshold is not the user-approved materiality definition, but the
result prevents treating the pooled-rank/resource screens as goal completion.
Per-pair region correction can vary across nearby encodes; investigate raw
choices and human-label harm before accepting this architecture. All maps
remain on r5900xt in `stratified-{aic4,cid22}-2026-09-26/`; the Mac holds
compact metadata, panels, and choice diagnostics. These maps are not yet
mirrored to Tower and must not be deleted.

The fuller multirate candidate's [teacher-regret diagnostic](benchmarks/margarine_multirate_choices_cid22_2026-09-26.tsv)
is substantially closer: at 1% relative teacher regret, max exceeds at 194
of 4,285 budgets, p1/p2 at zero, p3 at 13 and p6 at 11. Human-label loss
curves are recorded for [multirate](benchmarks/margarine_multirate_human_choices_cid22_2026-09-26.tsv)
and [stratified correction](benchmarks/margarine_stratified_human_choices_cid22_2026-09-26.tsv).
At a diagnostic loss greater than five native CID22 MCOS units, multirate
exceeds at 40 max-norm budgets and 12 p3 budgets; stratified correction
exceeds at 262 and 137. Five MCOS units is not an agreed materiality threshold.
The observed mean human-label change can improve while harmful tails remain.
These comparisons use raw point labels, not participant uncertainty; the
CID22 CSV provides opinion counts but no per-stimulus standard errors.

The initial `bounded-aic4-2026-09-26` run at `44f6051f` bypassed the new
channel scheduler in its default CLI route and exercised the existing full
multirate path. It must not establish corpus equivalence of the new schedule.
Its `--memory-native` and `--bench-direct` resource arms did exercise the
new schedule. The default route is corrected in the next commit; a separate
corpus run must validate it. No raw artifacts from the initial run were removed.

The [bounded scheduler resource sweep](benchmarks/margarine_bounded_resources_2026-09-26.tsv)
at `44f6051f` fails the speed goal: 1.775× at 1 MP and 1.337× at 8.44 MP.
RSS fractions are 32.29% and 16.94%, respectively, so the 1 MP memory
condition also fails. Tiny/small time and RSS remain below the matched
teacher; tiny RSS is close (5,632,000 versus 5,693,440 bytes). The 8.44 MP
run completed twenty rounds; zenbench reports nineteen noisy rounds while
the JSON unreliable flag is false. Guard peak RSS 1.57 GiB, minimum available
56,635 MiB, peak load 1.96. These observations do not qualify this candidate.

The corrected scheduler route at `4a78e85f` is now verified against all
300 multirate AIC4 results: [zero differing map hashes or scalar norms](benchmarks/margarine_bounded_aic4_parity_2026-09-26.json).
Its 1 MP callgrind run counts 2,165,689,200 instructions, with the original
Malta bank at 36.72% and buffer clearing at 3.85%. The subsequent sparse
Malta experiment counts 2,046,835,258: sampled filtering falls to 14.25%,
but reconstruction becomes the largest cost at 18.90%. Profiles and logs
are retained on r5900xt in `bounded-profile-2026-09-26/` and
`lattice-profile-2026-09-26/`; these are instruction diagnostics, not timing
or RSS claims. The initial lattice 1 MP resource invocation persisted its
measurements but then failed because a single size cannot fit overhead and
slope. It is not a completed resource sweep. Use `margarine-direct-timing`
for one-size diagnostics and the four-size sweep for resource qualification.

Reconstruction vectorization (`e7bc7678`) preserves the scalar formula in
bitwise tests over all interpolation phases and tails. Its instruction count
falls from 386,951,904 to 112,752,336; whole-process instructions fall from
2,046,835,258 to 1,772,823,032. [Profile hashes and source commits](benchmarks/margarine_direct_profiles_2026-09-26.pointer.json)
pin the raw captures. The [native 1 MP timing](benchmarks/margarine_lattice_opt_1mp_2026-09-26.json)
still misses 4×: teacher/candidate means 149.515/78.386 ms for scoring and
168.354/92.429 ms including decoding, across twenty-one rounds. Instruction
reduction is not a measured proportional wall-time improvement.

## Human-rated encoder-choice harm and broader evaluation

The user clarified on 2026-09-26 that the 1% material-reversal limit concerns
**human-rated quality loss**, not Butteraugli-score disagreement. Teacher regret
curves remain diagnostics. Native human-label loss curves are available above;
the user subsequently specified statistically distinguishable decreases. Do not
substitute point-label loss or teacher-relative regret for this acceptance gate.

The first untouched broader-corpus run, [CSIQ](benchmarks/margarine_lattice_csiq_2026-09-26.tsv),
is complete for all 866 pairs and thirty sources using frozen lattice
`e7bc7678`. Every staged image hash, size and dimension matched the source
audit, and the pair-manifest bytes matched before scoring. All five pooled
SROCC/KROCC losses are below 0.01; largest SROCC loss is 0.00551936 (max).
Max Z-RMSE rises from 0.55586309 to 0.56352007; p1 improves, other norms
worsen slightly. These are point panels, not uncertainty or joint acceptance.
Full teacher/candidate maps are retained on r5900xt in
`lattice-csiq-2026-09-26/`. Source pixels remain at
`lilith:/mnt/v/dataset/csiq`; the staged audit is
`human-inputs-audited-2026-09-26/csiq/` on both source and scoring hosts.

The [CID22 paper](https://cloudinary-marketing-res.cloudinary.com/image/upload/v1682076683/CID22.pdf)
explains that MCOS combines graded anchors and pairwise-derived interpolation;
its confidence intervals bootstrap both opinion collections. Opinion counts
alone cannot reconstruct those intervals. The local validation CSV has no
interval columns, so uncertainty-aware materiality needs additional published
data rather than an inferred standard error.

The two-dimensional tile scheduler at `0fdb5a55` reduces total-process RSS
to 17.02% at 1 MP and 8.21% at 8.44 MP, but its
[resource sweep](benchmarks/margarine_tiles_resources_2026-09-26.tsv) still fails
speed: 1.579× and 1.606× scoring, respectively. Decode-inclusive speedups
are 1.569× and 1.509×. Tiny RSS slightly exceeds teacher (5,996,544 versus
5,967,872 bytes); small RSS is 76.43%. All four sizes reached at least twenty
rounds without an unreliable flag. The run-heavy guard reports peak RSS
1.55 GiB, minimum available 56,765 MiB and peak load 1.53. Raw measurements
remain in `tiles-resources-2026-09-26/` on r5900xt and the Mac. Tile/strip
map equality is covered by odd, strided RGB16 tests; corpus-wide equality
and joint acceptance are not established.

The native-band lattice candidate `e7bc7678` also completed all 300 AIC4
pairs. Its [panel](benchmarks/margarine_lattice_aic4_2026-09-26.tsv) has
maximum SROCC loss 0.00334404 (p6); all five SROCC/KROCC losses stay below
0.01. Z-RMSE worsens for all five norms. These are development point
screens, not encoder-choice or resource qualification. Full maps remain
on r5900xt in `lattice-aic4-2026-09-26/`.

Participant uncertainty is required for the agreed material-choice criterion.
The published CID22 annotation ZIP contains only the validation CSV and license.
The full 7,288,601,687-byte archive's central directory contains 24,382 entries;
its only non-image, non-directory entries are that CSV and the license. The
inspected image plot (`1025469`) contains point curves without uncertainty fields.
Raw HTML, annotation ZIP, source URLs and the full archive index are preserved
in Mac `cid22-published-uncertainty-2026-09-26/`. No participant uncertainty
was inferred from opinion counts or image-to-image bootstrap variation.

The evaluator now supports a separate `--published-sigma` panel, using
zenstats's per-sample OR and Z-RMSE after logistic rescaling. This preserves
the original corpus-standardized panels. A missing dispersion value makes
the complete corpus's supplied-sigma panel unavailable; it does not silently
select a different subset. Published dispersion is not automatically a
standard error, a paired-loss interval, or the encoder-choice significance gate.

The [lattice CID22 panel](benchmarks/margarine_lattice_cid22_2026-09-26.tsv)
completed all 4,292 pairs. Every pooled SROCC/KROCC loss stays below 0.01;
largest SROCC loss is 0.00845873 (max). The
[human-choice point diagnostics](benchmarks/margarine_lattice_human_choices_cid22_2026-09-26.tsv)
show any observed label decrease at 483/4,285 max budgets, 13 p1, 21 p2,
69 p3 and 191 p6 budgets. These counts are not statistically distinguished
losses and do not resolve the agreed significance gate.

Reference-only two-region correction (`71063c59`) is rejected by the
[AIC4 panel](benchmarks/margarine_reference_regions_aic4_2026-09-26.tsv):
max SROCC loses 0.01788109 and p6 loses 0.01858510. p1 improves; p2 and p3
remain within the rank-loss screen. Maps remain in r5900xt
`reference-regions-aic4-2026-09-26/`.

Disagreement probes are in `{lattice,stratified}-disagreements-cid22-2026-09-26/`
on r5900xt and Mac. On the selected beneficial/harmful point-label tails, the
candidate value at Butteraugli's peak, normalized by its p1 score ratio, has
median 0.88279 across 79 lattice maps and 0.66758 across 64 stratified maps.
This measures spatial peak discrepancies after global scale removal, not a
representative corpus statistic or participant significance.

KADID pixels and raw ratings also exist on `dev.lan` (`ssh dev`) at
`/mnt/v/dataset/kadid10k/`. The raw-rating SHA-256 matches lilith:
`ba06cbe6c5783ad3a5aa84b13458a703222e8d98c4851a9393d306351a997bab`;
the label CSV is `573e2ed98fdaa2a5aed7b50ad716906e125e1001ed0e97c8e93e79d7b997ba99`.
Use this existing copy instead of transferring the image corpus. The separate
`/mnt/v/datasets/kadid10k/` directory contains raw-data material, not the same
image-root layout.

The raw crowd export contains TID controls and KADID stimuli using a different
distortion-ID order. The mapping in `prepare_kadid_opinions.py` reproduces all
10,125 published means within their printed rounding intervals. The published
`var` field matches population **standard deviation** for every mapped image,
not variance. The user approved correcting `prepare_human.py` and its incorrect
sigma test on 2026-09-26. Previously prepared KADID sigma metadata must be
regenerated; no KADID Margarine quality run used it. Raw data have 30–33 eligible
ratings per image and six repeated worker/image observations. Preserve them for
reconciliation and participant-cluster analysis; do not silently trim to thirty.

The corrected dev audit (`kadid-opinions-reconciled-2026-09-26/`) reproduces
all published means and population standard deviations with zero mismatches.
It retains 304,406 eligible observations from 2,058 workers. The six repeated
worker/image observations have distinct judgment IDs and equal ratings one
second apart. Preserve those observations together in participant resamples;
they are not six independent additional participants. The separate image audit
verified 10,206 files totaling 3,070,504,062 bytes. Two published dispersions
are zero. The approved evaluator correction accepts zero dispersion and marks
the complete supplied-sigma panel unavailable instead of dividing by zero;
ordinary panels retain every image.

KADID images were staged from dev to r5900xt under
`human-corpora-2026-09-26/kadid/` because dev's data volume had only 22 GB free.
The source audit is `kadid-input-r5900-2026-09-26/`; scoring must verify that
audit's image hashes and pair-manifest hash before computing results.

The separate `stable-peak` experiment (`8d99ccec`) also fails the AIC4
rank-loss screen: [max SROCC loses 0.01319081](benchmarks/margarine_stable_peak_aic4_2026-09-26.tsv),
and p6 KROCC loses 0.01667781. Its p6 SROCC loss is 0.00983878; that does
not excuse the other failures. Full maps remain on r5900xt in
`stable-peak-aic4-2026-09-26/`; no resource qualification was run.

The [whole-image lattice timing diagnostic](benchmarks/margarine_lattice_whole_1mp_2026-09-26.json)
at 1 MP with 1,024-row interiors gives teacher/candidate metric means
171.480/124.575 ms over twenty rounds, or 1.3765×. Decode-inclusive means
are 192.174/140.659 ms. It is slower than the earlier 128-row path; removing
strip overlap did not produce the hoped-for speedup. This single-size
diagnostic has no fresh RSS measurement and does not qualify resource use.

KADID lattice scoring (`f5137a75`) completed all 10,125 pairs after verifying
the staged input audit. All maps and scalar norms remain on r5900xt in
`lattice-kadid-2026-09-26/`. The [pooled panel](benchmarks/margarine_lattice_kadid_2026-09-26.tsv)
has largest SROCC loss 0.00532125 (max); all five pooled SROCC/KROCC losses
are below 0.01. This does not imply every distortion family passes:
[published distortion 15](benchmarks/margarine_lattice_kadid_distortion15_2026-09-26.tsv)
loses 0.01040244 SROCC for max and 0.01107207 for p6. Full family and source
panels remain with the participant artifacts below and in the Mac metadata copy.

Restricting the same participant disagreements to KADID's published JPEG and
JPEG2000 distortion classes gives [35 max-norm reversals among 2,025 cross-codec pairs](benchmarks/margarine_lattice_kadid_compression_2026-09-26.tsv),
with four pointwise 95% harmful decreases and none under the simultaneous
family bounds. p1/p2/p3/p6 have no pointwise harmful decreases in that subset.
The denominator includes every combination of five levels per codec for each
of 81 sources. It is not matched by bitrate and cannot replace encoder-choice
acceptance. Class identities follow the [KADID distortion catalog](https://database.mmsp-kn.de/kadid-10k-database.html).

The [participant diagnostic](benchmarks/margarine_lattice_kadid_participants_2026-09-26.tsv)
uses 2,000 worker-cluster bootstrap draws over 2,058 workers and 304,406
observations, retaining all 10,125 images. Among 627,750 within-source pairs,
max pooling reverses 5,813; 2,784 have positive lower pointwise 95% bounds,
and 1,274 have positive simultaneous-family lower bounds. The corresponding
p3 counts are 2,251, 1,091 and 514. These are statistically supported ranking
harms, not matched-rate encoder-choice rates. The simultaneous radius is
1.42396110 native rating units and covers all within-source pairs. Provenance,
seed, binary and output hashes are in the [measurement record](benchmarks/margarine_lattice_kadid_2026-09-26.meta.json).
Bootstrap means and full disagreements remain on r5900xt in
`lattice-kadid-participants-2026-09-26/`; maps are not yet mirrored to Tower.
The complete supplied-sigma panels are unavailable because zero dispersion
(two KADID and twenty-two CSIQ images) makes sigma-normalized residuals
undefined. The user approved accepting those observations on 2026-09-26;
ordinary panels retain them. No joint quality/resource acceptance is established.

Direct planar ingress (`0cef9206`) preserves all 300 AIC4 maps and all five
scalar norms [exactly](benchmarks/margarine_planar_aic4_parity_2026-09-26.json)
against frozen lattice results. Its [four-size resource sweep](benchmarks/margarine_planar_resources_2026-09-26.tsv)
still fails the goal: 2.228×/1.669× metric speed at 1 MP/8.44 MP, with
process RSS fractions 25.102%/14.134%. Decode-inclusive speedups are
2.092×/1.640×. At 64² the candidate uses 6,180,864 bytes versus teacher
5,632,000 bytes, failing the small-image memory condition. All sizes reached
at least twenty timing rounds without an unreliable flag. The guard recorded
peak RSS 1.57 GiB, minimum available 56,807 MiB and peak load 2.53;
[metadata](benchmarks/margarine_planar_resources_2026-09-26.meta.json)
retains timing fits, inputs, binary identity and configuration. Full records
remain on r5900xt and the Mac in `planar-resources-2026-09-26/`.

The lattice [human-quality rank-band diagnostics](benchmarks/margarine_lattice_quality_bands_2026-09-26.tsv)
cover five bands for each norm on AIC4, CID22, CSIQ and KADID, retaining tied
targets together and every row once. The largest observed SROCC loss is
0.04101139 in the lowest-quality AIC4 max band (60 pairs). Other conditional
panels also exceed 0.01, despite all four pooled screens passing. These are
narrow-range point diagnostics without source-cluster confidence intervals;
they do not establish a new acceptance rule. Complete six-stat/composite panels
are in Mac `lattice-quality-bands-2026-09-26/`, with [hashes and evaluator identity](benchmarks/margarine_lattice_quality_bands_2026-09-26.meta.json).

Frozen stratified correction (`76728638`) completed all 10,125 KADID pairs
using the already verified teacher ledger. Its [pooled panel](benchmarks/margarine_stratified_kadid_2026-09-26.tsv)
has largest SROCC loss 0.00693954 (max); all five pooled rank losses remain
below 0.01. Its [participant disagreement counts](benchmarks/margarine_stratified_kadid_participants_2026-09-26.tsv)
are larger than lattice: 16,409 max reversals, 6,722 pointwise harmful decreases
and 2,825 simultaneous-family harmful decreases among 627,750 source-local
pairs. p3 counts are 10,562, 4,290 and 1,825. Full maps remain in r5900xt
`stratified-kadid-2026-09-26/`; bootstrap outputs and panels remain in
`stratified-kadid-participants-2026-09-26/` with a compact Mac copy.
The [measurement record](benchmarks/margarine_stratified_kadid_2026-09-26.meta.json)
pins both scorer and evaluator provenance. These counts are not matched-rate
encoder choices and do not establish the requested choice gate.

The original AIC4 sample CSV also supplies marginal 95% JND intervals;
these are now checked against all 300 frozen score identities. Among 7,500
within-source cross-codec pairs, max has 74 [lattice](benchmarks/margarine_lattice_aic4_intervals_2026-09-26.tsv)
versus 191 [stratified](benchmarks/margarine_stratified_aic4_intervals_2026-09-26.tsv)
reversals whose candidate lower JND bound exceeds the teacher upper bound.
For p3 the corresponding counts are 19 and 113. Interval non-overlap is
reported literally: no paired-difference or simultaneous-coverage claim is
made. The sample's full-resolution files on lilith are PNG reconstructions,
not the original codec bitstreams, and this report has no matched byte budgets.
The [source methodology](https://arxiv.org/html/2504.06301v1) describes resampling
BTC/PTC responses and reconstructing scales. Original CSV and README links
remain preserved in Mac `aic4-sample-audit-2026-09-25/`.

Combining planar ingress with 256-column tiles (`0b13ee78`, 128-row interiors)
reduces [process RSS](benchmarks/margarine_planar_tiles_resources_2026-09-26.tsv)
to 14.80% at 1 MP and 7.58% at 8.44 MP. Metric speedups remain 1.902×
and 1.825×, so it fails the speed target. At 64², RSS is 6,152,192 versus
5,615,616 bytes for teacher, also failing the small-image memory condition.
The original tile/strip bitwise tests pass; no new corpus map run qualifies
this combination yet. Every size has at least twenty interleaved rounds.
The [metadata](benchmarks/margarine_planar_tiles_resources_2026-09-26.meta.json)
records the full size curve and timing fits. Guard peak RSS is 1.55 GiB,
minimum available RAM 56,682 MiB and peak load 1.46. Raw measurements remain
in `planar-tiles-resources-2026-09-26/` on r5900xt and the Mac.

The stratified candidate also completed all 866 CSIQ pairs with the audited
BMP-capable frozen binary (`4c09ec0e`). Its [ordinary panel](benchmarks/margarine_stratified_csiq_2026-09-26.tsv)
has largest SROCC loss 0.00689496 (max); all five pooled SROCC/KROCC losses
stay below 0.01. Max Z-RMSE rises from 0.55586309 to 0.56558895. This replay
explicitly requests ordinary panels only; supplied-sigma panels are not
evaluated, and no zero-dispersion rows are discarded. Native maps remain on
r5900xt in `stratified-csiq-2026-09-26/`; full ordinary panels and
[evaluation provenance](benchmarks/margarine_stratified_csiq_2026-09-26.meta.json)
are in `stratified-csiq-panels-2026-09-26/` on r5900xt and the Mac.

Anchoring each reduced reference sample to its block average (`f240e625`)
is rejected by the [AIC4 panel](benchmarks/margarine_anchored_rejected_2026-09-26.tsv).
Max/p3/p6 SROCC losses are 0.04058001/0.01064723/0.02350737; p2 KROCC
also loses 0.01030100. Fixing the reference sample while retaining the selected
native error vector did not preserve the required rankings. Full maps remain
on r5900xt in `anchored-aic4-2026-09-26/`; compact metadata and panels are
copied to the Mac. No resource qualification or additional corpus run was made.

The rolling horizontal Gaussian buffer (`0afa1014`) has a
[four-size resource sweep](benchmarks/margarine_stream_blur_resources_2026-09-26.tsv):
metric speedups are 2.332× at 1 MP and 1.671× at 8.44 MP, with process RSS
fractions 25.491% and 13.582%. At 64² RSS is 6,053,888 versus 5,910,528 bytes
for teacher. It still fails speed, 1 MP memory, and tiny memory requirements.
Guard peak RSS was 1.60 GiB, minimum available RAM 56,643 MiB, peak load 1.48.
The [AIC4 replay](benchmarks/margarine_stream_blur_aic4_2026-09-26.tsv)
retains the lattice corpus SROCC/KROCC values for all five norms. All 300 map
hashes change through rounding; the largest absolute scalar change is
2.1457672119140625e-6 (max). This is not bit parity. Full maps remain on
r5900xt in `stream-blur-aic4-2026-09-26/`; resource records and compact AIC
metadata are also on the Mac. Single-region pool retention was subsequently
changed in `21b19447` and is being measured separately.

Disabling unused scratch retention on single-region scales (`21b19447`)
passes the tiny/small comparisons in the [four-size sweep](benchmarks/margarine_single_region_resources_2026-09-26.tsv):
64² process RSS is 5,705,728 versus teacher 6,053,888 bytes and metric speedup
is 2.045×; 256² RSS fraction is 74.58% and speedup 1.652×. The larger-image
requirements remain unmet: 1 MP/8.44 MP speedups are 2.249×/1.692× and RSS
fractions 25.20%/13.93%. All sizes have at least twenty interleaved rounds.
Full records remain in `single-region-resources-2026-09-26/` on r5900xt and Mac.

Fusing the three lattice Malta reconstructions with accumulation (`bde9f310`)
[retains every AIC4 map hash and scalar](benchmarks/margarine_fused_malta_aic4_parity_2026-09-26.json)
from the rolling-Gaussian run. The [resource sweep](benchmarks/margarine_fused_malta_resources_2026-09-26.tsv)
measures 2.444×/1.700× speed at 1 MP/8.44 MP and process RSS fractions
26.01%/14.26%; it does not meet the requested larger-image targets. Tiny and
small time/RAM remain below teacher in this run. Full maps are in r5900xt
`fused-malta-aic4-2026-09-26/`; resources and compact replay metadata also
exist on the Mac. Fewer live map intermediates did not lower measured peak RSS
in this run; the measured process peak, not a buffer count, remains the gate.

Moving the fine Gaussian band onto a reduced grid (`f2e0b55a`) fails the
[AIC4 rank screen](benchmarks/margarine_coarse_gaussian_rejected_2026-09-26.tsv):
max KROCC loses 0.01132664 and p6 KROCC loses 0.01123746. All five SROCC
losses remain below 0.01, but that does not excuse the KROCC failures.
No resource qualification or further corpus evaluation was run. Full maps
remain in r5900xt `coarse-gaussian-aic4-2026-09-26/`; compact metadata and
panels also exist on the Mac.

The `row-psycho` experiment pulls linear RGB, opsin and frequency rows through
bounded per-stage caches and then uses the existing local Malta/mask scorer.
Its cache support includes downstream lookahead and both sides of scoring-strip
overlap. Tests compare all ten frequency planes exactly against shared source
for strided RGB16, odd/tiny dimensions and both scales, and verify each stage
produces each row once across overlapping strips. This is a scheduling change;
corpus parity and process resources still require measurement.

The first row-frequency schedule (`30733a96`) [preserves all 300 AIC4 maps and
five scalar norms exactly](benchmarks/margarine_row_psycho_aic4_parity_2026-09-26.json)
against the fused rolling-Gaussian build. Its [initial resource sweep](benchmarks/margarine_row_psycho_resources_2026-09-26.tsv)
fails: 1 MP/8.44 MP metric speedups are 1.583×/1.437× and process RSS fractions
37.59%/18.39%; 64² is slower than teacher (0.918×). The initial caches retained
the entire downstream delay. A subsequent execution change sizes caches by
direct-consumer delay; the unchanged tests still verify exact planes and no
row recomputation. That change needs its own measurements. Full initial maps
remain on r5900xt in `row-psycho-aic4-2026-09-26/`; resources and compact replay
metadata also exist on the Mac. Guard peak RSS was 1.56 GiB, minimum available
RAM 56,641 MiB, peak load 1.60.

Direct-consumer row caches and vectorized expansion (`61a9fd95`) retain
[all AIC4 maps/scalars exactly](benchmarks/margarine_row_cache_aic4_parity_2026-09-26.json).
The [three-fresh-process-per-arm resource sweep](benchmarks/margarine_row_cache_resources_2026-09-26.tsv)
reports the largest peak for each arm: 1 MP/8.44 MP speedups are 2.471×/2.371×,
RSS fractions 32.06%/15.95%. Tiny RSS still exceeds teacher (104.64%).
The [1 MP instruction profile](benchmarks/margarine_row_cache_profile_2026-09-26.txt)
identifies row reduction as 18.65% of 1,490,651,174 instructions, followed by
native Malta evaluation at 13.85%. That profile prompted fixed-width row
reduction; no throughput improvement is inferred from instruction counts.
Full profile data remain in r5900xt `row-cache-profile-2026-09-26/` with input
hashes and build commit; resources and compact AIC replay metadata also exist
on the Mac. Guard peak RSS was 1.60 GiB, minimum available RAM 56,799 MiB,
peak load 1.49.

Vectorized row reduction, single-region bypass and strip-shape recycling
(`288b3fd7`) [preserve all AIC4 maps and norms](benchmarks/margarine_row_reduce_aic4_parity_2026-09-26.json).
The [resource sweep](benchmarks/margarine_row_reduce_resources_2026-09-26.tsv)
reaches 3.006×/2.648× metric speed at 1 MP/8.44 MP, with largest-of-three
fresh-process RSS fractions 24.62%/13.25%. Decode-inclusive speedups are
2.644×/2.452×. Tiny and small measurements remain below teacher for both
resources. This passes memory on these crops, but still fails 4× speed and
has no broader content or matched-rate human-choice qualification. Full maps
remain in r5900xt `row-reduce-aic4-2026-09-26/`; compact results and resources
also exist on the Mac. Guard peak RSS was 1.65 GiB, minimum available RAM
56,350 MiB, peak load 4.96.

Streaming frequency rows within 512-column tiles (`1072eeb6`, measured from
`1e2554fe`) [preserves every AIC4 map and scalar](benchmarks/margarine_row_tiles_aic4_parity_2026-09-26.json).
Its [three-trial process-resource sweep](benchmarks/margarine_row_tiles_resources_2026-09-26.tsv)
measures 3.031×/3.064× metric speed at 1 MP/8.44 MP and RSS fractions
18.67%/8.08%. Decode-inclusive speedups are 2.664×/2.732×. Tiny and small
remain below teacher for time and RAM. The 4× target is still unmet.
Full maps remain in r5900xt `row-tiles-aic4-2026-09-26/`; resources and compact
replay metadata also exist on the Mac. The updated [untiled instruction
profile](benchmarks/margarine_row_reduce_profile_2026-09-26.txt) records
1,152,280,842 instructions, with row reduction reduced to 2.18% and Malta
now the largest metric stage at 17.92%. These instruction counts do not
substitute for the separately measured timing results.

The rolling Malta phase rows (`7e2fac48`) [preserve all 300 AIC4 maps and
scalar norms](benchmarks/margarine_phase_rows_aic4_parity_2026-09-26.json).
The [resource sweep](benchmarks/margarine_phase_rows_resources_2026-09-26.tsv)
measures 3.194×/2.756× metric speed at 1 MP/8.44 MP and RSS fractions
24.96%/13.26%. Tiny RSS is 101.68% of teacher, failing that condition.
This untiled run does not meet the goal. Each peak is the largest of three
fresh-process measurements. Full maps remain on r5900xt in
`phase-rows-aic4-2026-09-26/`; compact replay and resource records also exist
on the Mac. Guard peak RSS was 1.58 GiB, minimum available RAM 56,596 MiB,
peak load 1.40.

After the approved zero-dispersion correction (`0e9ec80a`), [replayed panels](benchmarks/margarine_zero_sigma_panels_2026-09-26.tsv)
retain all 10,125 KADID and 866 CSIQ pairs for lattice and stratified across
all five norms. Ordinary SROCC, PLCC, KROCC, OR, PWRC, Z-RMSE and composites
are measured. Every whole-corpus supplied-sigma panel explicitly reports
unavailable, with full sigma counts; no epsilon or row removal substitutes
for the two KADID and twenty-two CSIQ zero dispersions. [Provenance and hashes](benchmarks/margarine_zero_sigma_panels_2026-09-26.meta.json)
pin the unchanged score ledgers and new evaluator. Full replay outputs are
in Mac `{lattice,stratified}-{kadid,csiq}-zero-sigma-panels-2026-09-26/`.

Combining column tiling and rolling Malta (`7bc82d8f`) [preserves all AIC4
maps and norms](benchmarks/margarine_phase_tiles_aic4_parity_2026-09-26.json).
The [resource sweep](benchmarks/margarine_phase_tiles_resources_2026-09-26.tsv)
measures 3.062×/3.231× metric speed and 18.89%/8.15% process RSS at
1 MP/8.44 MP. Decode-inclusive speedups are 2.706×/2.894×. Tiny and small
time/RAM remain below teacher, but 4× speed remains unmet. Full maps remain
on r5900xt in `phase-tiles-aic4-2026-09-26/`; compact replay and all resource
records also exist on the Mac. Each peak is the largest of three fresh
processes. Guard peak RSS was 1.54 GiB, minimum available RAM 56,610 MiB,
peak load 1.34.

Fixed-width expansion (`a121ab5c`, measured from `4363b038`) [retains all
AIC4 maps and scalar norms](benchmarks/margarine_expand_lanes_aic4_parity_2026-09-26.json).
The [128-row resource run](benchmarks/margarine_expand_lanes_resources_2026-09-26.tsv)
measures 3.281×/3.047× metric speed at 1 MP/8.44 MP, with RSS fractions
18.17%/8.10%. The [256-row run](benchmarks/margarine_expand_lanes_rows256_resources_2026-09-26.tsv)
is slower at 2.878×/2.974×, with RSS fractions 24.64%/8.98%. Both retain
small-image resource use below teacher. Neither qualifies 4× speed; the
change does not establish a uniform speed improvement over the preceding
phase-tiles measurement. Full maps remain on r5900xt in
`expand-lanes-aic4-2026-09-26/`; compact replay and all resource records also
exist on the Mac. Raw resource directories are `expand-lanes-resources-2026-09-26/`
and `expand-lanes-rows256-resources-2026-09-26/`.

The wider streaming FIR accumulator schedule (`6186fc47`) [preserves all
AIC4 maps and norms](benchmarks/margarine_fir_lanes_aic4_parity_2026-09-26.json).
The [resource sweep](benchmarks/margarine_fir_lanes_resources_2026-09-26.tsv)
measures 3.358×/3.221× metric speed and 18.84%/8.13% process RSS at
1 MP/8.44 MP. Decode-inclusive speedups are 2.940×/2.884×; small-image
time/RAM remain below teacher. This still fails 4× speed. Full maps and
profiles remain in r5900xt `fir-lanes-{aic4,profile}-2026-09-26/`; compact
replay, flat profile and complete resources also exist on the Mac. The
[1 MP profile](benchmarks/margarine_fir_lanes_profile_2026-09-26.txt) now places
rolling Malta at 16.70%, horizontal Gaussian at 4.66% and vertical Gaussian
at 2.50% of instructions, including decoding. Tiling differs from the older
untiled profile, so these percentages are not a controlled stage speedup.
Guard peak RSS was 1.55 GiB, minimum available RAM 56,537 MiB, peak load 1.55.

The [official LIVE Release 1 JPEG](https://live.ece.utexas.edu/research/quality/JPEG/readme.txt)
and [JPEG2000](https://live.ece.utexas.edu/research/quality/JPEG2000/readme.txt)
archives contain individual raw responses, published processed observer
matrices, processing MATLAB code and bitrate tables. The downloaded originals
and metadata are in Mac `live-release1-metadata-2026-09-26/`; extracted images
are in `live-release1-input-2026-09-26/` on Mac and r5900xt. Release 1 is a
separate target from Release 2 DMOS. Its processing independently normalizes
four codec/session cohorts. The new adapter retains those cohort boundaries,
checks published means/sample deviations against nonzero processed opinions,
and explicitly excludes documented zero-bitrate lossless controls. Bootstrap
results from these matrices are conditional on published normalization and
outlier selection; they cannot establish cross-cohort or Release 2 uncertainty.

The [LIVE Release 1 panels](benchmarks/margarine_live1_quality_2026-09-26.tsv)
cover 344 compressed stimuli in four separately normalized codec/session
cohorts. Matching candidate and teacher norms exposes failures: faithful max
KROCC loses 0.01176313 in JPEG2000 session 1; stratified max SROCC/KROCC
lose 0.02145588/0.04314940 there. These results supersede any inference of
all-dataset rank qualification from the earlier four pooled corpus screens.
The [matched-rate choices](benchmarks/margarine_live1_choices_2026-09-26.tsv)
use every observed source-local bitrate budget and 2,000 observer-column
bootstrap draws per cohort. Faithful max has one simultaneously supported
harmful choice among 87 JPEG2000 session-1 budgets (1.1494%). Stratified max
has one pointwise-supported harmful choice among 78 JPEG session-2 budgets
(1.2821%), but none under that cohort's simultaneous bound. These are
within-codec decisions, conditional on published processing; they do not
establish cross-codec choice safety. [Provenance](benchmarks/margarine_live1_2026-09-26.meta.json)
pins score ledgers, audited opinions and evaluator identities. Full maps
remain on r5900xt in `{faithful,stratified}-live1-2026-09-26/`; compact panels,
bootstrap arrays and choice records also exist on the Mac in
`{faithful,stratified}-live1-participants-2026-09-26/`. Tower backup remains
outstanding. Neither candidate currently meets the combined goal.

The `peak-stratified` experiment isolates correction-region selection: its
first exact tile contains the largest proxy-map value, while its second tile
still represents the remaining image's mean cubic error. The previous
stratified selector ranked the first tile by total cubic energy. A focused
selection test verifies that an isolated peak wins over a higher-energy tile
and that area weights still cover the image. Existing finite-halo and ingress
tests remain unchanged. Quality and resource qualification require separate
measurements; no improvement is inferred from this selection rule.

The [fixed-teacher-max pooling diagnostic](benchmarks/margarine_fixed_max_diagnostics_2026-09-26.tsv)
compares each stratified norm against Butteraugli's default max, without
changing the acceptance rule. Stratified p2 has no participant-supported
harmful LIVE choices in the [observed budgets](benchmarks/margarine_fixed_max_live1_choices_2026-09-26.tsv),
but loses pooled SROCC 0.073367 on KADID and 0.029822 on CSIQ against that
baseline. Switching the primary scalar therefore does not solve the combined
quality problem. All five norms and all ordinary panel differences remain
reported. [Input hashes and scope](benchmarks/margarine_fixed_max_diagnostics_2026-09-26.meta.json)
preserve the alternative comparison explicitly; these results are not
replacement matching-norm acceptance measurements.

Peak-stratified selection (`e3e402a6`) is rejected by the [LIVE rank
screen](benchmarks/margarine_peak_stratified_live1_rejected_2026-09-26.tsv).
JPEG2000 session-1 max SROCC/KROCC losses are 0.02735669/0.05346879,
worse than the frozen stratified candidate on that cohort. All 344 maps
remain in r5900xt `peak-stratified-live1-2026-09-26/`; compact panels and
participant outputs also exist on the Mac. [Measurement metadata](benchmarks/margarine_peak_stratified_live1_rejected_2026-09-26.meta.json)
pins the score ledger and binary. No resource qualification was run for this
rejected variant. Isolating peak selection did not repair the quality loss.

The `native-gaussian` diagnostic keeps the row/column schedule and sampled
Malta bank but disables Gaussian grid reduction. It returns factor one with
the original sigma for every blur. Existing exact-plane, seam, stride and
row-reuse tests pass without relaxed expectations; this control isolates
blur reduction from directional-filter sampling and has no speed claim.

The [LIVE approximation controls](benchmarks/margarine_live1_ablation_quality_2026-09-26.tsv)
separate blur reduction from Malta sampling. `bounded` (`e3e402a6`) retains
reduced-grid Gaussian blurs but evaluates the complete native Malta outputs;
`native-gaussian` (`961fac3c`) retains sampled Malta but uses native-grid
Gaussian blurs. Both pass all twenty matching-norm SROCC/KROCC cohort
screens at loss <=0.01. The former also has zero pointwise or simultaneously
supported harmful choices in the [328 observed budgets per norm](benchmarks/margarine_live1_bounded_choices_2026-09-26.tsv).
The previously measured combination failed, so this experiment implicates an
interaction rather than either approximation alone. [Provenance](benchmarks/margarine_live1_ablation_2026-09-26.meta.json)
pins both binaries and uses the same participant draws for the choice
comparison. Full maps remain on r5900xt in `{bounded,native-gaussian}-live1-2026-09-26/`;
compact records also exist on the Mac. These controls do not yet establish
the combined quality/resource goal.

The `native-mask` control narrows native-grid Gaussian filtering to
`MASK_RADIUS`; frequency-separation blur reduction and sampled Malta remain.
It retains the existing shared sigma and coefficients and introduces no
fitted constant. The unchanged row-plane, seam and stride tests pass. This
isolates masking from frequency blur reduction; it is not yet a quality or
resource qualification.

The native-Gaussian control (`961fac3c`) also passes the [AIC4 pooled
rank screen](benchmarks/margarine_native_gaussian_aic4_2026-09-26.tsv), but
retains one simultaneously supported harmful LIVE max choice among 87
JPEG2000 session-1 budgets. [Choice counts and source metadata](benchmarks/margarine_native_gaussian_quality_2026-09-26.meta.json)
retain that failure. The [resource curve](benchmarks/margarine_native_gaussian_resources_2026-09-26.tsv)
measures 2.178×/2.386× metric speed and 19.15%/8.16% process RSS at
1 MP/8.44 MP; decode-inclusive speeds are 2.035×/2.268×. Tiny and small
time/RAM remain below teacher. All sizes have at least twenty interleaved
rounds and largest-of-three process peaks. Guard peak RSS was 1.54 GiB,
minimum available RAM 56,577 MiB and peak load 1.56. Full maps remain in
r5900xt `native-gaussian-aic4-2026-09-26/`; compact quality, choice and
resource artifacts also exist on the Mac. This control fails speed and the
per-cohort choice screen despite passing both measured rank panels.

The `full-malta` control combines bounded frequency-row caches and column
tiling with the shared complete Malta kernel. Both the whole-pipeline control
and streaming path select that same bank; the first draft wired only the
streaming arm, and the unchanged parity tests caught the mismatch. The
corrected wiring passes the existing channel/asymmetry, seam and strided-plane
tests. Gaussian grid reduction remains enabled. This provides a scheduled
counterpart of `bounded` for measuring the cost of restoring native Malta
outputs, without assuming its runtime from the sampled-kernel profile.

Native-mask (`86d43a83`) passes the [AIC4 pooled rank screen](benchmarks/margarine_native_mask_quality_2026-09-26.tsv)
but still fails LIVE: max KROCC loses 0.01122845 in JPEG2000 session 1,
and p6 KROCC loses 0.01098615 in JPEG session 2. Keeping only the masking
blur on the native grid does not repair the combined approximation. [Binary
and ledger provenance](benchmarks/margarine_native_mask_quality_2026-09-26.meta.json)
are retained; full maps remain in r5900xt `native-mask-{aic4,live1}-2026-09-26/`
with compact records also on the Mac. No resource qualification was run for
this rejected control.

The row opsin transform now exposes eight pixels through fixed-size channel
slices while retaining each pixel's arithmetic. Existing exact frequency-plane
and map-seam tests pass. This targets the previously measured opsin stage
(6.02% of the FIR-lanes instruction profile); a speed improvement requires
new measurement and is not inferred from the source rewrite.

`full-malta,coarse-gaussian` has the distinct identity `coarse-full-malta`.
It restores every native Malta output while applying the existing analytical
reduced-grid blur geometry to the fine band as well. This reuses the prior
coarse-blur control without selecting new weights or using human labels in
scoring. The unchanged stage/seam tests pass; corpus and resource measurements
must determine whether restoring Malta resolves that control's earlier
quality failure.

Full-Malta streaming (`3a42d971`) [passes the twenty LIVE cohort rank
screens](benchmarks/margarine_full_malta_live1_2026-09-26.tsv) and has
[zero participant-supported harmful choices](benchmarks/margarine_full_malta_live1_choices_2026-09-26.tsv)
in its 328 observed budgets per norm. Its [resource sweep](benchmarks/margarine_full_malta_resources_2026-09-26.tsv)
measures 2.659×/2.943× metric speed and 17.99%/7.99% process RSS at
1 MP/8.44 MP. Decode-inclusive speeds are 2.399×/2.667×; small sizes stay
below teacher resources. The [instruction profile](benchmarks/margarine_full_malta_profile_2026-09-26.txt)
places the native Malta dispatcher at 605,585,808 instructions (40.40% of
1,498,962,789 including decoding). Full maps/profile remain on r5900xt in
`full-malta-{live1,profile}-2026-09-26/`; compact records also exist on the Mac.
[Quality provenance](benchmarks/margarine_full_malta_quality_2026-09-26.meta.json)
pins the participant join; resource metadata pins three fresh-process peaks
per arm. This candidate still fails 4× speed.

The experimental full-Malta kernel now supplies nine fixed-size row windows
to the existing complete stencil bank, retaining the shared nonlinear
scaled-difference transform. A direct comparison against the shared native
kernel verifies every output at tiny/odd/vector-tail dimensions, both banks
and asymmetric weights. Existing seam and frequency tests remain unchanged.
No speed gain or corpus bit parity is inferred before replay and measurement.

The fixed-width opsin rewrite (`fde7c208`, measured from `a357f0cf`) is
reverted after a measured regression. [All 344 LIVE maps and five scalar
norms remain identical](benchmarks/margarine_opsin_lanes_live1_parity_2026-09-26.json),
but [metric speed](benchmarks/margarine_opsin_lanes_resources_2026-09-26.tsv)
is only 1.957×/2.136× teacher at 1 MP/8.44 MP, versus 2.659×/2.943× for
the prior full-Malta build. Candidate means rose from 51.31/446.46 ms to
69.09/602.91 ms while teacher means stayed within the measured baseline
range. The row transform's previous loop is restored; the separate Malta
row-window kernel remains for its own measurement. Raw maps and resource
records remain on r5900xt in `opsin-lanes-{live1,resources}-2026-09-26/`, with
compact records on the Mac. No test expectation was relaxed.

The LIVE Release 1 original JPEG/JPEG2000 ZIPs and extracted metadata now
have a [verified Tower mirror](benchmarks/margarine_live1_originals_tower_2026-09-26.pointer.json)
at `/mnt/user/coefficient/output/margarine/live-release1-metadata-2026-09-26/`.
Tower's cache returned ENOSPC despite array space, so the completed copy was
written through `/mnt/user0/` and verified through the canonical `/mnt/user/`
share. Both complete ZIP hashes and both source-manifest hashes match the
Mac. The failed partial cache copy was preserved on the array with the suffix
`.failed-enospc`; no source data or share configuration was changed. Generated
score-map backups remain separate and outstanding.

## Primary scalar acceptance clarification (2026-09-26)

The user selected primary-scalar comparison against Butteraugli’s default
maximum score for the 0.01 rank-loss gate. Every pooling variant remains in
the reports; earlier matching-norm screens remain diagnostics. The user has
no additional AIC4 original bitstreams or bitrate table, so its matched-byte
choice result remains unavailable. Human-ranking and published-interval
diagnostics retain all audited pairs.

## Native Malta tail repair and replay (2026-09-26)

The separate final-block path introduced in `a85ba912` fails an expanded
exact comparison on optimized x86 (first reproduction: 3×5, LF bank).
`2d9d736a` uses one overlapping-block loop and adds release tests to the
research recipe and CI. [All 344 LIVE maps and all five norms reproduce the
prior full-Malta implementation exactly](benchmarks/margarine_malta_uniform_live1_parity_2026-09-26.json).
The [fresh resource sweep](benchmarks/margarine_malta_uniform_resources_2026-09-26.tsv)
measures 2.736×/2.690× scoring speed and 17.566%/8.046% total-process RSS
at 1 MP/8.44 MP. Small-image time and RSS remain below the teacher on the
measured crop family. Speed still fails the target; corpus parity does not
qualify resources. Full artifacts live in r5900xt
`malta-uniform-{live1,resources}-2026-09-26/`, with compact records on the Mac.
The old `malta-windows-*` and `coarse-full-*` runs from `f6cbfe73` retain the
incorrect tail path and must not be used as accepted candidate results.

## Malta row-sampling control

The `row-malta` experiment keeps the complete native input lattice and all
six sixteen-orientation Malta banks. It evaluates all columns on alternating
rows, then linearly reconstructs the missing response rows before the shared
scoring pipeline. The full-Malta entry point and its exact assertions remain
separate. This analytical ablation has no fitted constants and no quality or
resource qualification before corpus scoring and measurements.

Row sampling (`37fbc16e`) passes the four LIVE max rank screens, but retains
one statistically harmful JPEG2000 session-1 choice out of 87 budgets
(1.1494%), under both pointwise and simultaneous participant intervals.
The changed choice is `caps.bmp`: img68 instead of img49. The native peak
in img68 is (734,415), score 4.541418; row sampling gives 4.156408 there and
selects a different global peak. The missing peak remains the maximum inside
its 512-column, 128-row core region, motivating a local refinement diagnostic.
This is a rejected candidate, not an accepted relaxation of the choice limit.
[All five LIVE panels and full-Malta AIC4/CSIQ panels](benchmarks/margarine_malta_controls_quality_2026-09-26.tsv)
and [participant counts](benchmarks/margarine_row_malta_live1_choices_2026-09-26.tsv)
retain the measured differences. Full-Malta max rank loss is below 0.01 on
both new corpora; CID22 and KADID scoring remain separate.

[Row-sampling resources](benchmarks/margarine_row_malta_resources_2026-09-26.tsv)
are 3.217×/3.326× scoring speed and 17.793%/8.053% total-process RSS at
1 MP/8.44 MP. The guard reports 1.55 GiB peak RSS, 56,678 MiB minimum
available RAM and 1.73 peak load; reported arm peaks use fresh-process time.
The [256-row full-Malta control](benchmarks/margarine_malta_rows256_resources_2026-09-26.tsv)
measures 2.590×/2.790× and 23.417%/8.822% RSS. Increasing strip height does
not establish the speed target. These resource sweeps use the same crop
family and do not establish content-wide performance.

The full-Malta AIC4 [source-cluster bootstrap](benchmarks/margarine_malta_uniform_aic4_bootstrap_2026-09-26.tsv)
uses 2,000 paired draws of the five source groups, refitting the panel in every
draw. For the primary max scalar, candidate-minus-teacher SROCC has a central
95% interval [-0.001984, 0.002126], and KROCC [-0.002552, 0.004740]. Both lower
bounds exceed -0.01. Every norm and all ten statistics retain their intervals;
there are no undefined draws. These are per-statistic source-cluster intervals,
not a simultaneous multi-corpus result or evidence from more than five sources.
Full outputs and evaluator/input hashes live in Mac
`malta-uniform-aic4-bootstrap-2026-09-26/`.

The fixed full-Malta build `2d9d736a` also has complete CID22 (4,292) and
KADID (10,125) panels in `malta-uniform-cid22-v2-2026-09-26/` and
`malta-uniform-kadid-2026-09-26/` on r5900xt, with compact Mac copies.
[All five ordinary corpus panels](benchmarks/margarine_malta_uniform_cid22_kadid_2026-09-26.tsv)
include the zenstats statistics and composites. Primary max SROCC changes
are −0.000113683 on CID22 and +0.000120502 on KADID; KROCC changes are
−0.000100361 and +0.000174987. These point rankings meet the 0.01 screen.
This does not qualify encoder-choice uncertainty, missing corpora, or speed.

Explicit SIMD full-Malta build `404280ba` has a complete four-size
[resource sweep](benchmarks/margarine_simd_malta_resources_2026-09-26.tsv)
on r5900xt with two threads. At 1 MP and 8.44 MP its metric speedups are
3.369577× and 3.621800×; total-process peak RSS fractions are 18.0387% and
8.0432%. Decode-inclusive speedups are 2.925812× and 3.082626×. At 64²,
peak RSS is 1.007602 of the teacher, so the small-image memory condition
also fails this run. The source, binary, inputs, full raw outputs and fitted
time intercepts/slopes are pinned in the adjacent metadata. This is a resource
improvement experiment, not a goal pass or independent content coverage.

The explicit SIMD full-Malta replay retains identical map hashes and all five
scalar scores on [all 344 LIVE Release 1 pairs](benchmarks/margarine_simd_malta_live1_parity_2026-09-26.json),
relative to fixed build `2d9d736a`. Thus this replay preserves that candidate's
LIVE rankings and zero harmful observed-budget choices. No equivalent replay
on the other corpora is implied. The same SIMD implementation combined with
alternate-row sampling measures 3.629637×/3.804872× at 1 MP/8.44 MP and
18.1818%/8.0119% process RSS; [the resource curve](benchmarks/margarine_simd_row_malta_resources_2026-09-26.tsv)
passes the small-image taper but still misses the larger-image speed target.

The `cd15be17` tile-width diagnostics did not reach 4×. Full SIMD Malta at
768 columns × 128 rows measures 3.497244×/3.600771× at 1 MP/8.44 MP, with
21.1863%/8.6041% process RSS. At 1024 columns × 96 rows these are
3.542057×/3.599026× and 22.0622%/8.6070%. Tiny/small results, timing fits
and raw artifact hashes are in `benchmarks/margarine_simd_tiles*_resources*`.
Both runs use the same source pair across four sizes; neither chooses a new
production geometry. The expanded geometry parity test also passes on x86
in debug and release.

The owned-row-cache build `cb622180` preserves the complete LIVE score/map
[ledger byte for byte](benchmarks/margarine_borrowed_rows_live1_parity_2026-09-26.json)
against SIMD build `404280ba`. Its [resource curve](benchmarks/margarine_borrowed_rows_resources_2026-09-26.tsv)
measures 3.631065×/3.686722× at 1 MP/8.44 MP and 18.2964%/8.0060% RSS.
Tiny and small cases remain below teacher time and RSS in this run. The
remaining shortfall is the large-image speed target. This does not qualify
content diversity or the remaining human-rated corpora.

### Corpus execution geometry audit

The scorer previously invoked the candidate without `--native-strip`; in the
CLI that selects `rows = image.height`. Consequently the older LIVE replay
ledgers above did not exercise `Graph::prepare` or the borrowed row cache.
Their map-identity statements describe full-height execution only. Exact seam
tests cover streaming equivalence, but the corpus-level streamed checks remain
required. `score_manifest.py` now passes and records explicit strip rows
(default 128), alongside tile columns (default 512), and rejects the wrong
reported execution mode. Resource and quality runs must use identical geometry.

The power-of-two row-cache builds retain explicit CPU/RSS measurements in
[the full and alternate-row resource curves](benchmarks/margarine_masked_rows_resources_2026-09-26.tsv),
with raw artifacts under `masked-row-resources-2026-09-26/` and
`masked-sampled-resources-2026-09-26/`. Build `2b04e9e4` has 1 MP mean
speedups of 3.445001× and 3.825383× respectively; neither reaches 4× there.
The full-bank 64² run has RSS fraction 1.002841, so it also misses the small
image memory condition in that measurement. No new resource pass is claimed.

The `6bb371fb` explicit SIMD reconstruction build has the first streamed
full-bank LIVE replay in `vector-expand-full-streamed-live1-2026-09-26/`.
At 128 rows × 512 columns, its [complete ledger](benchmarks/margarine_vector_expand_full_streamed_live1_parity_2026-09-26.json)
is byte-identical to the older full-height SIMD ledger, including all 344 map
hashes and five norms. This run exercises the row cache.

The same build with `simd-malta,row-malta` passes the measured four-size
[mean resource screen](benchmarks/margarine_vector_expand_row_resources_2026-09-26.tsv):
4.370076×/4.252029× at 1 MP/8.44 MP, with 18.6441%/8.1470% total-process
RSS. Both small cases are below teacher time and RSS. This uses one source
pair cropped to four sizes, so content-diverse resource qualification remains.
Its streamed [AIC4 panel](benchmarks/margarine_vector_expand_row_aic4_2026-09-26.tsv)
retains all 300 pairs: primary SROCC loss 0.000442227 and KROCC loss
0.001694537 versus teacher max. The other human corpora and choice diagnostics
are still required for this frozen build and execution geometry.

The frozen `6bb371fb` streamed row-Malta replay now also covers all 344 LIVE
release-1, 866 CSIQ and 4,292 CID22 pairs. The [corpus panels](benchmarks/margarine_vector_expand_row_quality_2026-09-26.tsv)
report all five norms and the full statistic set. Primary SROCC losses are
0.003398255 (CSIQ), 0.004505481 (CID22), and at most 0.002481471 across the four
LIVE sessions; all meet the 0.01 point-estimate screen. [LIVE choice bounds](benchmarks/margarine_vector_expand_row_live1_choices_2026-09-26.tsv)
retain one statistically distinguishable harmful choice among 328 observed
budgets (0.3049%), concentrated in one of 87 JPEG2000 session-1 budgets
(1.1494%). Dataset-level and cohort-level rates are both reported; pooling
sessions does not erase that cohort result. CID22 lacks participant uncertainty.

Additional original inputs were verified on `lilith`: `/mnt/v/dataset/tid2013`
(3,000 pairs), `/mnt/v/dataset/aic3_ctc_epfl` (600 pairs, subjective/estimated
labels kept separate), `/mnt/v/dataset/pipal` (four distortion directories),
and `/mnt/v/datasets/LIVE/databaserelease2`. The committed `prepare_human.py`
adapter audits source images and labels before staging. Audit artifacts are
`extended-human-audits-2026-09-26/` under the Linux artifact root. These paths
identify evaluation inputs; they do not imply completed candidate scoring.

The same streamed build retains all 10,125 KADID pairs in its
[quality panel](benchmarks/margarine_vector_expand_row_kadid_2026-09-26.tsv).
Primary SROCC loss is 0.002757075 and KROCC loss is 0.002280405. All five
norms are retained; the two zero-dispersion images remain in ordinary panels.
The [AIC4 primary source bootstrap](benchmarks/margarine_vector_expand_row_aic4_bootstrap_2026-09-26.tsv)
uses 2,000 draws over five sources: the SROCC delta interval is
[-0.005227962, 0.003264655], while KROCC spans [-0.010790240, 0.003945304].
Thus point estimates pass, but the KROCC interval crosses the -0.01 boundary.
These are per-statistic intervals, with no simultaneous coverage claim.

Optional `simd-opsin` full-Malta build `a0cc0f0d` passes unchanged exact tests
in debug/release on x86 and ARM. Its [resource curve](benchmarks/margarine_opsin_full_resources_2026-09-26.tsv)
measures 4.094713× at 1 MP but 3.828098× at 8.44 MP, with process RSS fractions
18.8706% and 8.1146%. Tiny/small resources remain below the teacher in this
run. It therefore does not meet the complete speed screen and is not selected.
The first resource invocation used the Mac-path crop manifest and failed before
measurement; the completed `opsin-full-resources-v2-2026-09-26/` uses the Linux
manifest. Both artifacts are preserved.

The frozen r5900xt candidate requires glibc 2.39, newer than the WSL host's
runtime. Evaluation on `lilith` therefore uses the already-installed
`ghcr.io/imazen/zenfleet-worker:exec-gpu-cuda13-6d4f9963` image (verified glibc
2.39 and Python 3.12.3), with 12 GiB memory/no-swap and two-CPU caps. The image
ID is recorded beside the extended input audits. Scoring still runs serially
through the same manifest harness and frozen binaries; these are quality runs,
not host performance measurements. The failed native invocation is preserved.

[KonJND-1k](https://github.com/angchen-dev/KonJND-1k) supplies picturewise
noticeability thresholds rather than a MOS for every distorted image. Its
WSL `subjective_ratings.csv` contains compression-level ratings per source;
those values must not be reused as ordinary quality labels for arbitrary
JPEG/BPG files. A threshold-specific evaluation is distinct from the ordinary
TID/AIC3/LIVE/PIPAL panels.

The streamed row candidate's [KADID participant diagnosis](benchmarks/margarine_vector_expand_row_kadid_participants_2026-09-26.tsv)
uses all 304,406 reconciled opinions and 2,000 worker-bootstrap draws. Among
627,750 within-source pairs, primary max reverses 4,315; 1,898 have pointwise
95% human-supported harm and 817 retain harm under the simultaneous envelope.
These are pair-order diagnostics, not bitrate choices. The saved 162 MB draw
matrix permits future fixed-choice joins without rerunning the bootstrap.
The [AIC4 interval diagnosis](benchmarks/margarine_vector_expand_row_aic4_intervals_2026-09-26.tsv)
finds 64 harmful and 51 beneficial primary reversals with separated published
intervals among 8,850 pairs. AIC4 has no original byte-budget table here, so this
also remains a pairwise diagnostic rather than the encoder-choice gate.

The [independent content resource checks](benchmarks/margarine_vector_row_content_resources_2026-09-26.tsv)
use four held-out q50 JPEG pilot pairs. At 1 MP, line art, mixed and photo
measure 4.524022×, 4.214074× and 4.023647×; the native 12.19 MP photo measures
4.343482×. The 1440×900 screenshot misses the speed target at 3.467011×,
although its RSS fraction is 16.3576%. This prevents a content-wide resource
pass for 128-row/512-column execution. These diagnostic cases do not replace
a dense content/quality sweep or establish guarantees at unmeasured geometries.

TID2013 is scored completely (3,000 pairs, 25 sources) in WSL
`vector-expand-row-streamed-tid-container-2026-09-26/`. Primary SROCC improves
by 0.000840982 and KROCC by 0.000376725. The [default-teacher comparison](benchmarks/margarine_vector_row_primary_comparison_2026-09-26.tsv)
now reports every candidate norm against Butteraugli max on each completed
corpus. Alternative norms improve AIC4/CID22 but exceed the allowed rank loss
on CSIQ/KADID, so these measurements do not justify replacing primary max.
