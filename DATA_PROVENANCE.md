# Margarine data index

No trained Margarine model exists. The current candidate is an analytic box-filter
control; feature-extraction measurements are cost probes, not quality predictions.
Human-quality coverage measured so far is AIC4_sample and CID22 validation, not
every required dataset. The total-process memory gate includes decoding and inputs.

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
The CID22 clustered run is in progress at r5900xt
`cluster-cid22-2026-09-26/`; do not treat partial output as complete.

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
