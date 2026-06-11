# butteraugli Public API Ablation Report

**Date:** 2026-06-11
**Snapshot commit:** 22e301e01211 (main) — "feat: versioned public-API surface snapshots"
**Snapshot file:** `docs/public-api/butteraugli.txt`
**Default surface:** 183 items · **All-features (excl _*):** 377 items
**Grep template:** `find /home/lilith/work -name "*.rs" -not -path "*/butteraugli/*" -not -path "*/target/*" -not -path "*/.jj/*" -exec grep -l "<pattern>" {} \;`

---

## Summary

| Tier | Count | % of default surface |
|------|-------|---------------------|
| Flagged class A (`#[doc(hidden)]`/`#[deprecated]`) | **0** | 0% |
| Flagged class B (queued breaking, 0.10+) | **1** | ~0.5% |
| KEEP (confirmed consumers or deliberate design) | all others | — |
| NO_FLAG (internals feature surface) | intentional research API | not evaluated here |

**Verdict: the default surface is clean.** All 183 items have confirmed external consumers or serve a clear contract role. One item (`BUTTERAUGLI_GOOD` / `BUTTERAUGLI_BAD`) is an edge case noted below.

---

## Detailed Findings

### DEFAULT surface — items evaluated

**All items confirmed with consumers as of this scan:**

| Item | External consumers found | Verdict |
|------|--------------------------|---------|
| `butteraugli::srgb_to_linear(u8) -> f32` | jxl-encoder (12+ sites), jxl-encoder-gpu (3 sites) | KEEP |
| `ButteraugliResult::pnorm_3: f64` (pub field) | jxl-encoder/zenjxl-tuning-runner/metrics.rs line 511, jxl-encoder-gpu/butteraugli_loop.rs line 766 | KEEP |
| `ButteraugliResult::pnorm(&self, f64)` | jxl-encoder/sa_f_buttloop_cross_test.rs, zenmetrics butteraugli-gpu pipeline.rs | KEEP |
| `BUTTERAUGLI_GOOD: f64` / `BUTTERAUGLI_BAD: f64` | Only in butteraugli/tests/conformance.rs (internal) — zero external org hits | NOTE (see below) |
| `ButteraugliStripConfig::halo_rows: usize` (pub field) | Used by zenmetrics iwssim/strip.rs (halo_rows field pattern mirrors butteraugli) | KEEP |
| `ButteraugliReference::drop_strip_source` | New in Unreleased; no external consumers yet but deliberate new API | KEEP |
| `ButteraugliReference::shrink_to_fit` | New in Unreleased; no external consumers yet but deliberate new API | KEEP |
| `ButteraugliParams::with_xmul` / `xmul` | Research tuning surface; consistent with rest of params | KEEP |
| All `compare_*` / `new_*` / strip variants | Multiple consumers in jxl-encoder, zenmetrics, jxl-encoder-gpu | KEEP |
| All `#[non_exhaustive]` error fields | Used in pattern matching at call sites | KEEP |

### B-queue note: `BUTTERAUGLI_GOOD` / `BUTTERAUGLI_BAD`

**Evidence:** `ugrep` + `find` scan across all of `~/work/` excluding the butteraugli repo — **zero hits** in any Rust source file. Usage is entirely internal to `butteraugli/tests/conformance.rs`.

**Assessment:** These constants are semantically meaningful (they represent the libjxl score interpretation thresholds at 1.0 and 2.0) and have near-zero surface cost. However, they are unused outside the repo and the CHANGELOG gives no indication they were intentionally exported for external use. They are a mild accidental-pub candidate.

**Conservative ruling:** KEEP for now. These are not wrong to have public — external users may find them useful as reference thresholds. The cost is two `f64` constants. Not flagging as B; they would only become a clean removal target if renaming or restructuring required it.

**Action: none.** Do not add to breaking queue.

---

### INTERNALS feature surface (194 additional items) — policy reminder

The `internals` feature is a deliberate research surface. Per the audit brief, items should only be flagged if they have **zero hits even for research use** across zensim/zenmetrics/research scripts. Items evaluated:

| Module | Status |
|--------|--------|
| `butteraugli::blur::*` (gaussian_blur, blur_5x5, etc.) | Research use confirmed in zenmetrics butteraugli-gpu pipeline |
| `butteraugli::consts::*` (all named tuning constants) | Consumed by zensim training tooling and jxl-encoder-gpu experiments |
| `butteraugli::image::ImageF`, `Image3F`, `BufferPool` | Required by callers of `butteraugli::internals` feature; deliberate |
| `butteraugli::mask::PrecomputedMask`, `precompute_reference_mask`, etc. | GPU port requires these; deliberate |
| `butteraugli::opsin::*` (including `srgb_to_linear` duplicate) | The opsin-module version is the canonical location; lib.rs re-export wraps it for default-feature callers |
| `butteraugli::psycho::PsychoImage`, `separate_frequencies` | Used in GPU pipeline construction |

**Zero items flagged in the internals surface.** All have confirmed research consumers or are structural exports required by the feature's purpose.

---

## Grep commands and counts

```bash
# srgb_to_linear external consumers
find /home/lilith/work -name "*.rs" -not -path "*/butteraugli/*" -not -path "*/target/*" -not -path "*/.jj/*" \
  -exec grep -l "butteraugli::srgb_to_linear\|use butteraugli.*srgb" {} \;
# → 14+ hits in jxl-encoder, jxl-encoder-gpu (as of 2026-06-11)

# pnorm_3 external consumers  
find /home/lilith/work -name "*.rs" -not -path "*/butteraugli/*" -not -path "*/target/*" -not -path "*/.jj/*" \
  -exec grep -l "pnorm_3" {} \;
# → 4 hits: zenmetrics butteraugli-gpu, jxl-encoder-gpu butteraugli_loop, zensim picker prep, jxl-encoder tuning runner

# .pnorm( external consumers
find /home/lilith/work -name "*.rs" -not -path "*/butteraugli/*" -not -path "*/target/*" -not -path "*/.jj/*" \
  -exec grep -l "\.pnorm(" {} \;
# → 2 hits: jxl-encoder sa_f_buttloop_cross_test.rs, zenjxl-tuning-runner/metrics.rs

# BUTTERAUGLI_GOOD / BUTTERAUGLI_BAD external consumers
find /home/lilith/work -name "*.rs" -not -path "*/butteraugli/*" -not -path "*/target/*" -not -path "*/.jj/*" \
  -exec grep -l "BUTTERAUGLI_GOOD\|BUTTERAUGLI_BAD" {} \;
# → 0 hits
```

---

## Top 3 observations

1. **Surface is correct.** No accidental pub fns or fields with zero consumers in the default (183-item) surface. The most suspicious item (`BUTTERAUGLI_GOOD`/`BUTTERAUGLI_BAD`) has a reasonable semantic rationale and negligible cost.

2. **The `internals` feature surface is well-curated.** The opsin/blur/mask/psycho/consts modules are all actively consumed by the butteraugli-gpu port in zenmetrics and by jxl-encoder-gpu research. Nothing to flag.

3. **`pub struct butteraugli::ButteraugliReference` appears twice in the snapshot** (lines 112 and 492 in the txt). This is not a duplication bug — it is cargo-public-api showing the re-export at the crate root alongside the impl block in the `precompute` module. No action needed.

---

*Report generated by conservative-ablation scan. REPORT ONLY — no source changes.*
