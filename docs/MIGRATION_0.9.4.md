# butteraugli 0.9.4 — consumer inventory and optional migration

Status: **0.9.4 is additive. Nothing here is required.** Every existing call
site compiles and returns the same numbers on 0.9.4 as on 0.9.3. This document
exists so the two repositories that path-patch butteraugli can decide, with the
exact diff in front of them, whether to adopt the new `linear-planes` surface.

Nothing in this file was applied to `jxl-encoder` or `zenmetrics` — those repos
are outside this crate's boundary. The diffs below are proposals.

Audited at butteraugli `main` (0.9.4 line), 2026-08-28.

---

## 1. What consumers actually import today

Read-only grep of `~/work/zen/jxl-encoder`, `~/work/zen/zenmetrics` and
`~/work/zen/zenpipe` for `use butteraugli` / `butteraugli::`.

### 1.1 The headline finding: nobody enables `internals`

**No consumer of this crate enables the `internals` cargo feature.** The only
`butteraugli = { ..., features = ["internals"] }` in any of the three repos is
in this repository's own `butteraugli-bench/Cargo.toml`, which uses it to reach
`blur::`, `malta::` and `image::` for the per-kernel tier benchmarks.

The `internals` mentions in `zenmetrics` are a *different feature with the same
name*: `butteraugli-gpu` (`zenmetrics/crates/butteraugli-gpu`) declares its own
`internals` feature that exposes `compute_with_reference_from_linear_planes`,
and `zenmetrics-api` / `zenmetrics-cli` / `jxl-encoder` enable
**`butteraugli-gpu/internals`**, never `butteraugli/internals`. The
linear-planes API those crates care about
(`set_reference_from_linear_planes` / `compute_from_linear_planes` / the
multi-resolution strip walker of zenmetrics#47) is `butteraugli-gpu`'s GPU
surface; on the CPU side the equivalent has always been public in the default
surface (`ButteraugliReference::new_linear_planar` +
`compare_linear_planar`).

Consequence: the CPU crate's seven `internals` modules (`blur`, `consts`,
`image`, `malta`, `mask`, `opsin`, `psycho` — 125 public items) have exactly
one consumer, and it is in-repo. Nothing outside pins them.

### 1.2 Dependency declarations

| repo | manifest | declaration | features |
|---|---|---|---|
| jxl-encoder | `jxl-encoder/Cargo.toml` | `butteraugli = { version = "0.9", optional = true }` | default |
| jxl-encoder | `Cargo.toml` (workspace `[patch]`) | `butteraugli = { path = "../../butteraugli/butteraugli" }` | — |
| jxl-encoder | `zenjxl-tuning-runner/Cargo.toml` | `butteraugli = { version = "0.9.2", optional = true }` | default |
| zenmetrics | `Cargo.toml` (workspace dep) | `butteraugli = { version = "0.9.4" }` | default |
| zenmetrics | `Cargo.toml` (`[patch.crates-io]`) | `butteraugli = { path = "../../butteraugli/butteraugli" }` | — |
| zenmetrics | `crates/zenmetrics-api/Cargo.toml` | `butteraugli = { workspace = true, optional = true }` | default |
| zenmetrics | `crates/zenmetrics-orchestrator/Cargo.toml` | `butteraugli = { workspace = true, optional = true }` | default |
| zenmetrics | `crates/zenmetrics-cli/Cargo.toml` | `butteraugli = { version = "0.9.2", optional = true }` | default |
| zenmetrics | `crates/butteraugli-gpu/Cargo.toml` | `butteraugli = { workspace = true }` (**dev**-dependency) | default |
| zenmetrics | `benchmarks/heaptrack/drivers/cpu_profile/Cargo.toml` | `butteraugli = { path = "../../../../../../butteraugli/butteraugli" }` | default |
| zenpipe | — | **no dependency on butteraugli** | — |

`zenmetrics`'s workspace dep already pins `0.9.4`, which does not exist on
crates.io — that pin resolves only through the `[patch.crates-io]` path entry.
Publishing 0.9.4 is what makes that manifest resolvable without the patch.

### 1.3 Items imported

Every item below is in the **default** (non-feature) surface of 0.9.3 and
0.9.4. The list is exhaustive across both repos.

| item | used by |
|---|---|
| `butteraugli()` | jxl-encoder (`src/api_tests.rs`, `tests/it/clic2025.rs`, `zenjxl-tuning-runner/src/metrics.rs`), zenmetrics (`zenmetrics-api/src/cpu_dispatch.rs`, `zenmetrics-orchestrator/src/cpu_adapter.rs`, `zenmetrics-cli/src/jobexec.rs`, `butteraugli-gpu` examples/tests, heaptrack drivers) |
| `butteraugli_linear()` | jxl-encoder (≈100 `examples/*.rs`, `src/vardct/tile_distmap.rs`, several `tests/it/*.rs`), zenmetrics (`cpu_dispatch.rs`, `jobexec.rs`) |
| `butteraugli_strip()` | zenmetrics (`cpu_adapter.rs`, heaptrack drivers, `butteraugli-gpu/examples/strip_drift_corpus.rs`) |
| `srgb_to_linear()` | jxl-encoder (≈60 examples) |
| `ButteraugliParams` + `new` / `with_intensity_target` / `with_hf_asymmetry` / `with_compute_diffmap` / `default` | both repos, everywhere |
| `ButteraugliResult::{score, pnorm_3, diffmap}` | both repos |
| `ButteraugliReference::new` | zenmetrics (`cpu_dispatch.rs:678`, `cpu_adapter.rs:482`) |
| `ButteraugliReference::new_linear_planar` | jxl-encoder (`src/vardct/perceptual_backend.rs:604`, `examples/w44_111_metric_divergence.rs`) |
| `ButteraugliReference::compare_linear_planar` | jxl-encoder (`perceptual_backend.rs`, `JXL_W44_B7_DISABLE` A/B path) |
| `ButteraugliReference::compare_linear_planar_into` | jxl-encoder (`perceptual_backend.rs`, production buttloop path) |
| `ButteraugliReference::estimated_reference_bytes` | jxl-encoder (`src/vardct/perceptual_loop.rs`, `tests/it/alloc_budget.rs`) |
| `RGB8` re-export | jxl-encoder (`zenjxl-tuning-runner/src/metrics.rs`) |

No consumer touches `blur::`, `consts::`, `image::`, `malta::`, `mask::`,
`opsin::` or `psycho::`.

---

## 2. What 0.9.4 adds

Two things, both additive:

1. **`ButteraugliReference::compare_linear_planar_with_stop`** — in the default
   surface. Completes the `*_with_stop` family; the planar path was the only
   compare method without a cancellable variant.
2. **The `linear-planes` cargo feature** — off by default. Enables
   `butteraugli::linear_planes`, a six-type documented API over the same code
   paths: `LinearPlanes`, `LinearColorSpace`, `Resolution`, `Walk` /
   `StripMode`, `ScorerBuilder` / `Scorer`, `Scores`.

Scores from `linear_planes` are **bit-identical** to the equivalent default-API
calls; `butteraugli/tests/linear_planes_parity.rs` asserts that with
`assert_eq!` on `f64` (no tolerance), including pixel-for-pixel diffmap
equality, across default params, padded stride, single-scale, HDR intensity
target, and both strip variants.

---

## 3. Proposed migration — `jxl-encoder`

The only call site where the new API buys anything is
`jxl-encoder/jxl-encoder/src/vardct/perceptual_backend.rs`
(`CpuButteraugliBackend`). What it gains: input validation moves to a typed
constructor that names the colour space and checks stride/finiteness once, and
the `padded_width` stride stops being an untyped `usize` threaded through the
trait.

**Required first:** enable the feature.

```diff
--- a/jxl-encoder/Cargo.toml
+++ b/jxl-encoder/Cargo.toml
-butteraugli = { version = "0.9", optional = true }
+butteraugli = { version = "0.9.4", optional = true, features = ["linear-planes"] }
```

### 3.1 `set_reference`

```diff
--- a/jxl-encoder/src/vardct/perceptual_backend.rs
+++ b/jxl-encoder/src/vardct/perceptual_backend.rs
     fn set_reference(
         &mut self,
         ref_r: &[f32],
         ref_g: &[f32],
         ref_b: &[f32],
         width: usize,
         height: usize,
     ) -> Result<()> {
-        let r = butteraugli::ButteraugliReference::new_linear_planar(
-            ref_r,
-            ref_g,
-            ref_b,
-            width,
-            height,
-            width, // tight stride
-            self.params.clone(),
-        )
-        .map_err(|e| crate::error::Error::InvalidInput(format!("butteraugli reference: {e}")))?;
-        self.reference = Some(r);
+        use butteraugli::linear_planes::{LinearPlanes, Scorer};
+        // Tight stride by the trait contract (`set_reference` takes no stride).
+        let planes = LinearPlanes::new(ref_r, ref_g, ref_b, width, height)
+            .map_err(|e| crate::error::Error::InvalidInput(format!("butteraugli reference: {e}")))?;
+        let scorer = Scorer::builder()
+            .with_intensity_target(self.params.intensity_target())
+            .with_hf_asymmetry(self.params.hf_asymmetry())
+            .with_xmul(self.params.xmul())
+            .with_diffmap(true) // the buttloop always needs the per-block diffmap
+            .build(&planes)
+            .map_err(|e| crate::error::Error::InvalidInput(format!("butteraugli reference: {e}")))?;
+        self.reference = Some(scorer);
         Ok(())
     }
```

and the field type:

```diff
-    reference: Option<butteraugli::ButteraugliReference>,
+    reference: Option<butteraugli::linear_planes::Scorer>,
```

### 3.2 `compare_with_reference` — production path

```diff
-        let (score, _pnorm_3) = bref
-            .compare_linear_planar_into(dist_r, dist_g, dist_b, padded_width, diffmap_out)
-            .map_err(|e| crate::error::Error::InvalidInput(format!("butteraugli compare: {e}")))?;
+        let dist = butteraugli::linear_planes::LinearPlanes::with_stride(
+            dist_r, dist_g, dist_b, width, height, padded_width,
+        )
+        .map_err(|e| crate::error::Error::InvalidInput(format!("butteraugli compare: {e}")))?;
+        let scores = bref
+            .score_into(&dist, diffmap_out)
+            .map_err(|e| crate::error::Error::InvalidInput(format!("butteraugli compare: {e}")))?;
+        let score = scores.max_norm;
         debug_assert_eq!(diffmap_out.len(), width * height);
```

### 3.3 `compare_with_reference` — the `JXL_W44_B7_DISABLE` A/B path

```diff
-            let r = bref
-                .compare_linear_planar(dist_r, dist_g, dist_b, padded_width)
-                .map_err(|e| { ... })?;
-            let dm = r.diffmap.ok_or_else(|| { ... })?;
-            let buf = dm.into_buf();
+            let scores = bref
+                .score(&dist)
+                .map_err(|e| { ... })?;
+            let dm = scores.diffmap.ok_or_else(|| { ... })?;
+            let buf = dm.into_buf();
             debug_assert_eq!(buf.len(), width * height);
             *diffmap_out = buf;
-            return Ok(BackendCompareResult { score: r.score });
+            return Ok(BackendCompareResult { score: scores.max_norm });
```

Note one behaviour difference to be aware of in this path: the default API's
`ButteraugliReference::compare_*` returns `Some(diffmap)` **regardless** of
`compute_diffmap` (see §5). `Scorer::score` honours the documented contract and
returns `None` unless `with_diffmap(true)` was set — which the `set_reference`
diff above does. If jxl-encoder ever built the backend without
`with_compute_diffmap(true)` and still relied on the diffmap coming back, that
would surface here as a caught `InvalidInput` rather than silently working.

### 3.4 What must NOT change

- `ButteraugliReference::estimated_reference_bytes(w, h, &params)` in
  `perceptual_loop.rs` / `tests/it/alloc_budget.rs`. `Scorer` has
  `reference_bytes(&self)` (post-construction) but there is no pre-construction
  estimate on the new type; keep using `estimated_reference_bytes`. Both types
  size the same precompute, so the budget check stays valid.
- The ~100 `examples/*.rs` and `tests/it/*.rs` sites using
  `butteraugli_linear` / `butteraugli` / `srgb_to_linear`. They take
  interleaved input; `linear_planes` is for planar callers and offers them
  nothing.
- `GpuButteraugliBackend`. It talks to `butteraugli-gpu`, not this crate.

**Expected result:** byte-identical encodes. The parity suite proves the score
path is unchanged; the diff above changes only how the arguments are packaged.
Validate with `tests/it/buttloop_target_parity.rs` and
`tests/it/zenjxl_regression_gate.rs` before landing.

---

## 4. Proposed migration — `zenmetrics`

**Recommendation: none.** Every zenmetrics call site takes interleaved
`ImgRef<RGB8>` or `&[u8]` sRGB — `cpu_dispatch.rs`, `cpu_adapter.rs`,
`jobexec.rs` all feed `butteraugli()`, `butteraugli_linear()`,
`butteraugli_strip()` or `ButteraugliReference::new()`. None of them holds
planar linear-light f32, so `linear-planes` would mean interleave→planar
conversion for no gain.

Two changes are worth making, both trivial:

```diff
--- a/crates/zenmetrics-cli/Cargo.toml
+++ b/crates/zenmetrics-cli/Cargo.toml
-butteraugli = { version = "0.9.2", optional = true }
+butteraugli = { workspace = true, optional = true }
```

`zenmetrics-cli` pins `0.9.2` directly instead of using the workspace dep at
`0.9.4`; cargo unifies them today only because the `[patch.crates-io]` path
entry overrides both. Once 0.9.4 is on crates.io, the workspace dep is the
honest declaration.

```diff
--- a/Cargo.toml
+++ b/Cargo.toml
 # butteraugli: the local sibling ... 
-butteraugli  = { path = "../../butteraugli/butteraugli" }
+# (drop the [patch.crates-io] entry once 0.9.4 is published)
```

`butteraugli-gpu`'s CPU-parity examples and tests (`parity_vs_cpu.rs`,
`parity_real_image.rs`, `diffmap_inspect.rs`, `strip_drift_corpus.rs`) compare
GPU output against `butteraugli::butteraugli()` on sRGB input. If any of those
is ever reworked to compare against the CPU on *linear planes* — which is what
`butteraugli_gpu::set_reference_from_linear_planes` actually consumes —
`linear_planes::Scorer` is the surface to use, and it removes the sRGB-u8
round-trip that currently accounts for the documented 0.02% GPU/CPU drift
(`perceptual_backend.rs` module docs, W44-RECON-DEEP/A7). That is a real
follow-up, not a migration.

---

## 5. Discrepancy found during the audit (not fixed here)

`ButteraugliResult::diffmap` is documented as "only present if
`compute_diffmap` was true". That holds for the free functions
(`butteraugli`, `butteraugli_linear`, `butteraugli_strip`), but **not** for the
warm-reference methods: `ButteraugliReference::compare*` returns
`Some(diffmap)` unconditionally (`precompute.rs:1062`, `precompute.rs:1263`).
Callers that pass `compute_diffmap = false` to a `ButteraugliReference` are
paying for a full `width * height * 4 B` diffmap allocation they did not ask
for and are being handed it anyway.

This is a documentation-vs-behaviour mismatch in the default API. It is left
untouched in 0.9.4 — changing the behaviour would be a silent break for any
caller relying on the diffmap being present, and correcting the doc is a
maintainer call. `linear_planes::Scorer` honours the documented contract on its
own surface (it drops the diffmap when `with_diffmap(false)`), so the new API
is self-consistent either way. Flagged for the maintainer to decide.

---

## 6. `internals` — status

`internals` is unchanged and still exports the same seven modules. It is now
documented as unstable with no compatibility promise, and `CHANGELOG.md`
carries it under `QUEUED BREAKING CHANGES` for a future major.

If you are reaching for `internals` to score planar linear-light data, use
`linear-planes` instead — that is the entire reason it exists.
