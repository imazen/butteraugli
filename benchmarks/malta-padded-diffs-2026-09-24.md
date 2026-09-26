# Malta padded-difference pass (2026-09-24)

The Malta filter previously allocated a full-size image for scaled differences,
then copied it into a zero-padded image for the SIMD stencil. The first pass now
writes into the padded image directly. This removes one full-image temporary and
one copy per Malta call. The arithmetic and four-pixel zero border are unchanged.

This follows the scratch-buffer lifetime idea in
[fmetrics' Butteraugli implementation](https://github.com/halidecx/fmetrics/blob/main/src/butteraugli/butteraugli.c): keep transient storage only for the phase that consumes it. We retained Butteraugli's existing pooled buffers and SIMD dispatch.

Measurements on x86-64, release build, 2026-09-24 (local machine; separate
before/after runs, so wall times are indicative):

| Measure | Before | After |
| --- | ---: | ---: |
| iai-callgrind Malta instructions, 128² | 1,323,464 | 1,299,128 |
| iai-callgrind Malta instructions, 512² | 19,633,879 | 19,514,493 |
| zenbench Malta, V4 arm previously mislabeled V3, 512² | 616.6 µs | 591.4 µs |
| zenbench Malta, V4 arm previously mislabeled V3, 1920×1080 | 9.43 ms | 9.09 ms |
| zenbench full compare, 512², one thread | 20.34 ms | 20.09 ms |
| Massif peak heap, 512², one Rayon thread | 43,055,528 B | 43,055,528 B |

The peak stays at another phase of the pipeline. The benefit is lower transient
allocation and copy traffic inside Malta, not a demonstrated whole-pipeline peak
reduction. The iai-callgrind bench prepares source images outside the measured
region, then measures the Malta allocation and filter. `cargo-show-asm`
confirmed a vectorized v3 scaled-difference loop;
`cargo pgo` training on 256², 512², and 1024² images placed the v4 blur,
frequency decomposition, and Malta scaled-difference loops among the hot code.
Those loops already have explicit `archmage` tier annotations, so the profile
did not justify additional manual inline or cold attributes.

Methodology correction: this machine supports AVX-512, but the original
`kernel_tiers` benchmark toggled only the V3 token. Its enabled arm therefore
dispatched to V4. The before/after Malta wall times above came from that same
setup, but they are **not** an isolated V3 result. The benchmark now disables
V4 after enabling V3 and checks both token states before measurement. With the
corrected harness, the current 512² Malta path measured 744.1 µs at V3 versus
3884.8 µs scalar (30 interleaved rounds); this is a tier comparison, not a
corrected before/after estimate. Callgrind
instruction counts do not establish wall-time or RSS gains, and the 512²
whole-compare difference is small enough to require replication on more inputs.
The PGO training and all timing inputs were synthetic, so the hotspot ranking
and speed estimates may differ for photographic or highly textured images.

For repeatable checks, run
`cargo bench -p butteraugli-bench --bench malta_callgrind` and
`cargo bench -p butteraugli-bench --bench kernel_tiers -- --group=malta_diff_map/512x512`.
