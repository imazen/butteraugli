# Diffmap parity against libjxl `butteraugli_main` — 2026-10-24

Host i270 (Core Ultra 7 270K Plus, AVX2, no AVX-512; both sides run their AVX2
paths). libjxl `a7a9c787341cf703dede03c2009fa460cae5e5df`, built with system
lcms2 2.17 (`JPEGXL_ENABLE_SKCMS=OFF`), system Highway 1.3.0, Release:

```
cmake -S libjxl -B build-lcms -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  -DBUILD_SHARED_LIBS=OFF -DJPEGXL_ENABLE_DEVTOOLS=ON -DJPEGXL_ENABLE_SKCMS=OFF \
  -DJPEGXL_FORCE_SYSTEM_LCMS2=ON -DJPEGXL_FORCE_SYSTEM_HWY=ON -DJPEGXL_FORCE_SYSTEM_BROTLI=ON \
  -DJPEGXL_ENABLE_SJPEG=OFF -DJPEGXL_ENABLE_OPENEXR=OFF -DJPEGXL_ENABLE_JNI=OFF \
  -DJPEGXL_ENABLE_DOXYGEN=OFF -DJPEGXL_ENABLE_MANPAGES=OFF -DJPEGXL_ENABLE_BENCHMARK=OFF \
  -DJPEGXL_ENABLE_EXAMPLES=OFF -DJPEGXL_ENABLE_TCMALLOC=OFF -DJPEGXL_BUNDLE_LIBPNG=OFF
ninja -C build-lcms butteraugli_main
just libjxl-parity build-lcms/tools/butteraugli_main ~/tmp/ba-parity/run-main
awk -f benchmarks/libjxl_parity_2026-10-24/summarize.awk ~/tmp/ba-parity/run-main/pairs.tsv
```

Harness: `butteraugli/examples/libjxl_parity.rs`. Inputs: CID22-512 validation
(first 12, cropped centred to 7x9, 16x16, 65x49, 256x256, plus native 512x512),
all 25 gb82 references (576x576), first 4 CLIC 2025 final-test images (1024x1024
crop plus native 1360x2048 / 2048x1360). Distortions: mozjpeg q20/q50/q85,
posterize (`v & !7`), deterministic ±6 noise. 465 pairs per variant. Each pair
runs three ways on both sides: 8-bit PNG, linear PFM from the lcms2 u8 LUT
(copied from fast-ssim2 `017d242`), and linear PFM from libjxl's `TF_SRGB`
polynomial. Every comparison is of the full-resolution diffmap, its max-norm and
its 3-norm (computed with the crate's reduction on both maps).

Variants:
- **main** — `02b570b4` plus the harness. `Numerics::Fast` (default) and
  `Numerics::LibjxlExact` (#21).
- **opsin-fma** — the same, with only Fast's per-pixel opsin step
  (`opsin.rs`: absorbance, `gamma`, `fast_log2f`) rewritten in libjxl's nested
  `MulAdd` order. Not on main.

## sRGB ingress

- The lcms2 LUT and libjxl's `TF_SRGB` polynomial agree on all 256 entries,
  bit for bit. Fast's `powf` LUT differs from both in 202/256 entries
  (max 49 ULP, 1.79e-7 absolute). LibjxlExact already uses the polynomial.
- libjxl's PNG diffmaps equal its PFM diffmaps (either LUT) bit for bit on all
  465 pairs. At this commit `jxl_cms` converts sRGB to linear sRGB with
  `TF_SRGB` and skips lcms2/skcms, so the CMS backend does not matter here.

## Results, excluding 7x9 (405 pairs per row)

| variant | numerics | arm | maps bit-exact | max abs score diff | max rel score diff | max abs 3-norm diff | max abs map diff | map pixels differing |
|---|---|---|---:|---:|---:|---:|---:|---:|
| main | Fast | png | 0 | 2.689e-4 | 4.53e-5 | 1.718e-4 | 7.84e-4 | 98.94% |
| main | Fast | pfm | 0 | 2.689e-4 | 5.93e-5 | 1.717e-4 | 7.65e-4 | 98.84% |
| main | LibjxlExact | png / pfm | 0 | 9.537e-7 | 5.74e-7 | 6.537e-7 | 8.42e-5 | 30.63% |
| opsin-fma | Fast | png | 0 | 3.309e-4 | 7.24e-5 | 1.837e-4 | 6.72e-4 | 98.90% |
| opsin-fma | Fast | pfm | 0 | 3.309e-4 | 7.28e-5 | 1.837e-4 | 6.72e-4 | 98.70% |
| opsin-fma | LibjxlExact | png / pfm | 0 | 9.537e-7 | 5.74e-7 | 6.537e-7 | 8.42e-5 | 30.63% |

Max-norm bit-equal to libjxl: LibjxlExact 257/405, Fast 9/405 (png) and 16/405
(pfm). No 3-norm is bit-equal. Median |score diff| (png): Fast 3.3e-6 (main),
3.6e-6 (opsin-fma); LibjxlExact 0.

Across the 1215 Fast comparisons, opsin-fma moves the score closer to libjxl in
571, further in 562, and leaves 82 unchanged; mean |score diff| 1.212e-5 →
1.185e-5, mean |3-norm diff| 4.44e-6 → 3.85e-6. LibjxlExact is identical in both
variants (it has its own opsin).

Per size, Fast, png arm (main → opsin-fma):

| size | pairs | max abs score diff | max abs 3-norm diff |
|---|---:|---|---|
| 16x16 | 60 | 1.45e-4 → 4.67e-5 | 5.46e-5 → 4.61e-5 |
| 65x49 | 60 | 1.35e-4 → 6.96e-5 | 4.99e-5 → 3.35e-5 |
| 256x256 | 60 | 1.74e-4 → 1.51e-4 | 3.94e-5 → 3.81e-5 |
| 512x512 | 60 | 2.69e-4 → 3.31e-4 | 1.45e-4 → 7.09e-5 |
| 576x576 | 125 | 2.66e-4 → 2.54e-4 | 1.72e-4 → 1.84e-4 |
| 1024x1024 | 20 | 3.34e-5 → 4.01e-5 | 2.80e-6 → 1.16e-5 |
| 1360x2048 | 15 | 6.58e-5 → 7.06e-5 | 1.55e-5 → 2.13e-5 |
| 2048x1360 | 5 | 1.36e-5 → 1.45e-5 | 8.19e-7 → 9.46e-7 |

LibjxlExact's max abs score diff is ≤ 9.537e-7 at every size.

## 7x9 (60 pairs per row)

`butteraugli_main` goes through `ButteraugliComparator`, which zero-fills the
diffmap when either side is under 8 px (`butteraugli.cc` 1832, 1897). The
library's `ButteraugliDiffmap` instead pads to 8x8
(`ButteraugliDiffmapSmall`, 2024). Both crate modes follow the padding path, so
they report up to 5.52 where `butteraugli_main` reports 0. This harness does
not test the padding path against libjxl.

## Conclusions

- The per-pixel opsin FMA order alone does not bring Fast to libjxl: maps stay
  ~99% different and the worst score error stays ~3e-4. Fast's residual comes
  from the other stages LibjxlExact also reorders (transfer, Gaussian/FIR,
  reductions).
- LibjxlExact stays within its documented 1e-6 score tolerance on all 405
  real-image pairs (largest 9.537e-7) but is not bit-exact: 30.6% of diffmap
  pixels differ, by up to 8.4e-5.
- The lcms2 LUT is not needed to match current libjxl's butteraugli_main; it is
  the same table as libjxl's own polynomial.

## Raw data

On i270 under `~/tmp/ba-parity/` (not mirrored): `run-main/` and
`run-opsinfma/` each hold `pairs.tsv` (one row per pair × arm × numerics),
`libjxl_ingress.tsv` and `luts.txt`; `run-main/pairs/` holds the 4.5 GB of
PNG/PFM inputs and raw distmaps. sha256 prefixes: run-main/pairs.tsv
`4f7b4ab829deada7`, run-opsinfma/pairs.tsv `d03f165f7a93b9af`,
libjxl_ingress.tsv `4e856417f1cddda2` (both runs).
Receipt (both runs, serial): `rc=0 455s | peak-RSS 1.17GiB | min-avail 20664MiB | peak-load 7.45`.
