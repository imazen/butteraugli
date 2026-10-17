# libjxl v0.12 parity investigation

2026-10-17, r5900xt. Rust latest main `0e397e5e716cd70ca40605ad93d68df590869dda`, pin `791d08d725f27b5331ef7b96e73aad89ed895f45`; C++ libjxl v0.12.0 `a7a9c787341cf703dede03c2009fa460cae5e5df`. All comparisons use 80 nits, default FIR and sRGB RGB8 input. Both images are decoded to RGB8 once before scoring and stored without PNG colour metadata.

`generate.py` creates deterministic Pillow JPEG q20/q50/q80 distortions of three CID22 references. The first source is cropped/tiled for odd sizes and 2048x2048; these are boundary and multiresolution probes, not native high-resolution photographs. `measure.py` retains both diffmaps, calls the Rust example and `butteraugli_main`, and independently aggregates the exported C++ map in double precision. The C++ CLI 3-norm field initially has six decimals; subsequent pin runs use a printf-only patch to 17 significant digits.

The 18-pair latest ladder has max absolute max-norm discrepancy 6.67572021484375e-6. The additional original benchmark pair reproduces 5.0067901611328125e-5 on the pin. Investigation continues in subsequent commits.

Artifacts are retained on r5900xt under `/mnt/v/output/butteraugli/libjxl-parity-2026-10-17/data` (also `~/tmp/bu-libjxl-data`). No images or maps are committed. Generator uses Pillow 12.3.0; input/JPEG hashes are in the artifact manifest. The exact original benchmark RGB8 buffers came from i265's `~/tmp/iqa-compete/inputs/cid22-{reference,distorted}.rgb`.
