# libjxl v0.12 parity

The original benchmark discrepancy reproduces on latest main and the pin. All 19 tested score pairs match exactly between Rust revisions. The generated 18-pair ladder has max-norm error at most `6.67572021484375e-6`; the original pair reaches `5.0067901611328125e-5`. Google-old matches its pinned C++ oracle exactly on two pairs.

[Results and proposed fix](findings.md), [complete norm table](parity.tsv), [stage comparisons](stages.tsv), [controlled substitutions](ablations.tsv), [Google-old](google-old.tsv), [reduction checks](reductions.tsv), [input/compiler provenance](provenance.json), [resource receipts](resources.txt).

All builds and measurements ran on r5900xt at 80 nits with default FIR and one Rayon thread. JPEG distortions use Pillow 12.3.0 at q20/q50/q80, 4:2:0, without copied colour metadata. Three CID22 references cover 512x512; one is cropped/tiled to 257x193, 1001x667 and 2048x2048. These larger probes test boundaries and scales, not native high-resolution detail. Artifacts and complete logs remain at r5900xt `/mnt/v/output/butteraugli/libjxl-parity-2026-10-17/data`; no images or diffmaps are committed.

`just libjxl-parity-build` builds the Rust probe; `libjxl-parity-generate`, `libjxl-parity-measure`, `libjxl-parity-trace`, `libjxl-parity-ablations` and `libjxl-parity-google-build` expose the scripts. Run builds and measurements under a resource supervisor. The tracing/diagnostic scripts operate on separate archived sources in `~/tmp/bu-libjxl-instrumented` and the disposable libjxl checkout at `~/tmp/libjxl-v012`; they do not edit the production Rust checkout. Apply tracing once to clean sources. The standalone Google runners download pinned, licensed sources; Google-old exposes only max-norm here.
