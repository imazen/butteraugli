# fmetrics Butteraugli follow-up (2026-09-24)

Reviewed [fmetrics' Butteraugli C implementation](https://github.com/halidecx/fmetrics/blob/main/src/butteraugli/butteraugli.c) and its [fixed blur kernels](https://github.com/halidecx/fmetrics/blob/main/src/butteraugli/internal.h). Its allocation order and scratch lifetime motivated the earlier XYB recycle change. Its fixed Gaussian weights, delayed HF/UHF allocation, and direct use of LF blur outputs were tested separately here. These follow-up experiments were reverted because they did not establish a process-level win.

| Experiment | 1024² runtime, 25 runs per build | 1024² Massif heap peak, one Rayon thread |
| --- | ---: | ---: |
| Current owned-XYB path | 210.7 ± 2.1 ms | 159,637,264 B |
| Allocate HF/UHF at their stages | 138.2 ± 2.7 ms (multithreaded; baseline 136.9 ± 3.1 ms) | 159,637,264 B |
| Cache the four fixed blur kernels | 210.5 ± 2.7 ms (baseline 211.9 ± 2.6 ms) | Not measured |
| Build LF from blur outputs directly | 211.9 ± 2.3 ms (baseline 210.7 ± 2.1 ms) | 159,637,264 B |

The fixed-kernel cache had exact `f32` bit parity with the generic kernel generator. It was also flat at 128² (4.3 ms for both builds), so its small 1024² difference is not convincing enough to keep the extra dispatch and one-time caches. The allocation experiments printed the same four-decimal score on the synthetic 1024² input; that output cannot establish exact score parity. The retained owned-XYB path has exact frequency-plane parity tests at 33×37 and 769×769 (the latter crosses the parallel threshold). The table's Massif runs all used `RAYON_NUM_THREADS=1`; their peaks were identical because other phases of the comparison set the process peak.

The first allocation experiment's timing was multithreaded, unlike the other two. Each hyperfine invocation ran all samples of one command before the next, so temporal drift could affect small differences. The single-row cache difference of about 1.4 ms at 1024² should not be interpreted as a measured improvement.

The before/after builds were saved as local release binaries, without an
archived record of exact rustc flags, build IDs, and CPU state. This limits
independent reproduction. A stronger follow-up would build both revisions
from recorded source states with identical flags, alternate their runs, and
include varied image content and cold and reused references.

The C implementation's horizontal and vertical blur loops are simple scalar loops amenable to autovectorization. This crate already has explicit SIMD blur and Malta paths. A subsequent [matched-input head-to-head](fmetrics-head-to-head-2026-09-24.md) found this Rust crate faster on four local cases, although scores differed, so it does not isolate the blur loops or prove score parity. The larger remaining opportunity is to reduce the memory needed by the difference/map stages or to improve blur's interior convolution, with score parity and cold/warm RSS measured independently.
