# Recycle XYB after LF/MF extraction (2026-09-24)

`separate_frequencies` only reads its XYB input during LF/MF extraction.
The internal owned-input path now returns those three planes to the buffer pool
immediately after that stage. HF/UHF extraction can reuse their storage.
The borrowed function remains available to tests and the unstable `internals`
feature.

Single-thread Massif runs of `callgrind_single` (reference creation plus one
planar comparison) measured these peak heaps. Both builds contained the
direct-to-padded Malta change; only the XYB lifetime differed.

| Input | Previous | Early recycle | Reduction |
| --- | ---: | ---: | ---: |
| 512² | 43,055,528 B | 39,909,776 B | 3,145,752 B (7.3%) |
| 1024² | 172,232,296 B | 159,637,264 B | 12,595,032 B (7.3%) |

An additional 1024² Massif check with `RAYON_NUM_THREADS=8` measured
172,278,840 B before and 159,687,848 B after, a 12,590,992 B (7.3%)
reduction. Massif reports instrumented heap allocations, not operating-system
RSS; each peak figure is from one run.

The reduction is approximately three full-size `f32` planes. In a separate
three-run `/usr/bin/time` check at 1024², maximum resident set size was
174,480–175,200 KiB before and 159,984–160,640 KiB after. In a separate
single-thread `hyperfine` run at 1024² (25 runs per build, four warmups),
reference creation plus comparison went from 217.0 ± 2.2 ms to
211.7 ± 2.3 ms. This is a process-level result and includes setup and
allocation. A separate 512² process benchmark with one precomputed reference
and 20 comparisons improved from 970.5 ± 9.0 ms to 922.5 ± 4.0 ms
(12 runs per build). Its measured compare phase also fell from 250.3 ms to
240.1 ms in one direct run. A 512² zenbench run interleaved with libjxl gave
20.16 ms before and 21.01 ms after for its one-thread Rust arm, so there is
cross-harness variation. A follow-up 1024² check alternated the two saved
release binaries in 25 before/after pairs, reversing order each pair. With
`RAYON_NUM_THREADS=1`, the mean paired advantage was 5.30 ms, with a
95% bootstrap interval of 3.64–6.84 ms. With eight Rayon threads, the mean
advantage was 0.93 ms, interval −0.14–2.04 ms; this does not establish a
multithreaded speed gain. The bootstrap intervals characterize variation in
this one session, not performance across machines or image content.

The synthetic callgrind example printed the same score to four decimals; that
alone does not prove exact score parity. The owned and borrowed separation
paths now compare every frequency-plane pixel exactly at 33×37 and 769×769,
the latter exercising the parallel blur branch. The full reference, strip,
and planar parity suites provide broader content coverage. These benchmarks
use a smooth synthetic gradient, so the timing and heap results should not be
generalized to arbitrary photographs or other workloads without checking
representative images and repeated-reference use.
The `precompute_bench` process also builds 20 distorted inputs and runs a full
comparison loop before timing its precomputed-reference loop, so its total
process time is not an isolated measure of reference reuse. The saved
before/after release binaries were not accompanied by a manifest of their
exact compiler flags and build IDs.
