#!/usr/bin/env python3
"""Alternate one-shot Rust/other-backend runs and report metric time and RSS."""

import argparse
import os
import re
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary")
    parser.add_argument("image")
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--quality", type=int, default=75)
    parser.add_argument("--resize", help="optional WxH target")
    parser.add_argument("--rss-runs", type=int, default=3)
    parser.add_argument("--other", choices=("fmetrics", "libjxl"), default="fmetrics")
    args = parser.parse_args()
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = "1"
    env["MALLOC_ARENA_MAX"] = "1"

    def command(backend):
        cmd = [args.binary, backend, args.image, "1", str(args.quality)]
        if args.resize:
            cmd.append(args.resize)
        return cmd

    def timed(backend):
        output = subprocess.check_output(command(backend), env=env, text=True)
        ms = float(re.search(r"ms=([0-9.]+)", output).group(1))
        score = float(re.search(r"score=([0-9.]+)", output).group(1))
        return ms, score

    for _ in range(2):
        timed("rust")
        timed(args.other)

    times = {"rust": [], args.other: []}
    scores = {}
    for i in range(args.pairs):
        order = ("rust", args.other) if i % 2 == 0 else (args.other, "rust")
        for backend in order:
            ms, score = timed(backend)
            times[backend].append(ms)
            scores[backend] = score

    rss = {"rust": [], args.other: []}
    for i in range(args.rss_runs):
        order = ("rust", args.other) if i % 2 == 0 else (args.other, "rust")
        for backend in order:
            result = subprocess.run(
                ["/usr/bin/time", "-f", "RSS_KB=%M", *command(backend)],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            rss[backend].append(int(re.search(r"RSS_KB=(\d+)", result.stderr).group(1)))

    for backend in ("rust", args.other):
        print(
            f"{backend}: metric_ms_median={statistics.median(times[backend]):.3f} "
            f"metric_ms_range={min(times[backend]):.3f}..{max(times[backend]):.3f} "
            f"rss_kib_median={statistics.median(rss[backend]):.0f} "
            f"score_p3={scores[backend]:.17f}"
        )
    print(
        f"{args.other}/rust metric-time ratio: "
        f"{statistics.median(times[args.other]) / statistics.median(times['rust']):.3f}x"
    )


if __name__ == "__main__":
    main()
