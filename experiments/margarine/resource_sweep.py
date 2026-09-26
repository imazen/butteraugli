#!/usr/bin/env python3
"""Serial local zenbench and fresh-process RSS measurements on a crop manifest.

Run through run-heavy on Linux, or nice on macOS; set RAYON_NUM_THREADS.
Persists full subprocess logs, raw zenbench JSON and hashes before summarizing.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess

ARMS = ("teacher", "features228", "features228-strips")
BENCH_NAMES = dict(zip(ARMS, ("teacher_rgb8", "features228_rgb8_only", "features228_rgb8_strips_only")))


def rss_bytes(text, system):
    if system == "Darwin":
        values = re.findall(r"^\s*(\d+)\s+maximum resident set size\s*$", text, re.M)
        multiplier = 1
    elif system == "Linux":
        values = re.findall(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
        multiplier = 1024
    else:
        raise ValueError(f"unsupported process-memory reporter: {system}")
    if len(values) != 1 or int(values[0]) <= 0:
        raise ValueError("missing, duplicated or zero process peak RSS")
    return int(values[0]) * multiplier


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("crops", type=Path)
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--build-commit", required=True)
    args = parser.parse_args()
    system = platform.system()
    time_flag = {"Darwin": "-l", "Linux": "-v"}[system]
    if int(os.environ.get("RAYON_NUM_THREADS", "0")) <= 0:
        raise ValueError("set a positive RAYON_NUM_THREADS explicitly")
    args.binary = args.binary.resolve()
    rows = list(csv.DictReader(args.crops.open(), delimiter="\t"))
    if not rows: raise ValueError("empty crop manifest")
    args.output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, ZENBENCH_NO_SAVE="1", LC_ALL="C")
    records = []
    provenance = dict(build_commit=args.build_commit, binary=str(args.binary), binary_sha256=sha(args.binary),
        host=platform.node(), system=system, threads=int(environment["RAYON_NUM_THREADS"]),
        crop_manifest=str(args.crops.resolve()), crop_manifest_sha256=sha(args.crops), inputs={},
        timing="zenbench cold pairs, decode excluded, each metric sRGB conversion included",
        memory="fresh process platform time, decode and inputs included",
        limitation="same-image crops, feature extraction only; no trained score or coverage claim")
    with (args.output / "progress.log").open("x", buffering=1) as progress:
        def report(message):
            print(message, file=progress, flush=True)
            print(message, flush=True)
        for row in rows:
            w, h = int(row["width"]), int(row["height"])
            name = f"{w}x{h}"
            pair = [row["reference"], row["distorted"]]
            for path in pair: provenance["inputs"][path] = sha(Path(path))
            peaks = {}
            for arm in ARMS:
                report(f"{name}: measuring process peak {arm}")
                log = args.output / f"{name}-{arm}-memory.log"
                command = ["/usr/bin/time", time_flag, str(args.binary), "--memory-rgb8", arm, *pair]
                with log.open("x") as out: subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, env=environment, check=True)
                peaks[arm] = rss_bytes(log.read_text(), system)
            report(f"{name}: running interleaved timing")
            result_path = args.output / f"{name}.json"
            with (args.output / f"{name}-bench.log").open("x") as out:
                subprocess.run([str(args.binary), "--bench-rgb8", *pair, str(result_path)],
                    stdout=out, stderr=subprocess.STDOUT, env=environment, check=True)
            result = json.loads(result_path.read_text())
            group = result["comparisons"][0]
            bench = {b["name"]: b for b in group["benchmarks"]}
            teacher_ns = bench[BENCH_NAMES["teacher"]]["summary"]["mean"]
            for arm in ARMS:
                measured = bench[BENCH_NAMES[arm]]
                ns = measured["summary"]["mean"]
                records.append(dict(width=w, height=h, pixels=w*h, arm=arm, mean_ns=ns,
                    ns_per_pixel=ns/(w*h), rounds=measured["summary"]["n"],
                    peak_rss_bytes=peaks[arm], rss_fraction_of_teacher=peaks[arm]/peaks["teacher"],
                    mean_speedup=teacher_ns/ns, timing_unreliable=result["unreliable"]))
            report(f"{name}: saved timing and process peaks")
            # Persist each completed size so a later failed arm loses no results.
            with (args.output / "summary.tsv").open("w") as out:
                writer = csv.DictWriter(out, delimiter="\t", fieldnames=list(records[0]))
                writer.writeheader(); writer.writerows(records)
            (args.output / "_MANIFEST.json").write_text(json.dumps(provenance, indent=2)+"\n")
        fits = []
        for arm in ARMS:
            data = [r for r in records if r["arm"] == arm]
            xs, ys = [r["pixels"] for r in data], [r["mean_ns"] for r in data]
            xm, ym = sum(xs)/len(xs), sum(ys)/len(ys)
            denom = sum((x-xm)**2 for x in xs)
            if denom == 0: raise ValueError("need distinct sizes for resource fit")
            beta = sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/denom
            alpha = ym-beta*xm
            fits.append(dict(arm=arm, alpha_ns=alpha, beta_ns_per_pixel=beta,
                observed_sizes=len(data), residual_ns=[y-alpha-beta*x for x,y in zip(xs,ys)],
                note="OLS description of measured sizes, not extrapolation or a performance gate"))
        (args.output / "time_fits.json").write_text(json.dumps(fits, indent=2)+"\n")
        report("Complete: no acceptance verdict for an untrained extractor")


if __name__ == "__main__": main()
