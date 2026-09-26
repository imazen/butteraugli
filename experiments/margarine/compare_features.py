#!/usr/bin/env python3
"""Compare frozen edge exporters or direct-score maps; no fitting or thresholds."""
import argparse
import csv
import json
import io
from pathlib import Path
import struct
import subprocess
import time

from score_manifest import NORMS, digest, parse_features, parse_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", type=Path)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--build-commit", required=True)
    parser.add_argument("--diffmaps", action="store_true", help="compare box3 scalar scores and persisted native maps")
    parser.add_argument("--candidate", default="box3", choices=["box3", "multirate", "compact", "compact4", "sparse"])
    parser.add_argument("--after-strip-rows", type=int)
    args = parser.parse_args()
    if args.after_strip_rows is not None and (not args.diffmaps or args.after_strip_rows <= 0):
        parser.error("after-strip-rows requires diffmaps and positive rows")
    args.output.mkdir(parents=True, exist_ok=False)
    with args.pairs.open() as file:
        pairs = list(csv.DictReader(file, delimiter="\t"))
    if not pairs:
        raise ValueError("empty comparison")
    summary = dict(build_commit=args.build_commit, pairs_sha256=digest(args.pairs),
                   binaries={name: digest(getattr(args, name)) for name in ("before", "after")},
                   pairs=len(pairs), changed_pairs=0, changed_values=0, max_absolute=0.0,
                   max_relative=0.0, status="running")
    summary["mode"] = "diffmaps" if args.diffmaps else "features"
    summary["candidate"] = args.candidate
    summary["after_strip_rows"] = args.after_strip_rows
    summary["changed_map_samples"] = 0
    summary["changed_map_pairs"] = 0
    summary["max_map_absolute"] = 0.0
    with (args.output / "progress.log").open("x", buffering=1) as progress, \
            (args.output / "comparisons.jsonl").open("x", buffering=1) as results:
        for i, row in enumerate(pairs):
            values = []
            maps = []
            for name in ("before", "after"):
                output = args.output / f"{i}-{name}.{'f32le' if args.diffmaps else 'tsv'}"
                command = [str(getattr(args, name))] + ([] if args.diffmaps else ["--export-edges"])
                if args.diffmaps and name == "after" and args.after_strip_rows:
                    command += ["--native-strip", str(args.after_strip_rows)]
                run = subprocess.run(command + [row["reference"], row["distorted"], str(output)],
                                     capture_output=True, text=True, check=False)
                (args.output / f"{i}-{name}.log").write_text(run.stdout + run.stderr)
                run.check_returncode()
                if args.diffmaps:
                    extracted = next(csv.DictReader(io.StringIO(run.stdout), delimiter="\t"))
                else:
                    with output.open() as file:
                        extracted = next(csv.DictReader(file, delimiter="\t"))
                dimensions = int(extracted["width"]), int(extracted["height"])
                if args.diffmaps:
                    mode = args.candidate + ("-native-strip" if name == "after" and args.after_strip_rows else "")
                    scores = parse_score(run.stdout, mode, dimensions, output)
                    values.append([scores[n] for n in NORMS])
                    maps.append(output)
                else:
                    values.append(parse_features(output, dimensions))
                if name == "before":
                    before_dimensions = dimensions
                elif dimensions != before_dimensions:
                    raise ValueError("exporters disagree on dimensions")
            differences = [abs(a - b) for a, b in zip(*values)]
            relative = [d / max(abs(a), abs(b), 1e-300)
                        for d, a, b in zip(differences, *values)]
            changed = sum(d != 0 for d in differences)
            result = dict(pair=row["pair"], source=row["source"], dimensions=dimensions,
                          changed_values=changed, max_absolute=max(differences),
                          max_relative=max(relative),
                          reference_sha256=digest(Path(row["reference"])),
                          distorted_sha256=digest(Path(row["distorted"])))
            if args.diffmaps:
                result["map_sha256"] = [digest(p) for p in maps]
                result["changed_map_samples"] = 0
                result["max_map_absolute"] = 0.0
                if result["map_sha256"][0] != result["map_sha256"][1]:
                    summary["changed_map_pairs"] += 1
                    with maps[0].open("rb") as a, maps[1].open("rb") as b:
                        for block in iter(lambda: a.read(65536), b""):
                            for (x,), (y,) in zip(struct.iter_unpack("<f", block),
                                                  struct.iter_unpack("<f", b.read(len(block)))):
                                result["changed_map_samples"] += x != y
                                result["max_map_absolute"] = max(result["max_map_absolute"], abs(x-y))
                summary["changed_map_samples"] += result["changed_map_samples"]
                summary["max_map_absolute"] = max(summary["max_map_absolute"], result["max_map_absolute"])
            results.write(json.dumps(result) + "\n")
            summary["changed_pairs"] += bool(changed)
            summary["changed_values"] += changed
            for key in ("max_absolute", "max_relative"):
                summary[key] = max(summary[key], result[key])
            line = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + f" Compared {i + 1}/{len(pairs)}; changed={changed}"
            print(line, flush=True)
            print(line, file=progress)
    summary["status"] = "complete"
    summary["comparisons_sha256"] = digest(args.output / "comparisons.jsonl")
    (args.output / "_MANIFEST.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
