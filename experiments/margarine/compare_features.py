#!/usr/bin/env python3
"""Compare two frozen edge exporters on explicit pairs; no fitting or thresholds."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import time

from score_manifest import digest, parse_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", type=Path)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--build-commit", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with args.pairs.open() as file:
        pairs = list(csv.DictReader(file, delimiter="\t"))
    if not pairs:
        raise ValueError("empty comparison")
    summary = dict(build_commit=args.build_commit, pairs_sha256=digest(args.pairs),
                   binaries={name: digest(getattr(args, name)) for name in ("before", "after")},
                   pairs=len(pairs), changed_pairs=0, changed_values=0, max_absolute=0.0,
                   max_relative=0.0, status="running")
    with (args.output / "progress.log").open("x", buffering=1) as progress, \
            (args.output / "comparisons.jsonl").open("x", buffering=1) as results:
        for i, row in enumerate(pairs):
            values = []
            for name in ("before", "after"):
                output = args.output / f"{i}-{name}.tsv"
                run = subprocess.run([str(getattr(args, name)), "--export-edges",
                                      row["reference"], row["distorted"], str(output)],
                                     capture_output=True, text=True, check=False)
                (args.output / f"{i}-{name}.log").write_text(run.stdout + run.stderr)
                run.check_returncode()
                with output.open() as file:
                    extracted = next(csv.DictReader(file, delimiter="\t"))
                dimensions = int(extracted["width"]), int(extracted["height"])
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
