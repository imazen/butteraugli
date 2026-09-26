#!/usr/bin/env python3
"""Serial local evaluation of a frozen candidate; not a timing benchmark.

Persists all five scalar variants and content-addressed native diffmaps before
running zenstats. Distributed jobs belong in zenmetrics/zenfleet instead.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import time

FIELDS = "dataset source codec pair target direction reference distorted".split()
NORMS = "max p1 p2 p3 p6".split()


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit_png(path):
    """Require untagged RGB8: explicitly interpret both arms as common sRGB."""
    with path.open("rb") as f:
        if f.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"not PNG: {path}")
        dimensions = None
        while True:
            header = f.read(8)
            if len(header) != 8:
                raise ValueError(f"truncated PNG: {path}")
            n, tag = struct.unpack(">I4s", header)
            if tag in (b"gAMA", b"cHRM", b"sRGB", b"iCCP", b"cICP"):
                raise ValueError(f"color tag {tag!r} needs managed ingress: {path}")
            if tag == b"IHDR":
                data = f.read(n)
                if n != 13 or data[8:10] != bytes([8, 2]):
                    raise ValueError(f"not RGB8: {path}")
                dimensions = struct.unpack(">II", data[:8])
            else:
                f.seek(n, 1)
            if len(f.read(4)) != 4:
                raise ValueError(f"truncated PNG chunk: {path}")
            if tag == b"IEND":
                if dimensions is None:
                    raise ValueError(f"missing IHDR: {path}")
                return dimensions


def parse_score(stdout, mode, dimensions, map_path):
    rows = list(csv.DictReader(io.StringIO(stdout), delimiter="\t"))
    if len(rows) != 1:
        raise ValueError("scorer must return exactly one row")
    row = rows[0]
    if row["mode"] != mode or (int(row["width"]), int(row["height"])) != dimensions:
        raise ValueError("wrong scorer mode or map dimensions")
    if row["diffmap"] != str(map_path):
        raise ValueError("wrong diffmap path")
    scores = {key: float(row[key]) for key in NORMS}
    if any(not math.isfinite(v) or v < 0 for v in scores.values()):
        raise ValueError("invalid metric output")
    if map_path.stat().st_size != dimensions[0] * dimensions[1] * 4:
        raise ValueError("wrong diffmap byte count")
    with map_path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            if any(not math.isfinite(v) or v < 0 for (v,) in struct.iter_unpack("<f", block)):
                raise ValueError("nonfinite or negative diffmap sample")
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("binaries", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--build-commit", required=True)
    parser.add_argument("--ingress", choices=("aic-rgb8", "cid22-srgb"), default="aic-rgb8")
    args = parser.parse_args()
    with args.manifest.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames not in (FIELDS, FIELDS + ["bpp", "setting"]):
            raise ValueError(f"expected manifest header: {FIELDS}")
        rows = list(reader)
    if not rows or len({(r['dataset'], r['pair']) for r in rows}) != len(rows):
        raise ValueError("empty or duplicated manifest")
    args.output.mkdir(parents=True, exist_ok=False)
    maps = args.output / "maps"
    maps.mkdir()
    progress = (args.output / "progress.log").open("x", buffering=1)

    def report(message):
        message = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}"
        print(message, flush=True)
        print(message, file=progress, flush=True)

    binaries = {name: args.binaries.resolve() / name for name in
                ("margarine-score", "margarine-box3", "margarine-eval")}
    provenance = dict(build_commit=args.build_commit, pairs_sha256=digest(args.manifest),
                      binaries={n: digest(p) for n, p in binaries.items()},
                      ingress=args.ingress,
                      n_pairs=len(rows), images={}, status="running")
    shutil.copyfile(args.manifest, args.output / "input_pairs.tsv")
    manifest_path = args.output / "_MANIFEST.json"
    manifest_path.write_text(json.dumps(provenance, indent=2) + "\n")
    if args.ingress == "cid22-srgb":
        from cid22_manifest import audit as audit_image
    else:
        audit_image = audit_png
    dimensions = {}
    for row in rows:
        for name in ("reference", "distorted"):
            path = Path(row[name])
            if str(path) not in dimensions:
                dimensions[str(path)] = audit_image(path)
                provenance["images"][str(path)] = digest(path)
                if len(dimensions) % 100 == 0:
                    report(f"Audited {len(dimensions)} images")
        if dimensions[row["reference"]] != dimensions[row["distorted"]]:
            raise ValueError(f"mismatched dimensions: {row['pair']}")
    manifest_path.write_text(json.dumps(provenance, indent=2) + "\n")
    report(f"Audited {len(dimensions)} images, scoring {len(rows)} pairs")
    with (args.output / "cells.jsonl").open("x", buffering=1) as cells:
        for i, row in enumerate(rows):
            cell = dict(row)
            cell["scores"] = {}
            for mode in ("teacher", "box3"):
                path = maps / f"pending-{i}-{mode}.f32le"
                cmd = ([str(binaries["margarine-score"]), "teacher"] if mode == "teacher"
                       else [str(binaries["margarine-box3"])])
                run = subprocess.run(cmd + [row["reference"], row["distorted"], str(path)],
                                     capture_output=True, text=True, check=False)
                (args.output / f"cell-{i}-{mode}.log").write_text(run.stdout + run.stderr)
                if run.returncode:
                    raise RuntimeError(f"scorer failed: {i} {mode}, see cell log")
                scores = parse_score(run.stdout, mode, dimensions[row["reference"]], path)
                sha = digest(path)
                final = maps / f"{sha}.f32le"
                if final.exists():
                    if digest(final) != sha:
                        raise ValueError("existing content-addressed map corrupted")
                    # Preserve the duplicate too; never delete generated artifacts.
                else:
                    path.rename(final)
                cell["scores"][mode] = dict(**scores, diffmap_sha256=sha,
                                             width=dimensions[row["reference"]][0],
                                             height=dimensions[row["reference"]][1])
            cells.write(json.dumps(cell) + "\n")
            report(f"Scored {i + 1}/{len(rows)} {row['dataset']} {row['pair']}")
    with (args.output / "cells.jsonl").open() as f:
        cells = [json.loads(line) for line in f]
    for norm in NORMS:
        path = args.output / f"scores-{norm}.tsv"
        fields = FIELDS[:6] + ["teacher", "candidate"]
        with path.open("x", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            for cell in cells:
                writer.writerow(dict(**{key: cell[key] for key in FIELDS[:6]},
                                     teacher=cell["scores"]["teacher"][norm],
                                     candidate=cell["scores"]["box3"][norm]))
        with (args.output / f"eval-{norm}.log").open("x") as log:
            subprocess.run([str(binaries["margarine-eval"]), str(path),
                            str(args.output / f"panel-{norm}.tsv"), "0"],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        report(f"Evaluated {norm}")
    provenance["status"] = "complete"
    provenance["cells_sha256"] = digest(args.output / "cells.jsonl")
    manifest_path.write_text(json.dumps(provenance, indent=2) + "\n")
    report("Complete; resource benchmarks and remaining evaluation gates are separate")


if __name__ == "__main__":
    main()
