#!/usr/bin/env python3
"""Build a strict PTC-image manifest from the original AIC4_sample JND CSV."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import sys

LABEL_REV = "56723f70180637aa9916664c018cd82b955b6b4f"
LABEL_URL = f"https://raw.githubusercontent.com/jpeg-aic/JPEG-AIC-4-datasets/{LABEL_REV}/JPEG_AIC_reconstructed_jnd_scores.csv"
LABEL_SHA = "3f746bc6bc7f3a0fb3c7537ebc8b107f975014c26029333f5c9b4de4fffb59f9"
FIELDS = "dataset source codec pair target direction reference distorted".split()


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="JPEG_AIC-4_Sample_Dataset directory")
    parser.add_argument("labels", type=Path, help="original pinned reconstructed JND CSV")
    parser.add_argument("output", type=Path, help="new directory")
    parser.add_argument("--build-commit", required=True)
    args = parser.parse_args()
    if digest(args.labels) != LABEL_SHA:
        raise ValueError(f"expected original label CSV from {LABEL_URL}")
    data = list(csv.DictReader(args.labels.open()))
    if len(data) != 300:
        raise ValueError("expected all 300 published pairs")
    root = args.root.resolve()
    rows, images, seen = [], {}, set()
    for r in data:
        source = r["img_num"].zfill(5)
        pair = r["img_distorted"]
        codec = pair.split("_")[2]
        target = float(r["distortion"])
        if pair in seen or not math.isfinite(target):
            raise ValueError(f"duplicate pair or invalid JND: {pair}")
        seen.add(pair)
        paths = []
        for field in ("img_source", "img_distorted"):
            name = r[field]
            if Path(name).name != name:
                raise ValueError("image name must be a basename")
            relative = Path("PTC_images") / source / name
            path = root / relative
            if str(relative) not in images:
                with path.open("rb") as f:
                    header = f.read(29)
                if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
                    raise ValueError(f"invalid PNG: {path}")
                dimensions = struct.unpack(">II", header[16:24])
                if dimensions != (620, 800) or header[24:26] != bytes([8, 2]):
                    raise ValueError(f"unexpected PTC image geometry/format: {path}")
                images[str(relative)] = {"sha256": digest(path), "bytes": path.stat().st_size}
            paths.append(str(path))
        rows.append(dict(zip(FIELDS, ("aic4_sample", source, codec, pair,
                                     r["distortion"], "distortion", *paths))))
        if len(rows) % 25 == 0:
            print(f"Verified {len(rows)}/300 pairs", file=sys.stderr, flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "pairs.tsv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    shutil.copyfile(args.labels, args.output / "original_jnd.csv")
    shutil.copyfile(root / "README.md", args.output / "dataset_README.md")
    report = dict(build_commit=args.build_commit, dataset="aic4_sample", n_pairs=len(rows),
                  n_sources=len({r["source"] for r in rows}), codecs=sorted({r["codec"] for r in rows}),
                  dimensions=[620, 800], labels_url=LABEL_URL, labels_sha256=LABEL_SHA,
                  target="original reconstructed JND, larger means more distortion",
                  images=images, pairs_sha256=digest(args.output / "pairs.tsv"),
                  colour_management="not audited; RGB8 header checks only")
    with (args.output / "_MANIFEST.json").open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"Wrote {len(rows)} pairs, {len(images)} distinct images to {args.output}", flush=True)


if __name__ == "__main__":
    main()
