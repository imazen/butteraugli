#!/usr/bin/env python3
"""Audit AIC2026 CSV/ZIP alignment without extracting the image corpus.

Writes a scoring manifest with archive-member identities. Human targets and
split assignments are explicitly unavailable: metric-derived JND columns are
not subjective labels. Source metadata and hyperlinks are preserved verbatim.
"""

import argparse
import collections
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import sys
from zipfile import ZipFile


ARCHIVE = "AIC2026-dataset-complete.zip"
METRICS = "metrics_fullres.csv"
PROVENANCE = (
    "readme_AIC2026.md",
    "AIC2026_source_images_metadata_and_attribution.csv",
    "darus_dataset_metadata_v2.0.json",
    "encoding_recipes.md",
    "SHA256SUMS",
    "MD5SUMS.zips",
)
FIELDS = (
    "dataset", "pair", "source", "codec", "distortion_level", "bpp",
    "archive", "reference_member", "distorted_member",
    "reference_crc32", "distorted_crc32", "human_target_status", "split_status",
)


def basename(value):
    if not value or PurePosixPath(value).name != value or "\\" in value:
        raise ValueError(f"expected a filename, got {value!r}")
    return value


def audit(metrics_text, members, archive):
    """Pure alignment check, also used by the self-contained regression tests."""
    reader = csv.DictReader(io.StringIO(metrics_text))
    required = {"distorted", "source", "codec_acronym", "distortion_level", "bpp"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError(f"missing columns: {sorted(required - set(reader.fieldnames or []))}")
    # Duplicate ZIP members can decode differently depending on the reader.
    index = {}
    for member in members:
        if member.filename in index:
            raise ValueError(f"duplicate ZIP member: {member.filename}")
        index[member.filename] = member
    rows = []
    seen = set()
    for row in reader:
        pair = basename(row["distorted"])
        source = basename(row["source"])
        if pair in seen:
            raise ValueError(f"duplicate metric row: {pair}")
        seen.add(pair)
        codec = row["codec_acronym"]
        if not codec or any(c in codec for c in "\t\r\n"):
            raise ValueError(f"invalid codec: {codec!r}")
        level = int(row["distortion_level"])
        bpp = float(row["bpp"])
        if level <= 0 or not math.isfinite(bpp) or bpp < 0:
            raise ValueError(f"invalid level/bitrate for {pair}")
        ref_member = f"sources/{source}"
        dist_member = f"distorted/{pair}"
        for name in (ref_member, dist_member):
            if name not in index or index[name].is_dir():
                raise ValueError(f"missing image member: {name}")
        rows.append(dict(zip(FIELDS, (
            "aic2026-fullres", pair, source, codec, level, row["bpp"],
            str(archive), ref_member, dist_member,
            f"{index[ref_member].CRC:08x}", f"{index[dist_member].CRC:08x}",
            "unavailable", "unassigned",
        ))))
    if not rows:
        raise ValueError("empty metric table")
    named_distortions = {r["distorted_member"] for r in rows}
    unused = sorted(
        name for name, member in index.items()
        if not member.is_dir() and name.startswith("distorted/")
        and name not in named_distortions
    )
    if unused:
        raise ValueError(f"{len(unused)} distorted archive members lack metric rows: {unused[:3]}")
    return rows, reader.fieldnames


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path, help="new directory, never overwritten")
    parser.add_argument("--build-commit", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    archive = root / ARCHIVE
    for name in (METRICS, *PROVENANCE):
        if not (root / name).is_file():
            raise FileNotFoundError(f"required provenance missing: {root / name}")
    print(f"Auditing {archive}", file=sys.stderr, flush=True)
    with ZipFile(archive) as z:
        members = z.infolist()
        rows, columns = audit((root / METRICS).read_text(), members, archive)
    print(f"Aligned {len(rows)} pairs; writing manifest", file=sys.stderr, flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = args.output / "pairs.tsv"
    with manifest.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    # Copy, do not render or strip markup; all original links survive.
    provenance = args.output / "provenance"
    provenance.mkdir()
    files = {}
    for name in (METRICS, *PROVENANCE):
        source = root / name
        if not source.is_file():
            raise FileNotFoundError(f"required provenance missing: {source}")
        shutil.copyfile(source, provenance / name)
        files[name] = sha256(provenance / name)
    report = {
        "build_commit": args.build_commit,
        "dataset_url": "https://doi.org/10.18419/DARUS-6156",
        "root": str(root),
        "n_pairs": len(rows),
        "n_sources": len({r["source"] for r in rows}),
        "codecs": dict(sorted(collections.Counter(r["codec"] for r in rows).items())),
        "metric_columns": columns,
        "human_targets": "not imported by this instrument; never inferred from JND_*",
        "split_assignments": "unavailable",
        "zip_members": len(members),
        "zip_bytes": archive.stat().st_size,
        "zip_content_checksum_verified": False,
        "zip_crc_note": "CRCs copied from central directory, not independently verified",
        "pairs_sha256": sha256(manifest),
        "provenance_sha256": files,
    }
    with (args.output / "_MANIFEST.json").open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
