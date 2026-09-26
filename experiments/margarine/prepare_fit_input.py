#!/usr/bin/env python3
"""Validate Zenfleet's pairs export and freeze project-local source partitions.

This is a file-format adapter, not a job launcher or reconciler. Zenfleet owns
encoding, persistence and the ledger-to-pairs reduction.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

from score_manifest import TRAINING_FIELDS, digest


def broad_class(name):
    prefix = int(name.split("-", 1)[0])
    if prefix in (8000, 8100):
        return "screen"
    if prefix in (6000, 7000):
        return "line_art"
    if prefix < 4000 or prefix == 9226:
        return "photo_like"  # includes rendered/generated photo-like content
    return "mixed"


def partition_sources(sources, seed):
    groups = defaultdict(list)
    for source, content in sources.items():
        groups[broad_class(content)].append(source)
    result = {}
    for group, members in groups.items():
        ordered = sorted(members, key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).digest())
        n = math.ceil(len(ordered) / 5)
        if len(ordered) < 2 * n + 1:
            raise ValueError(f"not enough sources for fit/tune/test in {group}")
        for i, source in enumerate(ordered):
            result[source] = "test" if i < n else "tune" if i < 2 * n else "fit"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path, help="local stage holding refs/, blobs/, pairs.tsv and _MANIFEST.json")
    parser.add_argument("output", type=Path)
    parser.add_argument("--build-commit", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--container-root", type=Path, default=Path("/work"))
    parser.add_argument("--split-seed", default="20260926")
    args = parser.parse_args()
    manifest = json.loads((args.stage / "_MANIFEST.json").read_text())
    refs = {r["file"]: r for r in manifest["references"]}
    if len(refs) != len(manifest["references"]):
        raise ValueError("duplicate reference filename")
    sources = {r["source_id"]: r["content_class"] for r in refs.values()}
    partitions = partition_sources(sources, args.split_seed)
    args.output.mkdir(parents=True, exist_ok=False)
    progress = (args.output / "progress.log").open("x", buffering=1)

    def report(message):
        print(message, flush=True)
        print(message, file=progress, flush=True)

    for name, ref in refs.items():
        if digest(args.stage / "refs" / name) != ref["sha256"]:
            raise ValueError(f"reference hash mismatch: {name}")
    report(f"Verified {len(refs)} reference hashes")
    qualities, seen, blobs, modes = defaultdict(set), set(), set(), set()
    count, raw_map_bytes = 0, 0
    with (args.stage / "pairs.tsv").open() as pairs, (args.output / "extraction-input.tsv").open("x") as out:
        writer = csv.DictWriter(out, TRAINING_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in csv.DictReader(pairs, delimiter="\t"):
            if row["codec"] != manifest["codec"]:
                raise ValueError("codec disagrees with staged plan")
            ref_relative = Path(row["ref_path"]).relative_to(args.container_root)
            dist_relative = Path(row["dist_path"]).relative_to(args.container_root)
            if ".." in ref_relative.parts or ".." in dist_relative.parts or ref_relative.parent != Path("refs"):
                raise ValueError("unexpected pair path")
            ref = refs[ref_relative.name]
            reference, encoded = args.stage / ref_relative, args.stage / dist_relative
            if encoded not in blobs:
                if digest(encoded) != row["encode_sha"]:
                    raise ValueError(f"encoded hash mismatch: {encoded}")
                blobs.add(encoded)
            q = int(row["q"])
            knobs = json.loads(row["knob_tuple_json"])
            modes.add(knobs["cell"])
            key = (ref_relative.name, row["codec"], q, row["knob_tuple_json"])
            if key in seen:
                raise ValueError("duplicate declared cell")
            seen.add(key)
            qualities[ref_relative.name].add(q)
            pair = hashlib.sha256(json.dumps(key).encode()).hexdigest()
            writer.writerow(dict(dataset=args.dataset, source=ref["source_id"], codec=row["codec"], pair=pair,
                                 reference=str(reference.resolve()), distorted=str(encoded.resolve()),
                                 encoded=str(encoded.resolve()), bpp=encoded.stat().st_size * 8 / (ref["width"] * ref["height"]),
                                 setting=json.dumps(dict(q=q, knobs=knobs), separators=(",", ":"))))
            raw_map_bytes += ref["width"] * ref["height"] * 4
            count += 1
            if count % 1000 == 0:
                out.flush()
                report(f"Verified and translated {count} persisted encode cells")
    if set(qualities) != set(refs) or any(q != set(manifest["quality_grid"]) for q in qualities.values()):
        raise ValueError("reference/quality coverage differs from the staged grid")
    with (args.output / "splits.tsv").open("x") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source", "split", "broad_class", "original_content_class"])
        for source in sorted(sources):
            writer.writerow([source, partitions[source], broad_class(sources[source]), sources[source]])
    result = dict(build_commit=args.build_commit, input_manifest_sha256=digest(args.stage / "_MANIFEST.json"),
                  input_pairs_sha256=digest(args.stage / "pairs.tsv"), n_pairs=count,
                  n_encoded_blobs=len(blobs), n_references=len(refs), n_sources=len(sources),
                  raw_diffmap_storage_bytes=raw_map_bytes,
                  split_seed=args.split_seed, split_counts=dict(Counter(partitions.values())),
                  source_classes=dict(Counter(broad_class(c) for c in sources.values())),
                  encoder_modes=sorted(modes),
                  extraction_input_sha256=digest(args.output / "extraction-input.tsv"),
                  splits_sha256=digest(args.output / "splits.tsv"),
                  status="prepared; no fit performed",
                  limitation="single-mode feasibility data; class quotas and cross-corpus near-duplicate checks remain incomplete")
    (args.output / "_MANIFEST.json").write_text(json.dumps(result, indent=2) + "\n")
    report(f"Frozen source partitions and {count} extraction rows; raw map storage {raw_map_bytes} bytes")


if __name__ == "__main__":
    main()
