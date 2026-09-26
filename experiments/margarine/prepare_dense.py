#!/usr/bin/env python3
"""Prepare dense reference renders from existing train-only centroid representatives.

Pilot preparation only: all full-image representatives are retained. This does
not claim minimum per-class coverage, fit a model, or dispatch a fleet. The caller
must claim the canonical render checkout before reading it through this tool.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlparse


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("renders", type=Path)
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--build-commit", required=True)
    parser.add_argument("--render-commit", required=True)
    parser.add_argument("--metadata-commit", required=True)
    parser.add_argument("--prepare-only", action="store_true", help="audit sources without rendering")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    catalog = {r["id"]:r for r in csv.DictReader((args.metadata/"train.tsv").open(), delimiter="\t")}
    reps_path = args.metadata/"imazen26_representatives_K500_even_2026-06-18.tsv"
    reps = [r for r in csv.DictReader(reps_path.open(), delimiter="\t") if r["crop_label"] == "full"]
    if not reps: raise ValueError("no full-image representatives")
    sources, files, seen = [], [], set()
    def checkpoint(complete=False):
        manifest = dict(build_commit=args.build_commit, metadata_commit=args.metadata_commit,
            render_commit=args.render_commit, binary_sha256=sha(args.binary),
            metadata_sha256={p.name:sha(p) for p in (reps_path,args.metadata/"train.tsv")},
            selection="all full-image representatives from existing K500 even-source clustering; no quality labels used",
            limitation="pilot only; per-class coverage and cross-corpus near-duplicate audit not complete; no model fitted",
            source_classes=dict(Counter(s["content_class"] for s in sources)), sources=sources,
            complete=complete,
            quality_grid=list(range(0,101,2)), sizes_per_source=20, references=files,
            variant_registration="pending; no canonical registry changed")
        (args.output/"_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
    checkpoint()
    with (args.output/"progress.log").open("x", buffering=1) as progress:
        def report(s): print(s, flush=True); print(s,file=progress,flush=True)
        for i, rep in enumerate(reps):
            token = Path(unquote(urlparse(rep["url"]).path)).name.split("_")[0]
            if not token.isdigit() or int(token)%2 or token in seen:
                raise ValueError(f"invalid or repeated train source {token}")
            seen.add(token)
            cat = catalog[token]
            if cat["split"] != "train" or cat["content_class"] != rep["content_class"]:
                raise ValueError("representative disagrees with canonical training split/class")
            # Current catalog URL is authoritative; representative URLs may predate rotation fixes.
            url = cat["png_v3_sdr_url"]
            prefix = "/imazen-26-png-v3/"
            parsed = urlparse(url)
            if parsed.scheme != "https" or parsed.hostname != "codec-corpus.r2.imazen.org" or not parsed.path.startswith(prefix):
                raise ValueError("unexpected canonical render URL")
            relative = Path("png-v3")/unquote(parsed.path[len(prefix):])
            if ".." in relative.parts: raise ValueError("invalid render path")
            path = args.renders/relative
            pointer = subprocess.check_output(["git","-C",str(args.renders),"show",f"{args.render_commit}:{relative}"],text=True)
            match = re.search(r"^oid sha256:([0-9a-f]{64})$",pointer,re.M)
            digest = sha(path)
            if not match or digest != match.group(1): raise ValueError(f"render does not match pinned LFS object: {path}")
            source = dict(source_id=token, split="train", content_class=cat["content_class"],
                cluster_id=rep["cluster_id"], cluster_size=rep["cluster_size"], crop_label="full",
                reference_url=url, path=str(path.resolve()), sha256=digest)
            sources.append(source)
            checkpoint()
            report(f"{i+1}/{len(reps)}: verified source {token}, cluster {rep['cluster_id']}")
            if not args.prepare_only:
                directory = args.output/token
                with (args.output/f"{token}-render.log").open("x") as log:
                    subprocess.run([str(args.binary.resolve()),"--render-dense",str(path),str(directory)], stdout=log, stderr=subprocess.STDOUT, check=True)
                for rendered in csv.DictReader((directory/"renditions.tsv").open(),delimiter="\t"):
                    f=Path(rendered["path"])
                    files.append(dict(**source, width=int(rendered["width"]), height=int(rendered["height"]),
                        rendition=str(f.resolve()), rendition_sha256=sha(f), kernel=rendered["kernel"]))
                checkpoint()
                report(f"{token}: twenty sizes persisted")
        checkpoint(complete=True)
        report(f"Complete: {len(sources)} audited sources, {len(files)} rendered references")


if __name__ == "__main__": main()
