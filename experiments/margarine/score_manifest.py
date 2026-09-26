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
TRAINING_FIELDS = "dataset source codec pair reference distorted encoded bpp setting".split()
EDGE_COLUMNS = [f"feature_{i:03}" for i in range(228)
                if (i % 13 >= 3 if i < 156 else (i - 156) % 6 not in (0, 3))]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_pairs(path, training=False):
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        expected = ([TRAINING_FIELDS] if training else
                    [FIELDS, FIELDS + ["bpp", "setting"], FIELDS + ["sigma", "label_method"]])
        if reader.fieldnames not in expected:
            raise ValueError(f"expected manifest header: {expected}")
        rows = list(reader)
    if not rows or len({(r['dataset'], r['pair']) for r in rows}) != len(rows):
        raise ValueError("empty or duplicated manifest")
    return rows


def verify_input_audit(audit_path, pairs_path, images, dimensions):
    audit = json.loads(audit_path.read_text())
    if audit["status"] != "images-audited" or digest(pairs_path) != audit["pairs_sha256"]:
        raise ValueError("source audit is incomplete or pair manifest changed")
    root = Path(audit["destination_root"])
    expected = {str(root / name): meta for name, meta in audit["images"].items()}
    if set(expected) != set(images):
        raise ValueError("staged image set differs from source audit")
    for path, meta in expected.items():
        if (images[path] != meta["sha256"] or list(dimensions[path]) != meta["dimensions"]
                or Path(path).stat().st_size != meta["bytes"]):
            raise ValueError(f"staged image differs from source audit: {path}")


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


def parse_prediction(stdout, mode, dimensions):
    rows = list(csv.DictReader(io.StringIO(stdout), delimiter="\t"))
    if len(rows) != 1:
        raise ValueError("scorer must return exactly one row")
    row = rows[0]
    if row["mode"] != mode or (int(row["width"]), int(row["height"])) != dimensions:
        raise ValueError("wrong scorer mode or map dimensions")
    scores = {key: float(row[key]) for key in NORMS}
    if any(not math.isfinite(v) or v < 0 for v in scores.values()):
        raise ValueError("invalid metric output")
    return row, scores


def parse_score(stdout, mode, dimensions, map_path):
    row, scores = parse_prediction(stdout, mode, dimensions)
    if row["diffmap"] != str(map_path):
        raise ValueError("wrong diffmap path")
    if map_path.stat().st_size != dimensions[0] * dimensions[1] * 4:
        raise ValueError("wrong diffmap byte count")
    with map_path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            if any(not math.isfinite(v) or v < 0 for (v,) in struct.iter_unpack("<f", block)):
                raise ValueError("nonfinite or negative diffmap sample")
    return scores


def parse_features(path, dimensions):
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames != ["width", "height"] + EDGE_COLUMNS:
            raise ValueError("unexpected feature ordering")
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError("extractor must return exactly one row")
    row = rows[0]
    if (int(row["width"]), int(row["height"])) != dimensions:
        raise ValueError("wrong feature dimensions")
    values = [float(row[key]) for key in EDGE_COLUMNS]
    if any(not math.isfinite(v) for v in values):
        raise ValueError("nonfinite extracted feature")
    return values


def frozen_teacher(directory):
    manifest = json.loads((directory / "_MANIFEST.json").read_text())
    if manifest.get("status") not in ("complete", "scores-complete") or manifest.get("mode") not in (None, "quality-evaluation"):
        raise ValueError("requires complete human-evaluation teacher ledger")
    if digest(directory / "cells.jsonl") != manifest["cells_sha256"]:
        raise ValueError("teacher ledger hash mismatch")
    rows = {}
    with (directory / "cells.jsonl").open() as ledger:
        for line in ledger:
            row = json.loads(line)
            # Earlier frozen runs kept input hashes in their manifest, before cells
            # carried them directly. Resolve that recorded provenance, never rehash
            # today's files as a substitute for the original input identity.
            for name in ("reference", "distorted"):
                if name + "_sha256" not in row:
                    row[name + "_sha256"] = manifest["images"][row[name]]
            key = row["dataset"], row["pair"]
            if key in rows:
                raise ValueError("duplicate teacher pair")
            rows[key] = row
    return rows


def aligned_teacher(row, frozen, images, dimensions):
    if any(row[key] != frozen[key] for key in FIELDS[:6]):
        raise ValueError("teacher evaluation labels or identities differ")
    for key in ("reference", "distorted"):
        if images[row[key]] != frozen[key + "_sha256"]:
            raise ValueError("teacher input hash mismatch")
    scores = frozen["scores"]["teacher"]
    if (scores["width"], scores["height"]) != dimensions:
        raise ValueError("teacher dimensions differ")
    if any(not math.isfinite(scores[n]) or scores[n] < 0 for n in NORMS):
        raise ValueError("invalid cached teacher score")
    return scores


def evaluate_panels(cells, candidate, evaluator, output, report, published_sigma=True):
    for norm in NORMS:
        path = output / f"scores-{norm}.tsv"
        fields = FIELDS[:6] + ["teacher", "candidate"]
        with path.open("x", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            for cell in cells:
                writer.writerow(dict(**{key: cell[key] for key in FIELDS[:6]},
                                     teacher=cell["scores"]["teacher"][norm],
                                     candidate=cell["scores"][candidate][norm]))
        with (output / f"eval-{norm}.log").open("x") as log:
            subprocess.run([str(evaluator), str(path),
                            str(output / f"panel-{norm}.tsv"), "0"],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        if published_sigma and any("sigma" in cell for cell in cells):
            sigma_path = output / f"scores-published-sigma-{norm}.tsv"
            with sigma_path.open("x", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields + ["sigma"], delimiter="\t")
                writer.writeheader()
                for cell in cells:
                    writer.writerow(dict(**{key: cell[key] for key in FIELDS[:6]},
                                         teacher=cell["scores"]["teacher"][norm],
                                         candidate=cell["scores"][candidate][norm],
                                         sigma=cell.get("sigma", "")))
            with (output / f"eval-published-sigma-{norm}.log").open("x") as log:
                subprocess.run([str(evaluator), "--published-sigma", str(sigma_path),
                                str(output / f"panel-published-sigma-{norm}.tsv")],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        report(f"Evaluated {norm}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("binaries", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--build-commit", required=True)
    parser.add_argument("--ingress", choices=("aic-rgb8", "cid22-srgb", "common-srgb"), default="aic-rgb8")
    parser.add_argument("--teacher-features", action="store_true",
                        help="extract teacher maps/norms plus a separate 168-feature sidecar; no labels or quality evaluation")
    parser.add_argument("--features-only", action="store_true",
                        help="refresh the separate training feature sidecar without recomputing teacher maps")
    parser.add_argument("--feature-source", help="required extractor dependency commit for training feature runs")
    parser.add_argument("--candidate", default="box3", choices=["box3", "multirate", "compact", "compact4", "sparse", "pooled", "perceptual", "physical", "refined", "refined1", "refined2", "stratified", "anchored-pool", "bounded", "lattice", "tiles", "planar", "planar-tiles", "stream-blur", "reference-regions", "stable-peak"], help="direct approximation identity")
    parser.add_argument("--model", type=Path, help="frozen fit directory, with model.tsv and provenance")
    parser.add_argument("--teacher", type=Path, help="existing human-evaluation score directory")
    parser.add_argument("--input-audit", type=Path, help="source-audited prepare_human manifest; verify staged hashes before scoring")
    parser.add_argument("--defer-panels", action="store_true", help="persist scores/maps for separate panel evaluation")
    args = parser.parse_args()
    if args.defer_panels and (args.teacher_features or args.features_only):
        parser.error("defer-panels applies only to human-quality scoring")
    if args.teacher_features and args.features_only:
        parser.error("choose teacher-features or features-only")
    training = args.teacher_features or args.features_only
    if (args.model and not args.teacher) or (args.teacher and training):
        parser.error("--model requires --teacher; cached teachers apply only to quality evaluation")
    if training and not args.feature_source:
        parser.error("training extraction requires --feature-source")
    rows = read_pairs(args.manifest, training)
    args.output.mkdir(parents=True, exist_ok=False)
    maps = args.output / "maps"
    maps.mkdir()
    progress = (args.output / "progress.log").open("x", buffering=1)

    def report(message):
        message = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}"
        print(message, flush=True)
        print(message, file=progress, flush=True)

    names = ["margarine-box3"] if args.features_only or args.teacher else ["margarine-score", "margarine-box3"]
    if not training and not args.defer_panels:
        names.append("margarine-eval")
    binaries = {name: args.binaries.resolve() / name for name in names}
    provenance = dict(build_commit=args.build_commit, pairs_sha256=digest(args.manifest),
                      binaries={n: digest(p) for n, p in binaries.items()},
                      ingress=args.ingress,
                      n_pairs=len(rows), images={}, status="running")
    provenance["mode"] = ("features-only" if args.features_only else
                          "teacher-features" if training else "quality-evaluation")
    candidate = "student" if args.model else args.candidate
    provenance["candidate"] = candidate
    if args.model:
        model = args.model / "model.tsv"
        fitted = json.loads((args.model / "_MANIFEST.json").read_text())
        if digest(model) != fitted["model_sha256"]:
            raise ValueError("model hash mismatch")
        if provenance["binaries"]["margarine-box3"] != fitted["feature_binaries"]["margarine-box3"]:
            raise ValueError("runtime binary differs from fitted feature extractor")
        provenance.update(model_sha256=digest(model),
                          model_manifest_sha256=digest(args.model / "_MANIFEST.json"))
    if args.teacher:
        teachers = frozen_teacher(args.teacher)
        if set(teachers) != {(r["dataset"], r["pair"]) for r in rows}:
            raise ValueError("teacher and evaluation pair sets differ")
        provenance.update(teacher_directory=str(args.teacher.resolve()),
                          teacher_manifest_sha256=digest(args.teacher / "_MANIFEST.json"),
                          teacher_cells_sha256=digest(args.teacher / "cells.jsonl"),
                          spatial_output=("teacher maps retained in original store; student predicts scalar norms only"
                                          if args.model else "teacher maps retained in original store; candidate maps persisted"))
    if training:
        provenance["feature_columns"] = EDGE_COLUMNS
        provenance["feature_profile"] = "168 edges; 256-row strips, 64-row halo"
        provenance["feature_source"] = args.feature_source
    shutil.copyfile(args.manifest, args.output / "input_pairs.tsv")
    manifest_path = args.output / "_MANIFEST.json"
    manifest_path.write_text(json.dumps(provenance, indent=2) + "\n")
    if args.ingress in ("cid22-srgb", "common-srgb"):
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
        if training and row["encoded"] not in provenance["images"]:
            provenance["images"][row["encoded"]] = digest(Path(row["encoded"]))
    if args.input_audit:
        verify_input_audit(args.input_audit, args.manifest, provenance["images"], dimensions)
        provenance.update(input_audit=str(args.input_audit.resolve()), input_audit_sha256=digest(args.input_audit))
    manifest_path.write_text(json.dumps(provenance, indent=2) + "\n")
    report(f"Audited {len(dimensions)} images, scoring {len(rows)} pairs")
    from contextlib import ExitStack
    with ExitStack() as stack:
        cells = stack.enter_context((args.output / "cells.jsonl").open("x", buffering=1))
        features = (stack.enter_context((args.output / "features.jsonl").open("x", buffering=1))
                    if training else None)
        for i, row in enumerate(rows):
            cell = dict(row)
            cell["reference_sha256"] = provenance["images"][row["reference"]]
            cell["distorted_sha256"] = provenance["images"][row["distorted"]]
            cell["encoded_sha256"] = provenance["images"][row.get("encoded", row["distorted"])]
            cell["scores"] = {}
            if args.teacher:
                cell["scores"]["teacher"] = aligned_teacher(
                    row, teachers[row["dataset"], row["pair"]], provenance["images"],
                    dimensions[row["reference"]])
            if args.model:
                run = subprocess.run([str(binaries["margarine-box3"]), "--student", str(model),
                                      row["reference"], row["distorted"]],
                                     capture_output=True, text=True, check=False)
                (args.output / f"cell-{i}-student.log").write_text(run.stdout + run.stderr)
                run.check_returncode()
                _, scores = parse_prediction(run.stdout, "margarine-probe", dimensions[row["reference"]])
                cell["scores"][candidate] = scores
            modes = (() if args.features_only or args.model else
                     ("teacher",) if training else (candidate,) if args.teacher else ("teacher", candidate))
            for mode in modes:
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
            if features is not None:
                path = args.output / f"features-{i}.tsv"
                run = subprocess.run([str(binaries["margarine-box3"]), "--export-edges",
                                      row["reference"], row["distorted"], str(path)],
                                     capture_output=True, text=True, check=False)
                (args.output / f"cell-{i}-features.log").write_text(run.stdout + run.stderr)
                if run.returncode:
                    raise RuntimeError(f"feature extraction failed: {i}, see cell log")
                values = parse_features(path, dimensions[row["reference"]])
                features.write(json.dumps(dict(reference_sha256=cell["reference_sha256"],
                                               encoded_sha256=cell["encoded_sha256"],
                                               features=values)) + "\n")
            cells.write(json.dumps(cell) + "\n")
            report(f"Scored {i + 1}/{len(rows)} {row['dataset']} {row['pair']}")
    with (args.output / "cells.jsonl").open() as f:
        cells = [json.loads(line) for line in f]
    provenance["status"] = "scores-complete"
    provenance["cells_sha256"] = digest(args.output / "cells.jsonl")
    provenance["candidate"] = candidate
    manifest_path.write_text(json.dumps(provenance, indent=2) + "\n")
    if args.defer_panels:
        report("Scores and maps complete; statistical panels explicitly deferred")
        return
    if not training:
        evaluate_panels(cells, candidate, binaries["margarine-eval"], args.output, report)
    provenance["status"] = "complete"
    provenance["cells_sha256"] = digest(args.output / "cells.jsonl")
    if training:
        provenance["features_sha256"] = digest(args.output / "features.jsonl")
    manifest_path.write_text(json.dumps(provenance, indent=2) + "\n")
    report("Complete; resource benchmarks and remaining evaluation gates are separate")


if __name__ == "__main__":
    main()
