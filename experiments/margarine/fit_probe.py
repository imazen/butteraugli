#!/usr/bin/env python3
"""Private feasibility fit against teacher scores, never human evaluation labels.

The small nonnegative model is a diagnostic for the 168-feature representation.
Its output is an experiment artifact, not a shipped Margarine implementation.
NumPy/SciPy dependencies are pinned in requirements-training.txt.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import time

from score_manifest import EDGE_COLUMNS, NORMS, digest


def read_splits(path):
    with path.open() as file:
        rows = list(csv.DictReader(file, delimiter="\t"))
    result = {}
    for row in rows:
        source, split = row["source"], row["split"]
        if source in result or not source or split not in ("fit", "tune", "test"):
            raise ValueError("invalid or duplicate source partition")
        result[source] = split
    if set(result.values()) != {"fit", "tune", "test"}:
        raise ValueError("all three source partitions are required")
    return result


def json_rows(path):
    with path.open() as file:
        for line in file:
            yield json.loads(line)


def transform(features, scale):
    import numpy as np
    features, scale = np.asarray(features, dtype=np.float64), np.asarray(scale, dtype=np.float64)
    if (not np.isfinite(features).all() or (features < 0).any()
            or not np.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError("features must be finite/nonnegative and scales finite/positive")
    return np.log1p(features / scale)


def source_weights(sources):
    import numpy as np
    counts = Counter(sources)
    return np.array([1.0 / (len(counts) * counts[s]) for s in sources])


def fit_nonnegative(x, y, sources, penalty):
    import numpy as np
    from scipy.optimize import nnls
    if penalty <= 0:
        raise ValueError("regularization must be positive")
    weights = np.sqrt(source_weights(sources))
    a = np.vstack([x * weights[:, None], np.eye(x.shape[1]) * np.sqrt(penalty)])
    b = np.concatenate([y * weights, np.zeros(x.shape[1])])
    coefficients, _ = nnls(a, b, maxiter=50 * x.shape[1])
    return coefficients


def main():
    import numpy as np
    import scipy
    from scipy.stats import spearmanr
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extraction", type=Path)
    parser.add_argument("splits", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--build-commit", required=True)
    parser.add_argument("--features", type=Path,
                        help="complete refreshed feature extraction; teacher scores remain in extraction")
    args = parser.parse_args()
    splits = read_splits(args.splits)
    manifest = json.loads((args.extraction / "_MANIFEST.json").read_text())
    if (manifest.get("status") != "complete" or manifest.get("mode") != "teacher-features"
            or manifest.get("feature_columns") != EDGE_COLUMNS):
        raise ValueError("requires a complete extraction with the pinned feature layout")
    if digest(args.extraction / "cells.jsonl") != manifest["cells_sha256"]:
        raise ValueError("teacher extraction hash mismatch")
    feature_dir = args.features or args.extraction
    feature_manifest = json.loads((feature_dir / "_MANIFEST.json").read_text())
    if (feature_manifest.get("status") != "complete"
            or feature_manifest.get("mode") not in ("teacher-features", "features-only")
            or feature_manifest.get("feature_columns") != EDGE_COLUMNS):
        raise ValueError("requires complete compatible feature extraction")
    if digest(feature_dir / "features.jsonl") != feature_manifest["features_sha256"]:
        raise ValueError("feature extraction hash mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    progress = (args.output / "progress.log").open("x", buffering=1)

    def report(message):
        line = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + " " + message
        print(line, flush=True)
        print(line, file=progress, flush=True)

    features = {}
    for row in json_rows(feature_dir / "features.jsonl"):
        key = row["reference_sha256"], row["encoded_sha256"]
        values = row["features"]
        if len(values) != 168 or (key in features and features[key] != values):
            raise ValueError("inconsistent feature sidecar")
        features[key] = values
    cells, matrix, targets, sources, seen = [], [], [], [], set()
    reference_partitions = {}
    for row in json_rows(args.extraction / "cells.jsonl"):
        if any(token in row["dataset"].lower() for token in ("aic", "cid22")) or "target" in row:
            raise ValueError("human evaluation data cannot enter this feasibility fit")
        source = row["source"]
        if source not in splits:
            raise ValueError(f"source has no frozen partition: {source}")
        key = row["reference_sha256"], row["encoded_sha256"]
        previous = reference_partitions.setdefault(key[0], splits[source])
        if previous != splits[source]:
            raise ValueError("identical reference bytes cross source partitions")
        # Identical encoded bytes of a reference are one training observation,
        # regardless of how many quality settings produced them. Raw cells remain.
        if key in seen:
            continue
        seen.add(key)
        cells.append(row)
        matrix.append(features[key])
        targets.append([row["scores"]["teacher"][n] for n in NORMS])
        sources.append(source)
        if len(cells) % 10000 == 0:
            report(f"Loaded {len(cells)} unique reference/encode observations")
    if set(sources) != set(splits):
        raise ValueError("partition sources and extraction sources differ")
    if set(features) != seen:
        raise ValueError("teacher and feature observation keys differ")
    x = np.asarray(matrix, dtype=np.float64)
    raw_y, sources = np.asarray(targets, dtype=np.float64), np.asarray(sources)
    if not np.isfinite(raw_y).all() or (raw_y < 0).any():
        raise ValueError("invalid teacher targets")
    masks = {name: np.array([splits[s] == name for s in sources]) for name in ("fit", "tune", "test")}
    # Learn scale ONLY from fit sources. Keep identity at zero (no intercept).
    weights = source_weights(sources[masks["fit"]])
    scale = np.sqrt(np.sum(x[masks["fit"]] ** 2 * weights[:, None], axis=0))
    scale[scale == 0] = 1.0
    x = transform(x, scale)
    y = np.log1p(raw_y)
    coefficients = np.zeros((168, 5))
    trials, selected = [], {}
    for k, norm in enumerate(NORMS):
        best = None
        for penalty in (0.000001, 0.0001, 0.01, 0.1):
            c = fit_nonnegative(x[masks["fit"]], y[masks["fit"], k], sources[masks["fit"]], penalty)
            residual = x[masks["tune"]] @ c - y[masks["tune"], k]
            mse = float(np.sum(source_weights(sources[masks["tune"]]) * residual ** 2))
            trials.append(dict(norm=norm, penalty=penalty, tune_source_weighted_log_mse=mse))
            report(f"{norm} penalty={penalty:g}: tune log MSE={mse:.9g}")
            if best is None or mse < best[0]:
                best = (mse, penalty, c)
        _, selected[norm], coefficients[:, k] = best
    log_prediction = x @ coefficients
    prediction = np.expm1(log_prediction)
    if not np.isfinite(prediction).all():
        raise ValueError("nonfinite model prediction")
    metrics = {}
    for split, mask in masks.items():
        weights = source_weights(sources[mask])
        metrics[split] = dict(n_pairs=int(mask.sum()), n_sources=len(set(sources[mask])), norms={})
        for k, norm in enumerate(NORMS):
            target, estimate = raw_y[mask, k], prediction[mask, k]
            rank = (float(spearmanr(target, estimate).statistic)
                    if np.ptp(target) > 0 and np.ptp(estimate) > 0 else None)
            metrics[split]["norms"][norm] = dict(
                signed_teacher_srocc=rank,
                source_weighted_log_rmse=float(np.sqrt(np.sum(weights * (log_prediction[mask, k] - y[mask, k]) ** 2))),
                source_weighted_raw_rmse=float(np.sqrt(np.sum(weights * (estimate - target) ** 2))))
    model = args.output / "model.tsv"
    with model.open("x") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["feature", "scale"] + NORMS)
        for name, s, c in zip(EDGE_COLUMNS, scale, coefficients):
            writer.writerow([name, format(s, ".17e")] + [format(v, ".17e") for v in c])
    with (args.output / "predictions.tsv").open("x") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source", "split", "reference_sha256", "encoded_sha256"]
                        + [f"teacher_{n}" for n in NORMS] + [f"student_{n}" for n in NORMS])
        for row, truth, pred in zip(cells, raw_y, prediction):
            writer.writerow([row["source"], splits[row["source"]], row["reference_sha256"], row["encoded_sha256"]]
                            + truth.tolist() + pred.tolist())
    provenance = dict(build_commit=args.build_commit, input_manifest_sha256=digest(args.extraction / "_MANIFEST.json"),
                      input_cells_sha256=manifest["cells_sha256"],
                      input_features_sha256=feature_manifest["features_sha256"],
                      feature_manifest_sha256=digest(feature_dir / "_MANIFEST.json"),
                      feature_provenance=feature_manifest.get("feature_source", feature_manifest.get("feature_profile")),
                      feature_binaries=feature_manifest["binaries"],
                      splits_sha256=digest(args.splits), splits=splits,
                      numpy=np.__version__, scipy=scipy.__version__, python=sys.version,
                      model_sha256=digest(model), chosen_penalties=selected, trials=trials, metrics=metrics,
                      formula="expm1(sum_i weight_i * log1p(feature_i / scale_i))); no intercept",
                      status="feasibility-only; not a validated or released Margarine score",
                      limitations=["coverage is inherited from the input experiment, not certified by this fit",
                                   "source-ID partitions do not establish a cross-corpus near-duplicate audit",
                                   "no human-quality or encoder-choice acceptance result"])
    (args.output / "_MANIFEST.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    report("Feasibility fit persisted; test sources were used only for the final report")


if __name__ == "__main__":
    main()
