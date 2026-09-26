#!/usr/bin/env python3
"""Compare choices at every observed byte budget within each source.

This is a diagnostic regret curve, not the unresolved material-reversal gate.
Observed budgets weight densely sampled ladders more heavily; report source
means alongside pooled counts. No interpolation or additional encodes occur.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

NORMS = ("max", "p1", "p2", "p3", "p6")


def choices(rows, candidate, norm, human=False):
    if human:
        directions = {row["direction"] for row in rows}
        if len(directions) != 1 or not directions <= {"quality", "distortion"}:
            raise ValueError("human labels need one declared direction per source")
        if any(not math.isfinite(float(row["target"])) for row in rows):
            raise ValueError("non-finite human label")
    for row in rows:
        rate = float(row["bpp"])
        values = (row["scores"][arm][norm] for arm in ("teacher", candidate))
        if not math.isfinite(rate) or rate <= 0 or any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("invalid rate or metric value")
    for budget in sorted({float(row["bpp"]) for row in rows}):
        eligible = [row for row in rows if float(row["bpp"]) <= budget]
        # Stable score ties prefer fewer bytes, then the declared pair ID.
        def pick(arm):
            return min(eligible, key=lambda row: (row["scores"][arm][norm], float(row["bpp"]), row["pair"]))
        teacher, student = pick("teacher"), pick(candidate)
        optimum = teacher["scores"]["teacher"][norm]
        achieved = student["scores"]["teacher"][norm]
        absolute = achieved - optimum
        relative = absolute / optimum if optimum else (0.0 if absolute == 0 else math.inf)
        result = dict(budget_bpp=budget, eligible=len(eligible), teacher_pair=teacher["pair"],
                   candidate_pair=student["pair"], teacher_codec=teacher["codec"],
                   candidate_codec=student["codec"], teacher_optimum=optimum,
                   teacher_at_candidate=achieved, absolute_regret=absolute,
                   relative_regret=relative, different_pair=int(teacher["pair"] != student["pair"]))
        if human:
            t, s = float(teacher["target"]), float(student["target"])
            result.update(teacher_human_target=t, candidate_human_target=s,
                          human_quality_loss=(t-s) if teacher["direction"] == "quality" else (s-t))
        yield result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate", default="box3")
    parser.add_argument("--build-commit", required=True)
    parser.add_argument("--human-loss-threshold", type=float, action="append", default=[],
                        help="diagnostic loss in native label units; requires complete human labels")
    args = parser.parse_args()
    if any(not math.isfinite(x) or x < 0 for x in args.human_loss_threshold):
        parser.error("human-loss thresholds must be finite and nonnegative")
    groups = defaultdict(list)
    seen = set()
    for line in args.ledger.open():
        row = json.loads(line)
        key = row["dataset"], row["source"], row["pair"]
        if key in seen:
            raise ValueError(f"duplicate pair {key}")
        seen.add(key)
        groups[key[:2]].append(row)
    if not groups:
        raise ValueError("empty ledger")
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    human_summaries = []
    with (args.output / "progress.log").open("x", buffering=1) as log:
        for norm in NORMS:
            records = []
            for (dataset, source), rows in sorted(groups.items()):
                records.extend(dict(dataset=dataset, source=source, norm=norm, **r)
                               for r in choices(rows, args.candidate, norm, bool(args.human_loss_threshold)))
                print(f"{norm} {dataset} {source}: choices persisted below", file=log, flush=True)
            with (args.output / f"choices-{norm}.tsv").open("x") as out:
                writer = csv.DictWriter(out, delimiter="\t", fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
            for dataset in sorted({r["dataset"] for r in records}):
                dataset_rows = [r for r in records if r["dataset"] == dataset]
                for threshold in (0.0, 0.001, 0.01, 0.05, 0.1):
                    per_source = defaultdict(list)
                    for row in dataset_rows:
                        per_source[row["source"]].append(row["relative_regret"] > threshold)
                    count = sum(sum(values) for values in per_source.values())
                    summaries.append(dict(dataset=dataset, norm=norm,
                        diagnostic_relative_regret_threshold=threshold, sources=len(per_source),
                        budgets=len(dataset_rows), exceeding_budgets=count,
                        pooled_exceedance_rate=count/len(dataset_rows),
                        source_mean_exceedance_rate=sum(sum(v)/len(v) for v in per_source.values())/len(per_source),
                        maximum_absolute_regret=max(r["absolute_regret"] for r in dataset_rows)))
                for threshold in args.human_loss_threshold:
                    per_source = defaultdict(list)
                    for row in dataset_rows:
                        per_source[row["source"]].append(row["human_quality_loss"] > threshold)
                    count = sum(sum(values) for values in per_source.values())
                    human_summaries.append(dict(dataset=dataset, norm=norm,
                        diagnostic_native_label_loss_threshold=threshold, sources=len(per_source),
                        budgets=len(dataset_rows), exceeding_budgets=count,
                        pooled_exceedance_rate=count/len(dataset_rows),
                        source_mean_exceedance_rate=sum(sum(v)/len(v) for v in per_source.values())/len(per_source),
                        mean_signed_quality_loss=sum(r["human_quality_loss"] for r in dataset_rows)/len(dataset_rows),
                        maximum_quality_loss=max(r["human_quality_loss"] for r in dataset_rows)))
            print(f"{norm}: {len(records)} observed-budget comparisons", flush=True)
        with (args.output / "summary.tsv").open("x") as out:
            writer = csv.DictWriter(out, delimiter="\t", fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        if human_summaries:
            with (args.output / "human-summary.tsv").open("x") as out:
                writer = csv.DictWriter(out, delimiter="\t", fieldnames=list(human_summaries[0]))
                writer.writeheader()
                writer.writerows(human_summaries)
        provenance = dict(build_commit=args.build_commit, ledger=str(args.ledger.resolve()),
            ledger_sha256=hashlib.sha256(args.ledger.read_bytes()).hexdigest(), candidate=args.candidate,
            budget_policy="all distinct observed bpp values per source, no interpolation",
            tie_policy="minimum score, then minimum bpp, then lexicographic pair ID",
            human_loss_thresholds=args.human_loss_threshold,
            human_loss_policy="signed native-label loss versus teacher-selected encode; not an uncertainty test",
            acceptance_gate="unavailable: materiality definition not agreed; thresholds are diagnostics")
        (args.output / "_MANIFEST.json").write_text(json.dumps(provenance, indent=2)+"\n")
        print("Complete; diagnostic regret curves, no acceptance verdict", file=log, flush=True)


if __name__ == "__main__":
    main()
