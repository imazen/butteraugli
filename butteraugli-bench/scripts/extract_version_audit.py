#!/usr/bin/env python3
"""Extract zenbench CSV or a Markdown comparison table from a complete audit log."""

import csv
import io
import re
import sys
from pathlib import Path


def extract(log: Path) -> str:
    lines = log.read_text().splitlines()
    header = next(line for line in lines if line.startswith("group,benchmark,"))
    rows = [line for line in lines if re.match(r"^\d+x\d+_[18]t_[a-z-]+,", line)]
    assert rows, "no completed benchmark rows in log"
    return header + "\n" + "\n".join(rows) + "\n"


def main() -> None:
    log, output = map(Path, sys.argv[1:])
    data = extract(log)
    if output.suffix == ".md":
        rows = list(csv.DictReader(io.StringIO(data)))
        baseline = {r["group"]: r for r in rows if r["benchmark"] == "0.9.3"}
        table = ["| Case | 0.9.3 ms | Current ms | Change | Paired 95% CI |",
                 "| --- | ---: | ---: | ---: | --- |"]
        for row in rows:
            if row["benchmark"] != "main":
                continue
            old = float(baseline[row["group"]]["mean_ns"]) / 1e6
            new = float(row["mean_ns"]) / 1e6
            pct, lo, hi = (float(row[k]) for k in
                          ("vs_base_pct", "vs_base_ci_lo_pct", "vs_base_ci_hi_pct"))
            table.append(f"| {row['group']} | {old:.2f} | {new:.2f} | {pct:+.2f}% | [{lo:+.2f}%, {hi:+.2f}%] |")
        data = "\n".join(table) + "\n"
    assert len(data.encode()) <= 30_000, "generated output needs external storage"
    output.write_text(data)


if __name__ == "__main__":
    main()
