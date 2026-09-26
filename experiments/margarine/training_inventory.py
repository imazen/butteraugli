#!/usr/bin/env python3
"""Read metadata axes from existing teacher-training candidates, without fitting.

Uses column projection and batches; feature matrices and encoded bytes are not
loaded. This inventory does not verify whole-file hashes or teacher parity.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time


def main():
    import pyarrow.parquet as pq
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--build-commit", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "progress.log").open("x", buffering=1) as log:
        def report(text):
            text = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {text}"
            print(text, flush=True)
            print(text, file=log, flush=True)
        results = []
        for path in args.inputs:
            parquet = pq.ParquetFile(path)
            names = parquet.schema_arrow.names
            columns = [n for n in ("origin_id", "source_id", "width", "height", "q", "content_class", "size_class", "mode", "codec", "dist_name", "severity_level") if n in names]
            origins = defaultdict(set)
            class_origins = defaultdict(set)
            counts = {n: Counter() for n in columns if n not in ("origin_id", "source_id", "width", "height")}
            geometries = Counter()
            count = 0
            for batch in parquet.iter_batches(batch_size=16384, columns=columns):
                for row in batch.to_pylist():
                    origin = row.get("origin_id", row.get("source_id"))
                    if "width" in row and "height" in row:
                        shape = (row["width"], row["height"])
                        geometries[shape] += 1
                        origins[origin].add(shape)
                    else:
                        origins[origin]
                    class_origins[row.get("content_class", "unavailable")].add(origin)
                    for key, counter in counts.items(): counter[str(row[key])] += 1
                count += batch.num_rows
                report(f"{path.name}: {count}/{parquet.metadata.num_rows} metadata rows")
            result = dict(path=str(path.resolve()), bytes=path.stat().st_size, rows=count,
                source_count=len(origins), source_columns=[n for n in ("origin_id", "source_id") if n in names],
                feature_count=sum(n.startswith("feat_") or (n.startswith("f") and n[1:].isdigit()) for n in names),
                teacher_columns=[n for n in names if "butteraugli" in n],
                geometry_counts={str(k):v for k,v in sorted(geometries.items())},
                sizes_per_source={str(k):len(v) for k,v in origins.items()},
                class_source_counts={str(k):len(v) for k,v in class_origins.items()},
                axes={k:dict(v) for k,v in counts.items()},
                whole_file_hash_verified=False, teacher_parity_verified=False)
            results.append(result)
        (args.output / "inventory.json").write_text(json.dumps(dict(build_commit=args.build_commit, files=results), indent=2)+"\n")
        report("Complete; no training or acceptance claims")


if __name__ == "__main__":
    main()
