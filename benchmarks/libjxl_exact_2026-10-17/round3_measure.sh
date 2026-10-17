#!/usr/bin/env bash
set -euo pipefail
root="$HOME/tmp/bu-libjxl-data"
mkdir -p "$root/round3-after"
cargo build --release -p butteraugli-cli --example libjxl_parity
cp "$CARGO_TARGET_DIR/release/examples/libjxl_parity" "$root/round3-after/exact-runner"
python3 benchmarks/libjxl_exact_2026-10-17/measure.py "$root/round3-after/exact-runner" \
    --baseline="$root/round2-final/origin-main-runner" \
    --exact-baseline-dir="$root/round3-before" --output-dir=round3-after --parity-only
python3 benchmarks/libjxl_exact_2026-10-17/round3_zenbench.py
python3 benchmarks/libjxl_exact_2026-10-17/oracles.py "$root/round3-after/oracles.tsv"
