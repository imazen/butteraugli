#!/usr/bin/env bash
set -euo pipefail
export BUTTERAUGLI_BENCH_NO_CPP=1 CARGO_INCREMENTAL=0
cargo fmt --check
cargo test
cargo clippy --all-targets -- -D warnings
for features in linear-planes internals unsafe-performance iir-blur linear-planes,iir-blur linear-planes,internals,unsafe-performance; do
    cargo test -p butteraugli --features "$features"
    cargo clippy -p butteraugli --lib --tests --features "$features" -- -D warnings
    cargo doc -p butteraugli --no-deps --features "$features"
done
cargo test -p butteraugli --no-default-features
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER='qemu-aarch64 -L /usr/aarch64-linux-gnu' \
cargo test -p butteraugli --target aarch64-unknown-linux-gnu --features linear-planes --test exact_numerics -- --nocapture
