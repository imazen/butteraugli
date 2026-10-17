# butteraugli development recipes

# Format + regenerate the public-API surface snapshots (docs/public-api/).
# The snapshot runner lives in the workspace-excluded apidoc/ package, so it
# is never built or run by plain `cargo test` or any CI job.
fmt:
    cargo fmt --all
    cargo test --manifest-path apidoc/Cargo.toml

# Regenerate the public-API surface snapshots only
api-doc:
    cargo test --manifest-path apidoc/Cargo.toml

# Verify the committed snapshots are current
api-doc-check:
    ZEN_API_DOC=check cargo test --manifest-path apidoc/Cargo.toml

# Rust-only gates explicitly disable the optional C++ benchmark adapter.
audit-lint:
    cargo fmt -p butteraugli -p butteraugli-cli -p butteraugli-bench --check
    BUTTERAUGLI_BENCH_NO_CPP=1 cargo clippy --workspace --all-targets -- -D warnings

audit-test:
    BUTTERAUGLI_BENCH_NO_CPP=1 cargo test -p butteraugli -p butteraugli-cli --all-targets
    cargo test -p butteraugli --doc
    cargo test -p butteraugli --features linear-planes
    cargo test -p butteraugli --features iir-blur --lib

audit-semver features="--default-features":
    RUSTUP_TOOLCHAIN=1.98.1 cargo semver-checks check-release -p butteraugli --baseline-version 0.9.3 --release-type patch {{features}}

audit-build:
    cargo build --release -p butteraugli-bench --example version_audit

audit-golden:
    RAYON_NUM_THREADS=1 target/release/examples/version_audit --golden

audit-perf image rounds="15" +args="":
    target/release/examples/version_audit {{quote(image)}} {{rounds}} {{args}}

audit-scores image:
    target/release/examples/version_audit {{quote(image)}} 3 --scores-only

audit-msrv:
    cargo +1.89.0 check -p butteraugli -p butteraugli-cli --locked

audit-extract log output:
    python3 butteraugli-bench/scripts/extract_version_audit.py {{quote(log)}} {{quote(output)}}

# Run exactly one Rust version per process for external RSS/allocation profiling.
audit-profile image version size="1920x1080" threads="8" calls="20" +args="":
    target/release/examples/version_audit {{quote(image)}} --profile={{version}} --size={{size}} --threads={{threads}} --calls={{calls}} {{args}}

# Summarize complete paired cache runs and independent RSS profiles.
audit-cache-summary logs output +args="":
    python3 butteraugli-bench/scripts/cache_budget_summary.py {{quote(logs)}} {{quote(output)}} {{args}}

# Cache parity in safe and existing opt-in allocation modes.
audit-cache-test:
    cargo test -p butteraugli --test scratch_cache --features linear-planes
    cargo test -p butteraugli --test scratch_cache --features linear-planes,unsafe-performance

# Investigation runner writes raw little-endian f32 diffmaps (top row first).
libjxl-parity-build:
    cargo build --release -p butteraugli-cli --example libjxl_parity

libjxl-parity-run reference distorted diffmap +args="":
    RAYON_NUM_THREADS=1 target/release/examples/libjxl_parity {{quote(reference)}} {{quote(distorted)}} {{quote(diffmap)}} {{args}}

libjxl-parity-measure artifacts label binary:
    python3 benchmarks/libjxl_parity_2026-10-17/measure.py {{quote(artifacts)}} {{quote(label)}} {{quote(binary)}}

libjxl-parity-generate artifacts inputs:
    python3 benchmarks/libjxl_parity_2026-10-17/generate.py {{quote(artifacts)}} {{quote(inputs)}}

libjxl-parity-trace:
    python3 benchmarks/libjxl_parity_2026-10-17/trace.py

libjxl-parity-ablations:
    python3 benchmarks/libjxl_parity_2026-10-17/ablations.py

libjxl-parity-google-build output:
    python3 benchmarks/libjxl_parity_2026-10-17/google_old.py {{quote(output)}}

# Run these under the host's resource supervisor; raw artifacts stay off git.
libjxl-exact-measure binary:
    python3 benchmarks/libjxl_exact_2026-10-17/measure.py {{quote(binary)}}

libjxl-exact-oracles output:
    python3 benchmarks/libjxl_exact_2026-10-17/oracles.py {{quote(output)}}

libjxl-exact-test:
    cargo test -p butteraugli --test exact_numerics -- --nocapture
