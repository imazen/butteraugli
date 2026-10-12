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
audit-profile image version size="1920x1080" threads="8" calls="20":
    target/release/examples/version_audit {{quote(image)}} --profile={{version}} --size={{size}} --threads={{threads}} --calls={{calls}}
