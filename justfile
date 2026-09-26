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

# The unpublished Margarine research instruments; does not format sibling crates.
margarine-check:
    python3 -m unittest discover -s experiments/margarine -p 'test_*.py'
    cargo fmt --manifest-path experiments/margarine/Cargo.toml -p margarine-lab --check
    nice -n 19 cargo test --manifest-path experiments/margarine/Cargo.toml -j 2
    nice -n 19 cargo clippy --manifest-path experiments/margarine/Cargo.toml --all-targets -j 2 -- -D warnings

# Caller may wrap this in run-heavy on Linux; use a fresh output directory.
margarine-bootstrap scores output draws="2000" seed="20260926":
    nice -n 19 experiments/margarine/target/release/margarine-eval --bootstrap-all "{{scores}}" "{{output}}" "{{draws}}" "{{seed}}"
