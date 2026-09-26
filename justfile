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

# Requires an interpreter with experiments/margarine/requirements-training.txt.
margarine-fit-check python:
    cd experiments/margarine && OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 nice -n 19 "{{python}}" -m unittest fit_probe_checks

# Fresh-process memory plus interleaved timing; caller sets RAYON_NUM_THREADS.
margarine-resources crops output commit:
    nice -n 19 python3 experiments/margarine/resource_sweep.py "{{crops}}" experiments/margarine/target/release/margarine-box3 "{{output}}" --build-commit "{{commit}}"

# Broad-blur approximation; shared arithmetic and strided seam contracts.
margarine-multirate-check:
    cargo fmt --manifest-path experiments/margarine/Cargo.toml -p margarine-lab --check
    nice -n 19 cargo test --manifest-path experiments/margarine/Cargo.toml --features multirate -j 2
    nice -n 19 cargo clippy --manifest-path experiments/margarine/Cargo.toml --features multirate --all-targets -j 2 -- -D warnings

margarine-compact-check:
    cargo fmt --manifest-path experiments/margarine/Cargo.toml -p margarine-lab --check
    nice -n 19 cargo test --manifest-path experiments/margarine/Cargo.toml --features compact -j 2
    nice -n 19 cargo clippy --manifest-path experiments/margarine/Cargo.toml --features compact --all-targets -j 2 -- -D warnings

# Wrap these recipes with run-heavy on Linux; one heavy command at a time.
margarine-candidate-check features:
    cargo fmt --manifest-path experiments/margarine/Cargo.toml -p margarine-lab --check
    nice -n 19 cargo test --manifest-path experiments/margarine/Cargo.toml --features "{{features}}" -j 2
    nice -n 19 cargo clippy --manifest-path experiments/margarine/Cargo.toml --features "{{features}}" --all-targets -j 2 -- -D warnings
    nice -n 19 cargo build --release --manifest-path experiments/margarine/Cargo.toml --features "{{features}}" --bin margarine-box3 -j 2

margarine-direct-eval pairs binaries output candidate teacher commit ingress="aic-rgb8":
    nice -n 19 python3 experiments/margarine/score_manifest.py "{{pairs}}" "{{binaries}}" "{{output}}" --candidate "{{candidate}}" --teacher "{{teacher}}" --build-commit "{{commit}}" --ingress "{{ingress}}"

margarine-direct-resources crops binary output candidate commit rows="128":
    nice -n 19 python3 experiments/margarine/resource_sweep.py "{{crops}}" "{{binary}}" "{{output}}" --direct "{{candidate}}" --strip-rows "{{rows}}" --build-commit "{{commit}}"

# A single-size diagnostic cannot fit fixed overhead or qualify the size curve.
margarine-direct-timing binary reference distorted output rows="128":
    nice -n 19 "{{binary}}" --bench-direct "{{rows}}" "{{reference}}" "{{distorted}}" "{{output}}"

margarine-direct-profile binary reference distorted output commit rows="128":
    mkdir "{{output}}"
    printf '%s\n' "{{commit}}" > "{{output}}/build_commit.txt"
    shasum -a 256 "{{binary}}" "{{reference}}" "{{distorted}}" > "{{output}}/inputs.sha256"
    nice -n 19 valgrind --tool=callgrind --callgrind-out-file="{{output}}/callgrind.out" "{{binary}}" --memory-native "{{rows}}" "{{reference}}" "{{distorted}}" > "{{output}}/run.log" 2>&1
    callgrind_annotate --inclusive=no --threshold=99 "{{output}}/callgrind.out" > "{{output}}/flat.txt"

# Point-label diagnostics; participant significance remains a separate gate.
margarine-disagreements ledger output candidate teacher_maps candidate_maps commit:
    nice -n 19 python3 experiments/margarine/disagreements.py "{{ledger}}" "{{output}}" --candidate "{{candidate}}" --teacher-maps "{{teacher_maps}}" --candidate-maps "{{candidate_maps}}" --build-commit "{{commit}}"

margarine-kadid-opinions raw dmos output commit:
    nice -n 19 python3 experiments/margarine/prepare_kadid_opinions.py "{{raw}}" "{{dmos}}" "{{output}}" --build-commit "{{commit}}"

margarine-panels scored evaluator output candidate commit:
    nice -n 19 python3 experiments/margarine/evaluate_manifest.py "{{scored}}" "{{evaluator}}" "{{output}}" --candidate "{{candidate}}" --build-commit "{{commit}}"

margarine-participant-pairs scored opinions evaluator output candidate commit draws="2000" seed="20260926":
    nice -n 19 python3 experiments/margarine/participant_pairs.py "{{scored}}" "{{opinions}}" "{{evaluator}}" "{{output}}" --candidate "{{candidate}}" --build-commit "{{commit}}" --draws "{{draws}}" --seed "{{seed}}"
