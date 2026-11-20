# Reproduction and scope

This is a candidate-only version copied from the sealed42-cell source. It is
not a new fairness/JCT objective and does not change R0/E-legacy, alpha, phi,
the workload, arrival sequence, capacity, service locks or startup costs.

Source authority: preregistration.json source_hashes, source_snapshot.zip,
SOURCE_CHANGES.patch, SOURCE_VERSION.json. Both source directories are source
copies without Git metadata; no commit hash is fabricated.

Canonical source used for legacy path-sensitive tests: cc-joint-breakpoint-study.
Runtime source: this directory/source. Both are byte-identical. New candidate
rule and adapter: source/breakpoint_study.py; generic default-off generation and
precommit hook: source/sia-simulator/joint_prototype/joint_scheduler.py.

Original233 tests (229old+4candidate) passed. Red missing-knot test, old run
failure and statistical-adapter fault replay are kept. small_attempt2 is the
complete8-cell gate evidence; first attempt is diagnostic, never mixed in.

Python: workspace .venv-sia-baseline/Scripts/python.exe -X utf8 -B.
Dependencies are pinned in source/dependencies.lock.txt and preregistration.
Do not blindly rerun in this output folder: all scripts refuse overwrites.
To reproduce independently create a new output root and update only output
locations in maintenance wrappers, retain all scientific settings and input
hashes, freeze it and recheck tests before executing.

Existing running batch entry: maintenance/run_breakpoint30_20261029.py batch.
Before resuming inspect CURRENT_TASK.md/execution_manifest.jsonl, verify no
duplicate runner, completed output hashes and nonempty partial directories.
Never overwrite partial or historical results. Resource gates2GiB/5GiB.

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE.
