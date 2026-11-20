# Completion continuity and completion-priority experiment

Status: FAILED_NUMERICAL_GUARD_PRESERVED. Three baseline cells complete; SCORE_ONLY
stopped at time900. Raw failed attempt and 15-round reproduction retained here.
Resume only in sibling joint_completion_continuity_20261026_run2, which fixes
near-integral solver output with unchanged all-row validation and reruns all7.
User authorized improving the mechanism based on
the completed six-job ECT experiment. Actual source copied, including historical
untracked implementation, from sia-joint-completion-time/sia-simulator into
sia-joint-completion-continuity/sia-simulator. Parent six-job snapshot is the
reconstruction authority; bare service-horizon Git HEAD is incomplete.

This task implements one preregistered revision and tests the SAME six jobs,
work, arrivals, CPU/RAM/GPU capacities, Medium response, seed initialization,
batch adaptation, grids, 60s interval and 6000-round bound. No parameter search.
Memory >=2 GiB and disk >=5 GiB before separate serial processes. Preserve all
historical source/results; no full160/104 workload, no push or cleanup.

New mechanism: reciprocal current completion estimate, and completion-continuity
admission with FIFO head/drain. Real checkpoint is exact. GPU changes allowed
only at arrival/completion/first-valid-model events and after productive training;
the inherited 250s startup ledger is lifetime, never reset at 300s. CPU/RAM
can adapt each round without extra delay under the unchanged execution model.

Seven registered six-job cells, in this order: SIA_P, SIA_ECT_AUX, LEGACY_ECT,
SCORE_ONLY, CONTINUITY_ONLY, JOINT_CC_P, JOINT_CC. The middle three isolate old
goal/old service versus individual changes. Joint-CC-P disables only auxiliary
choice relative to Joint-CC. Stop after the same-input results and audit,
including if performance worsens. No outcome-driven second revision.

Recovery: DESIGN.md, red/green logs, preregistration.json, execution_manifest.jsonl,
cells/, independent_audit.json and REPORT.md. A nonempty cell is never reused.
SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE.
