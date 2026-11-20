# Completion continuity and completion-priority experiment

Status: ALL_SEVEN_COMPLETE_AND_INDEPENDENTLY_VALIDATED. 212 tests pass.
Batch session46573 exited successfully; do not restart any experiment.
All seven cells completed6/6. Independent raw event/segment audit passed.
CC mean4775.5 versus Sia5636.5 (15.28% lower), oldECT6832.5 (30.11% lower).
CC-P mean4476: auxiliary optimization raises mean6.69%, so incremental CPU JCT
benefit is NOT established. Three jobs improve and three worsen versus Sia;
CF2 dominates net benefit. Preserve this negative ablation result.
REPORT.md and mechanism_diagnostics.json contain the scope and local objective
counterexample; no outcome-driven further algorithm revision or larger run.
The exact16-round numerical repair reproduction crossed time900 safely.
Frozen source fingerprint21CA3574775F3A10030473BD22183802CC5DED06D2ED57EDE35F3D1A332F5A57.
Previous attempt preserved in sibling run1.
See NUMERICAL_REPAIR.md; no scientific parameter change. This attempt reruns the
complete7-cell matrix under one fixed repaired version after tests.
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
