# Current authorized task: CC-Joint code cleanup

Status: COMPLETE_VALIDATED. No pending cleanup or experiment execution.
Canonical code: D:/科研启航/Sia/cc-joint.
Final evidence: final_validation.json (valid=true), relocation_validation.json,
CLEANUP_REPORT.md. Successful final reporter: final_report.py.
Before/after tests: 212/212 each; eight frozen-vs-clean bounded probe pairs exact.
All 27 legacy trees archived recoverably; 19,693 working-file hashes and 22 Git
states preserved. Do not repeat relocation scripts or remove the recovery backup.
No algorithm changes, performance experiments, Git commit or push in this task.

User: “梳理目前代码，只保留目前CC-joint实现，其他无关内容可以清除”。
No new performance experiments or algorithm changes are authorized/needed.

Canonical development root: D:/科研启航/Sia/cc-joint.
Validated source authority: cc_joint_sustained_cpu_20job_20261027_run1/source,
404-file fingerprint21CA3574775F3A10030473BD22183802CC5DED06D2ED57EDE35F3D1A332F5A57.
Dependency inventory selected retained engine/tests/data, unchanged bytes.
New front door/factory/metric extraction only; no target, capacity, rho, queue,
checkpoint, work or scheduling change. Keep212regressions and four comparisons.

Inventory:27old top-level sia-* trees,0.731GiB; five plain snapshots,22Git trees
(17linked+one main joint repo,4linked to official). All dirty content must survive.
Archive, do NOT permanently delete, to _archive/legacy_code_20261027/.
Keep official-artifacts/research_sources/Synergy working files and HEAD unchanged;
Git worktree path registrations necessarily update when their checkouts move.
Do not touch research_runs/research_notes, environments, unknown tmp or user files.

Sequence: checks.py before -> backup_legacy.py (full byte archive/readback,
tracked/staged patches, Git bundles) -> relocate.ps1 dryrun -> relocate.ps1 -Apply
-> verify_relocation.py -> checks.py after -> new entry constructor check -> report.
No reset/clean/remove/worktree-remove/push. Explicit path boundary verification
before moves; refuse destination collision. Git worktree move + main repo repair,
not raw moves of linked worktrees. Archive/source/branch/hash verification after.

Before-check session26215:212tests and bounded official Sia, plus8pairs of
two-job/five-round frozen-vs-clean trajectory probes. Consult before/summary.json
for completion; never count these as performance runs. No20job simulation.

Only claim complete after post-archive212tests pass, Git states/file hashes
preserved, old reports/results intact and recovery instructions available.
Use backup_manifest.json/relocation_plan.json/relocation_journal.jsonl to recover
if interrupted. Never repeat moves blindly or overwrite an archive destination.
