# CC-Joint 12-job load expansion

Status: EIGHT_COMPLETE_AND_INDEPENDENTLY_VALIDATED. No pending training.
All8 cells complete12/12. Sessions90386/96296/19976 have exited. Do not restart.
Neutral: SIA_P15375.75, SIA_AUX15375.75, CC_P10606.4167, CC10606.4167.
Medium: SIA_P12376.75, SIA_AUX10975.75, CC_P7050.6667, CC7073.50 seconds mean JCT.
CC improves Sia42.85% and serial35.55%, but worsens its own quota ablation0.32%.
212 tests pass; original bounded Sia exact; raw audit capacity violations0.
Neutral pairs have exact GPU/progress/completion/restart trajectories.
Medium6 improve/6worsen vsSia; without largest beneficiary still improves35.37%.
CPU quota benefit beyond proportional remains unproven. No algorithm changes.
REPORT.md, run_manifest.json, independent_audit.json, comparison_summary.json,
per_job.csv, mechanism_analysis.json and exchange_composition.json complete.
Packaging verification/delivery is final; stop after delivery, no new experiments.

Historical execution and recovery notes follow. Source404 hashes exactly preserved;
Execution session90386 exited0 after safety stop; no active owned training child.
Neutral4 and Medium SIA_P/SIA_ECT_AUX complete12/12; Medium CC_P/CC never started.
Start-time gate for pending Medium CC_P saw1.8599GiB; next sample1.479GiB.
No lowered gate or partial result overwritten. Read safety_stop_snapshot.json,
which freezes completed-prefix hashes and proves pending directories absent.
Memory recovered to2.341GiB; partial raw audit passed for6 complete cells.
resume.py resume has now been dispatched ONCE; do not duplicate it.
Read resume_started.json and resume_execution_manifest.jsonl for suffix status.
Resume session96296 completedMedium CC_P then exited0 at a second memory gate.
The first resume_last.py invocation stopped at1.8207GiB before taking its marker;
after recovery the same helper started ONLY Medium JOINT_CC. Active session19976.
Read last_cell_started.json/last_execution_manifest.jsonl. Never duplicate it.
Final wall-time/manifests must combine original, resume and last manifests,
preserving all safety-stop records. Once last cell completes run frozen audit.py,
mechanism_analysis.py, report.py and package.py; deliver once and stop.
Recovery only when memory>=2GiB: experiment source/input hashes must pass,
then resume.py resume runs just the two frozen suffix cells with an exclusive
resume_started marker and separate resume_execution_manifest.jsonl.
Never call experiment.py batch again. Do not start duplicate suffix processes.
audit_available.py writes checks for6 available cells under audit_at_memory_stop/.
212 tests pass (tests_relocated), 3-round official Sia regression exact,
48 configuration rows covered, fully supported CPU exchange witnesses exist.
Do not duplicate startup. Read execution_manifest.jsonl and current cell log.

User approved the preceding fixed plan: retain the old JOINT_CC (15.28% mean
JCT benefit on six jobs), name it CC-Joint, expand only workload size to12.
Do not import the newer net-completion objective. No algorithm changes/tuning.

Authority: joint_completion_continuity_20261026_run2/preregistration.json and
its actual404-file source snapshot. Verify every byte before copying source/.
Parent source fingerprint21CA3574775F3A10030473BD22183802CC5DED06D2ED57EDE35F3D1A332F5A57.
Old source, official inputs, all historical outputs must remain unchanged.

Twelve original workload-1 rows: CF0,1,2,3,4,5; DS8,9,12,16,21,30.
Preserve GPU demand/batch/work; simultaneous t=0 synthetic controlled arrival,
source rows and original arrivals retained. Same AWS/DGX each2GPU12CPU250GB.
Same R0/E-legacy, alpha0/0.5, quota grid including inheritedCPU8, profile seed1,8.
Fixed runs: Neutral then Medium; each SIA_P, SIA_ECT_AUX, JOINT_CC_P, JOINT_CC.
Four methods are Sia-proportional, fixed-Sia-intent auxiliary optimization,
CC-Joint proportional ablation, and full CC-Joint. 8 independent serial children.
Interval60, max6000 rounds; complete early or report censor, no extension.
Before each training process memory>=2GiB and disk>=5GiB; no duplicate startup.

Recovery: experiment.py prepare/preflight/test/regression/freeze/validate/batch;
preregistration.json; execution_manifest.jsonl; cells/<response>/<method>.
Only start batch after coverage/preflight, unchanged tests and bounded original
Sia regression pass. On implementation failure preserve evidence and stop.
Do not expand beyond this12-job/8-cell task, restore104/160 or tune to win.
Audit raw completion/progress/time/resources and report all jobs, negative
auxiliary effects, clipping, waiting, tails and long-job concentration.
Final report REPORT.md, machine summaries/per_job.csv, reconstructable archive.
Stop after delivery. SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE.
