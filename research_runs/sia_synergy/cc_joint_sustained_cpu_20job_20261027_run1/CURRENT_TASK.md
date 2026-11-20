# CC-Joint sustained CPU-sensitive 20-job experiment

Status: ALL_EIGHT_COMPLETE_AND_INDEPENDENTLY_VALIDATED; REPORT_READY.
Unified session86092 and analysis43424 exited0. No training remains. All8
completed20/20,212 unchanged tests pass, bounded official Sia exact, capacity
violations0. Medium means SIA_P7357.45, SIA_AUX7491.45, CC_P4313.45, CC3984.25.
CC vsSia45.85%, vsserial46.82%, quota increment7.63%;16 improve/4worsen.
Compared CC-P: P95 worsens19.90%, P9917.97%, makespan17.60%, GPU-h9.73%.
153 mixed CF/DS node-rounds,136 CF-from-DS exchanges; all selected Medium CC
numerator/denominator coordinates supported. CPU quota benefit observed in this
synthetic cohort, RAM-specific benefit unproven. Read REPORT.md, interpretation.json,
all raw/analysis manifests. Never rerun training or extend this task.
Final package.py may run once only if delivery_manifest.json is absent; once
readback verification exists, delivery is complete and stop. Do not resimulate.
The
constructor session47476 has completed. Preflight,212tests, bounded Sia and
input acceptance all passed. Preregistration is frozen. Batch began once after
acceptance; a prior pre-acceptance command failed without starting training,
documented separately. Keep all frozen source/scripts unchanged.

Authorization: user approved the fixed proposed scenario, then said “好的，开始实验！”.
Retain old CC-Joint, not the rejected net-completion variant. No algorithm changes.

Source authority: cc_joint_12job_20261026_run1/preregistration.json, 404 actual
source files, fingerprint 21CA3574775F3A10030473BD22183802CC5DED06D2ED57EDE35F3D1A332F5A57.
Copy byte-exact source into source/ and project-root sia-joint-cc-sustained20/sia-simulator.
Bare Git HEAD is not a sufficient rebuild. Preserve all parent/official files.

Workload: first18 CIFAR10 and first2 DeepSpeech2 records of official workload-1,
original order within each application. No duplicated/shortened work; batch and
GPU demand retained. Controlled arrivals: t0 CF0,DS8,CF1,DS9 in that stable order;
remaining CIFAR pairs every300s, through2400s. Original arrivals/source row kept
separately; source_row_index represents the explicitly controlled arrival order.

Hardware unchanged: AWS and DGX-ext one2GPU node each,12 synthetic CPU units,
250GB RAM per node. Same R0/E-legacy, alpha0/.5, CPU grid including inherited8,
RAM grid, mapping, phi, seed_profiles(1,8), batch adaptation and 250s CC budget.
Four strategies: SIA_P, SIA_ECT_AUX, JOINT_CC_P, JOINT_CC. Neutral then Medium.
Eight serial independent processes,60s interval,6000rounds; censor/no imputation.
Before every process: memory>=2GiB, disk>=5GiB. Never duplicate or overwrite.

Gate BEFORE formal training: policy-independent reference training-time share
of CIFAR40–60%; supported budget-feasible CF8/DS4 versus6/6 exchange including
donor loss; GPU/CPU coupling and analytically positive complete-pair opportunity
after actual startup delay. Reference estimates do not change online information.
If gate fails, preserve preflight and report, do not select another load by JCT.
Then unchanged212tests, bounded original Sia regression, freeze inputs/scripts/
dependencies/eight cells, constructor checks, batch. Resume only unstarted suffix
after verifying completed prefix hashes and absence of duplicate processes.

Recovery entry: experiment.py prepare/relocate/preflight/test/regression/freeze/
validate/batch; prepared.json, coverage_preflight.json, preregistration.json,
execution_manifest.jsonl, cells/<response>/<method>. Record every safety stop.
Audit from raw events independently, including time partition, rho once, clipping,
capacity, locks, initial/post-start waiting, receiver and donor effects, all jobs.
Deliver REPORT.md, summaries, per-job data and verified reconstructable archive.
No 104/160, other scenarios or tuning. Stop after delivery or failed opportunity gate.

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE
