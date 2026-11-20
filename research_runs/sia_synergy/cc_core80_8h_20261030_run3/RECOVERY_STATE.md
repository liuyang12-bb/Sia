# Active8-hour80-job core experiment recovery

User authorized: preflight80, freeze, run11core cells, optional4if budget, preserve algorithms/history, no tuning/no shortened work, stop/report within8h. No subagents. Do not resume old104/160 experiment.

## Time and live process

- Global start2026-10-30 18:49:54UTC.
- Formal workers must stop2026-10-31 01:49:54UTC; hard report deadline02:49:54UTC.
- Budget is unchanged across run1/run2/run3; see budget.json.
- **Active supervisor unified exec session61215**, cwd D:/科研启航/Sia/cc-joint-core80-8h-v3. It runs supervise80.py with this directory's preregistration.json. Do not start another.
- Monitor batch_console.log, execution_manifest.jsonl, cell progress_*.json/console logs. Never modify frozen source.
- Last observed: SIA_P complete80/80,309rounds, meanJCT9290.875s; FULL active,3logged rounds with0completed, third round27.15sec at34658scored columns. Fourth round is computationally slow. No overall FULL JCT inference.
- Each cell has1800sec wall compute cap, original60sec/6000simulated-round/full-work semantics. Watchdog kills its OWN child tree, labels COMPUTE_TIMEOUT, retains partial logs. Process resource gates beforeeach2GiB/5GiB. Serial independent Python processes. Do not repair truncated gzip.
- **Cross-tool psutil cannot see an active process PID from another tool session.** NoSuchProcess in a separate tool is not global proof that an active worker stopped. Use unified session status/exit and artifact activity. Prior run1 session86954 and run2 session14630 were interrupted with Ctrl+C and returned exit1; their artifacts stopped growing. prior_processes_stopped.json is scoped PID evidence, not global OS inventory.

## Frozen source

- TREE D:/科研启航/Sia/cc-joint-core80-8h-v3.
- OUT D:/科研启航/Sia/research_runs/sia_synergy/cc_core80_8h_20261030_run3.
- Prereg SHA3A7372248D0948BCD651A1C814503A29899F394139C54673ECB7D61883CB582A.
-302file full snapshot zip SHA43887CBD457E216EE84BEDD807F5F3A5B26348B7AA3912F230D33C0CF365F52D.
- Baseline authoritative291files from cc_clean_ab_20261030_run3/frozen_source remain byte-identical. No official source modified, no commits/push/reset/clean.
- Root AGENTS files searched atworkspace/parents, none. Snapshot base435seal verified at start.

## Inputs and matrix

-80real declarations: first48CF,16DS,16BERT by officialtrace1..10/originalrow order, interleaveCF,CF,CF,DS,BERT. First40at0,20at600,20at1200. Original demand,batch,full epochs/work retained; original_name/source_trace/row/arrival saved. Controlled arrivals, NOT natural trace replay.
-20GPUs, one node/type: AWS4/DGX8/RTX8. Main perGPU3syntheticCPU/62.5GB =>12CPU250GB,24CPU500GB,24CPU500GB.
- R0+Elegacy fixed, alpha0/.5, phi/mappings/source grid unchanged. seed_profiles(1,8) common. No real hardware claim.
- Core order: dual Medium SIA_P,FULL,SIA_AUX,PROP,CPU,MEM,FO; then dual Neutral A00,A01,A10,A11.
- ActualSIA_P can also be Neutral reference: same proportional reference point,rho1all33configs, strict real bounded equality. Do not treat reuse as independent sample.
- Optional complete pairs: cpu_only PROP/CPU (RAM source-derived ample); ram_only PROP/MEM (CPU source-derived ample). Start only if all11core ALL_COMPLETE and pair has2*1800+60sec before formal stop. No result-based JCT selection.
- SIA_P = actual official Sia GPU/placement, noL2. SIA_AUX = actualSia intent/placement + existingECT aux optimization, NOT officialSynergy-TUNE.
- PROP/CPU/MEM/FULL = common reciprocal-current-ETA objective,FIFO,continuity, only aux factors differ. FO = same controller's prop-domain GPU/service proposal then freeze and auxopt.
- A00/01/10/11 = common Siafair objective with FIFO×continuity ablation, identical historical eligibility/lifetime250startup/locks. Not officialSia; A/B percentages not additive.

## Preflight and correctness

-33application/GPU configurations covered;49090single-column resourcepoints,30390fully source-supported. 6/33proportional configs have denominator clipping. Counts in preflight/coverage.json, fullcsv.gz.
- Genuine supported CF/BERT1GPU AWS pair: CF4.213316CPU93.468723GB,rho1.534810; BERT1.676408CPU26.969888GB,rho1; sum<=6CPU125GB. All coords inbounds. Entire20GPU embedding uses2such pairs plus16CF/DSproportional and fits60CPU1250GB. Static opportunity, not JCT proof.
- Worst complete grid1338144columns; dense89row numeric matrix0.887GiB beforePythonobjects/copies. Sparse representation and immutable response cache preserve values; unused historical diagnostic re-enumerations removed. Existing unknown-ETA nonprop filter moved BEFORE construction; trusted grid restored before audit. Complete scored-vector/trajectory equivalence tested.
- run1 preserved: Sia80complete, FULL rejected old20-job INSTALLER (before training). run2 preserves installation fix/strict source-equivalent cold filtering; interrupted for proven redundant solver-call optimization, no overall FULL JCT.
- run3 adds exact_optional_proof.py: under fixed primary and minimum-startup secondary optimum, nonrequired candidate utility0 + positive startup cannot be selected (remove column lowers startup without affecting required/locks/resources/primary). Only certifies redundant presence query0 using current fully revalidated vector; no column/weight/grid change. No effect onAgroup, noL2 intoSia. Actual backend settings30sec/gap0 unchanged.
- Tests276PASS(258old+12core80+6exact certificate). OriginalSia3round exact regressionPASS. Real80FULL3round prefix EXACT same selected/execution/full candidate score vector as literal run2;39/38/37redundant calls certified. Proof saves some overhead but does NOT solve main mixed-resource MIP scalability.
- run1/run2 Sia309round semantic equality: job_end_state,execution,official GPU intent. Initial literal comparison flagged relocated template paths only, source SHA equal. Both artifacts retained.

## Finish work (do not end turn with merely a progress update)

1. Keep monitoring existing session until supervisor ends/core attempted or formal deadline. No further source/parameter changes. If failure, retain and let registered other cells continue. Do not infer whole-run JCT for partials.
2. Independent raw audit: `.venv-sia-baseline/Scripts/python.exe -X utf8 -B maintenance/audit_core80_8h_20261030.py` fromworkspace. It currently targets v3/run3, creates OUT/audit/ (must not exist). It reconstructs completion/time/resources/rho/locks/capacity/events, paired comparisons, service/resource factorial effects, per_job.csv. Scripts outside frozen tree may be corrected for audit-only bugs, retaining failures. Never edit raw.
3. Report builder: maintenance/report_core80_8h_20261030.py (expects audit complete) writes NEW research_notes/synergy/CC_CORE80_8H_REPORT_20261030.md and OUT/REPORT.md. Enrich top with direct conclusions, completed vs failed slots, actual contribution figures. Do not overclaim incomplete pair. Record full walltime including allprep/retries from18:49:54.
4. Additional diagnostics to finish: source/delivery hashes, baseline291 unchanged, old435 historical seal retained; actual backend vs certified calls, full selection time vs candidate+commit wall; response clipping rates; per-app/net JCT decomposition + long-tail influence. Preserve latest frozen source and old source attempts. No post-hoc delete jobs.
5. Finalseal only AFTER all writers/logfiles closed; don't hash a still-growing seal-console file (past experiment pitfall). Include report and scripts/dependency lock/fullsource snapshot and rawfiles/hash. Add REPRODUCE.md, source_changes.patch and concise execution/status tables.
6. Stop; no Low/High,160jobs,104resume,newalgorithm tuning. User wants completion status and two mechanism lines' actual contributions, not a promise to continue later.

Warning to include: **SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE**.
Postbatch optional correctness-only diagnostic: if main11finished and>=15min remains before FORMALstop, may run maintenance/diagnose_core80_full_failure.py serially under an EXTERNAL720sec watchdog. Exactly5 original FULL boundaries (80declarations, full work, unchanged source), no JCT claim. Catches exception.solver_diagnostics omitted by generic failure.json. Must preserve old failure; no automatic tolerance/gap/target fixes. Do not run concurrently with formal worker. Need watchdog because script cannot interrupt an in-flight native solver. Not a new formal cell or repeat sample.

## 2026-10-30 22:19UTC checkpoint
Session4015 still runs recovery1 serial dispatcher. Source302 unchanged; budget deadlines unchanged. Complete: SIA_P9290.875 mean; A0011798.8875; A01same; A117684.35; SIA_AUX7420.4125 (all80/80, independent raw audits0errors). FULL failed numerical retry at300sim seconds; A10failed lexical inconsistency at780sim seconds. PROP currently90rounds/27completed/~2364sec, cap3000; then CPU,MEM,FO fixed. No optional4 gate due failures. Do not start duplicates or alter source.
New analysis artifacts: completed_four_interim_audit.json, SIA_AUX_interim_raw_audit.json, service_trajectory_diagnostic.json (A00/A01 physical allocation and training exact395round equivalence), source_changes.patch/source_patch_verification.json (291baseline unchanged,11added). SIA_AUX vsSia mean -20.13%,56jobsbetter24worse, mean saving1870.4625s=wait1591.5+checkpoint42.375+train236.5875. All provisional interpretation must preserve Elegacy boundary limitation; final audit splits supported/clipped local gains.
Final scripts maintenance/audit_core80_8h_20261030.py and report_core80_8h_20261030.py updated: canonical paths, independent missed-lock detection, direct pre-rho execution, perjob observed/censor times, SIA_AUXvsSia and resource controls vsSia. Final outputs audit/ and report do not yet exist; run only once after batch, preserve failures if reporting bug.
When batch ends and>=15min remains before01:49:54UTC, serial optional correctness diagnosis: maintenance/guard_core80_failure_diagnostic.py (external720sec watchdog, exactly5FULLboundaries, no parameter/source changes, no full-JCTclaim). Never alongside formal worker. This preserves missing solver_diagnostics field without fixing solver or rerunning performance to win. Stop by02:49:54UTC including report/seal.
