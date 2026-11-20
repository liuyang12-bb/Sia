# 80-job core experiment: reconstruction and evidence selection

**SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE**

## Authority

- Frozen algorithm/source directory: `D:/科研启航/Sia/cc-joint-core80-8h-v3`.
- Complete302-file archive: `frozen_source_snapshot.zip`, SHA256
  `43887CBD457E216EE84BEDD807F5F3A5B26348B7AA3912F230D33C0CF365F52D`.
- Original source/data hashes: `preregistration.json`, SHA256
  `3A7372248D0948BCD651A1C814503A29899F394139C54673ECB7D61883CB582A`.
- Baseline291files are unchanged. All additional implementation is opt-in;
  original entry points, official Sia/Synergy and historical results remain.
- Actual primary result directory for each cell is fixed in
  `canonical_result_paths.json`, not chosen by the best JCT.
- The operational revision is frozen in
  `execution_recovery1/preregistration.json`. It retains all302source hashes,
  workload, model, capacities,6000simulated rounds,60sec intervals and solver
  settings. Only total process compute budget became3000sec and cheap service
  cells were prioritized. The ORIGINAL global8-hour deadline is unchanged.
  See `BUDGET_FIELD_CLARIFICATION.md` for inherited parent narrative fields.

## Data and environment

`inputs/core80_workload.csv` contains full declarations and provenance, with
controlled0/600/1200 arrivals. `input_origin.json` documents deterministic source
trace/row selection. No application epochs or work are shortened. Exact GPU
profiles, Synergy matrices and fixed R0 registry are inside the source snapshot;
all are covered by the source/input hashes. Missing coverage is rejected rather
than filled withphi1. E-legacy clipping is preserved and audited explicitly.

Use the existing isolated Python3.10.10 environment:
`D:/科研启航/Sia/.venv-sia-baseline/Scripts/python.exe`.
Full package lock: `dependencies.lock.txt`; queried native/runtime information:
`environment.json`. NumPy1.23.5, SciPy1.10.1, CVXPY1.3.2 and CyLP0.91.6.
An independent CBC/HiGHS binary version was not exposed by the queried API;
do not infer one. `seed_profiles(1,8)` is profile initialization, not a newly
selected random seed. No seed search or global dependency update occurred.

## Checks and commands

From the frozen source's `sia-simulator` directory:

```powershell
D:/科研启航/Sia/.venv-sia-baseline/Scripts/python.exe -X utf8 -B -m unittest discover -s tests -p 'test_*.py' -v
```

Original officialSia bounded comparison is implemented by
`sia-simulator/check_stage1_legacy_regression.py --probe <simulator-directory>`.
Tests/ contains276-test output,80queue/real-unseeded-chain tests, source-storage
equivalence, current-oracle invariants and the literal-vs-certified-MIP prefix
comparison. A passing short prefix is NOT a completed80-job experiment.

A registered individual cell is executed from the frozen source root with:

```powershell
D:/科研启航/Sia/.venv-sia-baseline/Scripts/python.exe -X utf8 -B run_core80.py --manifest <new-preregistration.json> --cell dual_medium_CPU
```

Never point that command at an already-used output directory. To reproduce
later, copy the manifest to a NEW root, create its `cells/` parent, update only
output paths and the operational wall-clock timestamps, preserve every
scientific field, and hash the new manifest before starting. The original
absolute input paths can be restored directly; relocation must preserve every
content hash and explicitly record path substitutions. Changing the maximum
PROGRAM compute budget does not change a completed simulated event, whereas
changing6000SIMULATED rounds would change observation semantics and is not
authorized by this reproduction note.

Runs are independent Python processes and serial. Check2GiB available memory
and5GiB disk before launch. An external process watchdog must enforce the chosen
global compute deadline even while a native solver is running. Keep all failed,
interrupted and numerical-error attempts; do not repair raw gzip streams.

## Interpretation and raw recomputation

- `rounds.jsonl.gz`: directly recorded pre-rho rates, training segments, progress,
  completion events, selected GPU/CPU/RAM/placement, lock checks and solver traces.
  Full candidate score catalogs are hashed/counted to avoid repeated enormous
  logs; the frozen source/data regenerate them.
- `summary.json` exists only for completed or6000-round-finished worker loops.
  Compute interruption is separate and cannot be summarized by completed-only
  meanJCT or imputed unfinished completion times.
- Raw recomputation: `maintenance/audit_core80_8h_20261030.py`; it resolves the
  canonical paths and writes NEW `audit/` outputs. Never overwrite a prior audit
  attempt when repairing a reporting bug.
- Service four cells share the frozen Sia-style fairness utility and common
  eligibility; resource cells share completion priority/FIFO/continuity. Their
  percentages are conditional effects and cannot be added across the groups.
- SIA_P is actualSia GPU/placement with common auxiliary execution. SIA_AUX is
  actualSia intent plus existingECT auxiliary optimization, not official
  Synergy-TUNE. PROP is a CC control and must not be renamed originalSia.
- Primary full-JCT contrasts require all80jobs complete and all raw audits valid.
  A CPU/RAM opportunity or a faster partial progress curve is not that evidence.
- Clipping, synthetic transfer, zero extra CPU/RAM-only reallocation overhead,
  one-node-per-GPU-type topology and finite controlled workload limit conclusions.

Old run1/run2 and original run3's1800sec attempts are preserved. The revised
primary rows were fixed before their outcomes; no best-run selection is allowed.
This is one deterministic controlled cohort, not80independent replications.


## Terminal outcome and analysis reconstruction

Main11 were all attempted:5full80completion,3numerical/lexicographic failures,3confirmed external COMPUTE_TIMEOUT. Optional4not launched. Exact receipts: audit/termination_analysis.json; overviewCSV: audit/core80_coverage.csv. Preserve audit/raw_recomputed_metrics.json: its generic timeout-or-interruption label is refined by the separate watchdog-based classification, never by rewriting raw logs.

Analysis scripts are copied with hashes into analysis_source/. Run audit_core80_8h_20261030.py only against a NEW audit destination when reproducing; its default refuses overwrite. Then supplement_core80_results.py and report_core80_8h_20261030.py. Existing output files are immutable. Report generation time and final seal time differ; final_receipt.json records total measured wall through final packaging.

One bounded correctness reproduction after the matrix used guard_core80_failure_diagnostic.py, max5boundaries/720sec, and captured the same FULL failure. inspect_core80_solver_model.py only multiplies captured vectors by frozen constraints; it does not call a solver or train. Complete modelSHA and prefix equivalence are preserved. No tolerance/target/model repair was applied.
