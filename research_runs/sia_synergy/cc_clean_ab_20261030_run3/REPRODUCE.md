# Reproduce the frozen experiment

Authoritative scientific definition: preregistration.json + DESIGN.md. All289 base files are unchanged; final adapters and tests are in frozen_source/ and frozen_source_snapshot.zip. Runtime pins: frozen_source/dependencies.lock.txt. Original official sources and old experiments are read-only.

Replay ONE cell, with identical source/input/dependency checks and a NEW destination:

```powershell
D:/科研启航/Sia/.venv-sia-baseline/Scripts/python.exe -X utf8 -B D:/科研启航/Sia/research_runs/sia_synergy/cc_clean_ab_20261030_run3/rebuild_scripts/replay_clean_ab_20261030.py --artifact D:/科研启航/Sia/research_runs/sia_synergy/cc_clean_ab_20261030_run3 --dest D:/科研启航/Sia/research_runs/sia_synergy/NEW_REPLAY --workload COMPACT --name B_JO
```

The destination must not exist. Each replay is an independent process; CPU/RAM/model/work are frozen. The helper changes only output destination, reads the original frozen source, and refuses input/source/dependency drift. Do not replay into the original directory. For all15, use the fixed order from preregistration and separate destinations. No replay was automatically executed during delivery.

Raw data: cells/*/rounds.jsonl.gz, summary.json, output_hashes.json. Recompute: audit_clean_ab_r3_20261030.py (writes exclusive new files, use a copied artifact with original analyses preserved/redirected). Metrics derive from actual completion events, captured raw rates, and allocation-time integrals, not summary alone. Test suites require the retained historical fixture files referenced by existing unit tests; formal-run GPU/profile/model data are included in frozen_source.

Earlier attempts run1/run2 contain interrupted control-variable debugging evidence and must not be merged with run3. C is an offline precheck, not a new response model. All conclusions are synthetic cross-hardware simulation evidence.
