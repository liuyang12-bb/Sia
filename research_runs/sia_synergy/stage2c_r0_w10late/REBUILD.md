# Fixed R0 four-cell experiment reconstruction

Use a NEW directory. Never restore over a protected checkout or an old run.

1. Clone `attempt1/source.bundle` and check out the exact commit in
   `attempt1/preregistration.json`. The bundle contains reachable Git history.
2. Restore byte-exact files from `final_full_source.zip`; validate every entry
   against `source_manifest.json`. This preserves the tested Windows line endings.
   The smaller per-attempt source ZIP contains the simulator execution subtree;
   the final ZIP contains the entire tracked checkout.
3. Restore official matrix files from the Stage2B `synergy_matrix_snapshot.zip`
   to `research_sources/Synergy/simulator/models`, preserving hashes. R0 registry
   and original cohort/config are snapshotted inside each attempt directory.
4. Use independent Python 3.10 and the saved dependency lock. No offline wheels
   or operating-system image are provided, so arbitrary-platform bitwise solver
   reproducibility is not claimed. HiGHS/CBC engine versions were not separately
   queried; SciPy/CyLP versions and exact solver options are recorded.
5. Run the 94-test suite in `sia-simulator` with `python -B -m unittest discover
   -s tests -p test_*.py -v`, directing STAGE1_EVIDENCE_DIR, STAGE2B_EVIDENCE_DIR,
   STAGE2C_EVIDENCE_DIR to NEW empty evidence directories. Tests contain only the
   bounded fixtures, not the 20-job matrix.
6. `freeze_and_run.py freeze <new_attempt>` rebuilds absolute paths from the
   current workspace and freezes all input/code/environment hashes before work.
   Preserve file contents; relocated path strings require a new manifest and
   are not the original run's identical manifest bytes.
7. `freeze_and_run.py run <new_attempt>` executes exactly Neutral FP/FO/JP/JO,
   then Medium FP/FO/JP/JO, separate processes, 6000 rounds maximum, serial,
   >=2 GiB at each launch. It refuses existing directories/logs. It stops on
   technical errors. Do not rerun into an occupied attempt directory.
8. `analyze.py <new_attempt>` re-reads raw compressed JSONL, validates the
   execution/response/geometry/time invariants and creates new summary files.

The fixed R0 numbers are used only as response demand scales, never as online
Goodput samples. The cohort keeps the prior fixed Sia seed_profiles(1,8) setting;
unseeded correctness fixtures are separately retained. L2 parameters, fairness,
resource capacities, quota grid, response exponent and solver options are frozen.

All scripts in the output root are included in delivery_manifest.json. Initial
setup should not be rerun against existing trees. Historical failure evidence
red1/green2/green3 is retained. No nine-cohort or 160-job command is part of this
reconstruction procedure.
