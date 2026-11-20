# Stage1 source reconstruction (new destinations only)

No reconstruction command below should target an existing repository or old run directory.

## Exact bytes

`source_snapshot.zip` contains the actual pre-repair frozen worktree, including its three tracked edits and six necessary untracked source/config files. `snapshot_manifest.json::source_hashes` lists all 782 file hashes. Extract it only into a new directory and verify every hash before using it.

`repair_source_snapshot.zip` similarly contains all tracked repair-version files. Its hashes and local commit are in `repair_manifest.json`. This is the byte-exact option when Windows Git autocrlf would otherwise change line endings. Generated Python caches are excluded unless already part of the inherited tracked repository.

`external_response_templates.zip` supplies the 12 official literal response matrices under their original `research_sources/Synergy/simulator/models/` paths. Their hashes are also present in the response preflight. Do not overwrite the existing Synergy checkout when extracting.

## Git history

The bundles are self-contained local Git history, not patches that require network access:

```powershell
git clone <absolute-path-to-repair.bundle> <NEW-repair-clone>
git -C <NEW-repair-clone> switch --detach 0461dcdf061197fdfffe23b2b3d07e46e0d9efb2
```

The repair commit's parent is `1cbc3f5abb4de18ccca3a75749e1fc704fd46027` (actual dirty-source archival baseline). The original inherited HEAD is `be03f95c7bc641cbef0e2a4452bc4c7a3b5404e4`.

`tracked.patch` documents the pre-existing changes against the original HEAD; `repair.patch` documents this stage against the archival baseline. Do not reapply either on top of a commit already containing it. Patch application alone does not restore the pre-existing untracked files; use the archive or complete baseline commit.

## Runtime

Python 3.10.10; exact installed package pins are in `requirements.lock.txt`. No packages were installed or upgraded. Recreate a separate virtual environment, never alter system Python. The dependency listing is not a vendored wheel repository: OS ABI and linked solver builds still need verification. `dependencies.json` records installed distributions; no independent CBC/HiGHS version was guessed.

Place the reconstructed simulator repository next to a reconstructed `research_sources/Synergy/simulator/models/` directory, as expected by tests. Run from `sia-simulator`:

```powershell
<separate-venv-python> -B -m unittest discover -s tests -p test_*.py -v
```

This runs bounded tests, not a 20-job experiment. The optional `STAGE1_EVIDENCE_DIR` must point to a fresh location if tiny-fixture JSON is desired.

Read-only data preflight additionally needs the OLD run2 inputs/logs, which are intentionally not duplicated into source archives; use their recorded hashes. It does not generate training data. Never point any output at the old run1/run2 or existing report paths.

## Boundaries

This version does not resolve the B3-PROP empty-optimum absorbing state, unify 2x2 admission objectives, or freeze a new response model. Do not use its passing engineering tests as approval for formal experiments.
