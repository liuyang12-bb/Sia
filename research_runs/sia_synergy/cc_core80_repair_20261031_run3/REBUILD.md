# Rebuild without losing historical work

Source archive: source_snapshot.zip (315files), verify every entry using source_hashes.json. This isolated source copy is not a Git checkout; hashes/full bytes are the version authority. Extract only into a new directory below the workspace; never overwrite an existing tree.

Legacy test fixtures/official baseline: external_test_dependencies.zip, 401files at workspace-relative paths; verify existing files rather than overwrite. The original protected repositories remain untouched. The legacy small_mechanism preregistration contains absolute D:/科研启航/Sia paths: tests reproduce directly under that workspace layout. When relocating, repoint ONLY copied path strings and record original/relocated hashes; do not change numerical configuration, data or original artifacts.

Runtime: Python3.10.10 and dependencies.lock.txt/environment.json; use independent venv, never global upgrades. All302 software tests: from NEW_TREE/sia-simulator run the matching venv python -X utf8 -B -m unittest discover -s tests -p test_*.py -v. This command is software verification; no formal launcher is invoked.

Formal11-cell restart remains blocked. planned_preregistration.json is a reviewable protocol, deliberately not an executable launch manifest. Do not use old wait_accept/accept_and_launch scripts to bypass FORMAL_BLOCKED_AUTO_REVIEW.json.
