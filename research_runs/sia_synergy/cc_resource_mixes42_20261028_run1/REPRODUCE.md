# Reproduce frozen42 cells
Use source_snapshot.zip or source/ with282 SHA256-pinned files, dependencies.lock.txt and inputs/M1-M3 CSVs. Original source is cc-joint-resource-study; unchanged-source hashes in preregistration.json.
Runner: maintenance/run_resource_mixes42_20261028.py; copy in this directory. ROOT resolution assumes script placed in project maintenance/. Output path is fixed to cc_resource_mixes42_20261028_run1 and refuses existing cell directories; for an independent reproduction use a newly preregistered output root, retain all scientific parameters and original evidence.
Command: .venv-sia-baseline/Scripts/python.exe -X utf8 -B maintenance/run_resource_mixes42_20261028.py batch
Do NOT run this command while current batchsession99340 remains active. No duplicated runner.
Independent auditor: maintenance/audit_resource_mixes42_20261028.py, using pinned helper maintenance/audit_cc_resource_factorial_complete_20261028.py. It reads compressed raw events and reconstructs metrics; never runs training. Original helper SHA recorded in audit_helper.sha256.json.
All formal cells use20jobs; acceptance_fixtures/ are short3-job correctness fixtures only. Frozen E-legacy constant boundary extension, not measured response outside source domain. No complete-domain empirical support claim.
SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE.
