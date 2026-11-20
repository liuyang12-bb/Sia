# Six-job heterogeneous CPU competition: four-policy completion experiment

Status: COMPLETE_ALL_FOUR_VALID. Four policies completed 6/6; no algorithm changes.
Final interpretation: RESULTS_AND_CAUSES.md; full tables: REPORT.md.
Mean JCT SIA-P=5636.50, SIA-ECT-AUX=5616.33, ECT-P=7308.83, ECT=6832.50 seconds.
ECT vs ECT-P improves 6.52%, all six jobs; ECT vs Sia worsens 21.22%.
Raw event/time/segment/capacity audits passed. Stop; no automatic optimization loop.

User authorized this six-job experiment after reviewing the workload and CPU sensitivity.
This authorization supersedes the previous task's stop for these four cells only.
Do not resume full160/104 runs or modify algorithm/model parameters.

Freeze actual JOINT-ECT source from sia-joint-completion-time, not bare Git HEAD.
Workload: existing CF0/1/2 and DS8/9/12, unchanged work/batch, synthetic coarrival 0.
AWS and DGX each 2GPU/12 synthetic CPU units/250GB RAM. Medium R0 E-legacy alpha .5.
Order SIA-P, SIA-ECT-AUX, JOINT-ECT-P, JOINT-ECT. 60s, max 6000 rounds, stop on completion.
Independent processes, memory >=2GiB and disk >=5GiB before each launch.
No result-driven tuning, new seeds, capacity changes, grid changes, or window extension.
Known response support gaps retained and counted; never delete a legal GPU config.

Recovery: delivery_manifest.json, independent_audit.json, preregistration.json and execution_manifest.jsonl;
never reuse a nonempty cell. All authorized cells are finished; do not start more.
After four cells, audit raw completion events, training segments, resource hours and locks,
report completion/censoring/failure, paired per-job results, and stop.

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE.
