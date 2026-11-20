"""Recheck raw receipts and write the combined result for the two fixed studies."""

from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

from verify_search import canonical


OUT = Path(__file__).resolve().parents[2] / "outputs"
scale = json.loads((OUT / "search_scale12_timing.json").read_text())
assert scale["valid"] and len(scale["repeats"]) == 3
reference = scale["repeats"][0]["methods"]["FO"]["totals"]
for repeat in scale["repeats"]:
    for run in repeat["methods"].values():
        assert run["totals"]["completed"] == 12
        assert run["totals"]["trajectory"] == reference["trajectory"]
        assert run["totals"]["completed_jct_seconds"] == reference["completed_jct_seconds"]

cells = {}
raw = {}
for directory in sorted((OUT / "cpu20_comparison").iterdir()):
    if not directory.is_dir():
        continue
    summary = json.loads((directory / "summary.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    assert summary["completed"] == summary["total"] == 20
    assert summary["actual_node_capacities"] == manifest["capacities"]
    time_totals = defaultdict(lambda: defaultdict(float))
    digest = hashlib.sha256()
    count = 0
    with gzip.open(directory / "rounds.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            count += 1
            selected = record["decision_for_next_interval"]["result"]["selected"]
            physical = {"selected": selected, "execution": record["execution"],
                        "rescale_events": record["rescale_events"], "migrate_events": record["migrate_events"]}
            digest.update((canonical(physical) + "\n").encode())
            for event in record["execution"]:
                assert event["independent_execution_audit"]["valid"]
                for key, field in (("wait", "waiting_seconds"), ("training", "training_seconds"),
                                   ("checkpoint", "checkpoint_delay_consumed")):
                    time_totals[event["job_id"]][key] += event[field]
    assert count == summary["rounds"]
    for name, totals in time_totals.items():
        for key, value in totals.items():
            assert abs(value - summary["jobs"][name][key]) < 1e-6
    raw[directory.name] = {"rounds": count, "physical_trajectory_sha256": digest.hexdigest()}
    cells[directory.name] = summary

base = cells["SIA_P_medium"]
joint = cells["LAZY_FO_medium"]
local = cells["LOCAL_medium"]
assert raw["LAZY_FO_medium"] == raw["LOCAL_medium"]
assert all(cells[name]["jobs"].keys() == base["jobs"].keys() for name in cells)
assert base["rho_range"][0] != 1.0
assert cells["SIA_P_neutral"]["rho_range"] == [1.0, 1.0]
by_job = {
    name: {"sia_jct": row["jct"], "lazy_jct": joint["jobs"][name]["jct"],
           "change_seconds": joint["jobs"][name]["jct"] - row["jct"]}
    for name, row in base["jobs"].items()
}
comparison = {
    "valid": True, "scale12_all_complete_and_equivalent": True,
    "cpu20_all_complete": True, "cpu20_raw_reconciliation": raw,
    "cpu20_per_job": by_job,
    "faster_jobs": sum(row["change_seconds"] < 0 for row in by_job.values()),
    "slower_jobs": sum(row["change_seconds"] > 0 for row in by_job.values()),
    "same_jobs": sum(row["change_seconds"] == 0 for row in by_job.values()),
    "mean_jct_reduction_vs_sia": 1 - joint["mean_jct"] / base["mean_jct"],
    "p95_jct_change_vs_sia": joint["p95_jct"] / base["p95_jct"] - 1,
    "local_accepted": local["local_status"].get("accepted", 0),
    "scope": "single frozen synthetic cross-hardware workload, simulator decision wall excluded from JCT",
}
with (OUT / "two_studies_validation.json").open("x", encoding="utf-8") as stream:
    json.dump(comparison, stream, indent=2)
print(json.dumps(comparison, indent=2))
