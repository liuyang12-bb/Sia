"""Validate and summarize the controlled S1/S2 synthetic suite."""

import argparse
from collections import defaultdict
import json
from pathlib import Path


SCENARIOS = ("neutral", "cpu_sensitive", "memory_sensitive", "mixed")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def core_round(record):
    record = {key: value for key, value in record.items()
              if key not in ("synthetic_resource_model", "sia_gpu_decision_changes")}
    record["submitted_jobs"] = [
        {key: value for key, value in job.items()
         if key not in ("auxiliary_multiplier", "auxiliary_profile")}
        for job in record["submitted_jobs"]
    ]
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--historical-sia-log", type=Path, default=None,
                        help="optional prior unmodified --policy=sia JSONL for core-field regression")
    args = parser.parse_args()
    root = args.root.resolve()
    summaries, logs = {}, {}
    for scenario in SCENARIOS:
        for mode in ("s1", "s2"):
            name = f"{scenario}_{mode}"
            summaries[name] = read_json(root / "summaries" / f"{name}.json")
            logs[name] = read_jsonl(root / "runs" / f"{name}.jsonl")

    validations = {}
    neutral_s1 = [core_round(row) for row in logs["neutral_s1"]]
    neutral_s2 = [core_round(row) for row in logs["neutral_s2"]]
    validations["neutral_s1_s2_round_exact"] = neutral_s1 == neutral_s2
    if args.historical_sia_log is not None:
        historical = [core_round(row) for row in read_jsonl(
            args.historical_sia_log.resolve())]
        validations["historical_sia_s1_neutral_round_exact"] = (
            historical == neutral_s1)
    validations["neutral_s1_s2_metrics_exact"] = all(
        summaries["neutral_s1"][key] == summaries["neutral_s2"][key]
        for key in ("completed_jobs", "mean_jct_seconds", "p95_jct_seconds",
                    "p99_jct_seconds", "makespan_seconds", "gpu_hours",
                    "restart_count", "sia_gpu_decision_changes"))
    validations["s1_all_scenarios_core_exact"] = all(
        [core_round(row) for row in logs[f"{scenario}_s1"]] == neutral_s1
        for scenario in SCENARIOS)
    validations["all_complete"] = all(
        summary["completed_jobs"] == summary["total_jobs"]
        for summary in summaries.values())
    validations["no_capacity_violations"] = all(
        not row["synthetic_resource_model"]["capacity_violations"]
        for run in logs.values() for row in run)
    validations["s1_multiplier_always_one"] = all(
        job["multiplier"] == 1.0
        for name, run in logs.items() if name.endswith("_s1")
        for row in run
        for job in row["synthetic_resource_model"]["jobs"].values())
    validations["neutral_multiplier_always_one"] = all(
        job["multiplier"] == 1.0
        for name in ("neutral_s1", "neutral_s2")
        for row in logs[name]
        for job in row["synthetic_resource_model"]["jobs"].values())
    validations["insensitive_multiplier_always_one"] = all(
        job["multiplier"] == 1.0
        for name, run in logs.items() if name.endswith("_s2")
        for row in run
        for job in row["synthetic_resource_model"]["jobs"].values()
        if job["profile"] == "insensitive")
    validations["sensitive_gain_observed"] = all(
        any(job["multiplier"] > 1.0
            for row in logs[f"{scenario}_s2"]
            for job in row["synthetic_resource_model"]["jobs"].values())
        for scenario in ("cpu_sensitive", "memory_sensitive", "mixed"))
    validations["no_solver_fallback"] = all(
        summary["fallback_count"] == 0 for summary in summaries.values())
    validations["simulation_time_monotonic"] = all(
        all(earlier["timestamp"] < later["timestamp"]
            for earlier, later in zip(run, run[1:]))
        for run in logs.values())

    comparisons = {}
    task_level = {}
    for scenario in SCENARIOS:
        s1 = summaries[f"{scenario}_s1"]
        s2 = summaries[f"{scenario}_s2"]
        comparisons[scenario] = {
            metric: {
                "s1": s1[metric],
                "s2": s2[metric],
                "delta": s2[metric] - s1[metric],
                "delta_percent": ((s2[metric] / s1[metric] - 1.0) * 100.0
                                  if s1[metric] else 0.0),
            }
            for metric in ("mean_jct_seconds", "p95_jct_seconds", "p99_jct_seconds",
                           "makespan_seconds", "gpu_hours", "cpu_core_hours",
                           "ram_gb_hours", "restart_count",
                           "sia_gpu_decision_changes")
        }
        s1_jobs = {job["name"]: job for job in s1["jobs"]}
        s2_jobs = {job["name"]: job for job in s2["jobs"]}
        task_level[scenario] = []
        for name in sorted(s1_jobs):
            jct_s1 = s1_jobs[name]["jct_seconds"]
            jct_s2 = s2_jobs[name]["jct_seconds"]
            task_level[scenario].append({
                "name": name,
                "profile": s2_jobs[name]["synthetic_profile"],
                "s1_jct_seconds": jct_s1,
                "s2_jct_seconds": jct_s2,
                "delta_seconds": jct_s2 - jct_s1,
                "delta_percent": (jct_s2 / jct_s1 - 1.0) * 100.0,
            })

    allocation_evidence = {}
    for scenario in SCENARIOS:
        choices, max_multiplier = defaultdict(set), defaultdict(lambda: 1.0)
        for row in logs[f"{scenario}_s2"]:
            for name, job in row["synthetic_resource_model"]["jobs"].items():
                choices[name].add(job["choice"])
                max_multiplier[name] = max(max_multiplier[name], job["multiplier"])
        allocation_evidence[scenario] = {
            name: {"choices": sorted(choices[name]), "max_multiplier": max_multiplier[name]}
            for name in sorted(choices)
        }

    result = {
        "claim_boundary": "SYNTHETIC RESOURCE MODEL — NOT REAL SATURN PERFORMANCE",
        "validations": validations,
        "comparisons": comparisons,
        "task_level": task_level,
        "allocation_evidence": allocation_evidence,
        "summaries": summaries,
    }
    if not all(validations.values()):
        result["classification"] = "INVALID"
        result["classification_reason"] = "one or more correctness validations failed"
    else:
        clear_tail_regression = any(
            comparisons[scenario][metric]["delta_percent"] > 5.0
            for scenario in ("cpu_sensitive", "memory_sensitive", "mixed")
            for metric in ("p95_jct_seconds", "p99_jct_seconds", "makespan_seconds"))
        has_primary_benefit = any(
            comparisons[scenario][metric]["delta_percent"] < 0.0
            for scenario in ("cpu_sensitive", "memory_sensitive", "mixed")
            for metric in ("mean_jct_seconds", "makespan_seconds", "gpu_hours"))
        if clear_tail_regression:
            result["classification"] = "MIXED"
            result["classification_reason"] = (
                "correctness passed, but at least one sensitive scenario regressed "
                "P95, P99, or makespan by more than 5%")
        elif has_primary_benefit:
            result["classification"] = "MECHANISM_VALID"
            result["classification_reason"] = (
                "correctness passed and sensitive scenarios produced primary-metric benefit")
        else:
            result["classification"] = "NO_BENEFIT"
            result["classification_reason"] = (
                "correctness passed but no sensitive scenario improved mean JCT, makespan, or GPU-hours")
    output = root / "synthetic_comparison.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "classification": result["classification"],
        "validations": validations,
        "output": str(output),
    }, indent=2))
    return 0 if result["classification"] != "INVALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
