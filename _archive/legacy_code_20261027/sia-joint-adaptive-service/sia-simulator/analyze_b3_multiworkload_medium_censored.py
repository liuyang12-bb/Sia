"""Audit a Medium matrix which contains an explicitly censored policy run.

This reads the compact per-cohort manifests rather than the very large round logs.
It never imputes an unfinished JCT and excludes censored pairs from JCT comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


POLICIES = ("b1", "sy", "b4", "b5", "b3")
COMPARATORS = ("b1", "sy", "b4", "b5")
COHORT_ORDER = (
    "w1_early", "w1_middle", "w1_late",
    "w5_early", "w5_middle", "w5_late",
    "w10_early", "w10_middle", "w10_late",
)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def percentile(values, q):
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * q
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def compact_metrics(record):
    jcts = {key: float(value) for key, value in record["jcts"].items()}
    completion_times = {
        key: float(value) for key, value in record["completion_times"].items()
        if value is not None
    }
    jobs = record["jobs"]
    return {
        "completed_jobs": int(record["completed_jobs"]),
        "total_jobs": int(record["total_jobs"]),
        "completion_rate": float(record["completion_rate"]),
        "all_jobs_completed": bool(record["all_jobs_completed"]),
        "hit_round_limit": bool(record["hit_round_limit"]),
        "average_jct_completed_only_seconds": (
            float(record["average_jct_completed_only"])
            if record["average_jct_completed_only"] is not None else None
        ),
        "median_jct_completed_only_seconds": statistics.median(jcts.values()) if jcts else None,
        "p95_jct_completed_only_seconds": percentile(jcts.values(), 0.95) if jcts else None,
        "p99_jct_completed_only_seconds": percentile(jcts.values(), 0.99) if jcts else None,
        "makespan_completed_only_seconds": max(completion_times.values()) if completion_times else None,
        "jcts": jcts,
        "completion_times": completion_times,
        "waiting_seconds": record["waiting_seconds"],
        "total_waiting_seconds": sum(float(value) for value in record["waiting_seconds"].values()),
        "resource_hours": record["resource_hours"],
        "restarts": sum(int(item["num_restarts"]) for item in jobs.values()),
        "wasted_gpu_seconds": sum(float(item["wasted_gpu_seconds"]) for item in jobs.values()),
        "rounds": int(record["rounds"]),
        "simulated_seconds": float(record["simulated_seconds"]),
        "wall_seconds": float(record["wall_seconds"]),
        "min_rho": float(record["min_rho"]),
        "max_rho": float(record["max_rho"]),
        "optimizer_total_seconds": float(record["optimizer_total_seconds"]),
        "optimizer_p99_seconds": float(record["optimizer_p99_seconds"]),
        "capacity_violation_count": len(record["capacity_violations"]),
        "cpu_saturated_gpu_idle_event_count": int(
            record["contention_evidence"]["cpu_saturated_gpu_idle_event_count"]
        ),
        "wait_reason_rounds": record["contention_evidence"]["wait_reason_rounds"],
        "jobs": jobs,
    }


def compare(base, b3):
    if not base["all_jobs_completed"] or not b3["all_jobs_completed"]:
        return {
            "status": "CENSORED_NO_IMPUTATION",
            "reason": "at least one policy did not complete all 20 jobs",
        }
    shared = sorted(set(base["jcts"]) & set(b3["jcts"]))
    if len(shared) != 20:
        raise ValueError(f"complete comparison has {len(shared)} shared jobs")
    contributions = {job: base["jcts"][job] - b3["jcts"][job] for job in shared}
    high_job = max(shared, key=lambda job: abs(contributions[job]))
    remaining = [job for job in shared if job != high_job]
    base_mean = base["average_jct_completed_only_seconds"]
    b3_mean = b3["average_jct_completed_only_seconds"]
    delta = base_mean - b3_mean
    base_without = statistics.mean(base["jcts"][job] for job in remaining)
    b3_without = statistics.mean(b3["jcts"][job] for job in remaining)
    return {
        "status": "COMPLETE_PAIR",
        "average_jct_delta_seconds_base_minus_b3": delta,
        "average_jct_relative_change_percent": 100.0 * delta / base_mean,
        "median_jct_delta_seconds_base_minus_b3": (
            base["median_jct_completed_only_seconds"] - b3["median_jct_completed_only_seconds"]
        ),
        "p95_delta_seconds_base_minus_b3": (
            base["p95_jct_completed_only_seconds"] - b3["p95_jct_completed_only_seconds"]
        ),
        "p99_delta_seconds_base_minus_b3": (
            base["p99_jct_completed_only_seconds"] - b3["p99_jct_completed_only_seconds"]
        ),
        "makespan_delta_seconds_base_minus_b3": (
            base["makespan_completed_only_seconds"] - b3["makespan_completed_only_seconds"]
        ),
        "jobs_b3_better": sum(value > 0 for value in contributions.values()),
        "jobs_equal": sum(value == 0 for value in contributions.values()),
        "jobs_b3_worse": sum(value < 0 for value in contributions.values()),
        "per_job_jct_contribution_seconds": contributions,
        "largest_absolute_contribution_job": high_job,
        "largest_absolute_contribution_seconds": contributions[high_job],
        "without_largest_contribution": {
            "removed_job": high_job,
            "base_average_jct_seconds": base_without,
            "b3_average_jct_seconds": b3_without,
            "delta_seconds_base_minus_b3": base_without - b3_without,
            "relative_change_percent": 100.0 * (base_without - b3_without) / base_without,
            "diagnostic_only": True,
        },
        "deepspeech2_31_contribution_seconds": contributions.get("deepspeech2-31"),
    }


def distribution(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--neutral-aggregate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    roots = [Path(value).resolve() for value in args.inputs]
    target = Path(args.output).resolve()
    neutral_path = Path(args.neutral_aggregate).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite audit: {target}")
    neutral = load(neutral_path)
    if neutral.get("status") != "PASS":
        raise ValueError("Neutral aggregate must be PASS")

    cohorts = {}
    execution_heads = set()
    invariant_hashes = set()
    for root in roots:
        master_path = root / "run_manifest.json"
        master = load(master_path)
        if len(master.get("cohort_runs", [])) != 1 or master.get("response") != "medium":
            raise ValueError(f"not a single Medium cohort root: {root}")
        record = master["cohort_runs"][0]
        cohort_id = record["cohort_id"]
        cohort_root = Path(record["root"])
        validation_path = cohort_root / "validation_summary.json"
        validation = load(validation_path)
        runs = {item["policy"]: compact_metrics(item) for item in record["runs"]}
        if set(runs) != set(POLICIES):
            raise ValueError(f"policy set mismatch for {cohort_id}")
        comparisons = {name: compare(runs[name], runs["b3"]) for name in COMPARATORS}
        cohorts[cohort_id] = {
            "root": str(cohort_root),
            "master_status": master["status"],
            "master_manifest_sha256": sha256(master_path),
            "validation_status": validation["status"],
            "validation_sha256": sha256(validation_path),
            "failed_checks": validation.get("failures", []),
            "workload_sha256": record["workload_sha256"],
            "config_sha256": record["config_sha256"],
            "metrics": runs,
            "b3_comparisons": comparisons,
        }
        execution_heads.add(master["git_head"])
        invariant_hashes.add((master["preregistration_sha256"], master["frozen_selection_sha256"]))

    if set(cohorts) != set(COHORT_ORDER):
        raise ValueError(f"cohort set mismatch: {sorted(cohorts)}")
    if len(execution_heads) != 1 or len(invariant_hashes) != 1:
        raise ValueError("execution commits or frozen inputs differ")

    cross = {}
    for comparator in COMPARATORS:
        complete = [
            (cohort_id, cohorts[cohort_id]["b3_comparisons"][comparator])
            for cohort_id in COHORT_ORDER
            if cohorts[cohort_id]["b3_comparisons"][comparator]["status"] == "COMPLETE_PAIR"
        ]
        relative = [item[1]["average_jct_relative_change_percent"] for item in complete]
        absolute = [item[1]["average_jct_delta_seconds_base_minus_b3"] for item in complete]
        worst = min(complete, key=lambda item: item[1]["average_jct_relative_change_percent"])
        neutral_stats = neutral["cross_cohort"][comparator]
        cross[comparator] = {
            "complete_pair_cohort_count": len(complete),
            "censored_cohorts": [
                cohort_id for cohort_id in COHORT_ORDER
                if cohorts[cohort_id]["b3_comparisons"][comparator]["status"] != "COMPLETE_PAIR"
            ],
            "b3_win_count": sum(value > 0 for value in absolute),
            "tie_count": sum(value == 0 for value in absolute),
            "b3_loss_count": sum(value < 0 for value in absolute),
            "average_jct_delta_seconds": distribution(absolute),
            "average_jct_relative_change_percent": distribution(relative),
            "worst_complete_cohort": {
                "cohort_id": worst[0],
                "relative_change_percent": worst[1]["average_jct_relative_change_percent"],
                "delta_seconds": worst[1]["average_jct_delta_seconds_base_minus_b3"],
            },
            "median_improvement_change_vs_neutral_percentage_points": (
                statistics.median(relative)
                - neutral_stats["average_jct_relative_change_percent"]["median"]
            ),
        }

    equal_weight = {}
    neutral_vs_medium = {}
    for policy in POLICIES:
        complete_ids = [
            cohort_id for cohort_id in COHORT_ORDER
            if cohorts[cohort_id]["metrics"][policy]["all_jobs_completed"]
        ]
        medium_values = [
            cohorts[cohort_id]["metrics"][policy]["average_jct_completed_only_seconds"]
            for cohort_id in complete_ids
        ]
        neutral_values = [
            neutral["cohorts"][cohort_id]["metrics"][policy]["average_jct_seconds"]
            for cohort_id in complete_ids
        ]
        changes = [
            100.0 * (neutral_value - medium_value) / neutral_value
            for neutral_value, medium_value in zip(neutral_values, medium_values)
        ]
        equal_weight[policy] = {
            "complete_cohort_count": len(complete_ids),
            "excluded_censored_cohorts": sorted(set(COHORT_ORDER) - set(complete_ids)),
            "medium_mean_of_cohort_average_jct_seconds": statistics.mean(medium_values),
            "neutral_mean_over_same_cohorts_seconds": statistics.mean(neutral_values),
        }
        neutral_vs_medium[policy] = {
            "per_cohort_medium_improvement_percent": dict(zip(complete_ids, changes)),
            "median_percent": statistics.median(changes),
            "min_percent": min(changes),
            "max_percent": max(changes),
            "equal_weight_mean_jct_improvement_percent": 100.0 * (
                statistics.mean(neutral_values) - statistics.mean(medium_values)
            ) / statistics.mean(neutral_values),
        }

    conflicts = []
    for cohort_id in COHORT_ORDER:
        for comparator, item in cohorts[cohort_id]["b3_comparisons"].items():
            if item["status"] != "COMPLETE_PAIR" or item["average_jct_delta_seconds_base_minus_b3"] <= 0:
                continue
            worsened = [
                metric for metric, key in (
                    ("p95", "p95_delta_seconds_base_minus_b3"),
                    ("p99", "p99_delta_seconds_base_minus_b3"),
                    ("makespan", "makespan_delta_seconds_base_minus_b3"),
                ) if item[key] < 0
            ]
            if worsened:
                conflicts.append({
                    "cohort_id": cohort_id,
                    "comparator": comparator,
                    "b3_average_improves_but_worsens": worsened,
                })

    censored = [
        {"cohort_id": cohort_id, "policy": policy,
         "completed_jobs": cohorts[cohort_id]["metrics"][policy]["completed_jobs"],
         "rounds": cohorts[cohort_id]["metrics"][policy]["rounds"]}
        for cohort_id in COHORT_ORDER for policy in POLICIES
        if not cohorts[cohort_id]["metrics"][policy]["all_jobs_completed"]
    ]
    report = {
        "status": "FAIL_INCOMPLETE_MATRIX" if censored else "PASS",
        "scope": "MULTI_WORKLOAD_20_JOB_MEDIUM_SYNTHETIC_COMPARISON",
        "warning": "Synthetic cross-hardware transfer; P95/P99 use only 20 jobs; censored JCTs are never imputed.",
        "analysis_unit": "workload cohort",
        "execution_git_head": next(iter(execution_heads)),
        "neutral_aggregate": str(neutral_path),
        "neutral_aggregate_sha256": sha256(neutral_path),
        "censored_policy_runs": censored,
        "all_capacity_violation_count": sum(
            cohorts[c]["metrics"][p]["capacity_violation_count"]
            for c in COHORT_ORDER for p in POLICIES
        ),
        "cohorts": cohorts,
        "cross_cohort": cross,
        "equal_weight_cohort_average_jct": equal_weight,
        "neutral_vs_medium": neutral_vs_medium,
        "average_tail_makespan_conflicts": conflicts,
        "source_roots": [str(root) for root in roots],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "censored": censored,
        "cross_cohort": cross, "neutral_vs_medium": neutral_vs_medium,
        "output": str(target),
    }, indent=2))


if __name__ == "__main__":
    main()
