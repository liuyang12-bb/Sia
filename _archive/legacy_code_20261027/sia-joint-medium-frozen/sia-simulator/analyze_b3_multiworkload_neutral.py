"""Aggregate the frozen multi-workload Neutral experiment artifacts."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path


POLICIES = ("b1", "sy", "b4", "b5", "b3")
COMPARATORS = ("b1", "sy", "b4", "b5")


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


def true_fallback(status):
    if not status:
        return False
    return (
        "FALLBACK" in status
        or "ALL_WAIT" in status
        or status == "NO_COLUMNS"
        or status.startswith("MIP_NO")
    )


def policy_metrics(payload, payload_path):
    summary = payload["summary"]
    jcts = {key: float(value) for key, value in summary["jcts"].items()}
    statuses = Counter()
    fallback_statuses = Counter()
    audit_reason_rounds = 0
    checkpoint_seconds = 0.0
    unallocated_job_rounds = 0
    for record in payload["joint_round_logs"]:
        result = record["decision_for_next_interval"].get("result", {})
        status = result.get("status")
        if status:
            statuses[status] += 1
            if true_fallback(status):
                fallback_statuses[status] += 1
        if result.get("audit", {}).get("fallback_reason"):
            audit_reason_rounds += 1
        unallocated_job_rounds += len(result.get("unallocated_jobs", []))
        checkpoint_seconds += sum(
            float(event["checkpoint_delay_consumed"])
            for event in record["execution"]
        )
    jobs = summary["jobs"]
    completion_times = {
        key: float(value) for key, value in summary["completion_times"].items()
    }
    return {
        "completed_jobs": int(summary["completed_jobs"]),
        "total_jobs": int(summary["total_jobs"]),
        "completion_rate": float(summary["completion_rate"]),
        "average_jct_seconds": float(summary["average_jct_completed_only"]),
        "median_jct_seconds": statistics.median(jcts.values()),
        "p95_jct_seconds": percentile(jcts.values(), 0.95),
        "p99_jct_seconds": percentile(jcts.values(), 0.99),
        "makespan_seconds": max(completion_times.values()),
        "jcts": jcts,
        "completion_times": completion_times,
        "waiting_seconds": summary["waiting_seconds"],
        "total_waiting_seconds": sum(float(x) for x in summary["waiting_seconds"].values()),
        "resource_hours": summary["resource_hours"],
        "restarts": sum(int(job["num_restarts"]) for job in jobs.values()),
        "wasted_gpu_seconds": sum(float(job["wasted_gpu_seconds"]) for job in jobs.values()),
        "checkpoint_restore_seconds": checkpoint_seconds,
        "cpu_saturated_gpu_idle_event_count": int(
            payload["contention_evidence"]["cpu_saturated_gpu_idle_event_count"]
        ),
        "wait_reason_rounds": payload["contention_evidence"]["wait_reason_rounds"],
        "unallocated_job_rounds": unallocated_job_rounds,
        "solve_status_counts": dict(statuses),
        "true_fallback_status_counts": dict(fallback_statuses),
        "audit_reason_rounds": audit_reason_rounds,
        "optimizer_total_seconds": float(summary["optimizer_total_seconds"]),
        "optimizer_p99_seconds": float(summary["optimizer_p99_seconds"]),
        "rounds": int(summary["rounds"]),
        "simulated_seconds": float(summary["simulated_seconds"]),
        "wall_seconds": float(summary["wall_seconds"]),
        "min_rho": float(summary["min_rho"]),
        "max_rho": float(summary["max_rho"]),
        "capacity_violations": summary["capacity_violations"],
        "result_sha256": sha256(payload_path),
    }


def compare(base, b3):
    shared = sorted(set(base["jcts"]) & set(b3["jcts"]))
    if len(shared) != 20:
        raise ValueError(f"pairwise comparison has {len(shared)} shared jobs")
    contributions = {
        job: float(base["jcts"][job] - b3["jcts"][job]) for job in shared
    }
    high_job = max(shared, key=lambda job: abs(contributions[job]))
    remaining = [job for job in shared if job != high_job]
    base_without = statistics.mean(base["jcts"][job] for job in remaining)
    b3_without = statistics.mean(b3["jcts"][job] for job in remaining)
    delta = base["average_jct_seconds"] - b3["average_jct_seconds"]
    return {
        "average_jct_delta_seconds_base_minus_b3": delta,
        "average_jct_relative_change_percent": 100.0 * delta / base["average_jct_seconds"],
        "median_jct_delta_seconds_base_minus_b3": base["median_jct_seconds"] - b3["median_jct_seconds"],
        "p95_delta_seconds_base_minus_b3": base["p95_jct_seconds"] - b3["p95_jct_seconds"],
        "p99_delta_seconds_base_minus_b3": base["p99_jct_seconds"] - b3["p99_jct_seconds"],
        "makespan_delta_seconds_base_minus_b3": base["makespan_seconds"] - b3["makespan_seconds"],
        "jobs_b3_better": sum(contributions[job] > 0 for job in shared),
        "jobs_equal": sum(contributions[job] == 0 for job in shared),
        "jobs_b3_worse": sum(contributions[job] < 0 for job in shared),
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
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.input).resolve()
    target = Path(args.output).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite aggregate: {target}")
    master = load(root / "run_manifest.json")
    if master.get("status") != "COMPLETE":
        raise RuntimeError(f"experiment is incomplete: {master.get('status')}")

    cohorts = {}
    all_validation_pass = True
    for cohort_record in master["cohort_runs"]:
        cohort_id = cohort_record["cohort_id"]
        cohort_root = Path(cohort_record["root"])
        validation_path = cohort_root / "validation_summary.json"
        validation = load(validation_path)
        all_validation_pass &= bool(validation.get("all_checks_passed"))
        manifest = load(cohort_root / "cohort_manifest.json")
        metrics = {}
        for policy in POLICIES:
            payload_path = cohort_root / "policies" / policy / "result.json"
            payload = load(payload_path)
            metrics[policy] = policy_metrics(payload, payload_path)
            del payload
            gc.collect()
        comparisons = {
            comparator: compare(metrics[comparator], metrics["b3"])
            for comparator in COMPARATORS
        }
        cohorts[cohort_id] = {
            "trace": manifest["trace"],
            "cohort_label": manifest["cohort_label"],
            "job_names": manifest["job_names"],
            "workload_sha256": manifest["workload_sha256"],
            "derived_config_sha256": manifest["derived_config_sha256"],
            "validation_status": validation["status"],
            "validation_sha256": sha256(validation_path),
            "metrics": metrics,
            "b3_comparisons": comparisons,
        }

    cross = {}
    for comparator in COMPARATORS:
        items = [
            (cohort_id, cohort["b3_comparisons"][comparator])
            for cohort_id, cohort in cohorts.items()
        ]
        relative = [item[1]["average_jct_relative_change_percent"] for item in items]
        absolute = [item[1]["average_jct_delta_seconds_base_minus_b3"] for item in items]
        worst = min(items, key=lambda item: item[1]["average_jct_relative_change_percent"])
        cross[comparator] = {
            "cohort_count": len(items),
            "b3_win_count": sum(value > 0 for value in absolute),
            "tie_count": sum(value == 0 for value in absolute),
            "b3_loss_count": sum(value < 0 for value in absolute),
            "average_jct_delta_seconds": distribution(absolute),
            "average_jct_relative_change_percent": distribution(relative),
            "worst_cohort": {
                "cohort_id": worst[0],
                "relative_change_percent": worst[1]["average_jct_relative_change_percent"],
                "delta_seconds": worst[1]["average_jct_delta_seconds_base_minus_b3"],
            },
        }

    conflicts = []
    for cohort_id, cohort in cohorts.items():
        for comparator, item in cohort["b3_comparisons"].items():
            if item["average_jct_delta_seconds_base_minus_b3"] > 0:
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

    report = {
        "status": "PASS" if all_validation_pass and len(cohorts) == 9 else "FAIL",
        "scope": "MULTI_WORKLOAD_20_JOB_NEUTRAL_SYNTHETIC_COMPARISON",
        "warning": "Each P95/P99 is computed from only 20 jobs and is not a high-precision population tail estimate.",
        "analysis_unit": "workload cohort; jobs within one cohort are not independent repetitions",
        "experiment_root": str(root),
        "experiment_manifest_sha256": sha256(root / "run_manifest.json"),
        "git_head": master["git_head"],
        "frozen_selection_sha256": master["frozen_selection_sha256"],
        "all_nine_cohorts_validation_pass": all_validation_pass,
        "cohorts": cohorts,
        "cross_cohort": cross,
        "average_tail_makespan_conflicts": conflicts,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "cohorts": len(cohorts),
        "all_validation_pass": all_validation_pass,
        "cross_cohort": cross,
        "output": str(target),
    }, indent=2))


if __name__ == "__main__":
    main()
