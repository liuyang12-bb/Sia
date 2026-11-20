"""Replace only w10_late with the uniformly extended five-policy follow-up."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import analyze_b3_multiworkload_medium_censored as audit


CORE_FILES = (
    "job.py",
    "run_cpu_bottleneck_smoke.py",
    "joint_prototype/joint_integration.py",
    "joint_prototype/joint_profile.py",
    "joint_prototype/joint_scheduler.py",
    "joint_prototype/synergy_adapter.py",
)


def write(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-censored-aggregate", required=True)
    parser.add_argument("--old-w10-root", required=True)
    parser.add_argument("--followup-root", required=True)
    parser.add_argument("--neutral-aggregate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    old_path = Path(args.old_censored_aggregate).resolve()
    old_w10_root = Path(args.old_w10_root).resolve()
    followup_root = Path(args.followup_root).resolve()
    neutral_path = Path(args.neutral_aggregate).resolve()
    target = Path(args.output).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite follow-up aggregate: {target}")
    old = audit.load(old_path)
    neutral = audit.load(neutral_path)
    frozen = audit.load(followup_root / "frozen_manifest.json")
    manifest = audit.load(followup_root / "run_manifest.json")
    validation = audit.load(followup_root / "validation_summary.json")
    if validation.get("status") != "PASS" or not validation.get("all_checks_passed"):
        raise ValueError("follow-up validation is not PASS")
    if manifest.get("status") != "COMPLETE" or set(x["policy"] for x in manifest["runs"]) != set(audit.POLICIES):
        raise ValueError("follow-up five-policy manifest is incomplete")

    old_master = audit.load(old_w10_root / "run_manifest.json")
    old_hashes = old_master["required_source_hashes"]
    for name in CORE_FILES:
        if old_hashes[name] != frozen["required_source_hashes"][name]:
            raise ValueError(f"algorithm/input code hash changed: {name}")
    old_w10 = old["cohorts"]["w10_late"]
    if old_w10["workload_sha256"] != frozen["workload_sha256"]:
        raise ValueError("w10_late workload changed")
    if frozen["config_differences"] != [
        {"path": "simulation.max_rounds", "old": 5000, "new": 6000}
    ]:
        raise ValueError("follow-up config has a non-window difference")

    new_metrics = {
        record["policy"]: audit.compact_metrics(record) for record in manifest["runs"]
    }
    if not all(metric["all_jobs_completed"] for metric in new_metrics.values()):
        raise ValueError("not all follow-up policies completed 20/20")
    historical_equality = {}
    for policy in audit.POLICIES:
        old_metric = old_w10["metrics"][policy]
        shared = sorted(set(old_metric["jcts"]) & set(new_metrics[policy]["jcts"]))
        historical_equality[policy] = {
            "historical_completed_jobs": old_metric["completed_jobs"],
            "followup_completed_jobs": new_metrics[policy]["completed_jobs"],
            "shared_job_count": len(shared),
            "shared_jcts_exactly_equal": all(
                old_metric["jcts"][job] == new_metrics[policy]["jcts"][job] for job in shared
            ),
            "all_historical_completion_times_equal": all(
                old_metric["completion_times"].get(job)
                == new_metrics[policy]["completion_times"].get(job)
                for job in old_metric["completion_times"]
            ),
        }
        if not historical_equality[policy]["shared_jcts_exactly_equal"]:
            raise ValueError(f"previously completed JCT changed for {policy}")

    cohorts = {
        cohort_id: old["cohorts"][cohort_id]
        for cohort_id in audit.COHORT_ORDER if cohort_id != "w10_late"
    }
    cohorts["w10_late"] = {
        "root": str(followup_root),
        "master_status": manifest["status"],
        "validation_status": validation["status"],
        "validation_sha256": audit.sha256(followup_root / "validation_summary.json"),
        "workload_sha256": frozen["workload_sha256"],
        "config_sha256": frozen["frozen_config_sha256"],
        "metrics": new_metrics,
        "b3_comparisons": {
            comparator: audit.compare(new_metrics[comparator], new_metrics["b3"])
            for comparator in audit.COMPARATORS
        },
        "observation_window_rounds": 6000,
    }

    cross = {}
    for comparator in audit.COMPARATORS:
        items = [
            (cohort_id, cohorts[cohort_id]["b3_comparisons"][comparator])
            for cohort_id in audit.COHORT_ORDER
        ]
        if any(item[1]["status"] != "COMPLETE_PAIR" for item in items):
            raise ValueError(f"incomplete comparison remains: {comparator}")
        relative = [item[1]["average_jct_relative_change_percent"] for item in items]
        absolute = [item[1]["average_jct_delta_seconds_base_minus_b3"] for item in items]
        worst = min(items, key=lambda item: item[1]["average_jct_relative_change_percent"])
        neutral_stats = neutral["cross_cohort"][comparator]
        cross[comparator] = {
            "cohort_count": 9,
            "b3_win_count": sum(value > 0 for value in absolute),
            "tie_count": sum(value == 0 for value in absolute),
            "b3_loss_count": sum(value < 0 for value in absolute),
            "average_jct_delta_seconds": audit.distribution(absolute),
            "average_jct_relative_change_percent": audit.distribution(relative),
            "worst_cohort": {
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
    for policy in audit.POLICIES:
        medium_values = [
            cohorts[c]["metrics"][policy]["average_jct_completed_only_seconds"]
            for c in audit.COHORT_ORDER
        ]
        neutral_values = [
            neutral["cohorts"][c]["metrics"][policy]["average_jct_seconds"]
            for c in audit.COHORT_ORDER
        ]
        changes = [
            100.0 * (n - m) / n for n, m in zip(neutral_values, medium_values)
        ]
        equal_weight[policy] = {
            "cohort_count": 9,
            "medium_mean_of_cohort_average_jct_seconds": statistics.mean(medium_values),
            "neutral_mean_of_cohort_average_jct_seconds": statistics.mean(neutral_values),
        }
        neutral_vs_medium[policy] = {
            "per_cohort_medium_improvement_percent": dict(zip(audit.COHORT_ORDER, changes)),
            "median_percent": statistics.median(changes),
            "min_percent": min(changes), "max_percent": max(changes),
            "equal_weight_mean_jct_improvement_percent": 100.0 * (
                statistics.mean(neutral_values) - statistics.mean(medium_values)
            ) / statistics.mean(neutral_values),
        }

    conflicts = []
    for cohort_id in audit.COHORT_ORDER:
        for comparator, item in cohorts[cohort_id]["b3_comparisons"].items():
            if item["average_jct_delta_seconds_base_minus_b3"] <= 0:
                continue
            worsened = [
                metric for metric, key in (
                    ("p95", "p95_delta_seconds_base_minus_b3"),
                    ("p99", "p99_delta_seconds_base_minus_b3"),
                    ("makespan", "makespan_delta_seconds_base_minus_b3"),
                ) if item[key] < 0
            ]
            if worsened:
                conflicts.append({"cohort_id": cohort_id, "comparator": comparator, "worsened": worsened})

    report = {
        "status": "PASS",
        "scope": "MULTI_WORKLOAD_20_JOB_MEDIUM_SYNTHETIC_COMPARISON_WITH_W10_LATE_6000_FOLLOWUP",
        "warning": "Synthetic cross-hardware transfer; P95/P99 use only 20 jobs.",
        "analysis_unit": "workload cohort",
        "mixed_observation_windows": {
            "eight_original_cohorts_max_rounds": 5000,
            "w10_late_followup_max_rounds": 6000,
            "interpretation": "All jobs in the eight original cohorts completed before 5000; only w10_late was uniformly rerun for all five policies at 6000.",
        },
        "old_censored_aggregate": str(old_path),
        "old_censored_aggregate_sha256": audit.sha256(old_path),
        "followup_root": str(followup_root),
        "followup_validation_sha256": audit.sha256(followup_root / "validation_summary.json"),
        "neutral_aggregate": str(neutral_path),
        "neutral_aggregate_sha256": audit.sha256(neutral_path),
        "execution_git_head": frozen["git_head"],
        "core_hash_equality_with_original_execution": True,
        "historical_completed_job_equality": historical_equality,
        "cohorts": cohorts,
        "cross_cohort": cross,
        "equal_weight_cohort_average_jct": equal_weight,
        "neutral_vs_medium": neutral_vs_medium,
        "average_tail_makespan_conflicts": conflicts,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    write(target, report)
    print(json.dumps({
        "status": report["status"],
        "w10_late": cohorts["w10_late"]["b3_comparisons"],
        "cross_cohort": cross,
        "output": str(target),
    }, indent=2))


if __name__ == "__main__":
    main()
