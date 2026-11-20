"""Aggregate Medium results and compare them with the frozen Neutral run."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path

import analyze_b3_multiworkload_neutral as common


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--neutral-aggregate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.input).resolve()
    neutral_path = Path(args.neutral_aggregate).resolve()
    target = Path(args.output).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite aggregate: {target}")
    master = common.load(root / "run_manifest.json")
    neutral = common.load(neutral_path)
    if master.get("status") != "COMPLETE":
        raise RuntimeError(f"Medium experiment is incomplete: {master.get('status')}")
    if neutral.get("status") != "PASS":
        raise RuntimeError("Neutral aggregate is not PASS")

    cohorts = {}
    all_validation_pass = True
    for cohort_record in master["cohort_runs"]:
        cohort_id = cohort_record["cohort_id"]
        cohort_root = Path(cohort_record["root"])
        validation_path = cohort_root / "validation_summary.json"
        validation = common.load(validation_path)
        all_validation_pass &= bool(validation.get("all_checks_passed"))
        manifest = common.load(cohort_root / "cohort_manifest.json")
        metrics = {}
        for policy in common.POLICIES:
            payload_path = cohort_root / "policies" / policy / "result.json"
            payload = common.load(payload_path)
            metrics[policy] = common.policy_metrics(payload, payload_path)
            del payload
            gc.collect()
        comparisons = {
            comparator: common.compare(metrics[comparator], metrics["b3"])
            for comparator in common.COMPARATORS
        }
        cohorts[cohort_id] = {
            "trace": manifest["trace"],
            "cohort_label": manifest["cohort_label"],
            "job_names": manifest["job_names"],
            "workload_sha256": manifest["workload_sha256"],
            "derived_config_sha256": manifest["derived_config_sha256"],
            "validation_status": validation["status"],
            "validation_sha256": common.sha256(validation_path),
            "metrics": metrics,
            "b3_comparisons": comparisons,
        }

    if set(cohorts) != set(neutral["cohorts"]):
        raise ValueError("Medium and Neutral cohort sets differ")

    cross = {}
    for comparator in common.COMPARATORS:
        items = [
            (cohort_id, cohort["b3_comparisons"][comparator])
            for cohort_id, cohort in cohorts.items()
        ]
        relative = [item[1]["average_jct_relative_change_percent"] for item in items]
        absolute = [item[1]["average_jct_delta_seconds_base_minus_b3"] for item in items]
        worst = min(items, key=lambda item: item[1]["average_jct_relative_change_percent"])
        medium_stats = {
            "cohort_count": len(items),
            "b3_win_count": sum(value > 0 for value in absolute),
            "tie_count": sum(value == 0 for value in absolute),
            "b3_loss_count": sum(value < 0 for value in absolute),
            "average_jct_delta_seconds": common.distribution(absolute),
            "average_jct_relative_change_percent": common.distribution(relative),
            "worst_cohort": {
                "cohort_id": worst[0],
                "relative_change_percent": worst[1]["average_jct_relative_change_percent"],
                "delta_seconds": worst[1]["average_jct_delta_seconds_base_minus_b3"],
            },
        }
        neutral_stats = neutral["cross_cohort"][comparator]
        medium_stats["median_improvement_change_vs_neutral_percentage_points"] = (
            medium_stats["average_jct_relative_change_percent"]["median"]
            - neutral_stats["average_jct_relative_change_percent"]["median"]
        )
        cross[comparator] = medium_stats

    equal_weight = {}
    neutral_vs_medium = {}
    for policy in common.POLICIES:
        medium_values = [
            cohorts[cohort_id]["metrics"][policy]["average_jct_seconds"]
            for cohort_id in sorted(cohorts)
        ]
        neutral_values = [
            neutral["cohorts"][cohort_id]["metrics"][policy]["average_jct_seconds"]
            for cohort_id in sorted(cohorts)
        ]
        changes = [
            100.0 * (neutral_value - medium_value) / neutral_value
            for neutral_value, medium_value in zip(neutral_values, medium_values)
        ]
        equal_weight[policy] = {
            "medium_mean_of_cohort_average_jct_seconds": statistics.mean(medium_values),
            "neutral_mean_of_cohort_average_jct_seconds": statistics.mean(neutral_values),
        }
        neutral_vs_medium[policy] = {
            "per_cohort_medium_improvement_percent": dict(zip(sorted(cohorts), changes)),
            "median_percent": statistics.median(changes),
            "min_percent": min(changes),
            "max_percent": max(changes),
            "equal_weight_mean_jct_improvement_percent": 100.0 * (
                statistics.mean(neutral_values) - statistics.mean(medium_values)
            ) / statistics.mean(neutral_values),
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
        "scope": "MULTI_WORKLOAD_20_JOB_MEDIUM_SYNTHETIC_COMPARISON",
        "warning": "Synthetic cross-hardware transfer; each P95/P99 uses only 20 jobs.",
        "analysis_unit": "workload cohort",
        "experiment_root": str(root),
        "experiment_manifest_sha256": common.sha256(root / "run_manifest.json"),
        "neutral_aggregate": str(neutral_path),
        "neutral_aggregate_sha256": common.sha256(neutral_path),
        "git_head": master["git_head"],
        "frozen_selection_sha256": master["frozen_selection_sha256"],
        "all_nine_cohorts_validation_pass": all_validation_pass,
        "cohorts": cohorts,
        "cross_cohort": cross,
        "equal_weight_cohort_average_jct": equal_weight,
        "neutral_vs_medium": neutral_vs_medium,
        "average_tail_makespan_conflicts": conflicts,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "cohorts": len(cohorts),
        "all_validation_pass": all_validation_pass,
        "cross_cohort": cross,
        "neutral_vs_medium": neutral_vs_medium,
        "output": str(target),
    }, indent=2))


if __name__ == "__main__":
    main()
