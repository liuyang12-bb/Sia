"""Validate the isolated post-repair B3 20-job run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    root = Path(args.input).resolve()
    manifest = load(root / "run_manifest.json")
    payload = load(root / "cpu_constrained.neutral.b3.json")
    summary = payload["summary"]
    evidence = payload["contention_evidence"]
    failures = []
    details = {}

    def check(name, condition, detail=None):
        if not condition:
            failures.append({"check": name, "detail": detail})
        details[name] = bool(condition)

    check("manifest_complete", manifest["status"] == "COMPLETE", manifest["status"])
    check("only_targeted_b3", manifest["frozen_matrix"]["policies"] == ["b3"])
    check("all_20_jobs_complete", summary["completed_jobs"] == 20 and summary["all_jobs_completed"])
    check("no_round_limit", not summary["hit_round_limit"])
    check("no_capacity_violations", not summary["capacity_violations"], summary["capacity_violations"])
    check("neutral_rho_exactly_one", summary["min_rho"] == 1.0 and summary["max_rho"] == 1.0)
    check("multiplier_applied_once", evidence["single_multiplier_max_abs_residual"] < 1e-9)
    util_errors = []
    checkpoint_errors = []
    status_counts = {}
    fallback_counts = {}
    candidate_max = 0
    repaired_failure_round = None
    for record in payload["joint_round_logs"]:
        duration = float(record["interval_end"] - record["interval_start"])
        decision = record["decision_for_next_interval"]
        result = decision.get("result", {})
        status = result.get("status")
        if status:
            status_counts[status] = status_counts.get(status, 0) + 1
            if "FALLBACK" in status or "ALL_WAIT" in status:
                fallback_counts[status] = fallback_counts.get(status, 0) + 1
        for item in decision.get("candidate_summary", {}).values():
            candidate_max = max(candidate_max, int(item["count"]))
        if float(record["interval_end"]) == 32880.0:
            candidates = decision.get("candidate_summary", {})
            repaired_failure_round = {
                "result_status": status,
                "candidate_summary": candidates,
                "candidate_columns": sum(int(item["count"]) for item in candidates.values()),
                "selected_configs": {
                    job_id: item.get("config", {}).get("config_id")
                    for job_id, item in result.get("selected", {}).items()
                },
            }
        for event in record["execution"]:
            checkpoint = float(event["checkpoint_delay_consumed"])
            training = float(event["training_seconds"])
            if checkpoint + training > duration + 1e-9:
                checkpoint_errors.append((event["job_id"], record["interval_start"], "time_overlap"))
            if checkpoint >= duration - 1e-9:
                if training > 1e-9 or abs(float(event["progress_delta"])) > 1e-9:
                    checkpoint_errors.append((event["job_id"], record["interval_start"], "progress_during_full_delay"))
                old = event.get("resource_allocation")
                new = result.get("selected", {}).get(event["job_id"])
                if old and new:
                    if old["config_id"] != new["config"]["config_id"] or old["placement"] != new["placement"]:
                        checkpoint_errors.append((event["job_id"], record["interval_start"], "locked_config_changed"))
    for node_id, util in evidence["node_time_weighted_utilization"].items():
        for resource in ("gpu_utilization", "cpu_utilization", "ram_utilization"):
            if util[resource] > 1.0 + 1e-9:
                util_errors.append((node_id, resource, util[resource]))
    check("time_weighted_capacity", not util_errors, util_errors)
    check("checkpoint_lock_and_delay_semantics", not checkpoint_errors, checkpoint_errors[:20])
    check("multiple_gpu_candidates_considered", candidate_max > 1, candidate_max)
    check(
        "former_failure_time_has_candidate",
        repaired_failure_round is not None
        and repaired_failure_round["result_status"] == "OPTIMAL"
        and repaired_failure_round["candidate_columns"] > 0,
        repaired_failure_round,
    )
    check("no_failure_or_fallback_status", not fallback_counts and all(not key.startswith("MIP_NO") and key != "NO_COLUMNS" for key in status_counts), {"statuses": status_counts, "fallbacks": fallback_counts})

    result = {
        "status": "PASS" if not failures else "FAIL",
        "checks": details,
        "failures": failures,
        "summary": {
            "completed_jobs": summary["completed_jobs"],
            "average_jct_seconds": summary["average_jct_completed_only"],
            "rounds": summary["rounds"],
            "wall_seconds": summary["wall_seconds"],
            "rho_range": [summary["min_rho"], summary["max_rho"]],
            "capacity_violations": summary["capacity_violations"],
            "status_counts": status_counts,
            "fallback_status_counts": fallback_counts,
            "former_failure_round": repaired_failure_round,
        },
    }
    (root / "targeted_validation_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
