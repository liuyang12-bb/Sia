"""Validate the pre-registered four-job contention artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


POLICIES = ("b1", "sy", "b4", "b5", "b3")
CAPACITIES = ("A_cpu_sufficient", "B_cpu_constrained", "C_mixed_pressure")
RESPONSES = ("neutral", "medium")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--previous", required=True)
    args = parser.parse_args()
    root = Path(args.input).resolve()
    previous = Path(args.previous).resolve()
    manifest = load(root / "run_manifest.json")
    payloads = {}
    for record in manifest["runs"]:
        key = (record["capacity"], record["response"], record["policy"])
        payloads[key] = load(root / record["file"])

    failures = []

    def check(name, condition, detail=None):
        if not condition:
            failures.append({"check": name, "detail": detail})
        return bool(condition)

    expected = {(c, r, p) for c in CAPACITIES for r in RESPONSES for p in POLICIES}
    checks = {}
    checks["full_factorial_30_runs"] = check(
        "full_factorial_30_runs", set(payloads) == expected, sorted(set(expected) - set(payloads))
    )
    checks["all_four_jobs_complete"] = check(
        "all_four_jobs_complete",
        all(item["summary"]["completed_jobs"] == 4 for item in payloads.values()),
    )
    checks["no_capacity_violations"] = check(
        "no_capacity_violations",
        all(not item["summary"]["capacity_violations"] for item in payloads.values()),
    )
    checks["neutral_rho_exactly_one"] = check(
        "neutral_rho_exactly_one",
        all(
            item["summary"]["min_rho"] == 1.0 and item["summary"]["max_rho"] == 1.0
            for (capacity, response, policy), item in payloads.items()
            if response == "neutral"
        ),
    )
    checks["goodput_multiplier_applied_once"] = check(
        "goodput_multiplier_applied_once",
        all(
            item["contention_evidence"]["single_multiplier_max_abs_residual"] < 1e-9
            for item in payloads.values()
        ),
    )

    utilization_errors = []
    fixed_errors = []
    synergy_errors = []
    b3_candidate_counts = []
    checkpoint_errors = []
    tune_actions = []
    for key, payload in payloads.items():
        for node_id, util in payload["contention_evidence"]["node_time_weighted_utilization"].items():
            for resource in ("gpu_utilization", "cpu_utilization", "ram_utilization"):
                if util[resource] > 1.0 + 1e-9:
                    utilization_errors.append((key, node_id, resource, util[resource]))
        for record in payload["joint_round_logs"]:
            decision = record["decision_for_next_interval"]
            result = decision.get("result", {})
            audit = result.get("audit", {})
            selected = result.get("selected", {})
            fixed = decision.get("sia_fixed_gpu_configs", {})
            if key[2] == "b5":
                for job_id, item in selected.items():
                    if item["config"]["config_id"] != fixed.get(job_id):
                        fixed_errors.append((key, decision.get("round"), job_id))
            if key[2] == "sy" and result:
                if audit.get("official_core") != "SYNERGY_TUNE":
                    synergy_errors.append((key, decision.get("round")))
            if key[2] in {"sy", "b5"}:
                tune_actions.extend(item.get("action") for item in audit.get("tune_trace", []))
            if key[2] == "b3":
                b3_candidate_counts.extend(
                    value["count"] for value in decision.get("candidate_summary", {}).values()
                )
            for event in record["execution"]:
                if (
                    event["checkpoint_delay_consumed"] >= 60.0
                    and event["training_seconds"] == 0.0
                    and abs(event["progress_delta"]) > 1e-9
                ):
                    checkpoint_errors.append((key, event["job_id"], record["interval_start"]))
    checks["time_weighted_utilization_within_capacity"] = check(
        "time_weighted_utilization_within_capacity", not utilization_errors, utilization_errors
    )
    checks["b5_gpu_config_frozen"] = check("b5_gpu_config_frozen", not fixed_errors, fixed_errors)
    checks["sy_uses_tune_core"] = check("sy_uses_tune_core", not synergy_errors, synergy_errors)
    checks["b3_has_multiple_gpu_candidates"] = check(
        "b3_has_multiple_gpu_candidates",
        bool(b3_candidate_counts) and max(b3_candidate_counts) > 1,
        max(b3_candidate_counts) if b3_candidate_counts else None,
    )
    checks["checkpoint_delay_has_no_progress"] = check(
        "checkpoint_delay_has_no_progress", not checkpoint_errors, checkpoint_errors
    )
    checks["tune_resource_downgrade_observed"] = check(
        "tune_resource_downgrade_observed",
        "DOWNGRADE_CURRENT_TO_FAIR" in tune_actions
        or "ATOMIC_PEER_DOWNGRADE_TO_FAIR" in tune_actions,
        sorted(set(tune_actions)),
    )

    pressure_payloads = {
        key: value for key, value in payloads.items() if key[0] != "A_cpu_sufficient"
    }
    actual_wait_seconds = sum(
        sum(item["summary"]["waiting_seconds"].values()) for item in pressure_payloads.values()
    )
    cpu_idle_events = sum(
        item["contention_evidence"]["cpu_saturated_gpu_idle_event_count"]
        for item in pressure_payloads.values()
    )
    cpu_wait_reason_rounds = sum(
        item["contention_evidence"]["wait_reason_rounds"].get(
            "CPU_SATURATED_WITH_IDLE_GPU", 0
        )
        for item in pressure_payloads.values()
    )
    checks["actual_waiting_observed"] = check(
        "actual_waiting_observed", actual_wait_seconds > 0, actual_wait_seconds
    )
    checks["runtime_cpu_saturation_with_idle_gpu_observed"] = check(
        "runtime_cpu_saturation_with_idle_gpu_observed",
        cpu_idle_events > 0,
        cpu_idle_events,
    )
    checks["unallocated_job_coincides_with_cpu_idle_gpu"] = check(
        "unallocated_job_coincides_with_cpu_idle_gpu",
        cpu_wait_reason_rounds > 0,
        cpu_wait_reason_rounds,
    )

    sensitive_jobs = ("cifar10-29", "deepspeech2-30")
    response_effect = {}
    for capacity in CAPACITIES:
        for policy in POLICIES:
            neutral = payloads[(capacity, "neutral", policy)]["summary"]
            medium = payloads[(capacity, "medium", policy)]["summary"]
            response_effect[f"{capacity}.{policy}"] = {
                job_id: {
                    "neutral_jct": neutral["jcts"][job_id],
                    "medium_jct": medium["jcts"][job_id],
                    "medium_rho": medium["rho_range_by_job"][job_id],
                }
                for job_id in sensitive_jobs
            }
    checks["medium_changes_sensitive_execution"] = check(
        "medium_changes_sensitive_execution",
        any(
            entry["neutral_jct"] != entry["medium_jct"]
            for group in response_effect.values()
            for entry in group.values()
        ),
        response_effect,
    )

    official = previous / "official_sia_regression.log"
    experimental = previous / "experimental_sia_regression.log"
    regression_hashes = {
        "official": sha256(official) if official.is_file() else None,
        "experimental": sha256(experimental) if experimental.is_file() else None,
    }
    checks["original_sia_regression_preserved"] = check(
        "original_sia_regression_preserved",
        regression_hashes["official"] is not None
        and regression_hashes["official"] == regression_hashes["experimental"],
        regression_hashes,
    )

    table = []
    for key in sorted(payloads):
        summary = payloads[key]["summary"]
        evidence = payloads[key]["contention_evidence"]
        checkpoint_delay_seconds = sum(
            float(event["checkpoint_delay_consumed"])
            for record in payloads[key]["joint_round_logs"]
            for event in record["execution"]
        )
        solve_statuses = sorted(
            {
                record["decision_for_next_interval"].get("result", {}).get("status")
                for record in payloads[key]["joint_round_logs"]
                if record["decision_for_next_interval"].get("result", {}).get("status")
            }
        )
        table.append(
            {
                "capacity": key[0],
                "response": key[1],
                "policy": key[2],
                "completed": summary["completed_jobs"],
                "average_jct": summary["average_jct_completed_only"],
                "jcts": summary["jcts"],
                "waiting_seconds": summary["waiting_seconds"],
                "total_waiting_seconds": sum(summary["waiting_seconds"].values()),
                "rho_range": [summary["min_rho"], summary["max_rho"]],
                "resource_hours": summary["resource_hours"],
                "restarts": sum(job["num_restarts"] for job in summary["jobs"].values()),
                "migrations": sum(job["num_migrations"] for job in summary["jobs"].values()),
                "wasted_gpu_seconds": sum(
                    job["wasted_gpu_seconds"] for job in summary["jobs"].values()
                ),
                "checkpoint_delay_seconds": checkpoint_delay_seconds,
                "gpu_config_changes": summary["gpu_config_changes"],
                "tune_action_counts": summary["tune_action_counts"],
                "solve_statuses": solve_statuses,
                "optimizer_total_seconds": summary["optimizer_total_seconds"],
                "optimizer_p99_seconds": summary["optimizer_p99_seconds"],
                "fallback_count": len(summary["fallback_events"]),
                "cpu_saturated_gpu_idle_event_count": evidence[
                    "cpu_saturated_gpu_idle_event_count"
                ],
                "wait_reason_rounds": evidence["wait_reason_rounds"],
            }
        )
    comparisons = {}
    for capacity in CAPACITIES:
        for response in RESPONSES:
            b3 = payloads[(capacity, response, "b3")]["summary"][
                "average_jct_completed_only"
            ]
            comparisons[f"{capacity}.{response}"] = {
                f"b3_minus_{policy}": b3
                - payloads[(capacity, response, policy)]["summary"][
                    "average_jct_completed_only"
                ]
                for policy in ("b1", "sy", "b4", "b5")
            }

    comparison = {
        "label": manifest["label"],
        "table": table,
        "b3_pairwise_average_jct_seconds": comparisons,
        "response_effect_on_sensitive_jobs": response_effect,
        "runtime_contention_totals": {
            "pressure_wait_seconds": actual_wait_seconds,
            "cpu_saturated_gpu_idle_events": cpu_idle_events,
            "unallocated_job_cpu_idle_gpu_rounds": cpu_wait_reason_rounds,
        },
    }
    validation = {
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "all_checks_passed": not failures,
        "failures": failures,
        "original_sia_regression_sha256": regression_hashes,
    }
    (root / "comparison_summary.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    (root / "validation_summary.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    print(json.dumps(validation, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
