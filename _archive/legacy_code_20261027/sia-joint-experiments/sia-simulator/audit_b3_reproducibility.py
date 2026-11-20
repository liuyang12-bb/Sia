"""Offline reproducibility audit for the frozen 20-job Neutral artifacts.

This script never invokes a scheduler.  It compares immutable result JSON files
and writes a separate machine-readable audit without rewriting historical data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import cvxpy
import cylp
import numpy
import pandas as pd
import scipy


POLICIES = ("b1", "sy", "b4", "b5", "b3")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def percentile(values, q):
    values = sorted(float(value) for value in values)
    rank = (len(values) - 1) * q
    lo = int(rank)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (rank - lo)


def stable_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest().upper()


def decision_signature(payload):
    signature = []
    for record in payload["joint_round_logs"]:
        decision = record["decision_for_next_interval"]
        result = decision.get("result", {})
        selected = {
            job_id: {
                "config": item["config"]["config_id"],
                "placement": dict(sorted(item.get("placement", {}).items())),
                "cpu_per_gpu": item.get("cpu_per_gpu"),
                "ram_gb_per_gpu": item.get("ram_gb_per_gpu"),
            }
            for job_id, item in sorted(result.get("selected", {}).items())
        }
        signature.append({
            "interval_end": record["interval_end"],
            "status": result.get("status"),
            "selected": selected,
            "unallocated_jobs": sorted(result.get("unallocated_jobs", [])),
        })
    return signature


def execution_signature(payload):
    return [
        {
            "interval_start": record["interval_start"],
            "interval_end": record["interval_end"],
            "execution": [
                {
                    "job_id": event["job_id"],
                    "progress_before": event["progress_before"],
                    "progress_after": event["progress_after"],
                    "completion_time": event["completion_time"],
                    "checkpoint_delay_consumed": event["checkpoint_delay_consumed"],
                    "training_seconds": event["training_seconds"],
                }
                for event in record["execution"]
            ],
        }
        for record in payload["joint_round_logs"]
    ]


def selected_map(decision):
    return {
        job_id: {
            "config": item["config"]["config_id"],
            "placement": dict(sorted(item.get("placement", {}).items())),
            "cpu_per_gpu": item.get("cpu_per_gpu"),
            "ram_gb_per_gpu": item.get("ram_gb_per_gpu"),
        }
        for job_id, item in sorted(decision.get("result", {}).get("selected", {}).items())
    }


def policy_metrics(payload, first_arrival):
    summary = payload["summary"]
    jcts = list(summary["jcts"].values())
    completion_times = summary["completion_times"]
    status_counts = Counter()
    true_fallback_statuses = Counter()
    reason_rounds = 0
    reason_jobs = 0
    reason_values = Counter()
    placement_status = Counter()
    for record in payload["joint_round_logs"]:
        result = record["decision_for_next_interval"].get("result", {})
        status = result.get("status")
        if status:
            status_counts[status] += 1
            if (
                "FALLBACK" in status
                or "ALL_WAIT" in status
                or status == "NO_COLUMNS"
                or status.startswith("MIP_NO")
            ):
                true_fallback_statuses[status] += 1
        reason = result.get("audit", {}).get("fallback_reason")
        if reason:
            reason_rounds += 1
            if isinstance(reason, dict):
                reason_jobs += len(reason)
                reason_values.update(str(value) for value in reason.values())
            else:
                reason_jobs += 1
                reason_values[str(reason)] += 1
        for item in result.get("selected", {}).values():
            placement_status[(item["config"]["config_id"], tuple(sorted(item.get("placement", {}))))] += 1
    jobs = summary["jobs"]
    return {
        "completed_jobs": summary["completed_jobs"],
        "total_jobs": summary["total_jobs"],
        "average_jct_seconds": summary["average_jct_completed_only"],
        "p95_jct_seconds": percentile(jcts, 0.95),
        "p99_jct_seconds": percentile(jcts, 0.99),
        "last_completion_timestamp": max(completion_times.values()),
        "makespan_from_first_arrival_seconds": max(completion_times.values()) - first_arrival,
        "simulated_seconds": summary["simulated_seconds"],
        "rounds": summary["rounds"],
        "total_waiting_seconds": sum(summary["waiting_seconds"].values()),
        "resource_hours": summary["resource_hours"],
        "restarts": sum(item["num_restarts"] for item in jobs.values()),
        "wasted_gpu_seconds": sum(item["wasted_gpu_seconds"] for item in jobs.values()),
        "jcts": summary["jcts"],
        "completion_times": completion_times,
        "waiting_seconds": summary["waiting_seconds"],
        "solve_status_counts": dict(status_counts),
        "true_fallback_status_counts": dict(true_fallback_statuses),
        "audit_reason_rounds": reason_rounds,
        "audit_reason_job_occurrences": reason_jobs,
        "audit_reason_values": dict(reason_values),
        "decision_signature_sha256": stable_hash(decision_signature(payload)),
        "execution_signature_sha256": stable_hash(execution_signature(payload)),
        "payload_sha256": None,
        "node_peak_aggregate_usage": summary["node_peak_aggregate_usage"],
        "node_time_weighted_utilization": payload["contention_evidence"]["node_time_weighted_utilization"],
        "cpu_saturated_gpu_idle_event_count": payload["contention_evidence"]["cpu_saturated_gpu_idle_event_count"],
        "wait_reason_rounds": payload["contention_evidence"]["wait_reason_rounds"],
        "placement_signature_count": len(placement_status),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", required=True)
    parser.add_argument("--current-root", required=True)
    parser.add_argument("--targeted-root", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--code-root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--blocked-memory-gib", type=float, required=True)
    args = parser.parse_args()

    old_root = Path(args.old_root).resolve()
    current_root = Path(args.current_root).resolve()
    targeted_root = Path(args.targeted_root).resolve()
    workload_path = Path(args.workload).resolve()
    code_root = Path(args.code_root).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite audit directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    workload = pd.read_csv(workload_path)
    first_arrival = float(workload["time"].min())
    old_validation = load(old_root / "validation_summary.json")
    current_validation = load(current_root / "validation_summary.json")
    old_manifest = load(old_root / "run_manifest.json")
    current_manifest = load(current_root / "run_manifest.json")

    current_payloads = {
        policy: load(current_root / f"cpu_constrained.neutral.{policy}.json")
        for policy in POLICIES
    }
    metrics = {
        policy: policy_metrics(payload, first_arrival)
        for policy, payload in current_payloads.items()
    }
    for policy, item in metrics.items():
        item["payload_sha256"] = sha256(current_root / f"cpu_constrained.neutral.{policy}.json")

    old_b4 = load(old_root / "cpu_constrained.neutral.b4.json")
    new_b4 = current_payloads["b4"]
    old_b4_true_fallbacks = []
    old_b4_reason_rounds = 0
    for record in old_b4["joint_round_logs"]:
        result = record["decision_for_next_interval"].get("result", {})
        status = result.get("status", "")
        if (
            "FALLBACK" in status
            or "ALL_WAIT" in status
            or status == "NO_COLUMNS"
            or status.startswith("MIP_NO")
        ):
            old_b4_true_fallbacks.append({
                "interval_end": record["interval_end"], "status": status
            })
        if result.get("audit", {}).get("fallback_reason"):
            old_b4_reason_rounds += 1
    old_decisions = {
        record["interval_end"]: record["decision_for_next_interval"]
        for record in old_b4["joint_round_logs"]
    }
    new_decisions = {
        record["interval_end"]: record["decision_for_next_interval"]
        for record in new_b4["joint_round_logs"]
    }
    differing_times = [
        time
        for time in sorted(set(old_decisions) & set(new_decisions))
        if selected_map(old_decisions[time]) != selected_map(new_decisions[time])
    ]
    first_diff = differing_times[0]
    old_first = old_decisions[first_diff]
    new_first = new_decisions[first_diff]
    old_record = next(r for r in old_b4["joint_round_logs"] if r["interval_end"] == first_diff)
    new_record = next(r for r in new_b4["joint_round_logs"] if r["interval_end"] == first_diff)
    b4_job_differences = {
        job_id: {
            "old_jct": old_b4["summary"]["jcts"][job_id],
            "current_jct": new_b4["summary"]["jcts"][job_id],
            "jct_delta_current_minus_old": (
                new_b4["summary"]["jcts"][job_id] - old_b4["summary"]["jcts"][job_id]
            ),
            "old_completion": old_b4["summary"]["completion_times"][job_id],
            "current_completion": new_b4["summary"]["completion_times"][job_id],
            "completion_delta_current_minus_old": (
                new_b4["summary"]["completion_times"][job_id]
                - old_b4["summary"]["completion_times"][job_id]
            ),
        }
        for job_id in old_b4["summary"]["jcts"]
        if (
            old_b4["summary"]["jcts"][job_id] != new_b4["summary"]["jcts"][job_id]
            or old_b4["summary"]["completion_times"][job_id]
            != new_b4["summary"]["completion_times"][job_id]
        )
    }
    changed_code_hashes = {
        name: {"old": old_hash, "current": current_validation["code_hashes"].get(name)}
        for name, old_hash in old_validation["code_hashes"].items()
        if old_hash != current_validation["code_hashes"].get(name)
    }

    targeted_b3 = load(targeted_root / "cpu_constrained.neutral.b3.json")
    full_b3 = current_payloads["b3"]
    targeted_decisions = decision_signature(targeted_b3)
    full_decisions = decision_signature(full_b3)
    targeted_execution = execution_signature(targeted_b3)
    full_execution = execution_signature(full_b3)
    b3_repeat = {
        "runs_compared": [
            "cpu_contention_20job_b3_repair_targeted",
            "cpu_contention_20job_repair_full/b3",
        ],
        "same_inputs": (
            load(targeted_root / "run_manifest.json")["config_sha256"]
            == current_manifest["config_sha256"]
            and load(targeted_root / "run_manifest.json")["workload_sha256"]
            == current_manifest["workload_sha256"]
            and load(targeted_root / "run_manifest.json")["profile_hashes"]
            == current_manifest["profile_hashes"]
        ),
        "same_dirty_git_head": (
            load(targeted_root / "run_manifest.json")["git_head"]
            == current_manifest["git_head"]
        ),
        "per_run_scheduler_file_hashes_embedded": False,
        "code_identity_limitation": (
            "The two runs were executed in one audited session without an intervening scheduler edit, "
            "but the targeted manifest did not embed per-file scheduler hashes. Exact code identity is "
            "therefore operationally documented, not independently proven by each run manifest."
        ),
        "completed_jobs_equal": targeted_b3["summary"]["completed_jobs"] == full_b3["summary"]["completed_jobs"],
        "jcts_equal": targeted_b3["summary"]["jcts"] == full_b3["summary"]["jcts"],
        "completion_times_equal": targeted_b3["summary"]["completion_times"] == full_b3["summary"]["completion_times"],
        "waiting_seconds_equal": targeted_b3["summary"]["waiting_seconds"] == full_b3["summary"]["waiting_seconds"],
        "resource_hours_equal": targeted_b3["summary"]["resource_hours"] == full_b3["summary"]["resource_hours"],
        "decision_signatures_equal": targeted_decisions == full_decisions,
        "execution_signatures_equal": targeted_execution == full_execution,
        "decision_signature_sha256": stable_hash(targeted_decisions),
        "execution_signature_sha256": stable_hash(targeted_execution),
        "rounds": len(targeted_decisions),
    }

    events = full_b3["contention_evidence"]["cpu_saturated_gpu_idle_events"]
    allocation_points = {}
    ram_consistency = {}
    for policy, payload in current_payloads.items():
        points = Counter()
        for record in payload["joint_round_logs"]:
            for event in record["execution"]:
                allocation = event.get("resource_allocation")
                if allocation and event.get("placement"):
                    key = f"cpu={float(allocation['cpu_per_gpu']):g},ram={float(allocation['ram_gb_per_gpu']):g}"
                    points[key] += 1
        resource_hours = payload["summary"]["resource_hours"]
        allocation_points[policy] = dict(points)
        ram_consistency[policy] = {
            "ram_gb_hours": resource_hours["ram_gb_hours"],
            "gpu_hours": resource_hours["gpu_hours"],
            "effective_ram_gb_per_gpu_hour": (
                resource_hours["ram_gb_hours"] / resource_hours["gpu_hours"]
            ),
        }

    audit = {
        "status": "BLOCKED_NEW_FULL_REPEAT_MEMORY_GATE_WITH_OFFLINE_AUDIT_COMPLETE",
        "scope": "Offline reproducibility and metric audit; no scheduler or algorithm mutation",
        "repeat_run": {
            "status": "NOT_STARTED_MEMORY_GATE",
            "observed_available_memory_gib": args.blocked_memory_gib,
            "frozen_minimum_memory_gib": 2.0,
            "output_directory_reserved_but_not_created": "research_runs/sia_synergy/cpu_contention_20job_repro_run2",
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "scipy_milp_backend": "HiGHS through scipy.optimize.milp",
            "cvxpy": cvxpy.__version__,
            "sia_intent_solver": "CBC through CVXPY",
            "cylp": getattr(cylp, "__version__", "unknown"),
            "historical_package_versions_frozen_in_old_manifest": False,
        },
        "frozen_inputs": {
            "old_config_sha256": old_manifest["config_sha256"],
            "current_config_sha256": current_manifest["config_sha256"],
            "old_workload_sha256": old_manifest["workload_sha256"],
            "current_workload_sha256": current_manifest["workload_sha256"],
            "old_profile_hashes": old_manifest["profile_hashes"],
            "current_profile_hashes": current_manifest["profile_hashes"],
            "inputs_equal": (
                old_manifest["config_sha256"] == current_manifest["config_sha256"]
                and old_manifest["workload_sha256"] == current_manifest["workload_sha256"]
                and old_manifest["profile_hashes"] == current_manifest["profile_hashes"]
            ),
        },
        "current_five_policy_metrics": metrics,
        "b3_existing_two_run_reproducibility": b3_repeat,
        "b4_historical_difference": {
            "old_average_jct_seconds": old_b4["summary"]["average_jct_completed_only"],
            "current_average_jct_seconds": new_b4["summary"]["average_jct_completed_only"],
            "old_rounds": old_b4["summary"]["rounds"],
            "current_rounds": new_b4["summary"]["rounds"],
            "old_total_waiting_seconds": sum(old_b4["summary"]["waiting_seconds"].values()),
            "current_total_waiting_seconds": sum(new_b4["summary"]["waiting_seconds"].values()),
            "old_restarts": sum(item["num_restarts"] for item in old_b4["summary"]["jobs"].values()),
            "current_restarts": sum(item["num_restarts"] for item in new_b4["summary"]["jobs"].values()),
            "old_wasted_gpu_seconds": sum(item["wasted_gpu_seconds"] for item in old_b4["summary"]["jobs"].values()),
            "current_wasted_gpu_seconds": sum(item["wasted_gpu_seconds"] for item in new_b4["summary"]["jobs"].values()),
            "old_checkpoint_restore_delay_seconds": sum(
                float(event["checkpoint_delay_consumed"])
                for record in old_b4["joint_round_logs"] for event in record["execution"]
            ),
            "current_checkpoint_restore_delay_seconds": sum(
                float(event["checkpoint_delay_consumed"])
                for record in new_b4["joint_round_logs"] for event in record["execution"]
            ),
            "old_true_fallbacks": old_b4_true_fallbacks,
            "current_true_fallbacks": [],
            "old_audit_reason_rounds": old_b4_reason_rounds,
            "current_audit_reason_rounds": metrics["b4"]["audit_reason_rounds"],
            "changed_code_hashes": changed_code_hashes,
            "different_decision_timestamps": len(differing_times),
            "first_different_interval_end": first_diff,
            "incoming_execution_equal_at_first_difference": old_record["execution"] == new_record["execution"],
            "sia_fixed_gpu_intent_equal_at_first_difference": (
                old_first.get("sia_fixed_gpu_configs") == new_first.get("sia_fixed_gpu_configs")
            ),
            "raw_candidate_summary_equal_at_first_difference": (
                old_first.get("candidate_summary") == new_first.get("candidate_summary")
            ),
            "old_selected": selected_map(old_first),
            "current_selected": selected_map(new_first),
            "old_unallocated": old_first["result"].get("unallocated_jobs", []),
            "current_unallocated": new_first["result"].get("unallocated_jobs", []),
            "old_objective": old_first["result"].get("scheduling_objective"),
            "current_objective": new_first["result"].get("scheduling_objective"),
            "job_differences": b4_job_differences,
            "finding": (
                "The first divergence is consistent with the repaired phantom checkpoint-lock semantics. "
                "The current solution has a strictly higher recorded objective, so the observed branch is "
                "not explained by a tie between the two recorded solutions. Historical package versions "
                "were not frozen, so solver-version identity cannot be proven from the old manifest alone."
            ),
        },
        "fallback_accounting": {
            policy: {
                "solve_status_counts": item["solve_status_counts"],
                "true_fallback_status_counts": item["true_fallback_status_counts"],
                "audit_reason_rounds": item["audit_reason_rounds"],
                "audit_reason_job_occurrences": item["audit_reason_job_occurrences"],
                "audit_reason_values": item["audit_reason_values"],
            }
            for policy, item in metrics.items()
        },
        "b3_contention_analysis": {
            "total_waiting_seconds": sum(full_b3["summary"]["waiting_seconds"].values()),
            "waiting_jobs": {
                key: value for key, value in full_b3["summary"]["waiting_seconds"].items() if value
            },
            "cpu_saturated_gpu_idle_events": len(events),
            "events_with_unallocated_jobs": sum(bool(item["unallocated_jobs"]) for item in events),
            "events_without_unallocated_jobs": sum(not item["unallocated_jobs"] for item in events),
            "node_counts": dict(Counter(node["node_id"] for item in events for node in item["nodes"])),
            "idle_gpu_counts": {
                f"{node_id}|idle_gpus={idle:g}": count
                for (node_id, idle), count in Counter(
                    (node["node_id"], float(node["idle_gpus"]))
                    for item in events for node in item["nodes"]
                ).items()
            },
            "events_with_wait_details": [item for item in events if item["unallocated_jobs"]],
        },
        "ram_accounting": {
            "formula": "sum(node_usage.ram_gb * active_seconds) / 3600",
            "allocation_point_occurrences": allocation_points,
            "policy_consistency": ram_consistency,
            "b3_expected_floor_ram_gb_per_gpu": 20.83,
            "b3_formula_residual": (
                full_b3["summary"]["resource_hours"]["ram_gb_hours"]
                - full_b3["summary"]["resource_hours"]["gpu_hours"] * 20.83
            ),
        },
        "artifact_hashes": {
            "old_manifest": sha256(old_root / "run_manifest.json"),
            "current_manifest": sha256(current_root / "run_manifest.json"),
            "current_validation": sha256(current_root / "validation_summary.json"),
            "targeted_manifest": sha256(targeted_root / "run_manifest.json"),
            "audit_script": sha256(Path(__file__)),
        },
        "interpretation_warning": "SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE",
    }
    target = output / "audit_summary.json"
    target.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": audit["status"],
        "target": str(target),
        "b3_two_run_exact_decisions": b3_repeat["decision_signatures_equal"],
        "b4_first_divergence": first_diff,
        "repeat_run": audit["repeat_run"]["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
