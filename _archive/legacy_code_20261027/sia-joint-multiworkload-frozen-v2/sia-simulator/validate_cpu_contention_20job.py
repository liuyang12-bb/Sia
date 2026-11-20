"""Validate and summarize the frozen 20-job stability smoke artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


POLICIES = ("b1", "sy", "b4", "b5", "b3")
CAPACITY = "cpu_constrained"
RESPONSE = "neutral"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def percentile(values, q):
    values = sorted(float(x) for x in values)
    if not values:
        return 0.0
    rank = (len(values) - 1) * q
    lo = int(rank)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (rank - lo)


def decision_map(payload):
    return {
        float(record["interval_end"]): record["decision_for_next_interval"]
        for record in payload["joint_round_logs"]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--code-root", default=".")
    parser.add_argument("--previous-regression", required=True)
    parser.add_argument(
        "--validation-output-dir",
        help="Optional separate directory for validation outputs; preserves source run artifacts.",
    )
    parser.add_argument(
        "--contention-observation-mode",
        choices=("required", "record-only"),
        default="required",
        help=(
            "Keep the historical single-cohort mechanism trigger checks as required, "
            "or record their counts without requiring every cohort to trigger them."
        ),
    )
    args = parser.parse_args()
    root = Path(args.input).resolve()
    validation_output = (
        Path(args.validation_output_dir).resolve()
        if args.validation_output_dir else root
    )
    validation_output.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).resolve()
    code_root = Path(args.code_root).resolve()
    previous = Path(args.previous_regression).resolve()
    manifest = load(root / "run_manifest.json")
    preflight = load(root / "preflight.json")
    config = load(config_path)
    payloads = {}
    for record in manifest["runs"]:
        payloads[record["policy"]] = load(root / record["file"])

    if manifest["status"] != "COMPLETE":
        official = previous / "official_sia_regression.log"
        experimental = previous / "experimental_sia_regression.log"
        regression_hashes = {
            "official": sha256(official) if official.is_file() else None,
            "experimental": sha256(experimental) if experimental.is_file() else None,
        }
        code_files = [
            "joint_prototype/joint_integration.py",
            "joint_prototype/joint_scheduler.py",
            "joint_prototype/joint_profile.py",
            "joint_prototype/synergy_adapter.py",
            "job.py",
            "run_cpu_bottleneck_smoke.py",
            "run_cpu_contention_4job.py",
            "run_cpu_contention_20job.py",
            "validate_cpu_contention_20job.py",
        ]
        is_memory_block = manifest["status"] == "ABORTED_MEMORY_GATE"
        failure_path = root / "failure_event.json"
        failure_event = load(failure_path) if failure_path.is_file() else None
        fixed_errors = []
        synergy_errors = []
        checkpoint_errors = []
        table = []
        for policy, payload in payloads.items():
            summary = payload["summary"]
            evidence = payload["contention_evidence"]
            solve_status_counts = Counter()
            for record in payload["joint_round_logs"]:
                decision = record["decision_for_next_interval"]
                result = decision.get("result", {})
                status = result.get("status")
                if status:
                    solve_status_counts[status] += 1
                fixed = decision.get("sia_fixed_gpu_configs", {})
                if policy in {"b4", "b5"}:
                    for job_id, item in result.get("selected", {}).items():
                        if item["config"]["config_id"] != fixed.get(job_id):
                            fixed_errors.append((policy, decision.get("round"), job_id))
                if policy == "sy" and result and result.get("audit", {}).get("official_core") != "SYNERGY_TUNE":
                    synergy_errors.append((decision.get("round"), result.get("audit", {}).get("official_core")))
                duration = float(record["interval_end"] - record["interval_start"])
                for event in record["execution"]:
                    checkpoint = float(event["checkpoint_delay_consumed"])
                    training = float(event["training_seconds"])
                    if checkpoint + training > duration + 1e-9:
                        checkpoint_errors.append((policy, event["job_id"], record["interval_start"]))
                    if checkpoint >= duration - 1e-9 and (training > 1e-9 or abs(float(event["progress_delta"])) > 1e-9):
                        checkpoint_errors.append((policy, event["job_id"], record["interval_start"], "full_delay_progress"))
            table.append({
                "policy": policy,
                "completed_jobs": summary["completed_jobs"],
                "completion_rate": summary["completion_rate"],
                "average_jct_seconds": summary["average_jct_completed_only"],
                "jcts": summary["jcts"],
                "total_waiting_seconds": sum(summary["waiting_seconds"].values()),
                "waiting_seconds": summary["waiting_seconds"],
                "rounds": summary["rounds"],
                "simulated_seconds": summary["simulated_seconds"],
                "wall_seconds": summary["wall_seconds"],
                "resource_hours": summary["resource_hours"],
                "rho_range": [summary["min_rho"], summary["max_rho"]],
                "restarts": sum(job["num_restarts"] for job in summary["jobs"].values()),
                "migrations": sum(job["num_migrations"] for job in summary["jobs"].values()),
                "wasted_gpu_seconds": sum(job["wasted_gpu_seconds"] for job in summary["jobs"].values()),
                "gpu_config_changes": summary["gpu_config_changes"],
                "solve_status_counts": dict(solve_status_counts),
                "optimizer_total_seconds": summary["optimizer_total_seconds"],
                "optimizer_p99_seconds": summary["optimizer_p99_seconds"],
                "fallback_count": len(summary["fallback_events"]),
                "tune_action_counts": summary["tune_action_counts"],
                "node_peak_aggregate_usage": summary["node_peak_aggregate_usage"],
                "node_time_weighted_utilization": evidence["node_time_weighted_utilization"],
                "cpu_saturated_gpu_idle_event_count": evidence["cpu_saturated_gpu_idle_event_count"],
                "wait_reason_rounds": evidence["wait_reason_rounds"],
            })
        available_checks = {
            "available_policy_results_complete_20_jobs": bool(payloads) and all(x["summary"]["completed_jobs"] == 20 for x in payloads.values()),
            "available_policy_results_no_capacity_violations": bool(payloads) and all(not x["summary"]["capacity_violations"] for x in payloads.values()),
            "available_policy_results_neutral_rho_one": bool(payloads) and all(x["summary"]["min_rho"] == 1.0 and x["summary"]["max_rho"] == 1.0 for x in payloads.values()),
            "available_policy_results_multiplier_once": bool(payloads) and all(x["contention_evidence"]["single_multiplier_max_abs_residual"] < 1e-9 for x in payloads.values()),
            "b4_b5_gpu_config_frozen": not fixed_errors if payloads else None,
            "sy_uses_adapted_tune_core": not synergy_errors if "sy" in payloads else None,
            "checkpoint_restore_semantics": not checkpoint_errors if payloads else None,
            "actual_queueing_observed": any(sum(x["summary"]["waiting_seconds"].values()) > 0 for x in payloads.values()) if payloads else None,
            "cpu_saturation_with_idle_gpu_observed": any(x["contention_evidence"]["cpu_saturated_gpu_idle_event_count"] > 0 for x in payloads.values()) if payloads else None,
        }
        validation_status = "BLOCKED_MEMORY_GATE" if is_memory_block else "FAIL_B3_EXECUTION"
        validation = {
            "status": validation_status,
            "all_checks_passed": False,
            "blocking_reason": manifest.get("blocking_reason") if is_memory_block else (failure_event or {}).get("exception_message", "incomplete run without persisted exception"),
            "checks": {
                "frozen_inputs_preflight_valid": preflight["cohort"]["job_count"] == 20,
                "memory_gate_honored": all(x["available_gib"] >= manifest["memory_threshold_gib"] for x in manifest["available_memory_samples"]) if not is_memory_block else manifest["completed_run_count"] == 0,
                "full_five_policy_matrix_complete": False,
                **available_checks,
                "b3_joint_choice_difference": None,
            },
            "not_run_checks_reason": "No policy was started because the pre-run memory gate failed." if is_memory_block else "B3 terminated before its policy artifact and last decision were committed.",
            "completed_policy_results": sorted(payloads),
            "missing_policy_results": sorted(set(POLICIES) - set(payloads)),
            "failure_event": failure_event,
            "original_sia_regression_sha256": regression_hashes,
            "original_sia_regression_preserved": (
                regression_hashes["official"] is not None
                and regression_hashes["official"] == regression_hashes["experimental"]
            ),
            "input_hashes": {
                "config": manifest["config_sha256"],
                "workload": manifest["workload_sha256"],
                "source_workload": manifest["source_workload_sha256"],
                "profiles": manifest["profile_hashes"],
            },
            "code_hashes": {name: sha256(code_root / name) for name in code_files},
            "preflight_sha256": sha256(root / "preflight.json"),
            "run_manifest_sha256_before_validation_outputs": sha256(root / "run_manifest.json"),
        }
        comparison = {
            "label": manifest["label"],
            "scope": manifest["experiment_scope"],
            "status": "NOT_RUN_MEMORY_GATE" if is_memory_block else "PARTIAL_FOUR_OF_FIVE_B3_FAILED",
            "table": sorted(table, key=lambda item: POLICIES.index(item["policy"])),
            "b3_pairwise_average_jct_seconds": None,
            "runtime_contention": {
                item["policy"]: {
                    "total_waiting_seconds": item["total_waiting_seconds"],
                    "cpu_saturated_gpu_idle_event_count": item["cpu_saturated_gpu_idle_event_count"],
                    "wait_reason_rounds": item["wait_reason_rounds"],
                }
                for item in table
            } if table else None,
            "tune_events": {
                item["policy"]: item["tune_action_counts"] for item in table
                if item["policy"] in {"sy", "b5"}
            } if table else None,
            "deferred_medium": manifest["deferred_medium"],
            "interpretation_warning": "The five-policy matrix is incomplete. Four policy results are valid for stability diagnostics only; no B3 pairwise performance claim is allowed.",
        }
        (validation_output / "comparison_summary.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        (validation_output / "validation_summary.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
        print(json.dumps(validation, indent=2))
        raise SystemExit(2)

    failures = []
    checks = {}

    def check(name, condition, detail=None):
        checks[name] = bool(condition)
        if not condition:
            failures.append({"check": name, "detail": detail})

    check("manifest_complete", manifest["status"] == "COMPLETE", manifest["status"])
    check("frozen_five_policy_matrix", set(payloads) == set(POLICIES), sorted(payloads))
    check("memory_gate_passed", all(x["available_gib"] >= manifest["memory_threshold_gib"] for x in manifest["available_memory_samples"]), manifest["available_memory_samples"])
    check("all_20_jobs_complete", all(x["summary"]["completed_jobs"] == 20 and x["summary"]["all_jobs_completed"] for x in payloads.values()))
    check("no_round_limit", all(not x["summary"]["hit_round_limit"] for x in payloads.values()))
    check("no_capacity_audit_violations", all(not x["summary"]["capacity_violations"] for x in payloads.values()))
    check("neutral_rho_exactly_one", all(x["summary"]["min_rho"] == 1.0 and x["summary"]["max_rho"] == 1.0 for x in payloads.values()))
    check("goodput_multiplier_applied_once", all(x["contention_evidence"]["single_multiplier_max_abs_residual"] < 1e-9 for x in payloads.values()))

    gpu_caps = config["cluster"]["ngpus_per_node"]
    aux_caps = config["cluster"]["capacity_scenarios"][CAPACITY]
    capacity_errors = []
    utilization_errors = []
    fixed_errors = []
    b1_intent_errors = []
    sy_core_errors = []
    checkpoint_errors = []
    progress_errors = []
    solve_statuses = defaultdict(Counter)
    tune_actions = defaultdict(Counter)
    candidate_counts = []
    fallback_counts = {}
    solve_times = defaultdict(list)
    for policy, payload in payloads.items():
        fallback_counts[policy] = len(payload["summary"]["fallback_events"])
        previous_progress = defaultdict(float)
        for node_id, util in payload["contention_evidence"]["node_time_weighted_utilization"].items():
            for resource in ("gpu_utilization", "cpu_utilization", "ram_utilization"):
                if util[resource] > 1.0 + 1e-9:
                    utilization_errors.append((policy, node_id, resource, util[resource]))
        for record in payload["joint_round_logs"]:
            decision = record["decision_for_next_interval"]
            result = decision.get("result", {})
            audit = result.get("audit", {})
            status = result.get("status")
            if status:
                solve_statuses[policy][status] += 1
            solve_times[policy].append(float(result.get("solve_seconds", 0.0)))
            selected = result.get("selected", {})
            fixed = decision.get("sia_fixed_gpu_configs", {})
            if policy in {"b4", "b5"}:
                for job_id, item in selected.items():
                    if item["config"]["config_id"] != fixed.get(job_id):
                        fixed_errors.append((policy, decision.get("round"), job_id, item["config"]["config_id"], fixed.get(job_id)))
            if policy == "b1":
                for job_id, item in selected.items():
                    if item["config"]["config_id"] != fixed.get(job_id):
                        b1_intent_errors.append((decision.get("round"), job_id))
            if policy == "sy" and result and audit.get("official_core") != "SYNERGY_TUNE":
                sy_core_errors.append((decision.get("round"), audit.get("official_core")))
            if policy in {"sy", "b5"}:
                tune_actions[policy].update(item.get("action") for item in audit.get("tune_trace", []))
            if policy == "b3":
                candidate_counts.extend(value["count"] for value in decision.get("candidate_summary", {}).values())
            for node_id, usage in audit.get("node_totals", {}).items():
                gpu_type = node_id.rsplit(":", 1)[0]
                if float(usage["gpus"]) > float(gpu_caps[gpu_type]) + 1e-9:
                    capacity_errors.append((policy, decision.get("round"), node_id, "gpu", usage["gpus"]))
                if float(usage["cpu"]) > float(aux_caps[gpu_type]["cpu_cores"]) + 1e-9:
                    capacity_errors.append((policy, decision.get("round"), node_id, "cpu", usage["cpu"]))
                if float(usage["ram_gb"]) > float(aux_caps[gpu_type]["ram_gb"]) + 1e-9:
                    capacity_errors.append((policy, decision.get("round"), node_id, "ram", usage["ram_gb"]))
            for event in record["execution"]:
                job_id = event["job_id"]
                before = float(event["progress_before"])
                after = float(event["progress_after"])
                if after + 1e-9 < before or before + 1e-9 < previous_progress[job_id]:
                    progress_errors.append((policy, job_id, record["interval_start"], previous_progress[job_id], before, after))
                previous_progress[job_id] = after
                duration = float(record["interval_end"] - record["interval_start"])
                checkpoint = float(event["checkpoint_delay_consumed"])
                training = float(event["training_seconds"])
                if checkpoint + training > duration + 1e-9:
                    checkpoint_errors.append((policy, job_id, record["interval_start"], checkpoint, training, duration))
                if checkpoint >= duration - 1e-9 and (training > 1e-9 or abs(float(event["progress_delta"])) > 1e-9):
                    checkpoint_errors.append((policy, job_id, record["interval_start"], "full_delay_progress"))

    check("instantaneous_gpu_cpu_ram_capacity", not capacity_errors, capacity_errors[:20])
    check("time_weighted_utilization_within_capacity", not utilization_errors, utilization_errors)
    check("b1_admitted_config_equals_sia_intent", not b1_intent_errors, b1_intent_errors[:20])
    check("b4_b5_gpu_config_frozen", not fixed_errors, fixed_errors[:20])
    check("sy_uses_adapted_tune_core", not sy_core_errors, sy_core_errors[:20])
    check("b3_enumerates_multiple_gpu_candidates", bool(candidate_counts) and max(candidate_counts) > 1, max(candidate_counts) if candidate_counts else None)
    check("checkpoint_restore_consumes_time_before_training", not checkpoint_errors, checkpoint_errors[:20])
    check("training_progress_monotonic", not progress_errors, progress_errors[:20])

    cpu_idle_events = {p: x["contention_evidence"]["cpu_saturated_gpu_idle_event_count"] for p, x in payloads.items()}
    total_waits = {p: sum(x["summary"]["waiting_seconds"].values()) for p, x in payloads.items()}
    cpu_wait_rounds = {
        p: x["contention_evidence"]["wait_reason_rounds"].get("CPU_SATURATED_WITH_IDLE_GPU", 0)
        for p, x in payloads.items()
    }
    contention_observations = {
        "actual_queueing_observed": sum(total_waits.values()) > 0,
        "cpu_saturation_with_idle_gpu_observed": sum(cpu_idle_events.values()) > 0,
        "unallocated_jobs_coincide_with_cpu_idle_gpu": sum(cpu_wait_rounds.values()) > 0,
        "total_waiting_seconds_by_policy": total_waits,
        "cpu_saturated_gpu_idle_events_by_policy": cpu_idle_events,
        "cpu_saturated_idle_gpu_wait_rounds_by_policy": cpu_wait_rounds,
    }
    if args.contention_observation_mode == "required":
        check("actual_queueing_observed", contention_observations["actual_queueing_observed"], total_waits)
        check(
            "cpu_saturation_with_idle_gpu_observed",
            contention_observations["cpu_saturation_with_idle_gpu_observed"],
            cpu_idle_events,
        )
        check(
            "unallocated_jobs_coincide_with_cpu_idle_gpu",
            contention_observations["unallocated_jobs_coincide_with_cpu_idle_gpu"],
            cpu_wait_rounds,
        )
    else:
        check("contention_observation_fields_recorded", True)

    b1_decisions = decision_map(payloads["b1"])
    b3_decisions = decision_map(payloads["b3"])
    config_differences = []
    for timestamp in sorted(set(b1_decisions) & set(b3_decisions)):
        b1_fixed = b1_decisions[timestamp].get("sia_fixed_gpu_configs", {})
        b3_selected = b3_decisions[timestamp].get("result", {}).get("selected", {})
        for job_id in sorted(set(b1_fixed) & set(b3_selected)):
            selected = b3_selected[job_id]["config"]["config_id"]
            if selected != b1_fixed[job_id]:
                config_differences.append({
                    "interval_end": timestamp,
                    "job_id": job_id,
                    "b1_sia_intent": b1_fixed[job_id],
                    "b3_selected": selected,
                    "comparison_scope": "same simulated timestamp; states may have diverged after earlier rounds",
                })
    check("b3_differs_from_b1_sia_intent_on_shared_timeline", bool(config_differences), config_differences[:5])

    official = previous / "official_sia_regression.log"
    experimental = previous / "experimental_sia_regression.log"
    regression_hashes = {
        "official": sha256(official) if official.is_file() else None,
        "experimental": sha256(experimental) if experimental.is_file() else None,
    }
    check("original_sia_regression_preserved", regression_hashes["official"] is not None and regression_hashes["official"] == regression_hashes["experimental"], regression_hashes)

    table = []
    for policy in POLICIES:
        payload = payloads[policy]
        summary = payload["summary"]
        evidence = payload["contention_evidence"]
        checkpoint_seconds = sum(float(event["checkpoint_delay_consumed"]) for record in payload["joint_round_logs"] for event in record["execution"])
        table.append({
            "policy": policy,
            "completed_jobs": summary["completed_jobs"],
            "completion_rate": summary["completion_rate"],
            "average_jct_seconds": summary["average_jct_completed_only"],
            "jcts": summary["jcts"],
            "completion_times": summary["completion_times"],
            "total_waiting_seconds": total_waits[policy],
            "waiting_seconds": summary["waiting_seconds"],
            "simulated_seconds": summary["simulated_seconds"],
            "rounds": summary["rounds"],
            "wall_seconds": summary["wall_seconds"],
            "resource_hours": summary["resource_hours"],
            "node_peak_aggregate_usage": summary["node_peak_aggregate_usage"],
            "node_time_weighted_utilization": evidence["node_time_weighted_utilization"],
            "rho_range": [summary["min_rho"], summary["max_rho"]],
            "gpu_config_changes": summary["gpu_config_changes"],
            "restarts": sum(job["num_restarts"] for job in summary["jobs"].values()),
            "migrations": sum(job["num_migrations"] for job in summary["jobs"].values()),
            "wasted_gpu_seconds": sum(job["wasted_gpu_seconds"] for job in summary["jobs"].values()),
            "checkpoint_restore_delay_seconds": checkpoint_seconds,
            "solve_status_counts": dict(solve_statuses[policy]),
            "optimizer_total_seconds": sum(solve_times[policy]),
            "optimizer_p99_seconds": percentile(solve_times[policy], 0.99),
            "fallback_count": fallback_counts[policy],
            "tune_action_counts": dict(tune_actions[policy]),
            "cpu_saturated_gpu_idle_event_count": cpu_idle_events[policy],
            "wait_reason_rounds": evidence["wait_reason_rounds"],
        })

    b3_jct = payloads["b3"]["summary"]["average_jct_completed_only"]
    comparisons = {
        f"b3_minus_{policy}_average_jct_seconds": b3_jct - payloads[policy]["summary"]["average_jct_completed_only"]
        for policy in ("b1", "sy", "b4", "b5")
    }
    code_files = [
        "joint_prototype/joint_integration.py",
        "joint_prototype/joint_scheduler.py",
        "joint_prototype/joint_profile.py",
        "joint_prototype/synergy_adapter.py",
        "job.py",
        "run_cpu_bottleneck_smoke.py",
        "run_cpu_contention_4job.py",
        "run_cpu_contention_20job.py",
        "validate_cpu_contention_20job.py",
    ]
    code_hashes = {name: sha256(code_root / name) for name in code_files}
    comparison = {
        "label": manifest["label"],
        "scope": manifest["experiment_scope"],
        "interpretation_warning": "Stability/mechanism smoke only. Pairwise JCT differences are descriptive and are not formal performance evidence.",
        "table": table,
        "b3_pairwise_average_jct_seconds": comparisons,
        "b3_vs_b1_same_timeline_config_differences_count": len(config_differences),
        "b3_vs_b1_same_timeline_config_difference_examples": config_differences[:50],
        "runtime_contention": {
            "total_waiting_seconds": total_waits,
            "cpu_saturated_gpu_idle_events": cpu_idle_events,
            "unallocated_job_cpu_idle_gpu_rounds": cpu_wait_rounds,
        },
        "tune_events": {policy: dict(actions) for policy, actions in tune_actions.items()},
        "deferred_medium": manifest["deferred_medium"],
    }
    validation = {
        "status": "PASS" if not failures else "FAIL",
        "all_checks_passed": not failures,
        "checks": checks,
        "contention_observation_mode": args.contention_observation_mode,
        "contention_observations": contention_observations,
        "failures": failures,
        "original_sia_regression_sha256": regression_hashes,
        "input_hashes": {
            "config": manifest["config_sha256"],
            "workload": manifest["workload_sha256"],
            "source_workload": manifest["source_workload_sha256"],
            "profiles": manifest["profile_hashes"],
        },
        "code_hashes": code_hashes,
        "preflight_sha256": sha256(root / "preflight.json"),
        "run_manifest_sha256_before_validation_outputs": sha256(root / "run_manifest.json"),
    }
    (validation_output / "comparison_summary.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    (validation_output / "validation_summary.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
