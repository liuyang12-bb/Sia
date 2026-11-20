"""Validate the frozen w10_late Medium 2x2 ablation end to end."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from run_cpu_bottleneck_smoke import json_default, sha256


POLICIES = ("b1", "b4", "b3_prop", "b3")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value):
    path.write_text(json.dumps(value, indent=2, default=json_default), encoding="utf-8")


def close(left, right, tolerance=1e-9):
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    frozen = load(root / "frozen_manifest.json")
    config = load(Path(frozen["frozen_config"]))
    failures = []
    checks = {}
    payloads = {}
    manifests = {}

    def check(name, condition, detail=None):
        checks[name] = bool(condition)
        if not condition:
            failures.append({"check": name, "detail": detail})

    for policy in POLICIES:
        policy_root = root / "policies" / policy
        result_path = policy_root / "result.json"
        manifest_path = policy_root / "policy_manifest.json"
        if result_path.is_file() and manifest_path.is_file():
            payloads[policy] = load(result_path)
            manifests[policy] = load(manifest_path)

    check("four_policy_matrix_present", set(payloads) == set(POLICIES), sorted(payloads))
    check(
        "all_inputs_traceable",
        bool(manifests) and all(
            item["config_sha256"] == frozen["frozen_config_sha256"]
            and item["workload_sha256"] == frozen["workload_sha256"]
            and item["response_manifest_sha256"] == frozen["response_manifest_sha256"]
            for item in manifests.values()
        ),
    )
    check(
        "all_20_jobs_complete",
        bool(payloads) and all(
            item["status"] == "SUCCESS"
            and item["summary"]["completed_jobs"] == 20
            and item["summary"]["total_jobs"] == 20
            for item in payloads.values()
        ),
    )
    check(
        "no_round_limit",
        bool(payloads) and all(not item["summary"]["hit_round_limit"] for item in payloads.values()),
    )
    check(
        "memory_gate_passed",
        bool(manifests) and all(
            item["available_memory_gib_before_run"] >= item["memory_threshold_gib"]
            for item in manifests.values()
        ),
    )

    capacities = config["cluster"]["capacity_scenarios"]["cpu_constrained"]
    node_gpus = config["cluster"]["ngpus_per_node"]
    minimum_cpu = float(config["response"]["minimum_legal_profile_point"]["cpu_per_gpu"])
    minimum_ram = float(config["response"]["minimum_legal_profile_point"]["ram_gb_per_gpu"])
    exponent = float(config["response"]["response_settings"]["medium"]["exponent"])
    capacity_errors = []
    rho_errors = []
    checkpoint_errors = []
    progress_errors = []
    audit_errors = []
    b4_freeze_errors = []
    prop_errors = []
    solve_statuses = {}
    b3_prop_multiple_candidates = False

    for policy, payload in payloads.items():
        previous_progress = {}
        statuses = Counter()
        for record in payload["joint_round_logs"]:
            decision = record["decision_for_next_interval"]
            result = decision.get("result", {})
            if result:
                statuses[result.get("status", "MISSING_STATUS")] += 1
                audit = result.get("audit", {})
                if not audit.get("valid", False):
                    audit_errors.append((policy, decision.get("round"), audit))
                for node_id, usage in audit.get("node_totals", {}).items():
                    gpu_type = node_id.rsplit(":", 1)[0]
                    cap = capacities[gpu_type]
                    if float(usage["gpus"]) > float(node_gpus[gpu_type]) + 1e-9:
                        capacity_errors.append((policy, node_id, "gpu", usage["gpus"]))
                    if float(usage["cpu"]) > float(cap["cpu_cores"]) + 1e-9:
                        capacity_errors.append((policy, node_id, "cpu", usage["cpu"]))
                    if float(usage["ram_gb"]) > float(cap["ram_gb"]) + 1e-9:
                        capacity_errors.append((policy, node_id, "ram", usage["ram_gb"]))
                for job_id, selected in result.get("selected", {}).items():
                    raw_rho = float(selected["raw_rho"])
                    rho = float(selected["rho"])
                    effective = float(selected["effective_goodput"])
                    raw_goodput = float(selected["raw_sia_goodput"])
                    if not close(rho, raw_rho ** exponent, 1e-11):
                        rho_errors.append((policy, job_id, "formula", raw_rho, rho))
                    if not close(effective, raw_goodput * rho, 1e-11):
                        rho_errors.append((policy, job_id, "single_multiplier", raw_goodput, rho, effective))
                    if policy == "b4":
                        fixed = decision.get("sia_fixed_gpu_configs", {}).get(job_id)
                        if fixed != selected["config"]["config_id"]:
                            b4_freeze_errors.append((job_id, fixed, selected["config"]["config_id"]))
                    if policy == "b3_prop":
                        placement = selected["placement"]
                        points = set()
                        for node_id in placement:
                            gpu_type = node_id.rsplit(":", 1)[0]
                            points.add((
                                float(capacities[gpu_type]["cpu_cores"]) / float(node_gpus[gpu_type]),
                                float(capacities[gpu_type]["ram_gb"]) / float(node_gpus[gpu_type]),
                            ))
                        if len(points) != 1:
                            prop_errors.append((job_id, "inconsistent_multinode_ratio", sorted(points)))
                        else:
                            point = next(iter(points))
                            expected_cpu = max(point[0], minimum_cpu)
                            expected_ram = max(point[1], minimum_ram)
                            if not close(selected["cpu_per_gpu"], expected_cpu):
                                prop_errors.append((job_id, "cpu", selected["cpu_per_gpu"], expected_cpu))
                            if not close(selected["ram_gb_per_gpu"], expected_ram):
                                prop_errors.append((job_id, "ram", selected["ram_gb_per_gpu"], expected_ram))
                if policy == "b3_prop":
                    summary = decision.get("candidate_summary", {})
                    if any(len(item.get("config_ids", [])) > 1 for item in summary.values()):
                        b3_prop_multiple_candidates = True

            duration = float(record["interval_end"] - record["interval_start"])
            for event in record["execution"]:
                job_id = event["job_id"]
                before = float(event["progress_before"])
                after = float(event["progress_after"])
                if after + 1e-9 < before:
                    progress_errors.append((policy, job_id, record["interval_start"], before, after))
                prior = previous_progress.get(job_id)
                if prior is not None and before + 1e-9 < prior:
                    progress_errors.append((policy, job_id, "cross_round", prior, before))
                previous_progress[job_id] = after
                checkpoint = float(event["checkpoint_delay_consumed"])
                training = float(event["training_seconds"])
                if checkpoint + training > duration + 1e-9:
                    checkpoint_errors.append((policy, job_id, checkpoint, training, duration))
                if checkpoint >= duration - 1e-9 and (
                    training > 1e-9 or abs(float(event["progress_delta"])) > 1e-9
                ):
                    checkpoint_errors.append((policy, job_id, "progress_during_full_delay"))
        solve_statuses[policy] = dict(statuses)

    check("all_decisions_atomically_audited", not audit_errors, audit_errors[:10])
    check("instantaneous_gpu_cpu_ram_capacity", not capacity_errors, capacity_errors[:10])
    check("medium_response_formula_exact", not rho_errors, rho_errors[:10])
    check(
        "goodput_multiplier_applied_once",
        bool(payloads) and all(
            item["contention_evidence"]["single_multiplier_max_abs_residual"] < 1e-9
            for item in payloads.values()
        ),
    )
    check("checkpoint_restore_before_training", not checkpoint_errors, checkpoint_errors[:10])
    check("training_progress_monotonic", not progress_errors, progress_errors[:10])
    check("b4_holds_sia_gpu_config", not b4_freeze_errors, b4_freeze_errors[:10])
    check("b3_prop_strictly_proportional", not prop_errors, prop_errors[:10])
    check("b3_prop_enumerates_multiple_gpu_candidates", b3_prop_multiple_candidates)
    check(
        "b3_prop_uses_expected_solver_path",
        bool(payloads.get("b3_prop")) and all(
            record["decision_for_next_interval"].get("result", {}).get("policy")
            in {"B3_PROP_SIA_GOODPUT_PROPORTIONAL_AUX_MIP", "B3_PRESERVE_PREVIOUS_VALID_ALLOCATION", "B3_LEGAL_ALL_WAIT"}
            for record in payloads["b3_prop"]["joint_round_logs"]
        ),
    )

    table = []
    for policy in POLICIES:
        if policy not in payloads:
            continue
        payload = payloads[policy]
        summary = payload["summary"]
        jcts = sorted(float(value) for value in summary["jcts"].values())
        completion_times = [
            float(value)
            for value in summary["completion_times"].values()
            if value is not None
        ]
        checkpoint_seconds = sum(
            float(event["checkpoint_delay_consumed"])
            for record in payload["joint_round_logs"]
            for event in record["execution"]
        )
        table.append({
            "policy": policy,
            "completed_jobs": summary["completed_jobs"],
            "average_jct_seconds": summary["average_jct_completed_only"],
            "median_jct_seconds": float(np.percentile(jcts, 50)) if jcts else None,
            "p95_jct_seconds": float(np.percentile(jcts, 95)) if jcts else None,
            "p99_jct_seconds": float(np.percentile(jcts, 99)) if jcts else None,
            "makespan_seconds": max(completion_times) if completion_times else None,
            "resource_hours": summary["resource_hours"],
            "restart_count": sum(item["num_restarts"] for item in summary["jobs"].values()),
            "checkpoint_restore_seconds": checkpoint_seconds,
            "wasted_gpu_seconds": sum(item["wasted_gpu_seconds"] for item in summary["jobs"].values()),
            "total_waiting_seconds": sum(summary["waiting_seconds"].values()),
            "rho_range": [summary["min_rho"], summary["max_rho"]],
            "rounds": summary["rounds"],
            "solve_status_counts": solve_statuses.get(policy, {}),
            "capacity_violation_count": len(summary["capacity_violations"]),
        })

    validation = {
        "status": "PASS" if not failures else "FAIL",
        "all_checks_passed": not failures,
        "checks": checks,
        "failures": failures,
        "table": table,
        "solve_status_counts": solve_statuses,
        "frozen_manifest_sha256": sha256(root / "frozen_manifest.json"),
        "input_hashes": {
            "config": frozen["frozen_config_sha256"],
            "workload": frozen["workload_sha256"],
            "response_manifest": frozen["response_manifest_sha256"],
        },
        "source_hashes": frozen["required_source_hashes"],
    }
    write(root / "validation_summary.json", validation)
    print(json.dumps(validation, indent=2, default=json_default))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
