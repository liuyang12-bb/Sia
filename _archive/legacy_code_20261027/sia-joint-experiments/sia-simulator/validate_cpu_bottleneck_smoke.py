"""Validate the two-job CPU-bottleneck smoke artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def selected_config(payload, job_id):
    for record in payload["joint_round_logs"]:
        selected = record["decision_for_next_interval"].get("result", {}).get("selected", {})
        if job_id in selected:
            yield selected[job_id]["config"]["config_id"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    root = Path(args.input).resolve()
    manifest = load(root / "run_manifest.json")
    payloads = {}
    failures = []
    for record in manifest["runs"]:
        key = (record["capacity"], record["response"], record["policy"])
        payloads[key] = load(root / record["file"])

    def check(name, condition, detail=None):
        if not condition:
            failures.append({"check": name, "detail": detail})
        return bool(condition)

    checks = {}
    checks["full_factorial_20_runs"] = check("full_factorial_20_runs", len(payloads) == 20, len(payloads))
    checks["all_two_jobs_complete"] = check(
        "all_two_jobs_complete",
        all(item["summary"]["completed_jobs"] == 2 for item in payloads.values()),
    )
    checks["no_capacity_violations"] = check(
        "no_capacity_violations",
        all(not item["summary"]["capacity_violations"] for item in payloads.values()),
    )
    neutral_rhos = [
        (item["summary"]["min_rho"], item["summary"]["max_rho"])
        for (capacity, response, policy), item in payloads.items()
        if response == "neutral"
    ]
    checks["neutral_rho_exactly_one"] = check(
        "neutral_rho_exactly_one",
        all(lo == 1.0 and hi == 1.0 for lo, hi in neutral_rhos),
        neutral_rhos,
    )

    fixed_errors = []
    synergy_core_errors = []
    b3_candidate_counts = []
    checkpoint_progress_errors = []
    for key, payload in payloads.items():
        policy = key[2]
        for record in payload["joint_round_logs"]:
            decision = record["decision_for_next_interval"]
            result = decision.get("result", {})
            selected = result.get("selected", {})
            fixed = decision.get("sia_fixed_gpu_configs", {})
            if policy == "b5":
                for job_id, item in selected.items():
                    if item["config"]["config_id"] != fixed.get(job_id):
                        fixed_errors.append((key, decision.get("round"), job_id))
            if policy == "sy" and result:
                if result.get("audit", {}).get("official_core") != "SYNERGY_TUNE":
                    synergy_core_errors.append((key, decision.get("round")))
            if policy == "b3":
                b3_candidate_counts.extend(
                    item["count"] for item in decision.get("candidate_summary", {}).values()
                )
            for event in record["execution"]:
                if (
                    event["checkpoint_delay_consumed"] >= 60.0
                    and event["training_seconds"] == 0.0
                    and abs(event["progress_delta"]) > 1e-9
                ):
                    checkpoint_progress_errors.append((key, event["job_id"], record["interval_start"]))
    checks["b5_never_changes_sia_gpu_config"] = check(
        "b5_never_changes_sia_gpu_config", not fixed_errors, fixed_errors
    )
    checks["sy_uses_audited_tune_core"] = check(
        "sy_uses_audited_tune_core", not synergy_core_errors, synergy_core_errors
    )
    sy_expected = {"bert-28": "aws:1:4", "cifar10-29": "dgx-ext:1:4"}
    sy_fixed_observed = {
        f"{capacity}.{response}.{job_id}": sorted(set(selected_config(payload, job_id)))
        for (capacity, response, policy), payload in payloads.items()
        if policy == "sy"
        for job_id in sy_expected
    }
    checks["sy_gpu_type_and_demand_are_fixed"] = check(
        "sy_gpu_type_and_demand_are_fixed",
        all(values == [sy_expected[key.rsplit(".", 1)[-1]]] for key, values in sy_fixed_observed.items()),
        sy_fixed_observed,
    )
    checks["b3_enumerates_multiple_gpu_candidates"] = check(
        "b3_enumerates_multiple_gpu_candidates",
        bool(b3_candidate_counts) and max(b3_candidate_counts) > 1,
        max(b3_candidate_counts) if b3_candidate_counts else None,
    )
    checks["checkpoint_full_delay_has_no_progress"] = check(
        "checkpoint_full_delay_has_no_progress",
        not checkpoint_progress_errors,
        checkpoint_progress_errors,
    )

    hard = manifest["hard_cpu_capacity_audit"]
    forced_idle = hard["cpu_constrained"]["rtx"]["gpus_forced_idle_if_node_otherwise_full"]
    checks["cpu_floor_makes_full_rtx_node_infeasible"] = check(
        "cpu_floor_makes_full_rtx_node_infeasible", forced_idle == 2, forced_idle
    )
    checks["sufficient_full_nodes_feasible"] = check(
        "sufficient_full_nodes_feasible",
        all(
            item["full_node_gpu_config_cpu_feasible"]
            for item in hard["cpu_sufficient"].values()
        ),
    )

    sensitive = "cifar10-29"
    pressure_effect = {}
    for policy in ("b1", "sy", "b4", "b5", "b3"):
        sufficient = payloads[("cpu_sufficient", "medium", policy)]["summary"]
        constrained = payloads[("cpu_constrained", "medium", policy)]["summary"]
        pressure_effect[policy] = {
            "sufficient_jct": sufficient["jcts"][sensitive],
            "constrained_jct": constrained["jcts"][sensitive],
            "sufficient_rho": sufficient["rho_range_by_job"][sensitive],
            "constrained_rho": constrained["rho_range_by_job"][sensitive],
        }
    checks["cpu_pressure_changes_sensitive_execution"] = check(
        "cpu_pressure_changes_sensitive_execution",
        any(
            item["sufficient_jct"] != item["constrained_jct"]
            or item["sufficient_rho"] != item["constrained_rho"]
            for item in pressure_effect.values()
        ),
        pressure_effect,
    )
    official_regression = root / "official_sia_regression.log"
    experimental_regression = root / "experimental_sia_regression.log"
    regression_hashes = {
        "official": sha256(official_regression) if official_regression.is_file() else None,
        "experimental": sha256(experimental_regression) if experimental_regression.is_file() else None,
    }
    checks["original_sia_cli_byte_identical"] = check(
        "original_sia_cli_byte_identical",
        regression_hashes["official"] is not None
        and regression_hashes["official"] == regression_hashes["experimental"],
        regression_hashes,
    )

    table = []
    for key in sorted(payloads):
        summary = payloads[key]["summary"]
        table.append(
            {
                "capacity": key[0],
                "response": key[1],
                "policy": key[2],
                "completed": summary["completed_jobs"],
                "average_jct": summary["average_jct_completed_only"],
                "jcts": summary["jcts"],
                "waiting_seconds": summary["waiting_seconds"],
                "min_rho": summary["min_rho"],
                "max_rho": summary["max_rho"],
                "gpu_hours": summary["resource_hours"]["gpu_hours"],
                "cpu_core_hours": summary["resource_hours"]["cpu_core_hours"],
                "restarts": sum(job["num_restarts"] for job in summary["jobs"].values()),
                "optimizer_total_seconds": summary["optimizer_total_seconds"],
            }
        )
    comparison = {
        "label": manifest["label"],
        "table": table,
        "cpu_pressure_effect_on_sensitive_job": pressure_effect,
        "b3_minus_b5_average_jct": {
            f"{capacity}.{response}": (
                payloads[(capacity, response, "b3")]["summary"]["average_jct_completed_only"]
                - payloads[(capacity, response, "b5")]["summary"]["average_jct_completed_only"]
            )
            for capacity in ("cpu_sufficient", "cpu_constrained")
            for response in ("neutral", "medium")
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
