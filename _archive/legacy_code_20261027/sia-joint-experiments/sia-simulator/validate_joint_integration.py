"""Validate persisted joint integration smoke outputs and emit an audit JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    names = [
        "b1.sensitive.json", "b4.sensitive.json", "b3.sensitive.json",
        "b1.neutral.json", "b4.neutral.json", "b3.neutral.json", "sia.control.json",
    ]
    runs = {name: json.loads((root / name).read_text(encoding="utf-8")) for name in names}
    checks = {}

    checks["all_jobs_completed"] = all(run["all_jobs_completed"] for run in runs.values())
    checks["jct_from_completion_events"] = all(
        math.isclose(job["jct"], job["completion_time"] - job["submission_time"], abs_tol=1e-9)
        for run in runs.values() for job in run["jobs"].values()
    )
    joint = {name: run for name, run in runs.items() if name != "sia.control.json"}
    decisions = {
        name: [r["decision_for_next_interval"] for r in run["joint_round_logs"] if r["decision_for_next_interval"]]
        for name, run in joint.items()
    }
    executions = {
        name: [e for r in run["joint_round_logs"] for e in r["execution"]]
        for name, run in joint.items()
    }
    checks["all_capacity_audits_valid"] = all(
        decision["result"]["audit"]["valid"]
        for policy_decisions in decisions.values() for decision in policy_decisions
    )
    checks["progress_monotonic"] = all(
        execution["progress_delta"] >= -1e-9
        for policy_executions in executions.values() for execution in policy_executions
    )
    checks["no_progress_without_training"] = all(
        execution["progress_delta"] <= 1e-9 or execution["training_seconds"] > 0
        for policy_executions in executions.values() for execution in policy_executions
    )
    checks["goodput_applied_once"] = all(
        execution["training_seconds"] == 0
        or math.isclose(
            execution["effective_goodput"], execution["raw_goodput"] * execution["rho"],
            rel_tol=1e-9, abs_tol=1e-9,
        )
        for policy_executions in executions.values() for execution in policy_executions
    )
    checks["checkpoint_delay_has_no_progress_when_full_interval"] = all(
        execution["progress_delta"] <= 1e-9
        for policy_executions in executions.values() for execution in policy_executions
        if execution["checkpoint_delay_consumed"] >= 60 and execution["training_seconds"] == 0
    )
    checks["neutral_rho_exactly_one"] = all(
        execution["rho"] == 1.0
        for name, policy_executions in executions.items() if ".neutral." in name
        for execution in policy_executions
    )
    checks["neutral_b1_matches_original_sia_jct"] = runs["b1.neutral.json"]["jcts"] == runs["sia.control.json"]["jcts"]
    checks["neutral_b4_matches_original_sia_jct"] = runs["b4.neutral.json"]["jcts"] == runs["sia.control.json"]["jcts"]
    checks["b4_gpu_config_fixed_each_round"] = all(
        selected["config"]["config_id"] == decision["sia_fixed_gpu_configs"][job_id]
        for name, policy_decisions in decisions.items() if name.startswith("b4.")
        for decision in policy_decisions
        for job_id, selected in decision["result"]["selected"].items()
    )
    checks["b3_has_multiple_legal_candidates"] = all(
        summary["count"] > 1
        for decision in decisions["b3.sensitive.json"]
        for summary in decision["candidate_summary"].values()
        if summary["goodput_sources"] != ["unprofiled_sia_fallback"]
    )
    checks["direct_profiles_observed"] = any(
        resource["goodput_provenance"]["goodput_source"] == "direct_profile"
        for policy_decisions in decisions.values() for decision in policy_decisions
        for resource in decision["resources"].values()
    )
    checks["unprofiled_fallback_observed"] = any(
        resource["goodput_provenance"]["goodput_source"] == "unprofiled_sia_fallback"
        for policy_decisions in decisions.values() for decision in policy_decisions
        for resource in decision["resources"].values()
    )
    official = root / "official_sia_regression.log"
    experimental = root / "experimental_sia_regression.log"
    checks["official_sia_regression_byte_identical"] = sha256(official) == sha256(experimental)

    source_counts = Counter(
        resource["goodput_provenance"]["goodput_source"]
        for policy_decisions in decisions.values() for decision in policy_decisions
        for resource in decision["resources"].values()
    )
    summary = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "jcts": {name: run["jcts"] for name, run in runs.items()},
        "average_jct": {name: run["average_jct"] for name, run in runs.items()},
        "resource_hours": {name: run["resource_hours"] for name, run in joint.items()},
        "goodput_source_counts": dict(source_counts),
        "official_sia_regression_sha256": sha256(official),
        "experimental_sia_regression_sha256": sha256(experimental),
    }
    (root / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
