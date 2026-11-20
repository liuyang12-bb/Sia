"""Validate the frozen response-audit artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--historical", required=True)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    manifest = json.loads((root / "sensitivity_manifest.json").read_text(encoding="utf-8"))
    runs = {
        item["file"]: json.loads((root / item["file"]).read_text(encoding="utf-8"))
        for item in manifest["runs"]
    }
    checks = {}
    checks["twelve_sequential_runs"] = len(runs) == 12 and manifest["sequential_execution"]
    checks["two_jobs_only"] = all(len(run["jobs"]) == 2 for run in runs.values())
    checks["all_jobs_completed"] = all(run["all_jobs_completed"] for run in runs.values())
    checks["jct_from_completion"] = all(
        math.isclose(job["jct"], job["completion_time"] - job["submission_time"], abs_tol=1e-12)
        for run in runs.values() for job in run["jobs"].values()
    )
    decisions = [
        record["decision_for_next_interval"]
        for run in runs.values() for record in run["joint_round_logs"]
        if record["decision_for_next_interval"]
    ]
    executions = [
        execution
        for run in runs.values() for record in run["joint_round_logs"]
        for execution in record["execution"]
    ]
    checks["all_capacity_audits_valid"] = all(d["result"]["audit"]["valid"] for d in decisions)
    checks["progress_monotonic"] = all(e["progress_delta"] >= -1e-9 for e in executions)
    checks["goodput_applied_once"] = all(
        e["training_seconds"] == 0
        or math.isclose(e["effective_goodput"], e["raw_goodput"] * e["rho"], rel_tol=1e-9, abs_tol=1e-9)
        for e in executions
    )
    checks["checkpoint_restore_has_no_training_progress"] = all(
        e["progress_delta"] <= 1e-9
        for e in executions
        if e["checkpoint_delay_consumed"] >= 60 and e["training_seconds"] == 0
    )
    resources = [resource for d in decisions for resource in d["resources"].values()]
    checks["response_exponent_identity"] = all(
        math.isclose(r["rho"], r["raw_rho"] ** r["response_exponent"], rel_tol=1e-12, abs_tol=1e-12)
        for r in resources
    )
    checks["coordinate_trace_complete"] = all(
        "source_coordinates" in r and "baseline_source_coordinates" in r
        for r in resources
    )
    checks["neutral_rho_exact"] = all(
        e["rho"] == 1.0
        for name, run in runs.items() if name.endswith(".neutral.json")
        for record in run["joint_round_logs"] for e in record["execution"]
    )
    checks["neutral_b1_b4_exact"] = runs["b1.neutral.json"]["jcts"] == runs["b4.neutral.json"]["jcts"]
    checks["b4_gpu_config_fixed"] = all(
        selected["config"]["config_id"] == d["sia_fixed_gpu_configs"][job_id]
        for d in decisions if d["mode"] == "b4"
        for job_id, selected in d["result"]["selected"].items()
    )
    coverage = json.loads((root / "coverage_summary.json").read_text(encoding="utf-8"))
    all_tests = json.loads((root / "all_tests_summary.json").read_text(encoding="utf-8"))
    checks["expanded_coverage_8_of_8"] = coverage["status"] == "PASS" and coverage["tests_run"] == 8
    checks["all_tests_27_of_27"] = all_tests["status"] == "PASS" and all_tests["tests_run"] == 27
    checks["original_sia_cli_byte_identical"] = sha(root / "official_sia_cli.log") == sha(root / "experimental_sia_cli.log")
    historical = Path(args.historical).resolve()
    expected_historical = {
        "run_manifest.json": "DB68695977A16813E33D967E0E1272998519CC9EDCE302458033441E1F6071F3",
        "validation_summary.json": "025D72CD163E9739DB68F4628BAD32151C236A61DCBC3BEF1BF496E1C4E37D3E",
        "comparison_summary.json": "66B3A615F7BFF1570F269BB53E4FF4A345DA73E6C18D56B62B6BC35ABC6C63DC",
    }
    historical_actual = {name: sha(historical / name) for name in expected_historical}
    checks["historical_artifacts_unchanged"] = historical_actual == expected_historical
    reproduction = json.loads((root / "rho_4_2981_reproduction.json").read_text(encoding="utf-8"))
    checks["historical_rho_exactly_reproduced"] = reproduction["recomputed_matches"]
    summary = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "official_sia_cli_sha256": sha(root / "official_sia_cli.log"),
        "experimental_sia_cli_sha256": sha(root / "experimental_sia_cli.log"),
        "historical_artifact_sha256": historical_actual,
        "sensitivity_average_jct": {name: run["average_jct"] for name, run in runs.items()},
    }
    (root / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
