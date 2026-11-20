"""Assemble and validate five independent w10_late policy processes."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

from run_cpu_bottleneck_smoke import POLICIES, hard_cpu_capacity_audit, json_default, sha256


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value):
    path.write_text(json.dumps(value, indent=2, default=json_default), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-root", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--previous-regression", required=True)
    args = parser.parse_args()
    root = Path(args.frozen_root).resolve()
    code_root = Path(args.code_root).resolve()
    previous = Path(args.previous_regression).resolve()
    frozen = load(root / "frozen_manifest.json")
    config_path = Path(frozen["frozen_config"])
    workload_path = Path(frozen["workload"])
    config = load(config_path)
    if (root / "run_manifest.json").exists():
        raise FileExistsError("refusing to overwrite follow-up run_manifest.json")

    records = []
    memory_samples = []
    commands = []
    for policy in POLICIES:
        policy_root = root / "policies" / policy
        result_path = policy_root / "result.json"
        policy_manifest_path = policy_root / "policy_manifest.json"
        if not result_path.is_file() or not policy_manifest_path.is_file():
            raise FileNotFoundError(f"missing completed policy artifact: {policy}")
        policy_manifest = load(policy_manifest_path)
        result = load(result_path)
        if policy_manifest["git_head"] != frozen["git_head"]:
            raise ValueError(f"policy commit mismatch: {policy}")
        if sha256(result_path) != policy_manifest["result_sha256"]:
            raise ValueError(f"policy result hash mismatch: {policy}")
        memory_samples.append({
            "phase": "before_run", "cohort_id": "w10_late",
            "capacity": "cpu_constrained", "response": "medium", "policy": policy,
            "available_gib": policy_manifest["available_memory_gib_before_run"],
        })
        commands.append(policy_manifest["command"])
        evidence = result["contention_evidence"]
        records.append({
            "capacity": "cpu_constrained", "response": "medium", "policy": policy,
            "file": f"policies/{policy}/result.json",
            **result["summary"],
            "contention_evidence": {
                "wait_reason_rounds": evidence["wait_reason_rounds"],
                "cpu_saturated_gpu_idle_event_count": evidence["cpu_saturated_gpu_idle_event_count"],
                "single_multiplier_max_abs_residual": evidence["single_multiplier_max_abs_residual"],
            },
        })
        del result

    source_workload = Path(config["source_workload"])
    cohort = {
        "cohort_id": "w10_late",
        "cohort_rule": config["cohort_rule"],
        "job_count": 20,
        "numeric_job_ids": list(range(140, 160)),
        "mapping_complete": True,
    }
    manifest = {
        "status": "COMPLETE",
        "label": frozen["label"],
        "experiment_scope": frozen["scope"],
        "platform": platform.platform(),
        "python": sys.version,
        "git_head": frozen["git_head"],
        "git_branch": frozen["git_branch"],
        "available_memory_samples": memory_samples,
        "memory_threshold_gib": frozen["minimum_available_memory_gib"],
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "workload": str(workload_path),
        "workload_sha256": sha256(workload_path),
        "source_workload": str(source_workload),
        "source_workload_sha256": sha256(source_workload),
        "profile_hashes": frozen["profile_hashes"],
        "cohort": cohort,
        "hard_cpu_capacity_audit": hard_cpu_capacity_audit(config),
        "frozen_matrix": {"capacities": ["cpu_constrained"], "responses": ["medium"], "policies": list(POLICIES)},
        "deferred_medium": ["Low", "High", "parameter search", "160-job"],
        "commands": commands,
        "completed_run_count": 5,
        "expected_run_count": 5,
        "runs": records,
        "failure": None,
        "required_source_hashes": frozen["required_source_hashes"],
        "dependencies": frozen["dependencies"],
    }
    write(root / "run_manifest.json", manifest)
    validator = code_root / "validate_cpu_contention_20job.py"
    command = [
        sys.executable, str(validator), "--input", str(root),
        "--config", str(config_path), "--code-root", str(code_root),
        "--previous-regression", str(previous),
        "--contention-observation-mode", "record-only",
        "--expected-response", "medium",
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    (root / "validation_console.log").write_text(
        completed.stdout + ("\nSTDERR\n" + completed.stderr if completed.stderr else ""),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"follow-up validator failed with exit {completed.returncode}")
    validation = load(root / "validation_summary.json")
    if validation["status"] != "PASS" or not validation["all_checks_passed"]:
        raise RuntimeError("follow-up validator did not pass")

    wrapper = root / "combine_input"
    wrapper.mkdir(exist_ok=False)
    cohort_record = {
        "cohort_id": "w10_late", "status": "COMPLETE", "root": str(root),
        "config": str(config_path), "config_sha256": sha256(config_path),
        "workload": str(workload_path), "workload_sha256": sha256(workload_path),
        "source_workload": str(source_workload),
        "source_workload_sha256": sha256(source_workload),
        "runs": records, "validation": "PASS",
        "validation_summary_sha256": sha256(root / "validation_summary.json"),
    }
    wrapper_manifest = {
        "status": "COMPLETE", "label": frozen["label"], "scope": frozen["scope"],
        "git_head": frozen["git_head"], "git_branch": frozen["git_branch"],
        "response": "medium", "preregistration_sha256": frozen["preregistration_sha256"],
        "frozen_selection_sha256": frozen["workload_sha256"],
        "required_source_hashes": frozen["required_source_hashes"],
        "profile_hashes": frozen["profile_hashes"], "dependencies": frozen["dependencies"],
        "memory_threshold_gib": frozen["minimum_available_memory_gib"],
        "available_memory_samples": memory_samples,
        "expected_policy_runs": 5, "completed_policy_runs": 5, "completed_cohorts": 1,
        "cohort_runs": [cohort_record], "failure": None, "all_cohorts_validated": True,
        "command": " ".join(sys.argv),
    }
    write(wrapper / "run_manifest.json", wrapper_manifest)
    print(json.dumps({
        "status": "PASS", "completed": {item["policy"]: item["completed_jobs"] for item in records},
        "validation": str(root / "validation_summary.json"),
        "combine_input": str(wrapper),
    }, indent=2))


if __name__ == "__main__":
    main()
