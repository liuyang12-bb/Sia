"""Freeze the only permitted 5000 -> 6000 round follow-up difference."""

from __future__ import annotations

import argparse
import copy
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import cvxpy
import cylp
import numpy
import pandas as pd
import psutil
import scipy

from run_cpu_bottleneck_smoke import hard_cpu_capacity_audit, sha256


POLICIES = ("b1", "sy", "b4", "b5", "b3")
CODE_FILES = (
    "job.py",
    "run_cpu_bottleneck_smoke.py",
    "run_cpu_contention_4job.py",
    "validate_cpu_contention_20job.py",
    "freeze_medium_w10_late_6000_followup.py",
    "run_medium_w10_late_6000_policy.py",
    "finalize_medium_w10_late_6000_followup.py",
    "analyze_medium_w10_late_6000_followup.py",
    "analyze_b3_multiworkload_medium_censored.py",
    "joint_prototype/joint_integration.py",
    "joint_prototype/joint_profile.py",
    "joint_prototype/joint_scheduler.py",
    "joint_prototype/synergy_adapter.py",
    "tests/test_medium_followup.py",
)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        keys = sorted(set(left) | set(right))
        result = []
        for key in keys:
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                result.append({"path": path, "old": left.get(key), "new": right.get(key)})
            else:
                result.extend(differences(left[key], right[key], path))
        return result
    if left != right:
        return [{"path": prefix, "old": left, "new": right}]
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--source-workload", required=True)
    parser.add_argument("--response-manifest", required=True)
    parser.add_argument("--medium-preregistration", required=True)
    parser.add_argument("--synergy-models", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    simulator_root = Path(__file__).resolve().parent
    repo_root = simulator_root.parent
    prereg_path = Path(args.preregistration).resolve()
    source_config = Path(args.source_config).resolve()
    source_workload = Path(args.source_workload).resolve()
    response_manifest = Path(args.response_manifest).resolve()
    medium_prereg = Path(args.medium_preregistration).resolve()
    models = Path(args.synergy_models).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite follow-up output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    prereg = load(prereg_path)
    if tuple(prereg["policies"]) != POLICIES:
        raise ValueError("policy order differs from pre-registration")
    if float(prereg["minimum_available_memory_gib"]) != 2.0:
        raise ValueError("memory gate must remain 2 GiB")
    expected_hashes = {
        source_config: prereg["source_config_sha256"],
        source_workload: prereg["source_workload_sha256"],
        response_manifest: prereg["response_manifest_sha256"],
        medium_prereg: prereg["medium_preregistration_sha256"],
    }
    for path, expected in expected_hashes.items():
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"frozen input hash mismatch for {path}: {actual} != {expected}")

    original = load(source_config)
    frozen = copy.deepcopy(original)
    if int(original["simulation"]["max_rounds"]) != int(prereg["original_max_rounds"]):
        raise ValueError("source config does not have the registered 5000-round window")
    frozen["simulation"]["max_rounds"] = int(prereg["followup_max_rounds"])
    diff = differences(original, frozen)
    if diff != [prereg["allowed_config_difference"]]:
        raise ValueError(f"unexpected config differences: {diff}")
    if float(frozen["response"]["response_settings"]["medium"]["exponent"]) != 0.5:
        raise ValueError("Medium alpha differs from 0.5")

    workload = pd.read_csv(source_workload)
    expected_jobs = original["workload_rows"]
    if workload["name"].tolist() != expected_jobs or len(workload) != 20:
        raise ValueError("w10_late workload order or membership changed")
    templates = sorted(set(original["response"]["template_map"].values()))
    profile_hashes = {}
    for name in templates:
        path = models / name
        if not path.is_file():
            raise FileNotFoundError(path)
        profile_hashes[name] = sha256(path)

    config_out = output / "frozen_config_6000.json"
    workload_out = output / "workload.csv"
    prereg_out = output / "followup_preregistration.json"
    write(config_out, frozen)
    shutil.copyfile(source_workload, workload_out)
    shutil.copyfile(prereg_path, prereg_out)
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    git_branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repo_root, text=True).strip()
    dependencies = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "cvxpy": cvxpy.__version__,
        "cylp": cylp.__version__,
        "cvxpy_installed_solvers": cvxpy.installed_solvers(),
        "cbc_version": "not exposed by installed CyLP",
        "highs_version": "not exposed by installed SciPy public API",
        "joint_milp_options": {"time_limit_seconds": 30.0, "mip_rel_gap": 0.0},
        "sia_cbc_options": {"numberThreads": 8, "allowablePercentageGap": 0.01},
    }
    manifest = {
        "status": "FROZEN",
        "label": prereg["disclaimer"],
        "scope": prereg["scope"],
        "git_head": git_head,
        "git_branch": git_branch,
        "preregistration": str(prereg_out),
        "preregistration_sha256": sha256(prereg_out),
        "source_config": str(source_config),
        "source_config_sha256": sha256(source_config),
        "frozen_config": str(config_out),
        "frozen_config_sha256": sha256(config_out),
        "config_differences": diff,
        "workload": str(workload_out),
        "workload_sha256": sha256(workload_out),
        "job_names": workload["name"].tolist(),
        "response_manifest": str(response_manifest),
        "response_manifest_sha256": sha256(response_manifest),
        "response_manifest_git_head": load(response_manifest)["git_head"],
        "medium_preregistration": str(medium_prereg),
        "medium_preregistration_sha256": sha256(medium_prereg),
        "profile_hashes": profile_hashes,
        "required_source_hashes": {name: sha256(simulator_root / name) for name in CODE_FILES},
        "dependencies": dependencies,
        "minimum_available_memory_gib": 2.0,
        "available_memory_gib_at_freeze": psutil.virtual_memory().available / 1024**3,
        "policies": list(POLICIES),
        "capacity": "cpu_constrained",
        "response": "medium",
        "response_exponent_alpha": 0.5,
        "max_rounds": 6000,
        "interval_seconds": 60,
        "hard_cpu_capacity_audit": hard_cpu_capacity_audit(frozen),
        "command": " ".join(sys.argv),
    }
    write(output / "frozen_manifest.json", manifest)
    write(output / "preflight.json", {
        "status": "PASS" if manifest["available_memory_gib_at_freeze"] >= 2.0 else "BLOCKED_MEMORY_GATE",
        "cohort": {"cohort_id": "w10_late", "job_count": 20},
        "available_memory_gib": manifest["available_memory_gib_at_freeze"],
        "minimum_available_memory_gib": 2.0,
        "config_sha256": manifest["frozen_config_sha256"],
        "workload_sha256": manifest["workload_sha256"],
        "source_workload_sha256": sha256(Path(frozen["source_workload"])),
        "profile_hashes": profile_hashes,
        "hard_cpu_capacity_audit": manifest["hard_cpu_capacity_audit"],
        "git_head": git_head,
    })
    print(json.dumps({
        "status": manifest["status"], "git_head": git_head,
        "config_sha256": manifest["frozen_config_sha256"],
        "workload_sha256": manifest["workload_sha256"],
        "difference": diff, "available_memory_gib": manifest["available_memory_gib_at_freeze"],
    }, indent=2))


if __name__ == "__main__":
    main()
