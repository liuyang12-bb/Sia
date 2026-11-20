"""Freeze the w10_late Medium 2x2 mechanism ablation without committing code."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import cvxpy
import cylp
import numpy
import psutil
import scipy

from run_cpu_bottleneck_smoke import hard_cpu_capacity_audit, sha256


POLICIES = ("b1", "b4", "b3_prop", "b3")
CODE_FILES = (
    "job.py",
    "run_cpu_bottleneck_smoke.py",
    "run_cpu_contention_4job.py",
    "joint_prototype/joint_integration.py",
    "joint_prototype/joint_profile.py",
    "joint_prototype/joint_scheduler.py",
    "freeze_b3_w10_late_ablation.py",
    "run_b3_w10_late_ablation_policy.py",
    "validate_b3_w10_late_ablation.py",
    "analyze_b3_w10_late_ablation.py",
    "tests/test_b3_prop_ablation.py",
)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--source-workload", required=True)
    parser.add_argument("--response-manifest", required=True)
    parser.add_argument("--synergy-models", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    simulator_root = Path(__file__).resolve().parent
    repo_root = simulator_root.parent
    prereg_path = Path(args.preregistration).resolve()
    source_config = Path(args.source_config).resolve()
    source_workload = Path(args.source_workload).resolve()
    response_manifest = Path(args.response_manifest).resolve()
    models = Path(args.synergy_models).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite ablation output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    prereg = load(prereg_path)
    if tuple(prereg["policies"]) != POLICIES:
        raise ValueError("policy matrix/order differs from preregistration")
    expected = {
        source_config: prereg["source_config_sha256"],
        source_workload: prereg["source_workload_sha256"],
        response_manifest: prereg["response_manifest_sha256"],
    }
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise ValueError(f"frozen input hash mismatch: {path}")

    config = load(source_config)
    if int(config["simulation"]["max_rounds"]) != 6000:
        raise ValueError("ablation requires the frozen 6000-round window")
    if int(config["simulation"]["interval_seconds"]) != 60:
        raise ValueError("ablation requires the frozen 60-second interval")
    if float(config["response"]["response_settings"]["medium"]["exponent"]) != 0.5:
        raise ValueError("ablation requires Medium alpha=0.5")
    if len(config["workload_rows"]) != 20:
        raise ValueError("ablation requires 20 frozen workload rows")

    config_out = output / "frozen_config.json"
    workload_out = output / "workload.csv"
    prereg_out = output / "preregistration.json"
    shutil.copyfile(source_config, config_out)
    shutil.copyfile(source_workload, workload_out)
    shutil.copyfile(prereg_path, prereg_out)

    templates = sorted(set(config["response"]["template_map"].values()))
    profile_hashes = {name: sha256(models / name) for name in templates}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repo_root, text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--short", "--untracked-files=all"], cwd=repo_root, text=True
    ).splitlines()
    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=repo_root)
    dependencies = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "cvxpy": cvxpy.__version__,
        "cylp": cylp.__version__,
        "cvxpy_installed_solvers": cvxpy.installed_solvers(),
        "joint_milp_options": {"time_limit_seconds": 30.0, "mip_rel_gap": 0.0},
        "sia_cbc_options": {"numberThreads": 8, "allowablePercentageGap": 0.01},
    }
    manifest = {
        "status": "FROZEN",
        "scope": prereg["scope"],
        "label": prereg["disclaimer"],
        "git_head": head,
        "git_branch": branch,
        "git_status_at_freeze": status,
        "tracked_git_diff_sha256": hashlib.sha256(diff).hexdigest().upper(),
        "algorithm_commit_created": False,
        "preregistration": str(prereg_out),
        "preregistration_sha256": sha256(prereg_out),
        "frozen_config": str(config_out),
        "frozen_config_sha256": sha256(config_out),
        "workload": str(workload_out),
        "workload_sha256": sha256(workload_out),
        "response_manifest": str(response_manifest),
        "response_manifest_sha256": sha256(response_manifest),
        "profile_hashes": profile_hashes,
        "required_source_hashes": {
            name: sha256(simulator_root / name) for name in CODE_FILES
        },
        "dependencies": dependencies,
        "policies": list(POLICIES),
        "capacity": "cpu_constrained",
        "response": "medium",
        "response_exponent_alpha": 0.5,
        "max_rounds": 6000,
        "interval_seconds": 60,
        "minimum_available_memory_gib": 2.0,
        "available_memory_gib_at_freeze": psutil.virtual_memory().available / 1024**3,
        "hard_cpu_capacity_audit": hard_cpu_capacity_audit(config),
    }
    write(output / "frozen_manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "git_head": head,
        "config_sha256": manifest["frozen_config_sha256"],
        "workload_sha256": manifest["workload_sha256"],
        "available_memory_gib": manifest["available_memory_gib_at_freeze"],
    }, indent=2))


if __name__ == "__main__":
    main()
