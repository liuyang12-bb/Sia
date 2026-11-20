"""Run one frozen w10_late Medium policy in a fresh Python process."""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import psutil

import simulator_config as sim_config
from run_cpu_bottleneck_smoke import POLICIES, json_default, run_one, sha256
from run_cpu_contention_4job import execution_evidence


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value):
    path.write_text(json.dumps(value, indent=2, default=json_default), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-root", required=True)
    parser.add_argument("--synergy-models", required=True)
    parser.add_argument("--policy", required=True, choices=POLICIES)
    args = parser.parse_args()

    simulator_root = Path(__file__).resolve().parent
    repo_root = simulator_root.parent
    root = Path(args.frozen_root).resolve()
    frozen_path = root / "frozen_manifest.json"
    frozen = load(frozen_path)
    if frozen["status"] != "FROZEN":
        raise ValueError("follow-up is not frozen")
    current_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    if current_head != frozen["git_head"]:
        raise ValueError(f"execution commit changed: {current_head} != {frozen['git_head']}")
    config_path = Path(frozen["frozen_config"])
    workload_path = Path(frozen["workload"])
    if sha256(config_path) != frozen["frozen_config_sha256"]:
        raise ValueError("frozen config hash changed")
    if sha256(workload_path) != frozen["workload_sha256"]:
        raise ValueError("frozen workload hash changed")
    if sha256(Path(frozen["response_manifest"])) != frozen["response_manifest_sha256"]:
        raise ValueError("response manifest hash changed")
    for name, expected in frozen["required_source_hashes"].items():
        if sha256(simulator_root / name) != expected:
            raise ValueError(f"source hash changed after freeze: {name}")
    models = Path(args.synergy_models).resolve()
    for name, expected in frozen["profile_hashes"].items():
        if sha256(models / name) != expected:
            raise ValueError(f"profile hash changed: {name}")

    available = psutil.virtual_memory().available / 1024**3
    threshold = float(frozen["minimum_available_memory_gib"])
    if available < threshold:
        raise RuntimeError(
            f"available memory {available:.3f} GiB is below frozen {threshold:.3f} GiB gate"
        )
    policy_index = list(POLICIES).index(args.policy)
    for prior in POLICIES[:policy_index]:
        if not (root / "policies" / prior / "policy_manifest.json").is_file():
            raise RuntimeError(f"frozen policy order violation: {prior} must finish before {args.policy}")
    for later in POLICIES[policy_index + 1:]:
        if (root / "policies" / later).exists():
            raise RuntimeError(f"frozen policy order violation: later policy already exists: {later}")
    policy_root = root / "policies" / args.policy
    if policy_root.exists() and any(policy_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite policy output: {policy_root}")
    policy_root.mkdir(parents=True, exist_ok=True)
    console_path = policy_root / "console.log"
    result_path = policy_root / "result.json"
    config = load(config_path)
    workload = pd.read_csv(workload_path)
    if int(config["simulation"]["max_rounds"]) != 6000:
        raise ValueError("worker requires the frozen 6000-round config")
    if config["response"]["response_settings"]["medium"]["exponent"] != 0.5:
        raise ValueError("worker requires frozen Medium alpha=0.5")

    sim_config.num_parallel_jobs = 1
    sim_config.num_parallel_traces = 1
    sim_config.sia_solver_verbose = False
    started = time.perf_counter()
    try:
        with console_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
            result = run_one(args, config, workload, args.policy, "cpu_constrained", "medium")
        evidence = execution_evidence(result, config, "cpu_constrained")
        result["contention_evidence"] = evidence
        write(result_path, result)
    except Exception as exc:
        failure = {
            "status": "FAILED_POLICY_EXECUTION",
            "policy": args.policy,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "diagnostic": getattr(exc, "joint_diagnostic", None),
            "available_memory_gib_before_run": available,
        }
        write(policy_root / "failure.json", failure)
        raise

    summary = result["summary"]
    manifest = {
        "status": result["status"],
        "policy": args.policy,
        "git_head": current_head,
        "platform": platform.platform(),
        "python": sys.version,
        "available_memory_gib_before_run": available,
        "memory_threshold_gib": threshold,
        "config_sha256": sha256(config_path),
        "workload_sha256": sha256(workload_path),
        "response_manifest_sha256": sha256(Path(frozen["response_manifest"])),
        "profile_hashes": frozen["profile_hashes"],
        "required_source_hashes": frozen["required_source_hashes"],
        "completed_jobs": summary["completed_jobs"],
        "total_jobs": summary["total_jobs"],
        "rounds": summary["rounds"],
        "hit_round_limit": summary["hit_round_limit"],
        "result_sha256": sha256(result_path),
        "console_sha256": sha256(console_path),
        "wall_seconds_outer": time.perf_counter() - started,
        "command": " ".join(sys.argv),
    }
    write(policy_root / "policy_manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"], "policy": args.policy,
        "completed_jobs": manifest["completed_jobs"], "rounds": manifest["rounds"],
        "average_jct": summary["average_jct_completed_only"],
        "available_memory_gib_before_run": available,
    }, indent=2))


if __name__ == "__main__":
    main()
