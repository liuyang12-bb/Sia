"""Run the pre-registered four-level response audit on the frozen two-job input."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

import simulator_config as sim_config
from joint_prototype.joint_profile import assign_templates_category_mix, assign_templates_fixed_hash
from run_joint_integration_smoke import _json_default, run_one


LEVELS = {
    "neutral": 0.0,
    "low": 0.25,
    "medium": 0.5,
    "high": 1.0,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def git_text(*args):
    return subprocess.check_output(["git", *args], text=True, encoding="utf-8").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="joint_prototype/integration_smoke_config.json")
    parser.add_argument("--workload", default="joint_prototype/integration_smoke_workload.csv")
    parser.add_argument("--synergy-models", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).resolve()
    workload_path = Path(args.workload).resolve()
    models = Path(args.synergy_models).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    workload = pd.read_csv(workload_path)
    source_path = Path(config["source_workload"]).resolve()
    source = pd.read_csv(source_path)
    expected = source[source["name"].isin(config["workload_rows"])].reset_index(drop=True)
    if not workload.reset_index(drop=True).equals(expected):
        raise ValueError("audit workload differs from the declared Saturn extraction")

    sim_config.num_parallel_jobs = 1
    sim_config.num_parallel_traces = 1
    sim_config.sia_solver_verbose = False
    fixed_map = assign_templates_fixed_hash(
        list(workload["name"]),
        config["response"]["template_files"],
        int(config["response"]["hash_seed"]),
    )
    # Frozen before looking at the new JCT outcomes.  This is a design manifest,
    # not an assertion that categories are measurements of the Saturn jobs.
    category_plan = {
        "mix": {"responsive": 0.5, "flat": 0.5},
        "templates": {
            "responsive": ["alexnet_1.py", "m5_1.py", "res18_1.py", "shufflenet_1.py"],
            "flat": ["bert_1.py", "gnmt_1.py", "lstm_1.py", "transformer_1.py", "vae_1.py"],
        },
    }
    category_map = assign_templates_category_mix(
        list(workload["name"]), category_plan["templates"], category_plan["mix"]
    )

    records = []
    start = time.perf_counter()
    for level, exponent in LEVELS.items():
        for policy in ("b1", "b4", "b3"):
            result = run_one(args, config, workload, policy, level, response_exponent=exponent)
            filename = f"{policy}.{level}.json"
            (output / filename).write_text(
                json.dumps(result, indent=2, default=_json_default), encoding="utf-8"
            )
            rhos = [
                execution["rho"]
                for record in result["joint_round_logs"]
                for execution in record["execution"]
                if execution["training_seconds"] > 0
            ]
            records.append(
                {
                    "level": level,
                    "response_exponent": exponent,
                    "policy": policy,
                    "file": filename,
                    "average_jct": result["average_jct"],
                    "jcts": result["jcts"],
                    "rounds": result["rounds"],
                    "min_executed_rho": min(rhos) if rhos else None,
                    "max_executed_rho": max(rhos) if rhos else None,
                    "wall_seconds": result["wall_seconds"],
                }
            )
            print(level, policy, result["average_jct"], records[-1]["max_executed_rho"], flush=True)

    model_hashes = {path.name: sha256(path) for path in sorted(models.glob("*_1.py"))}
    code_paths = [
        Path("joint_prototype/joint_profile.py"),
        Path("joint_prototype/joint_integration.py"),
        Path("joint_prototype/joint_scheduler.py"),
        Path("run_joint_response_sensitivity.py"),
    ]
    command = (
        "python run_joint_response_sensitivity.py "
        f"--config {args.config} --workload {args.workload} "
        f"--synergy-models {args.synergy_models} --output {args.output}"
    )
    manifest = {
        "status": "SUCCESS",
        "label": "SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE",
        "pre_registered_response_levels": {
            key: {
                "alpha": value,
                "definition": "rho_alpha = exp(alpha * ln(raw_rho)) = raw_rho ** alpha",
            }
            for key, value in LEVELS.items()
        },
        "selection_rule": "levels fixed before inspecting these runs; no JCT-dependent cap or selection",
        "platform": platform.platform(),
        "python": sys.version,
        "workload": str(workload_path),
        "workload_sha256": sha256(workload_path),
        "source_workload": str(source_path),
        "source_workload_sha256": sha256(source_path),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "profile_directory": str(models),
        "profile_sha256": model_hashes,
        "mapping": {
            "mode_a_fixed_hash": {
                "seed": config["response"]["hash_seed"],
                "eligible_templates": config["response"]["template_files"],
                "result": fixed_map,
            },
            "mode_b_pre_registered_category_mix": {
                **category_plan,
                "two_job_demonstration_mapping": category_map,
                "status": "DESIGN_FROZEN_NOT_USED_FOR_THESE_RUNS",
            },
        },
        "demand_scaling": {
            "phi_c": "clip((raw_goodput/gpus) / minimum_candidate_per_gpu_goodput, 0.5, 4.0)",
            "phi_m": "clip(sqrt((raw_goodput/gpus) / minimum_candidate_per_gpu_goodput), 0.5, 2.0)",
            "origin": "project synthetic assumption; not Synergy measurement",
        },
        "node_capacities": config["cluster"]["synthetic_node_capacity"],
        "reference_point": config["response"]["fixed_reference"],
        "normalization_mode": config["response"]["normalization_mode"],
        "git": {
            "branch": git_text("branch", "--show-current"),
            "head": git_text("rev-parse", "HEAD"),
            "dirty_status": git_text("status", "--short"),
        },
        "code_sha256": {str(path): sha256(path) for path in code_paths},
        "command": command,
        "sequential_execution": True,
        "run_count": len(records),
        "duration_seconds": time.perf_counter() - start,
        "runs": records,
    }
    (output / "sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output / "sensitivity_summary.json").write_text(
        json.dumps({"label": manifest["label"], "runs": records}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "SUCCESS", "run_count": len(records)}, indent=2))


if __name__ == "__main__":
    main()
