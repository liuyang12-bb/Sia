"""Run the two-job Sia/Synergy end-to-end integration smoke test."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import simulator_config as sim_config
from cluster import Cluster
from sia import SiaPolicy
from joint_prototype.joint_integration import JointCluster, JointSiaPolicy
from joint_prototype.joint_profile import JointProfileOracle, assign_templates_fixed_hash


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _resource_hours(round_logs):
    totals = {"gpu_hours": 0.0, "cpu_core_hours": 0.0, "ram_gb_hours": 0.0}
    for record in round_logs:
        seconds = record["interval_end"] - record["interval_start"]
        for execution in record["execution"]:
            allocation = execution.get("resource_allocation")
            if not allocation or not execution["placement"]:
                continue
            active_seconds = seconds
            completion = execution.get("completion_time")
            if completion is not None and completion < record["interval_end"]:
                active_seconds = max(0.0, completion - record["interval_start"])
            totals["gpu_hours"] += allocation["gpus"] * active_seconds / 3600.0
            totals["cpu_core_hours"] += sum(x["cpu"] for x in allocation["node_usage"]) * active_seconds / 3600.0
            totals["ram_gb_hours"] += sum(x["ram_gb"] for x in allocation["node_usage"]) * active_seconds / 3600.0
    return totals


def run_one(args, config, workload, policy_name, response_mode, response_exponent=None):
    cluster_cfg = config["cluster"]
    sim_cfg = config["simulation"]
    response_cfg = config["response"]
    interval = int(sim_cfg["interval_seconds"])
    common_policy = dict(
        p_fairness=float(sim_cfg["p_fairness"]),
        restart_penalty=30,
        lambda_a=float(sim_cfg["lambda_a"]),
        lambda_n=float(sim_cfg["lambda_n"]),
        project_throughputs=bool(sim_cfg["project_throughputs"]),
        share_max_replicas=bool(sim_cfg["share_max_replicas"]),
        sched_interval=interval,
        enable_bsz_tuning=not bool(sim_cfg["disable_batch_size_tuning"]),
    )
    if policy_name == "sia":
        policy = SiaPolicy(**common_policy)
        cluster_class = Cluster
        template_map = {}
    else:
        template_map = assign_templates_fixed_hash(
            list(workload["name"]), response_cfg["template_files"], int(response_cfg["hash_seed"])
        )
        if response_exponent is None:
            response_exponent = 0.0 if response_mode == "neutral" else 1.0
        oracle = JointProfileOracle(
            Path(args.synergy_models),
            normalization_mode=response_cfg["normalization_mode"],
            fixed_reference=(
                float(response_cfg["fixed_reference"]["cpu_per_gpu"]),
                float(response_cfg["fixed_reference"]["ram_gb_per_gpu"]),
            ),
            neutral_response=response_mode == "neutral",
            response_exponent=float(response_exponent),
        )
        policy = JointSiaPolicy(
            mode=policy_name,
            oracle=oracle,
            node_capacities=cluster_cfg["synthetic_node_capacity"],
            template_map=template_map,
            aux_cpu_points=response_cfg["aux_cpu_points"],
            aux_ram_points=response_cfg["aux_ram_points"],
            **common_policy,
        )
        cluster_class = JointCluster
    policy.populate_valid_configs(
        cluster_cfg["nnodes"], cluster_cfg["ngpus_per_node"], cluster_cfg["max_physical_nodes"]
    )
    cluster = cluster_class(
        workload.copy(),
        policy,
        cluster_cfg["nnodes"],
        cluster_cfg["ngpus_per_node"],
        max_physical_nodes=cluster_cfg["max_physical_nodes"],
    )
    if sim_cfg["disable_batch_size_tuning"]:
        cluster.disable_bsz_tuning()
    for job in cluster.jobs:
        job.seed_profiles(
            int(sim_cfg["seed_profiles_num_nodes"]),
            int(sim_cfg["seed_profiles_num_replicas"]),
        )

    start = time.perf_counter()
    rounds = 0
    while not cluster.all_complete() and rounds < int(sim_cfg["max_rounds"]):
        cluster.step(interval)
        rounds += 1
    elapsed = time.perf_counter() - start
    if not cluster.all_complete():
        raise RuntimeError(f"simulation did not finish within {rounds} rounds")

    jcts = {job.name: job.completion_time - job.submission_time for job in cluster.jobs}
    jobs = {
        job.name: {
            "submission_time": job.submission_time,
            "start_time": job.start_time,
            "completion_time": job.completion_time,
            "jct": jcts[job.name],
            "final_progress": job.progress,
            "final_epoch": job.epoch,
            "num_restarts": job.num_restarts,
            "num_migrations": job.num_migrations,
            "used_gpu_seconds": job.used_gpu_seconds,
            "wasted_gpu_seconds": job.wasted_gpu_seconds,
        }
        for job in cluster.jobs
    }
    joint_rounds = cluster.joint_round_logs if isinstance(cluster, JointCluster) else []
    result = {
        "status": "SUCCESS",
        "label": config["disclaimer"] if policy_name != "sia" else "ORIGINAL_SIA_CONTROL_PATH",
        "policy": policy_name,
        "response_mode": response_mode if policy_name != "sia" else None,
        "response_exponent": response_exponent if policy_name != "sia" else None,
        "rounds": rounds,
        "simulated_seconds": cluster.current_time,
        "wall_seconds": elapsed,
        "all_jobs_completed": cluster.all_complete(),
        "average_jct": sum(jcts.values()) / len(jcts),
        "jcts": jcts,
        "jobs": jobs,
        "resource_hours": _resource_hours(joint_rounds),
        "template_map": template_map,
        "joint_round_logs": joint_rounds,
        "standard_logs": cluster.logs,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="joint_prototype/integration_smoke_config.json")
    parser.add_argument("--workload", default="joint_prototype/integration_smoke_workload.csv")
    parser.add_argument("--synergy-models", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy", choices=["sia", "b1", "b4", "b3", "joint-all", "all"], default="joint-all")
    parser.add_argument("--response", choices=["sensitive", "neutral", "both"], default="sensitive")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    workload_path = Path(args.workload).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    workload = pd.read_csv(workload_path)
    source = pd.read_csv(Path(config["source_workload"]).resolve())
    expected = source[source["name"].isin(config["workload_rows"])].reset_index(drop=True)
    if not workload.reset_index(drop=True).equals(expected):
        raise ValueError("smoke workload is not an exact extraction of the declared Saturn rows")

    sim_config.num_parallel_jobs = 1
    sim_config.num_parallel_traces = 1
    sim_config.sia_solver_verbose = False
    if args.policy == "joint-all":
        policies = ["b1", "b4", "b3"]
    elif args.policy == "all":
        policies = ["sia", "b1", "b4", "b3"]
    else:
        policies = [args.policy]
    responses = ["sensitive", "neutral"] if args.response == "both" else [args.response]
    runs = []
    overall_start = time.perf_counter()
    for response in responses:
        for policy_name in policies:
            if policy_name == "sia" and response != responses[0]:
                continue
            result = run_one(args, config, workload, policy_name, response)
            filename = f"{policy_name}.{response if policy_name != 'sia' else 'control'}.json"
            (output_dir / filename).write_text(
                json.dumps(result, indent=2, default=_json_default), encoding="utf-8"
            )
            runs.append({"policy": policy_name, "response": response, "file": filename, "summary": {
                "average_jct": result["average_jct"],
                "jcts": result["jcts"],
                "rounds": result["rounds"],
                "wall_seconds": result["wall_seconds"],
            }})
    manifest = {
        "status": "SUCCESS",
        "label": config["disclaimer"],
        "platform": platform.platform(),
        "python": sys.version,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "workload": str(workload_path),
        "workload_sha256": _sha256(workload_path),
        "source_workload": str(Path(config["source_workload"]).resolve()),
        "source_workload_sha256": _sha256(Path(config["source_workload"]).resolve()),
        "run_count": len(runs),
        "duration_seconds": time.perf_counter() - overall_start,
        "runs": runs,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
