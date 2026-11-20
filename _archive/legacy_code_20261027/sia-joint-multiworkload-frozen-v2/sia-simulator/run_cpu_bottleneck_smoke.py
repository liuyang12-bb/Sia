"""Run the pre-registered two-job CPU-bottleneck mechanism matrix.

This is deliberately not a 20-job performance experiment.  It executes the
same two Saturn workload rows under two synthetic capacity scenarios, two
response settings, and B1/SY/B4/B5/B3 sequentially.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import simulator_config as sim_config
from joint_prototype.joint_integration import JointCluster, JointSiaPolicy
from joint_prototype.joint_profile import JointProfileOracle


POLICIES = ("b1", "sy", "b4", "b5", "b3")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(type(value).__name__)


def hard_cpu_capacity_audit(config):
    cluster = config["cluster"]
    minimum = float(config["response"]["minimum_legal_profile_point"]["cpu_per_gpu"])
    result = {}
    for scenario, capacities in cluster["capacity_scenarios"].items():
        result[scenario] = {}
        for gpu_type, gpu_count in cluster["ngpus_per_node"].items():
            cpu = float(capacities[gpu_type]["cpu_cores"])
            max_gpus = min(int(gpu_count), int(np.floor(cpu / minimum + 1e-12)))
            result[scenario][gpu_type] = {
                "node_gpus": int(gpu_count),
                "node_cpu_cores": cpu,
                "minimum_cpu_per_gpu": minimum,
                "max_gpus_usable_by_cpu_floor": max_gpus,
                "gpus_forced_idle_if_node_otherwise_full": int(gpu_count) - max_gpus,
                "full_node_gpu_config_cpu_feasible": max_gpus == int(gpu_count),
            }
    return result


def summarize(cluster, wall_seconds: float, max_rounds: int, capacities):
    completed = [job for job in cluster.jobs if job.completion_time is not None]
    jcts = {
        job.name: float(job.completion_time - job.submission_time)
        for job in completed
    }
    waits = {job.name: 0.0 for job in cluster.jobs}
    resource_hours = {"gpu_hours": 0.0, "cpu_core_hours": 0.0, "ram_gb_hours": 0.0}
    max_rho = 0.0
    min_rho = None
    rho_by_job = {job.name: [] for job in cluster.jobs}
    node_peaks = {
        f"{gpu_type}:{idx}": {"gpus": 0.0, "cpu": 0.0, "ram_gb": 0.0}
        for gpu_type, count in cluster.num_nodes.items()
        for idx in range(count)
    }
    capacity_violations = []
    tune_actions = []
    fallback_events = []
    solve_seconds = []
    gpu_config_history = {job.name: [] for job in cluster.jobs}
    for record in cluster.joint_round_logs:
        duration = float(record["interval_end"] - record["interval_start"])
        for event in record["execution"]:
            if event["submitted"] and not event["placement"] and event["completion_time"] is None:
                waits[event["job_id"]] += duration
            allocation = event.get("resource_allocation")
            if allocation and event["placement"]:
                active = duration
                completion = event.get("completion_time")
                if completion is not None and completion < record["interval_end"]:
                    active = max(0.0, completion - record["interval_start"])
                resource_hours["gpu_hours"] += allocation["gpus"] * active / 3600.0
                resource_hours["cpu_core_hours"] += sum(
                    item["cpu"] for item in allocation["node_usage"]
                ) * active / 3600.0
                resource_hours["ram_gb_hours"] += sum(
                    item["ram_gb"] for item in allocation["node_usage"]
                ) * active / 3600.0
                max_rho = max(max_rho, float(allocation["rho"]))
                min_rho = (
                    float(allocation["rho"])
                    if min_rho is None
                    else min(min_rho, float(allocation["rho"]))
                )
                rho_by_job[event["job_id"]].append(float(allocation["rho"]))
        decision = record["decision_for_next_interval"]
        result = decision.get("result", {})
        audit = result.get("audit", {})
        if result and not audit.get("valid", False):
            capacity_violations.append(
                {"round": decision.get("round"), "violations": audit.get("violations", [])}
            )
        for node_id, totals in audit.get("node_totals", {}).items():
            peak = node_peaks[node_id]
            peak["gpus"] = max(peak["gpus"], float(totals["gpus"]))
            peak["cpu"] = max(peak["cpu"], float(totals["cpu"]))
            peak["ram_gb"] = max(peak["ram_gb"], float(totals["ram_gb"]))
        tune_actions.extend(audit.get("tune_trace", []))
        if audit.get("fallback_reason"):
            fallback_events.append(
                {"round": decision.get("round"), "reason": audit["fallback_reason"]}
            )
        solve_seconds.append(float(result.get("solve_seconds", 0.0)))
        selected = result.get("selected", {})
        for job_id in gpu_config_history:
            gpu_config_history[job_id].append(
                selected.get(job_id, {}).get("config", {}).get("config_id")
            )
    gpu_changes = {}
    for job_id, history in gpu_config_history.items():
        active = [config for config in history if config is not None]
        gpu_changes[job_id] = sum(a != b for a, b in zip(active, active[1:]))
    return {
        "all_jobs_completed": cluster.all_complete(),
        "completed_jobs": len(completed),
        "total_jobs": len(cluster.jobs),
        "completion_rate": len(completed) / len(cluster.jobs),
        "average_jct_completed_only": (
            sum(jcts.values()) / len(jcts) if jcts else None
        ),
        "jcts": jcts,
        "completion_times": {
            job.name: job.completion_time for job in cluster.jobs
        },
        "waiting_seconds": waits,
        "simulated_seconds": float(cluster.current_time),
        "rounds": len(cluster.joint_round_logs),
        "hit_round_limit": len(cluster.joint_round_logs) >= max_rounds and not cluster.all_complete(),
        "wall_seconds": wall_seconds,
        "resource_hours": resource_hours,
        "max_rho": max_rho,
        "min_rho": min_rho,
        "rho_range_by_job": {
            job_id: {
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
            for job_id, values in rho_by_job.items()
        },
        "node_peak_aggregate_usage": node_peaks,
        "node_capacity": capacities,
        "capacity_violations": capacity_violations,
        "tune_action_counts": {
            action: sum(item["action"] == action for item in tune_actions)
            for action in sorted({item["action"] for item in tune_actions})
        },
        "fallback_events": fallback_events,
        "gpu_config_changes": gpu_changes,
        "optimizer_total_seconds": sum(solve_seconds),
        "optimizer_p99_seconds": float(np.percentile(solve_seconds, 99)) if solve_seconds else 0.0,
        "jobs": {
            job.name: {
                "start_time": job.start_time,
                "completion_time": job.completion_time,
                "progress": float(job.progress),
                "epoch": int(job.epoch),
                "num_restarts": int(job.num_restarts),
                "num_migrations": int(job.num_migrations or 0),
                "used_gpu_seconds": float(job.used_gpu_seconds),
                "wasted_gpu_seconds": float(job.wasted_gpu_seconds),
            }
            for job in cluster.jobs
        },
    }


def run_one(args, config, workload, policy_name, capacity_name, response_name):
    cluster_cfg = config["cluster"]
    response_cfg = config["response"]
    simulation = config["simulation"]
    exponent = float(response_cfg["response_settings"][response_name]["exponent"])
    oracle = JointProfileOracle(
        Path(args.synergy_models),
        normalization_mode=response_cfg["normalization_mode"],
        fixed_reference=(
            response_cfg["fixed_reference"]["cpu_per_gpu"],
            response_cfg["fixed_reference"]["ram_gb_per_gpu"],
        ),
        neutral_response=response_name == "neutral",
        response_exponent=exponent,
    )
    policy = JointSiaPolicy(
        mode=policy_name,
        oracle=oracle,
        node_capacities=cluster_cfg["capacity_scenarios"][capacity_name],
        template_map=response_cfg["template_map"],
        aux_cpu_points=response_cfg["aux_cpu_points"],
        aux_ram_points=response_cfg["aux_ram_points"],
        synergy_gpu_type_map=config["synergy_adaptation"]["fixed_gpu_type_map"],
        synergy_gpu_demand_map=config["synergy_adaptation"]["fixed_gpu_demand_map"],
        minimum_cpu_per_gpu=response_cfg["minimum_legal_profile_point"]["cpu_per_gpu"],
        minimum_ram_gb_per_gpu=response_cfg["minimum_legal_profile_point"]["ram_gb_per_gpu"],
        p_fairness=float(simulation["p_fairness"]),
        restart_penalty=30,
        lambda_a=float(simulation["lambda_a"]),
        lambda_n=float(simulation["lambda_n"]),
        project_throughputs=bool(simulation["project_throughputs"]),
        share_max_replicas=bool(simulation["share_max_replicas"]),
        sched_interval=int(simulation["interval_seconds"]),
        enable_bsz_tuning=not bool(simulation["disable_batch_size_tuning"]),
    )
    policy.populate_valid_configs(
        cluster_cfg["nnodes"], cluster_cfg["ngpus_per_node"], cluster_cfg["max_physical_nodes"]
    )
    cluster = JointCluster(
        workload.copy(),
        policy,
        cluster_cfg["nnodes"],
        cluster_cfg["ngpus_per_node"],
        max_physical_nodes=cluster_cfg["max_physical_nodes"],
    )
    if simulation["disable_batch_size_tuning"]:
        cluster.disable_bsz_tuning()
    for job in cluster.jobs:
        job.seed_profiles(
            int(simulation["seed_profiles_num_nodes"]),
            int(simulation["seed_profiles_num_replicas"]),
        )
    start = time.perf_counter()
    rounds = 0
    max_rounds = int(simulation["max_rounds"])
    while not cluster.all_complete() and rounds < max_rounds:
        try:
            cluster.step(int(simulation["interval_seconds"]))
        except Exception as exc:
            exc.joint_diagnostic = {
                "policy": policy_name,
                "capacity_scenario": capacity_name,
                "response_setting": response_name,
                "cluster_current_time": float(cluster.current_time),
                "completed_rounds": rounds,
                "policy_round_index": int(policy.round_index),
                "policy_last_failure": policy.last_failure,
                "cluster_allocations": cluster.allocations,
                "current_resources": policy.current_resources,
                "jobs": {
                    job.name: {
                        "completion_time": job.completion_time,
                        "current_cluster": job.current_cluster,
                        "placement": list(job.placement),
                        "rescale_time": float(job.rescale_time),
                        "progress": float(job.progress),
                    }
                    for job in cluster.jobs
                },
            }
            raise
        rounds += 1
    wall = time.perf_counter() - start
    return {
        "status": "SUCCESS" if cluster.all_complete() else "ROUND_LIMIT_WITH_UNFINISHED_JOBS",
        "label": config["disclaimer"],
        "policy": policy_name,
        "capacity_scenario": capacity_name,
        "response_setting": response_name,
        "response_exponent": exponent,
        "template_map": response_cfg["template_map"],
        "synthetic_node_capacity": cluster_cfg["capacity_scenarios"][capacity_name],
        "summary": summarize(
            cluster, wall, max_rounds, cluster_cfg["capacity_scenarios"][capacity_name]
        ),
        "joint_round_logs": cluster.joint_round_logs,
        "standard_logs": cluster.logs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="joint_prototype/cpu_bottleneck_smoke_config.json")
    parser.add_argument("--workload", default="joint_prototype/cpu_bottleneck_smoke_workload.csv")
    parser.add_argument("--synergy-models", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--capacities", default="cpu_sufficient,cpu_constrained")
    parser.add_argument("--responses", default="neutral,medium")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    workload_path = Path(args.workload).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    workload = pd.read_csv(workload_path)
    source_path = Path(config["source_workload"]).resolve()
    source = pd.read_csv(source_path)
    expected = source[source["name"].isin(config["workload_rows"])].reset_index(drop=True)
    if not workload.reset_index(drop=True).equals(expected):
        raise ValueError("smoke workload is not an exact extraction of declared Saturn rows")

    sim_config.num_parallel_jobs = 1
    sim_config.num_parallel_traces = 1
    sim_config.sia_solver_verbose = False
    run_records = []
    start = time.perf_counter()
    policies = tuple(item.strip() for item in args.policies.split(",") if item.strip())
    capacities = tuple(item.strip() for item in args.capacities.split(",") if item.strip())
    responses = tuple(item.strip() for item in args.responses.split(",") if item.strip())
    if not set(policies) <= set(POLICIES):
        raise ValueError(f"unsupported policies: {set(policies) - set(POLICIES)}")
    if not set(capacities) <= set(config["cluster"]["capacity_scenarios"]):
        raise ValueError("unsupported capacity scenario")
    if not set(responses) <= set(config["response"]["response_settings"]):
        raise ValueError("unsupported response setting")
    for capacity in capacities:
        for response in responses:
            for policy in policies:
                result = run_one(args, config, workload, policy, capacity, response)
                filename = f"{capacity}.{response}.{policy}.json"
                (output / filename).write_text(
                    json.dumps(result, indent=2, default=json_default), encoding="utf-8"
                )
                run_records.append(
                    {
                        "capacity": capacity,
                        "response": response,
                        "policy": policy,
                        "file": filename,
                        **result["summary"],
                    }
                )
                print(
                    f"{capacity}/{response}/{policy}: "
                    f"completed={result['summary']['completed_jobs']}/2, "
                    f"avg_jct={result['summary']['average_jct_completed_only']}"
                )

    code_root = Path(__file__).resolve().parents[1]
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=code_root, text=True
    ).strip()
    profile_files = sorted(set(config["response"]["template_map"].values()))
    models = Path(args.synergy_models).resolve()
    manifest = {
        "status": "COMPLETE",
        "label": config["disclaimer"],
        "experiment_scope": "TWO_JOB_MECHANISM_SMOKE_NOT_20_JOB_PERFORMANCE_EXPERIMENT",
        "platform": platform.platform(),
        "python": sys.version,
        "git_head": git_head,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "workload": str(workload_path),
        "workload_sha256": sha256(workload_path),
        "source_workload": str(source_path),
        "source_workload_sha256": sha256(source_path),
        "profile_hashes": {
            filename: sha256(models / filename) for filename in profile_files
        },
        "hard_cpu_capacity_audit": hard_cpu_capacity_audit(config),
        "commands": [" ".join(sys.argv)],
        "run_count": len(run_records),
        "duration_seconds": time.perf_counter() - start,
        "runs": run_records,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=json_default), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, default=json_default))


if __name__ == "__main__":
    main()
