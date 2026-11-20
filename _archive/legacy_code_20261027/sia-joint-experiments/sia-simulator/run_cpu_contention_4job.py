"""Run the pre-registered four-job auxiliary-resource contention matrix."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import psutil
import pandas as pd

import simulator_config as sim_config
from run_cpu_bottleneck_smoke import (
    POLICIES,
    hard_cpu_capacity_audit,
    json_default,
    run_one,
    sha256,
)


def execution_evidence(payload, config, capacity_name):
    capacities = config["cluster"]["capacity_scenarios"][capacity_name]
    gpu_caps = config["cluster"]["ngpus_per_node"]
    interval = float(config["simulation"]["interval_seconds"])
    first_arrival = min(float(row["submission_time"]) for row in payload["standard_logs"][-1]["submitted_jobs"])
    observation_seconds = max(float(payload["summary"]["simulated_seconds"]) - first_arrival, interval)
    node_seconds = defaultdict(lambda: {"gpu": 0.0, "cpu": 0.0, "ram": 0.0})
    wait_reason_rounds = defaultdict(int)
    cpu_idle_events = []
    multiplier_residual = 0.0
    goodput_ranges = defaultdict(lambda: {"raw": [], "effective": [], "rho": [], "predicted": []})

    for record in payload["joint_round_logs"]:
        duration = float(record["interval_end"] - record["interval_start"])
        for event in record["execution"]:
            allocation = event.get("resource_allocation")
            if allocation and event.get("placement"):
                active = duration
                completion = event.get("completion_time")
                if completion is not None and completion < record["interval_end"]:
                    active = max(0.0, completion - record["interval_start"])
                for usage in allocation["node_usage"]:
                    node_seconds[usage["node_id"]]["gpu"] += float(usage["gpus"]) * active
                    node_seconds[usage["node_id"]]["cpu"] += float(usage["cpu"]) * active
                    node_seconds[usage["node_id"]]["ram"] += float(usage["ram_gb"]) * active
                if event["training_seconds"] > 0:
                    residual = abs(
                        float(event["effective_goodput"])
                        - float(event["raw_goodput"]) * float(event["rho"])
                    )
                    multiplier_residual = max(multiplier_residual, residual)
                    values = goodput_ranges[event["job_id"]]
                    values["raw"].append(float(event["raw_goodput"]))
                    values["effective"].append(float(event["effective_goodput"]))
                    values["rho"].append(float(event["rho"]))
                    values["predicted"].append(float(allocation["predicted_effective_goodput"]))

        result = record["decision_for_next_interval"].get("result", {})
        audit = result.get("audit", {})
        totals = audit.get("node_totals", {})
        idle_due_cpu = []
        total_free_gpu = 0.0
        total_free_cpu = 0.0
        for node_id, usage in totals.items():
            gpu_type = node_id.rsplit(":", 1)[0]
            free_gpu = float(gpu_caps[gpu_type]) - float(usage["gpus"])
            free_cpu = float(capacities[gpu_type]["cpu_cores"]) - float(usage["cpu"])
            total_free_gpu += max(0.0, free_gpu)
            total_free_cpu += max(0.0, free_cpu)
            if free_gpu > 1e-9 and free_cpu < 1.0 - 1e-9:
                idle_due_cpu.append(
                    {"node_id": node_id, "idle_gpus": free_gpu, "free_cpu": free_cpu}
                )
        if idle_due_cpu:
            cpu_idle_events.append(
                {
                    "round": record["decision_for_next_interval"].get("round"),
                    "interval_end": record["interval_end"],
                    "nodes": idle_due_cpu,
                    "unallocated_jobs": result.get("unallocated_jobs", []),
                }
            )
        for _job_id in result.get("unallocated_jobs", []):
            if idle_due_cpu:
                wait_reason_rounds["CPU_SATURATED_WITH_IDLE_GPU"] += 1
            elif total_free_gpu < 1.0 - 1e-9:
                wait_reason_rounds["GPU_CAPACITY_FULL"] += 1
            elif total_free_cpu < 1.0 - 1e-9:
                wait_reason_rounds["CPU_CAPACITY_FULL"] += 1
            else:
                wait_reason_rounds["PLACEMENT_OR_CONFIG_FRAGMENTATION"] += 1

    node_utilization = {}
    for gpu_type, node_count in config["cluster"]["nnodes"].items():
        for idx in range(int(node_count)):
            node_id = f"{gpu_type}:{idx}"
            used = node_seconds[node_id]
            cap = capacities[gpu_type]
            node_utilization[node_id] = {
                "observation_seconds": observation_seconds,
                "gpu_utilization": used["gpu"] / (float(gpu_caps[gpu_type]) * observation_seconds),
                "cpu_utilization": used["cpu"] / (float(cap["cpu_cores"]) * observation_seconds),
                "ram_utilization": used["ram"] / (float(cap["ram_gb"]) * observation_seconds),
                "gpu_hours": used["gpu"] / 3600.0,
                "cpu_core_hours": used["cpu"] / 3600.0,
                "ram_gb_hours": used["ram"] / 3600.0,
            }

    ranges = {}
    for job_id, values in goodput_ranges.items():
        ranges[job_id] = {
            key: {"min": min(items), "max": max(items)} if items else None
            for key, items in values.items()
        }
    return {
        "node_time_weighted_utilization": node_utilization,
        "wait_reason_rounds": dict(wait_reason_rounds),
        "cpu_saturated_gpu_idle_event_count": len(cpu_idle_events),
        "cpu_saturated_gpu_idle_events": cpu_idle_events,
        "single_multiplier_max_abs_residual": multiplier_residual,
        "candidate_vs_execution_goodput_ranges": ranges,
        "goodput_units_note": "Candidate values are Sia-normalized scheduling Goodput; execution values are Job.step progress/second and are not numerically interchangeable.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="joint_prototype/cpu_contention_4job_config.json")
    parser.add_argument("--workload", default="joint_prototype/cpu_contention_4job_workload.csv")
    parser.add_argument("--synergy-models", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--capacities", default="A_cpu_sufficient,B_cpu_constrained,C_mixed_pressure")
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
        raise ValueError("workload is not an exact extraction of declared Saturn rows")

    memory = psutil.virtual_memory()
    threshold = float(config["simulation"]["minimum_available_memory_gib"]) * 1024**3
    if memory.available < threshold:
        raise RuntimeError(
            f"available memory {memory.available / 1024**3:.3f} GiB is below threshold"
        )
    sim_config.num_parallel_jobs = 1
    sim_config.num_parallel_traces = 1
    sim_config.sia_solver_verbose = False
    policies = tuple(item.strip() for item in args.policies.split(",") if item.strip())
    capacities = tuple(item.strip() for item in args.capacities.split(",") if item.strip())
    responses = tuple(item.strip() for item in args.responses.split(",") if item.strip())
    if not set(policies) <= set(POLICIES):
        raise ValueError("unsupported policy")
    if not set(capacities) <= set(config["cluster"]["capacity_scenarios"]):
        raise ValueError("unsupported capacity scenario")
    if not set(responses) <= set(config["response"]["response_settings"]):
        raise ValueError("unsupported response setting")

    start = time.perf_counter()
    records = []
    for capacity in capacities:
        for response in responses:
            for policy in policies:
                result = run_one(args, config, workload, policy, capacity, response)
                evidence = execution_evidence(result, config, capacity)
                result["contention_evidence"] = evidence
                filename = f"{capacity}.{response}.{policy}.json"
                (output / filename).write_text(
                    json.dumps(result, indent=2, default=json_default), encoding="utf-8"
                )
                records.append(
                    {
                        "capacity": capacity,
                        "response": response,
                        "policy": policy,
                        "file": filename,
                        **result["summary"],
                        "contention_evidence": {
                            "wait_reason_rounds": evidence["wait_reason_rounds"],
                            "cpu_saturated_gpu_idle_event_count": evidence["cpu_saturated_gpu_idle_event_count"],
                            "single_multiplier_max_abs_residual": evidence["single_multiplier_max_abs_residual"],
                        },
                    }
                )
                print(
                    f"{capacity}/{response}/{policy}: "
                    f"completed={result['summary']['completed_jobs']}/{len(workload)}, "
                    f"avg_jct={result['summary']['average_jct_completed_only']}, "
                    f"wait={sum(result['summary']['waiting_seconds'].values())}"
                )

    code_root = Path(__file__).resolve().parents[1]
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=code_root, text=True
    ).strip()
    models = Path(args.synergy_models).resolve()
    templates = sorted(set(config["response"]["template_map"].values()))
    manifest = {
        "status": "COMPLETE",
        "label": config["disclaimer"],
        "experiment_scope": config["experiment_scope"],
        "platform": platform.platform(),
        "python": sys.version,
        "git_head": git_head,
        "available_memory_gib_at_start": memory.available / 1024**3,
        "memory_threshold_gib": threshold / 1024**3,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "workload": str(workload_path),
        "workload_sha256": sha256(workload_path),
        "source_workload": str(source_path),
        "source_workload_sha256": sha256(source_path),
        "profile_hashes": {name: sha256(models / name) for name in templates},
        "hard_cpu_capacity_audit": hard_cpu_capacity_audit(config),
        "commands": [" ".join(sys.argv)],
        "run_count": len(records),
        "duration_seconds": time.perf_counter() - start,
        "runs": records,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=json_default), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, default=json_default))


if __name__ == "__main__":
    main()
