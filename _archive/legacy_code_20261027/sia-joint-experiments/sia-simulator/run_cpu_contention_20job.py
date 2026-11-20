"""Run the frozen 20-job stability smoke sequentially.

This driver deliberately runs one policy at a time, refuses to start below the
2 GiB memory gate, and writes a progress manifest after every completed policy.
It is a stability/mechanism smoke, not a performance experiment.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import re
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import pandas as pd
import psutil

import simulator_config as sim_config
from run_cpu_bottleneck_smoke import POLICIES, hard_cpu_capacity_audit, json_default, run_one, sha256
from run_cpu_contention_4job import execution_evidence


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=json_default), encoding="utf-8")


def available_memory_gib() -> float:
    return psutil.virtual_memory().available / 1024**3


def validate_frozen_workload(config, workload, source):
    expected = source[source["name"].isin(config["workload_rows"])].reset_index(drop=True)
    if not workload.reset_index(drop=True).equals(expected):
        raise ValueError("20-job workload is not an exact ordered extraction of declared Saturn rows")
    if list(workload["name"]) != config["workload_rows"]:
        raise ValueError("workload order differs from frozen workload_rows")
    numeric_ids = [int(re.search(r"-(\d+)$", name).group(1)) for name in workload["name"]]
    if numeric_ids != list(range(20, 40)):
        raise ValueError(f"cohort is not the frozen consecutive ID range 20..39: {numeric_ids}")
    template_map = config["response"]["template_map"]
    if set(template_map) != set(workload["name"]):
        raise ValueError("template map does not cover exactly the frozen cohort")
    type_map = config["synergy_adaptation"]["fixed_gpu_type_map"]
    demand_map = config["synergy_adaptation"]["fixed_gpu_demand_map"]
    if set(type_map) != set(workload["name"]) or set(demand_map) != set(workload["name"]):
        raise ValueError("Synergy fixed maps do not cover exactly the frozen cohort")
    if any(int(row.num_replicas) != int(demand_map[row.name]) for row in workload.itertuples()):
        raise ValueError("Synergy demand map differs from source workload demand")
    return {
        "cohort_rule": config["cohort_rule"],
        "job_count": len(workload),
        "numeric_job_ids": numeric_ids,
        "arrival_min": float(workload["time"].min()),
        "arrival_max": float(workload["time"].max()),
        "arrival_span_seconds": float(workload["time"].max() - workload["time"].min()),
        "application_counts": dict(sorted(Counter(workload["application"]).items())),
        "gpu_demand_counts": {
            str(key): int(value)
            for key, value in sorted(Counter(int(x) for x in workload["num_replicas"]).items())
        },
        "synergy_gpu_type_counts": dict(sorted(Counter(type_map.values()).items())),
        "mapping_complete": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="joint_prototype/cpu_contention_20job_config.json")
    parser.add_argument("--workload", default="joint_prototype/cpu_contention_20job_workload.csv")
    parser.add_argument("--synergy-models", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--capacities", default="cpu_constrained")
    parser.add_argument("--responses", default="neutral")
    parser.add_argument(
        "--targeted-b3", action="store_true",
        help="Run only the frozen B3 case for post-repair validation.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    workload_path = Path(args.workload).resolve()
    models = Path(args.synergy_models).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite historical run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    workload = pd.read_csv(workload_path)
    source_path = Path(config["source_workload"]).resolve()
    source = pd.read_csv(source_path)
    cohort = validate_frozen_workload(config, workload, source)

    policies = tuple(item.strip() for item in args.policies.split(",") if item.strip())
    capacities = tuple(item.strip() for item in args.capacities.split(",") if item.strip())
    responses = tuple(item.strip() for item in args.responses.split(",") if item.strip())
    if not set(policies) <= set(POLICIES):
        raise ValueError("unsupported policy")
    if not set(capacities) <= set(config["cluster"]["capacity_scenarios"]):
        raise ValueError("unsupported capacity scenario")
    if not set(responses) <= set(config["response"]["response_settings"]):
        raise ValueError("unsupported response setting")

    planned = config["planned_runs"]["formal_first_smoke"]
    expected_policies = ["b3"] if args.targeted_b3 else planned["policies"]
    if list(capacities) != planned["capacities"] or list(responses) != planned["responses"] or list(policies) != expected_policies:
        raise ValueError("requested matrix differs from frozen formal_first_smoke")

    template_files = sorted(set(config["response"]["template_map"].values()))
    missing_profiles = [name for name in template_files if not (models / name).is_file()]
    if missing_profiles:
        raise FileNotFoundError(f"missing Synergy profiles: {missing_profiles}")
    threshold_gib = float(config["simulation"]["minimum_available_memory_gib"])
    memory_at_preflight = available_memory_gib()

    code_root = Path(__file__).resolve().parents[1]
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code_root, text=True).strip()
    preflight = {
        "status": "PASS" if memory_at_preflight >= threshold_gib else "BLOCKED_MEMORY_GATE",
        "label": config["disclaimer"],
        "experiment_scope": config["experiment_scope"],
        "cohort": cohort,
        "available_memory_gib": memory_at_preflight,
        "minimum_available_memory_gib": threshold_gib,
        "config_sha256": sha256(config_path),
        "workload_sha256": sha256(workload_path),
        "source_workload_sha256": sha256(source_path),
        "profile_hashes": {name: sha256(models / name) for name in template_files},
        "hard_cpu_capacity_audit": hard_cpu_capacity_audit(config),
        "frozen_matrix": {"capacities": list(capacities), "responses": list(responses), "policies": list(policies)},
        "run_kind": "TARGETED_B3_REPAIR_VALIDATION" if args.targeted_b3 else "FULL_FIVE_POLICY_MATRIX",
        "deferred_medium": config["planned_runs"]["deferred_sensitivity"],
        "git_head": git_head,
        "command": " ".join(sys.argv),
    }
    write_json(output / "preflight.json", preflight)
    if memory_at_preflight < threshold_gib:
        write_json(output / "run_manifest.json", {
            "status": "ABORTED_MEMORY_GATE",
            "label": config["disclaimer"],
            "experiment_scope": config["experiment_scope"],
            "git_head": git_head,
            "available_memory_samples": [
                {"phase": "preflight", "available_gib": memory_at_preflight}
            ],
            "memory_threshold_gib": threshold_gib,
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "workload": str(workload_path),
            "workload_sha256": sha256(workload_path),
            "source_workload": str(source_path),
            "source_workload_sha256": sha256(source_path),
            "profile_hashes": preflight["profile_hashes"],
            "cohort": cohort,
            "hard_cpu_capacity_audit": preflight["hard_cpu_capacity_audit"],
            "frozen_matrix": preflight["frozen_matrix"],
            "deferred_medium": preflight["deferred_medium"],
            "commands": [" ".join(sys.argv)],
            "completed_run_count": 0,
            "expected_run_count": len(capacities) * len(responses) * len(policies),
            "runs": [],
            "blocking_reason": (
                f"available memory {memory_at_preflight:.3f} GiB is below frozen "
                f"{threshold_gib:.3f} GiB gate"
            ),
        })
        raise RuntimeError(
            f"available memory {memory_at_preflight:.3f} GiB is below frozen {threshold_gib:.3f} GiB gate"
        )

    sim_config.num_parallel_jobs = 1
    sim_config.num_parallel_traces = 1
    sim_config.sia_solver_verbose = False
    start = time.perf_counter()
    records = []
    failure_record = None
    memory_samples = [{"phase": "preflight", "available_gib": memory_at_preflight}]

    def write_progress(status):
        manifest = {
            "status": status,
            "label": config["disclaimer"],
            "experiment_scope": config["experiment_scope"],
            "platform": platform.platform(),
            "python": sys.version,
            "git_head": git_head,
            "available_memory_samples": memory_samples,
            "memory_threshold_gib": threshold_gib,
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "workload": str(workload_path),
            "workload_sha256": sha256(workload_path),
            "source_workload": str(source_path),
            "source_workload_sha256": sha256(source_path),
            "profile_hashes": preflight["profile_hashes"],
            "cohort": cohort,
            "hard_cpu_capacity_audit": preflight["hard_cpu_capacity_audit"],
            "frozen_matrix": preflight["frozen_matrix"],
            "deferred_medium": preflight["deferred_medium"],
            "commands": [" ".join(sys.argv)],
            "completed_run_count": len(records),
            "expected_run_count": len(capacities) * len(responses) * len(policies),
            "duration_seconds": time.perf_counter() - start,
            "runs": records,
            "failure": failure_record,
        }
        write_json(output / "run_manifest.json", manifest)

    write_progress("RUNNING")
    for capacity in capacities:
        for response in responses:
            for policy in policies:
                gc.collect()
                available = available_memory_gib()
                memory_samples.append({
                    "phase": "before_run", "capacity": capacity, "response": response,
                    "policy": policy, "available_gib": available,
                })
                write_progress("RUNNING")
                if available < threshold_gib:
                    write_progress("ABORTED_MEMORY_GATE")
                    raise RuntimeError(
                        f"before {capacity}/{response}/{policy}, available memory {available:.3f} GiB "
                        f"fell below frozen {threshold_gib:.3f} GiB gate"
                    )
                target = output / f"{capacity}.{response}.{policy}.json"
                if target.exists():
                    raise FileExistsError(f"refusing to overwrite historical artifact: {target}")
                try:
                    result = run_one(args, config, workload, policy, capacity, response)
                except Exception as exc:
                    failure_record = {
                        "capacity": capacity,
                        "response": response,
                        "policy": policy,
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                        "traceback": traceback.format_exc(),
                        "diagnostic": getattr(exc, "joint_diagnostic", None),
                    }
                    failure_name = f"{capacity}.{response}.{policy}.failure.json"
                    write_json(output / failure_name, failure_record)
                    failure_record["file"] = failure_name
                    write_progress("FAILED_POLICY_EXECUTION")
                    raise
                evidence = execution_evidence(result, config, capacity)
                result["contention_evidence"] = evidence
                write_json(target, result)
                records.append({
                    "capacity": capacity,
                    "response": response,
                    "policy": policy,
                    "file": target.name,
                    **result["summary"],
                    "contention_evidence": {
                        "wait_reason_rounds": evidence["wait_reason_rounds"],
                        "cpu_saturated_gpu_idle_event_count": evidence["cpu_saturated_gpu_idle_event_count"],
                        "single_multiplier_max_abs_residual": evidence["single_multiplier_max_abs_residual"],
                    },
                })
                print(
                    f"{capacity}/{response}/{policy}: completed={result['summary']['completed_jobs']}/20, "
                    f"avg_jct={result['summary']['average_jct_completed_only']}, "
                    f"rounds={result['summary']['rounds']}, wall={result['summary']['wall_seconds']:.3f}s",
                    flush=True,
                )
                del result
                gc.collect()
                write_progress("RUNNING")
    write_progress("COMPLETE")


if __name__ == "__main__":
    main()
