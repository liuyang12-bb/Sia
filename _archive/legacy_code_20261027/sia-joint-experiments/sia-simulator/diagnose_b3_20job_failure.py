"""Reproduce the frozen B3 failure while adding diagnostics only.

This script does not change solver inputs, objective, constraints, or fallback
semantics.  It wraps JointScheduler.solve and persists the first non-valid
result before the production atomic guard raises.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import psutil

import run_cpu_bottleneck_smoke as smoke
from joint_prototype.joint_integration import JointCluster, JointSiaPolicy


DIAGNOSTIC_PATH = None
LAST_POLICY = None
LAST_CLUSTER = None


def json_default(value):
    return smoke.json_default(value)


def previous_resource_audit(policy, nodes):
    totals = defaultdict(lambda: {"gpus": 0.0, "cpu": 0.0, "ram_gb": 0.0})
    violations = []
    for job_id, resource in policy.current_resources.items():
        for item in resource.get("node_usage", []):
            node = totals[item["node_id"]]
            node["gpus"] += float(item["gpus"])
            node["cpu"] += float(item["cpu"])
            node["ram_gb"] += float(item["ram_gb"])
    node_caps = {node.node_id: node for node in nodes}
    for node_id, usage in totals.items():
        capacity = node_caps[node_id]
        if usage["gpus"] > float(capacity.gpus) + 1e-9:
            violations.append(f"{node_id}:gpu")
        if usage["cpu"] > float(capacity.cpu_cores) + 1e-9:
            violations.append(f"{node_id}:cpu")
        if usage["ram_gb"] > float(capacity.ram_gb) + 1e-9:
            violations.append(f"{node_id}:ram")
    return {"valid": not violations, "violations": violations, "node_totals": dict(totals)}


class DiagnosticPolicy(JointSiaPolicy):
    def __init__(self, *args, **kwargs):
        global LAST_POLICY
        super().__init__(*args, **kwargs)
        LAST_POLICY = self

    def _scheduler(self, nodes, specs):
        scheduler = super()._scheduler(nodes, specs)
        original = scheduler.solve

        def solve(policy_name, columns, dimensions):
            result = original(policy_name, columns, dimensions)
            if not result.audit.get("valid", False):
                counts = Counter(column.job_id for column in columns)
                locked = {
                    spec.job_id: {
                        "locked_config": spec.current_config_id,
                        "locked_nodes": list(spec.current_nodes),
                    }
                    for spec in specs if spec.checkpoint_locked
                }
                previous = previous_resource_audit(self, nodes)
                locked_preserved = all(
                    job_id in self.current_resources
                    and self.current_resources[job_id]["config_id"] == item["locked_config"]
                    and sorted(self.current_resources[job_id]["placement"]) == sorted(item["locked_nodes"])
                    for job_id, item in locked.items()
                )
                snapshot = {
                    "policy": policy_name,
                    "policy_round_index": self.round_index,
                    "result_status": result.status,
                    "result_audit": result.audit,
                    "result_unallocated_jobs": result.unallocated_jobs,
                    "dimensions": list(dimensions),
                    "candidate_column_count": len(columns),
                    "candidate_counts_by_job": dict(sorted(counts.items())),
                    "active_jobs": [spec.job_id for spec in specs],
                    "jobs_without_candidates": sorted(set(spec.job_id for spec in specs) - set(counts)),
                    "job_state": {
                        spec.job_id: {
                            "checkpoint_locked": spec.checkpoint_locked,
                            "current_config_id": spec.current_config_id,
                            "current_nodes": list(spec.current_nodes),
                            "legal_gpu_config_count": sum(config.legal for config in spec.configs),
                        }
                        for spec in specs
                    },
                    "checkpoint_locked_jobs": locked,
                    "previous_resources": self.current_resources,
                    "previous_resource_audit": previous,
                    "previous_allocation_preserves_all_locks": locked_preserved,
                }
                DIAGNOSTIC_PATH.write_text(
                    json.dumps(snapshot, indent=2, default=json_default), encoding="utf-8"
                )
            return result

        scheduler.solve = solve
        return scheduler


class DiagnosticCluster(JointCluster):
    def __init__(self, *args, **kwargs):
        global LAST_CLUSTER
        super().__init__(*args, **kwargs)
        LAST_CLUSTER = self


def main():
    global DIAGNOSTIC_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--synergy-models", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty diagnostic output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    DIAGNOSTIC_PATH = output / "solve_failure_snapshot.json"
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    workload = pd.read_csv(args.workload)
    available = psutil.virtual_memory().available / 1024**3
    threshold = float(config["simulation"]["minimum_available_memory_gib"])
    if available < threshold:
        raise RuntimeError(f"available memory {available:.3f} GiB below {threshold:.3f} GiB")
    smoke.JointSiaPolicy = DiagnosticPolicy
    smoke.JointCluster = DiagnosticCluster
    run_args = SimpleNamespace(synergy_models=args.synergy_models)
    log_path = output / "b3_diagnostic_console.log"
    summary = {
        "status": "RUNNING",
        "available_memory_gib": available,
        "config": str(Path(args.config).resolve()),
        "workload": str(Path(args.workload).resolve()),
    }
    try:
        with log_path.open("w", encoding="utf-8") as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            smoke.run_one(run_args, config, workload, "b3", "cpu_constrained", "neutral")
        summary["status"] = "UNEXPECTED_SUCCESS"
    except Exception as exc:
        summary.update({
            "status": "REPRODUCED_FAILURE",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "cluster_current_time": getattr(LAST_CLUSTER, "current_time", None),
            "completed_jobs": sum(job.completion_time is not None for job in LAST_CLUSTER.jobs) if LAST_CLUSTER else None,
            "policy_round_index": getattr(LAST_POLICY, "round_index", None),
            "snapshot_file": DIAGNOSTIC_PATH.name if DIAGNOSTIC_PATH.is_file() else None,
        })
    (output / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=json_default))


if __name__ == "__main__":
    main()
