"""Run a tiny CPU-only Synergy-TUNE reference simulation without editing Synergy.

The official checkout imports Linux NUMA and deployment RPC modules even when
only the simulator is used.  On Windows the compatibility modules below are
import-only shims; no deployment or NUMA code is exercised by this smoke test.
"""

from __future__ import annotations

import argparse
import collections
import collections.abc
import json
import logging
import random
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SIMULATOR = ROOT / "research_sources" / "Synergy" / "simulator"


def install_import_compatibility() -> None:
    """Install import-only compatibility for Python 3.10 and Windows."""
    collections.Mapping = collections.abc.Mapping  # type: ignore[attr-defined]
    sys.modules.setdefault("numa", types.SimpleNamespace())

    deployment = types.ModuleType("deployment")
    deployment.__path__ = []  # type: ignore[attr-defined]
    runtime = types.ModuleType("deployment.runtime")
    runtime.__path__ = []  # type: ignore[attr-defined]
    rpc = types.ModuleType("deployment.runtime.rpc")
    rpc.__path__ = []  # type: ignore[attr-defined]
    scheduler_server = types.ModuleType(
        "deployment.runtime.rpc.scheduler_server"
    )
    scheduler_client = types.ModuleType(
        "deployment.runtime.rpc.scheduler_client"
    )
    helper = types.ModuleType("deployment.helper")
    helper.get_self_ip = lambda: "127.0.0.1"  # type: ignore[attr-defined]
    sys.modules.update(
        {
            "deployment": deployment,
            "deployment.runtime": runtime,
            "deployment.runtime.rpc": rpc,
            "deployment.runtime.rpc.scheduler_server": scheduler_server,
            "deployment.runtime.rpc.scheduler_client": scheduler_client,
            "deployment.helper": helper,
        }
    )


def run_smoke(num_jobs: int = 5, mode: str = "tune") -> dict:
    if mode not in {"fair", "tune"}:
        raise ValueError("mode must be 'fair' or 'tune'")
    install_import_compatibility()
    sys.path.insert(0, str(SIMULATOR))
    random.seed(42)

    from runner import Runner

    runner = Runner(
        cluster_job_log=None,
        scheduler="FIFO",
        jobs_per_hour=9,
        series_id_filter=(0, num_jobs),
        model_class_split=(20, 70, 10),
        exponential=True,
        multigpu=False,
        placement=True,
        prioritize=False,
        fair=(mode == "fair"),
        tune=(mode == "tune"),
        simulate=True,
        config_file=str(SIMULATOR / "configs/default_cluster.ini"),
        num_jobs_default=num_jobs,
    )
    _, jcts = runner.run_simulation()
    return {
        "status": "ok",
        "seed": 42,
        "policy": "FIFO+GPU-proportional" if mode == "fair" else "FIFO+Synergy-TUNE",
        "num_jobs": num_jobs,
        "finished_jobs": len(runner.finished_jobs),
        "pending_jobs": len(runner.runnable_jobs),
        "simulated_finish_time_seconds": runner.time,
        "cluster": {
            "servers": runner.max_servers,
            "gpus": runner.max_gpus,
            "cpus": runner.max_cpus,
            "memory_gb": runner.max_mem,
        },
        "jct_seconds": [
            {"job_id": int(job_id), "jct": float(jct)}
            for job_id, jct in jcts.data_series
        ],
        "models_finished": [job.job_model.model_name for job in runner.finished_jobs],
        "profiled_demands": [
            {
                "job_id": job.job_id,
                "model": job.job_model.model_name,
                "gpu_demand": job.job_gpu_demand,
                "profile_demand": list(job.get_job_demand_vector),
                "gpu_proportional_demand": [
                    value * job.job_gpu_demand
                    for value in runner.cluster.per_server_size_fair
                ],
                "profile_speedup_over_proportional": job.synergy_speedup_orig,
            }
            for job in sorted(runner.finished_jobs, key=lambda item: item.job_id)
        ],
        "compatibility": [
            "collections.Mapping alias for Python 3.10",
            "import-only NUMA shim for Windows",
            "import-only deployment RPC shims; simulation path only",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument("--mode", choices=("fair", "tune"), default="tune")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(module)s - %(funcName)s - %(levelname)s - %(message)s",
    )
    print(json.dumps(run_smoke(args.jobs, args.mode), indent=2))


if __name__ == "__main__":
    main()
