"""Three-round, one-job deterministic original-Sia probe; not a cohort run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def probe(simdir):
    sys.path.insert(0, str(simdir))
    os.chdir(simdir)
    import pandas as pd
    import simulator_config as cfg
    from cluster import Cluster
    from sia import SiaPolicy
    cfg.num_parallel_jobs = 1
    cfg.sia_solver_verbose = False
    nodes = {"aws": 1, "dgx-ext": 1, "rtx": 1}
    gpus = {"aws": 4, "dgx-ext": 8, "rtx": 8}
    policy = SiaPolicy(p_fairness=-.5, restart_penalty=30, lambda_a=.01, lambda_n=1.1,
                       project_throughputs=False, share_max_replicas=False, sched_interval=60, enable_bsz_tuning=True)
    policy.populate_valid_configs(nodes, gpus, nodes)
    c = Cluster(pd.DataFrame([dict(name="cifar10-regression", time=17, application="cifar10", num_replicas=1, batch_size=128)]), policy, nodes, gpus, nodes)
    for job in c.jobs:
        job.seed_profiles(1, 8)
    states = []
    for _ in range(3):
        c.step(60)
        j = c.jobs[0]
        states.append({"time": c.current_time, "allocations": c.allocations, "progress": j.progress, "epoch": j.epoch,
                       "execution_time": j.execution_time, "rescale_time": j.rescale_time, "completion": j.completion_time,
                       "restarts": j.num_restarts, "used": j.used_gpu_seconds, "wasted": j.wasted_gpu_seconds,
                       "start_time": j.start_time})
    print("PROBE_JSON=" + json.dumps(states, sort_keys=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe", type=Path)
    p.add_argument("--root", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.probe:
        return probe(a.probe)
    if a.output.exists():
        raise RuntimeError("refuse to overwrite regression")
    a.output.mkdir(parents=True)
    results = {}
    for name in ("sia-joint-medium-frozen", "sia-joint-correctness-stage1"):
        cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "--probe", str(a.root / name / "sia-simulator")]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"})
        (a.output / (name + ".log")).write_bytes(r.stdout)
        if r.returncode:
            raise RuntimeError(f"probe failed: {name}, exit {r.returncode}")
        results[name] = json.loads(next(x[len("PROBE_JSON="):] for x in r.stdout.decode().splitlines() if x.startswith("PROBE_JSON=")))
    matched = results["sia-joint-medium-frozen"] == results["sia-joint-correctness-stage1"]
    (a.output / "comparison.json").write_text(json.dumps({"exact_state_match": matched, "rounds": 3, "jobs": 1, "states": results}, indent=2), encoding="utf-8")
    assert matched
    print("original Sia: 3/3 round state snapshots exactly match")


if __name__ == "__main__":
    main()
