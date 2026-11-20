"""Fixed twenty-job comparison with streamed execution and allocation audits."""

import argparse
from collections import Counter
from contextlib import redirect_stdout
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

from core80 import ROOT, build
from run_core80 import compact_decision, encode
from run_stage2c_cell import Accumulator, independent_rho
from joint_prototype.execution_audit import audit_execution
from verify_search import sample_stats


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, default=encode, indent=2, allow_nan=False)


def run(method, response, output):
    output.mkdir(parents=True, exist_ok=False)
    workload = ROOT / "inputs/workload.csv"
    config = ROOT / "inputs/config_symmetric.json"
    inputs = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (workload, config, ROOT / "inputs/registry.json")}
    source = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in ROOT.rglob("*.py")}
    capacity = {typ: dict(cpu_cores=12.0, ram_gb=250.0) for typ in ("aws", "dgx-ext")}
    save(output / "manifest.json", dict(
        method=method, response=response, inputs=inputs, source=source, capacities=capacity,
        topology={"aws": 2, "dgx-ext": 2}, period=60, max_rounds=6000, seed_profiles=True,
        evidence="synthetic cross-hardware CPU/RAM transfer", decision_delay_in_jct=False))
    timings, statuses = [], Counter()
    local_status, recovery = Counter(), Counter()
    lock_checks = 0
    rho_values = []
    previous_round = None
    with (output / "console.log").open("x", encoding="utf-8") as console, redirect_stdout(console):
        cluster = build(method, response=response, workload_path=workload, config_path=config,
                        topology={"aws": 2, "dgx-ext": 2})
        # Match the frozen CPU-complementary scenario's explicit node budgets.
        cluster.policy.node_capacities = capacity
        accumulator = Accumulator(cluster)
        started = time.perf_counter()
        with gzip.open(output / "rounds.jsonl.gz", "xt", encoding="utf-8", compresslevel=1) as raw:
            for number in range(1, 6001):
                if cluster.all_complete():
                    break
                cluster.step(60)
                record = cluster.joint_round_logs[-1]
                decision = record["decision_for_next_interval"]
                fresh = decision.get("round") != previous_round
                result = decision["result"]
                if fresh:
                    previous_round = decision.get("round")
                    assert result["audit"]["valid"]
                    statuses[result["status"]] += 1
                    timings.append(decision.get("engineering_decision_seconds", result["solve_seconds"]))
                    accumulator.solve.append(result["solve_seconds"])
                    specs = {job["job_id"]: job for job in decision["probe"]["specs"]}
                    totals = {typ + ":0": [0.0, 0.0, 0.0] for typ in capacity}
                    for name, column in result["selected"].items():
                        cfg = column["config"]
                        expected = independent_rho(cluster.policy.oracle, name, cfg["config_id"],
                                                   column["cpu_per_gpu"], column["ram_gb_per_gpu"])
                        assert math.isclose(expected, column["rho"], rel_tol=1e-12)
                        job = specs[name]
                        if job["checkpoint_locked"] and job["current_config_id"]:
                            assert cfg["config_id"] == job["current_config_id"]
                            assert column["placement"] == dict(job["current_placement"])
                            assert (column["cpu_per_gpu"], column["ram_gb_per_gpu"]) == (
                                job["current_cpu_per_gpu"], job["current_ram_gb_per_gpu"])
                            lock_checks += 1
                        for node, gpus in column["placement"].items():
                            for i, value in enumerate((gpus, gpus * column["cpu_per_gpu"],
                                                       gpus * column["ram_gb_per_gpu"])):
                                totals[node][i] += value
                    for node, used in totals.items():
                        assert all(value <= maximum + 1e-7 for value, maximum in zip(used, (2, 12, 250)))
                        for key, value in zip(("gpu", "cpu", "ram"), used):
                            accumulator.node_peaks[node][key] = max(accumulator.node_peaks[node][key], value)
                    search = result["audit"].get("search", {})
                    local_status[search.get("local", {}).get("local_status", "none")] += 1
                    recovery["offers"] += result["audit"].get("recovery_offer_count", 0)
                    recovery["commits"] += len(result["audit"].get("tail_recoveries", []))
                    recovery["eager"] += search.get("generation") == "RECOVERY_EAGER"
                accumulator.rounds += 1
                for event in record["execution"]:
                    assert audit_execution(event)["valid"]
                    accumulator.audit_count += 1
                    row = accumulator.rows[event["job_id"]]
                    for key, field in (("wait", "waiting_seconds"), ("training", "training_seconds"),
                                       ("checkpoint", "checkpoint_delay_consumed")):
                        row[key] += event[field]
                    allocation = event["resource_allocation"]
                    if allocation:
                        expected = independent_rho(cluster.policy.oracle, event["job_id"], allocation["config_id"],
                                                   allocation["cpu_per_gpu"], allocation["ram_gb_per_gpu"])
                        assert math.isclose(event["rho"], expected, rel_tol=1e-12)
                        rho_values.append(expected)
                        name = event["job_id"]
                        if allocation["config_id"] != accumulator.previous.get(name, allocation["config_id"]):
                            row["gpu_changes"] += 1
                        accumulator.previous[name] = allocation["config_id"]
                        for segment in event["training_segments"]:
                            assert math.isclose(segment["effective_rate"],
                                                segment["raw_rate_before_rho"] * expected, rel_tol=1e-12)
                        hours = event["allocated_seconds"] / 3600
                        row["gpu_hours"] += allocation["gpus"] * hours
                        for usage in allocation["node_usage"]:
                            row["cpu_core_hours"] += usage["cpu"] * hours
                            row["ram_gb_hours"] += usage["ram_gb"] * hours
                            node_hours = accumulator.node_hours[usage["node_id"]]
                            for key, field in (("gpu", "gpus"), ("cpu", "cpu"), ("ram", "ram_gb")):
                                node_hours[key] += usage[field] * hours
                compact_decision(decision)
                raw.write(json.dumps(record, default=encode, separators=(",", ":")) + "\n")
                if number % 25 == 0:
                    raw.flush()
                    print(f"{method}/{response} round {number}, completed "
                          f"{sum(job.completion_time is not None for job in cluster.jobs)}/20",
                          file=sys.stderr, flush=True)
                cluster.joint_round_logs.clear()
                cluster.logs.clear()
                cluster.rescale_events.clear()
                cluster.migrate_events.clear()
                for name in ("exchange_columns", "exchange_scores", "continuity_scores", "completion_scores",
                             "actual_completion_columns"):
                    if hasattr(cluster.policy, name):
                        setattr(cluster.policy, name, [])
        accumulator.rhos = rho_values
        accumulator.status = statuses
        summary = accumulator.summary(cluster, {"rmst_tau_seconds": 360000})
        summary.update(method=method, response=response, actual_node_capacities=capacity,
                       local_status=dict(local_status), recovery=dict(recovery),
                       checkpoint_lock_checks=lock_checks, decision_seconds=sample_stats(timings),
                       total_wall_seconds=time.perf_counter() - started)
        save(output / "summary.json", summary)
    print(f"Completed {method}/{response}, {summary['completed']}/20, mean JCT {summary['mean_jct']}",
          flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        for method, response in (("SIA_P", "neutral"), ("SIA_P", "medium"),
                                 ("LAZY_FO", "medium"), ("LOCAL", "medium")):
            run(method, response, args.output / f"{method}_{response}")
    except Exception:
        save(args.output / "failure.json", {"traceback": traceback.format_exc()})
        raise
