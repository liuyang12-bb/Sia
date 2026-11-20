"""Recompute Sia single-trace metrics using print_run_stats.py semantics."""

import argparse
import collections
import json
import re
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--stdout-log")
    parser.add_argument("--output")
    args = parser.parse_args()

    records = []
    with open(args.log, encoding="utf-8") as stream:
        for line in stream:
            records.append(json.loads(line))
    if not records:
        raise RuntimeError("empty simulator log")

    final_jobs = records[-1]["submitted_jobs"]
    jcts = {
        job["name"]: job["completion_time"] - job["submission_time"]
        for job in final_jobs
        if job["completion_time"] is not None
    }
    ordered_jcts = sorted(jcts.values())
    starts = [job["submission_time"] for job in final_jobs]
    completions = [job["completion_time"] for job in final_jobs]

    restart_by_type = collections.defaultdict(list)
    jct_by_type = collections.defaultdict(list)
    for job in final_jobs:
        app = job["name"].split("-")[0]
        restart_by_type[app].append(job["num_restarts"])
        if job["completion_time"] is not None:
            jct_by_type[app].append(
                job["completion_time"] - job["submission_time"]
            )

    contention = []
    queues = []
    gpu_hours = collections.defaultdict(lambda: collections.defaultdict(float))
    job_types = {}
    for record in records:
        running = queued = 0
        for job in record["submitted_jobs"]:
            if job["completion_time"] is not None:
                continue
            allocation = job["allocation"]
            if not allocation or allocation[0] is None:
                queued += 1
                continue
            running += 1
            gpu_type, gpu_list = allocation
            app = job["name"].split("-")[0]
            job_types[job["name"]] = app
            gpu_hours[app][gpu_type] += len(gpu_list) * args.interval / 3600.0
        contention.append(running + queued)
        queues.append(queued)

    job_counts = collections.Counter(job["name"].split("-")[0] for job in final_jobs)
    per_job_gpu_hours = {
        app: {gpu: hours / job_counts[app] for gpu, hours in values.items()}
        for app, values in gpu_hours.items()
    }
    total_gpu_hours = sum(sum(values.values()) for values in gpu_hours.values())

    cluster_gpu_capacity = {"aws": 24, "dgx-ext": 16, "rtx": 24}
    maximum_allocated_gpus = collections.defaultdict(int)
    capacity_violations = []
    for record in records:
        allocated = collections.defaultdict(int)
        for job in record["submitted_jobs"]:
            allocation = job["allocation"]
            if allocation and allocation[0] is not None:
                allocated[allocation[0]] += len(allocation[1])
        for cluster, count in allocated.items():
            maximum_allocated_gpus[cluster] = max(
                maximum_allocated_gpus[cluster], count
            )
            if count > cluster_gpu_capacity[cluster]:
                capacity_violations.append(
                    {
                        "timestamp": record["timestamp"],
                        "cluster": cluster,
                        "allocated": count,
                        "capacity": cluster_gpu_capacity[cluster],
                    }
                )

    scheduling = None
    if args.stdout_log:
        with open(args.stdout_log, "rb") as stream:
            stdout_bytes = stream.read()
        if stdout_bytes.startswith((b"\xff\xfe", b"\xfe\xff")):
            stdout = stdout_bytes.decode("utf-16", errors="replace")
        else:
            stdout = stdout_bytes.decode("utf-8", errors="replace")
        solver_ms = [
            float(value)
            for value in re.findall(r"Solver took: ([0-9.]+)ms", stdout)
        ]
        policy_ms = [
            float(value)
            for value in re.findall(
                r"Policy optimization took ([0-9.]+)ms", stdout
            )
        ]
        scheduling = {
            "solver_rounds_measured": len(solver_ms),
            "solver_time_total_seconds": sum(solver_ms) / 1000.0,
            "solver_time_mean_ms": statistics.mean(solver_ms) if solver_ms else None,
            "solver_time_p99_ms": sorted(solver_ms)[int(0.99 * len(solver_ms))]
            if solver_ms
            else None,
            "policy_rounds_measured": len(policy_ms),
            "policy_time_total_seconds": sum(policy_ms) / 1000.0,
            "policy_time_mean_ms": statistics.mean(policy_ms) if policy_ms else None,
        }

    result = {
        "records": len(records),
        "jobs": len(final_jobs),
        "completed_jobs": len(jcts),
        "makespan_seconds": max(completions) - min(starts),
        "makespan_hours": (max(completions) - min(starts)) / 3600.0,
        "average_jct_seconds": statistics.mean(ordered_jcts),
        "average_jct_hours": statistics.mean(ordered_jcts) / 3600.0,
        "p99_jct_seconds_official_index": ordered_jcts[int(0.99 * len(ordered_jcts))],
        "p99_jct_hours_official_index": ordered_jcts[int(0.99 * len(ordered_jcts))] / 3600.0,
        "average_restarts_per_job": statistics.mean(job["num_restarts"] for job in final_jobs),
        "average_restarts_per_type": {
            app: statistics.mean(values) for app, values in sorted(restart_by_type.items())
        },
        "average_jct_seconds_per_type": {
            app: statistics.mean(values) for app, values in sorted(jct_by_type.items())
        },
        "contention_mean": statistics.mean(contention),
        "contention_max": max(contention),
        "queue_mean": statistics.mean(queues),
        "queue_median": statistics.median(queues),
        "queue_max": max(queues),
        "gpu_hours_mean_per_job": total_gpu_hours / len(final_jobs),
        "gpu_hours_per_job_type_and_gpu": per_job_gpu_hours,
        "maximum_allocated_gpus": dict(sorted(maximum_allocated_gpus.items())),
        "capacity_violation_count": len(capacity_violations),
        "capacity_violations": capacity_violations,
        "scheduling_time": scheduling,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
