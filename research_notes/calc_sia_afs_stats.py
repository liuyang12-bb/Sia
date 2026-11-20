"""Compute comparable Sia/AFS metrics for one simulator JSONL trace."""

import argparse
import collections
import json
import re
import statistics


CAPACITY = {"aws": 24, "dgx-ext": 16, "rtx": 24}


def read_stdout(path):
    if not path:
        return ""
    with open(path, "rb") as stream:
        data = stream.read()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    return data.decode("utf-8", errors="replace")


def indexed_percentile(values, percentile):
    ordered = sorted(values)
    return ordered[min(int(percentile * len(ordered)), len(ordered) - 1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--stdout-log")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.log, encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if not records:
        raise RuntimeError("empty simulator log")

    final_jobs = records[-1]["submitted_jobs"]
    task_jcts = {
        job["name"]: job["completion_time"] - job["submission_time"]
        for job in final_jobs
        if job["completion_time"] is not None
    }
    jcts = list(task_jcts.values())
    starts = [job["submission_time"] for job in final_jobs]
    completions = [
        job["completion_time"]
        for job in final_jobs
        if job["completion_time"] is not None
    ]

    jct_by_app = collections.defaultdict(list)
    restart_by_app = collections.defaultdict(list)
    for job in final_jobs:
        app = job["name"].split("-")[0]
        restart_by_app[app].append(job["num_restarts"])
        if job["completion_time"] is not None:
            jct_by_app[app].append(task_jcts[job["name"]])

    queues = []
    contention = []
    gpu_hours_by_type = collections.defaultdict(float)
    maximum_allocated = collections.defaultdict(int)
    capacity_violations = []
    for record in records:
        running = queued = 0
        allocated = collections.defaultdict(int)
        for job in record["submitted_jobs"]:
            if job["completion_time"] is not None:
                continue
            allocation = job["allocation"]
            if not allocation or allocation[0] is None:
                queued += 1
                continue
            running += 1
            cluster, gpu_list = allocation
            count = len(gpu_list)
            allocated[cluster] += count
            gpu_hours_by_type[cluster] += count * args.interval / 3600.0
        queues.append(queued)
        contention.append(running + queued)
        for cluster, count in allocated.items():
            maximum_allocated[cluster] = max(maximum_allocated[cluster], count)
            capacity = CAPACITY.get(cluster)
            if capacity is not None and count > capacity:
                capacity_violations.append(
                    {
                        "timestamp": record["timestamp"],
                        "cluster": cluster,
                        "allocated": count,
                        "capacity": capacity,
                    }
                )

    stdout = read_stdout(args.stdout_log)
    solver_ms = [float(x) for x in re.findall(r"Solver took: ([0-9.]+)ms", stdout)]
    policy_ms = [
        float(x)
        for x in re.findall(r"Policy optimization took ([0-9.]+)ms", stdout)
    ]
    afs_states = []
    for line in stdout.splitlines():
        marker = "AFS_STATE "
        position = line.find(marker)
        if position < 0:
            continue
        try:
            afs_states.append(json.loads(line[position + len(marker) :]))
        except json.JSONDecodeError:
            continue
    mapping_ms = [
        state["mapping_ms"] for state in afs_states if "mapping_ms" in state
    ]

    total_gpu_hours = sum(gpu_hours_by_type.values())
    result = {
        "records": len(records),
        "jobs": len(final_jobs),
        "completed_jobs": len(task_jcts),
        "completion_fraction": len(task_jcts) / len(final_jobs),
        "average_jct_seconds": statistics.mean(jcts),
        "p95_jct_seconds_official_index": indexed_percentile(jcts, 0.95),
        "p99_jct_seconds_official_index": indexed_percentile(jcts, 0.99),
        "makespan_seconds": max(completions) - min(starts),
        "average_restarts_per_job": statistics.mean(
            job["num_restarts"] for job in final_jobs
        ),
        "gpu_hours_mean_per_job": total_gpu_hours / len(final_jobs),
        "gpu_hours_by_type": dict(sorted(gpu_hours_by_type.items())),
        "gpu_type_share_fraction": {
            cluster: hours / total_gpu_hours
            for cluster, hours in sorted(gpu_hours_by_type.items())
        },
        "queue_mean": statistics.mean(queues),
        "queue_median": statistics.median(queues),
        "queue_max": max(queues),
        "contention_mean": statistics.mean(contention),
        "contention_max": max(contention),
        "average_jct_seconds_per_app": {
            app: statistics.mean(values) for app, values in sorted(jct_by_app.items())
        },
        "average_restarts_per_app": {
            app: statistics.mean(values)
            for app, values in sorted(restart_by_app.items())
        },
        "task_jct_seconds": dict(sorted(task_jcts.items())),
        "maximum_allocated_gpus": dict(sorted(maximum_allocated.items())),
        "capacity_violation_count": len(capacity_violations),
        "capacity_violations": capacity_violations,
        "scheduling_time": {
            "solver_rounds_measured": len(solver_ms),
            "solver_time_total_seconds": sum(solver_ms) / 1000.0,
            "solver_time_mean_ms": statistics.mean(solver_ms) if solver_ms else None,
            "solver_time_p99_ms": indexed_percentile(solver_ms, 0.99)
            if solver_ms
            else None,
            "policy_rounds_measured": len(policy_ms),
            "policy_time_total_seconds": sum(policy_ms) / 1000.0,
            "policy_time_mean_ms": statistics.mean(policy_ms) if policy_ms else None,
            "policy_time_p99_ms": indexed_percentile(policy_ms, 0.99)
            if policy_ms
            else None,
            "afs_mapping_rounds": len(mapping_ms),
            "afs_mapping_time_total_seconds": sum(mapping_ms) / 1000.0,
            "afs_mapping_time_mean_ms": statistics.mean(mapping_ms)
            if mapping_ms
            else None,
            "afs_mapping_time_p99_ms": indexed_percentile(mapping_ms, 0.99)
            if mapping_ms
            else None,
        },
        "afs_state": {
            "records": len(afs_states),
            "stage_counts": dict(
                collections.Counter(state.get("stage") for state in afs_states)
            ),
            "reason_counts": dict(
                collections.Counter(state.get("reason") for state in afs_states)
            ),
            "events_with_unmapped_jobs": sum(
                bool(state.get("unmapped_jobs")) for state in afs_states
            ),
        },
    }
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
