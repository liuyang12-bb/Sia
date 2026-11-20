"""Compare preregistered Sia and Sia+AFS-P single-trace statistics."""

import argparse
import json
import statistics


SCALARS = (
    "completed_jobs",
    "average_jct_seconds",
    "p95_jct_seconds_official_index",
    "p99_jct_seconds_official_index",
    "makespan_seconds",
    "average_restarts_per_job",
    "gpu_hours_mean_per_job",
    "queue_mean",
    "queue_max",
    "contention_mean",
    "contention_max",
)


def load(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def relative(value, baseline):
    if baseline == 0:
        return None if value == 0 else "undefined_baseline_zero"
    return (value / baseline - 1.0) * 100.0


def compare_trace(baseline, candidate):
    scalar = {
        key: {
            "sia": baseline[key],
            "sia_afs": candidate[key],
            "relative_percent": relative(candidate[key], baseline[key]),
        }
        for key in SCALARS
    }
    apps = {}
    for app in sorted(
        set(baseline["average_jct_seconds_per_app"])
        | set(candidate["average_jct_seconds_per_app"])
    ):
        before = baseline["average_jct_seconds_per_app"].get(app)
        after = candidate["average_jct_seconds_per_app"].get(app)
        apps[app] = {
            "sia": before,
            "sia_afs": after,
            "relative_percent": relative(after, before)
            if before is not None and after is not None
            else None,
        }

    task_deltas = {}
    for name in sorted(
        set(baseline["task_jct_seconds"]) & set(candidate["task_jct_seconds"])
    ):
        before = baseline["task_jct_seconds"][name]
        after = candidate["task_jct_seconds"][name]
        task_deltas[name] = {
            "sia": before,
            "sia_afs": after,
            "delta_seconds": after - before,
            "relative_percent": relative(after, before),
        }
    deltas = [item["delta_seconds"] for item in task_deltas.values()]
    task_summary = {
        "compared_jobs": len(deltas),
        "improved_jobs": sum(delta < 0 for delta in deltas),
        "unchanged_jobs": sum(delta == 0 for delta in deltas),
        "worsened_jobs": sum(delta > 0 for delta in deltas),
        "mean_delta_seconds": statistics.mean(deltas) if deltas else None,
        "median_delta_seconds": statistics.median(deltas) if deltas else None,
    }
    return {
        "metrics": scalar,
        "application_jct": apps,
        "gpu_hours_by_type": {
            "sia": baseline["gpu_hours_by_type"],
            "sia_afs": candidate["gpu_hours_by_type"],
        },
        "gpu_type_share_fraction": {
            "sia": baseline["gpu_type_share_fraction"],
            "sia_afs": candidate["gpu_type_share_fraction"],
        },
        "capacity": {
            "sia_violations": baseline["capacity_violation_count"],
            "sia_afs_violations": candidate["capacity_violation_count"],
            "sia_maximum": baseline["maximum_allocated_gpus"],
            "sia_afs_maximum": candidate["maximum_allocated_gpus"],
        },
        "scheduling_time": {
            "sia": baseline["scheduling_time"],
            "sia_afs": candidate["scheduling_time"],
        },
        "afs_state": candidate["afs_state"],
        "task_summary": task_summary,
        "task_deltas": task_deltas,
    }


def main():
    parser = argparse.ArgumentParser()
    for trace in (1, 5, 10):
        parser.add_argument(f"--sia-{trace}", required=True)
        parser.add_argument(f"--afs-{trace}", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    traces = {}
    for trace in (1, 5, 10):
        traces[f"workload-{trace}"] = compare_trace(
            load(getattr(args, f"sia_{trace}")),
            load(getattr(args, f"afs_{trace}")),
        )
    result = {"traces": traces}
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
