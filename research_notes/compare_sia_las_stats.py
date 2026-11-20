import argparse
import json


SCALAR_METRICS = (
    "completed_jobs",
    "average_jct_seconds",
    "p99_jct_seconds_official_index",
    "makespan_seconds",
    "average_restarts_per_job",
    "gpu_hours_mean_per_job",
    "contention_mean",
    "contention_max",
    "queue_mean",
    "queue_median",
    "queue_max",
)


def load(path):
    with open(path, encoding="utf-8-sig") as stream:
        return json.load(stream)


def relative_percent(value, baseline):
    if baseline == 0:
        return None if value == 0 else "undefined_baseline_zero"
    return (value / baseline - 1.0) * 100.0


def compare(candidate, baseline):
    result = {}
    for metric in SCALAR_METRICS:
        result[metric] = {
            "value": candidate[metric],
            "relative_to_sia_percent": relative_percent(
                candidate[metric], baseline[metric]
            ),
        }
    result["average_jct_seconds_per_type"] = {
        app: {
            "value": value,
            "relative_to_sia_percent": relative_percent(
                value, baseline["average_jct_seconds_per_type"][app]
            ),
        }
        for app, value in sorted(
            candidate["average_jct_seconds_per_type"].items()
        )
    }
    result["capacity_violation_count"] = candidate[
        "capacity_violation_count"
    ]
    result["maximum_allocated_gpus"] = candidate[
        "maximum_allocated_gpus"
    ]
    result["scheduling_time"] = candidate["scheduling_time"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--factor1_5", required=True)
    parser.add_argument("--factor2_0", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = load(args.baseline)
    output = {
        "baseline": baseline,
        "sia_las_factor1_5": compare(load(args.factor1_5), baseline),
        "sia_las_factor2_0": compare(load(args.factor2_0), baseline),
    }
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
