"""Audit preemption/reconfiguration events in saved Sia JSONL logs.

The saved logs expose 60-second snapshots, allocation choices, progress and the
cumulative restart count.  They do *not* expose separate pause/load phases,
remaining duration, decision cause, or the in-flight rescale countdown.  This
script therefore reports exact snapshot-observable events and a clearly marked
reconstruction under the simulator's fixed per-application rescale durations.
It never labels a reconstructed event as a paper-defined Lazer futile
preemption.
"""

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict


RESCALE_SECONDS = {
    "cifar10": 50.0,
    "deepspeech2": 25.0,
    "bert": 120.0,
    "yolov3": 80.0,
    "imagenet": 250.0,
    "ncf": 15.0,
}


def application(name):
    return name.rsplit("-", 1)[0]


def percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(int(quantile * len(ordered)), len(ordered) - 1)]


def describe(values):
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "min": min(values),
        "max": max(values),
    }


def allocation_of(snapshot):
    allocation = snapshot.get("allocation")
    if not allocation or len(allocation) != 2:
        return None
    cluster, node_ids = allocation
    if cluster is None or not node_ids:
        return None
    return {
        "cluster": cluster,
        "node_ids": tuple(node_ids),
        "gpu_count": len(node_ids),
        "placement": tuple(snapshot.get("placement") or ()),
    }


def allocation_key(allocation):
    if allocation is None:
        return None
    return allocation["cluster"], allocation["node_ids"]


def public_allocation(allocation):
    if allocation is None:
        return None
    return {
        "cluster": allocation["cluster"],
        "node_ids": list(allocation["node_ids"]),
        "gpu_count": allocation["gpu_count"],
        "placement": list(allocation["placement"]),
    }


def load_records(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def finalize_episode(episode, timestamp, snapshot, reason, submissions):
    completion_time = snapshot.get("completion_time")
    end_time = min(timestamp, completion_time) if completion_time is not None else timestamp
    elapsed = max(float(end_time) - episode["timestamp"], 0.0)
    progress_delta = float(snapshot.get("progress", 0.0)) - episode["progress"]
    rescale_seconds = episode["rescale_seconds"]
    gpu_count = episode["new_allocation"]["gpu_count"]
    load_end = episode["timestamp"] + rescale_seconds
    arrivals_during_rescale = sorted(
        name
        for name, submitted in submissions.items()
        if episode["timestamp"] < submitted <= load_end and name != episode["name"]
    )
    ended_by_config_change = reason in {"restart", "suspend"}
    episode.update(
        {
            "end_timestamp": float(end_time),
            "end_reason": reason,
            "seconds_to_next_change_or_completion": elapsed,
            "progress_delta": progress_delta,
            "zero_progress": abs(progress_delta) <= 1e-12,
            "reconstructed_effective_training_seconds": max(
                elapsed - rescale_seconds, 0.0
            ),
            "reconstructed_wasted_gpu_seconds": min(
                elapsed, rescale_seconds
            )
            * gpu_count,
            "changed_before_rescale_complete": (
                ended_by_config_change and elapsed < rescale_seconds
            ),
            "changed_at_rescale_boundary_without_progress": (
                ended_by_config_change
                and math.isclose(elapsed, rescale_seconds)
                and abs(progress_delta) <= 1e-12
            ),
            "changed_without_observed_progress": (
                ended_by_config_change and abs(progress_delta) <= 1e-12
            ),
            "arrivals_during_reconstructed_rescale": arrivals_during_rescale,
        }
    )


def audit(path, interval):
    records = load_records(path)
    if not records:
        raise ValueError("empty log")

    submissions = {}
    for record in records:
        for snapshot in record["submitted_jobs"]:
            submissions[snapshot["name"]] = float(snapshot["submission_time"])

    state = {}
    per_job = defaultdict(
        lambda: {
            "final_num_restarts": None,
            "first_observed_timestamp": None,
            "first_allocation_timestamp": None,
            "completion_time": None,
            "completion_observed_timestamp": None,
            "allocation_change_events": [],
            "load_episodes": [],
        }
    )

    for record in records:
        timestamp = float(record["timestamp"])
        for snapshot in record["submitted_jobs"]:
            name = snapshot["name"]
            app = application(name)
            job_result = per_job[name]
            if job_result["first_observed_timestamp"] is None:
                job_result["first_observed_timestamp"] = timestamp

            new_allocation = allocation_of(snapshot)
            new_key = allocation_key(new_allocation)
            previous = state.get(name)
            old_key = previous["allocation_key"] if previous else None
            previous_restarts = previous["num_restarts"] if previous else None
            current_restarts = snapshot.get("num_restarts")
            restart_increment = (
                current_restarts is not None
                and previous_restarts is not None
                and current_restarts > previous_restarts
            )

            if previous and previous.get("active_episode") is not None:
                if snapshot.get("completion_time") is not None:
                    finalize_episode(
                        previous["active_episode"],
                        timestamp,
                        snapshot,
                        "completion",
                        submissions,
                    )
                    previous["active_episode"] = None
                elif restart_increment:
                    finalize_episode(
                        previous["active_episode"],
                        timestamp,
                        snapshot,
                        "restart",
                        submissions,
                    )
                    previous["active_episode"] = None
                elif new_allocation is None and previous["allocation"] is not None:
                    finalize_episode(
                        previous["active_episode"],
                        timestamp,
                        snapshot,
                        "suspend",
                        submissions,
                    )
                    previous["active_episode"] = None

            allocation_changed = previous is None or new_key != old_key
            episode = previous.get("active_episode") if previous else None

            if allocation_changed:
                old_allocation = previous["allocation"] if previous else None
                last_allocated = previous["last_allocated"] if previous else None
                if old_allocation is None and new_allocation is not None:
                    kind = "initial_start" if last_allocated is None else "resume"
                elif old_allocation is not None and new_allocation is None:
                    kind = "suspend"
                elif old_allocation is not None and new_allocation is not None:
                    kind = "direct_reconfiguration"
                else:
                    kind = "first_observed_unallocated"

                event = {
                    "timestamp": timestamp,
                    "kind": kind,
                    "old_allocation": public_allocation(old_allocation),
                    "new_allocation": public_allocation(new_allocation),
                    "last_allocated_before_event": public_allocation(last_allocated),
                    "num_restarts": current_restarts,
                    "restart_increment_observed": restart_increment,
                }
                job_result["allocation_change_events"].append(event)

                if new_allocation is not None:
                    is_initial_start = job_result["first_allocation_timestamp"] is None
                    if is_initial_start:
                        job_result["first_allocation_timestamp"] = timestamp
                    if is_initial_start or restart_increment:
                        comparison = last_allocated
                        rescale_seconds = RESCALE_SECONDS.get(app, 30.0)
                        episode = {
                            "name": name,
                            "application": app,
                            "timestamp": timestamp,
                            "kind": kind,
                            "is_initial_start": is_initial_start,
                            "is_restart": restart_increment,
                            "num_restarts": current_restarts,
                            "previous_allocated_config": public_allocation(comparison),
                            "new_allocation": public_allocation(new_allocation),
                            "gpu_type_changed": (
                                comparison is not None
                                and comparison["cluster"] != new_allocation["cluster"]
                            ),
                            "gpu_count_changed": (
                                comparison is not None
                                and comparison["gpu_count"] != new_allocation["gpu_count"]
                            ),
                            "node_ids_changed": (
                                comparison is not None
                                and comparison["node_ids"] != new_allocation["node_ids"]
                            ),
                            "progress": float(snapshot.get("progress", 0.0)),
                            "rescale_seconds": rescale_seconds,
                        }
                        job_result["load_episodes"].append(episode)
                else:
                    episode = None

            if snapshot.get("completion_time") is not None:
                job_result["completion_time"] = float(snapshot["completion_time"])
                if job_result["completion_observed_timestamp"] is None:
                    job_result["completion_observed_timestamp"] = timestamp

            last_allocated = previous["last_allocated"] if previous else None
            if new_allocation is not None:
                last_allocated = new_allocation
            state[name] = {
                "allocation": new_allocation,
                "allocation_key": new_key,
                "last_allocated": last_allocated,
                "num_restarts": current_restarts,
                "active_episode": episode,
            }

    last_timestamp = float(records[-1]["timestamp"])
    last_snapshot = {
        snapshot["name"]: snapshot
        for snapshot in records[-1]["submitted_jobs"]
    }
    for name, current in state.items():
        if current.get("active_episode") is not None:
            snapshot = last_snapshot[name]
            finalize_episode(
                current["active_episode"],
                last_timestamp,
                snapshot,
                "log_end",
                submissions,
            )
        per_job[name]["final_num_restarts"] = current["num_restarts"] or 0

    jobs = dict(sorted(per_job.items()))
    episodes = [episode for item in jobs.values() for episode in item["load_episodes"]]
    restarts = [episode for episode in episodes if episode["is_restart"]]
    first_starts = [episode for episode in episodes if episode["is_initial_start"]]
    zero_progress_restarts = [
        episode
        for episode in restarts
        if episode.get("changed_without_observed_progress")
    ]
    strict_futile_candidates = [
        episode
        for episode in zero_progress_restarts
        if episode.get("arrivals_during_reconstructed_rescale")
    ]

    transition_categories = Counter()
    for episode in restarts:
        type_change = episode["gpu_type_changed"]
        count_change = episode["gpu_count_changed"]
        if type_change and count_change:
            transition_categories["gpu_type_and_count_changed"] += 1
        elif type_change:
            transition_categories["gpu_type_only_changed"] += 1
        elif count_change:
            transition_categories["gpu_count_only_changed"] += 1
        elif episode["node_ids_changed"]:
            transition_categories["node_or_placement_only_changed"] += 1
        else:
            transition_categories["same_config_resume"] += 1

    by_application = {}
    for app in sorted({episode["application"] for episode in episodes}):
        app_episodes = [episode for episode in episodes if episode["application"] == app]
        app_restarts = [episode for episode in app_episodes if episode["is_restart"]]
        by_application[app] = {
            "load_episodes": len(app_episodes),
            "restart_episodes": len(app_restarts),
            "reconstructed_wasted_gpu_seconds": sum(
                episode["reconstructed_wasted_gpu_seconds"]
                for episode in app_episodes
            ),
            "zero_progress_restart_then_change": sum(
                episode.get("changed_without_observed_progress", False)
                for episode in app_restarts
            ),
        }

    submit_quantization = []
    first_allocation_wait = []
    completion_release_delay = []
    for name, item in jobs.items():
        submit_quantization.append(
            item["first_observed_timestamp"] - submissions[name]
        )
        if item["first_allocation_timestamp"] is not None:
            first_allocation_wait.append(
                item["first_allocation_timestamp"] - submissions[name]
            )
        if item["completion_time"] is not None:
            completion_release_delay.append(
                item["completion_observed_timestamp"] - item["completion_time"]
            )

    total_restart_count = sum(item["final_num_restarts"] for item in jobs.values())
    reconstructed_waste = sum(
        episode["reconstructed_wasted_gpu_seconds"] for episode in episodes
    )
    restart_waste = sum(
        episode["reconstructed_wasted_gpu_seconds"] for episode in restarts
    )
    zero_progress_upper_bound_waste = sum(
        episode["reconstructed_wasted_gpu_seconds"]
        for episode in zero_progress_restarts
    )

    return {
        "source_log": path,
        "interval_seconds": interval,
        "records": len(records),
        "jobs": len(jobs),
        "strict_lazer_futile_preemption_count": None,
        "strict_count_status": "UNKNOWN_FROM_SAVED_JSONL",
        "strict_count_reason": (
            "The JSONL has no separate pause/load phase timestamps, remaining "
            "duration or scheduler-decision cause. Zero-progress reconfiguration "
            "is only an observable candidate, not Lazer Eq. (2)."
        ),
        "summary": {
            "final_restart_count": total_restart_count,
            "observed_restart_episode_count": len(restarts),
            "initial_load_episode_count": len(first_starts),
            "all_load_episode_count": len(episodes),
            "changed_before_reconstructed_rescale_complete": sum(
                episode.get("changed_before_rescale_complete", False)
                for episode in episodes
            ),
            "restart_then_change_without_observed_progress": len(
                zero_progress_restarts
            ),
            "zero_progress_restart_with_arrival_during_reconstructed_rescale": len(
                strict_futile_candidates
            ),
            "changed_at_rescale_boundary_without_progress": sum(
                episode.get("changed_at_rescale_boundary_without_progress", False)
                for episode in restarts
            ),
            "reconstructed_all_checkpoint_restore_gpu_seconds": reconstructed_waste,
            "reconstructed_restart_checkpoint_restore_gpu_seconds": restart_waste,
            "zero_progress_restart_gpu_seconds_upper_bound": zero_progress_upper_bound_waste,
            "restart_transition_categories": dict(transition_categories),
            "gpu_type_changed_restart_fraction": (
                sum(episode["gpu_type_changed"] for episode in restarts) / len(restarts)
                if restarts
                else None
            ),
            "gpu_count_changed_restart_fraction": (
                sum(episode["gpu_count_changed"] for episode in restarts) / len(restarts)
                if restarts
                else None
            ),
            "restart_to_next_change_or_completion_seconds": describe(
                [episode["seconds_to_next_change_or_completion"] for episode in restarts]
            ),
            "reconstructed_effective_training_seconds_after_restart": describe(
                [episode["reconstructed_effective_training_seconds"] for episode in restarts]
            ),
            "submission_to_first_scheduler_boundary_seconds": describe(
                submit_quantization
            ),
            "submission_to_first_allocation_seconds": describe(first_allocation_wait),
            "completion_to_observed_scheduler_boundary_seconds": describe(
                completion_release_delay
            ),
        },
        "by_application": by_application,
        "per_job": jobs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.log, args.interval)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"summary": result["summary"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
