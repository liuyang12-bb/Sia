"""Analyze the completed w10_late Medium 2x2 mechanism ablation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from run_cpu_bottleneck_smoke import json_default, sha256


POLICIES = ("b1", "b4", "b3_prop", "b3")
TARGET = "deepspeech2-159"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value):
    path.write_text(json.dumps(value, indent=2, default=json_default), encoding="utf-8")


def allocation_signature(allocation):
    if not allocation:
        return None
    return (
        allocation["config_id"],
        allocation["gpu_type"],
        int(allocation["gpus"]),
        tuple(sorted((str(key), int(value)) for key, value in allocation["placement"].items())),
        round(float(allocation["cpu_per_gpu"]), 9),
        round(float(allocation["ram_gb_per_gpu"]), 9),
        round(float(allocation["rho"]), 12),
    )


def job_trace(payload, job_id):
    segments = []
    config_sequence = []
    total_training = 0.0
    raw_work = 0.0
    effective_work = 0.0
    checkpoint = 0.0
    allocated_intervals = 0
    for record in payload["joint_round_logs"]:
        event = next(item for item in record["execution"] if item["job_id"] == job_id)
        allocation = event.get("resource_allocation")
        signature = allocation_signature(allocation)
        checkpoint += float(event["checkpoint_delay_consumed"])
        training = float(event["training_seconds"])
        total_training += training
        raw_work += float(event["raw_goodput"]) * training
        effective_work += float(event["effective_goodput"]) * training
        if not allocation or not event["placement"]:
            continue
        allocated_intervals += 1
        config_sequence.append(allocation["config_id"])
        entry = {
            "interval_start": record["interval_start"],
            "interval_end": record["interval_end"],
            "config_id": allocation["config_id"],
            "gpu_type": allocation["gpu_type"],
            "gpus": allocation["gpus"],
            "placement": allocation["placement"],
            "cpu_per_gpu": allocation["cpu_per_gpu"],
            "ram_gb_per_gpu": allocation["ram_gb_per_gpu"],
            "rho": allocation["rho"],
            "training_seconds": training,
            "checkpoint_seconds": float(event["checkpoint_delay_consumed"]),
            "raw_goodput_training_weighted_sum": float(event["raw_goodput"]) * training,
            "effective_goodput_training_weighted_sum": float(event["effective_goodput"]) * training,
            "interval_count": 1,
        }
        if segments and segments[-1]["signature"] == signature:
            segment = segments[-1]
            segment["interval_end"] = entry["interval_end"]
            segment["training_seconds"] += entry["training_seconds"]
            segment["checkpoint_seconds"] += entry["checkpoint_seconds"]
            segment["raw_goodput_training_weighted_sum"] += entry["raw_goodput_training_weighted_sum"]
            segment["effective_goodput_training_weighted_sum"] += entry["effective_goodput_training_weighted_sum"]
            segment["interval_count"] += 1
        else:
            entry["signature"] = signature
            segments.append(entry)
    for segment in segments:
        training = segment["training_seconds"]
        segment["mean_raw_goodput_during_training"] = (
            segment.pop("raw_goodput_training_weighted_sum") / training if training else 0.0
        )
        segment["mean_effective_goodput_during_training"] = (
            segment.pop("effective_goodput_training_weighted_sum") / training if training else 0.0
        )
        segment.pop("signature")
    summary = payload["summary"]
    job = summary["jobs"][job_id]
    return {
        "completed": summary["completion_times"].get(job_id) is not None,
        "jct_seconds": summary["jcts"].get(job_id),
        "completion_time": summary["completion_times"][job_id],
        "waiting_seconds": summary["waiting_seconds"][job_id],
        "restarts": job["num_restarts"],
        "wasted_gpu_seconds": job["wasted_gpu_seconds"],
        "checkpoint_restore_seconds": checkpoint,
        "training_seconds": total_training,
        "training_weighted_mean_raw_goodput": raw_work / total_training if total_training else 0.0,
        "training_weighted_mean_effective_goodput": effective_work / total_training if total_training else 0.0,
        "gpu_config_changes": sum(left != right for left, right in zip(config_sequence, config_sequence[1:])),
        "allocated_intervals": allocated_intervals,
        "segments": segments,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    validation = load(root / "validation_summary.json")
    payloads = {
        policy: load(root / "policies" / policy / "result.json")
        for policy in POLICIES
    }
    jobs = list(payloads["b1"]["summary"]["completion_times"])
    if len(jobs) != 20 or any(
        set(payload["summary"]["completion_times"]) != set(jobs)
        for payload in payloads.values()
    ):
        raise ValueError("analysis requires the same 20 submitted jobs in every cell")

    metrics = {}
    for policy, payload in payloads.items():
        summary = payload["summary"]
        jcts = np.asarray(list(summary["jcts"].values()), dtype=float)
        completions = [
            float(value) for value in summary["completion_times"].values()
            if value is not None
        ]
        checkpoint = sum(
            float(event["checkpoint_delay_consumed"])
            for record in payload["joint_round_logs"]
            for event in record["execution"]
        )
        statuses = Counter(
            record["decision_for_next_interval"].get("result", {}).get("status", "MISSING")
            for record in payload["joint_round_logs"]
        )
        metrics[policy] = {
            "completed_jobs": summary["completed_jobs"],
            "average_jct_seconds": float(np.mean(jcts)) if len(jcts) else None,
            "median_jct_seconds": float(np.percentile(jcts, 50)) if len(jcts) else None,
            "p95_jct_seconds": float(np.percentile(jcts, 95)) if len(jcts) else None,
            "p99_jct_seconds": float(np.percentile(jcts, 99)) if len(jcts) else None,
            "makespan_seconds": max(completions) if completions else None,
            "resource_hours": summary["resource_hours"],
            "restart_count": sum(item["num_restarts"] for item in summary["jobs"].values()),
            "checkpoint_restore_seconds": checkpoint,
            "wasted_gpu_seconds": sum(item["wasted_gpu_seconds"] for item in summary["jobs"].values()),
            "total_waiting_seconds": sum(summary["waiting_seconds"].values()),
            "rounds": summary["rounds"],
            "rho_range": [summary["min_rho"], summary["max_rho"]],
            "solve_status_counts": dict(statuses),
            "optimizer_total_seconds": summary["optimizer_total_seconds"],
            "optimizer_p99_seconds": summary["optimizer_p99_seconds"],
        }

    jct = {policy: metrics[policy]["average_jct_seconds"] for policy in POLICIES}

    def effect(name, baseline, treatment):
        if jct[baseline] is None or jct[treatment] is None:
            return {
                "name": name,
                "baseline": baseline,
                "treatment": treatment,
                "status": "NOT_IDENTIFIABLE_CENSORED_CELL",
                "absolute_jct_reduction_seconds": None,
                "relative_jct_reduction_percent": None,
            }
        delta = jct[baseline] - jct[treatment]
        return {
            "name": name,
            "baseline": baseline,
            "treatment": treatment,
            "absolute_jct_reduction_seconds": delta,
            "relative_jct_reduction_percent": 100.0 * delta / jct[baseline],
        }

    effects = {
        "aux_optimization_under_sia_gpu": effect("B4 vs B1", "b1", "b4"),
        "joint_gpu_choice_under_proportional_aux": effect("B3-PROP vs B1", "b1", "b3_prop"),
        "joint_gpu_choice_under_optimized_aux": effect("B3 vs B4", "b4", "b3"),
        "aux_optimization_under_joint_gpu": effect("B3 vs B3-PROP", "b3_prop", "b3"),
    }
    interaction = (
        jct["b3_prop"] + jct["b4"] - jct["b3"] - jct["b1"]
        if jct["b3_prop"] is not None
        else None
    )
    effects["interaction"] = {
        "definition": "(B3_PROP-B3) - (B1-B4) = B3_PROP+B4-B3-B1",
        "positive_sign_meaning": "auxiliary optimization provides more JCT reduction when joint GPU choice is enabled",
        "status": "IDENTIFIED" if interaction is not None else "NOT_IDENTIFIABLE_CENSORED_B3_PROP",
        "interaction_gain_seconds": interaction,
        "interaction_percent_of_b1": (
            100.0 * interaction / jct["b1"] if interaction is not None else None
        ),
    }
    effects["factorial_average_main_effects"] = {
        "status": "NOT_IDENTIFIABLE_CENSORED_B3_PROP",
        "gpu_choice_jct_reduction_seconds": None,
        "auxiliary_choice_jct_reduction_seconds": None,
    }

    per_job = {}
    for job in jobs:
        values = {
            policy: (
                float(payloads[policy]["summary"]["jcts"][job])
                if job in payloads[policy]["summary"]["jcts"]
                else None
            )
            for policy in POLICIES
        }
        def delta(left, right):
            return values[left] - values[right] if values[left] is not None and values[right] is not None else None
        per_job[job] = {
            "completed": {
                policy: payloads[policy]["summary"]["completion_times"].get(job) is not None
                for policy in POLICIES
            },
            "jct_seconds": values,
            "completion_time_seconds": {
                policy: payloads[policy]["summary"]["completion_times"].get(job)
                for policy in POLICIES
            },
            "waiting_seconds": {
                policy: payloads[policy]["summary"]["waiting_seconds"].get(job)
                for policy in POLICIES
            },
            "effects_seconds": {
                "aux_under_sia": delta("b1", "b4"),
                "gpu_under_proportional": delta("b1", "b3_prop"),
                "gpu_under_optimized_aux": delta("b4", "b3"),
                "aux_under_joint": delta("b3_prop", "b3"),
                "interaction": (
                    values["b3_prop"] + values["b4"] - values["b3"] - values["b1"]
                    if values["b3_prop"] is not None else None
                ),
            },
        }

    without_target = {}
    remaining = [job for job in jobs if job != TARGET]
    for policy in POLICIES:
        available = [
            payloads[policy]["summary"]["jcts"][job]
            for job in remaining
            if job in payloads[policy]["summary"]["jcts"]
        ]
        without_target[policy] = float(np.mean(available)) if len(available) == len(remaining) else None
    without_target_effects = {}
    for key, baseline, treatment in (
        ("b4_vs_b1", "b1", "b4"),
        ("b3_prop_vs_b1", "b1", "b3_prop"),
        ("b3_vs_b4", "b4", "b3"),
        ("b3_vs_b3_prop", "b3_prop", "b3"),
    ):
        if without_target[baseline] is None or without_target[treatment] is None:
            without_target_effects[key] = {
                "status": "NOT_IDENTIFIABLE_CENSORED_CELL",
                "absolute_jct_reduction_seconds": None,
                "relative_jct_reduction_percent": None,
            }
        else:
            delta_value = without_target[baseline] - without_target[treatment]
            without_target_effects[key] = {
                "status": "IDENTIFIED",
                "absolute_jct_reduction_seconds": delta_value,
                "relative_jct_reduction_percent": 100.0 * delta_value / without_target[baseline],
            }

    trace = {policy: job_trace(payload, TARGET) for policy, payload in payloads.items()}
    prop_rounds = payloads["b3_prop"]["joint_round_logs"]
    prop_liveness = {
        "rounds": len(prop_rounds),
        "optimal_empty_selection_rounds": sum(
            record["decision_for_next_interval"].get("result", {}).get("status") == "OPTIMAL"
            and not record["decision_for_next_interval"].get("result", {}).get("selected")
            for record in prop_rounds
        ),
        "rounds_with_nonempty_candidate_set": sum(
            any(
                int(item.get("count", 0)) > 0
                for item in record["decision_for_next_interval"].get("candidate_summary", {}).values()
            )
            for record in prop_rounds
        ),
        "maximum_candidate_columns_in_a_round": max(
            (
                sum(
                    int(item.get("count", 0))
                    for item in record["decision_for_next_interval"].get("candidate_summary", {}).values()
                )
                for record in prop_rounds
            ),
            default=0,
        ),
        "scheduling_objective_values": sorted({
            float(record["decision_for_next_interval"].get("result", {}).get("scheduling_objective", 0.0))
            for record in prop_rounds
        }),
        "jobs_with_zero_progress": sorted(
            job_id
            for job_id, item in payloads["b3_prop"]["summary"]["jobs"].items()
            if float(item["progress"]) == 0.0
        ),
        "interpretation": (
            "Legal candidates existed, but the frozen transformed Sia objective preferred "
            "the permitted empty allocation in every round. This is not NO_COLUMNS or a solver failure."
        ),
    }
    analysis = {
        "status": "PASS_COMPLETE_MATRIX" if validation["status"] == "PASS" else "INCOMPLETE_CENSORED_MATRIX",
        "scope": "w10_late Medium 2x2 B3 mechanism ablation",
        "warning": "Synthetic cross-hardware transfer; not real Saturn performance.",
        "metrics": metrics,
        "effects": effects,
        "per_job": per_job,
        "b3_prop_liveness": prop_liveness,
        "deepspeech2_159": {
            "traces": trace,
            "diagnostic_without_job": {
                "mean_jct_seconds": without_target,
                "effects": without_target_effects,
                "diagnostic_only": True,
            },
        },
        "validation_sha256": sha256(root / "validation_summary.json"),
        "result_hashes": {
            policy: sha256(root / "policies" / policy / "result.json")
            for policy in POLICIES
        },
    }
    write(root / "ablation_analysis.json", analysis)
    print(json.dumps({
        "status": analysis["status"],
        "average_jct": jct,
        "effects": effects,
        "target_jct": {policy: trace[policy]["jct_seconds"] for policy in POLICIES},
    }, indent=2, default=json_default))


if __name__ == "__main__":
    main()
