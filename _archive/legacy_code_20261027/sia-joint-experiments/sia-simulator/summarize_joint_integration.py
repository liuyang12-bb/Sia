"""Create a compact, reproducible comparison from full integration logs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    files = ["b1.sensitive.json", "b4.sensitive.json", "b3.sensitive.json",
             "b1.neutral.json", "b4.neutral.json", "b3.neutral.json", "sia.control.json"]
    runs = {name: json.loads((root / name).read_text(encoding="utf-8")) for name in files}
    summary = {"label": "SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE", "policies": {}}
    for name, run in runs.items():
        decisions = [r["decision_for_next_interval"] for r in run.get("joint_round_logs", []) if r["decision_for_next_interval"]]
        executions = [e for r in run.get("joint_round_logs", []) for e in r["execution"]]
        configs = Counter(v["config_id"] for d in decisions for v in d["resources"].values())
        rhos = [e["rho"] for e in executions if e["resource_allocation"]]
        solve = [d["result"]["solve_seconds"] for d in decisions]
        summary["policies"][name] = {
            "jcts": run["jcts"],
            "average_jct": run["average_jct"],
            "rounds": run["rounds"],
            "resource_hours": run["resource_hours"],
            "restarts": {job: state["num_restarts"] for job, state in run["jobs"].items()},
            "migrations": {job: state["num_migrations"] for job, state in run["jobs"].items()},
            "selected_config_rounds": dict(configs),
            "rho_min": min(rhos) if rhos else None,
            "rho_max": max(rhos) if rhos else None,
            "solver_seconds_total": sum(solve),
            "solver_seconds_p99": float(np.percentile(solve, 99)) if solve else 0.0,
            "completion_events": {
                job: {"submission_time": state["submission_time"], "completion_time": state["completion_time"], "jct": state["jct"]}
                for job, state in run["jobs"].items()
            },
        }
    b1 = runs["b1.sensitive.json"]["average_jct"]
    b4 = runs["b4.sensitive.json"]["average_jct"]
    b3 = runs["b3.sensitive.json"]["average_jct"]
    summary["comparisons"] = {
        "b4_vs_b1_average_jct_percent": (b4 / b1 - 1) * 100,
        "b3_vs_b1_average_jct_percent": (b3 / b1 - 1) * 100,
        "b3_vs_b4_average_jct_percent": (b3 / b4 - 1) * 100,
        "neutral_b1_equals_sia": runs["b1.neutral.json"]["jcts"] == runs["sia.control.json"]["jcts"],
        "neutral_b4_equals_sia": runs["b4.neutral.json"]["jcts"] == runs["sia.control.json"]["jcts"],
        "neutral_b3_delta_seconds": {
            job: runs["b3.neutral.json"]["jcts"][job] - runs["sia.control.json"]["jcts"][job]
            for job in runs["sia.control.json"]["jcts"]
        },
    }
    active_timeline = {}
    for name in ("b1.sensitive.json", "b4.sensitive.json", "b3.sensitive.json"):
        records = []
        for round_record in runs[name]["joint_round_logs"]:
            active_execution = [
                e for e in round_record["execution"]
                if e["resource_allocation"] or e["progress_delta"] > 0 or e["completion_time"] is not None
            ]
            next_resources = round_record["decision_for_next_interval"].get("resources", {})
            if active_execution or next_resources:
                records.append({
                    "interval_start": round_record["interval_start"],
                    "interval_end": round_record["interval_end"],
                    "execution": active_execution,
                    "next_resources": next_resources,
                    "rescale_events": round_record["rescale_events"],
                    "migrate_events": round_record["migrate_events"],
                })
        active_timeline[name] = records
    summary["active_timeline"] = active_timeline
    (root / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"comparisons": summary["comparisons"], "policy_metrics": summary["policies"]}, indent=2))


if __name__ == "__main__":
    main()
