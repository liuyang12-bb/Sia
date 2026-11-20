"""Read-only paired analysis of the preregistered half-CPU eight-cell run."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / "preregistration.json").read_text(encoding="utf-8"))
LOCATION = {
    "original20_half_medium_joint_g": "attempt2",
    "holdout20_half_medium_sia_p": "attempt2",
    "holdout20_half_medium_sia_aux": "attempt4",
    "holdout20_half_medium_b3_free": "attempt4",
    "holdout20_half_medium_joint_g": "attempt4",
}
METHODS = ("sia_p", "sia_aux", "b3_free", "joint_g")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_cell(cell):
    directory = ROOT / LOCATION.get(cell["id"], "") / cell["id"]
    summary_path = directory / "summary.json"
    validation_path = directory / "independent_validation.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    start = json.loads((directory / "start.json").read_text(encoding="utf-8"))
    assert summary["status"] == "COMPLETE" and summary["completed"] == summary["total"] == 20
    assert validation["valid"] and validation["completed"] == 20
    assert start["manifest_sha256"] == sha(ROOT / "preregistration.json")
    assert start["source_snapshot_sha256"] == PLAN["source_snapshot_hash"]
    assert validation["raw_rounds_sha256"] == sha(directory / "rounds.jsonl.gz")
    assert math.isclose(summary["mean_jct"], validation["independent_mean_jct"], abs_tol=1e-8)
    assert all(j["completed"] and j["completion"] is not None for j in summary["jobs"].values())
    assert not summary["violations"]
    return {
        "cell": cell,
        "location": str(directory),
        "summary_hash": sha(summary_path),
        "validation_hash": sha(validation_path),
        "rounds_hash": validation["raw_rounds_sha256"],
        "summary": summary,
        "validation": validation,
    }


def contrast(base, new):
    old = base["summary"]["jobs"]
    fresh = new["summary"]["jobs"]
    assert set(old) == set(fresh)
    jobs = []
    for name in sorted(old):
        assert old[name]["arrival"] == fresh[name]["arrival"]
        diff = old[name]["jct"] - fresh[name]["jct"]
        jobs.append({"job": name, "base_jct": old[name]["jct"],
                     "new_jct": fresh[name]["jct"], "improvement_seconds": diff})
    delta = base["summary"]["mean_jct"] - new["summary"]["mean_jct"]
    assert math.isclose(sum(j["improvement_seconds"] for j in jobs) / 20, delta, abs_tol=1e-7)
    ranked = sorted(jobs, key=lambda x: x["improvement_seconds"], reverse=True)
    return {
        "base": base["cell"]["method"], "new": new["cell"]["method"],
        "mean_jct_improvement_seconds": delta,
        "mean_jct_improvement_percent": 100 * delta / base["summary"]["mean_jct"],
        "improved_jobs": sum(j["improvement_seconds"] > 0 for j in jobs),
        "unchanged_jobs": sum(j["improvement_seconds"] == 0 for j in jobs),
        "degraded_jobs": sum(j["improvement_seconds"] < 0 for j in jobs),
        "largest_positive_contributors": ranked[:5],
        "largest_negative_contributors": ranked[-5:],
        "jobs": jobs,
    }


def main():
    cells = {c["id"]: load_cell(c) for c in PLAN["run_order"]}
    data = {
        "scope": PLAN["scope"],
        "preregistration_sha256": sha(ROOT / "preregistration.json"),
        "source_snapshot_sha256": PLAN["source_snapshot_hash"],
        "dataset_results": {},
        "all_eight_complete_and_independently_valid": True,
    }
    for dataset in ("original20", "holdout20"):
        group = {m: cells[f"{dataset}_half_medium_{m}"] for m in METHODS}
        metrics = {}
        for method, item in group.items():
            s = item["summary"]
            v = item["validation"]
            metrics[method] = {
                "location": item["location"],
                "summary_sha256": item["summary_hash"],
                "rounds_sha256": item["rounds_hash"],
                "validation_sha256": item["validation_hash"],
                "rounds": s["rounds"], "completed": s["completed"],
                "mean_jct": s["mean_jct"], "median_jct": s["median_jct"],
                "p95_jct": s["p95_jct"], "p99_jct": s["p99_jct"],
                "makespan": s["makespan"], "totals": s["totals"],
                "solver_status": s["solver_status"],
                "fallback_count": s["fallback_count"],
                "cpu_saturated_gpu_idle_node_round_events": s["cpu_saturated_gpu_idle_node_round_events"],
                "clipping_counts": v["clipping"],
                "selected_columns": v["counts"]["selected_columns"],
                "independent_audit": v["counts"],
                "rho_range": s["rho_range"],
                "wall_seconds": s["wall_seconds"],
                "jobs": s["jobs"],
            }
        comparisons = {}
        for base, new in (("sia_p", "sia_aux"), ("sia_p", "b3_free"),
                          ("sia_p", "joint_g"), ("sia_aux", "joint_g"),
                          ("b3_free", "joint_g")):
            comparisons[f"{new}_vs_{base}"] = contrast(group[base], group[new])
        data["dataset_results"][dataset] = {"metrics": metrics, "comparisons": comparisons}
    output = ROOT / "comparison_summary.json"
    assert not output.exists()
    with output.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print("VALID 8/8; paired analysis:", output)
    for ds, value in data["dataset_results"].items():
        print(ds)
        for key, x in value["comparisons"].items():
            print(key, round(x["mean_jct_improvement_percent"], 3),
                  x["improved_jobs"], x["degraded_jobs"],
                  x["largest_positive_contributors"][0]["job"])


if __name__ == "__main__":
    main()
