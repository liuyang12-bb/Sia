"""Run a pre-registered multi-workload five-policy experiment.

All cohorts are extracted and hashed before the first scheduler is invoked.
Each policy writes to its own directory.  Runs are strictly sequential and the
2 GiB available-memory gate is checked immediately before every policy.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import cvxpy
import cylp
import numpy
import pandas as pd
import psutil
import scipy

import simulator_config as sim_config
from run_cpu_bottleneck_smoke import (
    POLICIES,
    hard_cpu_capacity_audit,
    json_default,
    run_one,
    sha256,
)
from run_cpu_contention_4job import execution_evidence


EXPECTED_COLUMNS = ["name", "time", "application", "num_replicas", "batch_size"]
CAPACITY = "cpu_constrained"
SUPPORTED_RESPONSES = {"neutral", "medium"}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=json_default), encoding="utf-8")


def available_memory_gib() -> float:
    return psutil.virtual_memory().available / 1024**3


def stable_json_hash(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest().upper()


def numeric_job_id(name: str) -> int:
    match = re.search(r"-(\d+)$", name)
    if not match:
        raise ValueError(f"job name lacks numeric suffix: {name}")
    return int(match.group(1))


def synergy_gpu_type(job_name: str, demand: int) -> str:
    job_number = numeric_job_id(job_name)
    if demand <= 6:
        return "rtx"
    if demand == 8:
        return "dgx-ext"
    if demand >= 10:
        return "aws" if job_number % 2 == 0 else "dgx-ext"
    raise ValueError(f"pre-registered SY mapping has no rule for demand={demand}")


def dependency_manifest() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "cvxpy": cvxpy.__version__,
        "cylp": cylp.__version__,
        "cvxpy_installed_solvers": cvxpy.installed_solvers(),
        "cbc_version": "not exposed by installed CyLP and no standalone version is assumed",
        "highs_version": "not exposed by installed SciPy public API",
        "sia_gpu_intent_solver": "CVXPY+CBC",
        "joint_column_solver": "scipy.optimize.milp (HiGHS backend)",
        "joint_milp_options": {"time_limit_seconds": 30.0, "mip_rel_gap": 0.0},
        "sia_cbc_options": {
            "numberThreads": int(sim_config.sia_solver_num_threads),
            "allowablePercentageGap": float(sim_config.sia_solver_thresh * 100),
        },
    }


def required_source_hashes(simulator_root: Path) -> dict:
    names = [
        "job.py",
        "run_cpu_bottleneck_smoke.py",
        "run_cpu_contention_4job.py",
        "run_cpu_contention_20job.py",
        "validate_cpu_contention_20job.py",
        "run_b3_multiworkload_neutral.py",
        "analyze_b3_multiworkload_neutral.py",
        "analyze_b3_multiworkload_medium.py",
        "combine_medium_cohort_runs.py",
        "freeze_medium_response_manifest.py",
        "joint_prototype/joint_integration.py",
        "joint_prototype/joint_profile.py",
        "joint_prototype/joint_scheduler.py",
        "joint_prototype/synergy_adapter.py",
        "joint_prototype/multiworkload_neutral_preregistration.json",
        "joint_prototype/multiworkload_medium_preregistration.json",
        "joint_prototype/cpu_contention_20job_config.json",
        "tests/test_medium_response.py",
    ]
    return {name: sha256(simulator_root / name) for name in names}


def prepare_all_cohorts(prereg, prereg_path, output, simulator_root, response):
    base_config_path = (simulator_root / prereg["base_config"]).resolve()
    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    app_templates = base_config["response"]["application_template_rule"]
    expected_rows = int(prereg["required_trace_rows"])
    cohort_records = []
    trace_records = []

    for trace_order, trace_rel in enumerate(prereg["source_traces"]):
        trace_path = (simulator_root / trace_rel).resolve()
        if not trace_path.is_file():
            raise FileNotFoundError(f"missing pre-registered official trace: {trace_path}")
        source = pd.read_csv(trace_path)
        if list(source.columns) != EXPECTED_COLUMNS:
            raise ValueError(f"unexpected columns in {trace_path.name}: {list(source.columns)}")
        if len(source) != expected_rows:
            raise ValueError(f"{trace_path.name} has {len(source)} rows, expected {expected_rows}")
        if source["name"].duplicated().any():
            raise ValueError(f"duplicate job names in {trace_path.name}")
        missing_apps = sorted(set(source["application"]) - set(app_templates))
        if missing_apps:
            raise ValueError(f"unmapped applications in {trace_path.name}: {missing_apps}")

        indexed = source.copy()
        indexed["__original_row_index"] = range(len(indexed))
        ordered = indexed.sort_values(
            ["time", "__original_row_index"], kind="mergesort"
        ).reset_index(drop=True)
        trace_slug = trace_path.stem.replace("workload-", "w")
        trace_records.append({
            "trace_order": trace_order,
            "trace": trace_rel,
            "path": str(trace_path),
            "sha256": sha256(trace_path),
            "rows": len(source),
            "columns": list(source.columns),
        })

        for cohort_order, (label, window) in enumerate(prereg["cohorts"].items()):
            start = int(window["start_position_one_based"]) - 1
            end = int(window["end_position_one_based"])
            selected = ordered.iloc[start:end].copy()
            if len(selected) != 20:
                raise ValueError(f"{trace_path.name}/{label} selected {len(selected)} jobs")
            original_arrivals = selected["time"].astype(float).tolist()
            first_arrival = float(selected["time"].min())
            selected["time"] = selected["time"].astype(float) - first_arrival
            source_rows = selected["__original_row_index"].astype(int).tolist()
            selected = selected[EXPECTED_COLUMNS].reset_index(drop=True)

            cohort_id = f"{trace_slug}_{label}"
            cohort_root = output / "cohorts" / cohort_id
            cohort_root.mkdir(parents=True, exist_ok=False)
            workload_path = cohort_root / "workload.csv"
            selected.to_csv(workload_path, index=False)

            config = copy.deepcopy(base_config)
            names = selected["name"].tolist()
            template_map = {
                row.name: app_templates[row.application]
                for row in selected.itertuples(index=False)
            }
            type_map = {
                row.name: synergy_gpu_type(row.name, int(row.num_replicas))
                for row in selected.itertuples(index=False)
            }
            demand_map = {
                row.name: int(row.num_replicas)
                for row in selected.itertuples(index=False)
            }
            config.update({
                "experiment_scope": prereg["scope"],
                "preregistration_note": prereg["parameter_policy"],
                "cohort_rule": (
                    f"{trace_path.name} stable arrival order positions "
                    f"{window['start_position_one_based']}..{window['end_position_one_based']}"
                ),
                "source_workload": str(trace_path),
                "workload_rows": names,
                "disclaimer": prereg["disclaimer"],
            })
            config["response"]["template_map"] = template_map
            config["synergy_adaptation"]["fixed_gpu_type_map"] = type_map
            config["synergy_adaptation"]["fixed_gpu_demand_map"] = demand_map
            config["planned_runs"]["formal_first_smoke"] = {
                "capacities": [CAPACITY], "responses": [response],
                "policies": list(prereg["policies"]),
            }
            config["simulation"]["minimum_available_memory_gib"] = float(
                prereg["minimum_available_memory_gib"]
            )
            config_path = cohort_root / "derived_config.json"
            write_json(config_path, config)

            cohort_manifest = {
                "cohort_id": cohort_id,
                "trace": trace_rel,
                "trace_sha256": sha256(trace_path),
                "trace_order": trace_order,
                "cohort_label": label,
                "cohort_order": cohort_order,
                "stable_position_range_one_based": [start + 1, end],
                "job_count": 20,
                "job_names": names,
                "original_zero_based_row_indices": source_rows,
                "source_csv_line_numbers": [value + 2 for value in source_rows],
                "original_arrivals": original_arrivals,
                "arrival_offset_seconds": first_arrival,
                "normalized_arrivals": selected["time"].astype(float).tolist(),
                "relative_gaps_preserved": [
                    float(value - original_arrivals[0]) for value in original_arrivals
                ] == selected["time"].astype(float).tolist(),
                "application_counts": dict(sorted(Counter(selected["application"]).items())),
                "gpu_demand_counts": {
                    str(key): int(value)
                    for key, value in sorted(Counter(selected["num_replicas"].astype(int)).items())
                },
                "template_map": template_map,
                "synergy_gpu_type_map": type_map,
                "synergy_gpu_demand_map": demand_map,
                "workload_sha256": sha256(workload_path),
                "derived_config_sha256": sha256(config_path),
            }
            write_json(cohort_root / "cohort_manifest.json", cohort_manifest)
            cohort_records.append({
                **cohort_manifest,
                "root": str(cohort_root),
                "workload": str(workload_path),
                "config": str(config_path),
            })

    expected_cohorts = len(prereg["source_traces"]) * len(prereg["cohorts"])
    if len(cohort_records) != expected_cohorts:
        raise AssertionError(f"prepared {len(cohort_records)} cohorts, expected {expected_cohorts}")
    return base_config_path, trace_records, cohort_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preregistration",
        default="joint_prototype/multiworkload_neutral_preregistration.json",
    )
    parser.add_argument("--synergy-models", required=True)
    parser.add_argument("--previous-regression", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--cohort-ids",
        help="Optional comma-separated pre-registered cohort subset for targeted validation.",
    )
    args = parser.parse_args()

    simulator_root = Path(__file__).resolve().parent
    prereg_path = Path(args.preregistration).resolve()
    models = Path(args.synergy_models).resolve()
    previous_regression = Path(args.previous_regression).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite historical output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    if prereg["policies"] != list(POLICIES):
        raise ValueError(f"policy order differs from frozen order: {prereg['policies']}")
    response = str(prereg["response"])
    if prereg["capacity"] != CAPACITY or response not in SUPPORTED_RESPONSES:
        raise ValueError("only frozen cpu_constrained neutral/medium matrices are allowed")
    if float(prereg["minimum_available_memory_gib"]) != 2.0:
        raise ValueError("memory gate must remain exactly 2 GiB")
    if not previous_regression.is_dir():
        raise FileNotFoundError(previous_regression)

    base_config_path, traces, all_cohorts = prepare_all_cohorts(
        prereg, prereg_path, output, simulator_root, response
    )
    if args.cohort_ids:
        requested = [item.strip() for item in args.cohort_ids.split(",") if item.strip()]
        known = {item["cohort_id"] for item in all_cohorts}
        if not requested or len(requested) != len(set(requested)) or not set(requested) <= known:
            raise ValueError(f"invalid targeted cohort set: {requested}; known={sorted(known)}")
        cohorts = [item for item in all_cohorts if item["cohort_id"] in set(requested)]
    else:
        cohorts = all_cohorts
    template_files = sorted({
        template for cohort in cohorts for template in cohort["template_map"].values()
    })
    missing = [name for name in template_files if not (models / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing Synergy templates: {missing}")

    repo_root = simulator_root.parent
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    git_branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=repo_root, text=True
    ).strip() or "DETACHED"
    threshold = float(prereg["minimum_available_memory_gib"])
    memory_preflight = available_memory_gib()
    dependencies = dependency_manifest()
    source_hashes = required_source_hashes(simulator_root)
    profile_hashes = {name: sha256(models / name) for name in template_files}
    frozen_selection_hash = stable_json_hash([
        {
            "cohort_id": item["cohort_id"],
            "trace_sha256": item["trace_sha256"],
            "rows": item["original_zero_based_row_indices"],
            "jobs": item["job_names"],
            "workload_sha256": item["workload_sha256"],
            "config_sha256": item["derived_config_sha256"],
        }
        for item in all_cohorts
    ])
    preflight = {
        "status": "PASS" if memory_preflight >= threshold else "BLOCKED_MEMORY_GATE",
        "label": prereg["disclaimer"],
        "scope": prereg["scope"],
        "git_head": git_head,
        "git_branch": git_branch,
        "preregistration": str(prereg_path),
        "preregistration_sha256": sha256(prereg_path),
        "base_config": str(base_config_path),
        "base_config_sha256": sha256(base_config_path),
        "trace_inputs": traces,
        "cohorts": all_cohorts,
        "frozen_selection_sha256": frozen_selection_hash,
        "required_source_hashes": source_hashes,
        "profile_hashes": profile_hashes,
        "dependencies": dependencies,
        "available_memory_gib": memory_preflight,
        "minimum_available_memory_gib": threshold,
        "hard_cpu_capacity_audit": hard_cpu_capacity_audit(
            json.loads(base_config_path.read_text(encoding="utf-8"))
        ),
        "frozen_matrix": {
            "prepared_cohort_count": len(all_cohorts),
            "run_cohort_ids": [item["cohort_id"] for item in cohorts],
            "policies": list(POLICIES),
            "capacity": CAPACITY, "response": response,
        },
        "command": " ".join(sys.argv),
    }
    write_json(output / "preflight.json", preflight)

    master = {
        "status": "RUNNING" if memory_preflight >= threshold else "ABORTED_MEMORY_GATE",
        "label": prereg["disclaimer"],
        "scope": prereg["scope"],
        "git_head": git_head,
        "git_branch": git_branch,
        "preregistration_sha256": sha256(prereg_path),
        "frozen_selection_sha256": frozen_selection_hash,
        "required_source_hashes": source_hashes,
        "profile_hashes": profile_hashes,
        "dependencies": dependencies,
        "memory_threshold_gib": threshold,
        "available_memory_samples": [
            {"phase": "preflight", "available_gib": memory_preflight}
        ],
        "expected_policy_runs": len(cohorts) * len(POLICIES),
        "response": response,
        "completed_policy_runs": 0,
        "completed_cohorts": 0,
        "cohort_runs": [],
        "failure": None,
        "command": " ".join(sys.argv),
    }

    def write_master():
        write_json(output / "run_manifest.json", master)

    write_master()
    if memory_preflight < threshold:
        raise RuntimeError(
            f"available memory {memory_preflight:.3f} GiB below frozen {threshold:.3f} GiB gate"
        )

    sim_config.num_parallel_jobs = 1
    sim_config.num_parallel_traces = 1
    sim_config.sia_solver_verbose = False
    experiment_start = time.perf_counter()
    for cohort in cohorts:
        cohort_root = Path(cohort["root"])
        workload_path = Path(cohort["workload"])
        config_path = Path(cohort["config"])
        workload = pd.read_csv(workload_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        cohort_run = {
            "cohort_id": cohort["cohort_id"],
            "status": "RUNNING",
            "root": str(cohort_root),
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "workload": str(workload_path),
            "workload_sha256": sha256(workload_path),
            "source_workload": cohort["trace"],
            "source_workload_sha256": cohort["trace_sha256"],
            "runs": [],
        }
        master["cohort_runs"].append(cohort_run)
        cohort_preflight = {
            "status": "PASS",
            "label": prereg["disclaimer"],
            "experiment_scope": prereg["scope"],
            "cohort": {
                "cohort_id": cohort["cohort_id"],
                "cohort_rule": config["cohort_rule"],
                "job_count": 20,
                "numeric_job_ids": [numeric_job_id(name) for name in cohort["job_names"]],
                "arrival_min": float(workload["time"].min()),
                "arrival_max": float(workload["time"].max()),
                "arrival_span_seconds": float(workload["time"].max() - workload["time"].min()),
                "application_counts": cohort["application_counts"],
                "gpu_demand_counts": cohort["gpu_demand_counts"],
                "synergy_gpu_type_counts": dict(sorted(Counter(cohort["synergy_gpu_type_map"].values()).items())),
                "mapping_complete": True,
            },
            "available_memory_gib": available_memory_gib(),
            "minimum_available_memory_gib": threshold,
            "config_sha256": sha256(config_path),
            "workload_sha256": sha256(workload_path),
            "source_workload_sha256": cohort["trace_sha256"],
            "profile_hashes": profile_hashes,
            "hard_cpu_capacity_audit": hard_cpu_capacity_audit(config),
            "frozen_matrix": {"capacities": [CAPACITY], "responses": [response], "policies": list(POLICIES)},
            "run_kind": f"MULTI_WORKLOAD_{response.upper()}_COHORT",
            "deferred_medium": prereg.get("deferred", []),
            "git_head": git_head,
            "required_source_hashes": source_hashes,
            "dependencies": dependencies,
        }
        write_json(cohort_root / "preflight.json", cohort_preflight)
        cohort_memory = []
        cohort_start = time.perf_counter()

        def write_cohort(status):
            manifest = {
                "status": status,
                "label": prereg["disclaimer"],
                "experiment_scope": prereg["scope"],
                "platform": platform.platform(),
                "python": sys.version,
                "git_head": git_head,
                "git_branch": git_branch,
                "available_memory_samples": cohort_memory,
                "memory_threshold_gib": threshold,
                "config": str(config_path),
                "config_sha256": sha256(config_path),
                "workload": str(workload_path),
                "workload_sha256": sha256(workload_path),
                "source_workload": cohort["trace"],
                "source_workload_sha256": cohort["trace_sha256"],
                "profile_hashes": profile_hashes,
                "cohort": cohort_preflight["cohort"],
                "hard_cpu_capacity_audit": cohort_preflight["hard_cpu_capacity_audit"],
                "frozen_matrix": cohort_preflight["frozen_matrix"],
                "deferred_medium": cohort_preflight["deferred_medium"],
                "commands": [" ".join(sys.argv)],
                "completed_run_count": len(cohort_run["runs"]),
                "expected_run_count": len(POLICIES),
                "duration_seconds": time.perf_counter() - cohort_start,
                "runs": cohort_run["runs"],
                "failure": master["failure"],
                "required_source_hashes": source_hashes,
                "dependencies": dependencies,
            }
            write_json(cohort_root / "run_manifest.json", manifest)

        write_cohort("RUNNING")
        for policy in POLICIES:
            gc.collect()
            available = available_memory_gib()
            sample = {
                "phase": "before_run", "cohort_id": cohort["cohort_id"],
                "capacity": CAPACITY, "response": response, "policy": policy,
                "available_gib": available,
            }
            cohort_memory.append(sample)
            master["available_memory_samples"].append(sample)
            write_cohort("RUNNING")
            write_master()
            if available < threshold:
                master["status"] = "ABORTED_MEMORY_GATE"
                cohort_run["status"] = "ABORTED_MEMORY_GATE"
                master["failure"] = {
                    **sample,
                    "reason": f"available memory {available:.3f} GiB below {threshold:.3f} GiB",
                }
                write_cohort("ABORTED_MEMORY_GATE")
                write_master()
                raise RuntimeError(master["failure"]["reason"])

            policy_root = cohort_root / "policies" / policy
            policy_root.mkdir(parents=True, exist_ok=False)
            result_path = policy_root / "result.json"
            console_path = policy_root / "console.log"
            policy_start = time.perf_counter()
            try:
                with console_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
                    result = run_one(args, config, workload, policy, CAPACITY, response)
            except Exception as exc:
                failure = {
                    "cohort_id": cohort["cohort_id"], "policy": policy,
                    "exception_type": type(exc).__name__, "exception_message": str(exc),
                    "traceback": traceback.format_exc(),
                    "diagnostic": getattr(exc, "joint_diagnostic", None),
                }
                master["failure"] = failure
                master["status"] = "FAILED_POLICY_EXECUTION"
                cohort_run["status"] = "FAILED_POLICY_EXECUTION"
                write_json(policy_root / "failure.json", failure)
                write_cohort("FAILED_POLICY_EXECUTION")
                write_master()
                raise

            evidence = execution_evidence(result, config, CAPACITY)
            result["contention_evidence"] = evidence
            write_json(result_path, result)
            policy_manifest = {
                "status": "SUCCESS",
                "cohort_id": cohort["cohort_id"], "policy": policy,
                "git_head": git_head,
                "config_sha256": sha256(config_path),
                "workload_sha256": sha256(workload_path),
                "source_workload_sha256": cohort["trace_sha256"],
                "profile_hashes": profile_hashes,
                "required_source_hashes": source_hashes,
                "dependencies": dependencies,
                "available_memory_gib_before_run": available,
                "wall_seconds_outer": time.perf_counter() - policy_start,
                "result_sha256": sha256(result_path),
                "console_sha256": sha256(console_path),
            }
            write_json(policy_root / "policy_manifest.json", policy_manifest)
            record = {
                "capacity": CAPACITY, "response": response, "policy": policy,
                "file": str(result_path.relative_to(cohort_root)).replace("\\", "/"),
                **result["summary"],
                "contention_evidence": {
                    "wait_reason_rounds": evidence["wait_reason_rounds"],
                    "cpu_saturated_gpu_idle_event_count": evidence["cpu_saturated_gpu_idle_event_count"],
                    "single_multiplier_max_abs_residual": evidence["single_multiplier_max_abs_residual"],
                },
            }
            cohort_run["runs"].append(record)
            master["completed_policy_runs"] += 1
            print(
                f"{cohort['cohort_id']} {policy}: "
                f"{result['summary']['completed_jobs']}/20, "
                f"avg_jct={result['summary']['average_jct_completed_only']}, "
                f"rounds={result['summary']['rounds']}", flush=True,
            )
            del result
            gc.collect()
            write_cohort("RUNNING")
            write_master()

        cohort_run["status"] = "COMPLETE"
        write_cohort("COMPLETE")
        validator = simulator_root / "validate_cpu_contention_20job.py"
        validation_log = cohort_root / "validation_console.log"
        command = [
            sys.executable, str(validator), "--input", str(cohort_root),
            "--config", str(config_path), "--code-root", str(simulator_root),
            "--previous-regression", str(previous_regression),
            "--contention-observation-mode", "record-only",
            "--expected-response", response,
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        validation_log.write_text(
            completed.stdout + ("\nSTDERR\n" + completed.stderr if completed.stderr else ""),
            encoding="utf-8",
        )
        if completed.returncode != 0:
            master["status"] = "FAILED_VALIDATION"
            cohort_run["status"] = "FAILED_VALIDATION"
            master["failure"] = {
                "cohort_id": cohort["cohort_id"],
                "reason": "unified validator failed",
                "returncode": completed.returncode,
                "validation_log": str(validation_log),
            }
            write_master()
            raise RuntimeError(f"validator failed for {cohort['cohort_id']}")
        cohort_run["validation"] = "PASS"
        cohort_run["validation_summary_sha256"] = sha256(cohort_root / "validation_summary.json")
        master["completed_cohorts"] += 1
        write_master()

    master["status"] = "COMPLETE"
    master["duration_seconds"] = time.perf_counter() - experiment_start
    master["all_cohorts_validated"] = all(
        item.get("validation") == "PASS" for item in master["cohort_runs"]
    )
    write_master()


if __name__ == "__main__":
    main()
