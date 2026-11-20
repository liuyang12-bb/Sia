"""Compare two frozen five-policy Neutral runs without rewriting either run.

Wall-clock and optimizer-duration fields are intentionally excluded from the
deterministic simulation signature.  Decisions, placements, execution
progress, completion events, resource accounting, and solver statuses remain
in scope.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
from pathlib import Path

import cvxpy
import cylp
import numpy
import scipy

from audit_b3_reproducibility import (
    POLICIES,
    decision_signature,
    execution_signature,
    load,
    policy_metrics,
    sha256,
    stable_hash,
)


NONDETERMINISTIC_SUMMARY_FIELDS = {
    "wall_seconds",
    "optimizer_total_seconds",
    "optimizer_p99_seconds",
}


def deterministic_summary(payload):
    return {
        key: value
        for key, value in payload["summary"].items()
        if key not in NONDETERMINISTIC_SUMMARY_FIELDS
    }


def first_mismatch(reference, repeat):
    limit = min(len(reference), len(repeat))
    for index in range(limit):
        if reference[index] != repeat[index]:
            return {
                "index": index,
                "reference": reference[index],
                "repeat": repeat[index],
            }
    if len(reference) != len(repeat):
        return {
            "index": limit,
            "reference_length": len(reference),
            "repeat_length": len(repeat),
        }
    return None


def manifest_identity(manifest):
    return {
        "git_head": manifest.get("git_head"),
        "config_sha256": manifest.get("config_sha256"),
        "workload_sha256": manifest.get("workload_sha256"),
        "source_workload_sha256": manifest.get("source_workload_sha256"),
        "profile_hashes": manifest.get("profile_hashes"),
        "frozen_matrix": manifest.get("frozen_matrix"),
        "memory_threshold_gib": manifest.get("memory_threshold_gib"),
        "python": manifest.get("python"),
        "platform": manifest.get("platform"),
    }


def dependency_versions():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "cvxpy": cvxpy.__version__,
        "cylp": cylp.__version__,
        "cvxpy_installed_solvers": cvxpy.installed_solvers(),
        "sia_gpu_intent_solver": "CVXPY+CBC",
        "joint_column_mip_solver": "scipy.optimize.milp (HiGHS backend)",
        "highs_embedded_version": "not exposed by the installed SciPy 1.10.1 public API",
    }


def compare_policy(reference_root, repeat_root, policy, first_arrival):
    filename = f"cpu_constrained.neutral.{policy}.json"
    reference_path = reference_root / filename
    repeat_path = repeat_root / filename
    reference = load(reference_path)
    repeat = load(repeat_path)

    reference_decisions = decision_signature(reference)
    repeat_decisions = decision_signature(repeat)
    reference_execution = execution_signature(reference)
    repeat_execution = execution_signature(repeat)
    reference_summary = deterministic_summary(reference)
    repeat_summary = deterministic_summary(repeat)

    reference_metrics = policy_metrics(reference, first_arrival)
    repeat_metrics = policy_metrics(repeat, first_arrival)
    reference_metrics["payload_sha256"] = sha256(reference_path)
    repeat_metrics["payload_sha256"] = sha256(repeat_path)

    result = {
        "policy": policy,
        "reference_file": str(reference_path.resolve()),
        "repeat_file": str(repeat_path.resolve()),
        "raw_payload_sha256": {
            "reference": sha256(reference_path),
            "repeat": sha256(repeat_path),
            "equal": sha256(reference_path) == sha256(repeat_path),
            "note": "Raw payloads include wall-clock and optimizer timings and need not be byte-identical.",
        },
        "deterministic_summary": {
            "equal": reference_summary == repeat_summary,
            "reference_sha256": stable_hash(reference_summary),
            "repeat_sha256": stable_hash(repeat_summary),
        },
        "decision_trace": {
            "equal": reference_decisions == repeat_decisions,
            "reference_sha256": stable_hash(reference_decisions),
            "repeat_sha256": stable_hash(repeat_decisions),
            "first_mismatch": first_mismatch(reference_decisions, repeat_decisions),
        },
        "execution_trace": {
            "equal": reference_execution == repeat_execution,
            "reference_sha256": stable_hash(reference_execution),
            "repeat_sha256": stable_hash(repeat_execution),
            "first_mismatch": first_mismatch(reference_execution, repeat_execution),
        },
        "reference_metrics": reference_metrics,
        "repeat_metrics": repeat_metrics,
    }
    result["exact_simulation_reproduction"] = all(
        (
            result["deterministic_summary"]["equal"],
            result["decision_trace"]["equal"],
            result["execution_trace"]["equal"],
        )
    )

    del reference, repeat, reference_decisions, repeat_decisions
    del reference_execution, repeat_execution, reference_summary, repeat_summary
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--repeat-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reference_root = Path(args.reference_root).resolve()
    repeat_root = Path(args.repeat_root).resolve()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=False)

    reference_manifest_path = reference_root / "run_manifest.json"
    repeat_manifest_path = repeat_root / "run_manifest.json"
    reference_manifest = load(reference_manifest_path)
    repeat_manifest = load(repeat_manifest_path)
    first_arrival = float(reference_manifest["cohort"]["arrival_min"])

    policies = {
        policy: compare_policy(reference_root, repeat_root, policy, first_arrival)
        for policy in POLICIES
    }
    manifest_reference = manifest_identity(reference_manifest)
    manifest_repeat = manifest_identity(repeat_manifest)
    validation_reference = load(reference_root / "validation_summary.json")
    validation_repeat = load(repeat_root / "validation_summary.json")
    all_policy_reproductions = all(
        item["exact_simulation_reproduction"] for item in policies.values()
    )
    provenance_equal = all((
        manifest_reference == manifest_repeat,
        validation_reference.get("input_hashes")
        == validation_repeat.get("input_hashes"),
        validation_reference.get("code_hashes")
        == validation_repeat.get("code_hashes"),
        validation_reference.get("all_checks_passed") is True,
        validation_repeat.get("all_checks_passed") is True,
    ))

    report = {
        "status": "PASS" if all_policy_reproductions and provenance_equal else "FAIL",
        "scope": "FROZEN_NEUTRAL_FIVE_POLICY_REPRODUCIBILITY_AUDIT",
        "reference_root": str(reference_root),
        "repeat_root": str(repeat_root),
        "reference_manifest_sha256": sha256(reference_manifest_path),
        "repeat_manifest_sha256": sha256(repeat_manifest_path),
        "manifest_identity_equal": manifest_reference == manifest_repeat,
        "manifest_identity": {
            "reference": manifest_reference,
            "repeat": manifest_repeat,
        },
        "validation": {
            "reference_status": validation_reference.get("status"),
            "reference_all_checks_passed": validation_reference.get("all_checks_passed"),
            "repeat_status": validation_repeat.get("status"),
            "repeat_all_checks_passed": validation_repeat.get("all_checks_passed"),
            "input_hashes_equal": (
                validation_reference.get("input_hashes")
                == validation_repeat.get("input_hashes")
            ),
            "code_hashes_equal": (
                validation_reference.get("code_hashes")
                == validation_repeat.get("code_hashes")
            ),
            "reference_input_hashes": validation_reference.get("input_hashes"),
            "repeat_input_hashes": validation_repeat.get("input_hashes"),
            "reference_code_hashes": validation_reference.get("code_hashes"),
            "repeat_code_hashes": validation_repeat.get("code_hashes"),
        },
        "dependencies_observed_at_audit_time": dependency_versions(),
        "policies": policies,
        "all_five_exact_simulation_reproductions": all_policy_reproductions,
        "provenance_and_validation_equal": provenance_equal,
        "excluded_nondeterministic_summary_fields": sorted(
            NONDETERMINISTIC_SUMMARY_FIELDS
        ),
    }
    target = output_root / "repeat_comparison.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "output": str(target),
        "manifest_identity_equal": report["manifest_identity_equal"],
        "policy_reproduction": {
            policy: item["exact_simulation_reproduction"]
            for policy, item in policies.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
