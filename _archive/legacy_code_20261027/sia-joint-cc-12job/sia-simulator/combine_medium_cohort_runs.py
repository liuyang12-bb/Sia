"""Combine nine independently validated Medium cohort runs without copying results."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    roots = [Path(item).resolve() for item in args.inputs]
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite combined output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    manifests = []
    records = []
    for root in roots:
        manifest_path = root / "run_manifest.json"
        manifest = load(manifest_path)
        if manifest.get("status") != "COMPLETE":
            raise ValueError(f"incomplete cohort run: {root}")
        if manifest.get("response") != "medium":
            raise ValueError(f"not a Medium run: {root}")
        if manifest.get("completed_policy_runs") != 5 or manifest.get("completed_cohorts") != 1:
            raise ValueError(f"expected one complete five-policy cohort: {root}")
        if not manifest.get("all_cohorts_validated") or len(manifest["cohort_runs"]) != 1:
            raise ValueError(f"cohort validation missing: {root}")
        record = manifest["cohort_runs"][0]
        validation = load(Path(record["root"]) / "validation_summary.json")
        if validation.get("status") != "PASS" or not validation.get("all_checks_passed"):
            raise ValueError(f"validation failed: {root}")
        manifests.append((root, manifest_path, manifest))
        records.append(record)

    cohort_ids = [item["cohort_id"] for item in records]
    expected = {
        "w1_early", "w1_middle", "w1_late",
        "w5_early", "w5_middle", "w5_late",
        "w10_early", "w10_middle", "w10_late",
    }
    if len(cohort_ids) != len(set(cohort_ids)) or set(cohort_ids) != expected:
        raise ValueError(f"cohort set mismatch: {cohort_ids}")
    invariant_fields = (
        "git_head", "preregistration_sha256", "frozen_selection_sha256",
        "required_source_hashes", "profile_hashes", "dependencies",
        "memory_threshold_gib", "response",
    )
    first = manifests[0][2]
    for field in invariant_fields:
        if any(manifest[field] != first[field] for _, _, manifest in manifests[1:]):
            raise ValueError(f"invariant differs across cohort runs: {field}")

    order = {name: idx for idx, name in enumerate(
        ("w1_early", "w1_middle", "w1_late", "w5_early", "w5_middle", "w5_late",
         "w10_early", "w10_middle", "w10_late")
    )}
    records.sort(key=lambda item: order[item["cohort_id"]])
    combined = {
        "status": "COMPLETE",
        "label": first["label"],
        "scope": first["scope"],
        "git_head": first["git_head"],
        "git_branch": first["git_branch"],
        "response": "medium",
        "preregistration_sha256": first["preregistration_sha256"],
        "frozen_selection_sha256": first["frozen_selection_sha256"],
        "required_source_hashes": first["required_source_hashes"],
        "profile_hashes": first["profile_hashes"],
        "dependencies": first["dependencies"],
        "memory_threshold_gib": first["memory_threshold_gib"],
        "available_memory_samples": [
            {**sample, "source_run": str(root)}
            for root, _, manifest in manifests
            for sample in manifest["available_memory_samples"]
        ],
        "expected_policy_runs": 45,
        "completed_policy_runs": 45,
        "completed_cohorts": 9,
        "cohort_runs": records,
        "failure": None,
        "all_cohorts_validated": True,
        "execution_isolation": (
            "Each cohort ran in a fresh Python process after the monolithic attempt honored "
            "the 2 GiB gate. Policy order within every cohort remained B1/SY/B4/B5/B3."
        ),
        "source_run_manifests": [
            {"root": str(root), "manifest_sha256": sha256(path)}
            for root, path, _ in manifests
        ],
        "command": " ".join(sys.argv),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(combined, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "status": "COMPLETE",
        "cohorts": cohort_ids,
        "output": str(output / "run_manifest.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
