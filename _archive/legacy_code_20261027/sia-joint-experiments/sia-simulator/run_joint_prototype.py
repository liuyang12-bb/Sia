"""Run the three pre-registered 2-job joint scheduling scenarios."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

from joint_prototype.joint_profile import JointProfileOracle
from joint_prototype.joint_scheduler import JointScheduler
from joint_prototype.scenario_io import load_scenario


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=Path, default=Path("joint_prototype/scenarios"))
    parser.add_argument("--synergy-models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--normalization",
        choices=("fixed_reference", "capacity_relative", "all"),
        default="all",
    )
    args = parser.parse_args()
    scenario_paths = sorted(args.scenarios.glob("scenario_*.json"))
    if len(scenario_paths) != 3:
        raise ValueError(f"expected exactly three scenarios, found {len(scenario_paths)}")
    modes = ("fixed_reference", "capacity_relative") if args.normalization == "all" else (args.normalization,)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    suite_start = time.perf_counter()
    records = []
    for scenario_path in scenario_paths:
        scenario = load_scenario(scenario_path)
        for mode in modes:
            oracle = JointProfileOracle(args.synergy_models, mode)
            scheduler = JointScheduler(
                scenario.nodes,
                scenario.jobs,
                oracle,
                scenario.aux_cpu_points,
                scenario.aux_ram_points,
                scenario.allocation_bonus,
            )
            b1 = scheduler.run_b1()
            b4 = scheduler.run_b4(b1)
            b3 = scheduler.run_b3()
            if not all(result.exact_match for result in (b1, b4, b3)):
                raise RuntimeError(f"MIP/exhaustive mismatch in {scenario.scenario_id}/{mode}")
            result_record = {
                "scenario": {
                    "id": scenario.scenario_id,
                    "title": scenario.title,
                    "mechanism": scenario.mechanism,
                    "resource_model_type": scenario.resource_model_type,
                    "mapping_mode": scenario.mapping_mode,
                    "source_path": scenario.source_path,
                    "source_sha256": scenario.source_sha256,
                },
                "normalization_mode": mode,
                "oracle_provenance_digest": oracle.provenance_digest(),
                "policies": {"B1": b1.to_dict(), "B4": b4.to_dict(), "B3": b3.to_dict()},
                "comparisons": {
                    "B3_minus_B1_actual_goodput": b3.actual_total_goodput - b1.actual_total_goodput,
                    "B3_minus_B4_actual_goodput": b3.actual_total_goodput - b4.actual_total_goodput,
                    "B4_minus_B1_actual_goodput": b4.actual_total_goodput - b1.actual_total_goodput,
                    "B3_differs_from_B4_gpu_config": any(
                        job_id in b4.selected
                        and b3.selected[job_id].config.config_id != b4.selected[job_id].config.config_id
                        for job_id in b3.selected
                    ),
                },
                "label": "SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE",
            }
            target = output / f"{scenario.scenario_id}.{mode}.json"
            write_json(target, result_record)
            records.append(result_record)
    manifest = {
        "status": "SUCCESS",
        "exit_code": 0,
        "command": [sys.executable, *sys.argv],
        "scenario_count": len(scenario_paths),
        "run_count": len(records),
        "normalization_modes": list(modes),
        "python": sys.version,
        "platform": platform.platform(),
        "duration_seconds": time.perf_counter() - suite_start,
        "scenario_inputs": {
            record["scenario"]["id"]: {
                "path": record["scenario"]["source_path"],
                "sha256": record["scenario"]["source_sha256"],
            }
            for record in records
        },
        "results": [f"{r['scenario']['id']}.{r['normalization_mode']}.json" for r in records],
        "all_mip_exact_matches": all(
            result["exact_match"]
            for record in records
            for result in record["policies"].values()
        ),
        "all_capacity_audits_valid": all(
            result["audit"]["valid"]
            for record in records
            for result in record["policies"].values()
        ),
        "label": "SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE",
    }
    write_json(output / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
