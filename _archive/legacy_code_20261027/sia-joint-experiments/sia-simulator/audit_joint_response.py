"""Create source-matrix and historical-rho audit artifacts without rerunning jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from joint_prototype.joint_profile import (
    CPU_COORDS,
    RAM_COORDS,
    GpuConfig,
    JobSpec,
    JointProfileOracle,
    load_synergy_template,
)


def _point(matrix, cpu_index, ram_index):
    return {
        "cpu": float(CPU_COORDS[cpu_index]),
        "ram_gb": float(RAM_COORDS[ram_index]),
        "response": float(matrix[cpu_index, ram_index]),
    }


def audit_matrices(models_dir: Path):
    records = []
    for path in sorted(models_dir.glob("*_1.py")):
        template = load_synergy_template(path)
        matrix = template.matrix
        min_index = np.unravel_index(int(np.argmin(matrix)), matrix.shape)
        max_index = np.unravel_index(int(np.argmax(matrix)), matrix.shape)
        records.append(
            {
                "template": template.template_id,
                "source": template.source_path,
                "sha256": template.source_sha256,
                "declared_cpu_demand": template.cpu_demand,
                "declared_memory_demand_gb": template.memory_demand_gb,
                "declared_batch": template.batch,
                "shape": list(matrix.shape),
                "min": float(matrix.min()),
                "max": float(matrix.max()),
                "max_over_min": float(matrix.max() / matrix.min()),
                "min_point": _point(matrix, *min_index),
                "max_point": _point(matrix, *max_index),
                "key_points": {
                    "grid_min": _point(matrix, 0, 0),
                    "source_reference_3_cpu_62_5_gb": _point(matrix, 2, 1),
                    "declared_9_cpu_62_5_gb": _point(matrix, 6, 1),
                    "grid_max": _point(matrix, -1, -1),
                },
                "matrix": matrix.tolist(),
            }
        )
    return {
        "cpu_grid": CPU_COORDS.tolist(),
        "ram_gb_grid": RAM_COORDS.tolist(),
        "source_reference": {"cpu": 3.0, "ram_gb": 62.5},
        "template_count": len(records),
        "templates": records,
        "global_min": min(item["min"] for item in records),
        "global_max": max(item["max"] for item in records),
        "global_max_over_min": max(item["max"] for item in records)
        / min(item["min"] for item in records),
    }


def reproduce_historical(models_dir: Path, history_path: Path):
    history = json.loads(history_path.read_text(encoding="utf-8"))
    best = None
    for round_record in history["joint_round_logs"]:
        decision = round_record["decision_for_next_interval"]
        for job_id, allocation in decision.get("resources", {}).items():
            if best is None or allocation["rho"] > best["allocation"]["rho"]:
                best = {
                    "decision_interval_start": round_record["interval_start"],
                    "job_id": job_id,
                    "allocation": allocation,
                }
    allocation = best["allocation"]
    provenance = allocation["goodput_provenance"]
    config = GpuConfig(
        allocation["config_id"],
        allocation["gpu_type"],
        int(allocation["gpus"]),
        int(allocation["num_nodes"]),
        float(allocation["normalized_sia_goodput"]),
        float(provenance["cpu_demand_scale"]),
        float(provenance["memory_demand_scale"]),
    )
    job = JobSpec(best["job_id"], f"{allocation['template']}.py", (config,))
    oracle = JointProfileOracle(models_dir, "fixed_reference", (3.0, 62.5))
    query = oracle.query(
        job,
        config,
        float(allocation["cpu_per_gpu"]),
        float(allocation["ram_gb_per_gpu"]),
        (2.25, 50.0),
    )
    recomputed = query.to_dict()
    return {
        "historical_artifact": str(history_path.resolve()),
        "historical_job": best["job_id"],
        "historical_decision_interval_start": best["decision_interval_start"],
        "historical_rho": allocation["rho"],
        "historical_allocation": allocation,
        "formula": {
            "source_cpu": "3 * (cpu_per_gpu / 3) / phi_c",
            "source_ram": "62.5 * (ram_gb_per_gpu / 62.5) / phi_m",
            "baseline_source_cpu": "3 / phi_c",
            "baseline_source_ram": "62.5 / phi_m",
            "raw_rho": "bilinear(F, source_cpu, source_ram) / bilinear(F, baseline_source_cpu, baseline_source_ram)",
        },
        "recomputed_query": recomputed,
        "recomputed_matches": bool(np.isclose(query.rho, allocation["rho"], rtol=0, atol=1e-12)),
        "numerical_identity": f"{query.response_raw:.15g} / {query.response_baseline:.15g} = {query.raw_rho:.15g}",
        "interpretation": {
            "phi_c_source": "synthetic min-per-GPU-rate ratio, clipped to [0.5, 4.0]",
            "phi_m_source": "synthetic sqrt(phi_c preclip ratio), clipped to [0.5, 2.0]",
            "official_measurement": "only the source matrix cells",
            "synthetic_assumptions": [
                "fixed cross-hardware reference",
                "GPU-rate-to-demand scaling",
                "bilinear interpolation",
                "boundary clipping",
                "multiplication of Sia Goodput by the normalized response",
            ],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True)
    parser.add_argument("--history", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    models = Path(args.models).resolve()
    matrix_audit = audit_matrices(models)
    reproduction = reproduce_historical(models, Path(args.history).resolve())
    (output / "source_matrix_audit.json").write_text(
        json.dumps(matrix_audit, indent=2), encoding="utf-8"
    )
    (output / "rho_4_2981_reproduction.json").write_text(
        json.dumps(reproduction, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "template_count": matrix_audit["template_count"],
        "global_range": [matrix_audit["global_min"], matrix_audit["global_max"]],
        "historical_rho": reproduction["historical_rho"],
        "recomputed": reproduction["recomputed_query"]["rho"],
        "matches": reproduction["recomputed_matches"],
    }, indent=2))


if __name__ == "__main__":
    main()
