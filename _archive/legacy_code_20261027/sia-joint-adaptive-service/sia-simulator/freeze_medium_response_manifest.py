"""Freeze and audit the existing pre-registered Medium response model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from joint_prototype.joint_profile import (
    CPU_COORDS,
    RAM_COORDS,
    SOURCE_REFERENCE,
    _bilinear,
    load_synergy_template,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def adjacent_audit(matrix: np.ndarray, axis: int, coordinates: np.ndarray) -> dict:
    values = np.moveaxis(matrix, axis, 0)
    deltas = np.diff(values, axis=0)
    widths = np.diff(coordinates).reshape((-1,) + (1,) * (deltas.ndim - 1))
    slopes = deltas / widths
    second = np.diff(slopes, axis=0)
    return {
        "adjacent_comparisons": int(deltas.size),
        "strict_decrease_count": int(np.sum(deltas < -1e-12)),
        "non_decreasing": bool(np.all(deltas >= -1e-12)),
        "diminishing_slope_comparisons": int(second.size),
        "increasing_marginal_slope_count": int(np.sum(second > 1e-12)),
        "strict_global_diminishing_returns": bool(np.all(second <= 1e-12)),
    }


def response(template, cpu_per_gpu, ram_per_gpu, per_gpu_ratio, alpha):
    phi_c = float(np.clip(per_gpu_ratio, 0.5, 4.0))
    phi_m = float(np.clip(math.sqrt(per_gpu_ratio), 0.5, 2.0))
    source_cpu_requested = SOURCE_REFERENCE[0] * (cpu_per_gpu / SOURCE_REFERENCE[0]) / phi_c
    source_ram_requested = SOURCE_REFERENCE[1] * (ram_per_gpu / SOURCE_REFERENCE[1]) / phi_m
    baseline_cpu_requested = SOURCE_REFERENCE[0] / phi_c
    baseline_ram_requested = SOURCE_REFERENCE[1] / phi_m
    raw, source_cpu, source_ram = _bilinear(
        template.matrix, source_cpu_requested, source_ram_requested
    )
    baseline, baseline_cpu, baseline_ram = _bilinear(
        template.matrix, baseline_cpu_requested, baseline_ram_requested
    )
    raw_rho = raw / baseline
    return {
        "phi_c": phi_c,
        "phi_m": phi_m,
        "raw_rho": raw_rho,
        "rho": raw_rho ** alpha,
        "source_coordinate_clipped": (
            source_cpu != source_cpu_requested or source_ram != source_ram_requested
        ),
        "baseline_coordinate_clipped": (
            baseline_cpu != baseline_cpu_requested or baseline_ram != baseline_ram_requested
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--synergy-models", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    simulator_root = Path(__file__).resolve().parent
    prereg_path = Path(args.preregistration).resolve()
    models_dir = Path(args.synergy_models).resolve()
    target = Path(args.output).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite response manifest: {target}")
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    base_config_path = (simulator_root / prereg["base_config"]).resolve()
    config = json.loads(base_config_path.read_text(encoding="utf-8"))
    response_cfg = config["response"]
    registered = response_cfg["response_settings"]["medium"]
    alpha = float(registered["exponent"])
    declared = prereg["response_model"]
    if prereg["response"] != "medium" or alpha != 0.5:
        raise ValueError("Medium must be the existing pre-registered exponent 0.5")
    if response_cfg["normalization_mode"] != declared["normalization_mode"]:
        raise ValueError("normalization mode differs from preregistration")
    fixed = response_cfg["fixed_reference"]
    if (
        float(fixed["cpu_per_gpu"]) != float(declared["fixed_reference"]["cpu_per_gpu"])
        or float(fixed["ram_gb_per_gpu"]) != float(declared["fixed_reference"]["ram_gb_per_gpu"])
    ):
        raise ValueError("fixed reference differs from preregistration")

    templates = sorted(set(response_cfg["application_template_rule"].values()))
    ratios = (0.25, 0.5, 1.0, 2.0, 4.0, 16.0)
    template_records = {}
    global_rhos = []
    representative = {}
    for filename in templates:
        path = models_dir / filename
        template = load_synergy_template(path)
        samples = []
        for ratio in ratios:
            for cpu in response_cfg["aux_cpu_points"]:
                for ram in response_cfg["aux_ram_points"]:
                    item = response(template, float(cpu), float(ram), ratio, alpha)
                    global_rhos.append(float(item["rho"]))
                    samples.append({
                        "candidate_per_gpu_ratio": ratio,
                        "cpu_per_gpu": float(cpu),
                        "ram_gb_per_gpu": float(ram),
                        **item,
                    })
        reference = response(template, 3.0, 62.5, 1.0, alpha)
        minimum = response(template, 1.0, 20.83, 1.0, alpha)
        representative[filename] = minimum["rho"]
        template_records[filename] = {
            "source_path": str(path.resolve()),
            "source_sha256": sha256(path),
            "matrix_shape": list(template.matrix.shape),
            "matrix_min": float(np.min(template.matrix)),
            "matrix_max": float(np.max(template.matrix)),
            "source_declared_cpu_demand": template.cpu_demand,
            "source_declared_ram_gb_demand": template.memory_demand_gb,
            "source_declared_batch": template.batch,
            "cpu_axis_audit": adjacent_audit(template.matrix, 0, CPU_COORDS),
            "ram_axis_audit": adjacent_audit(template.matrix, 1, RAM_COORDS),
            "fixed_reference_query": reference,
            "minimum_legal_point_at_phi_one": minimum,
            "sampled_medium_rho_min": min(float(item["rho"]) for item in samples),
            "sampled_medium_rho_max": max(float(item["rho"]) for item in samples),
            "sampled_source_clip_count": sum(item["source_coordinate_clipped"] for item in samples),
            "sampled_baseline_clip_count": sum(item["baseline_coordinate_clipped"] for item in samples),
            "sample_count": len(samples),
        }

    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=simulator_root.parent, text=True
    ).strip()
    payload = {
        "status": "FROZEN",
        "scope": prereg["scope"],
        "disclaimer": prereg["disclaimer"],
        "git_head": git_head,
        "preregistration": str(prereg_path),
        "preregistration_sha256": sha256(prereg_path),
        "base_config": str(base_config_path),
        "base_config_sha256": sha256(base_config_path),
        "joint_profile_sha256": sha256(simulator_root / "joint_prototype/joint_profile.py"),
        "joint_integration_sha256": sha256(simulator_root / "joint_prototype/joint_integration.py"),
        "response_setting": "medium",
        "response_exponent_alpha": alpha,
        "normalization_mode": response_cfg["normalization_mode"],
        "fixed_reference": fixed,
        "minimum_legal_profile_point": response_cfg["minimum_legal_profile_point"],
        "cpu_coordinates": CPU_COORDS.tolist(),
        "ram_coordinates": RAM_COORDS.tolist(),
        "aux_cpu_points": response_cfg["aux_cpu_points"],
        "aux_ram_points": response_cfg["aux_ram_points"],
        "formula": declared["formula"],
        "phi_c": declared["phi_c"],
        "phi_m": declared["phi_m"],
        "interpolation": declared["interpolation"],
        "out_of_grid": declared["out_of_grid"],
        "missing_data_rule": declared["missing_template_or_invalid_matrix"],
        "application_template_rule": response_cfg["application_template_rule"],
        "mapping_warning": response_cfg["mapping_warning"],
        "template_records": template_records,
        "representative_minimum_point_rho_by_template": representative,
        "templates_have_distinct_representative_response": len(
            {round(value, 12) for value in representative.values()}
        ) > 1,
        "sampled_response_envelope": {
            "candidate_per_gpu_ratios": list(ratios),
            "rho_min": min(global_rhos),
            "rho_max": max(global_rhos),
            "sample_count": len(global_rhos),
            "not_a_runtime_observation": True,
        },
        "diminishing_returns_interpretation": (
            "Official measured matrices are audited rather than forced monotone/concave. "
            "Strict violations are retained as measurement shape; no smoothing or clipping of rho is added."
        ),
        "goodput_application": (
            "JointCluster sets job.goodput_multiplier once after allocation; Job.step applies it "
            "only to training progress after checkpoint/restore delay."
        ),
        "parameter_policy": prereg["parameter_policy"],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, payload)
    print(json.dumps({
        "status": payload["status"],
        "templates": templates,
        "representative": representative,
        "sampled_response_envelope": payload["sampled_response_envelope"],
        "output": str(target),
    }, indent=2))


if __name__ == "__main__":
    main()
