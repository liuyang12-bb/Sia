"""Scenario schema and loader for the tiny joint scheduling experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from .joint_profile import GpuConfig, JobSpec, NodeSpec


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    title: str
    mechanism: str
    resource_model_type: str
    mapping_mode: str
    nodes: Tuple[NodeSpec, ...]
    jobs: Tuple[JobSpec, ...]
    aux_cpu_points: Tuple[float, ...]
    aux_ram_points: Tuple[float, ...]
    allocation_bonus: float
    source_path: str
    source_sha256: str


def load_scenario(path: Path) -> Scenario:
    path = Path(path).resolve()
    raw_bytes = path.read_bytes()
    data = json.loads(raw_bytes.decode("utf-8"))
    if data.get("resource_model_type") != "synthetic_cross_hardware_transfer":
        raise ValueError("scenario must be explicitly synthetic_cross_hardware_transfer")
    if data.get("mapping_mode") not in {"fixed_hash", "category_mix"}:
        raise ValueError("mapping_mode must be fixed_hash or category_mix")
    nodes = tuple(NodeSpec(**node) for node in data["nodes"])
    jobs: List[JobSpec] = []
    for item in data["jobs"]:
        configs = tuple(GpuConfig(**config) for config in item["configs"])
        jobs.append(
            JobSpec(
                job_id=item["job_id"],
                template_file=item["template_file"],
                configs=configs,
                arrival_time=float(item.get("arrival_time", 0)),
                observed_progress=float(item.get("observed_progress", 0)),
                checkpoint_locked=bool(item.get("checkpoint_locked", False)),
                current_config_id=item.get("current_config_id"),
                current_nodes=tuple(item.get("current_nodes", [])),
                mapping_mode=data["mapping_mode"],
            )
        )
    return Scenario(
        scenario_id=data["scenario_id"],
        title=data["title"],
        mechanism=data["mechanism"],
        resource_model_type=data["resource_model_type"],
        mapping_mode=data["mapping_mode"],
        nodes=nodes,
        jobs=tuple(jobs),
        aux_cpu_points=tuple(map(float, data["aux_cpu_points"])),
        aux_ram_points=tuple(map(float, data["aux_ram_points"])),
        allocation_bonus=float(data.get("allocation_bonus", 10)),
        source_path=str(path),
        source_sha256=hashlib.sha256(raw_bytes).hexdigest().upper(),
    )
