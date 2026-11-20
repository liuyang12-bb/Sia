"""Unified Sia-Goodput and Synergy-response query layer.

This module deliberately reads the official Synergy ``*_1.py`` matrices at
runtime instead of copying their numbers.  The transfer to Sia GPU configs is
synthetic and every result carries source/provenance metadata.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np


CPU_COORDS = np.asarray([1, 2, 3, 4, 5, 6, 9, 12, 24], dtype=float)
RAM_COORDS = np.asarray([20.83, 62.5, 125, 187.5, 250], dtype=float)
SOURCE_REFERENCE = (3.0, 62.5)


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    gpu_type: str
    gpus: int
    cpu_cores: float
    ram_gb: float


@dataclass(frozen=True)
class GpuConfig:
    config_id: str
    gpu_type: str
    gpus: int
    num_nodes: int
    sia_goodput: float
    cpu_demand_scale: float = 1.0
    memory_demand_scale: float = 1.0
    legal: bool = True


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    template_file: str
    configs: Tuple[GpuConfig, ...]
    arrival_time: float = 0.0
    observed_progress: float = 0.0
    checkpoint_locked: bool = False
    current_config_id: str | None = None
    current_nodes: Tuple[str, ...] = ()
    mapping_mode: str = "category_mix"


@dataclass(frozen=True)
class ProfileTemplate:
    template_id: str
    source_path: str
    source_sha256: str
    cpu_demand: float
    memory_demand_gb: float
    batch: float
    matrix: np.ndarray


@dataclass(frozen=True)
class ProfileQuery:
    job_id: str
    config_id: str
    gpu_type: str
    gpus: int
    cpu_per_gpu: float
    ram_gb_per_gpu: float
    raw_sia_goodput: float
    response_raw: float
    response_baseline: float
    raw_rho: float
    response_exponent: float
    rho: float
    effective_goodput: float
    template_id: str
    template_source: str
    template_sha256: str
    profile_baseline_cpu: float
    profile_baseline_ram_gb: float
    source_cpu_coordinate: float
    source_ram_coordinate: float
    requested_source_cpu_coordinate: float
    requested_source_ram_coordinate: float
    baseline_source_cpu_coordinate: float
    baseline_source_ram_coordinate: float
    requested_baseline_source_cpu_coordinate: float
    requested_baseline_source_ram_coordinate: float
    source_coordinate_clipped: bool
    baseline_coordinate_clipped: bool
    synthetic_cpu_demand: float
    synthetic_ram_demand_gb: float
    normalization_mode: str
    evidence: str = "synthetic_cross_hardware_transfer"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _safe_numeric_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _safe_numeric_eval(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left, right = _safe_numeric_eval(node.left), _safe_numeric_eval(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        return left / right
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"int", "float"} and len(node.args) == 1:
        value = _safe_numeric_eval(node.args[0])
        return float(int(value)) if node.func.id == "int" else float(value)
    raise ValueError(f"unsupported numeric expression: {ast.dump(node)}")


def _literal_assignment(tree: ast.AST, attribute: str):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Attribute) and target.attr == attribute:
            value = node.value
            if attribute == "tput" and isinstance(value, ast.Call) and value.args:
                value = value.args[0]
            if attribute == "tput":
                return ast.literal_eval(value)
            return _safe_numeric_eval(value)
    raise ValueError(f"missing self.{attribute} assignment")


def load_synergy_template(path: Path) -> ProfileTemplate:
    """Load only literal profile data from an official Synergy Python file."""
    path = path.resolve()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matrix = np.asarray(_literal_assignment(tree, "tput"), dtype=float)
    if matrix.shape != (len(CPU_COORDS), len(RAM_COORDS)):
        raise ValueError(f"{path}: expected 9x5 tput matrix, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)) or np.any(matrix <= 0):
        raise ValueError(f"{path}: matrix must be finite and positive")
    template = ProfileTemplate(
        template_id=path.stem,
        source_path=str(path),
        source_sha256=_sha256(path),
        cpu_demand=float(_literal_assignment(tree, "cpus")),
        memory_demand_gb=float(_literal_assignment(tree, "mem")),
        batch=float(_literal_assignment(tree, "batch")),
        matrix=matrix,
    )
    return template


def _bilinear(matrix: np.ndarray, cpu: float, memory: float) -> Tuple[float, float, float]:
    """Bilinear interpolation with conservative clipping, no extrapolation."""
    cpu_c = float(np.clip(cpu, CPU_COORDS[0], CPU_COORDS[-1]))
    mem_c = float(np.clip(memory, RAM_COORDS[0], RAM_COORDS[-1]))

    def bracket(axis: np.ndarray, value: float) -> Tuple[int, int, float]:
        hi = int(np.searchsorted(axis, value, side="right"))
        if hi == 0:
            return 0, 0, 0.0
        if hi >= len(axis):
            idx = len(axis) - 1
            return idx, idx, 0.0
        lo = hi - 1
        weight = (value - axis[lo]) / (axis[hi] - axis[lo])
        return lo, hi, float(weight)

    c0, c1, cw = bracket(CPU_COORDS, cpu_c)
    m0, m1, mw = bracket(RAM_COORDS, mem_c)
    low = matrix[c0, m0] * (1 - mw) + matrix[c0, m1] * mw
    high = matrix[c1, m0] * (1 - mw) + matrix[c1, m1] * mw
    return float(low * (1 - cw) + high * cw), cpu_c, mem_c


class JointProfileOracle:
    """The only performance query used by B1, B4 and B3.

    ``fixed_reference`` normalizes every GPU config at the same absolute
    (3 synthetic cores, 62.5 GB) point and permits slowdown below one.
    ``capacity_relative`` normalizes at the supplied proportional point.
    """

    def __init__(
        self,
        synergy_models_dir: Path,
        normalization_mode: str,
        fixed_reference: Tuple[float, float] = SOURCE_REFERENCE,
        neutral_response: bool = False,
        response_exponent: float = 1.0,
    ) -> None:
        if normalization_mode not in {"fixed_reference", "capacity_relative"}:
            raise ValueError(f"unsupported normalization_mode={normalization_mode}")
        self.synergy_models_dir = Path(synergy_models_dir).resolve()
        self.normalization_mode = normalization_mode
        self.fixed_reference = tuple(map(float, fixed_reference))
        self.neutral_response = bool(neutral_response)
        self.response_exponent = 0.0 if self.neutral_response else float(response_exponent)
        if not 0.0 <= self.response_exponent <= 1.0:
            raise ValueError("response_exponent must be in [0, 1]")
        self._templates: Dict[str, ProfileTemplate] = {}

    def template(self, filename: str) -> ProfileTemplate:
        if Path(filename).name != filename or not filename.endswith("_1.py"):
            raise ValueError(f"only official *_1.py filenames are accepted: {filename}")
        if filename not in self._templates:
            path = self.synergy_models_dir / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            self._templates[filename] = load_synergy_template(path)
        return self._templates[filename]

    def query(
        self,
        job: JobSpec,
        config: GpuConfig,
        cpu_per_gpu: float,
        ram_gb_per_gpu: float,
        proportional_point: Tuple[float, float],
    ) -> ProfileQuery:
        if config not in job.configs or not config.legal:
            raise ValueError(f"illegal config {config.config_id} for {job.job_id}")
        if cpu_per_gpu <= 0 or ram_gb_per_gpu <= 0:
            raise ValueError("CPU and RAM allocations must be positive")
        if config.gpus <= 0 or config.num_nodes <= 0 or config.sia_goodput <= 0:
            raise ValueError(f"invalid GPU config {config}")

        template = self.template(job.template_file)
        if self.normalization_mode == "fixed_reference":
            baseline_cpu, baseline_mem = self.fixed_reference
        else:
            baseline_cpu, baseline_mem = map(float, proportional_point)
        if baseline_cpu <= 0 or baseline_mem <= 0:
            raise ValueError("invalid normalization baseline")

        source_ref_cpu, source_ref_mem = SOURCE_REFERENCE
        source_cpu = source_ref_cpu * (cpu_per_gpu / baseline_cpu) / config.cpu_demand_scale
        source_mem = source_ref_mem * (ram_gb_per_gpu / baseline_mem) / config.memory_demand_scale
        baseline_source_cpu = source_ref_cpu / config.cpu_demand_scale
        baseline_source_mem = source_ref_mem / config.memory_demand_scale
        raw, source_cpu_c, source_mem_c = _bilinear(template.matrix, source_cpu, source_mem)
        baseline, baseline_cpu_c, baseline_mem_c = _bilinear(
            template.matrix, baseline_source_cpu, baseline_source_mem
        )
        if baseline <= 0:
            raise ValueError("non-positive response baseline")
        raw_rho = raw / baseline
        # Pre-registered log-effect attenuation.  It is symmetric for gains and
        # slowdowns, preserves rho=1, and does not clip favorable outcomes.
        rho = float(raw_rho ** self.response_exponent)
        effective = config.sia_goodput * rho
        return ProfileQuery(
            job_id=job.job_id,
            config_id=config.config_id,
            gpu_type=config.gpu_type,
            gpus=config.gpus,
            cpu_per_gpu=float(cpu_per_gpu),
            ram_gb_per_gpu=float(ram_gb_per_gpu),
            raw_sia_goodput=config.sia_goodput,
            response_raw=raw,
            response_baseline=baseline,
            raw_rho=raw_rho,
            response_exponent=self.response_exponent,
            rho=rho,
            effective_goodput=effective,
            template_id=template.template_id,
            template_source=template.source_path,
            template_sha256=template.source_sha256,
            profile_baseline_cpu=baseline_cpu,
            profile_baseline_ram_gb=baseline_mem,
            source_cpu_coordinate=source_cpu_c,
            source_ram_coordinate=source_mem_c,
            requested_source_cpu_coordinate=source_cpu,
            requested_source_ram_coordinate=source_mem,
            baseline_source_cpu_coordinate=baseline_cpu_c,
            baseline_source_ram_coordinate=baseline_mem_c,
            requested_baseline_source_cpu_coordinate=baseline_source_cpu,
            requested_baseline_source_ram_coordinate=baseline_source_mem,
            source_coordinate_clipped=(source_cpu != source_cpu_c or source_mem != source_mem_c),
            baseline_coordinate_clipped=(
                baseline_source_cpu != baseline_cpu_c or baseline_source_mem != baseline_mem_c
            ),
            synthetic_cpu_demand=template.cpu_demand * config.cpu_demand_scale,
            synthetic_ram_demand_gb=template.memory_demand_gb * config.memory_demand_scale,
            normalization_mode=self.normalization_mode,
        )

    def provenance_digest(self) -> str:
        payload = {
            "normalization_mode": self.normalization_mode,
            "fixed_reference": self.fixed_reference,
            "neutral_response": self.neutral_response,
            "response_exponent": self.response_exponent,
            "models": {k: v.source_sha256 for k, v in sorted(self._templates.items())},
            "evidence": "synthetic_cross_hardware_transfer",
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest().upper()


def assign_templates_fixed_hash(
    job_ids: Sequence[str], template_files: Sequence[str], seed: int
) -> Dict[str, str]:
    """Deterministic mapping mode A; never uses application labels."""
    if not template_files:
        raise ValueError("template_files cannot be empty")
    result = {}
    for job_id in job_ids:
        key = f"{seed}:{job_id}".encode()
        idx = int(hashlib.sha256(key).hexdigest(), 16) % len(template_files)
        result[job_id] = template_files[idx]
    return result


def assign_templates_category_mix(
    job_ids: Sequence[str], category_templates: Mapping[str, Sequence[str]], mix: Mapping[str, float]
) -> Dict[str, str]:
    """Deterministic mapping mode B with pre-registered category proportions."""
    if set(category_templates) != set(mix):
        raise ValueError("category templates and mix keys differ")
    total = sum(mix.values())
    if total <= 0:
        raise ValueError("mix must be positive")
    categories = sorted(mix)
    normalized = np.asarray([mix[c] / total for c in categories])
    counts = np.floor(normalized * len(job_ids)).astype(int)
    for idx in np.argsort(-(normalized * len(job_ids) - counts))[: len(job_ids) - counts.sum()]:
        counts[idx] += 1
    assignments: Dict[str, str] = {}
    cursor = 0
    for category, count in zip(categories, counts):
        templates = list(category_templates[category])
        if not templates:
            raise ValueError(f"empty template category {category}")
        for offset, job_id in enumerate(sorted(job_ids)[cursor : cursor + count]):
            assignments[job_id] = templates[offset % len(templates)]
        cursor += count
    return assignments
