"""Finite-column joint GPU/CPU/RAM/placement optimizer for tiny experiments."""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .joint_profile import GpuConfig, JobSpec, JointProfileOracle, NodeSpec, ProfileQuery


@dataclass(frozen=True)
class Placement:
    node_gpus: Tuple[Tuple[str, int], ...]

    @property
    def node_ids(self) -> Tuple[str, ...]:
        return tuple(node for node, _ in self.node_gpus)


@dataclass(frozen=True)
class JointColumn:
    column_id: str
    job_id: str
    config: GpuConfig
    placement: Placement
    query: ProfileQuery
    node_usage: Tuple[Tuple[str, float, float, float], ...]
    scheduling_utility: float
    proportional_cpu_per_gpu: float
    proportional_ram_per_gpu: float
    preserves_sia_config: bool
    preserves_node_ids: bool
    query_round: int = 0
    candidate_kind: str = "legacy"

    def usage_for(self, node_id: str) -> Tuple[float, float, float]:
        for nid, gpu, cpu, ram in self.node_usage:
            if nid == node_id:
                return gpu, cpu, ram
        return 0.0, 0.0, 0.0


@dataclass
class PolicyResult:
    policy: str
    status: str
    selected: Dict[str, JointColumn]
    unallocated_jobs: List[str]
    scheduling_objective: float
    actual_total_goodput: float
    solve_seconds: float
    exact_objective: float | None = None
    exact_match: bool | None = None
    audit: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        selected = {}
        for job_id, column in sorted(self.selected.items()):
            selected[job_id] = {
                "column_id": column.column_id,
                "query_round": column.query_round,
                "candidate_kind": column.candidate_kind,
                "config": asdict(column.config),
                "placement": dict(column.placement.node_gpus),
                "cpu_per_gpu": column.query.cpu_per_gpu,
                "ram_gb_per_gpu": column.query.ram_gb_per_gpu,
                "raw_sia_goodput": column.query.raw_sia_goodput,
                "response_raw": column.query.response_raw,
                "response_baseline": column.query.response_baseline,
                "raw_rho": column.query.raw_rho,
                "response_exponent": column.query.response_exponent,
                "rho": column.query.rho,
                "effective_goodput": column.query.effective_goodput,
                "template_id": column.query.template_id,
                "template_source": column.query.template_source,
                "template_sha256": column.query.template_sha256,
                "profile_baseline": {
                    "cpu": column.query.profile_baseline_cpu,
                    "ram_gb": column.query.profile_baseline_ram_gb,
                },
                "source_coordinates": {
                    "requested": {
                        "cpu": column.query.requested_source_cpu_coordinate,
                        "ram_gb": column.query.requested_source_ram_coordinate,
                    },
                    "used": {
                        "cpu": column.query.source_cpu_coordinate,
                        "ram_gb": column.query.source_ram_coordinate,
                    },
                    "clipped": column.query.source_coordinate_clipped,
                },
                "baseline_source_coordinates": {
                    "requested": {
                        "cpu": column.query.requested_baseline_source_cpu_coordinate,
                        "ram_gb": column.query.requested_baseline_source_ram_coordinate,
                    },
                    "used": {
                        "cpu": column.query.baseline_source_cpu_coordinate,
                        "ram_gb": column.query.baseline_source_ram_coordinate,
                    },
                    "clipped": column.query.baseline_coordinate_clipped,
                },
                "synthetic_demand": {
                    "cpu": column.query.synthetic_cpu_demand,
                    "ram_gb": column.query.synthetic_ram_demand_gb,
                },
                "node_usage": [
                    {"node_id": n, "gpus": g, "cpu": c, "ram_gb": m}
                    for n, g, c, m in column.node_usage
                ],
                "preserves_sia_config": column.preserves_sia_config,
                "preserves_node_ids": column.preserves_node_ids,
            }
        return {
            "policy": self.policy,
            "status": self.status,
            "selected": selected,
            "unallocated_jobs": self.unallocated_jobs,
            "scheduling_objective": self.scheduling_objective,
            "actual_total_goodput": self.actual_total_goodput,
            "solve_seconds": self.solve_seconds,
            "exact_objective": self.exact_objective,
            "exact_match": self.exact_match,
            "audit": self.audit,
        }


def enumerate_placements(config: GpuConfig, nodes: Sequence[NodeSpec]) -> List[Placement]:
    compatible = [n for n in nodes if n.gpu_type == config.gpu_type]
    placements: List[Placement] = []
    if config.num_nodes == 1:
        for node in compatible:
            if config.gpus <= node.gpus:
                placements.append(Placement(((node.node_id, config.gpus),)))
    else:
        # Match Sia multi-node candidates: selected nodes are occupied in full.
        for selected in itertools.combinations(compatible, config.num_nodes):
            if sum(n.gpus for n in selected) != config.gpus:
                continue
            ratios = {(round(n.cpu_cores / n.gpus, 9), round(n.ram_gb / n.gpus, 9)) for n in selected}
            if len(ratios) != 1:
                continue
            placements.append(Placement(tuple((n.node_id, n.gpus) for n in selected)))
    return placements


def _proportional_point(placement: Placement, node_map: Mapping[str, NodeSpec]) -> Tuple[float, float]:
    points = {
        (round(node_map[n].cpu_cores / node_map[n].gpus, 9), round(node_map[n].ram_gb / node_map[n].gpus, 9))
        for n, _ in placement.node_gpus
    }
    if len(points) != 1:
        raise ValueError("multi-node placement has inconsistent proportional resource ratios")
    return next(iter(points))


class JointScheduler:
    def __init__(
        self,
        nodes: Sequence[NodeSpec],
        jobs: Sequence[JobSpec],
        oracle: JointProfileOracle,
        aux_cpu_points: Sequence[float],
        aux_ram_points: Sequence[float],
        allocation_bonus: float = 10.0,
        enforce_proportional_progress_floor: bool = True,
        verify_exact: bool = True,
        minimum_cpu_per_gpu: float = 0.0,
        minimum_ram_gb_per_gpu: float = 0.0,
    ) -> None:
        self.nodes = tuple(nodes)
        self.node_map = {n.node_id: n for n in nodes}
        if len(self.node_map) != len(self.nodes):
            raise ValueError("duplicate node IDs")
        self.jobs = tuple(jobs)
        self.job_map = {j.job_id: j for j in jobs}
        if len(self.job_map) != len(self.jobs):
            raise ValueError("duplicate job IDs")
        self.oracle = oracle
        self.aux_cpu_points = tuple(sorted(set(map(float, aux_cpu_points))))
        self.aux_ram_points = tuple(sorted(set(map(float, aux_ram_points))))
        self.allocation_bonus = float(allocation_bonus)
        self.enforce_floor = enforce_proportional_progress_floor
        self.verify_exact = bool(verify_exact)
        self.minimum_cpu_per_gpu = float(minimum_cpu_per_gpu)
        self.minimum_ram_gb_per_gpu = float(minimum_ram_gb_per_gpu)
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        for job in self.jobs:
            # The caller supplies only arrived jobs; absolute arrival timestamps
            # must survive here for FIFO (zeroing them changes Synergy ordering).
            if not math.isfinite(job.arrival_time):
                raise ValueError("non-finite arrival timestamp")
            for config in job.configs:
                if not config.legal:
                    continue
                capacity = sum(n.gpus for n in self.nodes if n.gpu_type == config.gpu_type)
                if config.gpus > capacity:
                    raise ValueError(f"{job.job_id}/{config.config_id} exceeds GPU-type capacity")
                if not enumerate_placements(config, self.nodes):
                    raise ValueError(f"{job.job_id}/{config.config_id} has no legal placement")

    def _node_usage(
        self, placement: Placement, cpu_per_gpu: float, ram_per_gpu: float
    ) -> Tuple[Tuple[str, float, float, float], ...]:
        return tuple(
            (node_id, float(gpus), gpus * cpu_per_gpu, gpus * ram_per_gpu)
            for node_id, gpus in placement.node_gpus
        )

    def allocation_floor(self, proportional_point: Tuple[float, float]) -> Tuple[float, float]:
        """Return the legal GPU-proportional execution point.

        The optional lower bounds are disabled by default, preserving all
        historical experiments.  The CPU-bottleneck smoke enables the
        official Synergy profile grid minima (1 core and 20.83 GiB per GPU),
        so an 8-GPU configuration cannot silently execute with less than the
        smallest measured/profiled resource point.
        """
        return (
            max(float(proportional_point[0]), self.minimum_cpu_per_gpu),
            max(float(proportional_point[1]), self.minimum_ram_gb_per_gpu),
        )

    def _fits_column(self, usage: Sequence[Tuple[str, float, float, float]]) -> bool:
        for node_id, gpu, cpu, ram in usage:
            node = self.node_map[node_id]
            if gpu > node.gpus + 1e-9 or cpu > node.cpu_cores + 1e-9 or ram > node.ram_gb + 1e-9:
                return False
        return True

    @staticmethod
    def lock_active(job: JobSpec) -> bool:
        return bool(job.checkpoint_locked and job.current_config_id is not None)

    def refresh_column(self, old: JointColumn) -> JointColumn:
        """Reuse geometry/quotas ONLY; rebuild against the current round state."""
        job = self.job_map[old.job_id]
        expected_usage = self._node_usage(old.placement, old.query.cpu_per_gpu, old.query.ram_gb_per_gpu)
        if old.node_usage != expected_usage:
            raise ValueError("previous column has inconsistent geometry/quota evidence")
        config = next((c for c in job.configs if c.config_id == old.config.config_id and c.legal), None)
        if config is None or old.placement not in enumerate_placements(config, self.nodes):
            raise ValueError("previous configuration/placement is no longer legal")
        prop = _proportional_point(old.placement, self.node_map)
        query = self.oracle.query(job, config, old.query.cpu_per_gpu, old.query.ram_gb_per_gpu, prop)
        return replace(old, config=config, query=query, node_usage=expected_usage,
                       query_round=job.query_round, scheduling_utility=query.effective_goodput + self.allocation_bonus)

    def locked_column(self, job: JobSpec) -> JointColumn:
        """A lock is one exact allocation, not a filter permitting unselection."""
        if not job.current_placement or job.current_cpu_per_gpu is None or job.current_ram_gb_per_gpu is None:
            raise RuntimeError(f"INFEASIBLE_CHECKPOINT_LOCK missing full allocation: {job.job_id}")
        cfg = next((c for c in job.configs if c.config_id == job.current_config_id and c.legal), None)
        placement = Placement(job.current_placement)
        if cfg is None or placement not in enumerate_placements(cfg, self.nodes):
            raise RuntimeError(f"INFEASIBLE_CHECKPOINT_LOCK illegal geometry: {job.job_id}")
        prop = _proportional_point(placement, self.node_map)
        query = self.oracle.query(job, cfg, job.current_cpu_per_gpu, job.current_ram_gb_per_gpu, prop)
        return JointColumn(f"{job.job_id}|LOCK|{cfg.config_id}", job.job_id, cfg, placement, query,
                           self._node_usage(placement, query.cpu_per_gpu, query.ram_gb_per_gpu),
                           query.effective_goodput + self.allocation_bonus, prop[0], prop[1], True, True,
                           query_round=job.query_round, candidate_kind="locked")

    def locked_columns(self) -> Dict[str, JointColumn]:
        selected = {j.job_id: self.locked_column(j) for j in self.jobs if self.lock_active(j)}
        audit = self.audit(list(selected.values()))
        if not audit["valid"]:
            raise RuntimeError(f"INFEASIBLE_CHECKPOINT_LOCK: {audit}")
        return selected

    def build_joint_columns(
        self, allowed_configs: Mapping[str, Iterable[str]] | None = None
    ) -> List[JointColumn]:
        columns: List[JointColumn] = []
        for job in self.jobs:
            if self.lock_active(job):
                columns.append(self.locked_column(job))
                continue
            lock_active = (
                job.checkpoint_locked
                and job.current_config_id is not None
                and bool(job.current_nodes)
            )
            allowed = None if allowed_configs is None else set(allowed_configs[job.job_id])
            for config in job.configs:
                if not config.legal or (allowed is not None and config.config_id not in allowed):
                    continue
                if lock_active and config.config_id != job.current_config_id:
                    continue
                for placement in enumerate_placements(config, self.nodes):
                    if lock_active and tuple(sorted(placement.node_ids)) != tuple(sorted(job.current_nodes)):
                        continue
                    prop = _proportional_point(placement, self.node_map)
                    execution_floor = self.allocation_floor(prop)
                    prop_query = self.oracle.query(
                        job, config, execution_floor[0], execution_floor[1], prop
                    )
                    cpu_points = sorted(
                        point for point in set(self.aux_cpu_points + (execution_floor[0],))
                        if point + 1e-9 >= self.minimum_cpu_per_gpu
                    )
                    ram_points = sorted(
                        point for point in set(self.aux_ram_points + (execution_floor[1],))
                        if point + 1e-9 >= self.minimum_ram_gb_per_gpu
                    )
                    for cpu, ram in itertools.product(cpu_points, ram_points):
                        usage = self._node_usage(placement, cpu, ram)
                        if not self._fits_column(usage):
                            continue
                        query = self.oracle.query(job, config, cpu, ram, prop)
                        if self.enforce_floor and query.effective_goodput + 1e-9 < prop_query.effective_goodput:
                            continue
                        current_config = job.current_config_id
                        current_nodes = tuple(sorted(job.current_nodes))
                        preserves_config = current_config is not None and current_config == config.config_id
                        preserves_nodes = bool(current_nodes) and current_nodes == tuple(sorted(placement.node_ids))
                        restart_factor = 1.0
                        if current_config is not None and not preserves_config:
                            # Tiny prototype uses a declared, state-only factor; scenarios start fresh.
                            restart_factor = 0.9
                        effective = query.effective_goodput * restart_factor
                        placement_key = "+".join(f"{n}:{g}" for n, g in placement.node_gpus)
                        column_id = f"{job.job_id}|{config.config_id}|{cpu:g}|{ram:g}|{placement_key}"
                        columns.append(
                            JointColumn(
                                column_id=column_id,
                                job_id=job.job_id,
                                config=config,
                                placement=placement,
                                query=query,
                                node_usage=usage,
                                scheduling_utility=effective + self.allocation_bonus,
                                proportional_cpu_per_gpu=prop[0],
                                proportional_ram_per_gpu=prop[1],
                                preserves_sia_config=preserves_config,
                                preserves_node_ids=preserves_nodes,
                                query_round=job.query_round,
                                candidate_kind="joint_aux",
                            )
                        )
        return self._pareto_prune(columns)

    def build_proportional_columns(self) -> List[JointColumn]:
        """Enumerate GPU/placement choices at one deterministic auxiliary point.

        Each legal GPU configuration and placement receives exactly the same
        proportional execution rule used by the unified B1 path: the node's
        CPU/GPU and RAM/GPU ratios, raised only to the registered minimum
        profile-grid point.  No auxiliary-resource candidate grid is exposed
        to the optimizer.  Checkpoint locks constrain both configuration and
        placement exactly as in ``build_joint_columns``.
        """
        columns: List[JointColumn] = []
        for job in self.jobs:
            if self.lock_active(job):
                columns.append(self.locked_column(job))
                continue
            lock_active = (
                job.checkpoint_locked
                and job.current_config_id is not None
                and bool(job.current_nodes)
            )
            for config in job.configs:
                if not config.legal:
                    continue
                if lock_active and config.config_id != job.current_config_id:
                    continue
                for placement in enumerate_placements(config, self.nodes):
                    if lock_active and tuple(sorted(placement.node_ids)) != tuple(sorted(job.current_nodes)):
                        continue
                    proportional = _proportional_point(placement, self.node_map)
                    execution_point = self.allocation_floor(proportional)
                    usage = self._node_usage(
                        placement, execution_point[0], execution_point[1]
                    )
                    if not self._fits_column(usage):
                        continue
                    query = self.oracle.query(
                        job,
                        config,
                        execution_point[0],
                        execution_point[1],
                        proportional,
                    )
                    placement_key = "+".join(
                        f"{node_id}:{gpus}" for node_id, gpus in placement.node_gpus
                    )
                    columns.append(
                        JointColumn(
                            column_id=(
                                f"{job.job_id}|{config.config_id}|B3_PROP|{placement_key}"
                            ),
                            job_id=job.job_id,
                            config=config,
                            placement=placement,
                            query=query,
                            node_usage=usage,
                            scheduling_utility=config.sia_goodput + self.allocation_bonus,
                            proportional_cpu_per_gpu=execution_point[0],
                            proportional_ram_per_gpu=execution_point[1],
                            preserves_sia_config=config.config_id == job.current_config_id,
                            preserves_node_ids=(
                                tuple(sorted(placement.node_ids))
                                == tuple(sorted(job.current_nodes))
                            ),
                            query_round=job.query_round,
                            candidate_kind="proportional",
                        )
                    )
        return sorted(columns, key=lambda column: column.column_id)

    @staticmethod
    def _pareto_prune(columns: Sequence[JointColumn]) -> List[JointColumn]:
        groups: Dict[Tuple[str, str, Tuple[Tuple[str, int], ...]], List[JointColumn]] = {}
        for col in columns:
            groups.setdefault((col.job_id, col.config.config_id, col.placement.node_gpus), []).append(col)
        kept: List[JointColumn] = []
        for group in groups.values():
            for candidate in group:
                dominated = any(
                    other.column_id != candidate.column_id
                    and other.query.cpu_per_gpu <= candidate.query.cpu_per_gpu + 1e-9
                    and other.query.ram_gb_per_gpu <= candidate.query.ram_gb_per_gpu + 1e-9
                    and other.query.effective_goodput >= candidate.query.effective_goodput - 1e-9
                    and (
                        other.query.cpu_per_gpu < candidate.query.cpu_per_gpu - 1e-9
                        or other.query.ram_gb_per_gpu < candidate.query.ram_gb_per_gpu - 1e-9
                        or other.query.effective_goodput > candidate.query.effective_goodput + 1e-9
                    )
                    for other in group
                )
                if not dominated:
                    kept.append(candidate)
        return sorted(kept, key=lambda c: c.column_id)

    def build_b1_columns(self) -> List[JointColumn]:
        """GPU-only objective with proportional auxiliary allocation for execution."""
        columns: List[JointColumn] = []
        for job in self.jobs:
            if self.lock_active(job):
                columns.append(self.locked_column(job))
                continue
            lock_active = (
                job.checkpoint_locked
                and job.current_config_id is not None
                and bool(job.current_nodes)
            )
            for config in job.configs:
                if not config.legal:
                    continue
                if lock_active and config.config_id != job.current_config_id:
                    continue
                for placement in enumerate_placements(config, self.nodes):
                    prop = _proportional_point(placement, self.node_map)
                    execution_point = self.allocation_floor(prop)
                    query = self.oracle.query(
                        job, config, execution_point[0], execution_point[1], prop
                    )
                    usage = self._node_usage(placement, execution_point[0], execution_point[1])
                    if not self._fits_column(usage):
                        continue
                    placement_rank = sum(list(self.node_map).index(n) for n in placement.node_ids)
                    tie_break = -1e-7 * placement_rank
                    column_id = f"{job.job_id}|{config.config_id}|B1|" + "+".join(placement.node_ids)
                    columns.append(
                        JointColumn(
                            column_id=column_id,
                            job_id=job.job_id,
                            config=config,
                            placement=placement,
                            query=query,
                            node_usage=usage,
                            scheduling_utility=config.sia_goodput + self.allocation_bonus + tie_break,
                            proportional_cpu_per_gpu=execution_point[0],
                            proportional_ram_per_gpu=execution_point[1],
                            preserves_sia_config=config.config_id == job.current_config_id,
                            preserves_node_ids=tuple(sorted(placement.node_ids)) == tuple(sorted(job.current_nodes)),
                            query_round=job.query_round,
                            candidate_kind="proportional",
                        )
                    )
        return sorted(columns, key=lambda c: c.column_id)

    def _constraint_matrix(
        self, columns: Sequence[JointColumn], dimensions: Sequence[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        rows: List[List[float]] = []
        upper: List[float] = []
        for job in self.jobs:
            rows.append([1.0 if col.job_id == job.job_id else 0.0 for col in columns])
            upper.append(1.0)
        for node in self.nodes:
            for dim, cap in (("gpu", node.gpus), ("cpu", node.cpu_cores), ("ram", node.ram_gb)):
                if dim not in dimensions:
                    continue
                index = {"gpu": 0, "cpu": 1, "ram": 2}[dim]
                rows.append([col.usage_for(node.node_id)[index] for col in columns])
                upper.append(float(cap))
        return np.asarray(rows, dtype=float), np.asarray(upper, dtype=float)

    def _failure_audit(
        self,
        failure_reason: str,
        columns: Sequence[JointColumn],
        dimensions: Sequence[str],
        solver_result=None,
    ) -> Dict[str, object]:
        """Return a non-committable but fully diagnostic solver audit."""
        counts = {job.job_id: 0 for job in self.jobs}
        for column in columns:
            counts[column.job_id] += 1
        solver = None
        if solver_result is not None:
            solver = {
                "status": getattr(solver_result, "status", None),
                "success": bool(getattr(solver_result, "success", False)),
                "message": str(getattr(solver_result, "message", "")),
                "fun": getattr(solver_result, "fun", None),
                "mip_gap": getattr(solver_result, "mip_gap", None),
                "mip_node_count": getattr(solver_result, "mip_node_count", None),
            }
        return {
            "valid": False,
            "violations": [failure_reason],
            "failure_reason": failure_reason,
            "candidate_column_count": len(columns),
            "candidate_counts_by_job": counts,
            "jobs_without_candidates": sorted(
                job_id for job_id, count in counts.items() if count == 0
            ),
            "active_jobs": [job.job_id for job in self.jobs],
            "current_allocations": {
                job.job_id: {
                    "config_id": job.current_config_id,
                    "nodes": list(job.current_nodes),
                }
                for job in self.jobs
            },
            "checkpoint_locked_jobs": {
                job.job_id: {
                    "config_id": job.current_config_id,
                    "nodes": list(job.current_nodes),
                }
                for job in self.jobs if job.checkpoint_locked
            },
            "dimensions": list(dimensions),
            "solver": solver,
        }

    def solve(
        self, policy: str, columns: Sequence[JointColumn], dimensions: Sequence[str]
    ) -> PolicyResult:
        start = time.perf_counter()
        if not columns:
            audit = self._failure_audit("NO_CANDIDATE_COLUMNS", columns, dimensions)
            return PolicyResult(
                policy, "NO_COLUMNS", {}, [j.job_id for j in self.jobs], 0, 0, 0,
                audit=audit,
            )
        matrix, upper = self._constraint_matrix(columns, dimensions)
        lower = np.full(len(upper), -np.inf)
        for idx, job in enumerate(self.jobs):
            if self.lock_active(job):
                lower[idx] = 1.0  # sum of the exact locked job's columns == 1
        objective = -np.asarray([col.scheduling_utility for col in columns], dtype=float)
        result = milp(
            c=objective,
            integrality=np.ones(len(columns)),
            bounds=Bounds(np.zeros(len(columns)), np.ones(len(columns))),
            constraints=LinearConstraint(matrix, lower, upper),
            options={"time_limit": 30.0, "mip_rel_gap": 0.0},
        )
        elapsed = time.perf_counter() - start
        if result.x is None:
            audit = self._failure_audit(
                "MIP_NO_USABLE_SOLUTION", columns, dimensions, solver_result=result
            )
            return PolicyResult(
                policy, f"MIP_NO_SOLUTION_STATUS_{result.status}", {},
                [j.job_id for j in self.jobs], 0, 0, elapsed, audit=audit,
            )
        selected_cols = [columns[i] for i, val in enumerate(result.x) if val > 0.5]
        selected = {col.job_id: col for col in selected_cols}
        audit = self.audit(selected_cols)
        if not audit["valid"]:
            raise RuntimeError(f"refusing partial/invalid solution: {audit}")
        scheduling_objective = sum(col.scheduling_utility for col in selected_cols)
        actual_goodput = sum(col.query.effective_goodput for col in selected_cols)
        unallocated = sorted(set(self.job_map) - set(selected))
        exact_obj = None
        exact_match = None
        if self.verify_exact:
            exact_obj, _ = self.exhaustive_optimum(columns, dimensions)
            exact_match = math.isclose(scheduling_objective, exact_obj, rel_tol=1e-8, abs_tol=1e-8)
        return PolicyResult(
            policy=policy,
            status="OPTIMAL" if result.success else f"INCUMBENT_{result.message}",
            selected=selected,
            unallocated_jobs=unallocated,
            scheduling_objective=scheduling_objective,
            actual_total_goodput=actual_goodput,
            solve_seconds=elapsed,
            exact_objective=exact_obj,
            exact_match=exact_match,
            audit=audit,
        )

    def exhaustive_optimum(
        self, columns: Sequence[JointColumn], dimensions: Sequence[str]
    ) -> Tuple[float, Tuple[str, ...]]:
        by_job = {job.job_id: [None] + [c for c in columns if c.job_id == job.job_id] for job in self.jobs}
        best_obj = -math.inf
        best_ids: Tuple[str, ...] = ()
        for choice in itertools.product(*(by_job[j.job_id] for j in self.jobs)):
            chosen = [c for c in choice if c is not None]
            audit = self.audit(chosen, dimensions=dimensions)
            if not audit["valid"]:
                continue
            obj = sum(c.scheduling_utility for c in chosen)
            ids = tuple(sorted(c.column_id for c in chosen))
            if obj > best_obj + 1e-9 or (math.isclose(obj, best_obj, abs_tol=1e-9) and ids < best_ids):
                best_obj, best_ids = obj, ids
        return best_obj, best_ids

    def audit(
        self, columns: Sequence[JointColumn], dimensions: Sequence[str] = ("gpu", "cpu", "ram"),
        require_locks: bool = True,
    ) -> Dict[str, object]:
        violations: List[str] = []
        job_counts: Dict[str, int] = {}
        node_totals = {n.node_id: [0.0, 0.0, 0.0] for n in self.nodes}
        for col in columns:
            job_counts[col.job_id] = job_counts.get(col.job_id, 0) + 1
            job = self.job_map.get(col.job_id)
            if job is None:
                violations.append(f"unknown job {col.job_id}")
                continue
            if col.config not in job.configs or not col.config.legal:
                violations.append(f"stale/illegal config {col.job_id}")
            if col.placement not in enumerate_placements(col.config, self.nodes):
                violations.append(f"illegal placement {col.job_id}")
            if col.query_round != job.query_round:
                violations.append(f"stale query round {col.job_id}")
            try:
                prop = _proportional_point(col.placement, self.node_map)
                expected = self.oracle.query(job, col.config, col.query.cpu_per_gpu, col.query.ram_gb_per_gpu, prop)
                if expected != col.query:
                    violations.append(f"stale/invalid oracle query {col.job_id}")
                if col.node_usage != self._node_usage(col.placement, col.query.cpu_per_gpu, col.query.ram_gb_per_gpu):
                    violations.append(f"quota/usage mismatch {col.job_id}")
                if col.query.cpu_per_gpu < self.minimum_cpu_per_gpu or col.query.ram_gb_per_gpu < self.minimum_ram_gb_per_gpu:
                    violations.append(f"quota below execution floor {col.job_id}")
                if not self.lock_active(job):
                    point = (col.query.cpu_per_gpu, col.query.ram_gb_per_gpu)
                    floor = self.allocation_floor(prop)
                    if col.candidate_kind == "proportional" and point != floor:
                        violations.append(f"non-proportional candidate {col.job_id}")
                    if col.candidate_kind == "joint_aux":
                        if point[0] not in self.aux_cpu_points + (floor[0],) or point[1] not in self.aux_ram_points + (floor[1],):
                            violations.append(f"quota outside current candidate grid {col.job_id}")
                        baseline = self.oracle.query(job, col.config, *floor, prop)
                        if self.enforce_floor and col.query.effective_goodput + 1e-9 < baseline.effective_goodput:
                            violations.append(f"current proportional progress floor {col.job_id}")
            except (ValueError, KeyError, TypeError):
                violations.append(f"unverifiable candidate {col.job_id}")
            if self.lock_active(job):
                if (col.config.config_id != job.current_config_id or col.placement.node_gpus != job.current_placement
                    or col.query.cpu_per_gpu != job.current_cpu_per_gpu or col.query.ram_gb_per_gpu != job.current_ram_gb_per_gpu):
                    violations.append(f"checkpoint allocation changed {col.job_id}")
            for node_id, gpu, cpu, ram in col.node_usage:
                if node_id not in node_totals or not all(math.isfinite(v) and v >= 0 for v in (gpu, cpu, ram)):
                    violations.append(f"invalid node usage {col.job_id}/{node_id}")
                    continue
                totals = node_totals[node_id]
                totals[0] += gpu
                totals[1] += cpu
                totals[2] += ram
        if require_locks:
            for job in self.jobs:
                if self.lock_active(job) and job_counts.get(job.job_id, 0) != 1:
                    violations.append(f"checkpoint job not selected exactly once {job.job_id}")
        for job_id, count in job_counts.items():
            if count > 1:
                violations.append(f"multiple columns for {job_id}")
        for node in self.nodes:
            gpu, cpu, ram = node_totals[node.node_id]
            if "gpu" in dimensions and gpu > node.gpus + 1e-8:
                violations.append(f"GPU capacity {node.node_id}: {gpu}>{node.gpus}")
            if "cpu" in dimensions and cpu > node.cpu_cores + 1e-8:
                violations.append(f"CPU capacity {node.node_id}: {cpu}>{node.cpu_cores}")
            if "ram" in dimensions and ram > node.ram_gb + 1e-8:
                violations.append(f"RAM capacity {node.node_id}: {ram}>{node.ram_gb}")
        return {
            "valid": not violations,
            "violations": violations,
            "node_totals": {
                nid: {"gpus": vals[0], "cpu": vals[1], "ram_gb": vals[2]}
                for nid, vals in node_totals.items()
            },
        }

    def run_b1(self) -> PolicyResult:
        return self.solve("B1_GPU_ONLY_UNIFIED_EXECUTION", self.build_b1_columns(), ("gpu",))

    def run_b4(self, b1: PolicyResult) -> PolicyResult:
        allowed = {
            job.job_id: ({b1.selected[job.job_id].config.config_id} if job.job_id in b1.selected else set())
            for job in self.jobs
        }
        columns = self.build_joint_columns(allowed)
        result = self.solve("B4_SERIAL_FIXED_GPU_AUX_PACKING", columns, ("gpu", "cpu", "ram"))
        missing_from_b1 = sorted(set(b1.selected) - set(result.selected))
        if missing_from_b1:
            # The B1 proportional map is a constructive, jointly feasible fallback.
            fallback_cols = [self.refresh_column(c) for c in b1.selected.values()]
            fallback_audit = self.audit(fallback_cols)
            if fallback_audit["valid"]:
                result = PolicyResult(
                    policy="B4_SERIAL_FIXED_GPU_PROPORTIONAL_FALLBACK",
                    status="FALLBACK_B1_PROPORTIONAL",
                    selected={c.job_id: c for c in fallback_cols},
                    unallocated_jobs=list(b1.unallocated_jobs),
                    scheduling_objective=sum(c.query.effective_goodput + self.allocation_bonus for c in fallback_cols),
                    actual_total_goodput=sum(c.query.effective_goodput for c in fallback_cols),
                    solve_seconds=result.solve_seconds,
                    exact_objective=result.exact_objective,
                    exact_match=None,
                    audit=fallback_audit,
                )
                result.audit["fallback_reason"] = {
                    job_id: "FIXED_GPU_JOINT_PACKING_DROPPED_JOB" for job_id in missing_from_b1
                }
            else:
                # Under an explicit hard profile-grid minimum the Sia GPU
                # intent itself may be infeasible.  Falling back to it would
                # overcommit CPU/RAM, so retain the all-or-nothing feasible
                # subset returned by the optimizer and record the refusal.
                result.audit["fallback_reason"] = {
                    job_id: "PROPORTIONAL_INTENT_INFEASIBLE_NO_PARTIAL_COMMIT"
                    for job_id in missing_from_b1
                }
                result.audit["rejected_fallback_audit"] = fallback_audit
        reasons = {}
        for job_id, col in result.selected.items():
            old = b1.selected[job_id]
            if col.placement.node_gpus == old.placement.node_gpus:
                reasons[job_id] = "KEEP_B1_PLACEMENT"
            elif col.config.config_id == old.config.config_id:
                reasons[job_id] = "REPACK_FIXED_GPU_CONFIG"
            else:
                raise RuntimeError("B4 illegally changed GPU config")
        result.audit["repair_rule"] = reasons
        result.audit["fixed_gpu_configs"] = {j: c.config.config_id for j, c in b1.selected.items()}
        result.audit["proportional_fallback_constructively_valid"] = self.audit(list(b1.selected.values()))["valid"]
        return result

    def run_b3(self) -> PolicyResult:
        return self.solve("B3_FINITE_COLUMN_JOINT_MIP", self.build_joint_columns(), ("gpu", "cpu", "ram"))
