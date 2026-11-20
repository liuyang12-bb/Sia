"""Minimal end-to-end integration of the joint prototype with Sia.

The original :class:`SiaPolicy` and :class:`Cluster` remain untouched.  B1 and
B4 invoke the original Sia MIP every round.  B3 enumerates candidates from the
same Sia configuration tables and calls ``_compute_goodputs_v2`` directly,
then performs one finite-column GPU/CPU/RAM/placement optimization.
"""

from __future__ import annotations

import collections
import math
import time
from dataclasses import replace
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

import simulator_config as sim_config
from cluster import Cluster
from sia import SiaPolicy, ZERO_ALLOC_GAIN, get_restart_penalty

from .joint_profile import GpuConfig, JobSpec, JointProfileOracle, NodeSpec
from .joint_scheduler import (
    JointColumn,
    JointScheduler,
    Placement,
    PolicyResult,
    _proportional_point,
)
from .synergy_adapter import SynergyTuneAdapter


def _config_id(cluster: str, num_nodes: int, num_gpus: int) -> str:
    return f"{cluster}:{int(num_nodes)}:{int(num_gpus)}"


class JointSiaPolicy(SiaPolicy):
    """Sia-compatible policy implementing B1, B4, or B3.

    B1/B4/B5 first execute ``SiaPolicy.optimize``.  SY uses a fixed,
    pre-registered GPU mapping.  B3 does not call Sia's optimizer; it reuses
    Sia's candidate and Goodput paths before one joint MIP.
    """

    MODES = {"b1", "b4", "b3", "sy", "b5"}

    def __init__(
        self,
        *,
        mode: str,
        oracle: JointProfileOracle,
        node_capacities: Mapping[str, Mapping[str, float]],
        template_map: Mapping[str, str],
        aux_cpu_points: Sequence[float],
        aux_ram_points: Sequence[float],
        synergy_gpu_type_map: Mapping[str, str] | None = None,
        synergy_gpu_demand_map: Mapping[str, int] | None = None,
        minimum_cpu_per_gpu: float = 0.0,
        minimum_ram_gb_per_gpu: float = 0.0,
        **sia_kwargs,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unsupported joint mode: {mode}")
        super().__init__(**sia_kwargs)
        self.mode = mode
        self.oracle = oracle
        self.node_capacities = {k: dict(v) for k, v in node_capacities.items()}
        self.template_map = dict(template_map)
        self.aux_cpu_points = tuple(map(float, aux_cpu_points))
        self.aux_ram_points = tuple(map(float, aux_ram_points))
        self.synergy_gpu_type_map = dict(synergy_gpu_type_map or {})
        self.synergy_gpu_demand_map = {
            str(job_id): int(gpus) for job_id, gpus in (synergy_gpu_demand_map or {}).items()
        }
        self.minimum_cpu_per_gpu = float(minimum_cpu_per_gpu)
        self.minimum_ram_gb_per_gpu = float(minimum_ram_gb_per_gpu)
        self.last_decision: Dict[str, object] = {}
        self.last_failure: Dict[str, object] = {}
        self.current_resources: Dict[str, Dict[str, object]] = {}
        self._last_committed_columns: Dict[str, JointColumn] = {}
        self.round_index = 0

    def _nodes(self, nodes) -> Tuple[NodeSpec, ...]:
        result = []
        for cluster in self.cluster_ordering:
            if cluster not in self.node_capacities:
                raise ValueError(f"missing synthetic capacity for {cluster}")
            capacity = self.node_capacities[cluster]
            for node_idx, node in nodes[cluster].items():
                node_capacity = dict(capacity)
                node_capacity.update(capacity.get("nodes", {}).get(str(node_idx), {}))
                result.append(
                    NodeSpec(
                        node_id=f"{cluster}:{int(node_idx)}",
                        gpu_type=cluster,
                        gpus=int(node.resources["nvidia.com/gpu"]),
                        cpu_cores=float(node_capacity["cpu_cores"]),
                        ram_gb=float(node_capacity["ram_gb"]),
                    )
                )
        return tuple(result)

    @staticmethod
    def _allocation_state(allocation) -> Tuple[str | None, str | None, Tuple[str, ...]]:
        if not allocation or allocation[0] is None or not allocation[1]:
            return None, None, ()
        cluster, gpu_ids = allocation
        node_ids = tuple(sorted(f"{cluster}:{int(idx)}" for idx in set(gpu_ids)))
        config_id = _config_id(cluster, len(node_ids), len(gpu_ids))
        return cluster, config_id, node_ids

    def _raw_candidates(
        self, job_info, max_gpu_override: Mapping[str, int] | None = None
    ) -> Tuple[Tuple[GpuConfig, ...], Dict[str, Dict[str, object]]]:
        provisional = []
        provenance: Dict[str, Dict[str, object]] = {}
        any_direct = any(fn is not None for fn in job_info.speedup_fn.values())
        for cluster in self.cluster_ordering:
            scale_unit = int(job_info.scale_unit[cluster])
            min_replicas = max(1, int(job_info.min_replicas[cluster]))
            max_replicas = max(1, int(job_info.max_replicas[cluster]))
            if max_gpu_override and cluster in max_gpu_override:
                max_replicas = max(
                    max_replicas,
                    int(math.ceil(max_gpu_override[cluster] / scale_unit)),
                )
            if getattr(job_info, "category", None) == "rigid":
                min_replicas = max_replicas
            alloc_nodes, alloc_gpus = self.configs[cluster]
            valid = (alloc_gpus <= max_replicas * scale_unit) & (alloc_gpus >= min_replicas * scale_unit)
            if scale_unit > 1:
                valid &= alloc_gpus % scale_unit == 0
            nnodes = alloc_nodes[valid].astype(np.uint32)
            ngpus = alloc_gpus[valid].astype(np.uint32)
            if not len(nnodes):
                continue
            speedup_fn = job_info.speedup_fn.get(cluster)
            if speedup_fn is not None:
                source = "direct_profile"
                values = self._compute_goodputs_v2(job_info, cluster, nnodes, ngpus)
            elif self.project_throughputs and any_direct:
                source = "projected_profile"
                values = self._compute_goodputs_v2(job_info, cluster, nnodes, ngpus)
            else:
                source = "unprofiled_sia_fallback"
                fallback_gpus = max_replicas * scale_unit if getattr(job_info, "category", None) == "rigid" else scale_unit
                values = np.where(ngpus == fallback_gpus, float(fallback_gpus), 0.0)
            values = np.broadcast_to(np.asarray(values, dtype=float), nnodes.shape)
            for num_nodes, num_gpus, raw in zip(nnodes, ngpus, values):
                item_source = source
                if not np.isfinite(raw) or raw <= ZERO_ALLOC_GAIN:
                    # SY's fixed GPU demand must not change merely because a
                    # new Sia Job has not profiled the frozen configuration
                    # yet.  The placeholder is used only for SY bookkeeping;
                    # GPU choice is fixed and actual progress still comes from
                    # Job.step plus the shared response oracle.
                    if (
                        self.mode == "sy"
                        and max_gpu_override
                        and cluster in max_gpu_override
                        and int(num_gpus) <= int(max_gpu_override[cluster])
                    ):
                        raw = float(num_gpus)
                        item_source = "sy_fixed_demand_unprofiled_placeholder"
                    else:
                        continue
                cid = _config_id(cluster, int(num_nodes), int(num_gpus))
                provisional.append((cid, cluster, int(num_nodes), int(num_gpus), float(raw), scale_unit))
                provenance[cid] = {
                    "goodput_source": item_source,
                    "pre_normalization_goodput": float(raw),
                    "scale_unit": scale_unit,
                }
        if not provisional:
            raise RuntimeError("Sia produced no positive legal candidate")

        min_item = min(provisional, key=lambda item: item[4])
        scale_factor = float(min_item[5]) / min_item[4]
        per_gpu_rates = [item[4] / item[3] for item in provisional]
        reference_rate = max(min(per_gpu_rates), 1e-12)
        configs = []
        for cid, cluster, num_nodes, num_gpus, raw, _ in provisional:
            normalized = raw * scale_factor
            per_gpu_ratio = raw / num_gpus / reference_rate
            cpu_scale = float(np.clip(per_gpu_ratio, 0.5, 4.0))
            memory_scale = float(np.clip(math.sqrt(per_gpu_ratio), 0.5, 2.0))
            configs.append(
                GpuConfig(
                    config_id=cid,
                    gpu_type=cluster,
                    gpus=num_gpus,
                    num_nodes=num_nodes,
                    sia_goodput=float(normalized),
                    cpu_demand_scale=cpu_scale,
                    memory_demand_scale=memory_scale,
                )
            )
            provenance[cid].update(
                {
                    "normalized_sia_goodput": float(normalized),
                    "normalization_scale": scale_factor,
                    "cpu_demand_scale": cpu_scale,
                    "memory_demand_scale": memory_scale,
                }
            )
        return tuple(configs), provenance

    def _job_specs(self, jobs, base_allocations):
        specs = []
        all_provenance = {}
        for job_id, job_info in jobs.items():
            override = None
            if self.mode == "sy":
                mapped_type = self.synergy_gpu_type_map.get(job_id)
                demand = self.synergy_gpu_demand_map.get(job_id)
                if mapped_type is None or demand is None:
                    raise ValueError(f"SY requires frozen GPU type and demand for {job_id}")
                override = {mapped_type: demand}
            configs, provenance = self._raw_candidates(job_info, override)
            _, current_config, current_nodes = self._allocation_state(base_allocations.get(job_id))
            checkpoint_locked = bool(
                getattr(job_info, "scaling_underway", False)
                and current_config is not None
                and current_nodes
            )
            template = self.template_map.get(job_id)
            if template is None:
                raise ValueError(f"missing pre-registered response template for {job_id}")
            specs.append(
                JobSpec(
                    job_id=job_id,
                    template_file=template,
                    configs=configs,
                    arrival_time=0.0,
                    observed_progress=float(getattr(job_info, "attained_service", 0.0)),
                    checkpoint_locked=checkpoint_locked,
                    current_config_id=current_config,
                    current_nodes=current_nodes,
                    mapping_mode="fixed_hash",
                )
            )
            all_provenance[job_id] = provenance
        return tuple(specs), all_provenance

    def _scheduler(self, nodes, specs) -> JointScheduler:
        return JointScheduler(
            nodes=nodes,
            jobs=specs,
            oracle=self.oracle,
            aux_cpu_points=self.aux_cpu_points,
            aux_ram_points=self.aux_ram_points,
            allocation_bonus=10.0,
            enforce_proportional_progress_floor=True,
            verify_exact=False,
            minimum_cpu_per_gpu=self.minimum_cpu_per_gpu,
            minimum_ram_gb_per_gpu=self.minimum_ram_gb_per_gpu,
        )

    @staticmethod
    def _placement_from_allocation(cluster: str, gpu_ids) -> Placement:
        counts = collections.Counter(int(idx) for idx in gpu_ids)
        return Placement(tuple((f"{cluster}:{idx}", count) for idx, count in sorted(counts.items())))

    def _b1_result(self, allocations, specs, scheduler) -> PolicyResult:
        spec_map = {job.job_id: job for job in specs}
        selected = {}
        for job_id, allocation in allocations.items():
            if not allocation or allocation[0] is None or not allocation[1]:
                continue
            cluster, gpu_ids = allocation
            placement = self._placement_from_allocation(cluster, gpu_ids)
            cid = _config_id(cluster, len(placement.node_ids), len(gpu_ids))
            job = spec_map[job_id]
            config = next((cfg for cfg in job.configs if cfg.config_id == cid), None)
            if config is None:
                raise RuntimeError(f"Sia selected a candidate missing from integration enumeration: {job_id}/{cid}")
            prop = _proportional_point(placement, scheduler.node_map)
            execution_point = scheduler.allocation_floor(prop)
            query = self.oracle.query(
                job, config, execution_point[0], execution_point[1], prop
            )
            usage = scheduler._node_usage(
                placement, execution_point[0], execution_point[1]
            )
            selected[job_id] = JointColumn(
                column_id=f"{job_id}|{cid}|ORIGINAL_SIA",
                job_id=job_id,
                config=config,
                placement=placement,
                query=query,
                node_usage=usage,
                scheduling_utility=config.sia_goodput,
                proportional_cpu_per_gpu=execution_point[0],
                proportional_ram_per_gpu=execution_point[1],
                preserves_sia_config=cid == job.current_config_id,
                preserves_node_ids=tuple(sorted(placement.node_ids)) == tuple(sorted(job.current_nodes)),
            )
        audit = scheduler.audit(list(selected.values()))
        if not audit["valid"] and self.minimum_cpu_per_gpu <= 0 and self.minimum_ram_gb_per_gpu <= 0:
            raise RuntimeError(f"original Sia proportional allocation failed joint audit: {audit}")
        return PolicyResult(
            policy="B1_ORIGINAL_SIA_GPU_ONLY_UNIFIED_EXECUTION",
            status="ORIGINAL_SIA_OPTIMAL",
            selected=selected,
            unallocated_jobs=sorted(set(spec_map) - set(selected)),
            scheduling_objective=sum(col.config.sia_goodput for col in selected.values()),
            actual_total_goodput=sum(col.query.effective_goodput for col in selected.values()),
            solve_seconds=0.0,
            audit=audit,
        )

    def _sy_fixed_configs(self, jobs, specs):
        """Select fixed GPU type/demand without consulting CPU/RAM responses."""
        fixed = {}
        rule = {}
        for spec in specs:
            info = jobs[spec.job_id]
            mapped_type = self.synergy_gpu_type_map.get(spec.job_id)
            if mapped_type is None:
                raise ValueError(f"SY requires a pre-registered GPU type for {spec.job_id}")
            if mapped_type not in self.cluster_ordering:
                raise ValueError(f"invalid SY GPU type mapping {spec.job_id}->{mapped_type}")
            candidates = [cfg for cfg in spec.configs if cfg.gpu_type == mapped_type and cfg.legal]
            if spec.checkpoint_locked and spec.current_config_id is not None:
                candidates = [cfg for cfg in candidates if cfg.config_id == spec.current_config_id]
            if not candidates:
                raise RuntimeError(f"no legal fixed SY config for {spec.job_id}/{mapped_type}")
            requested = self.synergy_gpu_demand_map.get(spec.job_id)
            if requested is None or requested <= 0:
                raise ValueError(f"missing positive frozen SY GPU demand for {spec.job_id}")
            exact = [cfg for cfg in candidates if cfg.gpus == requested]
            if exact:
                chosen = min(exact, key=lambda cfg: (cfg.num_nodes, cfg.config_id))
                adjustment = "EXACT_WORKLOAD_DEMAND"
            else:
                not_above = [cfg for cfg in candidates if cfg.gpus <= requested]
                pool = not_above if not_above else candidates
                chosen = min(pool, key=lambda cfg: (-cfg.gpus, cfg.num_nodes, cfg.config_id))
                adjustment = "CLIP_TO_LARGEST_LEGAL_NOT_ABOVE_DEMAND" if not_above else "SMALLEST_LEGAL"
            fixed[spec.job_id] = chosen
            rule[spec.job_id] = {
                "gpu_type": mapped_type,
                "requested_fixed_gpus": requested,
                "selected_config": chosen.config_id,
                "adjustment": adjustment,
                "uses_cpu_ram_response": False,
            }
        return fixed, rule

    def _restart_factor(self, job_id: str, job_info, config_id: str, current_config: str | None) -> float:
        if current_config is None or current_config == config_id:
            return 1.0
        penalty = get_restart_penalty(job_id)
        if job_info.age < penalty:
            return 1.0
        if sim_config.sia_use_pollux_realloc_factor:
            lost = job_info.num_restarts * penalty
            return max(job_info.age - lost, 0.0) / (job_info.age + penalty)
        used = job_info.used_gpu_seconds
        wasted = job_info.wasted_gpu_seconds
        return max(1.0 + used, 0.0) / (1.0 + used + wasted + penalty)

    def _b3_result(self, jobs, specs, scheduler) -> PolicyResult:
        columns = scheduler.build_joint_columns()
        spec_map = {job.job_id: job for job in specs}
        transformed = []
        lambda_n = self.lambda_n if self.lambda_n else -1.0
        sign = -1.0 if self.p_fairness < 0 else 1.0
        for column in columns:
            job_info = jobs[column.job_id]
            current = spec_map[column.job_id].current_config_id
            factor = self._restart_factor(column.job_id, job_info, column.config.config_id, current)
            adjusted = max(column.query.effective_goodput * factor, ZERO_ALLOC_GAIN)
            powered = round(adjusted ** self.p_fairness, 2)
            powered = min(powered, sim_config.sia_goodput_clip_val)
            utility = sign * (powered - lambda_n)
            transformed.append(replace(column, scheduling_utility=float(utility)))
        return scheduler.solve("B3_SIA_GOODPUT_JOINT_MIP", transformed, ("gpu", "cpu", "ram"))

    def _b3_failure_fallback(self, failed, specs, scheduler) -> PolicyResult | None:
        """Preserve only a complete, currently legal previous allocation."""
        selected = {}
        missing_current = []
        lock_violations = []
        for spec in specs:
            previous = self._last_committed_columns.get(spec.job_id)
            if spec.current_config_id is None:
                continue
            if previous is None:
                missing_current.append(spec.job_id)
                continue
            same_config = previous.config.config_id == spec.current_config_id
            same_nodes = tuple(sorted(previous.placement.node_ids)) == tuple(sorted(spec.current_nodes))
            if not same_config or not same_nodes:
                missing_current.append(spec.job_id)
                if spec.checkpoint_locked:
                    lock_violations.append(spec.job_id)
                continue
            selected[spec.job_id] = previous

        original_failure_audit = dict(failed.audit)
        fallback_audit = scheduler.audit(list(selected.values()))
        fallback_audit.update({
            "fallback_reason": failed.audit.get("failure_reason", failed.status),
            "original_failure_status": failed.status,
            "original_failure_audit": original_failure_audit,
            "missing_current_allocations": missing_current,
            "checkpoint_lock_violations": lock_violations,
            "preserved_previous_jobs": sorted(selected),
        })
        safe = fallback_audit["valid"] and not missing_current and not lock_violations
        failed.audit.update({
            "previous_allocation_audit": {
                "valid": fallback_audit["valid"],
                "violations": fallback_audit["violations"],
                "node_totals": fallback_audit["node_totals"],
            },
            "safe_previous_allocation_available": bool(safe and selected),
            "missing_current_allocations": missing_current,
            "checkpoint_lock_violations": lock_violations,
            "preserved_previous_jobs": sorted(selected),
        })
        if safe and selected:
            return PolicyResult(
                policy="B3_PRESERVE_PREVIOUS_VALID_ALLOCATION",
                status="FALLBACK_PRESERVE_PREVIOUS_VALID_ALLOCATION",
                selected=selected,
                unallocated_jobs=sorted(set(scheduler.job_map) - set(selected)),
                scheduling_objective=sum(col.scheduling_utility for col in selected.values()),
                actual_total_goodput=sum(col.query.effective_goodput for col in selected.values()),
                solve_seconds=failed.solve_seconds,
                audit=fallback_audit,
            )

        no_current_allocations = all(spec.current_config_id is None for spec in specs)
        no_real_locks = not any(spec.checkpoint_locked for spec in specs)
        if (
            failed.status == "NO_COLUMNS"
            and no_current_allocations
            and no_real_locks
            and fallback_audit["valid"]
        ):
            fallback_audit["fallback_reason"] = "LEGAL_ALL_WAIT_NO_CANDIDATE_COLUMNS"
            return PolicyResult(
                policy="B3_LEGAL_ALL_WAIT",
                status="ALL_WAIT_NO_FEASIBLE_COLUMNS",
                selected={},
                unallocated_jobs=sorted(scheduler.job_map),
                scheduling_objective=0.0,
                actual_total_goodput=0.0,
                solve_seconds=failed.solve_seconds,
                audit=fallback_audit,
            )
        return None

    @staticmethod
    def _allocations_from_result(result: PolicyResult, all_job_ids) -> Dict[str, Tuple[object, tuple]]:
        allocations = {job_id: (None, ()) for job_id in all_job_ids}
        for job_id, column in result.selected.items():
            cluster = column.config.gpu_type
            gpu_ids = []
            for node_id, count in column.placement.node_gpus:
                node_cluster, local_id = node_id.rsplit(":", 1)
                if node_cluster != cluster:
                    raise RuntimeError("cross-GPU-type placement in one column")
                gpu_ids.extend([int(local_id)] * int(count))
            allocations[job_id] = (cluster, tuple(gpu_ids))
        return allocations

    def _update_b3_previous(self, jobs, result: PolicyResult) -> None:
        active_rigid = set()
        active_adaptive = set()
        for job_id, info in jobs.items():
            column = result.selected.get(job_id)
            if info.category == "rigid":
                active_rigid.add(job_id)
                self.prev_allocs_v2["rigid"][job_id] = (
                    self.cluster_ordering.index(column.config.gpu_type) if column else None
                )
            else:
                active_adaptive.add(job_id)
                self.prev_allocs_v2["adaptive"][job_id] = (
                    self.config_to_idx_map[(column.config.gpu_type, column.config.num_nodes, column.config.gpus)]
                    if column else None
                )
        self.prev_allocs_v2["rigid"] = {k: v for k, v in self.prev_allocs_v2["rigid"].items() if k in active_rigid}
        self.prev_allocs_v2["adaptive"] = {k: v for k, v in self.prev_allocs_v2["adaptive"].items() if k in active_adaptive}

    def _commit_metadata(self, result: PolicyResult, provenance, sia_fixed=None) -> None:
        resources = {}
        for job_id, column in result.selected.items():
            resources[job_id] = {
                "config_id": column.config.config_id,
                "gpu_type": column.config.gpu_type,
                "gpus": column.config.gpus,
                "num_nodes": column.config.num_nodes,
                "placement": dict(column.placement.node_gpus),
                "cpu_per_gpu": column.query.cpu_per_gpu,
                "ram_gb_per_gpu": column.query.ram_gb_per_gpu,
                "rho": column.query.rho,
                "raw_rho": column.query.raw_rho,
                "response_exponent": column.query.response_exponent,
                "response_raw": column.query.response_raw,
                "response_baseline": column.query.response_baseline,
                "normalized_sia_goodput": column.query.raw_sia_goodput,
                "predicted_effective_goodput": column.query.effective_goodput,
                "template": column.query.template_id,
                "template_source": column.query.template_source,
                "template_sha256": column.query.template_sha256,
                "source_coordinates": {
                    "requested_cpu": column.query.requested_source_cpu_coordinate,
                    "requested_ram_gb": column.query.requested_source_ram_coordinate,
                    "used_cpu": column.query.source_cpu_coordinate,
                    "used_ram_gb": column.query.source_ram_coordinate,
                    "clipped": column.query.source_coordinate_clipped,
                },
                "baseline_source_coordinates": {
                    "requested_cpu": column.query.requested_baseline_source_cpu_coordinate,
                    "requested_ram_gb": column.query.requested_baseline_source_ram_coordinate,
                    "used_cpu": column.query.baseline_source_cpu_coordinate,
                    "used_ram_gb": column.query.baseline_source_ram_coordinate,
                    "clipped": column.query.baseline_coordinate_clipped,
                },
                "goodput_provenance": provenance[job_id][column.config.config_id],
                "node_usage": [
                    {"node_id": n, "gpus": g, "cpu": c, "ram_gb": r}
                    for n, g, c, r in column.node_usage
                ],
            }
        self.current_resources = resources
        self._last_committed_columns = dict(result.selected)
        self.last_failure = {}
        self.last_decision = {
            "round": self.round_index,
            "mode": self.mode,
            "resource_model": "synthetic_cross_hardware_transfer",
            "result": result.to_dict(),
            "resources": resources,
            "sia_fixed_gpu_configs": sia_fixed or {},
            "candidate_summary": {
                job_id: {
                    "count": len(items),
                    "config_ids": sorted(items),
                    "goodput_sources": sorted({item["goodput_source"] for item in items.values()}),
                }
                for job_id, items in provenance.items()
            },
        }
        self.round_index += 1

    def optimize(self, jobs, nodes, base_allocations):
        node_specs = self._nodes(nodes)
        specs, provenance = self._job_specs(jobs, base_allocations)
        scheduler = self._scheduler(node_specs, specs)
        if self.mode in {"b1", "b4", "b5"}:
            original_allocations = super().optimize(jobs, nodes, base_allocations)
            b1_intent = self._b1_result(original_allocations, specs, scheduler)
            if self.mode == "b1":
                if b1_intent.audit.get("valid", False):
                    result = b1_intent
                    allocations = original_allocations
                else:
                    result = scheduler.solve(
                        "B1_SIA_GPU_INTENT_CPU_ADMISSION",
                        list(b1_intent.selected.values()),
                        ("gpu", "cpu", "ram"),
                    )
                    result.audit["original_sia_intent_audit"] = b1_intent.audit
                    result.audit["admission_rule"] = "WHOLE_JOB_ONLY_NO_GPU_CONFIG_CHANGE"
                    allocations = self._allocations_from_result(result, jobs)
                fixed = {
                    job_id: col.config.config_id for job_id, col in b1_intent.selected.items()
                }
            elif self.mode == "b4":
                result = scheduler.run_b4(b1_intent)
                allocations = self._allocations_from_result(result, jobs)
                fixed = {
                    job_id: col.config.config_id for job_id, col in b1_intent.selected.items()
                }
                for job_id, column in result.selected.items():
                    if column.config.config_id != fixed[job_id]:
                        raise RuntimeError("B4 changed the Sia GPU configuration")
            else:
                fixed_configs = {
                    job_id: column.config for job_id, column in b1_intent.selected.items()
                }
                result = SynergyTuneAdapter(scheduler).run(
                    "B5_SIA_GPU_THEN_SYNERGY_TUNE",
                    fixed_configs,
                    {
                        "source": "ORIGINAL_SIA_MIP_THIS_ROUND",
                        "cpu_ram_response_used_for_gpu_choice": False,
                    },
                )
                allocations = self._allocations_from_result(result, jobs)
                fixed = {
                    job_id: config.config_id for job_id, config in fixed_configs.items()
                }
                for job_id, column in result.selected.items():
                    if column.config.config_id != fixed[job_id]:
                        raise RuntimeError("B5 changed the Sia GPU configuration")
        elif self.mode == "sy":
            fixed_configs, gpu_rule = self._sy_fixed_configs(jobs, specs)
            result = SynergyTuneAdapter(scheduler).run(
                "SY_SYNERGY_TUNE_FIXED_GPU_ADAPTED", fixed_configs, gpu_rule
            )
            allocations = self._allocations_from_result(result, jobs)
            fixed = {job_id: config.config_id for job_id, config in fixed_configs.items()}
        else:
            result = self._b3_result(jobs, specs, scheduler)
            if not result.audit.get("valid", False):
                fallback = self._b3_failure_fallback(result, specs, scheduler)
                if fallback is None:
                    self.last_failure = {
                        "round": self.round_index,
                        "mode": self.mode,
                        "result": result.to_dict(),
                        "current_resources": self.current_resources,
                        "active_jobs": [spec.job_id for spec in specs],
                    }
                    raise RuntimeError(
                        f"refusing non-atomic joint allocation: {result.audit}"
                    )
                result = fallback
            allocations = self._allocations_from_result(result, jobs)
            self._update_b3_previous(jobs, result)
            fixed = {}
        if not result.audit.get("valid", False):
            self.last_failure = {
                "round": self.round_index,
                "mode": self.mode,
                "result": result.to_dict(),
                "current_resources": self.current_resources,
            }
            raise RuntimeError(f"refusing non-atomic joint allocation: {result.audit}")
        self._commit_metadata(result, provenance, fixed)
        return allocations


class JointCluster(Cluster):
    """Cluster wrapper which applies rho only to training progress."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.process_pool is not None:
            raise ValueError("joint integration requires sequential Job.step for auditable multipliers")
        self.joint_profile_bridges = {}
        for app_name in self.cluster_applications:
            if app_name in self.cluster_throughput_ratios:
                continue
            available = {
                cluster: self.cluster_applications[app_name][cluster]
                for cluster in self.clusters
                if cluster in self.cluster_applications[app_name]
            }
            if not available:
                continue
            throughputs = {}
            for cluster, app in available.items():
                rows = app.placements[app.placements.num_replicas == app.num_stages]
                throughputs[cluster] = float(rows.local_bsz.max() / rows.step_time.max())
            self.cluster_throughputs[app_name] = throughputs
            self.cluster_throughput_ratios[app_name] = {
                source: {
                    dest: throughputs[source] / throughputs[dest]
                    for dest in throughputs if dest != source
                }
                for source in throughputs
            }
            self.joint_profile_bridges[app_name] = {
                "reason": "official Cluster skips PMP throughput maps",
                "source": "minimum-stage rows in the existing Sia placement traces",
                "throughputs": throughputs,
            }
        self.joint_round_logs = []

    def optimize_policy(self):
        events = super().optimize_policy()
        resources = self.policy.current_resources
        for job in self.jobs:
            job.goodput_multiplier = float(resources.get(job.name, {}).get("rho", 1.0))
        return events

    def step(self, seconds=60):
        interval_start = self.current_time
        before = {
            job.name: {
                "progress": job.progress,
                "epoch": job.epoch,
                "job_time": job.current_time,
                "execution_time": job.execution_time,
                "rescale_time": job.rescale_time,
                "placement": list(job.placement),
                "cluster": job.current_cluster,
                "rho": job.goodput_multiplier,
                "resource_allocation": self.policy.current_resources.get(job.name),
            }
            for job in self.jobs
        }
        start = time.perf_counter()
        self.step_jobs_sequential(seconds)
        job_step_seconds = time.perf_counter() - start
        execution = []
        for job in self.jobs:
            prev = before[job.name]
            progress_delta = float(job.progress - prev["progress"])
            if progress_delta < -1e-9:
                raise RuntimeError(f"non-monotonic progress for {job.name}")
            training_seconds = float(job.execution_time - prev["execution_time"])
            rho = float(prev["rho"])
            effective_goodput = progress_delta / training_seconds if training_seconds > 0 else 0.0
            raw_goodput = effective_goodput / rho if training_seconds > 0 and rho > 0 else 0.0
            delay_consumed = min(float(prev["rescale_time"]), float(seconds)) if prev["placement"] else 0.0
            execution.append(
                {
                    "job_id": job.name,
                    "interval_start": interval_start,
                    "interval_end": self.current_time,
                    "submitted": job.submission_time <= interval_start,
                    "cluster": prev["cluster"],
                    "placement": prev["placement"],
                    "progress_before": prev["progress"],
                    "progress_after": job.progress,
                    "progress_delta": progress_delta,
                    "epoch_before": prev["epoch"],
                    "epoch_after": job.epoch,
                    "checkpoint_delay_consumed": delay_consumed,
                    "training_seconds": training_seconds,
                    "raw_goodput": raw_goodput,
                    "rho": rho,
                    "effective_goodput": effective_goodput,
                    "completion_time": job.completion_time,
                    "resource_allocation": prev["resource_allocation"],
                }
            )
        rescale_events, migrate_events = self.optimize_policy()
        round_record = {
            "interval_start": interval_start,
            "interval_end": self.current_time,
            "job_step_wall_seconds": job_step_seconds,
            "execution": execution,
            "decision_for_next_interval": self.policy.last_decision,
            "rescale_events": rescale_events,
            "migrate_events": migrate_events,
        }
        self.joint_round_logs.append(round_record)
        self.rescale_events.append(rescale_events)
        self.migrate_events.append(migrate_events)
        self.step_log()
        self.logs[-1]["joint_execution"] = execution
        self.logs[-1]["joint_decision"] = self.policy.last_decision

