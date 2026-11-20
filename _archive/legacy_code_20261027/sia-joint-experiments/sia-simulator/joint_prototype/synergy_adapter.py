"""Auditable adaptation of the official Synergy-TUNE allocation core.

This module intentionally does not reuse the B3 finite-column objective.  It
keeps a fixed GPU configuration, tries the official profile demand, falls back
to GPU-proportional fair share, and may atomically downgrade already placed
peers before retrying.  The source algorithm is in
``research_sources/Synergy/simulator/resources/cluster.py`` lines 575-745.

Adaptations are explicit: GPU type/demand are supplied by the caller, Sia's
integer placement candidates replace Synergy's homogeneous GPU objects, and a
failed final retry leaves the job waiting instead of calling ``sys.exit(1)``.
"""

from __future__ import annotations

import time
from typing import Dict, List, Mapping, Sequence, Tuple

from .joint_profile import GpuConfig, JobSpec
from .joint_scheduler import (
    JointColumn,
    JointScheduler,
    Placement,
    PolicyResult,
    _proportional_point,
    enumerate_placements,
)


class SynergyTuneAdapter:
    """Faithful small-scale port of demand -> fair-share -> peer downgrade."""

    SOURCE_PATH = "research_sources/Synergy/simulator/resources/cluster.py"
    SOURCE_LINES = {
        "entry": "575-586",
        "demand_then_fair": "589-623",
        "peer_downgrade": "624-666,698-745",
        "placement": "382-431,480-518",
    }

    def __init__(self, scheduler: JointScheduler) -> None:
        self.scheduler = scheduler
        self.specs = {job.job_id: job for job in scheduler.jobs}

    def _column(
        self,
        job: JobSpec,
        config: GpuConfig,
        placement: Placement,
        point: Tuple[float, float],
        stage: str,
    ) -> JointColumn:
        proportional = _proportional_point(placement, self.scheduler.node_map)
        query = self.scheduler.oracle.query(job, config, point[0], point[1], proportional)
        usage = self.scheduler._node_usage(placement, point[0], point[1])
        placement_key = "+".join(f"{node}:{gpus}" for node, gpus in placement.node_gpus)
        return JointColumn(
            column_id=(
                f"{job.job_id}|{config.config_id}|SYNERGY_TUNE_{stage}|"
                f"{point[0]:g}|{point[1]:g}|{placement_key}"
            ),
            job_id=job.job_id,
            config=config,
            placement=placement,
            query=query,
            node_usage=usage,
            # Not an optimization objective: retained only for common logging.
            scheduling_utility=query.rho,
            proportional_cpu_per_gpu=proportional[0],
            proportional_ram_per_gpu=proportional[1],
            preserves_sia_config=config.config_id == job.current_config_id,
            preserves_node_ids=(
                tuple(sorted(placement.node_ids)) == tuple(sorted(job.current_nodes))
            ),
        )

    def _fits_with(self, candidate: JointColumn, selected: Mapping[str, JointColumn]) -> bool:
        return self.scheduler.audit(list(selected.values()) + [candidate])["valid"]

    def _ordered_placements(self, config: GpuConfig) -> List[Placement]:
        # Official placement prefers consolidation and then the server(s) with
        # the fewest free GPUs.  With a fresh round and fixed Sia candidates,
        # lexicographic node order is the deterministic equivalent tie-break.
        return sorted(
            enumerate_placements(config, self.scheduler.nodes),
            key=lambda p: (len(p.node_ids), tuple(p.node_ids)),
        )

    def _try_place(
        self,
        job: JobSpec,
        config: GpuConfig,
        selected: Mapping[str, JointColumn],
        use_fair: bool,
    ) -> JointColumn | None:
        template = self.scheduler.oracle.template(job.template_file)
        for placement in self._ordered_placements(config):
            proportional = _proportional_point(placement, self.scheduler.node_map)
            fair = self.scheduler.allocation_floor(proportional)
            if use_fair:
                point = fair
                stage = "FAIR"
            else:
                point = (
                    max(template.cpu_demand * config.cpu_demand_scale, self.scheduler.minimum_cpu_per_gpu),
                    max(
                        template.memory_demand_gb * config.memory_demand_scale,
                        self.scheduler.minimum_ram_gb_per_gpu,
                    ),
                )
                stage = "DEMAND"
            candidate = self._column(job, config, placement, point, stage)
            if self._fits_with(candidate, selected):
                return candidate
        return None

    def _repack_all_fair(
        self,
        ordered_job_ids: Sequence[str],
        fixed_configs: Mapping[str, GpuConfig],
    ) -> Dict[str, JointColumn] | None:
        rebuilt: Dict[str, JointColumn] = {}
        for job_id in ordered_job_ids:
            column = self._try_place(
                self.specs[job_id], fixed_configs[job_id], rebuilt, use_fair=True
            )
            if column is None:
                return None
            rebuilt[job_id] = column
        return rebuilt

    def run(
        self,
        policy: str,
        fixed_configs: Mapping[str, GpuConfig],
        gpu_rule: Mapping[str, object],
    ) -> PolicyResult:
        start = time.perf_counter()
        selected: Dict[str, JointColumn] = {}
        trace: List[Dict[str, object]] = []
        order = sorted(
            fixed_configs,
            key=lambda job_id: (self.specs[job_id].arrival_time, job_id),
        )
        for job_id in order:
            job = self.specs[job_id]
            config = fixed_configs[job_id]
            column = self._try_place(job, config, selected, use_fair=False)
            if column is not None:
                selected[job_id] = column
                trace.append({"job_id": job_id, "action": "ALLOCATE_PROFILE_DEMAND"})
                continue

            column = self._try_place(job, config, selected, use_fair=True)
            if column is not None:
                selected[job_id] = column
                trace.append({"job_id": job_id, "action": "DOWNGRADE_CURRENT_TO_FAIR"})
                continue

            # Mirrors _tune's peer-adjust branch, but stages the complete map
            # and commits only if every old peer plus the current job fits.
            rebuilt = self._repack_all_fair(list(selected) + [job_id], fixed_configs)
            if rebuilt is not None:
                changed = sorted(
                    peer for peer in selected
                    if selected[peer].query.cpu_per_gpu != rebuilt[peer].query.cpu_per_gpu
                    or selected[peer].query.ram_gb_per_gpu != rebuilt[peer].query.ram_gb_per_gpu
                    or selected[peer].placement != rebuilt[peer].placement
                )
                selected = rebuilt
                trace.append(
                    {
                        "job_id": job_id,
                        "action": "ATOMIC_PEER_DOWNGRADE_TO_FAIR",
                        "downgraded_peers": changed,
                    }
                )
            else:
                trace.append(
                    {
                        "job_id": job_id,
                        "action": "WAIT_FINAL_RETRY_INFEASIBLE",
                        "reason": "GPU_CPU_RAM_OR_PLACEMENT_CAPACITY",
                    }
                )

        audit = self.scheduler.audit(list(selected.values()))
        if not audit["valid"]:
            raise RuntimeError(f"refusing non-atomic Synergy adaptation: {audit}")
        audit.update(
            {
                "official_core": "SYNERGY_TUNE",
                "official_source": self.SOURCE_PATH,
                "official_source_lines": self.SOURCE_LINES,
                "adaptations": [
                    "CALLER_FIXED_GPU_TYPE_AND_DEMAND",
                    "SIA_INTEGER_PLACEMENT_CANDIDATES",
                    "FINAL_FAILURE_MEANS_WAIT_NOT_PROCESS_EXIT",
                    "UNIFIED_ORACLE_CONTROLS_ACTUAL_PROGRESS",
                ],
                "gpu_rule": dict(gpu_rule),
                "tune_trace": trace,
                "fixed_gpu_configs": {
                    job_id: config.config_id for job_id, config in fixed_configs.items()
                },
            }
        )
        return PolicyResult(
            policy=policy,
            status="SYNERGY_TUNE_ADAPTED",
            selected=selected,
            unallocated_jobs=sorted(set(self.specs) - set(selected)),
            scheduling_objective=sum(column.query.rho for column in selected.values()),
            actual_total_goodput=sum(
                column.query.effective_goodput for column in selected.values()
            ),
            solve_seconds=time.perf_counter() - start,
            audit=audit,
        )

