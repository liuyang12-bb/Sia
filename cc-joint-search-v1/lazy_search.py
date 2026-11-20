"""Per-round candidate generation for the two-stage FO decision."""

from time import perf_counter

from joint_prototype.joint_scheduler import CandidateLimitReached, _proportional_point
from joint_prototype.neutral_liveness import geometry
from recovery_precheck import certify_job


class LazySearch:
    def __init__(self, policy, scheduler, points, score, potential_jobs):
        self.scheduler = scheduler
        self.points = points
        self.score = score
        self.proportional = []
        self.eligible_keys = set()
        self.mode = policy._search_mode
        self.started = policy._search_started
        self.deadline = self.started + policy._search_budget_seconds
        self.audit_reserve = min(0.05, policy._search_budget_seconds / 10)
        self.potential_jobs = potential_jobs
        self.recovery_eager = bool(potential_jobs)
        scope = "solo" if len(scheduler.job_map) == 1 else "multi"
        self.stats = {
            "mode": self.mode,
            "generation": "RECOVERY_EAGER" if self.recovery_eager else "LAZY",
            "recovery_guard_potential": self.recovery_eager,
            "recovery_scope": scope if self.recovery_eager else None,
            "recovery_certificate": ("not_applicable" if not self.recovery_eager else
                                     "uncertain" if policy._recovery_precheck else "disabled"),
            "recovery_precheck_seconds": 0.0,
            "phase1_columns": 0,
            "initial_columns": 0,
            "initial_generation_seconds": 0.0,
            "aux_columns": 0,
            "aux_generation_seconds": 0.0,
            "local": {"local_status": "not_requested"},
        }

    def precheck(self, policy, bounds):
        started = perf_counter()
        if self.potential_jobs and all(
                certify_job(policy, self.scheduler, name, bounds.get(name, {}))
                for name in self.potential_jobs):
            self.scheduler.recovery_proved_no_offer = True
            self.recovery_eager = False
            self.stats.update(generation="LAZY", recovery_certificate="proved_no_offer")
        self.stats["recovery_precheck_seconds"] = perf_counter() - started

    def _generate(self, point_selector, allowed_configs=None, deadline=None, max_columns=None):
        scheduler = self.scheduler
        original = scheduler.resource_point_provider
        scheduler.resource_point_provider = point_selector
        try:
            columns = scheduler.build_joint_columns(
                allowed_configs,
                only_jobs=None if allowed_configs is None else set(allowed_configs),
                deadline=deadline,
                max_columns=max_columns,
            )
        finally:
            scheduler.resource_point_provider = original
        return self.score(columns)

    def initial(self):
        started = perf_counter()
        incumbent = {}

        def points(s, job, config, placement):
            cpu, ram = self.points(s, job, config, placement)
            pc, pm = _proportional_point(placement, s.node_map)
            cp = tuple(x for x in cpu if abs(x - pc) < 1e-8)
            mp = tuple(x for x in ram if abs(x - pm) < 1e-8)
            if (config.config_id == job.current_config_id
                    and placement.node_gpus == job.current_placement):
                # Existing allocations are also compared to the chosen bundle.
                # Materialize them only if their exact point is still on-grid.
                if job.current_cpu_per_gpu in cpu:
                    cp += (job.current_cpu_per_gpu,)
                if job.current_ram_gb_per_gpu in ram:
                    mp += (job.current_ram_gb_per_gpu,)
                incumbent[job.job_id] = (job.current_cpu_per_gpu, job.current_ram_gb_per_gpu)
            return cp, mp

        columns, evidence = self._generate(points)
        keep = [
            i for i, col in enumerate(columns)
            if self.scheduler.lock_active(self.scheduler.job_map[col.job_id])
            or (abs(col.query.cpu_per_gpu - col.proportional_cpu_per_gpu) < 1e-8
                and abs(col.query.ram_gb_per_gpu - col.proportional_ram_per_gpu) < 1e-8)
            or (col.config.config_id == self.scheduler.job_map[col.job_id].current_config_id
                and col.placement.node_gpus == self.scheduler.job_map[col.job_id].current_placement
                and (col.query.cpu_per_gpu, col.query.ram_gb_per_gpu) == incumbent.get(col.job_id))
        ]
        self.stats["initial_columns"] = len(columns)
        self.stats["initial_generation_seconds"] = perf_counter() - started
        return [columns[i] for i in keep], [evidence[i] for i in keep]

    def bind_admission(self, admission, columns, productive, now):
        self.admission = admission
        self.productive = productive
        self.now = now
        self.proportional = [
            col for col in columns
            if admission.is_prop(col) or self.scheduler.lock_active(self.scheduler.job_map[col.job_id])
        ]
        self.stats["phase1_columns"] = len(self.proportional)

    def expand(self, allowed_configs, *, deadline=None, max_columns=None):
        started = perf_counter()
        try:
            columns, evidence = self._generate(self.points, allowed_configs, deadline, max_columns)
        except CandidateLimitReached:
            self.stats["aux_columns"] += max_columns
            raise
        except TimeoutError:
            self.stats["aux_generation_timeout"] = True
            return []
        finally:
            self.stats["aux_generation_seconds"] += perf_counter() - started
        self.stats["aux_columns"] += len(columns)
        self.scheduler.current_completion_scores.update({e["column_id"]: e for e in evidence})
        return columns

    def eta(self, column):
        return self.scheduler.current_completion_scores[column.column_id]["estimated_completion"]

    def legal_local(self, columns):
        grouped = {}
        for column in columns:
            grouped.setdefault(column.job_id, []).append(column)
        eligible = []
        ledger = self.admission._pending_tail.copy()
        try:
            for name, candidates in grouped.items():
                history = self.admission._lifetime_history.get(name)
                if history is None:
                    eligible.extend(candidates)
                else:
                    accepted, _ = self.admission.shared_eligible(
                        self.scheduler, self.scheduler.job_map[name], history,
                        candidates, self.productive, self.now,
                    )
                    eligible.extend(accepted)
            if self.admission._pending_tail != ledger:
                raise RuntimeError("RECOVERY_REQUIRES_EAGER_CANDIDATE_DOMAIN")
        finally:
            self.admission._pending_tail = ledger
        self.eligible_keys.update(geometry(column) for column in eligible)
        return sorted(eligible, key=geometry)
