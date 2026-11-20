"""Safety regressions for the conservative no-offer recovery certificate."""

import math
from dataclasses import replace
from time import perf_counter
from types import SimpleNamespace
from unittest import TestCase, main

from core80 import ROOT
from factor_cache import full_max_rho
from joint_prototype.bounded_correction import BoundedCorrectionMixin
from joint_prototype.clean_ablation import CleanAdmission
from joint_prototype.joint_profile import GpuConfig, JobSpec, JointProfileOracle, NodeSpec
from joint_prototype.joint_scheduler import Placement
from joint_prototype.neutral_liveness import FactorScheduler, geometry
from lazy_search import LazySearch
from recovery_precheck import potential_recovery_jobs, proves_no_recovery
from verify_search import compare_exact


class NoRecoveryCertificateTests(TestCase):
    def test_legal_witness_dominates_all_optimistic_alternatives(self):
        self.assertTrue(proves_no_recovery(100.0, [100.0, 101.0, 200.0]))

    def test_potential_fast_alternative_must_still_go_to_original_recovery(self):
        self.assertFalse(proves_no_recovery(100.0, [99.0, 120.0]))

    def test_proportional_alternative_is_not_a_bound_for_single_job(self):
        # Solo recovery evaluates auxiliary quotas at alternative geometries.
        # An alternative's proportional ETA could be slow while its best
        # possible auxiliary ETA is fast.
        witness = 100.0
        self.assertTrue(proves_no_recovery(witness, [120.0]))
        self.assertFalse(proves_no_recovery(witness, [80.0]))

    def test_preserves_the_recovery_comparison_tolerance(self):
        threshold = 100.0 - 1e-9
        self.assertTrue(proves_no_recovery(100.0, [threshold]))
        self.assertFalse(proves_no_recovery(100.0, [math.nextafter(threshold, -math.inf)]))

    def test_missing_or_invalid_evidence_fails_closed(self):
        for invalid_eta in (None, math.inf, math.nan, 0.0, -1.0):
            with self.subTest(invalid_eta=invalid_eta):
                self.assertFalse(proves_no_recovery(invalid_eta, [120.0]))
                self.assertFalse(proves_no_recovery(100.0, [120.0, invalid_eta]))

    def test_state_comparison_catches_a_lost_unconsumed_permit(self):
        selected = {"a": ("aws:1:1", (("aws:0", 1),), 3.0, 62.5, 1.0)}
        original = {
            "selected": selected,
            "recovery_offers": {"a": {"column_id": "a|dgx-ext:1:1", "actual_startup": 120.0}},
            "recovery_commits": [],
        }
        modified = {**original, "recovery_offers": {}}
        self.assertEqual(compare_exact(original, modified)["field"], "recovery_offers")

    def test_witness_uses_slowest_response_of_serialized_column_alias(self):
        slow_alias, fast_alias = 1.234561, 1.234562
        self.assertEqual(f"{slow_alias:g}", f"{fast_alias:g}")
        node = NodeSpec("aws:0", "aws", 1, 3.0, 62.5)
        config = GpuConfig("aws:1:1", "aws", 1, 1, 1.0)
        job = JobSpec("j", "bert_1.py", (config,))
        placement = Placement((("aws:0", 1),))
        response = {slow_alias: 0.7, fast_alias: 0.9}
        oracle = SimpleNamespace(
            response_exponent=0.5,
            query=lambda _job, _config, cpu, _ram, _prop: SimpleNamespace(rho=response.get(cpu, 0.5)),
        )
        scheduler = SimpleNamespace(
            node_map={node.node_id: node}, nodes=[node], oracle=oracle,
            allocation_floor=lambda proportional: proportional,
            resource_points=lambda *_: ((slow_alias, fast_alias), (62.5,)),
            minimum_cpu_per_gpu=1.0, minimum_ram_gb_per_gpu=20.83,
            _node_usage=lambda _placement, cpu, ram: (cpu, ram), _fits_column=lambda _usage: True,
        )
        self.assertEqual(full_max_rho(scheduler, job, config, placement), (0.9, 3, None))
        self.assertEqual(full_max_rho(scheduler, job, config, placement, require_witness=True), (0.9, 3, 0.7))


class RecoveryAdmission(BoundedCorrectionMixin, CleanAdmission):
    pass


class RecoveryIntegrationTests(TestCase):
    def fixture(self, alternate_eta, *, ordinary_move=False, peer_type=None):
        """A paid recovery can move one started job to the other GPU type."""
        job = JobSpec("j", "bert_1.py", tuple(
            GpuConfig(f"{gpu}:1:1", gpu, 1, 1, rate)
            for gpu, rate in (("aws", 1.0), ("dgx-ext", 5.0))
        ))
        live_jobs = [job]
        if peer_type is not None:
            live_jobs.append(JobSpec("peer", "bert_1.py", (
                GpuConfig(f"{peer_type}:1:1", peer_type, 1, 1, 1.0),
            ), source_row_index=1))
        nodes = [NodeSpec(f"{gpu}:0", gpu, 1, 3.0, 62.5) for gpu in ("aws", "dgx-ext", "rtx")]
        admission = RecoveryAdmission(300.0, 250.0, 60.0)
        admission.factors = (1, 1, 1)
        admission.clean_gpu = "fixed"
        admission.clean_aux = "opt"
        admission.delay_costs = {
            spec.job_id: {config.gpu_type: 50.0 for config in spec.configs} for spec in live_jobs
        }
        admission.profile_sources = {
            spec.job_id: {config.config_id: "direct_profile" for config in spec.configs}
            for spec in live_jobs
        }

        def make_scheduler(specs):
            scheduler = FactorScheduler(
                nodes, specs, JointProfileOracle(ROOT / "inputs/models", "fixed_reference", neutral_response=True),
                [1.0, 3.0], [20.83, 62.5], verify_exact=False,
                minimum_cpu_per_gpu=1.0, minimum_ram_gb_per_gpu=20.83, candidate_policy="b3",
            )
            scheduler.current_known_configs = {
                (spec.job_id, config.config_id) for spec in specs for config in spec.configs
            }
            scheduler.resource_point_provider = points
            return scheduler

        def points(scheduler, *_):
            return scheduler.aux_cpu_points, scheduler.aux_ram_points

        def score(columns):
            scored, evidence = [], []
            for column in columns:
                aws = column.config.gpu_type == "aws"
                eta = (130.0 if admission.is_prop(column) else 100.0) if aws else alternate_eta
                if column.job_id == "peer":
                    eta = 100.0
                scored.append(replace(column, scheduling_utility=1.0 / eta))
                evidence.append(dict(
                    column_id=column.column_id, estimated_completion=eta, raw_rate=column.config.sia_goodput,
                    delay=0.0 if column.job_id == "peer" or aws else 50.0,
                ))
            return scored, evidence

        first = make_scheduler(live_jobs)
        initial, evidence = score(first.build_joint_columns())
        first.current_completion_scores = {entry["column_id"]: entry for entry in evidence}
        initial = [
            c for c in initial if (c.job_id == "peer" or c.config.gpu_type == "aws") and admission.is_prop(c)
        ]
        current_by_job = admission.decide(first, initial, 0.0, {}).selected
        if ordinary_move:
            admission.previous_known_configs.remove(("j", "dgx-ext:1:1"))
        else:
            admission.services["j"]["delay_charged"] = 250.0
        updated_jobs = []
        for spec in live_jobs:
            current = current_by_job[spec.job_id]
            updated_jobs.append(replace(
                spec, current_config_id=current.config.config_id, current_nodes=current.placement.node_ids,
                current_placement=current.placement.node_gpus, current_cpu_per_gpu=current.query.cpu_per_gpu,
                current_ram_gb_per_gpu=current.query.ram_gb_per_gpu, query_round=1,
            ))
        scheduler = make_scheduler(updated_jobs)
        policy = SimpleNamespace(
            liveness=admission, productive={spec.job_id: 100.0 for spec in live_jobs}, _search_mode="LAZY_FO",
            _search_started=perf_counter(), _search_budget_seconds=5.0, _recovery_precheck=True,
        )
        bounds = {"j": {
            ("aws:1:1", (("aws:0", 1),)): (100.0, 100.0),
            ("dgx-ext:1:1", (("dgx-ext:0", 1),)): (alternate_eta, alternate_eta),
        }}
        if peer_type is not None:
            bounds["peer"] = {(f"{peer_type}:1:1", ((f"{peer_type}:0", 1),)): (100.0, 100.0)}
        return policy, scheduler, points, score, bounds

    def decide(self, alternate_eta, precheck, *, drop_witness=False, ordinary_move=False, peer_type=None):
        policy, scheduler, points, score, bounds = self.fixture(
            alternate_eta, ordinary_move=ordinary_move, peer_type=peer_type,
        )
        search = LazySearch(policy, scheduler, points, score, potential_recovery_jobs(policy, scheduler))
        scheduler.lazy_search = search
        self.assertTrue(search.recovery_eager)
        if precheck:
            if drop_witness:
                bounds["j"].pop(("aws:1:1", (("aws:0", 1),)))
            search.precheck(policy, bounds)
        columns, evidence = score(scheduler.build_joint_columns()) if search.recovery_eager else search.initial()
        scheduler.current_completion_scores = {entry["column_id"]: entry for entry in evidence}
        decision = policy.liveness.decide(scheduler, columns, 360.0, policy.productive)
        return decision, search.stats

    def test_certificate_prevents_spurious_offer_from_sparse_incumbent_set(self):
        # The 110-second backup is faster than the visible 130-second live quota,
        # yet slower than a legal 100-second auxiliary quota on the current GPU.
        reference, _ = self.decide(110.0, precheck=False)
        optimized, stats = self.decide(110.0, precheck=True)
        self.assertEqual(stats["recovery_certificate"], "proved_no_offer")
        self.assertEqual(stats["generation"], "LAZY")
        self.assertFalse(optimized.audit["recovery_offers"])
        self.assertEqual(optimized.audit["recovery_offers"], reference.audit["recovery_offers"])
        self.assertEqual(geometry(optimized.selected["j"]), geometry(reference.selected["j"]))
        self.assertFalse(optimized.audit["tail_recoveries"])

    def test_possible_and_unknown_recovery_keep_the_paid_permit(self):
        for drop_witness in (False, True):
            with self.subTest(drop_witness=drop_witness):
                reference, _ = self.decide(80.0, precheck=False)
                optimized, stats = self.decide(80.0, precheck=True, drop_witness=drop_witness)
                self.assertEqual(stats["recovery_certificate"], "uncertain")
                self.assertEqual(stats["generation"], "RECOVERY_EAGER")
                self.assertIn("j", optimized.audit["recovery_offers"])
                self.assertEqual(optimized.audit["recovery_offers"], reference.audit["recovery_offers"])
                self.assertEqual(optimized.audit["tail_recoveries"], reference.audit["tail_recoveries"])
                self.assertEqual(optimized.audit["tail_recoveries"][0]["cumulative_delay"], 300.0)
                self.assertEqual(optimized.selected["j"].config.gpu_type, "dgx-ext")

    def test_no_offer_certificate_leaves_ordinary_gpu_changes_eligible(self):
        reference, _ = self.decide(110.0, precheck=False, ordinary_move=True)
        optimized, stats = self.decide(110.0, precheck=True, ordinary_move=True)
        self.assertEqual(stats["recovery_certificate"], "proved_no_offer")
        self.assertEqual(optimized.selected["j"].config.gpu_type, "dgx-ext")
        self.assertEqual(geometry(optimized.selected["j"]), geometry(reference.selected["j"]))
        self.assertFalse(optimized.audit["recovery_offers"])
        self.assertEqual(optimized.audit["service_budget"], reference.audit["service_budget"])

    def test_multi_job_certificate_accounts_for_fixed_peer_gpu_capacity(self):
        for peer_type, certificate, recovered in (
                ("dgx-ext", "proved_no_offer", False), ("rtx", "uncertain", True)):
            with self.subTest(peer_type=peer_type):
                reference, _ = self.decide(80.0, precheck=False, peer_type=peer_type)
                optimized, stats = self.decide(80.0, precheck=True, peer_type=peer_type)
                self.assertEqual(stats["recovery_scope"], "multi")
                self.assertEqual(stats["recovery_certificate"], certificate)
                self.assertEqual(optimized.audit["recovery_offers"], reference.audit["recovery_offers"])
                self.assertEqual(optimized.audit["tail_recoveries"], reference.audit["tail_recoveries"])
                self.assertEqual(bool(optimized.audit["tail_recoveries"]), recovered)
                self.assertEqual(optimized.selected["peer"].config.gpu_type, peer_type)


if __name__ == "__main__":
    main()
