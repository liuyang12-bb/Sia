"""Bounded red/green fixtures; no cohort simulation and no liveness redesign."""
import unittest
import json
import os
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import test_joint_integration_coverage as coverage_fixtures
import test_joint_integration as job_fixtures
import test_b3_failure_handling as failure_fixtures
from joint_prototype.joint_profile import JobSpec, GpuConfig, NodeSpec, JointProfileOracle
from joint_prototype.joint_scheduler import JointScheduler
from run_cpu_bottleneck_smoke import summarize
from joint_prototype.execution_audit import audit_execution
from joint_prototype.joint_scheduler import PolicyResult
from joint_prototype.joint_integration import JointSiaPolicy
from joint_prototype.synergy_adapter import SynergyTuneAdapter


class Stage1CorrectnessTests(unittest.TestCase):
    @staticmethod
    def evidence(name, data):
        folder = os.environ.get("STAGE1_EVIDENCE_DIR")
        if folder:
            path = Path(folder)
            path.mkdir(exist_ok=True, parents=True)
            target = path / (name + ".json")
            if target.exists():
                raise RuntimeError("refuse to overwrite tiny-fixture evidence")
            target.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def setUp(self):
        self.fixture = coverage_fixtures.JointIntegrationCoverageTests()
        self.fixture.setUp()
        self.small = failure_fixtures.B3FailureHandlingTests()

    def locked_spec(self):
        return JobSpec("locked", "res18_1.py", (GpuConfig("aws:1:1", "aws", 1, 1, 1.0),),
                       checkpoint_locked=True, current_config_id="aws:1:1", current_nodes=("aws:0",),
                       current_placement=(("aws:0", 1),), current_cpu_per_gpu=1.0, current_ram_gb_per_gpu=20.83)

    def test_negative_utility_lock_must_be_selected_and_keep_all_quotas(self):
        spec = self.locked_spec()
        s = self.small.scheduler([spec])
        for builder in (s.build_joint_columns, s.build_proportional_columns):
            columns = [replace(c, scheduling_utility=-100.0) for c in builder()]
            result = s.solve("negative-locked", columns, ("gpu", "cpu", "ram"))
            self.assertEqual(set(result.selected), {"locked"})
            c = result.selected["locked"]
            self.assertEqual(c.placement.node_gpus, spec.current_placement)
            self.assertEqual(c.query.ram_gb_per_gpu, 20.83)
            self.assertFalse(s.audit([])["valid"])

    def test_fifo_real_arrival_then_source_row_not_job_id(self):
        cfg = GpuConfig("aws:1:1", "aws", 1, 1, 1.0)
        jobs = [JobSpec("a", "res18_1.py", (cfg,), arrival_time=40, source_row_index=0),
                JobSpec("z", "res18_1.py", (cfg,), arrival_time=20, source_row_index=2)]
        s = self.small.scheduler(jobs)
        r = SynergyTuneAdapter(s).run("B5", {j.job_id: cfg for j in jobs}, {})
        self.assertEqual(set(r.selected), {"z"})
        jobs = [replace(j, arrival_time=20, source_row_index=1 if j.job_id=="z" else 2) for j in jobs]
        s = self.small.scheduler(jobs)
        self.assertEqual(set(SynergyTuneAdapter(s).run("B5", {j.job_id: cfg for j in jobs}, {}).selected), {"z"})

    def test_mid_interval_arrival_and_first_times(self):
        c = self.fixture._cluster([dict(name="bert-mid", time=17, application="bert", num_replicas=1, batch_size=384)], mode="b1")
        c.step(60)
        j = c.jobs[0]
        e = c.joint_round_logs[0]["execution"][0]
        self.assertEqual(e["waiting_seconds"], 43)
        self.assertEqual(j.first_allocation_time, 60)
        self.assertIsNone(j.first_productive_training_time)
        infos = c.get_job_infos()
        specs, _ = c.policy._job_specs(infos, c.allocations)
        self.assertEqual(specs[0].arrival_time, 17)
        self.assertEqual(specs[0].source_row_index, 0)
        c.step(60)
        c.step(60)
        # Isolate the first productive boundary from the next Sia reoptimization.
        with patch.object(c, "optimize_policy", return_value=([], [])):
            c.step(1)
        self.assertEqual(j.first_productive_training_time, 180)
        self.assertEqual(j.start_time, j.first_allocation_time)
        self.evidence("partial_arrival", c.joint_round_logs)

    def test_fallback_rebuilds_current_config_query_not_old_rho(self):
        old = self.small.job("keep")
        s = self.small.scheduler([old])
        previous = s.build_joint_columns()[0]
        newer = replace(old, configs=(replace(old.configs[0], sia_goodput=7, cpu_demand_scale=2),),
                        current_config_id="aws:1:1", current_nodes=("aws:0",))
        current = self.small.scheduler([newer])
        p = object.__new__(JointSiaPolicy)
        p._last_committed_columns = {"keep": previous}
        failed = PolicyResult("B3", "MIP_NO_SOLUTION_STATUS_1", {}, ["keep"], 0, 0, 0, audit={"valid": False})
        r = p._b3_failure_fallback(failed, [newer], current)
        self.assertIsNotNone(r)
        self.assertEqual(r.selected["keep"].query.raw_sia_goodput, 7)
        self.assertEqual(r.selected["keep"].config.cpu_demand_scale, 2)
        self.assertFalse(current.audit([previous])["valid"])
        self.evidence("fresh_fallback", r.to_dict())

    def test_direct_raw_execution_rate_not_reconstructed(self):
        j = job_fixtures.JointIntegrationTests()._job(2)
        j.enable_joint_accounting = True
        j.rescale_time = 0
        j.step(1)
        self.assertTrue(j.joint_step_segments)
        s = j.joint_step_segments[0]
        self.assertGreater(s["raw_rate_before_rho"], 0)
        self.assertEqual(s["effective_rate"], s["raw_rate_before_rho"] * 2)
        self.assertAlmostEqual(s["progress_delta"], s["effective_rate"] * s["seconds"])

    def test_all_modes_keep_full_lock_and_b1_secondary_admission(self):
        for mode in ("b1", "b4", "b5", "b3", "b3_prop"):
            with self.subTest(mode=mode):
                rows = [dict(name="bert-lock", time=0, application="bert", num_replicas=1, batch_size=384),
                        dict(name="bert-new", time=90, application="bert", num_replicas=1, batch_size=384)]
                c = self.fixture._cluster(rows, mode=mode, nnodes={"aws": 1}, ngpus={"aws": 4},
                                          capacities={"aws": {"cpu_cores": 1, "ram_gb": 500}})
                c.policy.minimum_cpu_per_gpu = 1.0
                c.policy.minimum_ram_gb_per_gpu = 20.83
                with patch("joint_prototype.joint_integration.SiaPolicy.optimize", return_value={"bert-lock": ("aws", (0,))}):
                    c.step(60)
                old = c.policy.current_resources["bert-lock"]
                with patch("joint_prototype.joint_integration.SiaPolicy.optimize", return_value={"bert-lock": ("aws", (0,)), "bert-new": ("aws", (0,))}):
                    c.step(60)
                new = c.policy.current_resources["bert-lock"]
                self.assertEqual({k: new[k] for k in ("config_id", "placement", "cpu_per_gpu", "ram_gb_per_gpu")},
                                 {k: old[k] for k in ("config_id", "placement", "cpu_per_gpu", "ram_gb_per_gpu")})
                self.assertNotIn("bert-new", c.policy.current_resources)
                result = c.policy.last_decision["result"]
                self.assertTrue(result["audit"]["precommit_independent_audit"]["valid"])
                if mode == "b1":
                    self.assertEqual(result["audit"]["admission_rule"], "WHOLE_JOB_ONLY_NO_GPU_CONFIG_CHANGE")
                if mode == "b5":
                    actions = [x["action"] for x in result["audit"]["tune_trace"]]
                    self.assertIn("PRESERVE_EXACT_CHECKPOINT_LOCK", actions)
                    self.assertIn("WAIT_FINAL_RETRY_INFEASIBLE", actions)
                self.evidence("full_lock_"+mode, c.joint_round_logs)

    def test_b5_fifo_through_cluster_with_real_arrivals_and_source_order(self):
        for arrival in (17, 40):
            rows = [dict(name="bert-z", time=17, application="bert", num_replicas=1, batch_size=384),
                    dict(name="bert-a", time=arrival, application="bert", num_replicas=1, batch_size=384)]
            c = self.fixture._cluster(rows, mode="b5", nnodes={"aws": 1}, ngpus={"aws": 4},
                                      capacities={"aws": {"cpu_cores": 1, "ram_gb": 500}})
            c.policy.minimum_cpu_per_gpu = 1
            c.policy.minimum_ram_gb_per_gpu = 20.83
            with patch("joint_prototype.joint_integration.SiaPolicy.optimize", return_value={"bert-a": ("aws", (0,)), "bert-z": ("aws", (0,))}):
                c.step(60)
            self.assertEqual(set(c.policy.current_resources), {"bert-z"})
            self.evidence("fifo_arrival_"+str(arrival), c.joint_round_logs)

    def test_locked_peer_cannot_be_downgraded_or_moved(self):
        spec = self.locked_spec()
        peer = replace(spec, job_id="peer", checkpoint_locked=False, current_config_id=None, current_nodes=(), current_placement=())
        s = self.small.scheduler([spec, peer])
        adapter = SynergyTuneAdapter(s)
        fixed = {j.job_id: j.configs[0] for j in (spec, peer)}
        self.assertIsNone(adapter._repack_all_fair(["locked", "peer"], fixed))
        r = adapter.run("B5", fixed, {})
        self.assertEqual(set(r.selected), {"locked"})
        self.assertEqual(r.selected["locked"].query.ram_gb_per_gpu, 20.83)
        self.assertFalse(s.audit([replace(r.selected["locked"], placement=replace(r.selected["locked"].placement, node_gpus=(("aws:0", 2),)))])["valid"])

    def test_infeasible_lock_stops_without_committing(self):
        c = self.fixture._cluster([dict(name="bert-lock", time=0, application="bert", num_replicas=1, batch_size=384)], mode="b3")
        c.step(60)
        before = dict(c.policy.current_resources)
        gpu_type = before["bert-lock"]["gpu_type"]
        c.policy.node_capacities[gpu_type]["cpu_cores"] = 0.01  # fault injection, NOT experimental capacity change
        with self.assertRaisesRegex(RuntimeError, "INFEASIBLE_CHECKPOINT_LOCK"):
            c.optimize_policy()
        self.assertEqual(c.policy.current_resources, before)

    def test_all_empty_absorbing_state_is_still_unresolved(self):
        c = self.fixture._cluster([dict(name="cifar10-cold", time=0, application="cifar10", num_replicas=1, batch_size=128)],
                                 mode="b3_prop", nnodes={"aws": 1}, ngpus={"aws": 4}, capacities={"aws": {"cpu_cores": 4, "ram_gb": 500}}, neutral=False)
        c.policy.oracle.response_exponent = 0.5
        c.policy.template_map["cifar10-cold"] = "alexnet_1.py"
        for _ in range(3):
            c.step(60)
            self.assertEqual(c.policy.last_decision["result"]["status"], "OPTIMAL")
            self.assertEqual(c.policy.current_resources, {})
        self.assertEqual(c.jobs[0].progress, 0)
        self.assertIsNone(c.jobs[0].first_productive_training_time)
        self.evidence("UNRESOLVED_empty_optimal", c.joint_round_logs)

    @staticmethod
    def artificial_job(rho, delay=0):
        j = job_fixtures.JointIntegrationTests()._job(rho)
        app = SimpleNamespace(max_epochs=100, init_batch_size=1,
                              get_grad_stats=lambda batch, epoch: (1.0, 0.0),
                              get_throughput=lambda placement, batch: (1.0, 0.0),
                              get_progress=lambda epoch: 10.0 * epoch,
                              get_completion_epoch=lambda batch: 1,
                              get_best_metric=lambda batch, epoch: 0)
        j.applications = {"rtx": app}
        j.atomic_bsz = 1
        j.accum_steps = 0
        j.calibration_factor = 1
        j.rescale_time = delay
        j.update_params = lambda *args: None
        j.update_local_bsz = lambda *args: None
        j.enable_joint_accounting = True
        return j

    def test_independent_multiplier_checkpoint_epoch_completion_boundaries(self):
        slow, fast = self.artificial_job(1, 3), self.artificial_job(2, 3)
        slow.step(3)
        fast.step(3)
        self.assertEqual((slow.progress, fast.progress), (0, 0))
        slow.step(6)
        fast.step(6)
        self.assertEqual((slow.progress, fast.progress), (6, 12))
        self.assertEqual(fast.epoch, 1)
        self.assertTrue(all(s["raw_rate_before_rho"] == 1 for s in fast.joint_step_segments))
        # Completion truncates productive work; never assert linearity past completion.
        fast.step(60)
        self.assertEqual(fast.progress, 20)
        self.assertEqual(fast.completion_time, 13)
        self.assertEqual(sum(s["seconds"] for s in fast.joint_step_segments), 4)
        self.assertEqual(fast.execution_time, 10)
        slow.step(60)
        self.assertEqual(slow.completion_time, 23)

    def test_resource_hours_stop_at_within_interval_completion(self):
        c = self.fixture._cluster([dict(name="bert-fixture", time=0, application="bert", num_replicas=1, batch_size=384)], mode="b1")
        j = self.artificial_job(2, 3)
        j.name = "bert-fixture"
        j.source_row_index = 0
        j.first_allocation_time = 0
        j.first_productive_training_time = None
        c.jobs = [j]
        c.policy.current_resources = {j.name: {"gpus": 1, "rho": 2, "node_usage": [{"cpu": 2, "ram_gb": 50}]}}
        with patch.object(c, "optimize_policy", return_value=([], [])), patch.object(c, "step_log", side_effect=lambda: c.logs.append({})):
            c.step(60)
        e = c.joint_round_logs[0]["execution"][0]
        self.assertEqual((e["waiting_seconds"], e["checkpoint_delay_consumed"], e["training_seconds"]), (0, 3, 10))
        self.assertEqual(e["allocated_seconds"], 13)
        self.assertEqual(e["post_completion_seconds"], 47)
        self.assertTrue(audit_execution(e)["valid"])
        forged = dict(e)
        forged["training_segments"] = [dict(s, effective_rate=s["effective_rate"]*2) for s in e["training_segments"]]
        self.assertFalse(audit_execution(forged)["valid"])
        m = summarize(c, 0, 2, {})
        self.assertAlmostEqual(m["resource_hours"]["gpu_hours"], 13/3600)
        self.assertAlmostEqual(m["resource_hours"]["cpu_core_hours"], 26/3600)
        self.assertAlmostEqual(m["resource_hours"]["ram_gb_hours"], 650/3600)
        self.evidence("completion_timing_and_hours", {"event": e, "summary": m})

    def test_epoch_rounding_bound_is_explicit_not_hidden_in_raw(self):
        j = self.artificial_job(3)
        j.step(4)
        self.assertEqual(j.progress, 13)  # 10 boundary at round(10/3)=3, then 1s*3.
        self.assertEqual(j.joint_step_segments[0]["rounding_progress_residual"], 1)
        self.assertEqual(j.joint_step_segments[0]["raw_rate_before_rho"], 1)

    def test_precommit_rejects_forged_success_with_dropped_lock(self):
        c = self.fixture._cluster([dict(name="bert-lock", time=0, application="bert", num_replicas=1, batch_size=384)], mode="b3")
        c.step(60)
        before = dict(c.policy.current_resources)
        fake = PolicyResult("B3", "OPTIMAL", {}, ["bert-lock"], 0, 0, 0, audit={"valid": True})
        with patch.object(c.policy, "_b3_result", return_value=fake):
            with self.assertRaisesRegex(RuntimeError, "refusing non-atomic"):
                c.optimize_policy()
        self.assertEqual(c.policy.current_resources, before)


if __name__ == "__main__":
    unittest.main()
