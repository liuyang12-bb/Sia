from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from joint_prototype.joint_integration import JointSiaPolicy
from joint_prototype.joint_profile import GpuConfig, JobSpec, JointProfileOracle, NodeSpec
from joint_prototype.joint_scheduler import JointScheduler, PolicyResult


ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT.parent / "research_sources" / "Synergy" / "simulator" / "models"


class B3FailureHandlingTests(unittest.TestCase):
    def scheduler(self, jobs, nodes=None):
        nodes = nodes or [NodeSpec("aws:0", "aws", 1, 1.0, 100.0)]
        return JointScheduler(
            nodes=nodes,
            jobs=jobs,
            oracle=JointProfileOracle(MODELS, "fixed_reference", neutral_response=True),
            aux_cpu_points=[1.0],
            aux_ram_points=[20.83],
            verify_exact=False,
            minimum_cpu_per_gpu=1.0,
            minimum_ram_gb_per_gpu=20.83,
        )

    @staticmethod
    def job(job_id, gpu_type="aws", checkpoint=False, current=None, nodes=()):
        return JobSpec(
            job_id=job_id,
            template_file="res18_1.py",
            configs=(GpuConfig(f"{gpu_type}:1:1", gpu_type, 1, 1, 1.0),),
            checkpoint_locked=checkpoint,
            current_config_id=current,
            current_nodes=tuple(nodes),
            current_placement=tuple((n, 1) for n in nodes),
            current_cpu_per_gpu=1.0 if nodes else None,
            current_ram_gb_per_gpu=20.83 if nodes else None,
        )

    def test_normal_b3_produces_valid_allocation(self):
        scheduler = self.scheduler([self.job("normal")])
        result = scheduler.run_b3()
        self.assertEqual(result.status, "OPTIMAL")
        self.assertTrue(result.audit["valid"])
        self.assertEqual(set(result.selected), {"normal"})

    def test_one_job_without_columns_does_not_block_feasible_peer(self):
        nodes = [
            NodeSpec("aws:0", "aws", 1, 1.0, 100.0),
            NodeSpec("rtx:0", "rtx", 1, 0.5, 100.0),
        ]
        scheduler = self.scheduler(
            [self.job("feasible", "aws"), self.job("wait", "rtx")], nodes
        )
        result = scheduler.run_b3()
        self.assertTrue(result.audit["valid"])
        self.assertIn("feasible", result.selected)
        self.assertIn("wait", result.unallocated_jobs)

    def test_unallocated_scaling_flag_does_not_create_phantom_lock(self):
        scheduler = self.scheduler([
            self.job("phantom", checkpoint=True, current=None, nodes=())
        ])
        columns = scheduler.build_joint_columns()
        self.assertTrue(columns)
        self.assertEqual({column.job_id for column in columns}, {"phantom"})

    def test_no_columns_has_explicit_failure_audit(self):
        nodes = [NodeSpec("rtx:0", "rtx", 1, 0.5, 100.0)]
        scheduler = self.scheduler([self.job("none", "rtx")], nodes)
        result = scheduler.run_b3()
        self.assertEqual(result.status, "NO_COLUMNS")
        self.assertFalse(result.audit["valid"])
        self.assertEqual(result.audit["failure_reason"], "NO_CANDIDATE_COLUMNS")
        self.assertEqual(result.audit["candidate_column_count"], 0)
        self.assertEqual(result.audit["candidate_counts_by_job"], {"none": 0})

    def test_milp_without_x_has_solver_diagnostics(self):
        scheduler = self.scheduler([self.job("solver")])
        fake = SimpleNamespace(
            x=None, status=2, success=False, message="forced no solution",
            fun=None, mip_gap=None, mip_node_count=0,
        )
        with patch("joint_prototype.joint_scheduler.milp", return_value=fake):
            result = scheduler.run_b3()
        self.assertEqual(result.status, "MIP_NO_SOLUTION_STATUS_2")
        self.assertFalse(result.audit["valid"])
        self.assertEqual(result.audit["failure_reason"], "MIP_NO_USABLE_SOLUTION")
        self.assertEqual(result.audit["solver"]["status"], 2)
        self.assertEqual(result.audit["candidate_column_count"], 1)

    def test_safe_previous_allocation_fallback(self):
        spec = self.job("keep", checkpoint=True, current="aws:1:1", nodes=("aws:0",))
        scheduler = self.scheduler([spec])
        previous = scheduler.build_joint_columns()[0]
        policy = object.__new__(JointSiaPolicy)
        policy._last_committed_columns = {"keep": previous}
        failed = PolicyResult(
            "B3", "MIP_NO_SOLUTION_STATUS_2", {}, ["keep"], 0, 0, 0,
            audit=scheduler._failure_audit("MIP_NO_USABLE_SOLUTION", [previous], ("gpu", "cpu", "ram")),
        )
        fallback = policy._b3_failure_fallback(failed, [spec], scheduler)
        self.assertIsNotNone(fallback)
        self.assertEqual(fallback.status, "FALLBACK_PRESERVE_PREVIOUS_VALID_ALLOCATION")
        self.assertEqual(fallback.selected["keep"].config.config_id, "aws:1:1")
        self.assertTrue(fallback.audit["valid"])

    def test_invalid_previous_allocation_refuses_fallback(self):
        spec = self.job("bad", checkpoint=True, current="aws:1:1", nodes=("aws:0",))
        scheduler = self.scheduler([spec])
        previous = scheduler.build_joint_columns()[0]
        invalid = replace(previous, node_usage=(("aws:0", 2.0, 2.0, 41.66),))
        policy = object.__new__(JointSiaPolicy)
        policy._last_committed_columns = {"bad": invalid}
        failed = PolicyResult(
            "B3", "MIP_NO_SOLUTION_STATUS_2", {}, ["bad"], 0, 0, 0,
            audit=scheduler._failure_audit("MIP_NO_USABLE_SOLUTION", [previous], ("gpu", "cpu", "ram")),
        )
        fallback = policy._b3_failure_fallback(failed, [spec], scheduler)
        self.assertIsNone(fallback)
        self.assertFalse(failed.audit["safe_previous_allocation_available"])
        self.assertFalse(failed.audit["previous_allocation_audit"]["valid"])

    def test_every_successful_selection_passes_capacity_audit(self):
        scheduler = self.scheduler([self.job("capacity")])
        result = scheduler.run_b3()
        repeated = scheduler.audit(list(result.selected.values()))
        self.assertTrue(repeated["valid"])
        self.assertLessEqual(repeated["node_totals"]["aws:0"]["gpus"], 1.0)
        self.assertLessEqual(repeated["node_totals"]["aws:0"]["cpu"], 1.0)
        self.assertLessEqual(repeated["node_totals"]["aws:0"]["ram_gb"], 100.0)


if __name__ == "__main__":
    unittest.main()
