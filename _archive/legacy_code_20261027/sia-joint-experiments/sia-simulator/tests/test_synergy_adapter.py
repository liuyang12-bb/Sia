import unittest
from pathlib import Path

from joint_prototype.joint_profile import GpuConfig, JobSpec, JointProfileOracle, NodeSpec
from joint_prototype.joint_scheduler import JointScheduler
from joint_prototype.synergy_adapter import SynergyTuneAdapter


WORKSPACE = Path(__file__).resolve().parents[3]
MODELS = WORKSPACE / "research_sources" / "Synergy" / "simulator" / "models"


class SynergyAdapterTests(unittest.TestCase):
    def _scheduler(self, nodes, jobs, neutral=False, minimum_cpu=0.0):
        return JointScheduler(
            nodes,
            jobs,
            JointProfileOracle(MODELS, "fixed_reference", neutral_response=neutral),
            aux_cpu_points=[1, 2, 3, 4, 6, 9, 12],
            aux_ram_points=[20.83, 62.5, 125],
            verify_exact=False,
            minimum_cpu_per_gpu=minimum_cpu,
            minimum_ram_gb_per_gpu=20.83 if minimum_cpu else 0.0,
        )

    def test_tune_atomically_downgrades_peer_to_fair_share(self):
        node = NodeSpec("x:0", "x", 2, 10, 125)
        config0 = GpuConfig("x:1:1", "x", 1, 1, 2.0)
        config1 = GpuConfig("x:1:1", "x", 1, 1, 3.0)
        jobs = (
            JobSpec("j0", "res18_1.py", (config0,)),
            JobSpec("j1", "res18_1.py", (config1,)),
        )
        scheduler = self._scheduler((node,), jobs)
        result = SynergyTuneAdapter(scheduler).run(
            "SY_TEST", {"j0": config0, "j1": config1}, {"source": "test"}
        )
        self.assertEqual(set(result.selected), {"j0", "j1"})
        self.assertEqual(result.selected["j0"].query.cpu_per_gpu, 5.0)
        self.assertEqual(result.selected["j1"].query.cpu_per_gpu, 5.0)
        self.assertIn(
            "ATOMIC_PEER_DOWNGRADE_TO_FAIR",
            [item["action"] for item in result.audit["tune_trace"]],
        )
        self.assertTrue(result.audit["valid"])

    def test_profile_grid_minimum_makes_oversized_gpu_config_wait(self):
        node = NodeSpec("x:0", "x", 8, 6, 500)
        config = GpuConfig("x:1:8", "x", 8, 1, 8.0)
        job = JobSpec("j0", "bert_1.py", (config,))
        scheduler = self._scheduler((node,), (job,), minimum_cpu=1.0)
        result = SynergyTuneAdapter(scheduler).run(
            "SY_TEST", {"j0": config}, {"source": "test"}
        )
        self.assertEqual(result.selected, {})
        self.assertEqual(result.unallocated_jobs, ["j0"])
        self.assertEqual(
            result.audit["tune_trace"][-1]["action"],
            "WAIT_FINAL_RETRY_INFEASIBLE",
        )
        self.assertEqual(result.audit["node_totals"]["x:0"]["cpu"], 0.0)

    def test_neutral_execution_has_exactly_one_multiplier(self):
        node = NodeSpec("x:0", "x", 1, 9, 125)
        config = GpuConfig("x:1:1", "x", 1, 1, 4.0)
        job = JobSpec("j0", "res18_1.py", (config,))
        scheduler = self._scheduler((node,), (job,), neutral=True)
        result = SynergyTuneAdapter(scheduler).run(
            "SY_TEST", {"j0": config}, {"source": "test"}
        )
        column = result.selected["j0"]
        self.assertEqual(column.query.rho, 1.0)
        self.assertEqual(column.query.effective_goodput, column.query.raw_sia_goodput)


if __name__ == "__main__":
    unittest.main()
