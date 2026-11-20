import math
import unittest
from pathlib import Path

import numpy as np

from applications import APPLICATIONS
from job import Job
from joint_prototype.joint_integration import JointSiaPolicy
from joint_prototype.joint_profile import GpuConfig, JobSpec, JointProfileOracle


ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT.parent / "research_sources" / "Synergy" / "simulator" / "models"


class JointIntegrationTests(unittest.TestCase):
    def _job(self, multiplier):
        applications = {name: APPLICATIONS[name]["bert"] for name in ("aws", "dgx-ext", "rtx")}
        job = Job(
            "bert-integration-test",
            applications,
            submission_time=0,
            target_num_replicas=1,
            target_batch_size=384,
        )
        job.migrate("rtx", [1])
        job.goodput_multiplier = multiplier
        return job

    def test_multiplier_does_not_advance_checkpoint_delay(self):
        job = self._job(4.0)
        progress = job.progress
        job.step(60)
        self.assertEqual(job.progress, progress)
        self.assertEqual(job.rescale_time, 60)
        job.step(60)
        self.assertEqual(job.progress, progress)
        self.assertEqual(job.rescale_time, 0)

    def test_multiplier_is_applied_once_to_training_progress(self):
        baseline = self._job(1.0)
        multiplied = self._job(2.0)
        baseline.rescale_time = 0
        multiplied.rescale_time = 0
        baseline.step(1)
        multiplied.step(1)
        self.assertGreater(baseline.progress, 0)
        self.assertTrue(math.isclose(multiplied.progress, 2 * baseline.progress, rel_tol=1e-10))

    def test_unallocated_job_waits_without_progress(self):
        job = self._job(3.0)
        job.placement = ()
        job.rescale_time = 0
        job.step(60)
        self.assertEqual(job.progress, 0)
        self.assertEqual(job.current_time, 60)

    def test_neutral_oracle_is_exactly_one(self):
        oracle = JointProfileOracle(MODELS, "fixed_reference", neutral_response=True)
        config = GpuConfig("rtx:1:1", "rtx", 1, 1, 2.0, cpu_demand_scale=2.0)
        job = JobSpec("job", "res18_1.py", (config,))
        for cpu, ram in ((1, 20.83), (3, 62.5), (12, 250)):
            query = oracle.query(job, config, cpu, ram, (2, 50))
            self.assertEqual(query.rho, 1.0)
            self.assertEqual(query.effective_goodput, query.raw_sia_goodput)

    def test_direct_and_projected_goodput_sources_remain_distinct(self):
        class Speedup:
            @staticmethod
            def get_goodput(num_nodes, num_replicas):
                return np.asarray(num_replicas, dtype=float)

        class Info:
            speedup_fn = {"aws": Speedup(), "rtx": None}
            scale_unit = {"aws": 1, "rtx": 1}
            min_replicas = {"aws": 0, "rtx": 0}
            max_replicas = {"aws": 4, "rtx": 4}
            cluster_throughput_ratios = {"rtx": {"aws": 1.5}, "aws": {"rtx": 2 / 3}}
            cluster_throughputs = {"aws": 1.0, "rtx": 1.5}

        policy = JointSiaPolicy(
            mode="b3",
            oracle=JointProfileOracle(MODELS, "fixed_reference"),
            node_capacities={"aws": {"cpu_cores": 8, "ram_gb": 200}, "rtx": {"cpu_cores": 16, "ram_gb": 500}},
            template_map={"job": "res18_1.py"},
            aux_cpu_points=[1, 2, 3],
            aux_ram_points=[20.83, 62.5],
            p_fairness=-0.5,
            project_throughputs=True,
        )
        policy.populate_valid_configs(
            {"aws": 1, "rtx": 1}, {"aws": 4, "rtx": 8}, {"aws": 1, "rtx": 1}
        )
        _, provenance = policy._raw_candidates(Info())
        sources = {item["goodput_source"] for item in provenance.values()}
        self.assertEqual(sources, {"direct_profile", "projected_profile"})


if __name__ == "__main__":
    unittest.main()
