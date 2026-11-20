import types
import unittest
from pathlib import Path

from joint_prototype.joint_integration import JointSiaPolicy
from joint_prototype.joint_profile import GpuConfig, JobSpec, JointProfileOracle, NodeSpec
from joint_prototype.joint_scheduler import JointScheduler, _proportional_point
from run_cpu_bottleneck_smoke import summarize


ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT.parent / "research_sources" / "Synergy" / "simulator" / "models"


class B3ProportionalAblationTests(unittest.TestCase):
    def setUp(self):
        self.nodes = (
            NodeSpec("slow:0", "slow", 4, 4, 500),
            NodeSpec("fast:0", "fast", 8, 8, 1000),
        )
        self.configs = (
            GpuConfig("slow:1:1", "slow", 1, 1, 1.0),
            GpuConfig("fast:1:1", "fast", 1, 1, 2.0),
            GpuConfig("fast:1:2", "fast", 2, 1, 3.0),
        )
        self.job = JobSpec("job", "alexnet_1.py", self.configs)
        self.oracle = JointProfileOracle(
            MODELS,
            normalization_mode="fixed_reference",
            fixed_reference=(3.0, 62.5),
            neutral_response=False,
            response_exponent=0.5,
        )
        self.scheduler = JointScheduler(
            self.nodes,
            (self.job,),
            self.oracle,
            aux_cpu_points=(2, 3, 6, 12),
            aux_ram_points=(20.83, 62.5, 125, 250),
            enforce_proportional_progress_floor=True,
            verify_exact=True,
            minimum_cpu_per_gpu=1.0,
            minimum_ram_gb_per_gpu=20.83,
        )

    def test_columns_use_only_deterministic_proportional_resources(self):
        columns = self.scheduler.build_proportional_columns()
        self.assertGreaterEqual(len({column.config.config_id for column in columns}), 2)
        keys = set()
        for column in columns:
            proportional = _proportional_point(column.placement, self.scheduler.node_map)
            expected = self.scheduler.allocation_floor(proportional)
            self.assertEqual(column.query.cpu_per_gpu, expected[0])
            self.assertEqual(column.query.ram_gb_per_gpu, expected[1])
            key = (column.job_id, column.config.config_id, column.placement.node_gpus)
            self.assertNotIn(key, keys)
            keys.add(key)

    def test_b3_prop_can_choose_across_gpu_candidates(self):
        policy = JointSiaPolicy(
            mode="b3_prop",
            oracle=self.oracle,
            node_capacities={
                "slow": {"cpu_cores": 4, "ram_gb": 500},
                "fast": {"cpu_cores": 8, "ram_gb": 1000},
            },
            template_map={"job": "alexnet_1.py"},
            aux_cpu_points=(2, 3, 6, 12),
            aux_ram_points=(20.83, 62.5, 125, 250),
            minimum_cpu_per_gpu=1.0,
            minimum_ram_gb_per_gpu=20.83,
            p_fairness=-0.5,
            lambda_n=1.1,
            project_throughputs=True,
        )
        info = types.SimpleNamespace(
            age=600,
            num_restarts=0,
            used_gpu_seconds=1000.0,
            wasted_gpu_seconds=0.0,
        )
        result = policy._b3_result({"job": info}, (self.job,), self.scheduler)
        self.assertEqual(result.status, "OPTIMAL")
        self.assertEqual(result.policy, "B3_PROP_SIA_GOODPUT_PROPORTIONAL_AUX_MIP")
        self.assertTrue(result.audit["valid"])
        self.assertIn(result.selected["job"].config.config_id, {cfg.config_id for cfg in self.configs})
        candidate_configs = {
            column.config.config_id
            for column in self.scheduler.build_proportional_columns()
            if column.job_id == "job"
        }
        self.assertGreaterEqual(len(candidate_configs), 2)

    def test_checkpoint_lock_constrains_config_and_placement(self):
        locked = JobSpec(
            job_id="job",
            template_file="alexnet_1.py",
            configs=self.configs,
            checkpoint_locked=True,
            current_config_id="fast:1:2",
            current_nodes=("fast:0",),
            current_placement=(("fast:0", 2),),
            current_cpu_per_gpu=1.0,
            current_ram_gb_per_gpu=125.0,
        )
        scheduler = JointScheduler(
            self.nodes,
            (locked,),
            self.oracle,
            aux_cpu_points=(2, 3, 6),
            aux_ram_points=(20.83, 62.5, 125),
            minimum_cpu_per_gpu=1.0,
            minimum_ram_gb_per_gpu=20.83,
        )
        columns = scheduler.build_proportional_columns()
        self.assertTrue(columns)
        self.assertTrue(all(column.config.config_id == "fast:1:2" for column in columns))
        self.assertTrue(all(column.placement.node_ids == ("fast:0",) for column in columns))

    def test_censored_never_started_job_is_serializable(self):
        job = types.SimpleNamespace(
            name="never-started",
            completion_time=None,
            start_time=None,
            progress=0.0,
            epoch=0,
            num_restarts=None,
            num_migrations=None,
            used_gpu_seconds=0.0,
            wasted_gpu_seconds=0.0,
        )
        cluster = types.SimpleNamespace(
            jobs=[job],
            joint_round_logs=[],
            current_time=60.0,
            num_nodes={"fast": 1},
            all_complete=lambda: False,
        )
        summary = summarize(
            cluster,
            wall_seconds=0.0,
            max_rounds=6000,
            capacities={"fast": {"cpu_cores": 8, "ram_gb": 1000}},
        )
        self.assertEqual(summary["completed_jobs"], 0)
        self.assertEqual(summary["jobs"]["never-started"]["num_restarts"], 0)


if __name__ == "__main__":
    unittest.main()
