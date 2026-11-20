import math
import unittest
from pathlib import Path

from joint_prototype.joint_profile import (
    GpuConfig,
    JobSpec,
    JointProfileOracle,
    NodeSpec,
    assign_templates_category_mix,
    assign_templates_fixed_hash,
    load_synergy_template,
)
from joint_prototype.joint_scheduler import JointScheduler, enumerate_placements
from joint_prototype.scenario_io import load_scenario


WORKSPACE = Path(__file__).resolve().parents[3]
SYNERGY_MODELS = WORKSPACE / "research_sources" / "Synergy" / "simulator" / "models"
SCENARIOS = Path(__file__).resolve().parents[1] / "joint_prototype" / "scenarios"


class JointPrototypeTests(unittest.TestCase):
    def scenario_scheduler(self, filename, mode="fixed_reference"):
        scenario = load_scenario(SCENARIOS / filename)
        oracle = JointProfileOracle(SYNERGY_MODELS, mode)
        scheduler = JointScheduler(
            scenario.nodes,
            scenario.jobs,
            oracle,
            scenario.aux_cpu_points,
            scenario.aux_ram_points,
            scenario.allocation_bonus,
        )
        return scenario, oracle, scheduler

    def test_official_matrix_is_loaded_with_provenance(self):
        template = load_synergy_template(SYNERGY_MODELS / "res18_1.py")
        self.assertEqual(template.matrix.shape, (9, 5))
        self.assertEqual(template.cpu_demand, 9)
        self.assertEqual(template.memory_demand_gb, 62.5)
        self.assertEqual(len(template.source_sha256), 64)
        self.assertAlmostEqual(template.matrix[2, 1], 1.0)

    def test_arithmetic_profile_literals_are_supported(self):
        template = load_synergy_template(SYNERGY_MODELS / "m5_1.py")
        self.assertAlmostEqual(template.memory_demand_gb, 187.5)
        neutral = load_synergy_template(SYNERGY_MODELS / "transformer_1.py")
        self.assertEqual(neutral.memory_demand_gb, 20.0)

    def test_fixed_reference_baseline_and_slowdown(self):
        scenario, oracle, _ = self.scenario_scheduler("scenario_a_fast_gpu_cpu_bottleneck.json")
        job = next(j for j in scenario.jobs if j.job_id == "target")
        config = next(c for c in job.configs if c.config_id == "fast-1g")
        baseline = oracle.query(job, config, 3, 62.5, (2, 62.5))
        shortage = oracle.query(job, config, 2, 62.5, (2, 62.5))
        self.assertAlmostEqual(baseline.rho, 1.0)
        self.assertLess(shortage.rho, 1.0)
        self.assertAlmostEqual(shortage.effective_goodput, shortage.raw_sia_goodput * shortage.rho)

    def test_capacity_relative_proportional_point_is_one(self):
        scenario, oracle, _ = self.scenario_scheduler(
            "scenario_a_fast_gpu_cpu_bottleneck.json", "capacity_relative"
        )
        job = next(j for j in scenario.jobs if j.job_id == "target")
        config = next(c for c in job.configs if c.config_id == "fast-1g")
        query = oracle.query(job, config, 2, 62.5, (2, 62.5))
        self.assertAlmostEqual(query.rho, 1.0)

    def test_illegal_config_is_rejected(self):
        scenario, oracle, _ = self.scenario_scheduler("scenario_a_fast_gpu_cpu_bottleneck.json")
        job = next(j for j in scenario.jobs if j.job_id == "target")
        fake = GpuConfig("fake", "fast", 1, 1, 1.0)
        with self.assertRaises(ValueError):
            oracle.query(job, fake, 3, 62.5, (2, 62.5))

    def test_mapping_modes_are_deterministic_and_explicit(self):
        jobs = ["j0", "j1", "j2", "j3"]
        templates = ["res18_1.py", "m5_1.py"]
        self.assertEqual(
            assign_templates_fixed_hash(jobs, templates, 17),
            assign_templates_fixed_hash(jobs, templates, 17),
        )
        mapping = assign_templates_category_mix(
            jobs,
            {"cpu": ["res18_1.py"], "memory": ["m5_1.py"]},
            {"cpu": 0.5, "memory": 0.5},
        )
        self.assertEqual(len(mapping), 4)
        self.assertEqual(set(mapping.values()), set(templates))

    def test_multinode_patterns_require_consistent_full_nodes(self):
        nodes = [
            NodeSpec("n0", "x", 2, 6, 125),
            NodeSpec("n1", "x", 2, 6, 125),
            NodeSpec("n2", "x", 2, 4, 125),
        ]
        config = GpuConfig("x-4g", "x", 4, 2, 2.0)
        placements = enumerate_placements(config, nodes)
        self.assertEqual([p.node_ids for p in placements], [("n0", "n1")])
        self.assertTrue(all(g == 2 for p in placements for _, g in p.node_gpus))

    def test_all_three_policies_match_independent_exhaustive_solver(self):
        for path in sorted(SCENARIOS.glob("scenario_*.json")):
            _, _, scheduler = self.scenario_scheduler(path.name)
            b1 = scheduler.run_b1()
            b4 = scheduler.run_b4(b1)
            b3 = scheduler.run_b3()
            for result in (b1, b4, b3):
                self.assertEqual(result.status, "OPTIMAL")
                self.assertTrue(result.exact_match)
                self.assertTrue(result.audit["valid"])

    def test_b4_never_changes_b1_gpu_config(self):
        for path in sorted(SCENARIOS.glob("scenario_*.json")):
            _, _, scheduler = self.scenario_scheduler(path.name)
            b1 = scheduler.run_b1()
            b4 = scheduler.run_b4(b1)
            for job_id, column in b4.selected.items():
                self.assertEqual(column.config.config_id, b1.selected[job_id].config.config_id)

    def test_scenario_a_joint_changes_gpu_type(self):
        _, _, scheduler = self.scenario_scheduler("scenario_a_fast_gpu_cpu_bottleneck.json")
        b1 = scheduler.run_b1()
        b4 = scheduler.run_b4(b1)
        b3 = scheduler.run_b3()
        self.assertEqual(b1.selected["target"].config.config_id, "fast-1g")
        self.assertEqual(b4.selected["target"].config.config_id, "fast-1g")
        self.assertEqual(b3.selected["target"].config.config_id, "slow-1g")
        self.assertGreater(b3.actual_total_goodput, b4.actual_total_goodput)

    def test_scenario_b_joint_scales_down_gpu_count(self):
        _, _, scheduler = self.scenario_scheduler("scenario_b_elastic_gpu_cpu_limit.json")
        b1 = scheduler.run_b1()
        b4 = scheduler.run_b4(b1)
        b3 = scheduler.run_b3()
        self.assertEqual(b1.selected["elastic-target"].config.gpus, 2)
        self.assertEqual(b4.selected["elastic-target"].config.gpus, 2)
        self.assertEqual(b3.selected["elastic-target"].config.gpus, 1)
        self.assertGreater(b3.actual_total_goodput, b4.actual_total_goodput)

    def test_scenario_c_b4_repairs_placement_and_matches_b3(self):
        _, _, scheduler = self.scenario_scheduler("scenario_c_multiresource_placement.json")
        b1 = scheduler.run_b1()
        b4 = scheduler.run_b4(b1)
        b3 = scheduler.run_b3()
        self.assertEqual(
            b1.selected["cpu-sensitive"].placement.node_ids,
            b1.selected["memory-sensitive"].placement.node_ids,
        )
        self.assertNotEqual(
            b4.selected["cpu-sensitive"].placement.node_ids,
            b4.selected["memory-sensitive"].placement.node_ids,
        )
        self.assertAlmostEqual(b4.actual_total_goodput, b3.actual_total_goodput)
        self.assertGreater(b4.actual_total_goodput, b1.actual_total_goodput)

    def test_job_input_contains_no_future_information(self):
        fields = set(JobSpec.__dataclass_fields__)
        forbidden = {"completion_time", "future_arrivals", "true_remaining_time", "future_runtime"}
        self.assertFalse(fields & forbidden)

    def test_checkpoint_lock_limits_config_and_nodes(self):
        scenario, oracle, _ = self.scenario_scheduler("scenario_a_fast_gpu_cpu_bottleneck.json")
        target = next(j for j in scenario.jobs if j.job_id == "target")
        locked = JobSpec(
            job_id=target.job_id,
            template_file=target.template_file,
            configs=target.configs,
            checkpoint_locked=True,
            current_config_id="slow-1g",
            current_nodes=("slow-0",),
            current_placement=(("slow-0", 1),),
            current_cpu_per_gpu=3.0,
            current_ram_gb_per_gpu=62.5,
        )
        scheduler = JointScheduler(
            scenario.nodes,
            [locked, next(j for j in scenario.jobs if j.job_id != "target")],
            oracle,
            scenario.aux_cpu_points,
            scenario.aux_ram_points,
        )
        columns = [c for c in scheduler.build_joint_columns() if c.job_id == "target"]
        self.assertTrue(columns)
        self.assertTrue(all(c.config.config_id == "slow-1g" for c in columns))
        self.assertTrue(all(c.placement.node_ids == ("slow-0",) for c in columns))


if __name__ == "__main__":
    unittest.main()
