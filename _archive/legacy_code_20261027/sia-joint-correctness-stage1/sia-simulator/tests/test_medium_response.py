import json
import math
import unittest
from pathlib import Path

from joint_prototype.joint_profile import GpuConfig, JobSpec, JointProfileOracle


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT.parents[1] / "research_sources" / "Synergy" / "simulator" / "models"
CONFIG = ROOT / "joint_prototype" / "cpu_contention_20job_config.json"


class MediumResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))
        response = cls.config["response"]
        cls.oracle = JointProfileOracle(
            MODELS,
            normalization_mode=response["normalization_mode"],
            fixed_reference=(
                response["fixed_reference"]["cpu_per_gpu"],
                response["fixed_reference"]["ram_gb_per_gpu"],
            ),
            neutral_response=False,
            response_exponent=response["response_settings"]["medium"]["exponent"],
        )

    def query(self, template, cpu, ram):
        config = GpuConfig("aws:1:1", "aws", 1, 1, 1.0)
        job = JobSpec("job", template, (config,))
        return self.oracle.query(job, config, cpu, ram, (1.0, 125.0))

    def test_existing_medium_definition_is_unique(self):
        response = self.config["response"]
        self.assertEqual(response["normalization_mode"], "fixed_reference")
        self.assertEqual(response["fixed_reference"], {"cpu_per_gpu": 3.0, "ram_gb_per_gpu": 62.5})
        self.assertEqual(response["response_settings"]["medium"]["exponent"], 0.5)

    def test_medium_uses_sqrt_of_raw_ratio_and_fixed_reference_is_one(self):
        low = self.query("alexnet_1.py", 1.0, 20.83)
        reference = self.query("alexnet_1.py", 3.0, 62.5)
        self.assertAlmostEqual(low.rho, math.sqrt(low.raw_rho), places=12)
        self.assertAlmostEqual(reference.raw_rho, 1.0, places=12)
        self.assertAlmostEqual(reference.rho, 1.0, places=12)

    def test_templates_have_distinct_medium_responses(self):
        templates = sorted(set(self.config["response"]["application_template_rule"].values()))
        responses = [round(self.query(template, 1.0, 20.83).rho, 12) for template in templates]
        self.assertGreater(len(set(responses)), 1)
        self.assertTrue(all(value > 0 for value in responses))


if __name__ == "__main__":
    unittest.main()
