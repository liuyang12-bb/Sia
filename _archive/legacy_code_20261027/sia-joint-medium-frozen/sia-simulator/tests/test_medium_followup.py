import json
import unittest
from pathlib import Path

from freeze_medium_w10_late_6000_followup import differences


ROOT = Path(__file__).resolve().parents[1]


class MediumFollowupTests(unittest.TestCase):
    def test_preregistration_changes_only_observation_window(self):
        prereg = json.loads(
            (ROOT / "joint_prototype/medium_w10_late_6000_followup_preregistration.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(prereg["original_max_rounds"], 5000)
        self.assertEqual(prereg["followup_max_rounds"], 6000)
        self.assertEqual(
            prereg["allowed_config_difference"],
            {"path": "simulation.max_rounds", "old": 5000, "new": 6000},
        )
        self.assertEqual(prereg["response_exponent_alpha"], 0.5)
        self.assertEqual(prereg["minimum_available_memory_gib"], 2.0)

    def test_structural_diff_rejects_hidden_changes(self):
        original = {"simulation": {"max_rounds": 5000, "interval_seconds": 60}}
        followup = {"simulation": {"max_rounds": 6000, "interval_seconds": 60}}
        self.assertEqual(
            differences(original, followup),
            [{"path": "simulation.max_rounds", "old": 5000, "new": 6000}],
        )
        followup["simulation"]["interval_seconds"] = 30
        self.assertEqual(len(differences(original, followup)), 2)


if __name__ == "__main__":
    unittest.main()
