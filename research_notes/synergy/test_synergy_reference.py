"""Read-only checks against the official Synergy reference checkout."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from native_synergy_smoke import install_import_compatibility


ROOT = Path(__file__).resolve().parents[2]
SIMULATOR = ROOT / "research_sources" / "Synergy" / "simulator"
install_import_compatibility()
sys.path.insert(0, str(SIMULATOR))

from jobs.model_zoo import ModelZoo
from resources.cluster import Cluster


class SynergyReferenceTests(unittest.TestCase):
    def test_default_cluster_capacity(self) -> None:
        cluster = Cluster(config_file=str(SIMULATOR / "configs/default_cluster.ini"))
        _, servers, gpus, cpus, memory, _, _ = cluster.size
        self.assertEqual((servers, gpus, cpus, memory), (16, 128, 384, 8000))

    def test_active_model_zoo_matches_reference_code(self) -> None:
        zoo = ModelZoo(image_percent=20, lang_percent=70, speech_percent=10)
        names = {model.model_name for model in zoo.model_zoo.values()}
        self.assertEqual(
            names,
            {
                "alexnet",
                "res18",
                "res50",
                "mobilenet",
                "shufflenet",
                "gnmt",
                "transformer",
                "lstm",
                "m5",
                "deepspeech",
            },
        )
        self.assertNotIn("bert", names)

    def test_active_profiles_have_expected_cpu_memory_grid(self) -> None:
        zoo = ModelZoo(image_percent=20, lang_percent=70, speech_percent=10)
        for model in zoo.model_zoo.values():
            self.assertEqual(model.tput.shape, (9, 5), model.model_name)

    def test_tune_fallback_reduces_to_gpu_proportional_share(self) -> None:
        cluster = Cluster(config_file=str(SIMULATOR / "configs/default_cluster.ini"))

        class FakeJob:
            job_gpu_demand = 1
            synergy_speedup = 2.4

        job = FakeJob()
        changed, demand = cluster._make_fair_share(job, [1, 12, 125, 125, 0])
        self.assertTrue(changed)
        self.assertEqual(demand[:4], [1, 3.0, 62.5, 62.5])
        self.assertEqual(job.synergy_speedup, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
