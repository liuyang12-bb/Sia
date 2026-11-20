import copy
import os
import sys
import unittest
from types import SimpleNamespace


SIMULATOR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SIMULATOR_DIR not in sys.path:
    sys.path.insert(0, SIMULATOR_DIR)

from cluster import Cluster, get_preemption_audit_job_state


def make_job():
    return SimpleNamespace(
        name="cifar10-0",
        epoch=0,
        progress=0.0,
        num_restarts=1,
        current_cluster="rtx",
        placement=(1,),
        atomic_bsz=32,
        accum_steps=0,
        num_stages=1,
        submission_time=0.0,
        start_time=None,
        completion_time=None,
        contention=[1],
        category="adaptive",
        rescale_time=20.0,
        attained_service=40.0,
        used_gpu_seconds=10.0,
        wasted_gpu_seconds=30.0,
        execution_time=10.0,
    )


class PreemptionAuditTest(unittest.TestCase):
    def make_cluster(self, enabled):
        cluster = Cluster.__new__(Cluster)
        cluster.current_time = 60
        cluster.num_nodes = {"rtx": 3}
        cluster.jobs = [make_job()]
        cluster.allocations = {"cifar10-0": ("rtx", (0,))}
        cluster.logs = []
        cluster.log_preemption_audit = enabled
        return cluster

    def test_job_state_distinguishes_restore_from_training(self):
        job = make_job()
        self.assertEqual(get_preemption_audit_job_state(job)["phase"],
                         "checkpoint_restore")
        job.rescale_time = 0
        self.assertEqual(get_preemption_audit_job_state(job)["phase"], "training")

    def test_default_log_schema_is_unchanged(self):
        cluster = self.make_cluster(False)
        cluster.step_log([("event",)], [])
        self.assertNotIn("preemption_audit", cluster.logs[-1])

    def test_enabling_audit_only_appends_a_top_level_field(self):
        plain = self.make_cluster(False)
        plain.step_log([("event",)], [])
        audited = self.make_cluster(True)
        audited.step_log([("event",)], [])

        audited_base = copy.deepcopy(audited.logs[-1])
        audit = audited_base.pop("preemption_audit")
        self.assertEqual(audited_base, plain.logs[-1])
        self.assertEqual(audit["jobs"]["cifar10-0"]["phase"],
                         "checkpoint_restore")
        self.assertEqual(audit["rescale_events"], [("event",)])


if __name__ == "__main__":
    unittest.main()
