import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from test_exchange_integration import ExchangeIntegration
from test_adaptive_service import spec, scheduler, columns
from joint_prototype.joint_profile import NodeSpec
from joint_prototype.protected_admission import ProtectedPolicy
from joint_prototype.service_horizon import ServiceHorizonPolicy, install


class ServiceHorizonTests(unittest.TestCase):
    def test_expired_commitment_scores_as_new_quantum(self):
        job = spec('head')
        base = columns(scheduler([job], [NodeSpec('aws:0', 'aws', 1, 3., 62.5)]))[0]
        policy = ServiceHorizonPolicy.__new__(ServiceHorizonPolicy)
        policy.liveness = SimpleNamespace(quantum=300., interval=60.,
            services={'head': {'productive_base': 0.}})
        policy.productive = {'head': 310.}
        policy.p_fairness = -.5
        policy.lambda_n = 1.1
        score = dict(normalized_rate_before_delay=1., candidate_delay=50., utility=0.)
        with patch.object(ProtectedPolicy, 'factor_columns', return_value=([base], [])):
            policy.exchange_scores = [score]
            result, _ = policy.factor_columns([], [], None)
        self.assertEqual(policy.exchange_scores[0]['service_horizon'], 300.)
        self.assertEqual(len(result), 1)

    def test_full_chain_uses_remaining_service_and_preserves_locks(self):
        with contextlib.redirect_stdout(io.StringIO()):
            cluster = install(ExchangeIntegration().build())
            shortened = 0
            training = 0.
            for _ in range(12):
                cluster.step(60)
                decision = cluster.policy.last_decision
                audit = decision['result']['audit']
                self.assertTrue(audit['valid'])
                self.assertTrue(set(audit['checkpoint_locks']) <= set(decision['result']['selected']))
                for score in decision['net_progress_scores']:
                    horizon = score['service_horizon']
                    self.assertGreaterEqual(horizon, 60.)
                    self.assertLessEqual(horizon, 300.)
                    shortened += horizon < 300.
                training += sum(e['training_seconds'] for e in cluster.joint_round_logs[-1]['execution'])
            self.assertGreater(shortened, 0)
            self.assertGreater(training, 0)


if __name__ == '__main__':
    unittest.main()
