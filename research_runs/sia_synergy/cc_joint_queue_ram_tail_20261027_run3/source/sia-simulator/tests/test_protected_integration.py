import contextlib,io,unittest
from test_exchange_integration import ExchangeIntegration
from joint_prototype.protected_admission import install
from joint_prototype.execution_audit import audit_execution

class ProtectedIntegration(unittest.TestCase):
 def test_unseeded_checkpoint_training_and_real_next_decisions(self):
  with contextlib.redirect_stdout(io.StringIO()):
   c=install(ExchangeIntegration().build());productive=0.;locks=0
   for _ in range(12):
    c.step(60);d=c.policy.last_decision;selected=d['result']['selected'];a=d['result']['audit']
    self.assertTrue(a['valid']);self.assertFalse(d['shared_contract']['admission_unchanged'])
    self.assertTrue(set(a['required_jobs'])<=set(selected))
    self.assertEqual(set(a['service_commitments']),set(selected))
    for e in c.joint_round_logs[-1]['execution']:
     self.assertTrue(audit_execution(e)['valid']);productive+=e['training_seconds'];locks+=e['checkpoint_delay_consumed']>0
   self.assertGreater(productive,0);self.assertGreater(locks,0)

if __name__=='__main__':unittest.main()
