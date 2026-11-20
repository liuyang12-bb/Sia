"""Real unseeded Job/Cluster chains, fixed <=10 rounds, <=1 job."""
import unittest,json
from pathlib import Path
import pandas as pd
from joint_prototype.full160 import make_cluster
from joint_prototype.adaptive_probe import install
from joint_prototype.execution_audit import audit_execution
ROOT=Path(__file__).resolve().parents[3]
P=ROOT/'research_runs/sia_synergy/small_mechanism_20261025/preregistration.json'

class AdaptiveIntegration(unittest.TestCase):
 def test_real_unseeded_bert_checkpoint_training_next_decision(self):
  p=json.loads(P.read_text());c=install(make_cluster(json.loads(Path(p['config']).read_text()),pd.DataFrame([dict(name='bert-mid',time=17,application='bert',num_replicas=1,batch_size=384)]),p['registry'],p['models'],'joint_o','medium','main',False));productive=False
  for _ in range(10):
   if c.all_complete():break
   c.step(60);r=c.joint_round_logs[-1]
   for e in r['execution']:
    self.assertTrue(audit_execution(e)['valid']);productive|=e['training_seconds']>0
   for ledger in c.policy.last_decision['result']['audit'].get('service_delay_ledgers',{}).values():self.assertLessEqual(ledger['delay_charged'],250)
   self.assertTrue(c.policy.last_decision['result']['audit']['valid'])
  self.assertTrue(productive);self.assertGreater(c.jobs[0].progress,0)
 def test_actual_delay_source_cifar_is_50_not_fairness_proxy_30(self):
  p=json.loads(P.read_text());c=install(make_cluster(json.loads(Path(p['config']).read_text()),pd.DataFrame([dict(name='cifar10-cost',time=0,application='cifar10',num_replicas=1,batch_size=128)]),p['registry'],p['models'],'joint_o','neutral','main',False));self.assertEqual(set(c.policy.liveness.delay_costs['cifar10-cost'].values()),{50.})
  c.step(60);a=c.policy.last_decision['result']['audit']['service_budget']['cifar10-cost'];self.assertEqual(a['charged_this_round'],50.)
if __name__=='__main__':unittest.main()
