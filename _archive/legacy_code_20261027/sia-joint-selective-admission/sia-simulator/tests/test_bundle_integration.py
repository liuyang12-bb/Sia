import unittest,json
from pathlib import Path
import pandas as pd
from joint_prototype.full160 import make_cluster
from joint_prototype.bundle_adaptation import install
from joint_prototype.execution_audit import audit_execution

class BundleIntegration(unittest.TestCase):
    def test_real_unseeded_checkpoint_to_training_and_next_decision(self):
        root=Path(__file__).resolve().parents[3]
        p=json.loads((root/'research_runs/sia_synergy/small_mechanism_20261025/preregistration.json').read_text())
        c=install(make_cluster(json.loads(Path(p['config']).read_text()),
            pd.DataFrame([dict(name='bert-mid',time=17,application='bert',num_replicas=1,batch_size=384)]),
            p['registry'],p['models'],'joint_o','medium','main',False))
        productive=0.;locks=0
        for _ in range(10):
            if c.all_complete():break
            c.step(60);r=c.joint_round_logs[-1];d=c.policy.last_decision
            self.assertTrue(d['result']['audit']['valid'])
            self.assertEqual(d['result']['policy'],'JOINT_G')
            for e in r['execution']:
                self.assertTrue(audit_execution(e)['valid']);productive+=e['training_seconds']
                locks+=int(e['checkpoint_delay_consumed']>0)
            for v in d['result']['audit']['service_delay_ledgers'].values():self.assertLessEqual(v['delay_charged'],250.)
        self.assertGreater(productive,0.);self.assertGreater(locks,0)
        self.assertGreater(c.jobs[0].progress,0.)
