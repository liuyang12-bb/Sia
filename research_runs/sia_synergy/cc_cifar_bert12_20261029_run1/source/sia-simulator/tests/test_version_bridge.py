import json,unittest
from dataclasses import replace
from pathlib import Path
import pandas as pd
from test_adaptive_service import spec,cfg,current,scheduler,columns
from joint_prototype.version_bridge import BridgeScheduler,FreeB3,install
from joint_prototype.joint_integration import JointSiaPolicy
from joint_prototype.full160 import make_cluster
from joint_prototype.execution_audit import audit_execution

def bridge(s):
    s.__class__=BridgeScheduler
    return s

class VersionBridgeTests(unittest.TestCase):
    def test_free_core_is_literal_legacy_optimize_and_no_l2(self):
        self.assertIs(FreeB3.optimize.__wrapped__ if hasattr(FreeB3.optimize,'__wrapped__') else FreeB3.__mro__[2].optimize,JointSiaPolicy.optimize)

    def test_negative_utility_remains_empty_not_hidden_liveness_fix(self):
        s=bridge(scheduler([spec()]))
        r=s.solve('B3_FREE',columns(s,negative=True),('gpu','cpu','ram'))
        self.assertEqual(r.selected,{})
        self.assertTrue(r.audit['valid'])

    def test_negative_true_lock_is_mandatory_full_geometry(self):
        s=bridge(scheduler([spec()]))
        first=s.solve('B3_FREE',columns(s),('gpu','cpu','ram'))
        s=bridge(scheduler([current(s.jobs[0],first.selected,locked=True)]))
        r=s.solve('B3_FREE',columns(s,negative=True),('gpu','cpu','ram'))
        self.assertEqual(r.selected['j'].placement,first.selected['j'].placement)
        self.assertEqual(r.selected['j'].query.cpu_per_gpu,first.selected['j'].query.cpu_per_gpu)

    def test_unlocked_configuration_is_free_to_change(self):
        j=spec(configs=[cfg('aws'),cfg('dgx-ext',rate=10.)]);s=bridge(scheduler([j]))
        first=s.solve('B3_FREE',[c for c in columns(s) if c.config.gpu_type=='aws'],('gpu','cpu','ram'))
        s=bridge(scheduler([current(j,first.selected)]));r=s.solve('B3_FREE',columns(s),('gpu','cpu','ram'))
        self.assertEqual(r.selected['j'].config.gpu_type,'dgx-ext')

    def test_real_unseeded_bert_lock_productivity_next_decision(self):
        root=Path(__file__).resolve().parents[3]
        p=json.loads((root/'research_runs/sia_synergy/small_mechanism_20261025/preregistration.json').read_text())
        c=install(make_cluster(json.loads(Path(p['config']).read_text()),pd.DataFrame([
            dict(name='bert-mid',time=17,application='bert',num_replicas=1,batch_size=384)]),
            p['registry'],p['models'],'sia_p','medium','main',False))
        self.assertFalse(hasattr(c.policy,'liveness'))
        productive=0.;checkpoint=0.
        for _ in range(10):
            c.step(60)
            for e in c.joint_round_logs[-1]['execution']:
                self.assertTrue(audit_execution(e)['valid']);productive+=e['training_seconds'];checkpoint+=e['checkpoint_delay_consumed']
            self.assertTrue(c.policy.last_decision['result']['audit']['valid'])
        self.assertGreater(productive,0);self.assertGreater(checkpoint,0)
