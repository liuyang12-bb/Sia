import unittest
from dataclasses import replace
from test_adaptive_service import spec,cfg,scheduler,columns
from joint_prototype.release_evaluation import evaluate_reconfig_plan
from joint_prototype.joint_profile import NodeSpec

class ReconfigRolloutTests(unittest.TestCase):
 def case(self,spent=0,locked=False):
  s=scheduler([spec('a'),spec('b',configs=[cfg('aws'),cfg('aws',2)],index=1)], [NodeSpec('aws:0','aws',2,6.,125.)]);s.enforce_floor=False
  cs=[c for c in columns(s) if c.query.cpu_per_gpu==3. and c.query.ram_gb_per_gpu==62.5]
  initial=[next(c for c in cs if c.job_id==j and c.config.gpus==1) for j in ('a','b')]
  scores={c.column_id:dict(raw_rate=10. if c.config.gpus==2 else 1.,rho=1.,remaining_work=60. if c.job_id=='a' else 6000.,delay=120. if locked and c.job_id=='b' else 0.) for c in cs}
  ctx={j:dict(spent=spent if j=='b' else 0.,limit=250.,frozen=False,productive=0.,costs={'aws':120.},known=['aws:1:1','aws:1:2']) for j in ('a','b')}
  return s,cs,scores,initial,ctx
 def test_release_pays_real_delay_and_uses_only_remaining_work(self):
  s,cs,v,initial,ctx=self.case();r=evaluate_reconfig_plan(s,cs,v,initial,[],60.,ctx)
  self.assertAlmostEqual(r['completion_times']['b'],774.)
  change=next(e for e in r['events'] if e['event']=='SURVIVOR_RECONFIGURE')
  self.assertEqual(change['startup'],120.);self.assertEqual(change['spent_after'],120.)
 def test_lifetime_budget_blocks_illegal_upgrade(self):
  s,cs,v,initial,ctx=self.case(spent=200.);r=evaluate_reconfig_plan(s,cs,v,initial,[],60.,ctx)
  self.assertEqual(r['completion_times']['b'],6000.)
  self.assertFalse(any(e['event']=='SURVIVOR_RECONFIGURE' for e in r['events']))
 def test_checkpoint_delay_cannot_be_skipped(self):
  s,cs,v,initial,ctx=self.case(locked=True);r=evaluate_reconfig_plan(s,cs,v,initial,[],60.,ctx)
  self.assertEqual(r['completion_times']['b'],6120.)
  self.assertFalse(any(e['event']=='SURVIVOR_RECONFIGURE' for e in r['events']))
 def test_used_recovery_freezes_gpu_geometry(self):
  s,cs,v,initial,ctx=self.case();ctx['b']['frozen']=True;r=evaluate_reconfig_plan(s,cs,v,initial,[],60.,ctx)
  self.assertEqual(r['completion_times']['b'],6000.)
if __name__=='__main__':unittest.main()
