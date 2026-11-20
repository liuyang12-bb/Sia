import unittest,copy
from event_gate import qualify
class GateTests(unittest.TestCase):
 def setUp(self):
  self.s=dict(time=60,event=dict(structural=True,changed=['a'],clipped_changed=[],fo_objective=1,full_objective=2),selected={m:{'a':{'config':{'config_id':c}}} for m,c in [('FO','aws:1:2'),('FULL','aws:1:1')]})
  self.rows=[dict(time=t,same_service_set=True,fo=dict(configs={'a':'aws:1:2'},progress={'a':1},geometry_event_reasons=['FIRST_VALID_CURRENT_MODEL'] if t==60 else []),full=dict(configs={'a':'aws:1:1'},progress={'a':2},geometry_event_reasons=[])) for t in range(60,481,60)]
 def q(self):return qualify(self.s,self.rows,'a',['FIRST_VALID_CURRENT_MODEL'])
 def test_event_then_persistent_execution_needs_no_repeated_event(self):self.assertIsNotNone(self.q())
 def test_no_legitimate_event_rejected(self):self.rows[0]['fo']['geometry_event_reasons']=[];self.assertIsNone(self.q())
 def test_checkpoint_or_zero_progress_not_productive(self):
  for r in self.rows:r['full']['progress']['a']=0
  self.assertIsNone(self.q())
 def test_transient_or_changed_geometry_rejected(self):self.rows[3]['full']['configs']['a']='rtx:1:1';self.assertIsNone(self.q())
 def test_key_clipped_rejected_other_donor_disclosed(self):
  self.s['event']['clipped_changed']=[['a','FO']];self.assertIsNone(self.q())
  self.s['event']['clipped_changed']=[['peer','FO']];self.assertEqual(self.q()['other_changed_point_clipping'],[['peer','FO']])
if __name__=='__main__':unittest.main(verbosity=2)
