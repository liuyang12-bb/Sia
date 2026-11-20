"""Synthetic evidence fixtures only, zero simulator/training calls."""
import gzip,json,subprocess,sys,tempfile,unittest
from pathlib import Path
O=Path(__file__).resolve().parent
class GateTests(unittest.TestCase):
 def run_gate(self,version,diverge=False,short=False,zero=False,nonpositive=False):
  with tempfile.TemporaryDirectory(prefix='gate_fixture_',dir=O) as t:
   d=Path(t);src=(O/f'gate_analyze{version}.py').read_text(encoding='utf8');(d/'check.py').write_text(src,encoding='utf8')
   (d/'preregistration.json').write_text(json.dumps(dict(gate_rounds=8,gate_consecutive=6,gate_productive=2)),encoding='utf8')
   def col(config):return dict(config={'config_id':config},source_coordinates={'clipped':False},baseline_source_coordinates={'clipped':False})
   for m in ('FO','FULL'):
    with gzip.open(d/f'gate_{m}.jsonl.gz','wt',encoding='utf8') as f:
     for i in range(1,9):
      selected={'A':col('aws:1:1' if m=='FO' or (short and i>=5) else 'rtx:1:1'),'B':col('aws:1:1')}
      if diverge and m=='FULL' and i>=2:selected['C']=col('rtx:1:1')
      ds={n:dict(result=dict(selected={'A':col('aws:1:1' if n=='FO' else 'rtx:1:1'),'B':col('aws:1:1')},scheduling_objective=1 if n=='FO' or nonpositive else 2)) for n in ('FO','FULL')} if i==1 else {}
      f.write(json.dumps(dict(time=i*60,decisions=ds,actual=dict(result=dict(selected=selected)),execution=[dict(job_id='A',progress_delta=0 if zero else 1)],geometry_events=['ARRIVAL']))+'\n')
   r=subprocess.run([sys.executable,'-B',d/'check.py'],capture_output=True,text=True)
   self.assertEqual(r.returncode,0,r.stderr)
   return json.loads((d/('gate_result_v2.json' if version=='_v2' else 'gate_result.json')).read_text())['passed']
 def test_red_old_rejects_extra_admission_despite_persistent_geometry(self):self.assertFalse(self.run_gate('',diverge=True))
 def test_green_allows_later_additional_admission(self):self.assertTrue(self.run_gate('_v2',diverge=True))
 def test_constant_service_passes(self):self.assertTrue(self.run_gate('_v2'))
 def test_short_gpu_difference_fails(self):self.assertFalse(self.run_gate('_v2',short=True))
 def test_zero_productive_progress_fails(self):self.assertFalse(self.run_gate('_v2',zero=True))
 def test_no_objective_improvement_fails(self):self.assertFalse(self.run_gate('_v2',nonpositive=True))
if __name__=='__main__':unittest.main(verbosity=2)
