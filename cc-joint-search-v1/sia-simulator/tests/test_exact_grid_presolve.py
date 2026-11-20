import contextlib,itertools,sys,unittest
from pathlib import Path
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from exact_grid_presolve import indices
from engineering_repair import repair_context
from core80 import Core80Admission
from test_clean_ablation import controller,state
from test_adaptive_service import spec,current
from joint_prototype.joint_profile import NodeSpec
from joint_prototype.neutral_liveness import geometry

class ExactGridPresolveTests(unittest.TestCase):
 def make(self):
  js=[spec('a'),spec('b',index=1)];s=state(js,[NodeSpec('aws:0','aws',2,6.,125.)]);s.enforce_floor=False
  points={(1.,20.83),(2.,20.83),(3.,62.5),(4.,62.5),(6.,125.)}
  cols=[replace(c,scheduling_utility=.5 if c.query.cpu_per_gpu>=3 and c.query.ram_gb_per_gpu>=62.5 else .3) for c in s.build_joint_columns() if(c.query.cpu_per_gpu,c.query.ram_gb_per_gpu) in points]
  return js,s,cols
 def test_removed_variables_cannot_change_canonical_optimum(self):
  js,s,cols=self.make();keep=indices(cols);self.assertLess(len(keep),len(cols))
  def exhaustive(domain):
   domains=[[c for c in domain if c.job_id==j.job_id] for j in js]
   feasible=[x for x in itertools.product(*domains) if s.audit(x)['valid']]
   return min(feasible,key=lambda x:(-sum(c.scheduling_utility for c in x),tuple(geometry(c) for c in x)))
  self.assertEqual([geometry(c) for c in exhaustive(cols)],[geometry(c) for c in exhaustive([cols[i] for i in keep])])
 def test_full_native_controller_matches_literal_allocation(self):
  runs=[]
  for fast in (False,True):
   with (repair_context(None) if fast else contextlib.nullcontext()):
    js,s,cols=self.make();x=controller(js,aux='opt');x.__class__=Core80Admission;x.resource_factor='FULL'
    r=x.decide(s,cols,0.,{})
   self.assertTrue(r.audit['valid']);runs.append({j:geometry(c) for j,c in r.selected.items()})
   self.assertFalse(any(k.startswith('_repair_') for k in x.__dict__))
   if fast:self.assertGreater(r.audit['engineering_solver']['variables_removed'],0)
  self.assertEqual(runs[0],runs[1])
 def test_tradeoff_points_are_kept(self):
  js,s,cols=self.make();a=[c for c in cols if c.job_id=='a']
  # Better utility with greater quota is a true tradeoff, never removed.
  a=[replace(c,scheduling_utility=c.query.cpu_per_gpu) for c in a]
  self.assertEqual(len(indices(a)),len(a))
 def test_zero_optional_variable_certificate_preserves_head_and_checkpoint(self):
  js=[spec('a'),spec('b',index=1)];tracks=[]
  for fast in (False,True):
   with (repair_context(None) if fast else contextlib.nullcontext()):
    s=state(js,[NodeSpec('aws:0','aws',2,6.,125.)]);s.enforce_floor=False
    cols=[replace(c,scheduling_utility=0.) for c in s.build_joint_columns()]
    x=controller(js,aux='opt');x.__class__=Core80Admission;x.resource_factor='FULL';x.factors=(1,1,1)
    first=x.decide(s,cols,0.,{});self.assertEqual(set(first.selected),{'a'})
    if fast:self.assertGreater(first.audit['engineering_solver']['zero_variables_removed'],0)
    s=state([current(js[0],first.selected,locked=True),js[1]],s.nodes);s.enforce_floor=False
    cols=[replace(c,scheduling_utility=0.) for c in s.build_joint_columns()]
    second=x.decide(s,cols,60.,{});self.assertEqual(set(second.selected),{'a','b'})
    tracks.append(({j:geometry(c) for j,c in first.selected.items()},{j:geometry(c) for j,c in second.selected.items()}))
  self.assertEqual(tracks[0],tracks[1])

if __name__=='__main__':unittest.main(verbosity=2)
