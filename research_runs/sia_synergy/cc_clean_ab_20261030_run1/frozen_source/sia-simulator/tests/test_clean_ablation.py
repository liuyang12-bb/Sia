import copy,importlib,unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
from test_adaptive_service import spec,cfg,current,scheduler,columns,admission
from joint_prototype.joint_profile import NodeSpec
from joint_prototype.neutral_liveness import geometry

def module():return importlib.import_module('joint_prototype.clean_ablation')
def controller(js,order=1,continuity=1,gpu='joint',aux='prop'):
 x=module().CleanAdmission(300.,250.,60.);b=admission(js,50.)
 x.delay_costs=b.delay_costs;x.profile_sources=b.profile_sources
 x.factors=(0,order,continuity);x.clean_gpu=gpu;x.clean_aux=aux
 return x
def state(js,nodes=None):
 s=scheduler(js,nodes or [NodeSpec('aws:0','aws',1,3.,62.5)])
 s.current_known_configs={(j.job_id,c.config_id) for j in js for c in j.configs}
 s.current_completion_scores={};return s
def cs(s,w):return [replace(c,scheduling_utility=w.get(c.job_id,1.)) for c in columns(s)
 if c.query.cpu_per_gpu==3. and c.query.ram_gb_per_gpu==62.5]

class CleanABTests(unittest.TestCase):
 def test_fifo_only_changes_head_requirement(self):
  js=[spec('a'),spec('b',arrival=30,index=1)];s=state(js)
  for order,target in [(0,'b'),(1,'a')]:
   x=controller(js,order);r=x.decide(s,cs(s,{'a':.1,'b':1.}),60.,{})
   self.assertEqual(set(r.selected),{target});self.assertTrue(r.audit['valid'])
 def test_negative_nonempty_no_bonus(self):
  js=[spec('a'),spec('b',index=1)];s=state(js);x=controller(js,0)
  r=x.decide(s,cs(s,{'a':-2.,'b':-1.}),0.,{})
  self.assertEqual(set(r.selected),{'b'});self.assertEqual(r.scheduling_objective,-1.)
 def test_expiry_only_changes_commitment_not_geometry_eligibility(self):
  js=[spec('a',configs=[cfg('aws'),cfg('dgx-ext',rate=3.)])]
  records=[]
  for keep in (0,1):
   x=controller(js,1,keep);s=state(js,[NodeSpec('aws:0','aws',1,3.,62.5),NodeSpec('dgx-ext:0','dgx-ext',1,3.,62.5)])
   first=x.decide(s,[c for c in cs(s,{}) if c.config.gpu_type=='aws'],0.,{})
   s=state([current(js[0],first.selected)],s.nodes)
   x.previous_active={'a'};x.previous_known_configs=set(s.current_known_configs)
   r=x.decide(s,cs(s,{}),360.,{'a':310.})
   records.append(r.audit['shared_eligibility']['a']['eligible_geometries'])
  self.assertEqual(records[0],records[1])
 def test_checkpoint_preserved_all_four_A_cells(self):
  js=[spec('a'),spec('b',arrival=60,index=1)]
  for order in (0,1):
   for keep in (0,1):
    x=controller(js,order,keep);s=state(js[:1]);r=x.decide(s,cs(s,{}),0.,{})
    s=state([current(js[0],r.selected,locked=True),js[1]])
    out=x.decide(s,cs(s,{'a':-100.,'b':100.}),60.,{})
    self.assertEqual(geometry(r.selected['a']),geometry(out.selected['a']))
 def test_continuity_service_requeue_effect(self):
  js=[spec('a'),spec('b',arrival=60,index=1)]
  for keep,target in [(0,'b'),(1,'a')]:
   x=controller(js,1,keep);s=state(js[:1]);r=x.decide(s,cs(s,{}),0.,{})
   s=state([current(js[0],r.selected),js[1]])
   out=x.decide(s,cs(s,{'a':.1,'b':1.}),360.,{'a':310.})
   self.assertEqual(set(out.selected),{target})
 def test_lifetime_ledger_and_non_destructive_expiry(self):
  js=[spec('a')];x=controller(js,1,0);s=state(js);r=x.decide(s,cs(s,{}),0.,{})
  x.services['a']['delay_charged']=250.;s=state([current(js[0],r.selected)])
  out=x.decide(s,cs(s,{}),360.,{'a':310.})
  self.assertIn('a',out.selected);self.assertIn('a',out.audit['budget_safety_holds'])
 def test_solver_failure_atomic(self):
  js=[spec('a')];s=state(js);x=controller(js,0,0);old=copy.deepcopy(x.__dict__)
  with patch('joint_prototype.adaptive_service.milp',return_value=SimpleNamespace(status=1,x=None,success=False,message='injected')):
   with self.assertRaisesRegex(RuntimeError,'SOLVER_UNDECIDED'):x.decide(s,cs(s,{}),0.,{})
  self.assertEqual(old,x.__dict__)
 def test_serial_freezes_config_but_allows_aux(self):
  js=[spec('a',configs=[cfg('aws'),cfg('dgx-ext',rate=2.)])]
  nodes=[NodeSpec('aws:0','aws',1,6.,125.),NodeSpec('dgx-ext:0','dgx-ext',1,6.,125.)]
  s=state(js,nodes);s.enforce_floor=False;allc=columns(s)
  # Proportional dgx wins, but a nonproportional aws point would win globally.
  scored=[replace(c,scheduling_utility=(100. if c.config.gpu_type=='aws' and c.query.cpu_per_gpu==3. else 3. if c.config.gpu_type=='dgx-ext' else 1.)) for c in allc]
  x=controller(js,gpu='fixed',aux='opt');r=x.decide(s,scored,0.,{})
  self.assertEqual(r.selected['a'].config.gpu_type,'dgx-ext')
  self.assertTrue(r.audit['serial_proposals']);self.assertTrue(r.audit['valid'])
  y=controller(js,gpu='joint',aux='opt');out=y.decide(s,scored,0.,{})
  self.assertEqual(out.selected['a'].config.gpu_type,'aws')
 def test_prop_contract_is_exact_node_proportion(self):
  js=[spec('a')];s=state(js,[NodeSpec('aws:0','aws',1,6.,125.)]);allc=columns(s)
  for gpu in ('fixed','joint'):
   x=controller(js,gpu=gpu,aux='prop');out=x.decide(s,allc,0.,{})
   c=out.selected['a'];self.assertEqual((c.query.cpu_per_gpu,c.query.ram_gb_per_gpu),(6.,125.))
 def test_contract_only_factor_fields_differ(self):
  m=module();contracts=[m.contract('B',1,1,g,a) for g in ('fixed','joint') for a in ('prop','opt')]
  clean=[{k:v for k,v in c.items() if k not in ('gpu_factor','aux_factor')} for c in contracts]
  self.assertTrue(all(c==clean[0] for c in clean))
 def test_real_unseeded_bert_full_chain_all_factor_cells(self):
  import contextlib,io
  from test_exchange_integration import ExchangeIntegration
  from joint_prototype.completion_continuity import install as base
  from joint_prototype.tail_recovery import install as tail
  from joint_prototype.execution_audit import audit_execution
  cases=[('A',a,b,'joint','prop') for a in (0,1) for b in (0,1)]
  cases += [('B',1,1,g,q) for g in ('fixed','joint') for q in ('prop','opt')]
  for group,a,b,g,q in cases:
   with contextlib.redirect_stdout(io.StringIO()):
    c=base(ExchangeIntegration().build(),proportional=q=='prop');tail(c)
    module().install(c,group,a,b,g,q)
    trained=0.;checkpoint=0.;known=False
    for _ in range(12):
     c.step(60);d=c.policy.last_decision
     self.assertTrue(d['result']['audit']['valid'])
     for e in c.joint_round_logs[-1]['execution']:
      self.assertTrue(audit_execution(e)['valid']);trained+=e['training_seconds'];checkpoint+=e['checkpoint_delay_consumed']
     known |= any(v['raw_rate'] is not None for v in d['continuity_scores'])
    self.assertTrue(known);self.assertGreater(trained,0.);self.assertGreater(checkpoint,0.)
    self.assertGreater(c.jobs[0].progress,0.)
    self.assertEqual(d['clean_contract'],module().contract(group,a,b,g,q))

if __name__=='__main__':unittest.main(verbosity=2)
