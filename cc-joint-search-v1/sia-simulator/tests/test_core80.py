"""Scale/representation contracts, retaining all80 queue entries and candidates."""
import contextlib,copy,dataclasses,io,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from core80 import Core80Admission,build
from scale_storage import storage_context,sparse_constraints
from test_adaptive_service import spec,cfg,current,admission,scheduler
from test_clean_ablation import state,cs,controller
from joint_prototype.joint_profile import NodeSpec

class Core80Tests(unittest.TestCase):
 def test_real80_installer_all_methods_preserves_entire_input(self):
  from core80 import METHODS
  for method in METHODS:
   with contextlib.redirect_stdout(io.StringIO()):c=build(method,seed=False)
   self.assertEqual(len(c.jobs),80);self.assertEqual(len({j.name for j in c.jobs}),80)
   self.assertTrue(all(j.progress==0. for j in c.jobs))
   if not method.startswith('SIA'):self.assertEqual(c.policy.liveness.MAX_ACTIVE_JOBS,80)
 def test_real80_optimize_policy_checkpoint_training_no_seed(self):
  import pandas as pd
  data=pd.read_csv(ROOT/'inputs/core80_workload.csv');data.time=0
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'all80.csv';data.to_csv(path,index=False)
   with contextlib.redirect_stdout(io.StringIO()):
    c=build('A11','neutral',workload_path=path,seed=False)
    for _ in range(4):
     c.step(60);self.assertTrue(c.policy.last_decision['result']['audit']['valid'])
     self.assertEqual(len(c.policy.liveness.seen),80)
     self.assertEqual(set(c.policy.liveness.ready)|set(c.policy.liveness.services),{j.name for j in c.jobs if j.completion_time is None})
    self.assertGreater(sum(j.progress for j in c.jobs),0.)
    self.assertTrue(any(j.rescale_time>0 or j.execution_time>0 for j in c.jobs))
 def make(self,js):
  x=controller(js);x.__class__=Core80Admission;x.resource_factor='PROP';return x
 def test_80_fifo_queue_no_truncation(self):
  js=[spec(f'job-{i:03}',index=i) for i in range(80)]
  s=state(js,[NodeSpec('aws:0','aws',4,12.,250.)]);x=self.make(js)
  r=x.decide(s,cs(s,{}),0.,{})
  self.assertEqual(set(r.selected),{j.job_id for j in js[:4]})
  self.assertEqual(x.ready,[j.job_id for j in js[4:]])
  self.assertEqual(len(x.seen),80);self.assertTrue(r.audit['valid'])
 def test_81_rejects_without_partial_queue(self):
  js=[spec(str(i),index=i) for i in range(81)];s=state(js);x=self.make(js);before=copy.deepcopy(x.__dict__)
  with self.assertRaises(RuntimeError):x.decide(s,cs(s,{}),0.,{})
  self.assertEqual(before,x.__dict__)
 def test_80_completion_next_head_and_full_lock(self):
  js=[spec(f'job-{i:03}',index=i) for i in range(80)]
  s=state(js,[NodeSpec('aws:0','aws',2,6.,125.)]);x=self.make(js)
  first=x.decide(s,cs(s,{}),0.,{})
  remaining=[current(js[1],first.selected,locked=True)]+js[2:]
  s=state(remaining,s.nodes);out=x.decide(s,cs(s,{}),60.,{})
  self.assertEqual(set(out.selected),{'job-001','job-002'})
  self.assertEqual(out.selected['job-001'].placement,first.selected['job-001'].placement)
  self.assertEqual(out.selected['job-001'].query.cpu_per_gpu,first.selected['job-001'].query.cpu_per_gpu)
  self.assertEqual(len(x.ready),77)
 def test_sparse_dense_matrix_exact_and_upper_not_mutated(self):
  js=[spec('a',configs=[cfg('aws'),cfg('aws',2)]),spec('b')]
  s=state(js,[NodeSpec('aws:0','aws',4,12.,250.)]);columns=s.build_joint_columns()
  dense,u=s._constraint_matrix(columns,('gpu','cpu','ram'));sp,v=sparse_constraints(s,columns,('gpu','cpu','ram'))
  np.testing.assert_array_equal(dense,sp.toarray());np.testing.assert_array_equal(u,v)
  v[0]=0.;_,fresh=sparse_constraints(s,columns,('gpu','cpu','ram'));self.assertEqual(fresh[0],1.)
 def test_response_storage_cache_current_raw_and_job_identity(self):
  js=[spec('a'),spec('b')];s=scheduler(js,medium=True);config=js[0].configs[0]
  expected=s.oracle.query(js[0],config,3.,62.5,(3.,62.5))
  with storage_context(s.oracle):
   actual=s.oracle.query(js[0],config,3.,62.5,(3.,62.5))
   self.assertEqual(dataclasses.asdict(expected),dataclasses.asdict(actual))
   changed=dataclasses.replace(config,sia_goodput=7.);other=dataclasses.replace(js[1],configs=(changed,))
   q=s.oracle.query(other,changed,3.,62.5,(3.,62.5));self.assertEqual(q.job_id,'b');self.assertEqual(q.raw_sia_goodput,7.)
   self.assertEqual(q.effective_goodput,7.*q.rho);self.assertFalse(hasattr(q,'__dict__'))
   columns=s.build_joint_columns();self.assertEqual(dataclasses.asdict(columns[0]),dataclasses.asdict(copy.deepcopy(columns[0])))
 def test_sparse_dense_full_lexicographic_same_selection(self):
  js=[spec('a'),spec('b',index=1)];s=state(js,[NodeSpec('aws:0','aws',2,6.,125.)]);columns=cs(s,{'a':.25,'b':.3})
  plain=self.make(js).decide(s,columns,0.,{})
  with storage_context(s.oracle):
   stored=cs(s,{'a':.25,'b':.3});fast=self.make(js).decide(s,stored,0.,{})
  self.assertEqual({j:c.column_id for j,c in plain.selected.items()},{j:c.column_id for j,c in fast.selected.items()})
  self.assertEqual(plain.scheduling_objective,fast.scheduling_objective)
 def test_cpu_mem_postscore_quota_contract(self):
  js=[spec('a')];s=state(js,[NodeSpec('aws:0','aws',2,6.,125.)]);s.enforce_floor=False;columns=s.build_joint_columns()
  for factor in ('CPU','MEM'):
   x=self.make(js);x.resource_factor=factor;x.clean_aux='opt';r=x.decide(s,columns,0.,{})
   for c in r.selected.values():
    if factor=='CPU':self.assertEqual(c.query.ram_gb_per_gpu,c.proportional_ram_per_gpu)
    else:self.assertEqual(c.query.cpu_per_gpu,c.proportional_cpu_per_gpu)
 def test_efficient_real_optimize_policy_matches_original_storage(self):
  # Real unseeded BERT: initial allocation, checkpoint, training, next schedule.
  from test_exchange_integration import ExchangeIntegration
  from joint_prototype.completion_continuity import install as cc
  from joint_prototype.tail_recovery import install as tail
  from joint_prototype.clean_ablation import install
  from core80 import EfficientTraceMixin
  tracks=[]
  for efficient in (False,True):
   with contextlib.redirect_stdout(io.StringIO()):
    c=cc(ExchangeIntegration().build());tail(c);install(c,'B',1,1,'joint','opt')
    if efficient:c.__class__=type('TestEfficient',(EfficientTraceMixin,c.__class__),{})
    trace=[]
    for _ in range(12):
     c.step(60);trace.append((copy.deepcopy(c.allocations),[(j.progress,j.epoch,j.rescale_time,j.num_restarts,j.completion_time) for j in c.jobs],copy.deepcopy(c.policy.current_resources),[dataclasses.asdict(x) for x in c.policy.exchange_columns]))
    self.assertGreater(c.jobs[0].progress,0.);tracks.append(trace)
  self.assertEqual(tracks[0],tracks[1])
 def test_cold_filter_pushdown_exact_complete_scored_domain(self):
  import pandas as pd
  data=pd.read_csv(ROOT/'inputs/core80_workload.csv').iloc[[0,3,4]].copy();data.time=0
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'three.csv';data.to_csv(path,index=False)
   for method in ('FULL','FO','SIA_AUX'):
    for seed in (False,True):
     tracks=[]
     for efficient in (False,True):
      with contextlib.redirect_stdout(io.StringIO()):
       c=build(method,workload_path=path,topology={'aws':2,'dgx-ext':2},seed=seed,efficient=efficient);trace=[]
       for _ in range(4):
        c.step(60);columns=getattr(c.policy,'exchange_columns',getattr(c.policy,'actual_completion_columns',[]))
        trace.append((copy.deepcopy(c.allocations),[(j.progress,j.num_restarts) for j in c.jobs],[dataclasses.asdict(x) for x in columns]))
       tracks.append(trace)
     self.assertEqual(tracks[0],tracks[1],(method,seed))
 def test_sia_neutral_medium_real_bounded_exact_regression(self):
  import pandas as pd
  data=pd.read_csv(ROOT/'inputs/core80_workload.csv').iloc[[0,4]].copy();data['time']=0
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'two.csv';data.to_csv(p,index=False);tracks=[]
   for response in ('neutral','medium'):
    with contextlib.redirect_stdout(io.StringIO()):
     c=build('SIA_P',response,workload_path=p,topology={'aws':2,'dgx-ext':2});trace=[]
     for _ in range(12):
      c.step(60);trace.append((copy.deepcopy(c.allocations),[(j.progress,j.epoch,j.num_restarts,j.completion_time) for j in c.jobs]))
      self.assertTrue(c.policy.last_decision['strict_full_intent'])
      self.assertTrue(all(x['rho']==1. for x in c.policy.current_resources.values()))
     tracks.append(trace)
   self.assertEqual(tracks[0],tracks[1])

if __name__=='__main__':unittest.main(verbosity=2)
