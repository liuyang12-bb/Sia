"""Hydrate ten surviving models from logged observations, NOT 80-job replay."""
import contextlib,copy,json,math,sys,traceback
from pathlib import Path
from types import SimpleNamespace
R=Path(r'D:/科研启航/Sia');T=R/'cc-joint-fo-audit-repair-v2';O=Path(__file__).resolve().parent
E=O/'reconstruction_attempt3'
sys.path[:0]=[str(T),str(T/'sia-simulator')]
import pandas as pd
from core80 import build
from joint_prototype.joint_profile import GpuConfig
from joint_prototype.joint_scheduler import JointColumn,Placement

def reconstruct():
 data=json.loads((O/'failure_observations.json').read_text());last=data['last'][-1];d=last['decision_for_next_interval'];names=data['active']
 w=pd.read_csv(T/'inputs/core80_workload.csv');w=w[w.name.isin(names)]
 w.to_csv(E/'survivors_input.csv',index=False)
 c=build('FO',workload_path=E/'survivors_input.csv');jobs={j.name:j for j in c.jobs}
 raw_errors=[];count=0
 for name,obs in data['observations'].items():
  j=jobs[name]
  for e in obs:
   placement=tuple(e['placement']);changed=(j.current_cluster!=e['cluster'] or j.placement!=placement)
   j.current_cluster=e['cluster'];j.placement=placement
   if changed:j.update_local_bsz(placement)
   app=j.applications[j.current_cluster];g=sum(placement)
   for seg in e['segments']:
    batch=g*j.atomic_bsz*(j.accum_steps+1)
    sq,var=app.get_grad_stats(batch,seg['epoch_before'])
    step,sync=app.get_throughput(placement,j.atomic_bsz)
    gain=(var+sq)/(var/(batch/app.init_batch_size)+sq)
    rate=gain/(step+(step-sync)*j.accum_steps)*j.calibration_factor
    count+=1
    if not math.isclose(rate,seg['raw_rate_before_rho'],rel_tol=1e-10):raw_errors.append(dict(job=name,time=e['time'],rate=rate,logged=seg['raw_rate_before_rho'],batch=batch))
    j.update_params(len(placement),g,j.atomic_bsz,step,sync,sq,var)
    if seg['epoch_after']>seg['epoch_before']:
     assert len(j.epoch_duration)==seg['epoch_before']
     j.epoch_duration.append(round(float((app.get_progress(seg['epoch_after'])-app.get_progress(seg['epoch_before']))/seg['effective_rate'])))
    j.epoch=seg['epoch_after'];j.update_local_bsz(placement)
  state=last['job_end_state'][name];info=d['probe']['job_state'][name];ex=next(e for e in last['execution'] if e['job_id']==name)
  j.progress=state['progress'];j.epoch=state['epoch'];j.current_time=last['interval_end'];j.execution_time=next(s['observed_progress'] for s in d['probe']['specs'] if s['job_id']==name)
  j.num_restarts=state['restarts'];j.num_migrations=state['migrations'];j.used_gpu_seconds=info['used_gpu_seconds'];j.wasted_gpu_seconds=state['wasted_gpu_seconds']
  j.attained_service=next(s['observed_progress'] for s in d['probe']['specs'] if s['job_id']==name)
  # observed_progress in JobSpec is attained GPU-service, not execution time.
  j.execution_time=d['probe']['l2_after']['services'][name]['productive_base']+d['result']['audit']['productive_service'][name]
  col=d['result']['selected'][name];typ=col['config']['gpu_type'];place=tuple(col['placement'].values())
  changed=j.current_cluster!=typ or j.placement!=place
  j.current_cluster=typ;j.placement=place
  if changed:j.update_local_bsz(place)
  j.rescale_time=float(j.applications[typ].rescale_time or 30) if changed else ex['checkpoint_remaining_before_decision']
  j.goodput_multiplier=col['rho'];j.source_row_index=next(s['source_row_index'] for s in d['probe']['specs'] if s['job_id']==name)
  c.allocations[name]=(typ,[int(n.rsplit(':',1)[1]) for n,gpu in col['placement'].items() for _ in range(gpu)])
 c.current_time=last['interval_end'];c.policy.round_index=d['round']
 infos=c.get_job_infos();specs,prov=c.policy._job_specs(infos,c.allocations)
 diffs=[]
 for s in specs:
  expected=next(v for v in d['probe']['specs'] if v['job_id']==s.job_id)
  for cfg in s.configs:
   before=next(v for v in expected['configs'] if v['config_id']==cfg.config_id)
   if not math.isclose(cfg.sia_goodput,before['sia_goodput'],rel_tol=1e-10):diffs.append(dict(job=s.job_id,config=cfg.config_id,actual=cfg.sia_goodput,expected=before['sia_goodput']))
 validation=dict(scope='Offline observed-profile updates only; ten surviving jobs, zero Cluster.step calls',observed_segments=count,raw_rate_mismatches=raw_errors,candidate_model_mismatches=diffs)
 (E/'model_reconstruction_validation.json').write_text(json.dumps(validation,indent=2),encoding='utf8')
 assert not raw_errors and not diffs,validation
 live=c.policy.liveness;live.ready=d['probe']['l2_after']['ready'];live.seen=set(pd.read_csv(T/'inputs/core80_workload.csv').name);live.sequence=80
 live.previous_active=set(names);live.previous_known_configs={(s.job_id,x.config_id) for s in specs for x in s.configs}
 live.services={}
 for name,h in d['probe']['l2_after']['services'].items():
  h=copy.deepcopy(h);old=h['column'];cfg=GpuConfig(**old['config']);p=Placement(tuple(tuple(x) for x in old['placement']));q=c.policy.oracle.query(next(s for s in specs if s.job_id==name),cfg,old['cpu_per_gpu'],old['ram_gb_per_gpu'],(3.,62.5))
  h['column']=JointColumn(old['column_id'],name,cfg,p,q,tuple(tuple(x) for x in old['node_usage']),old['scheduling_utility'],3.,62.5,True,True,old['query_round'],old['candidate_kind']);live.services[name]=h
 live._lifetime_history=copy.deepcopy(live.services)
 c.policy._last_committed_columns={name:h['column'] for name,h in live.services.items()}
 c.policy.current_resources=d['resources'];c.policy.round_index=d['round']+1
 return c

def run():
 c=reconstruct()
 # Exactly ONE bounded60s diagnostic interval for the ten surviving jobs;
 # no arrivals, full workload replay or performance experiment.
 for j in c.jobs:j.step(60)
 c.current_time+=60
 from joint_prototype import bundle_adaptation as bundle
 import types
 ns=dict(bundle.__dict__);oldfile=R/'cc-joint-bounded-correction-v1/sia-simulator/joint_prototype/bundle_adaptation.py'
 exec(compile(oldfile.read_text(encoding='utf8'),str(oldfile),'exec'),ns)
 old_fn=ns['BundleAdmission']._decide;actual_cls=bundle.BundleAdmission
 cell=(lambda:actual_cls).__closure__[0]
 replay_fn=types.FunctionType(old_fn.__code__,old_fn.__globals__,closure=(cell,))
 current_fn=bundle.BundleAdmission._decide;bundle.BundleAdmission._decide=replay_fn
 try:
  c.optimize_policy();result=dict(reproduced=False,time=c.current_time)
 except Exception as e:
  tb=e.__traceback__;target=None
  while tb:
   if tb.tb_frame.f_code.co_filename.endswith('bundle_adaptation.py') and tb.tb_frame.f_code.co_name=='_decide' and 'result' in tb.tb_frame.f_locals:target=tb.tb_frame.f_locals
   tb=tb.tb_next
  if target:
   r=target['result'];baseline=target['baseline'];fixed=r.audit.get('serial_proposals',[])
   result=dict(reproduced=str(e)=='GLOBAL_BUNDLE_OBJECTIVE_REGRESSION',error=repr(e),time=c.current_time,comparison=target['comparison'],
    selected={j:dict(config=x.config.config_id,cpu=x.query.cpu_per_gpu,ram=x.query.ram_gb_per_gpu,utility=x.scheduling_utility) for j,x in r.selected.items()},
    baseline={x.job_id:dict(config=x.config.config_id,cpu=x.query.cpu_per_gpu,ram=x.query.ram_gb_per_gpu,utility=x.scheduling_utility) for x in baseline},serial_proposals=fixed,traceback=traceback.format_exc())
  else:raise
 finally:bundle.BundleAdmission._decide=current_fn
 (E/'failed_decision_reconstruction.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(result.get('comparison'),result.get('reproduced'),flush=True)
 assert result['reproduced']
 c.optimize_policy()
 (E/'repaired_decision.json').write_text(json.dumps(c.policy.last_decision,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x),indent=2),encoding='utf8')
if __name__=='__main__':
 E.mkdir(exist_ok=False)
 with (E/'console.log').open('x',encoding='utf8') as f,contextlib.redirect_stdout(f):run()
