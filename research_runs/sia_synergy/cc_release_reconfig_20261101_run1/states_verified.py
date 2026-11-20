"""Replay logged profile OBSERVATIONS only. Zero Job.step/Cluster.step."""
import copy,dataclasses,gzip,json,math,sys
from pathlib import Path
R=Path(r'D:/科研启航/Sia');T=R/'cc-joint-release-reconfig-v2';O=Path(__file__).resolve().parent
OLD=R/'research_runs/sia_synergy/cc_release_evaluation_20261101_run1'
sys.path[:0]=[str(T),str(T/'sia-simulator')]
from core80 import build
from joint_prototype.completion_time import predict_job,declare_work
from joint_prototype.joint_profile import JobSpec,GpuConfig,NodeSpec
from joint_prototype.neutral_liveness import geometry
from joint_prototype.release_evaluation import evaluate_reconfig_plan
def main():
 c=build('FULL',workload_path=OLD/'inputs/workload.csv',topology={'aws':2,'dgx-ext':2});jobs={j.name:j for j in c.jobs};count=0;results=[]
 with gzip.open(OLD/'cells/NEW_FULL/rounds.jsonl.gz','rt',encoding='utf8') as f:
  for line in f:
   r=json.loads(line)
   for e in r['execution']:
    j=jobs[e['job_id']]
    if not e['resource_allocation']:continue
    placement=tuple(e['placement']);changed=j.current_cluster!=e['cluster'] or j.placement!=placement
    j.current_cluster=e['cluster'];j.placement=placement
    if changed:j.update_local_bsz(placement)
    app=j.applications[j.current_cluster];g=sum(placement)
    for seg in e['training_segments']:
     batch=g*j.atomic_bsz*(j.accum_steps+1);sq,var=app.get_grad_stats(batch,seg['epoch_before']);step,sync=app.get_throughput(placement,j.atomic_bsz)
     raw=(var+sq)/(var/(batch/app.init_batch_size)+sq)/(step+(step-sync)*j.accum_steps)*j.calibration_factor
     assert math.isclose(raw,seg['raw_rate_before_rho'],rel_tol=1e-10),(j.name,seg['start_time'])
     count+=1;j.update_params(len(placement),g,j.atomic_bsz,step,sync,sq,var);j.epoch=seg['epoch_after'];j.update_local_bsz(placement)
    j.progress=e['progress_after'];j.rescale_time=e['checkpoint_remaining_before_decision']
   if r['interval_end'] not in (4800,5820):continue
   d=r['decision_for_next_interval'];specs=[]
   for v0 in d['probe']['specs']:
    v=dict(v0);v['configs']=tuple(GpuConfig(**x) for x in v['configs']);v['current_nodes']=tuple(v['current_nodes']);v['current_placement']=tuple(tuple(x) for x in v['current_placement']);specs.append(JobSpec(**v))
   s=c.policy._scheduler([NodeSpec(t+':0',t,2,6.,125.) for t in ('aws','dgx-ext')],specs);s.locked_columns()
   columns=[];scores={};a=d['result']['audit']
   for col in s.build_joint_columns():
    e=a['shared_eligibility'].get(col.job_id);offer=e.get('bounded_correction_offer',{}).get('column_id') if e else None
    if e and str(geometry(col)) not in e['eligible_geometries'] and col.column_id!=offer:continue
    j=jobs[col.job_id];pred=predict_job(j,col.config);sp=s.job_map[col.job_id]
    delay=j.rescale_time if (sp.current_config_id==col.config.config_id and sp.current_placement==col.placement.node_gpus) else c.policy.completion_delay_costs[col.job_id][col.config.gpu_type]
    work=declare_work(j)['total_progress']-j.progress
    scores[col.column_id]=dict(raw_rate=pred['raw_rate'],remaining_work=work,rho=col.query.rho,delay=delay)
    columns.append(col)
   oldev=a['release_completion'];evaluations={}
   before=d['probe']['l2_before']['services']
   for plan in oldev['plans']:
    bundle=[next(col for col in columns if col.column_id==key) for key in plan['columns'].values()];context={}
    for spec in specs:
     j=spec.job_id;h=before.get(j,{});col=next(col for col in bundle if col.job_id==j);spent=h.get('delay_charged',0.);limit=250.+h.get('tail_extra_budget',0.);frozen=h.get('tail_recovery_used',False)
     if spec.current_config_id!=col.config.config_id or spec.current_placement!=col.placement.node_gpus:
      spent+=c.policy.completion_delay_costs[j][col.config.gpu_type]
      e=a['shared_eligibility'].get(j,{})
      offer=e.get('bounded_correction_offer',{})
      if offer.get('column_id')==col.column_id:limit=250.+max(h.get('tail_extra_budget',0.),offer['extra_budget']);frozen=True
     assert spent<=limit+1e-9
     context[j]=dict(spent=spent,limit=limit,frozen=frozen,costs=c.policy.completion_delay_costs[j],known=[cfg.config_id for cfg in spec.configs if predict_job(jobs[j],cfg)['raw_rate'] is not None])
    ev=evaluate_reconfig_plan(s,columns,scores,bundle,[],60.,context)
    assert ev['status']=='OK'
    resident={col.job_id:col for col in bundle};lookup={col.column_id:col for col in columns}
    spent={j:v['spent'] for j,v in context.items()};locked_until={col.job_id:scores[col.column_id]['delay'] for col in bundle}
    for event in ev['events']:
     if event['event']=='COMPLETE_RELEASE':resident.pop(event['job'],None)
     elif event['event']=='SURVIVOR_RECONFIGURE':
      j=event['job'];target=lookup[event['column']];old=resident[j]
      different=(old.config.config_id,old.placement.node_gpus)!=(target.config.config_id,target.placement.node_gpus)
      expected=context[j]['costs'][target.config.gpu_type] if different else 0.
      assert event['time']>=locked_until[j]-1e-9
      assert event['startup']==expected
      spent[j]+=expected;assert abs(spent[j]-event['spent_after'])<1e-9 and spent[j]<=context[j]['limit']+1e-9
      assert event['time']%60==0
      if different:assert not context[j]['frozen'] and target.config.config_id in context[j]['known']
      resident[j]=target;locked_until[j]=event['time']+expected
     for node in s.nodes:
      used=[sum(col.usage_for(node.node_id)[k] for col in resident.values()) for k in range(3)]
      assert all(x<=y+1e-7 for x,y in zip(used,(node.gpus,node.cpu_cores,node.ram_gb)))
    ev['independent_future_capacity_lock_budget_audit']=True
    evaluations[plan['name']]=dict(old_forecast=plan['evaluation']['sum_completion'],new_forecast=ev)
   results.append(dict(time=r['interval_end'],evaluations=evaluations,profile_segments_verified=count,training_steps=0))
   print('STATE',r['interval_end'],{k:round(v['new_forecast']['sum_completion'],3) for k,v in evaluations.items()},flush=True)
   if r['interval_end']==5820:break
 with (O/'same_state_verified.json').open('x',encoding='utf8') as f:json.dump(dict(results=results,gate=len(results)==2,raw_observation_mismatches=0),f,indent=2)
if __name__=='__main__':main()
