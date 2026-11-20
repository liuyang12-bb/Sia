import argparse,dataclasses,gzip,hashlib,importlib.metadata,json,sys,time,traceback,types
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2];T=R/'cc-joint-release-reconfig-v2';P=R/'research_runs/sia_synergy/cc_release_reconfig_20261101_run1'
sys.path[:0]=[str(T),str(T/'sia-simulator')]
from core80 import build
from run_core80 import Metrics,gate,compact_decision,encode
from joint_prototype.release_evaluation import ReleaseEvaluationMixin
from joint_prototype.neutral_liveness import geometry
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def save(p,v):
 with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,default=encode)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cell',required=True);a=ap.parse_args()
 p=json.loads((O/'preregistration.json').read_text());cell=next(x for x in p['cells'] if x['id']==a.cell)
 assert all(sha(T/n)==h for n,h in p['source_hashes'].items())
 assert all(sha(n)==h for n,h in p['input_hashes'].items())
 for n,v in p['environment']['packages'].items():assert importlib.metadata.version(n)==v
 out=O/'cells'/a.cell;resources=gate(O);out.mkdir(parents=True,exist_ok=False);start=time.time()
 save(out/'start.json',dict(cell=cell,resources=resources,start_unix=start,preregistration_sha256=sha(O/'preregistration.json')))
 c=build('FULL',workload_path=O/'inputs/workload.csv',topology=p['topology']);m=Metrics(c);hits=[]
 live=c.policy.liveness
 def refine(self,s,columns,legacy,now):
  if now>cell['fork_time']:return legacy,None
  chosen,ev=ReleaseEvaluationMixin.refine_completion_bundle(self,s,columns,legacy,now)
  if now==cell['fork_time']:
   assert ev['status']=='OK'
   plan=next(z for z in ev['plans'] if z['name']==cell['plan'])
   chosen=[next(v for v in columns if v.column_id==key) for key in plan['columns'].values()]
   assert s.audit(chosen)['valid'];assert {v.job_id for v in chosen}=={v.job_id for v in legacy}
   ev['selected_plan']=cell['plan'];ev['selected']=plan['evaluation'];ev['predicted_sum_completion_saved']=ev['baseline']['sum_completion']-ev['selected']['sum_completion']
   ev['per_job_predicted_saved']={j:ev['baseline']['completion_times'][j]-ev['selected']['completion_times'][j] for j in s.job_map}
   ev['receiver_gain']=sum(max(v,0) for v in ev['per_job_predicted_saved'].values());ev['donor_loss']=sum(max(-v,0) for v in ev['per_job_predicted_saved'].values())
   hits.append(now)
   save(out/'fork.json',dict(time=now,evaluation=ev,scores=s.current_completion_scores,
    columns=[dict(id=v.column_id,job=v.job_id,config=v.config.config_id,placement=v.placement.node_gpus,cpu=v.query.cpu_per_gpu,ram=v.query.ram_gb_per_gpu,rho=v.query.rho) for v in columns],
    jobs={j:dict(progress=x.progress,epoch=x.epoch,atomic_bsz=x.atomic_bsz,accum_steps=x.accum_steps,rescale=x.rescale_time) for j,x in c.policy.completion_jobs.items()}))
  return chosen,ev
 live.refine_completion_bundle=types.MethodType(refine,live)
 historical={}
 with gzip.open(P/'cells/NEW_FULL/rounds.jsonl.gz','rt',encoding='utf8') as f:
  for line in f:
   r=json.loads(line)
   if r['interval_end']<=cell['fork_time']:historical[r['interval_end']]=r
 prefix=0
 try:
  with gzip.open(out/'rounds.jsonl.gz','xt',encoding='utf8',compresslevel=3) as f:
   for i in range(p['max_rounds']):
    if c.all_complete():break
    tick=time.perf_counter();c.step(p['interval']);r=c.joint_round_logs[-1];r['round_wall_seconds']=time.perf_counter()-tick
    r['job_end_state']={j.name:dict(restarts=j.num_restarts or 0,migrations=j.num_migrations or 0,wasted_gpu_seconds=j.wasted_gpu_seconds,completion=j.completion_time,progress=float(j.progress),epoch=int(j.epoch)) for j in c.jobs}
    compact_decision(r['decision_for_next_interval'])
    t=r['interval_end']
    if t<=cell['fork_time']:
     old=historical[t]
     assert r['execution']==old['execution'],('PREFIX_EXECUTION',t)
     d=r['decision_for_next_interval'];od=old['decision_for_next_interval']
     assert json.loads(json.dumps(d['probe']['specs'],default=encode))==od['probe']['specs'],('PREFIX_STATE',t)
     assert json.loads(json.dumps(d['probe']['l2_before'],default=encode))==od['probe']['l2_before'],('PREFIX_LEDGER',t)
     if t<cell['fork_time']:assert json.loads(json.dumps(d['result']['selected'],default=encode))==od['result']['selected'],('PREFIX_DECISION',t)
     prefix+=1
    f.write(json.dumps(r,default=encode,separators=(',',':'))+'\n');f.flush();m.consume(c,r)
    if (i+1)%40==0:print('PROGRESS',cell['id'],i+1,sum(j.completion_time is not None for j in c.jobs),flush=True)
    c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()
    for key in ('exchange_columns','exchange_scores','continuity_scores','completion_scores','actual_completion_columns'):
     if hasattr(c.policy,key):setattr(c.policy,key,[])
  assert hits==[cell['fork_time']]
  result=m.summary(c);result.update(cell=cell,total_wall_seconds=time.time()-start,termination='ALL_COMPLETE' if c.all_complete() else 'ROUND_LIMIT',prefix_frames_exact=prefix,interventions=1,continuation='ORIGINAL_FULL_NO_EVALUATOR')
  save(out/'summary.json',result);save(out/'output_hashes.json',{v.name:sha(v) for v in out.iterdir() if v.is_file()})
  print('RESULT',a.cell,result['completed'],result['mean_jct'],flush=True)
 except Exception:
  save(out/'failure.json',dict(traceback=traceback.format_exc(),time=c.current_time));raise
if __name__=='__main__':main()
