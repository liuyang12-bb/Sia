import argparse,dataclasses,gzip,hashlib,json,sys,time,traceback
from pathlib import Path
OUT=Path(__file__).resolve().parent
P=json.loads((OUT/'preregistration.json').read_text());sys.path.insert(0,P['tree'])
import psutil
from scenario import make,opportunity
from metrics_legacy_adapter import Metrics,save,default
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--method',required=True);a=ap.parse_args()
 if a.method not in P['runs']:raise ValueError('UNREGISTERED_METHOD')
 if psutil.virtual_memory().available<2*2**30:raise RuntimeError('MEMORY_GATE')
 if psutil.disk_usage(str(OUT)).free<5*2**30:raise RuntimeError('DISK_GATE')
 for p,h in P['source_hashes'].items():assert sha(Path(P['tree'])/p)==h,p
 for p,h in P['input_hashes'].items():assert sha(p)==h,p
 for p,h in P['tools_hashes'].items():assert sha(OUT/p)==h,p
 out=OUT/'cells'/a.method;out.mkdir(parents=True,exist_ok=False)
 save(out/'start.json',dict(method=a.method,preregistration_sha256=sha(OUT/'preregistration.json'),memory_gib=psutil.virtual_memory().available/2**30))
 c=make(a.method);m=Metrics(c);tick=time.perf_counter();events=[];queue_streak=0;maxqueue=0;gate_done=False;passed=False
 try:
  with gzip.open(out/'rounds.jsonl.gz','xt',encoding='utf-8') as f:
   for i in range(P['scenario']['gates']['max_rounds']):
    if c.all_complete():break
    c.step(60);r=c.joint_round_logs[-1];d=c.policy.last_decision;res=d['result']
    # Persist the decision and full execution BEFORE audit; never repair logs.
    f.write(json.dumps(r,default=default,separators=(',',':'))+'\n');f.flush();m.consume(c,r)
    if a.method=='G111' and not gate_done:
     waiting=res['unallocated_jobs'];queue_streak=queue_streak+1 if waiting else 0;maxqueue=max(maxqueue,queue_streak)
     cols=c.policy.exchange_columns;_,flips,exchanges=opportunity(cols,P['scenario']['capacities'],P['scenario']['gpus'])
     selected={name:v['column'] for name,v in c.policy.liveness.services.items()}
     selected_flips=[(x,y) for x,y in flips if y.job_id in selected and selected[y.job_id].config.config_id==y.config.config_id]
     specs={j.job_id:j for j in c.policy.probe_specs};actual=[]
     for ja,aa in selected.items():
      typ=aa.config.gpu_type;pc=P['scenario']['capacities'][typ]['cpu_cores']/P['scenario']['gpus'][typ];pm=62.5
      basea=c.policy.oracle.query(specs[ja],aa.config,pc,pm,(pc,pm))
      if aa.query.rho<basea.rho*1.05:continue
      for jb,bb in selected.items():
       if ja==jb or aa.placement.node_ids!=bb.placement.node_ids:continue
       baseb=c.policy.oracle.query(specs[jb],bb.config,pc,pm,(pc,pm))
       if bb.query.rho+1e-9<baseb.rho*.9:continue
       if not (bb.query.cpu_per_gpu<pc-1e-8 or bb.query.ram_gb_per_gpu<pm-1e-8):continue
       if aa.query.effective_goodput+bb.query.effective_goodput<=basea.effective_goodput+baseb.effective_goodput+1e-8:continue
       actual.append(dict(receiver=dataclasses.asdict(aa),donor=dataclasses.asdict(bb),
           receiver_baseline=dataclasses.asdict(basea),donor_baseline=dataclasses.asdict(baseb)))
     events.append(dict(round=i+1,time=c.current_time,waiting=waiting,queue_streak=queue_streak,
       waiting_seconds=sum(e['waiting_seconds'] for e in r['execution']) if i>0 else 0,
       rank_reversals=len(flips),selected_rank_reversals=len(selected_flips),
       rank_witness=[[dataclasses.asdict(x),dataclasses.asdict(y)] for x,y in selected_flips[:1]],
       selected_exchanges=actual[:1],required_jobs=res['audit']['required_jobs'],optional_selected=res['audit']['optional_selected']))
     if i+1==60 or c.all_complete():
      gate_done=True
      checks=dict(persistent_queue=maxqueue>=3 and sum(e['waiting_seconds'] for e in events)>0,
         live_rank_reversal=any(e['rank_reversals'] for e in events),
         selected_rank_reversal=any(e['selected_rank_reversals'] for e in events),
         actual_resource_exchange=any(e['selected_exchanges'] for e in events))
      passed=all(checks.values());save(out/'opportunity_gate.json',dict(passed=passed,checks=checks,events=events))
      print('GATE',passed,checks,flush=True)
      if not passed:break
    c.joint_round_logs.clear();c.logs.clear();c.rescale_events.clear();c.migrate_events.clear()
   if a.method=='G111' and not passed:
    save(out/'stop.json',dict(status='STOP_NO_JCT_ANALYSIS_GATE_NOT_TRIGGERED',time=c.current_time,rounds=m.rounds))
   else:
    result=m.summary(c,dict(rmst_tau_seconds=36000));result.update(method=a.method,wall_seconds_total=time.perf_counter()-tick)
    save(out/'summary.json',result);print('COMPLETE',a.method,result['completed'],flush=True)
 except Exception as exc:
  save(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=c.current_time,last_decision=c.policy.last_decision));raise

if __name__=='__main__':main()
