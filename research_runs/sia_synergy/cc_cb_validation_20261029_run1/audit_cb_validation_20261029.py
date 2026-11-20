import collections,csv,gzip,hashlib,json,statistics,sys,subprocess,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_cb_validation_20261029_run1';OLD=ROOT/'research_runs/sia_synergy/cc_cifar_bert12_20261029_run1'
RETRY=ROOT/'research_runs/sia_synergy/cc_cb_validation_20261029_ample_retry1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(n,v):
 with (OUT/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
 p=json.loads((OUT/'preregistration.json').read_text());original=[json.loads(l) for l in (OUT/'execution_manifest.jsonl').read_text().splitlines()];retry=[json.loads(l) for l in (RETRY/'execution_manifest.jsonl').read_text().splitlines()]
 assert len(original)==31 and original[-1]['exit_code']!=0 and len(retry)==6
 records=original[:30]+retry;assert len(records)==36 and all(r['exit_code']==0 for r in records[:30])
 rp=json.loads((RETRY/'preregistration.json').read_text());assert rp['source_hashes']==p['source_hashes'] and rp['input_hashes']==p['input_hashes'] and rp['capacities']==p['capacities']
 save('accepted_execution_manifest.json',[dict(accepted_index=i,source_directory=str(RETRY if i>=30 else OUT),**r) for i,r in enumerate(records)])
 text=(ROOT/'maintenance/audit_breakpoint30_precision_v3_20261029.py').read_text();body=text.split('def main():\n',1)[1].split(" p=json.loads((OUT/'preregistration.json')",1)[0]
 body=body.replace(' DEST.mkdir(exist_ok=False)\n','').replace('cc_breakpoint_study_20261029_run1/attempt3','cc_cb_validation_20261029_run1')
 body=body.replace(" ns={'__file__':", " text=text.replace('tau=6000*60-2400','tau=6000*60-1200').replace('if len(vals)==20:', 'if len(vals)==12:')\n ns={'__file__':")
 body=body.replace(" ns={'__file__':", " text=text.replace('assert(cp,mm)==(cpu/2,ram/2)','assert cp in {cpu/2,round(cpu/2,9)} and mm in {ram/2,round(ram/2,9)}').replace('assert mm==ram/2','assert mm in {ram/2,round(ram/2,9)}').replace('assert cp==cpu/2','assert cp in {cpu/2,round(cpu/2,9)}')\n ns={'__file__':")
 scope=dict(ROOT=ROOT,OUT=OUT,DEST=OUT,np=np);exec('def bootstrap():\n'+body+'\n return ns\n',scope);ns=scope['bootstrap']()
 original_points=ns['independent_points']
 def canonical_points(name,col,cpu,ram,method):
  dims=original_points(name,col,cpu,ram,method)
  return (set(dims[0])|{round(cpu/2,9)},set(dims[1])|{round(ram/2,9)})
 ns['independent_points']=canonical_points
 results={};metrics=[];rows=[]
 for cell,rec in zip(p['cells'],records):
  name='_'.join(cell);assert rec['cell']==name;folder=(RETRY if cell[1]=='AMPLE' else OUT)/'cells'/name
  if rec['exit_code']:
   failure=json.loads((folder/'failure.json').read_text());round_count=0;previous={}
   from joint_prototype.execution_audit import audit_execution
   with gzip.open(folder/'rounds.jsonl.gz','rt',encoding='utf8') as f:
    for line in f:
     event=json.loads(line);round_count+=1
     for e in event['execution']:
      assert audit_execution(e)['valid']
      if e['allocated_seconds']>0:
       assert ns['geom'](e['resource_allocation'])==ns['geom'](previous[e['job_id']])
       a=e['resource_allocation'];assert abs(e['rho']-ns['rho'](e['job_id'],a['config_id'],a['cpu_per_gpu'],a['ram_gb_per_gpu'],.5))<1e-10
     d=event['decision_for_next_interval'];sel=d['result']['selected'];specs={q['job_id']:q for q in d['probe']['specs']};usage=collections.defaultdict(lambda:[0.,0.,0.])
     assert {j for j,q in specs.items() if q['checkpoint_locked']}<=set(sel)
     for j,col in sel.items():
      if specs[j]['checkpoint_locked']:
       q=specs[j];assert ns['geom'](col)==(q['current_config_id'],q['current_cpu_per_gpu'],q['current_ram_gb_per_gpu'],tuple(tuple(x) for x in q['current_placement']))
      for node,g in col['placement'].items():
       for k,value in enumerate([g,g*col['cpu_per_gpu'],g*col['ram_gb_per_gpu']]):usage[node][k]+=value
     cpu,ram=p['capacities'][cell[1]];assert all(all(x<=y+1e-7 for x,y in zip(v,[2,cpu,ram])) for v in usage.values());previous=sel
   numeric=['mean','median','p95','p99','makespan','gpu_hours','cpu_core_hours','ram_gb_hours','wait','training','checkpoint','restarts','wasted_gpu_seconds']
   metrics.append(dict(cell=name,cohort=cell[0],capacity=cell[1],method=cell[-1],completed=None,**{k:None for k in numeric}))
   results[name]=dict(valid=False,execution_failed=True,failure_reason=failure['error'],failure_time=failure['time'],committed_rounds_audited=round_count,summary=dict(completed=None,mean_jct=None,totals={}),jobs={})
   for row in csv.DictReader((OUT/'inputs'/f'{cell[0]}_workload.csv').open()):rows.append(dict(cell=name,job=row['name'],status='RUN_FAILED_NO_FINAL_JCT',arrival=float(row['time']),jct=None,completion=None))
   print('FAILED_CELL_PARTIAL_AUDITED',name,round_count,flush=True);continue
  for n,h in json.loads((folder/'output_hashes.json').read_text()).items():assert sha(folder/n)==h
  r=ns['audit'](folder,p);results[name]=r;s=r['summary']
  with gzip.open(folder/'rounds.jsonl.gz','rt',encoding='utf8') as f:
   for line in f:
    event=json.loads(line);decision=event['decision_for_next_interval']
    locks={q['job_id'] for q in decision['probe']['specs'] if q['checkpoint_locked']}
    assert locks<=set(decision['result']['selected']),'LOCKED_JOB_OMITTED'
  metrics.append(dict(cell=name,cohort=cell[0],capacity=cell[1],method=cell[-1],completed=s['completed'],mean=s['mean_jct'],median=s['median_jct'],p95=s['p95_jct'],p99=s['p99_jct'],makespan=s['makespan'],**s['totals']))
  rows.extend(dict(cell=name,job=j,**v) for j,v in r['jobs'].items());print('AUDITED',name,s['completed'],flush=True)
 save('raw_audit.json',results);save('metrics.json',metrics)
 with (OUT/'per_job.csv').open('x',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=sorted(set(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
 comparisons={}
 pairs=[('CC_FULL','SIA_P'),('CC_FULL','SIA_AUX'),('CC_P','SIA_P'),('CC_CPU','CC_P'),('CC_MEM','CC_P'),('CC_FULL','CC_P'),('CC_FULL','CC_CPU'),('CC_FULL','CC_MEM')]
 for cohort,cap in dict.fromkeys((r[0],r[1]) for r in p['cells']):
  comparisons[cohort+'/'+cap]={}
  for a,b in pairs:
   x=results[f'{cohort}_{cap}_medium_{a}'];y=results[f'{cohort}_{cap}_medium_{b}'];sx,sy=x['summary'],y['summary']
   if sx['completed']!=12 or sy['completed']!=12:comparisons[cohort+'/'+cap][a+'_vs_'+b]={'censored':True,'reason':'RUN_FAILURE_UNPAIRED; not observation-window right censoring','complete_jct_comparison_permitted':False};continue
   d={j:v['jct']-y['jobs'][j]['jct'] for j,v in x['jobs'].items()};winner=min(d,key=d.get)
   comparisons[cohort+'/'+cap][a+'_vs_'+b]=dict(improvement=1-sx['mean_jct']/sy['mean_jct'],job_delta=d,improved=sum(v<0 for v in d.values()),worsened=sum(v>0 for v in d.values()),unchanged=sum(v==0 for v in d.values()),time_delta={k:sx['totals'][k]-sy['totals'][k] for k in ['wait','training','checkpoint']},p95_change=sx['p95_jct']/sy['p95_jct']-1,p99_change=sx['p99_jct']/sy['p99_jct']-1,makespan_change=sx['makespan']/sy['makespan']-1,biggest_gain_job=winner,leave_best_out_diagnostic=1-sum(v['jct'] for j,v in x['jobs'].items() if j!=winner)/sum(v['jct'] for j,v in y['jobs'].items() if j!=winner))
 save('comparisons.json',comparisons)
 # Descriptive accounting: same accepted independent response integrals, no simulation.
 text=(OLD/'describe_response_v2.py').read_text().replace('cc_cifar_bert12_20261029_run1','cc_cb_validation_20261029_run1')
 text=text.replace("OUT/'cells'/name/'rounds.jsonl.gz'", "((ROOT/'research_runs/sia_synergy/cc_cb_validation_20261029_ample_retry1') if cap=='AMPLE' else OUT)/'cells'/name/'rounds.jsonl.gz'")
 text=text.replace("  out=collections.Counter();apps=", "  folder=((ROOT/'research_runs/sia_synergy/cc_cb_validation_20261029_ample_retry1') if cap=='AMPLE' else OUT)/'cells'/name\n  if not (folder/'summary.json').exists():\n   data[name]=dict(failed=True,interpretation='Solver failure; no overall response/JCT aggregate computed');continue\n  out=collections.Counter();apps=")
 (OUT/'response_analysis_script.py').write_text(text,encoding='utf8')
 with (OUT/'response_analysis.log').open('x',encoding='utf8') as f:proc=subprocess.run([sys.executable,'-X','utf8','-B',str(OUT/'response_analysis_script.py')],stdout=f,stderr=subprocess.STDOUT)
 assert proc.returncode==0
 for n,h in p['source_hashes'].items():assert sha(OUT/'source'/n)==h
 for n,h in p['input_hashes'].items():assert sha(OUT/'inputs'/n)==h
 seal=json.loads((OLD/'delivery_hashes.json').read_text());assert all(sha(OLD/n)==h for n,h in seal.items())
 for n,h in rp['source_hashes'].items():assert sha(RETRY/'source'/n)==h
 save('validation.json',dict(status='ALL_ATTEMPTS_AUDITED_WITH_FAILURES' if any(r.get('execution_failed') for r in results.values()) else 'COMPLETE36',cells=36,complete_cells=sum(r['summary']['completed']==12 for r in results.values()),failed_cells={n:r['failure_reason'] for n,r in results.items() if r.get('execution_failed')},capacity_violations=0,lock_violations=0,rho_violations=0,audit_scope='Complete cells and committed full rounds only; failed cells have no full JCT estimate',source_unchanged=True,historical_hashes_verified=len(seal),simulation_wall_seconds=sum(r['wall_seconds'] for r in records),failed_attempt_wall_seconds=original[-1]['wall_seconds'],repair_scope='Verifier proportional round9 canonicalization only; original30 complete cells retained, all6 AMPLE attempted in fresh retry directory. Source solver failures preserved without changing algorithm/parameters.'))
 print('ALL36_AUDITED',flush=True)
if __name__=='__main__':main()
