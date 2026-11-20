"""Independent raw-event/time/resource/response audit, no simulator stepping."""
import argparse,collections,csv,gzip,hashlib,json,math,sys
from pathlib import Path
import numpy as np
def close(a,b):
 assert math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-5),(a,b)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def main(out):
 p=json.loads((out/'preregistration.json').read_text(encoding='utf-8'))
 sys.path.insert(0,str(out/'source'))
 from joint_prototype.r0_liveness import R0Oracle
 from joint_prototype.execution_audit import audit_execution
 from run_stage2c_cell import independent_rho
 results=[]
 for cd in p['runs']:
  directory=out/'cells'/cd['id']
  if not (directory/'summary.json').exists():
   results.append(dict(id=cd['id'],status='NOT_COMPLETE_OR_NOT_STARTED'));continue
  s=json.loads((directory/'summary.json').read_text())
  rows=list(csv.DictReader((out/'inputs'/f"{cd['scenario']}.csv").open()))
  apps={r['name']:r['application'] for r in rows};arrivals={r['name']:float(r['time']) for r in rows}
  oracle=R0Oracle(out/'inputs/registry.json',out/'inputs/models',apps,cd['response'])
  totals=collections.defaultdict(collections.Counter);pr=collections.defaultdict(float);complete={};count=collections.Counter();last=None
  perround=[]
  with gzip.open(directory/'rounds.jsonl.gz','rt',encoding='utf-8') as f:
   for line in f:
    r=json.loads(line)
    for e in r['execution']:
     assert audit_execution(e)['valid'];j=e['job_id'];close(pr[j],e['progress_before']);pr[j]=e['progress_after']
     assert e['progress_after']>=e['progress_before']-1e-8
     for k,v in [('wait','waiting_seconds'),('checkpoint','checkpoint_delay_consumed'),('training','training_seconds')]:totals[j][k]+=e[v]
     if e['completion_time'] is not None:complete[j]=e['completion_time']
     a=e['resource_allocation']
     if a:
      rho=independent_rho(oracle,j,a['config_id'],a['cpu_per_gpu'],a['ram_gb_per_gpu']);close(rho,e['rho'])
      if cd['response']=='neutral':assert rho==1
      for seg in e['training_segments']:close(seg['effective_rate'],seg['raw_rate_before_rho']*rho)
      for k,fac in [('gpu_hours',1),('cpu_core_hours',a['cpu_per_gpu']),('ram_gb_hours',a['ram_gb_per_gpu'])]:totals[j][k]+=a['gpus']*fac*e['allocated_seconds']/3600
     count['execution']+=1
    d=r['decision_for_next_interval']
    if last==d['round']:continue
    last=d['round'];used=collections.defaultdict(lambda:[0,0,0]);selected=d['result']['selected']
    for j,c in selected.items():
     for node,g in c['placement'].items():used[node][0]+=g;used[node][1]+=g*c['cpu_per_gpu'];used[node][2]+=g*c['ram_gb_per_gpu']
    # Ratio changes are dimensionless; never add raw throughput of different apps.
    ratios={}
    for j,col in selected.items():
     cap=cd['nodes'][col['config']['gpu_type']];pc=cap['cpu']/cap['gpus'];pm=cap['ram']/cap['gpus']
     base=independent_rho(oracle,j,col['config']['config_id'],pc,pm)
     ratios[j]=dict(gain=col['rho']/base-1,donates=col['cpu_per_gpu']<pc-1e-8 or col['ram_gb_per_gpu']<pm-1e-8)
    exchange=[]
    for ja,a in selected.items():
     for jb,b in selected.items():
      if ja!=jb and set(a['placement'])==set(b['placement']) and ratios[ja]['gain']>=.05 and ratios[jb]['donates'] and ratios[jb]['gain']>=-.1 and ratios[ja]['gain']+ratios[jb]['gain']>0:
       exchange.append(dict(receiver=ja,donor=jb,receiver_gain=ratios[ja]['gain'],donor_gain=ratios[jb]['gain']))
    if exchange:count['selected_resource_exchange_rounds']+=1
    for node,v in used.items():
     cap=cd['nodes'][node.rsplit(':',1)[0]]
     assert all(x<=y+1e-7 for x,y in zip(v,[cap['gpus'],cap['cpu'],cap['ram']]));count['node_capacity']+=1
    for spec in d['probe']['specs']:
     if spec['checkpoint_locked'] and spec['current_config_id']:
      c=selected[spec['job_id']];assert c['config']['config_id']==spec['current_config_id'];assert c['placement']==dict(spec['current_placement'])
      close(c['cpu_per_gpu'],spec['current_cpu_per_gpu']);close(c['ram_gb_per_gpu'],spec['current_ram_gb_per_gpu']);count['checkpoint_lock']+=1
    perround.append(dict(round=d['round'],time=r.get('time'),status=d['result']['status'],objective=d['result']['scheduling_objective'],resource_exchange=exchange,selected={j:dict(config=c['config']['config_id'],cpu=c['cpu_per_gpu'],ram=c['ram_gb_per_gpu'],rho=c['rho'],raw=c['config']['sia_goodput'],effective=c['effective_goodput'],placement=c['placement']) for j,c in selected.items()}))
   # Recompute each completed JCT from raw completion events, never summary averages.
  for j,v in s['jobs'].items():
   if v['completed']:
    close(v['completion'],complete[j]);close(v['jct'],complete[j]-arrivals[j]);close(v['jct'],sum(totals[j][k] for k in ['wait','checkpoint','training']))
   for k,val in totals[j].items():close(v[k],val)
  if len(complete)==len(rows):close(s['mean_jct'],np.mean([complete[j]-arrivals[j] for j in complete]))
  with gzip.open(directory/'audited_decisions.jsonl.gz','xt',encoding='utf-8') as f:
   for r in perround:f.write(json.dumps(r)+'\n')
  results.append(dict(id=cd['id'],status=s['status'],valid=True,checks=dict(count),summary=s,hashes={f.name:sha(f) for f in directory.iterdir() if f.is_file()}))
 analysis=[]
 for scenario in sorted({r['scenario'] for r in p['runs']}):
  for control in sorted({r['capacity_control'] for r in p['runs'] if r['scenario']==scenario}):
   cells=[r for r in results if r.get('summary',{}).get('cell',{}).get('scenario')==scenario and r['summary']['cell']['capacity_control']==control]
   ss={r['summary']['cell']['method']:r['summary'] for r in cells}
   if len(ss)==3 and all(s['completion_rate']==1 for s in ss.values()):
    joint=ss['G112'];comparisons={}
    for method in ['sia_p','sia_aux']:
     base=ss[method];deltas={j:joint['jobs'][j]['jct']-v['jct'] for j,v in base['jobs'].items()}
     comparisons[method]=dict(mean_jct_improvement=1-joint['mean_jct']/base['mean_jct'],job_delta=deltas,improved=sum(v<0 for v in deltas.values()),worsened=sum(v>0 for v in deltas.values()),total_time_delta={k:joint['totals'][k]-base['totals'][k] for k in ['wait','checkpoint','training']})
    analysis.append(dict(scenario=scenario,control=control,comparisons=comparisons))
 with (out/'independent_audit.json').open('x',encoding='utf-8') as f:json.dump(dict(results=results,paired_analysis=analysis,all_available_valid=all(r.get('valid',False) for r in results if r['status']!='NOT_COMPLETE_OR_NOT_STARTED')),f,indent=2)
 print(json.dumps(analysis,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);main(ap.parse_args().output.resolve())
