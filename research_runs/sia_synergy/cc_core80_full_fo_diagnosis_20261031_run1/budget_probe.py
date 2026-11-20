"""Post-hoc structural explanation, no simulation or optimizer invocation.

Add at most two late states to the four first-condition states. Find the FIRST
state per method with empty ready queue, >1 remaining jobs and a DeepSpeech
job whose lifetime startup budget is spent, while a physically feasible
proportional GPU alternative has higher CURRENT normalized-Goodput*rho.
This is not an ETA/JCT counterfactual; history makes the alternative ineligible.
"""
import gzip,hashlib,json,sys
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2];T=R/'cc-joint-core80-full-lex-repair-v8'
S=R/'research_runs/sia_synergy/cc_core80_full_lex_repair_20261031_run5'
sys.path[:0]=[str(T),str(T/'sia-simulator')]
from joint_prototype.r0_liveness import R0Oracle
from run_stage2c_cell import independent_rho
import csv
with (T/'inputs/core80_workload.csv').open(encoding='utf8') as f:w={x['name']:x for x in csv.DictReader(f)}
oracle=R0Oracle(T/'inputs/registry.json',T/'inputs/models',{j:x['application'] for j,x in w.items()},'medium')
out=[]
for m in ('FULL','FO'):
 print('LATE',m,flush=True)
 with gzip.open(S/'cells'/('dual_medium_'+m)/'rounds.jsonl.gz','rt',encoding='utf8') as f:
  for line in f:
   r=json.loads(line);d=r['decision_for_next_interval'];a=d['result']['audit'];sel=d['result']['selected']
   if a['ready'] or len(sel)<2:continue
   specs={x['job_id']:x for x in d['probe']['specs']};found=None
   for j in sorted(sel):
    if w[j]['application']!='deepspeech2' or specs[j]['checkpoint_locked']:continue
    history=a['service_delay_ledgers'][j]
    if history['delay_charged']<250-1e-8:continue
    current=sel[j];cur=current['raw_sia_goodput']*current['rho'];opts=[]
    for c in specs[j]['configs']:
     if c['config_id']==current['config']['config_id']:continue
     typ=c['gpu_type'];n=typ+':0';g=c['gpus'];capg=4 if typ=='aws' else 8
     use=[x for k,x in sel.items() if k!=j and n in x['placement']]
     free=[capg-sum(x['placement'][n] for x in use),3*capg-sum(x['placement'][n]*x['cpu_per_gpu'] for x in use),62.5*capg-sum(x['placement'][n]*x['ram_gb_per_gpu'] for x in use)]
     if g>free[0]+1e-8 or 3*g>free[1]+1e-8 or 62.5*g>free[2]+1e-8:continue
     rho=independent_rho(oracle,j,c['config_id'],3.,62.5);v=c['sia_goodput']*rho
     if v>cur*(1+1e-8):opts.append(dict(config=c['config_id'],placement={n:g},cpu_per_gpu=3.,ram_per_gpu=62.5,rho=rho,normalized_model_goodput=c['sia_goodput'],effective_model_proxy=v,proxy_ratio_to_current=v/cur,node_free_excluding_this_job=free,profile=d['probe']['provenance'][j].get(c['config_id'])))
    if opts:
     found=dict(method=m,time=r['interval_end'],round=d['round'],job=j,remaining_active=len(specs),ready=a['ready'],
       current=current,delay_ledger=history,alternatives=sorted(opts,key=lambda x:x['proxy_ratio_to_current'],reverse=True),
       filters={k:v for k,v in a['shared_eligibility'][j].items() if k not in ('eligible_geometries','rejected')},
       scope='Physically feasible on SAME current state, others fixed, original proportional point; ineligible under existing lifetime startup cap. Normalized Sia-model proxy, not missing physical ETA, actual execution prediction or JCT. Post-hoc diagnostic criterion.');break
   if found:out.append(found);break
with (O/'late_budget_states.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print('BUDGET_PROBE_DONE',[(x['method'],x['time'],x['job']) for x in out],flush=True)
