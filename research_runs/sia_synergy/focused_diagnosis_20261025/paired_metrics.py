import json,csv
from pathlib import Path
P=Path(__file__).resolve().parent;load=lambda n:json.loads((P/n).read_text(encoding='utf-8'))
pairs=[]
for a,b in [('main_w1_medium_sia_p','main_w1_medium_sia_tune'),('main_w1_neutral_sia_p','main_w1_neutral_joint_o'),('main_w1_neutral_joint_p','main_w1_neutral_joint_o')]:
 x=load(a+'_raw_audit.json');y=load(b+'_raw_audit.json');ds={j:x['jobs'][j]['jct']-y['jobs'][j]['jct'] for j in x['jobs']};top=sorted(ds,key=ds.get,reverse=True);net=sum(ds.values())
 pairs.append(dict(baseline=a,comparison=b,mean_reduction=net/160,percent=100*net/sum(j['jct'] for j in x['jobs'].values()),improved=sum(d>0 for d in ds.values()),worse=sum(d<0 for d in ds.values()),tied=sum(d==0 for d in ds.values()),top5=[dict(job=j,difference=ds[j]) for j in top[:5]],top5_fraction=sum(ds[j] for j in top[:5])/net if net else None,diagnostic_remove_top1=(net-ds[top[0]])/159,per_job_difference=ds))
with (P/'paired_metrics.json').open('x',encoding='utf-8') as f:json.dump(pairs,f,indent=2)
manifest=json.loads((P.parent/'sia_r0_full160_comparison/attempt2/preregistration.json').read_text())
workload=list(csv.DictReader(Path(manifest['traces']['1']['file']).open()))
configs=load('response_config_table.json');keys=['job','arrival','application','template','config','baseline','best','best_rho','cpu_only_max_rho','ram_only_max_rho','cheap_no_slowdown','best_extra_cpu','best_extra_ram','confidence']
with (P/'task_config_evidence.csv').open('x',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
 for job in workload:
  for c in configs:
   if c['application']!=job['application']:continue
   row={k:c[k] for k in keys if k in c};row.update(job=job['name'],arrival=job['time'],confidence='official template values + synthetic cross-hardware transfer; solo feasibility only');w.writerow(row)
print('PAIRED',[(p['percent'],p['improved'],p['worse'],p['top5_fraction']) for p in pairs])
