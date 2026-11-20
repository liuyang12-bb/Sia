"""Read frozen inputs/logs; no Cluster, Job.step or policy.optimize invocation."""
import collections,csv,gzip,hashlib,importlib.util,itertools,json,math,sys,zipfile
from pathlib import Path
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
ROOT=Path(__file__).resolve().parents[3]; OUT=Path(__file__).resolve().parent
BASE=ROOT/'research_runs/sia_synergy/sia_r0_full160_comparison'; RUN=BASE/'attempt2'
WORK=ROOT/'sia-joint-full160'; sys.path.insert(0,str(WORK/'sia-simulator'));sys.path.insert(0,str(BASE))
from joint_prototype.r0_liveness import R0Oracle,APPLICATION_TEMPLATES
from joint_prototype.joint_profile import GpuConfig,JobSpec
from joint_prototype.full160 import capacities
import audit_results
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(name,v):
 with (OUT/name).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
def readrounds(path,partial=False):
 try:
  with gzip.open(path,'rt',encoding='utf-8') as f:
   for l in f:yield json.loads(l)
 except (EOFError,OSError):
  if not partial:raise
def main():
 p=json.loads((RUN/'preregistration.json').read_text());cfg=json.loads(Path(p['config']).read_text())
 with zipfile.ZipFile(RUN/'source_snapshot.zip') as z:
  diffs=[n for n in z.namelist() if (WORK/n).read_bytes()!=z.read(n)]
 save('version_check.json',dict(commit=p['commit'],snapshot_files=len(z.namelist()),byte_differences=diffs,preregistration_sha=sha(RUN/'preregistration.json')))
 rows=json.loads(Path(p['registry']).read_text()); oracle=R0Oracle(p['registry'],p['models'],{a:a for a in APPLICATION_TEMPLATES},'medium')
 cp=cfg['response']['aux_cpu_points'];mp=cfg['response']['aux_ram_points'];cap=capacities('main'); tables=[];points={}
 for r in rows:
  a=r['application'];typ,nn,g=r['config_id'].split(':');g=int(g);c=GpuConfig(r['config_id'],typ,g,1,1.,r['phi_c'],r['phi_m']);j=JobSpec(a,r['template'],(c,))
  def query(cpu,ram):return oracle.query(j,c,cpu,ram,(3.,62.5))
  feasible=[query(cpu,ram) for cpu,ram in itertools.product(cp,mp) if cpu*g<=cap[typ]['cpu_cores']+1e-8 and ram*g<=cap[typ]['ram_gb']+1e-8]
  best=max(feasible,key=lambda q:(q.rho,-q.cpu_per_gpu,-q.ram_gb_per_gpu));cheap=min((q for q in feasible if q.rho>=1-1e-9),key=lambda q:(q.cpu_per_gpu/3+q.ram_gb_per_gpu/62.5,q.cpu_per_gpu,q.ram_gb_per_gpu))
  cpu_best=max((q for q in feasible if q.ram_gb_per_gpu==62.5),key=lambda q:q.rho);mem_best=max((q for q in feasible if q.cpu_per_gpu==3),key=lambda q:q.rho)
  points[r['key']]=[dict(cpu=q.cpu_per_gpu,ram=q.ram_gb_per_gpu,rho=q.rho,clip=q.source_coordinate_clipped) for q in feasible]
  tables.append(dict(key=r['key'],application=a,template=r['template'],template_hash=best.template_sha256,config=c.config_id,gpus=g,phi_c=r['phi_c'],phi_m=r['phi_m'],source_rows=r['source_rows'],baseline=[3,62.5],baseline_clip=query(3,62.5).baseline_coordinate_clipped,best=[best.cpu_per_gpu,best.ram_gb_per_gpu],best_rho=best.rho,best_extra_cpu=g*(best.cpu_per_gpu-3),best_extra_ram=g*(best.ram_gb_per_gpu-62.5),cpu_only_max_rho=cpu_best.rho,ram_only_max_rho=mem_best.rho,cheap_no_slowdown=[cheap.cpu_per_gpu,cheap.ram_gb_per_gpu],cheap_rho=cheap.rho,grid_points=len(feasible),clipped_points=sum(q.source_coordinate_clipped for q in feasible),below_floor=sum(q.rho<1-1e-9 for q in feasible)))
 save('response_config_table.json',tables);save('feasible_response_points.json',points)
 with (OUT/'response_config_table.csv').open('x',encoding='utf-8-sig',newline='') as f:
  keys=[k for k in tables[0] if k!='source_rows'];w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(tables)
 apps=collections.Counter();wl=list(csv.DictReader(Path(p['traces']['1']['file']).open()));apps.update(r['application'] for r in wl)
 save('data_acceptance.json',dict(workload1_mix=dict(apps),configs=55,accelerable_configs=sum(r['best_rho']>1.01 for r in tables),cpu_accelerable=sum(r['cpu_only_max_rho']>1.01 for r in tables),ram_accelerable=sum(r['ram_only_max_rho']>1.01 for r in tables),baseline_clipped=sum(r['baseline_clip'] for r in tables),clipped_grid=sum(r['clipped_points'] for r in tables),grid=sum(r['grid_points'] for r in tables)))
 completed=[];selected_states=[];used=set();rules={}
 for cell in p['runs']:
  folder=RUN/cell['id']
  if not (folder/'summary.json').exists():continue
  print('RAW_AUDIT',cell['id'],flush=True); result=audit_results.cell_audit(folder,cell,p); save(cell['id']+'_raw_audit.json',result)
  hist=collections.Counter();training=boost=slow=0.; weighted=0.; nboost=0; clip=0;selected=0;serviceheld=0;decisions=0
  for r in readrounds(folder/'rounds.jsonl.gz'):
   d=r['decision_for_next_interval'];sel=d['result']['selected'];audit=d['result']['audit']; decisions+=1
   serviceheld+=len(audit.get('service_commitments',[])); selected+=len(sel)
   for c in sel.values():clip+=c['source_coordinates']['clipped']
   for e in r['execution']:
    dt=e['training_seconds'];training+=dt;weighted+=dt*e['rho'];boost+=dt if e['rho']>1.000001 else 0;slow+=dt if e['rho']<.999999 else 0
   if cell['id']=='main_w1_medium_sia_tune':
    usage={t:[0,0.,0.] for t in cap}
    for c in sel.values():
     t=c['config']['gpu_type'];g=c['config']['gpus'];usage[t][0]+=g;usage[t][1]+=g*c['cpu_per_gpu'];usage[t][2]+=g*c['ram_gb_per_gpu']
    tests={'first_two_running':len(sel)>=2,'first_reconfiguration':bool(r['rescale_events'] or r['migrate_events']),'first_sensitive_with_idle_cpu':any(c['rho']>1.0001 and usage[c['config']['gpu_type']][1]<cap[c['config']['gpu_type']]['cpu_cores']-1e-5 for c in sel.values()),'first_fair_with_waiters':bool(d['result']['unallocated_jobs']) and any(c['cpu_per_gpu']==3 for c in sel.values())}
   elif cell['id']=='main_w1_neutral_joint_o':tests={'first_drain':audit.get('drain',False),'first_held_with_waiters':bool(audit.get('service_commitments')) and bool(d['result']['unallocated_jobs'])}
   else:tests={}
   for rule,ok in tests.items():
    if ok and rule not in rules and (cell['id'],r['interval_end']) not in used:
     rules[rule]=True;used.add((cell['id'],r['interval_end']));selected_states.append(dict(rule=rule,cell=cell['id'],time=r['interval_end'],round=r))
  completed.append(dict(cell=cell['id'],metrics=result['metrics'],training_seconds=training,boosted_training_seconds=boost,slow_training_seconds=slow,time_weighted_rho=weighted/training,selected_clipped_fraction=clip/selected,mean_service_commitments=serviceheld/decisions))
 save('completed_results.json',completed)
 partial=RUN/'main_w1_medium_joint_p/rounds.jsonl.gz';n=0;last=None
 for r in readrounds(partial,True):n+=1;last=r
 save('partial_evidence.json',dict(readable_rounds=n,last_time=last['interval_end'],completed=sum(v['completion'] is not None for v in last['job_end_state'].values()),raw_hash=sha(partial),overall_jct=None))
 # Only logged configurations have online Goodput values. Re-optimize auxiliary
 # quotas for that same GPU map; do not invent unlogged online performance.
 comparisons=[]
 for state in selected_states[:6]:
  r=state['round'];d=r['decision_for_next_interval'];sel=d['result']['selected'];jobs=list(sel);held=set(d['result']['audit'].get('service_commitments',[]))
  oracle=R0Oracle(p['registry'],p['models'],{a:a for a in APPLICATION_TEMPLATES},'medium' if 'medium' in state['cell'] else 'neutral')
  options=[];base=[]
  for name,c in sel.items():
   a=name.split('-')[0];row=oracle.reference(a,c['config']['config_id']);gc=GpuConfig(c['config']['config_id'],c['config']['gpu_type'],c['config']['gpus'],c['config']['num_nodes'],c['raw_sia_goodput'],row['phi_c'],row['phi_m']);j=JobSpec(a,APPLICATION_TEMPLATES[a],(gc,));locked=c['candidate_kind']=='locked' or name in held
   def col(cpu,ram):
    q=oracle.query(j,gc,cpu,ram,(3.,62.5));return dict(job=name,gpu=gc.gpu_type,gpus=gc.gpus,config=gc.config_id,cpu=cpu,ram=ram,rho=q.rho,raw=gc.sia_goodput,effective=q.effective_goodput,placement=c['placement'],held=locked)
   base.append(col(c['cpu_per_gpu'],c['ram_gb_per_gpu']) if locked else col(3.,62.5))
   pts=[(c['cpu_per_gpu'],c['ram_gb_per_gpu'])] if locked else itertools.product(cp,mp)
   options.extend(col(cpu,ram) for cpu,ram in pts if cpu*gc.gpus<=cap[gc.gpu_type]['cpu_cores']+1e-8 and ram*gc.gpus<=cap[gc.gpu_type]['ram_gb']+1e-8 and (locked or col(cpu,ram)['rho']>=1-1e-9))
  mat=[];lo=[];hi=[]
  for name in jobs:mat.append([int(c['job']==name) for c in options]);lo.append(1);hi.append(1)
  for t in cap:
   for key,capacity in [('cpu',cap[t]['cpu_cores']),('ram',cap[t]['ram_gb'])]:mat.append([c[key]*c['gpus'] if c['gpu']==t else 0 for c in options]);lo.append(-np.inf);hi.append(capacity)
  result=milp(c=-np.array([c['effective'] for c in options]),integrality=np.ones(len(options)),bounds=Bounds(0,1),constraints=LinearConstraint(np.array(mat),lo,hi),options={'time_limit':10,'mip_rel_gap':0})
  chosen=[c for c,x in zip(options,result.x) if x>.5] if result.x is not None else []
  def total(cols):return dict(predicted_normalized_goodput=sum(c['effective'] for c in cols),cpu=sum(c['cpu']*c['gpus'] for c in cols),ram=sum(c['ram']*c['gpus'] for c in cols),gpus=sum(c['gpus'] for c in cols),jobs=cols)
  observed=sum(c['effective_goodput'] for c in sel.values())
  comparisons.append(dict(rule=state['rule'],cell=state['cell'],time=state['time'],A=total(base),B=total(chosen),observed_goodput=observed,solver_status=int(result.status),C='NOT_IDENTIFIABLE: unselected GPU candidate online Goodput, restart/age/profile state absent from persisted log',constraints='Fixed logged job set/GPU geometry, checkpoint and L2 service quotas held; Medium response; per-config progress floor. B maximizes sum normalized Goodput, not historical fairness utility.',historical_response='medium' if 'medium' in state['cell'] else 'neutral',restart='A/B keep GPU map and placement; no extra GPU restart assumed; auxiliary-only change overhead absent from existing model',unallocated=d['result']['unallocated_jobs']))
  save('state_'+str(len(comparisons))+'.json',state)
 save('same_state_comparisons.json',comparisons)
 save('diagnostic_manifest.json',dict(command='python -B diagnose.py',simulation_invocations=0,static_milp_count=len(comparisons),static_time_limit_seconds=10,input_hashes={str(x):sha(x) for x in [RUN/'preregistration.json',RUN/'execution_manifest.jsonl',Path(p['registry']),Path(p['config']),RUN/'main_w1_medium_joint_p/rounds.jsonl.gz']},source=sha(Path(__file__)),formal_experiments='PAUSED'))
 print('DONE',len(completed),len(comparisons),flush=True)
if __name__=='__main__':main()
