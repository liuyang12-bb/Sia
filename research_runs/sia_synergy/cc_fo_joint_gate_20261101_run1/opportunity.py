"""Three structural states, fixed recorded FO intent; no Job/Cluster.step."""
import collections,dataclasses,gzip,hashlib,json,math,sys
from pathlib import Path
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
R=Path(r'D:/科研启航/Sia');T=R/'cc-joint-fo-audit-repair-v2';O=Path(__file__).resolve().parent
sys.path[:0]=[str(T),str(T/'sia-simulator')]
from core80 import build
from joint_prototype.joint_profile import JobSpec,GpuConfig,NodeSpec
from joint_prototype.neutral_liveness import geometry

def specs_from(rows):
 out=[]
 for raw in rows:
  x=dict(raw);x['configs']=tuple(GpuConfig(**v) for v in x['configs']);x['current_nodes']=tuple(x['current_nodes']);x['current_placement']=tuple(tuple(v) for v in x['current_placement']);out.append(JobSpec(**x))
 return out
def solve(s,cols,required):
 cols=[c for c in cols if c.job_id in required]
 if any(not any(c.job_id==j for c in cols) for j in required):return dict(status='NO_COLUMN')
 m,up=s._constraint_matrix(cols,('gpu','cpu','ram'));lo=np.full(len(up),-np.inf)
 for i,j in enumerate(s.jobs):lo[i]=up[i]=float(j.job_id in required)
 w=np.asarray([c.scheduling_utility for c in cols])
 r=milp(c=-w,integrality=np.ones(len(cols)),bounds=Bounds(0,1),constraints=[LinearConstraint(m,lo,up)],options={'time_limit':30.,'mip_rel_gap':0.})
 if not r.success or r.status!=0:return dict(status='UNPROVEN',solver_status=int(r.status),message=str(r.message))
 selected=[c for c,x in zip(cols,r.x) if x>.5];a=s.audit(selected)
 assert a['valid'] and {c.job_id for c in selected}==required
 return dict(status='OPTIMAL',objective=sum(c.scheduling_utility for c in selected),selected=selected,audit=a)
def main():
 raw=R/'research_runs/sia_synergy/cc_clean_ab_20261030_run3/cells/COMPACT_B_FO/rounds.jsonl.gz'
 workload=R/'research_runs/sia_synergy/cc_clean_ab_20261030_run3/inputs/COMPACT_workload.csv'
 # Rules fixed before scanning outcomes; first chronological hit only.
 rules=['FIRST_MIXED_KNOWN_UNLOCKED_WITH_QUEUE','FIRST_COMPLETION_EVENT_MIXED_KNOWN_WITH_QUEUE','FIRST_AFTER_LAST_ARRIVAL_MIXED_KNOWN_WITH_QUEUE']
 (O/'opportunity_preregistration.json').write_text(json.dumps(dict(workload=str(workload),sha256=hashlib.sha256(workload.read_bytes()).hexdigest().upper(),rules=rules,quota='EXACT_EXISTING_GRID',response='UNCHANGED_R0_E_LEGACY_MEDIUM',topology={'aws':2,'dgx-ext':2},cpu_per_node=6,ram_per_node=125,gate='At least one certified same-state joint objective gain AND reduced sum constant-rate ETA after startup, including all served donors; actual JCT not inferred. No alternate workload search if failed.'),indent=2),encoding='utf8')
 chosen={}
 with gzip.open(raw,'rt',encoding='utf8') as f:
  for line in f:
   r=json.loads(line);d=r['decision_for_next_interval'];a=d['result']['audit'];selected=d['result']['selected'];ids=set(selected)
   apps={j.split('-')[0] for j in ids};scores=d.get('continuity_scores',[])
   known=all(any(v['job']==j and v['raw_rate'] is not None for v in scores) for j in ids)
   unlocked=not any(s['checkpoint_locked'] for s in d['probe']['specs'] if s['job_id'] in ids)
   common=bool(ids and {'cifar10','bert'}<=apps and known and unlocked and d['result']['unallocated_jobs'] and a.get('serial_proposals'))
   if not common:continue
   conditions=[True,'COMPLETION' in a.get('geometry_event_reasons',[]),r['interval_end']>=1200]
   for rule,ok in zip(rules,conditions):
    if ok and rule not in chosen:chosen[rule]=dict(time=r['interval_end'],decision=d)
   if len(chosen)==3:break
 c=build('FO',workload_path=workload,topology={'aws':2,'dgx-ext':2});results=[]
 for rule,item in chosen.items():
  d=item['decision'];a=d['result']['audit'];required=set(d['result']['selected']);ss=specs_from(d['probe']['specs'])
  s=c.policy._scheduler([NodeSpec(t+':0',t,2,6.,125.) for t in ('aws','dgx-ext')],ss);s.locked_columns()
  scores={v['column_id']:v for v in d['continuity_scores']};allcols=s.build_joint_columns();eligible=a['shared_eligibility'];cols=[]
  for col in allcols:
   if col.job_id not in required or col.column_id not in scores:continue
   e=eligible.get(col.job_id)
   if e and str(geometry(col)) not in e['eligible_geometries']:continue
   if scores[col.column_id]['raw_rate'] is None:continue
   cols.append(dataclasses.replace(col,scheduling_utility=scores[col.column_id]['utility']))
  fixed=a['serial_proposals'][-1]['fixed']
  fo=solve(s,[col for col in cols if fixed.get(col.job_id)==col.config.config_id],required)
  jo=solve(s,cols,required)
  out=dict(rule=rule,time=item['time'],required=sorted(required),waiting=d['result']['unallocated_jobs'],columns=len(cols),fo_status=fo['status'],joint_status=jo['status'])
  def metrics(v):
   selected=v['selected'];etas={col.job_id:scores[col.column_id]['estimated_completion'] for col in selected};quotas={col.job_id:dict(config=col.config.config_id,cpu=col.query.cpu_per_gpu,ram=col.query.ram_gb_per_gpu,rho=col.query.rho,score=scores[col.column_id],source_supported=not(col.query.source_coordinate_clipped or col.query.baseline_coordinate_clipped)) for col in selected}
   return dict(objective=v['objective'],sum_eta=sum(etas.values()),etas=etas,quotas=quotas,node_totals=v['audit']['node_totals'])
  if fo['status']==jo['status']=='OPTIMAL':
   out.update(fo=metrics(fo),joint=metrics(jo));out['recorded_fo_objective']=d['result']['scheduling_objective']
   assert math.isclose(fo['objective'],d['result']['scheduling_objective'],rel_tol=1e-6,abs_tol=1e-7),'FO_STATE_DOMAIN_RECONSTRUCTION_MISMATCH'
   out['objective_gain']=jo['objective']-fo['objective'];out['net_eta_saved']=out['fo']['sum_eta']-out['joint']['sum_eta'];out['gate_pass']=out['objective_gain']>1e-7 and out['net_eta_saved']>1e-7
   out['per_job_eta_saved']={j:out['fo']['etas'][j]-out['joint']['etas'][j] for j in required}
  else:out['gate_pass']=False
  results.append(out)
  print('STATE',rule,item['time'],out.get('objective_gain'),out.get('net_eta_saved'),out['gate_pass'],flush=True)
 (O/'opportunity_results.json').write_text(json.dumps(dict(results=results,missing_rules=sorted(set(rules)-set(chosen)),gate_pass=any(v['gate_pass'] for v in results),warning='Current constant-rate ETA and reciprocal utility only, not predicted closed-loop JCT; ties/global MILP30s explicit; no training simulation.'),indent=2),encoding='utf8')
if __name__=='__main__':main()
