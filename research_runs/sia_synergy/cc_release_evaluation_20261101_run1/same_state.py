"""Three PREVIOUSLY selected states. No training, new state search or future read."""
import dataclasses,gzip,json,sys,time
from pathlib import Path
R=Path(r'D:/科研启航/Sia');T=R/'cc-joint-release-evaluation-v1';O=Path(__file__).resolve().parent
OLD=R/'research_runs/sia_synergy/cc_fo_joint_gate_20261101_run1'
sys.path[:0]=[str(T),str(T/'sia-simulator'),str(OLD)]
from core80 import build
from joint_prototype.joint_profile import JobSpec,GpuConfig,NodeSpec
from joint_prototype.neutral_liveness import geometry
from joint_prototype.release_evaluation import install

def specs(rows):
 out=[]
 for x0 in rows:
  x=dict(x0);x['configs']=tuple(GpuConfig(**v) for v in x['configs']);x['current_nodes']=tuple(x['current_nodes']);x['current_placement']=tuple(tuple(v) for v in x['current_placement']);out.append(JobSpec(**x))
 return out

def run(d,topology,workload,label):
 c=install(build('FULL',workload_path=workload,topology=topology))
 s=c.policy._scheduler([NodeSpec(t+':0',t,g,3.*g,62.5*g) for t,g in topology.items()],specs(d['probe']['specs']));s.locked_columns()
 scores={v['column_id']:v for v in d['continuity_scores']};s.current_completion_scores=scores
 a=d['result']['audit'];cols=[]
 for col in s.build_joint_columns():
  if col.column_id not in scores:continue
  e=a['shared_eligibility'].get(col.job_id)
  offer=e.get('bounded_correction_offer',{}).get('column_id') if e else None
  if e and str(geometry(col)) not in e['eligible_geometries'] and col.column_id!=offer:continue
  cols.append(dataclasses.replace(col,scheduling_utility=scores[col.column_id]['utility']))
 legacy=[]
 for j,x in d['result']['selected'].items():
  legacy.append(next(col for col in cols if col.column_id==x['column_id']))
 live=c.policy.liveness;live.ready=list(a['ready']);live.services={}
 live.numerical_solver_retries=[];live.objective_unit_certifications=[];live._diagnostic_job_order=[j.job_id for j in s.jobs]
 # Include existing held allocation only when present in current domain.
 for j in s.jobs:
  match=next((col for col in cols if col.job_id==j.job_id and col.config.config_id==j.current_config_id and col.placement.node_gpus==j.current_placement and col.query.cpu_per_gpu==j.current_cpu_per_gpu and col.query.ram_gb_per_gpu==j.current_ram_gb_per_gpu),None)
  if match:live.services[j.job_id]={'column':match}
 out=dict(label=label,columns=len(cols),jobs=len(s.jobs),methods={})
 for method in ('FO','FULL'):
  live.clean_gpu='fixed' if method=='FO' else 'joint'
  chosen,ev=live.refine_completion_bundle(s,cols,legacy,0.)
  out['methods'][method]=ev
  print('STATE',label,method,ev['status'],ev.get('predicted_sum_completion_saved'),flush=True)
 return out

def main():
 results=[];trace=R/'research_runs/sia_synergy/cc_clean_ab_20261030_run3/cells/COMPACT_B_FO/rounds.jsonl.gz'
 with gzip.open(trace,'rt',encoding='utf8') as f:
  for line in f:
   row=json.loads(line)
   if row['interval_end'] in (420,1200):
    results.append(run(row['decision_for_next_interval'],{'aws':2,'dgx-ext':2},R/'research_runs/sia_synergy/cc_clean_ab_20261030_run3/inputs/COMPACT_workload.csv',str(row['interval_end'])))
   if row['interval_end']>=1200:break
 results.append(run(json.loads((OLD/'reconstruction_attempt3/repaired_decision.json').read_text()),{'aws':4,'dgx-ext':8,'rtx':8},OLD/'reconstruction_attempt3/survivors_input.csv','13500_RECONSTRUCTED_NO_TRAINING'))
 with (O/'same_state_results.json').open('x',encoding='utf8') as f:json.dump(results,f,indent=2)

if __name__=='__main__':main()
