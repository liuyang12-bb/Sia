import csv,gzip,json,math,sys
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2];T=R/'cc-joint-uncertainty-v4';P=R/'research_runs/sia_synergy/cc_release_reconfig_20261101_run1';F=R/'research_runs/sia_synergy/cc_fork_diagnosis_20261101_run3'
sys.path[:0]=[str(T),str(T/'sia-simulator')]
from joint_prototype.r0_liveness import R0Oracle
from run_stage2c_cell import independent_rho
def shape(c):return (c['config']['config_id'],c['placement'],c['cpu_per_gpu'],c['ram_gb_per_gpu'])
def main():
 v=json.loads((O/'same_state_final.json').read_text());observed={z['time']:z for z in json.loads((F/'fork_comparisons.json').read_text())}
 work={z['name']:z['application'] for z in csv.DictReader((P/'inputs/workload.csv').open())};oracle=R0Oracle(T/'inputs/registry.json',T/'inputs/models',work,'medium')
 with gzip.open(P/'cells/NEW_FULL/rounds.jsonl.gz','rt',encoding='utf8') as f:hist={r['interval_end']:r for r in map(json.loads,f) if r['interval_end'] in (4800,5820)}
 errors=[];counts=dict(capacity_nodes=0,locks=0,budgets=0,rho=0,single_multiplier=0,recovery_permits=0);comparison=[]
 for state in v['results']:
  t=state['time'];plans={p['name']:p['evaluation'] for p in state['evaluation']['plans']}
  for name,p in plans.items():
   previous={j:dict(config={'config_id':s['current_config_id']},placement=dict(s['current_placement']),cpu_per_gpu=s['current_cpu_per_gpu'],ram_gb_per_gpu=s['current_ram_gb_per_gpu']) for j,s in ((x['job_id'],x) for x in hist[t]['decision_for_next_interval']['probe']['specs'])}
   for event in p['events']:
    selected=event['selected'];used={n:[0.,0.,0.] for n in ('aws:0','dgx-ext:0')}
    for j,c in selected.items():
     rho=independent_rho(oracle,j,c['config']['config_id'],c['cpu_per_gpu'],c['ram_gb_per_gpu']);counts['rho']+=1
     if not math.isclose(rho,c['rho'],rel_tol=1e-10):errors.append('RHO')
     for node,g in c['placement'].items():
      for k,x in enumerate((g,g*c['cpu_per_gpu'],g*c['ram_gb_per_gpu'])):used[node][k]+=x
    for n,u in used.items():
     counts['capacity_nodes']+=1
     if any(x>y+1e-7 for x,y in zip(u,(2.,6.,125.))):errors.append('CAPACITY')
    for j in event['locks']:
     counts['locks']+=1
     if j not in previous or j not in selected or shape(previous[j])!=shape(selected[j]):errors.append('LOCK')
    for j,h in event['ledgers'].items():
     counts['budgets']+=1
     if h['delay_charged']>250.+event['tail_grants'].get(j,{}).get('extra_budget',0.)+1e-7:errors.append('BUDGET')
    counts['recovery_permits']+=len(event['recovery']);previous=selected
   original=json.loads((F/'cells'/f'T{t}_LEGACY'/'fork.json').read_text())
   remaining={x['job']:x['remaining_work'] for x in original['scores'].values()}
   for e in p['model_execution']:
    if e['raw_rate'] is None:
     if e['training']!=0:errors.append('CHECKPOINT_TRAINING')
    else:
     counts['single_multiplier']+=1
     if e['training']<0 or e['checkpoint']+e['training']>60.+1e-7:errors.append('NONEXCLUSIVE_TIME')
     expected=max(0.,remaining[e['job']]-e['training']*e['raw_rate']*e['rho'])
     if not math.isclose(expected,e['work_remaining'],abs_tol=1e-7):errors.append('MODEL_PROGRESS_SINGLE_RHO')
     remaining[e['job']]=e['work_remaining']
   pred=plans['LEGACY']['sum_completion']-plans['MIN_STATIC_SUM_ETA']['sum_completion']
  actual=observed[t]['actual_sum_saved']
  comparison.append(dict(time=t,predicted_sum_saved=pred,actual_sum_saved=actual,ranking_matches_independent_fork=(state['evaluation']['selected_plan']=='LEGACY')==(actual<=0),point_ranking_matches=(pred>0)==(actual>0),
    forecast_completion={k:z['completion_times'] for k,z in plans.items()},
    actual_completion={k:sum(x['completion']-t for x in json.loads((F/'per_job.json').read_text())[f'T{t}_{k}'].values() if x['completion']>t) for k in ('LEGACY','MIN_STATIC_SUM_ETA')}))
 result=dict(errors=errors,counts=counts,comparisons=comparison,technical_gate=not errors and v['gate'],ranking_gate=all(c['ranking_matches_independent_fork'] for c in comparison),formal_matrix_started=0)
 with (O/'state_validation_final.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
 print(json.dumps(result,indent=2));assert not errors
if __name__=='__main__':main()
