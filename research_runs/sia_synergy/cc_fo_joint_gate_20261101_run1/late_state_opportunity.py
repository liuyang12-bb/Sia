"""Posthoc explanation of the reconstructed failed state, not a new gate."""
import contextlib,dataclasses,json,math
from opportunity import O,T,build,specs_from,solve,NodeSpec,geometry
def main():
 d=json.loads((O/'reconstruction_attempt3/repaired_decision.json').read_text());a=d['result']['audit'];required=set(d['result']['selected'])
 c=build('FO',workload_path=O/'reconstruction_attempt3/survivors_input.csv')
 s=c.policy._scheduler([NodeSpec(t+':0',t,g,3.*g,62.5*g) for t,g in [('aws',4),('dgx-ext',8),('rtx',8)]],specs_from(d['probe']['specs']));s.locked_columns()
 scores={v['column_id']:v for v in d['continuity_scores']};cols=[]
 for col in s.build_joint_columns():
  if col.job_id not in required or col.column_id not in scores:continue
  e=a['shared_eligibility'].get(col.job_id);offered=e.get('bounded_correction_offer',{}).get('column_id') if e else None
  if e and str(geometry(col)) not in e['eligible_geometries'] and col.column_id!=offered:continue
  if scores[col.column_id]['raw_rate'] is None:continue
  cols.append(dataclasses.replace(col,scheduling_utility=scores[col.column_id]['utility']))
 fixed=a['serial_proposals'][-1]['fixed'];fo=solve(s,[c for c in cols if fixed.get(c.job_id)==c.config.config_id],required);jo=solve(s,cols,required)
 result=dict(scope='POSTHOC_RECONSTRUCTED_FAILURE_STATE_ONLY; not permission to select a new workload',time=13500,columns=len(cols),fo_status=fo['status'],joint_status=jo['status'])
 if fo['status']==jo['status']=='OPTIMAL':
  assert math.isclose(fo['objective'],d['result']['scheduling_objective'],abs_tol=1e-7)
  def metric(v):return dict(objective=v['objective'],sum_eta=sum(scores[c.column_id]['estimated_completion'] for c in v['selected']),jobs={c.job_id:dict(config=c.config.config_id,cpu=c.query.cpu_per_gpu,ram=c.query.ram_gb_per_gpu,rho=c.query.rho,eta=scores[c.column_id]['estimated_completion'],raw=scores[c.column_id]['raw_rate'],delay=scores[c.column_id]['delay'],source_supported=not(c.query.source_coordinate_clipped or c.query.baseline_coordinate_clipped)) for c in v['selected']})
  result.update(fo=metric(fo),joint=metric(jo));result['objective_gain']=jo['objective']-fo['objective'];result['net_eta_saved']=result['fo']['sum_eta']-result['joint']['sum_eta'];result['eta_saved_by_job']={j:result['fo']['jobs'][j]['eta']-result['joint']['jobs'][j]['eta'] for j in required}
 (O/'late_state_opportunity.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(result.get('objective_gain'),result.get('net_eta_saved'),flush=True)
if __name__=='__main__':main()
