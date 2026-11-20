"""Recompute witness responses without the oracle interpolation routine."""
import csv,hashlib,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_breakpoint_study_20261029_run1';SRC=OUT/'source'
sys.path[:0]=[str(SRC),str(SRC/'sia-simulator')]
from joint_prototype.joint_profile import load_synergy_template,CPU_COORDS,RAM_COORDS
from applications import APPLICATIONS
def interp(matrix,c,m):
 values=[np.interp(c,CPU_COORDS,matrix[:,j]) for j in range(len(RAM_COORDS))]
 return float(np.interp(m,RAM_COORDS,values))
def main():
 witnesses=json.loads((OUT/'mechanism_witnesses.json').read_text());rows=list(csv.DictReader((OUT/'inputs/M1_workload.csv').open(encoding='utf8')));by={r['name']:r for r in rows}
 registry={r['key']:r for r in json.loads((SRC/'inputs/registry.json').read_text())}
 result=[]
 for factor,w in witnesses.items():
  phases=[];values=[]
  for name,base,new in zip(w['jobs'],w['proportional'],w['alternative']):
   ref=registry[f"{by[name]['application']}/{w['config']}"];t=load_synergy_template(SRC/'inputs/models'/ref['template'])
   for point in [base,new]:
    c,m=point['cpu']/ref['phi_c'],point['ram']/ref['phi_m'];bc,bm=3/ref['phi_c'],62.5/ref['phi_m']
    assert 1<=c<=24 and 20.83<=m<=250 and 1<=bc<=24 and 20.83<=bm<=250
    rho=(interp(t.matrix,c,m)/interp(t.matrix,bc,bm))**.5
    assert math.isclose(rho,point['rho'],rel_tol=1e-12)
   values.append(dict(job=name,base_rho=base['rho'],new_rho=new['rho'],speed_ratio=new['rho']/base['rho']))
  # Original preflight used linear remaining-work fractions. Recheck using
  # actual declared progress at the sampled epoch, not a constant-work/epoch assumption.
  for phase,old in zip([0.,.25,.75],w['phases']):
   deltas=[]
   for i,name in enumerate(w['jobs']):
    app=APPLICATIONS[w['gpu']][by[name]['application']];epoch=min(app.max_epochs-1,int(app.max_epochs*phase))
    remaining=float(app.get_progress(app.max_epochs)-app.get_progress(epoch))
    original_remaining=float(app.get_progress(app.max_epochs))*(1-phase)
    delta=old['individual_eta_delta'][i]*(remaining/original_remaining)
    deltas.append(delta)
   assert sum(deltas)<0
   phases.append(dict(epoch_fraction=phase,individual_eta_delta=deltas,total_eta_delta=sum(deltas)))
  result.append(dict(factor=factor,queries_independently_verified=True,quotas=w['alternative'],rate_ratios=values,actual_epoch_remaining_work_recheck=phases))
 coverage=json.loads((OUT/'offline_coverage.json').read_text());counts={}
 for cap in ['R01','R11']:
  rows=[r for r in coverage if r['capacity']==cap];counts[cap]={f:dict(feasible_queries=sum(r['counts'][f]['feasible'] for r in rows),strict_supported=sum(r['counts'][f]['strict_supported'] for r in rows),numerator_clipped=sum(r['counts'][f]['numerator_clipped'] for r in rows),denominator_clipped=sum(r['counts'][f]['denominator_clipped'] for r in rows)) for f in ['CPU','MEM','FULL']}
 parent=json.loads((OUT/'old_source_hashes.json').read_text());old=ROOT/'research_runs/sia_synergy/cc_resource_mixes42_20261028_run1/source'
 for n,h in parent.items():assert hashlib.sha256((old/n).read_bytes()).hexdigest().upper()==h
 with (OUT/'independent_offline_audit.json').open('x',encoding='utf8') as f:json.dump(dict(valid=True,witnesses=result,coverage=counts,old_282_source_files_preserved=True,simulations=0),f,indent=2)
 print('INDEPENDENT_OFFLINE_PASS',counts)
if __name__=='__main__':main()
