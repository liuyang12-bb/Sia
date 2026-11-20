"""Full-domain offline census, unchanged regression tests and bounded opportunity gates."""
import collections,gzip,hashlib,itertools,json,math,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_cifar_bert12_20261029_run1';SRC=OUT/'source'
sys.path[:0]=[str(SRC),str(SRC/'sia-simulator')]
def save(n,x):
 with (OUT/n).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def main():
 assert json.loads((OUT/'tests_canonical.json').read_text())['exit_code']==0
 gates={}
 for method in ['SIA_P','CC_CPU','CC_FULL']:
  with (OUT/f'GATE_{method}_v2.log').open('x',encoding='utf8') as f:r=subprocess.run([sys.executable,'-X','utf8','-B',str(OUT/'run_scenario_v2.py'),'run','--mix','GATE','--scenario','RNEW','--response','medium','--method',method],stdout=f,stderr=subprocess.STDOUT)
  assert r.returncode==0,method
  pair=0.;gain=0.;supported_pair=0.
  with gzip.open(OUT/f'cells/GATE_RNEW_medium_{method}/rounds.jsonl.gz','rt',encoding='utf8') as f:
   for line in f:
    rr=json.loads(line);es=rr['execution']
    for a,b in itertools.combinations(es,2):
     if {a['job_id'].rsplit('-',1)[0],b['job_id'].rsplit('-',1)[0]}!={'cifar10','bert'}:continue
     if not a['resource_allocation'] or not b['resource_allocation']:continue
     aa,bb=a['resource_allocation'],b['resource_allocation']
     if not(set(aa['placement'])&set(bb['placement'])):continue
     for za,zb in itertools.product(a['training_segments'],b['training_segments']):
      dt=max(0,min(za['start_time']+za['seconds'],zb['start_time']+zb['seconds'])-max(za['start_time'],zb['start_time']))
      pair+=dt
      cf=a if a['job_id'].startswith('cifar10') else b;bt=b if cf is a else a
      if cf['rho']>1+1e-9 and abs(bt['rho']-1)<1e-12:gain+=dt
      def inside(x):
       return all(1-1e-12<=x[k]['requested_cpu']<=24+1e-12 and 20.83-1e-12<=x[k]['requested_ram_gb']<=250+1e-12 for k in ['source_coordinates','baseline_source_coordinates'])
      if inside(aa) and inside(bb):supported_pair+=dt
  gates[method]=dict(productive_coplace_seconds=pair,receiver_accelerated_donor_unchanged_seconds=gain,fully_supported_pair_seconds=supported_pair)
  print('GATE',method,gates[method],flush=True)
 passed=gates['CC_FULL']['receiver_accelerated_donor_unchanged_seconds']>=60
 save('actual_opportunity_gate.json',dict(passed=passed,methods=gates,boundary_warning='Actual E-legacy clipped points reported separately from domain-supported static witness.'))
 if not passed:raise RuntimeError('NO_ACTUAL_OPPORTUNITY; do not run formal6 or retune')
 with (OUT/'formal_batch_console.log').open('x',encoding='utf8') as f:r=subprocess.run([sys.executable,'-X','utf8','-B',str(OUT/'run_scenario_v2.py'),'batch'],stdout=f,stderr=subprocess.STDOUT)
 print('FORMAL_BATCH_EXIT',r.returncode,flush=True)
if __name__=='__main__':main()
