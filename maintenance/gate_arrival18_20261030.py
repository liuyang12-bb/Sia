import collections,gzip,hashlib,importlib.util,itertools,json,math,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_arrival_pressure18_20261030_run1';SRC=OUT/'source'
def save(n,v):
 with (OUT/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
def response_effect(raw,dt,work,remaining,rho,baseline):
 if min(raw,dt,work,rho,baseline)<=0:return (0.,0.)
 return raw*dt*(rho-baseline)/work,remaining/raw*(1/baseline-1/rho)
def analyze(folder):
 sys.path[:0]=[str(SRC),str(SRC/'sia-simulator')]
 from applications import APPLICATIONS
 from joint_prototype.joint_profile import CPU_COORDS,RAM_COORDS,load_synergy_template
 import numpy as np
 reg={r['key']:r for r in json.loads((SRC/'inputs/registry.json').read_text())};mat={};tot=collections.Counter();by_job=collections.defaultdict(collections.Counter)
 def rho(j,c,cp,mm):
  r=reg[j.rsplit('-',1)[0]+'/'+c];t=r['template']
  if t not in mat:mat[t]=load_synergy_template(SRC/'inputs/models'/t).matrix
  def f(x,y):return float(np.interp(x,CPU_COORDS,[np.interp(y,RAM_COORDS,row) for row in mat[t]]))
  return (f(cp/r['phi_c'],mm/r['phi_m'])/f(3/r['phi_c'],62.5/r['phi_m']))**.5
 def inside(a):return all(1-1e-12<=a[k]['requested_cpu']<=24+1e-12 and 20.83-1e-12<=a[k]['requested_ram_gb']<=250+1e-12 for k in ['source_coordinates','baseline_source_coordinates'])
 with gzip.open(folder/'rounds.jsonl.gz','rt',encoding='utf8') as f:
  for line in f:
   r=json.loads(line);executions=r['execution'];round_eta=0.;round_progress=0.
   for e in executions:
    a=e['resource_allocation']
    if not a:continue
    name=e['job_id'];app=name.rsplit('-',1)[0];work=float(APPLICATIONS['aws'][app].get_progress(APPLICATIONS['aws'][app].max_epochs));baseline=rho(name,a['config_id'],3.,62.5)
    remaining=max(0.,work-e['progress_before'])
    for z in e['training_segments']:
     delta,eta=response_effect(z['raw_rate_before_rho'],z['seconds'],work,remaining,z['rho'],baseline)
     round_progress+=delta;by_job[name]['net_normalized_progress']+=delta
     by_job[name]['receiver_fraction']+=max(0,delta);by_job[name]['donor_fraction_loss']+=max(0,-delta)
     if z is e['training_segments'][0]:round_eta+=eta;by_job[name]['same_state_eta_saving_sum']+=eta
    tot['productive_job_seconds']+=e['training_seconds']
   tot['net_normalized_progress']+=round_progress;tot['sum_round_same_state_eta_savings']+=round_eta
   for a,b in itertools.combinations(executions,2):
    if {a['job_id'].rsplit('-',1)[0],b['job_id'].rsplit('-',1)[0]}!={'cifar10','bert'}:continue
    aa,bb=a['resource_allocation'],b['resource_allocation']
    if not aa or not bb or not(set(aa['placement'])&set(bb['placement'])):continue
    for za,zb in itertools.product(a['training_segments'],b['training_segments']):
     dt=max(0,min(za['start_time']+za['seconds'],zb['start_time']+zb['seconds'])-max(za['start_time'],zb['start_time']))
     tot['productive_coplace_node_seconds']+=dt
     cf,bt=(a,b) if a['job_id'].startswith('cifar10') else (b,a)
     cr=rho(cf['job_id'],cf['resource_allocation']['config_id'],3,62.5);br=rho(bt['job_id'],bt['resource_allocation']['config_id'],3,62.5)
     if cf['rho']>cr+1e-9 and bt['rho']>=br-1e-12:tot['receiver_faster_donor_not_slower_node_seconds']+=dt
     if inside(aa) and inside(bb):tot['fully_supported_coplace_node_seconds']+=dt
 return dict(totals=dict(tot),per_job={j:dict(v) for j,v in by_job.items()},scope='Same executed GPU/batch/raw rate; proportional counterfactual changes only quotas. Normalized progress and round ETA sums are diagnostic, not realized JCT savings. All served jobs counted; delays unchanged by rho.')
def main():
 spec=importlib.util.spec_from_file_location('runner',OUT/'run.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner);p=runner.setup()
 if (OUT/'contract_tests.json').exists():assert json.loads((OUT/'contract_tests.json').read_text())['exit_code']==0
 else:
  start=time.time()
  with (OUT/'contract_tests.log').open('x',encoding='utf8') as f:r=subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'maintenance/test_arrival18_20261030.py')],stdout=f,stderr=subprocess.STDOUT)
  save('contract_tests.json',dict(exit_code=r.returncode,wall_seconds=time.time()-start));assert r.returncode==0
 gate={}
 for method in ['SIA_P','CC_CPU','CC_MEM','CC_FULL']:
  folder=OUT/f'cells/GATE_RNEW_medium_{method}'
  if folder.exists():
   assert (folder/'summary.json').exists() and (folder/'output_hashes.json').exists(),'REFUSE_PARTIAL_GATE'
   for n,h in json.loads((folder/'output_hashes.json').read_text()).items():assert runner.sha(folder/n)==h
  else:
   # Keep the launcher light: matrix/application analysis runs in a separate
   # process below, so its memory is released before the next simulation.
   start=time.time()
   while True:
    try:
     resource=runner.resources()
     if resource['memory_gib']<2.125:raise RuntimeError('STARTUP_HEADROOM_WAIT '+str(resource))
     break
    except RuntimeError as e:
     if time.time()-start>=600:raise
     print('RESOURCE_WAIT_BEFORE_GATE',method,str(e),flush=True);time.sleep(30)
   log=OUT/f'GATE_{method}.log'
   if log.exists():log=OUT/f'GATE_{method}_{time.time_ns()}.log'
   with log.open('x',encoding='utf8') as f:r=subprocess.run([sys.executable,'-X','utf8','-B',str(OUT/'run.py'),'run','--mix','GATE','--scenario','RNEW','--response','medium','--method',method],stdout=f,stderr=subprocess.STDOUT)
   if r.returncode!=0:
    if not folder.exists() and 'RESOURCE_GATE' in log.read_text(encoding='utf8'):
     raise RuntimeError('RESOURCE_GATE_CHILD_BEFORE_SIMULATION '+method)
    raise RuntimeError('GATE_CHILD_FAILED '+method)
  analysis=OUT/f'GATE_{method}_analysis.json'
  if not analysis.exists():
   with (OUT/f'GATE_{method}_analysis.log').open('x',encoding='utf8') as f:r=subprocess.run([sys.executable,'-X','utf8','-B',str(Path(__file__)),'analyze',str(folder),str(analysis)],stdout=f,stderr=subprocess.STDOUT)
   assert r.returncode==0
  gate[method]=json.loads(analysis.read_text());print('GATE_DONE',method,gate[method]['totals'],flush=True)
 t=gate['CC_FULL']['totals'];passed=t.get('receiver_faster_donor_not_slower_node_seconds',0)>=60 and t.get('net_normalized_progress',0)>0 and t.get('sum_round_same_state_eta_savings',0)>0
 save('opportunity_gate.json',dict(passed=passed,methods=gate,thresholds_frozen_before_runs=True,warning='E-legacy clipped exposure distinguished from supported points; fixed-state ETA sums not whole-run JCT proof. No tuning on failure.'))
 if not passed:
  (OUT/'CURRENT_TASK.md').write_text('STOPPED_OPPORTUNITY_GATE_FAILED. No formal18 launched. Preserve all gate logs; do not retune.\n',encoding='utf8');raise RuntimeError('ACTUAL_OPPORTUNITY_GATE_FAILED')
 with (OUT/'batch_console.log').open('x',encoding='utf8') as f:r=subprocess.run([sys.executable,'-X','utf8','-B',str(OUT/'run.py'),'batch'],stdout=f,stderr=subprocess.STDOUT)
 assert r.returncode==0
 records=[json.loads(l) for l in (OUT/'execution_manifest.jsonl').read_text().splitlines()];assert len(records)==18 and all(r['exit_code']==0 for r in records)
 with (OUT/'final_audit.log').open('x',encoding='utf8') as f:r=subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'maintenance/audit_arrival18_20261030.py'),'final'],stdout=f,stderr=subprocess.STDOUT)
 assert r.returncode==0
 print('ALL18_AUDITED_STOPPED',flush=True)
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='analyze':
  with Path(sys.argv[3]).open('x',encoding='utf8') as f:json.dump(analyze(Path(sys.argv[2])),f,indent=2,ensure_ascii=False)
 else:main()
