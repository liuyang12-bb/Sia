"""Offline proposed scenario proof. No simulation or scheduler modification."""
import ast,csv,hashlib,json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
SRC=ROOT/'research_runs/sia_synergy/cc_breakpoint_study_20261029_run1/attempt3/source'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def get_matrix(name):
 for n in ast.walk(ast.parse((SRC/'inputs/models'/name).read_text())):
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='tput' for t in n.targets):
   return np.array(ast.literal_eval(n.value.args[0]))
 raise ValueError(name)
def main():
 reg={r['key']:r for r in json.loads((SRC/'inputs/registry.json').read_text())}
 refs={app:reg[app+'/aws:1:1'] for app in ['cifar10','bert']}
 matrices={app:get_matrix(r['template']) for app,r in refs.items()}
 def query(app,c,m):
  r=refs[app];x,y=c/r['phi_c'],m/r['phi_m'];x0,y0=3/r['phi_c'],62.5/r['phi_m']
  def f(a,b):return float(np.interp(a,[1,2,3,4,5,6,9,12,24],[np.interp(b,[20.83,62.5,125,187.5,250],row) for row in matrices[app]]))
  inside=lambda a,b:1-1e-12<=a<=24+1e-12 and 20.83-1e-12<=b<=250+1e-12
  assert inside(x,y) and inside(x0,y0)
  return {'cpu':c,'ram':m,'requested_source':[x,y],'reference_source':[x0,y0],
          'supported_including_floating_endpoint_tolerance':True,'rho':(f(x,y)/f(x0,y0))**.5}
 bcpu=refs['bert']['phi_c'];bram=20.83*refs['bert']['phi_m'];cram=62.5*refs['cifar10']['phi_m']
 pairs={'P':((3,62.5),(3,62.5)), 'CPU':((6-bcpu,62.5),(bcpu,62.5)),
        'MEM':((3,cram),(3,bram)), 'FULL':((6-bcpu,cram),(bcpu,bram))}
 result={'scope':'DESIGN AND STATIC MODEL ARITHMETIC ONLY; NOT A TRAINING RUN',
         'node':{'gpu_type':'aws','gpus':2,'synthetic_cpu_units':6,'ram_gb':125},
         'reference':'existing fixed_reference 3CPU/62.5GB; R0 and alpha0.5 unchanged',
         'pairs':{},'limitations':['Fixed one GPU per job is a mechanism witness, not an imposed main-experiment allocation.',
         'Full heterogeneous domain remains E-legacy; bert DGX reference CPU=0.75 is out of source range.',
         'Existing lower quota points may be clipped even though this witness is supported; do not silently remove them.',
         'Rate ratios are not measured JCT ratios; actual overlap, GPU choice and queue response need future gates.'],
         'warning':'SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE'}
 for name,(cq,bq) in pairs.items():
  assert cq[0]+bq[0]<=6+1e-12 and cq[1]+bq[1]<=125+1e-12
  result['pairs'][name]={'cifar10':query('cifar10',*cq),'bert':query('bert',*bq),
      'cpu_sum':cq[0]+bq[0],'ram_sum':cq[1]+bq[1]}
  assert result['pairs'][name]['bert']['rho']==1
 p=result['pairs'];result['conditional_ratios']={'FULL_over_CPU':p['FULL']['cifar10']['rho']/p['CPU']['cifar10']['rho'],
   'FULL_over_MEM':p['FULL']['cifar10']['rho']/p['MEM']['cifar10']['rho'],
   'fixed_geometry_CF_productive_time_reduction':1-1/p['FULL']['cifar10']['rho']}
 trace=ROOT/'official-artifacts/sia-simulator/workloads/saturn/workload-1.csv'
 rows=list(csv.DictReader(trace.open()))
 result['available_real_bert_records']=[r for r in rows if r['application']=='bert']
 result['candidate_load_rule_not_yet_preregistered']='first6 real CF and all6 real BERT rows; retain native training parameters. Arrival/heterogeneous execution gates still to define before running.'
 files=[SRC/'inputs/registry.json',SRC/'inputs/models/alexnet_1.py',SRC/'inputs/models/bert_1.py',trace]
 result['input_hashes']={str(p):sha(p) for p in files}
 with (OUT/'scenario_opportunity.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
 print(json.dumps({'ratios':{n:x['cifar10']['rho'] for n,x in p.items()},'conditional':result['conditional_ratios']},ensure_ascii=False))
if __name__=='__main__':main()
