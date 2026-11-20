import hashlib,json,subprocess,zipfile,sys,platform,importlib.metadata as md
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2];TREE=ROOT/'sia-joint-version-bridge'
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest().upper()
def save(f,v):
 with f.open('x',encoding='utf-8') as o:json.dump(v,o,indent=2,ensure_ascii=False)
def main():
 p=json.loads((OUT/'input_seed.json').read_text(encoding='utf-8'));p['tree']=str(TREE);p['branch']='exp/b3-version-bridge'
 p['base_commit']=subprocess.check_output(['git','-C',str(TREE),'rev-parse','HEAD'],text=True).strip()
 p['hashes']={f:h for f,h in p['hashes'].items() if str(ROOT/'sia-joint-selective-admission') not in f}
 science=['job.py','cluster.py','sia.py','goodput.py','applications.py','simulator_config.py',
  'joint_prototype/joint_integration.py','joint_prototype/joint_scheduler.py','joint_prototype/joint_profile.py',
  'joint_prototype/r0_liveness.py','joint_prototype/full160.py','joint_prototype/small_mechanism.py','run_full160_cell.py']
 equivalence={r:dict(bridge=sha(TREE/'sia-simulator'/r),scale=sha(ROOT/'sia-joint-scale-check/sia-simulator'/r)) for r in science}
 assert all(v['bridge']==v['scale'] for v in equivalence.values()),equivalence
 save(OUT/'original_baseline_source_equivalence.json',equivalence)
 for f in (TREE/'sia-simulator').rglob('*'):
  if f.is_file() and f.suffix in ['.py','.csv','.json'] and '__pycache__' not in f.parts:p['hashes'][str(f)]=sha(f)
 for f in p['datasets'].values():p['hashes'][f]=sha(Path(f))
 for k in ['config','registry']:p['hashes'][p[k]]=sha(Path(p[k]))
 for f,h in p['hashes'].items():assert sha(Path(f))==h,f
 p.update(interval=60,max_rounds=600,seed_profiles=[1,8],capacity='main_3CPU_62.5GB_per_GPU',
  responses={'neutral':0.,'medium':.5},objective='unchanged p=-0.5 lambda_n=1.1 bonus=0',
  free_admission='all jobs optional except exact checkpoint locks; no L2/age/bonus/protection',
  g_admission='unchanged FIFO L2 Q=300 physical productive seconds D=250 actual startup',
  memory_gate_gib=2,disk_gate_gib=5,formal104='PAUSED',new_code_committed=False,
  missing_mean='incomplete pairs: no full-JCT mean, no imputation, no extension',
  stop_rule='implementation/capacity/lock/audit failure stops; legitimate censoring retained; 8 gate requires independent correctness, not winning or completion',
  historical_reuse='hash+science-source identity; new execution-audit does not affect Job.step or scheduling',
  interpretation='SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE')
 p['runs']=[dict(id=f'{d}_{r}_{m}',dataset=d,response=r,method=m) for d in ['original8','original20','holdout20'] for r in ['neutral','medium'] for m in (['b3_free','joint_g'] if d=='holdout20' else ['b3_free'])]
 p['reference_cells']=json.loads((OUT/'reference_cells.json').read_text(encoding='utf-8'));p['baseline_hashes']=json.loads((OUT/'baseline_hashes.json').read_text(encoding='utf-8'))
 p['packages']={k:md.version(k) for k in p['packages']};p['python']=sys.version
 save(OUT/'preregistration.json',p)
 with zipfile.ZipFile(OUT/'execution_source.zip','x',zipfile.ZIP_DEFLATED) as z:
  for f in sorted((TREE/'sia-simulator').rglob('*.py')):
   if '__pycache__' not in f.parts:z.write(f,f.relative_to(TREE))
 with zipfile.ZipFile(OUT/'frozen_inputs.zip','x',zipfile.ZIP_DEFLATED) as z:
  for f in p['hashes']:
   f=Path(f)
   if f.suffix!='.py' or str(ROOT/'research_sources') in str(f):z.write(f,f.relative_to(ROOT))
 save(OUT/'source_version.json',dict(tree=str(TREE),base_commit=p['base_commit'],git_status=subprocess.check_output(['git','-C',str(TREE),'status','--short','--branch'],text=True),
  python=sys.version,platform=platform.platform(),packages=p['packages'],execution_source_sha256=sha(OUT/'execution_source.zip')))
 print('FROZEN',len(p['hashes']),'source/input hashes;',len(p['runs']),'new cells; references',len(p['reference_cells']))
if __name__=='__main__':main()
