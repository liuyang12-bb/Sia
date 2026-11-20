"""Preserve actual parent source; restore only into newly created bridge tree."""
import hashlib,json,subprocess,zipfile,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
PARENT=ROOT/'sia-joint-budgeted-adaptation';TREE=ROOT/'sia-joint-version-bridge'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(p,v):
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
def main():
 files=sorted((PARENT/'sia-simulator').rglob('*.py'))
 files=[f for f in files if '__pycache__' not in f.parts]
 hashes={str(f.relative_to(PARENT)):sha(f) for f in files}
 with zipfile.ZipFile(OUT/'parent_actual_source.zip','x',zipfile.ZIP_DEFLATED) as z:
  for f in files:z.write(f,f.relative_to(PARENT))
 patch=subprocess.check_output(['git','-C',str(PARENT),'diff','--binary','HEAD'])
 with (OUT/'parent_tracked.patch').open('xb') as f:f.write(patch)
 for f in files:
  dst=TREE/f.relative_to(PARENT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst)
 assert all(sha(TREE/r)==h for r,h in hashes.items())
 save(OUT/'parent_restored_hashes.json',hashes)
 sel=ROOT/'research_runs/sia_synergy/joint_selective_admission_20261026/attempt2'
 prior=json.loads((sel/'preregistration.json').read_text())
 # Shared engine and official/full-intent baseline files must be byte-identical.
 shared=['job.py','cluster.py','sia.py','goodput.py','applications.py','simulator_config.py',
  'joint_prototype/full160.py','joint_prototype/joint_integration.py','joint_prototype/joint_scheduler.py',
  'joint_prototype/joint_profile.py','joint_prototype/r0_liveness.py','joint_prototype/small_mechanism.py',
  'joint_prototype/execution_audit.py','run_full160_cell.py']
 checks={r:dict(parent=sha(PARENT/'sia-simulator'/r),selective=sha(ROOT/'sia-joint-selective-admission/sia-simulator'/r)) for r in shared}
 assert all(v['parent']==v['selective'] for v in checks.values())
 save(OUT/'baseline_source_equivalence.json',checks)
 refs=[]
 for dataset,n in [('original8',8),('original20',20),('holdout20',20)]:
  for response in ['neutral','medium']:
   for method in ['sia_p','sia_aux','joint_g']:
    if dataset=='holdout20' and method=='joint_g':continue
    folder=(sel/f'{dataset}_{response}_{method}' if dataset=='holdout20' else
      ROOT/'research_runs/sia_synergy'/('joint_budgeted_adaptation_20261025/attempt3' if method=='joint_g' else 'joint_r_scale_check_20261025')/f'n{n}_{response}_{method}')
    assert (folder/'summary.json').is_file()
    refs.append(dict(dataset=dataset,response=response,method=method,folder=str(folder)))
 save(OUT/'reference_cells.json',refs)
 baseline_hashes={str(f):sha(f) for r in refs for f in Path(r['folder']).iterdir() if f.is_file()}
 save(OUT/'baseline_hashes.json',baseline_hashes)
 save(OUT/'input_seed.json',{k:prior[k] for k in ['config','registry','models','datasets','hashes','packages','python']})
 print('RESTORED',len(hashes),'BASELINES',len(refs),'HISTORICAL_FILES',len(baseline_hashes))
if __name__=='__main__':main()
