import json,hashlib,zipfile,subprocess,sys,importlib.metadata
from pathlib import Path
from difflib import unified_diff
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2];OLD=OUT.parent/'joint_r_scale_check_20261025'
WORK=ROOT/'sia-joint-budgeted-adaptation/sia-simulator'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def main():
    assert json.loads((OUT/'red_result.json').read_text())['exit_code']!=0
    assert json.loads((OUT/'green_final_result.json').read_text())['exit_code']==0
    p=json.loads((OLD/'preregistration_v2.json').read_text())
    for path,h in p['hashes'].items():assert sha(path)==h,path
    assert all(importlib.metadata.version(k)==v for k,v in p['packages'].items())
    p['baseline_directory']=str(OLD);p['methods']=['joint_g'];p['tree']=str(WORK.parent)
    p['new_design']='Joint bundle exchange under original cost-aware fairness; remove per-job no-slowdown/within-lease payback veto; real locks, observed-training, provenance, delay cap unchanged; exact primary ties minimize actual startup cost'
    p['planned_cells']=[dict(size=n,response=r,id=f'n{n}_{r}_joint_g') for n in [8,20] for r in ['neutral','medium']]
    p['gate']='8 two cells all complete, raw independent audits valid before20; not conditioned on JCT advantage'
    p['no_search']='Exactly one design and four runs; no response/capacity/Q/D/p/lambda/seed/grid changes; no automatic performance retuning'
    p['primary_endpoint']='20-job Medium relative to SIA-AUX; include all regressions and censored cases'
    for f in WORK.rglob('*.py'):p['hashes'][str(f)]=sha(f)
    p['baseline_hashes']={str(f):sha(f) for f in OLD.rglob('*') if f.is_file()}
    old=ROOT/'sia-joint-scale-check/sia-simulator/joint_prototype/adaptive_service.py'
    new=WORK/'joint_prototype/adaptive_service.py'
    with (OUT/'legacy_hook_diff.patch').open('x',encoding='utf-8') as f:f.write(''.join(unified_diff(old.read_text().splitlines(True),new.read_text().splitlines(True),fromfile=str(old),tofile=str(new))))
    with zipfile.ZipFile(OUT/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in WORK.rglob('*.py'):z.write(f,f.relative_to(WORK.parent))
    with (OUT/'preregistration.json').open('x',encoding='utf-8') as f:json.dump(p,f,indent=2)
    print('FROZEN',len(p['hashes']),'source/input hashes; baselines',len(p['baseline_hashes']))
if __name__=='__main__':main()
