"""Copy/verify actual source, capture a reproducible parent, run bounded tests."""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys, time, zipfile, difflib

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
PARENT=ROOT/'sia-joint-completion-continuity/sia-simulator'
TREE=ROOT/'sia-joint-net-completion/sia-simulator'
OLD=OUT.parent/'joint_completion_continuity_20261026_run2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
def freeze_parent():
    old=json.loads((OLD/'preregistration.json').read_text(encoding='utf-8'))
    for rel,h in old['source_hashes'].items():
        assert sha(PARENT/rel)==h,(rel,'parent differs from executed snapshot')
        assert sha(TREE/rel)==h,(rel,'new tree copy differs')
    for rel,h in old['input_hashes'].items():assert sha(OUT/rel)==h,rel
    save(OUT/'parent_verification.json',dict(parent_source_hashes=old['source_hashes'],
        input_hashes=old['input_hashes'],parent_preregistration_sha256=sha(OLD/'preregistration.json'),
        parent_code='actual snapshot (not incomplete historical Git HEAD)',verified_unix=time.time()))
    with zipfile.ZipFile(OUT/'parent_actual_source.zip','x',zipfile.ZIP_DEFLATED) as z:
        for rel in old['source_hashes']:z.write(PARENT/rel,rel)
    print('PARENT_VERIFIED',len(old['source_hashes']))
def test(label,pattern):
    cmd=[sys.executable,'-X','utf8','-B','-m','unittest','discover','-s','tests','-p',pattern,'-v']
    start=time.time()
    with (OUT/(label+'.log')).open('x',encoding='utf-8') as f:
        p=subprocess.run(cmd,cwd=TREE,stdout=f,stderr=subprocess.STDOUT)
    save(OUT/(label+'.json'),dict(command=cmd,cwd=str(TREE),exit_code=p.returncode,
        elapsed_seconds=time.time()-start,log_sha256=sha(OUT/(label+'.log'))))
    print(label,p.returncode,flush=True)
    return p.returncode
def snapshot():
    old=json.loads((OUT/'parent_verification.json').read_text(encoding='utf-8'))['parent_source_hashes']
    hashes={str(p.relative_to(TREE)):sha(p) for p in TREE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    changes=[];diff=[]
    for rel,h in hashes.items():
        if old.get(rel)==h:continue
        changes.append(dict(file=rel,before=old.get(rel),after=h))
        before=(PARENT/rel).read_text(encoding='utf-8').splitlines(True) if rel in old else []
        after=(TREE/rel).read_text(encoding='utf-8').splitlines(True)
        diff+=difflib.unified_diff(before,after,fromfile='parent/'+rel,tofile='current/'+rel)
    with (OUT/'source.patch').open('x',encoding='utf-8') as f:f.writelines(diff)
    save(OUT/'source_changes.json',dict(changes=changes,source_hashes=hashes,
        source_fingerprint=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest().upper()))
    print('SOURCE_SNAPSHOT',len(hashes),'changes',len(changes))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['parent','test','snapshot']);a.add_argument('--label');a.add_argument('--pattern',default='test_*.py');v=a.parse_args()
    if v.mode=='parent':freeze_parent()
    elif v.mode=='snapshot':snapshot()
    else:sys.exit(test(v.label,v.pattern))
