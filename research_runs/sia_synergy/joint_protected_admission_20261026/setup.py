import hashlib,json,shutil,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
PARENT=ROOT/'sia-joint-exchange-netprogress';TREE=ROOT/'sia-joint-protected-admission'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
files=sorted(p for p in (PARENT/'sia-simulator').rglob('*.py') if '__pycache__' not in p.parts)
hashes={str(p.relative_to(PARENT)):sha(p) for p in files}
with zipfile.ZipFile(OUT/'parent_source.zip','x',zipfile.ZIP_DEFLATED) as z:
 for p in files:z.write(p,p.relative_to(PARENT))
(OUT/'parent_tracked.patch').write_bytes(subprocess.check_output(['git','-C',str(PARENT),'diff','--binary','HEAD']))
head=subprocess.check_output(['git','-C',str(PARENT),'rev-parse','HEAD'],text=True).strip()
subprocess.run(['git','-C',str(PARENT),'worktree','add','-b','exp/joint-protected-admission',str(TREE),head],check=True)
for p in files:
 q=TREE/p.relative_to(PARENT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
assert all(sha(TREE/p)==h for p,h in hashes.items())
with (OUT/'parent_hashes.json').open('x',encoding='utf-8') as f:json.dump(hashes,f,indent=2)
print('RESTORED',len(files),head)
