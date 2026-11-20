import json,hashlib,platform,importlib.metadata,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=Path(__file__).resolve().parent;W=ROOT/'sia-joint-adaptive-service'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
p=json.loads((P/'preregistration.json').read_text());changed=[path for path,h in p['hashes'].items() if sha(Path(path))!=h];assert not changed,changed
old=ROOT/'research_runs/sia_synergy/sia_r0_full160_comparison/attempt2'
assert sha(old/'execution_manifest.jsonl')=='F5B8DEAC6BA612D44E0422CE18FFF2598DCD6EECE7DA870A525800377F557D77'
assert sha(old/'main_w1_medium_joint_p/rounds.jsonl.gz')=='B4A19C80BB17647FCB683EDCFCBE125F8182878408232B0D13ED1720FFBC1CAB'
assert sha(ROOT/'official-artifacts/sia-simulator/sia.py')==sha(W/'sia-simulator/sia.py')
files={str(f.relative_to(P)):sha(f) for f in P.rglob('*') if f.is_file() and '__pycache__' not in f.parts}
payload=dict(valid=True,checked_frozen_hashes=len(p['hashes']),base_commit=subprocess.check_output(['git','-C',str(W),'rev-parse','HEAD']).decode().strip(),git_status=subprocess.check_output(['git','-C',str(W),'status','--short']).decode(),source_snapshot_hash=sha(P/'source_snapshot.zip'),python=platform.python_version(),dependencies={k:importlib.metadata.version(k) for k in ['numpy','scipy','cvxpy','cylp','pandas','autograd','psutil']},old_batch_and_partial_preserved=True,formal_batch='PAUSED',files=files)
with (P/'final_manifest.json').open('x') as f:json.dump(payload,f,indent=2)
print('SEALED',len(files),'artifacts;',len(p['hashes']),'frozen hashes verified')
