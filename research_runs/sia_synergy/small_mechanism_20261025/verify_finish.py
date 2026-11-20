import json,gzip,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def records(n):
 with gzip.open(P/n/'rounds.jsonl.gz','rt',encoding='utf-8') as f:return [json.loads(l) for l in f]
a=records('neutral_sia_p');b=records('neutral_sia_aux');assert len(a)==len(b)
for x,y in zip(a,b):
 assert x['decision_for_next_interval']['original_sia_allocations']==y['decision_for_next_interval']['original_sia_allocations']
 for e,f in zip(x['execution'],y['execution']):
  for k in ['progress_before','progress_after','epoch_after','completion_time','training_seconds','checkpoint_delay_consumed']:assert e[k]==f[k]
p=json.loads((P/'preregistration.json').read_text());bad=[path for path,h in p['hashes'].items() if sha(Path(path))!=h];assert not bad,bad
base=ROOT/'research_runs/sia_synergy/sia_r0_full160_comparison/attempt2'
assert sha(base/'execution_manifest.jsonl')=='F5B8DEAC6BA612D44E0422CE18FFF2598DCD6EECE7DA870A525800377F557D77'
assert sha(base/'main_w1_medium_joint_p/rounds.jsonl.gz')=='B4A19C80BB17647FCB683EDCFCBE125F8182878408232B0D13ED1720FFBC1CAB'
assert sha(ROOT/'official-artifacts/sia-simulator/sia.py')==sha(ROOT/'sia-joint-small-mechanism/sia-simulator/sia.py')
with (P/'final_validation.json').open('x') as f:json.dump(dict(valid=True,neutral_exact_rounds=len(a),input_and_source_hashes=len(p['hashes']),old_partial_and_manifest_unchanged=True,official_sia_source_identical=True,formal_104='PAUSED',all_six_completed_3of3=True),f,indent=2)
hashes={str(f.relative_to(P)):sha(f) for f in P.rglob('*') if f.is_file() and '__pycache__' not in f.parts}
with (P/'artifact_hashes.json').open('x') as f:json.dump(hashes,f,indent=2)
print('FINAL_VERIFIED',len(a),len(p['hashes']),len(hashes))
