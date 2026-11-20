import json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
v=load(P/'review2/version_check.json');assert v['snapshot_files']==799 and not v['byte_differences']
results=load(P/'review2/completed_results.json');assert len(results)==6
for r in results:assert r['metrics']['completed']==160
states=load(P/'review2/same_state_comparisons.json');assert len(states)<=6
for s in states:
 assert s['solver_status']==0
 assert {x['job'] for x in s['A']['jobs']}=={x['job'] for x in s['B']['jobs']}
 old={c['job']:c for c in s['A']['jobs']}
 for c in s['B']['jobs']:
  a=old[c['job']];assert (c['config'],c['placement'])==(a['config'],a['placement'])
  if a['held']:assert (c['cpu'],c['ram'])==(a['cpu'],a['ram'])
 for t,g in [('aws',4),('dgx-ext',8),('rtx',8)]:
  cols=[c for c in s['B']['jobs'] if c['gpu']==t]
  assert sum(c['gpus'] for c in cols)<=g
  assert sum(c['gpus']*c['cpu'] for c in cols)<=g*3+1e-7
  assert sum(c['gpus']*c['ram'] for c in cols)<=g*62.5+1e-7
 partial=load(P/'review2/partial_evidence.json');assert partial['readable_rounds']==78 and partial['overall_jct'] is None
batch=P.parent/'sia_r0_full160_comparison/attempt2';manifest=load(batch/'preregistration.json')
mismatches=[]
for name,h in manifest['input_hashes'].items():
 path=Path(name)
 if hashlib.sha256(path.read_bytes()).hexdigest().upper()!=h.upper():mismatches.append(name)
assert not mismatches,mismatches
with (P/'verification.json').open('x',encoding='utf-8') as f:json.dump(dict(valid=True,raw_complete_cells=6,static_states=len(states),input_hashes_checked=len(manifest['input_hashes']),source_files_equal=799,static_capacity_and_protection=True,formal_experiments='PAUSED',new_simulations=0),f,indent=2)
print('VERIFIED',len(states),'static states; six cached independent raw audits; 799 identical source files')
