import hashlib,json
from pathlib import Path
OUT=Path(__file__).resolve().parent
OLD=OUT.parent/'joint_protected_admission_challenge_20261026'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
p=json.loads((OLD/'preregistration.json').read_text())
for name in ['workload.csv','scenario.json']:
 assert sha(OUT/name)==sha(OLD/name)
 p['input_hashes'][str(OUT/name)]=sha(OUT/name)
p['input_hashes'][str(OUT/'offline_reference_precheck.json')]=sha(OUT/'offline_reference_precheck.json')
p['tools_hashes']={f.name:sha(f) for f in OUT.glob('*.py')}
p['retry_context']=dict(previous_preregistration_sha256=sha(OLD/'preregistration.json'),
 reason='Gate logger read waiting instead of the actual unallocated_jobs API key; failure preserved at t60',
 algorithm_changed=False,scenario_parameters_changed=False,input_copy_equal=True)
with (OUT/'preregistration.json').open('x',encoding='utf-8') as f:json.dump(p,f,indent=2)
print('REFROZEN SAME SCENARIO; LOGGER CORRECTION ONLY')
