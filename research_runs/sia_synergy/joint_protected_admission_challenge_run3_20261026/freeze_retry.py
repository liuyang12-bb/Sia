import hashlib,json,zipfile
from pathlib import Path
OUT=Path(__file__).resolve().parent;OLD=OUT.parent/'joint_protected_admission_challenge_run2_20261026'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
p=json.loads((OLD/'preregistration.json').read_text())
for name in ['workload.csv','scenario.json']:
 assert sha(OUT/name)==sha(OLD/name);p['input_hashes'][str(OUT/name)]=sha(OUT/name)
tree=Path(p['tree'])
p['source_hashes']={str(f.relative_to(tree)):sha(f) for f in tree.rglob('*.py') if '__pycache__' not in f.parts}
p['tools_hashes']={f.name:sha(f) for f in OUT.glob('*.py')}
p['retry_context']=dict(previous_preregistration_sha256=sha(OLD/'preregistration.json'),
 reason='Artifact uses 8-GPU templates for small nodes; apply one physical-domain filter consistently to all five policies and rebuild index arrays',
 scenario_parameters_changed=False,response_changed=False,source_changed=True,objective_changed=False)
with zipfile.ZipFile(OUT/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
 for f in p['source_hashes']:z.write(tree/f,f)
with (OUT/'preregistration.json').open('x',encoding='utf-8') as f:json.dump(p,f,indent=2)
print('REFROZEN same data/capacity, consistent physical-domain adapter')
