import hashlib,json,platform,sys,zipfile,importlib.metadata
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2];T=R/'cc-joint-recovery-equivalent-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
plan=json.loads((O/'design_manifest.json').read_text());base=json.loads((O/'baseline_receipt.json').read_text())
hashes={str(p.relative_to(T)):sha(p) for p in sorted(T.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
changes=[n for n,h in hashes.items() if base['hashes'].get(n)!=h]
assert set(changes)=={r'sia-simulator\joint_prototype\clean_ablation.py',r'sia-simulator\joint_prototype\tail_recovery.py',r'sia-simulator\tests\test_recovery_equivalence.py'},changes
with zipfile.ZipFile(O/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
 for n in hashes:z.write(T/n,n)
old=json.loads((R/'research_runs/sia_synergy/cc_persistent_opportunity_experiment_20261101_run1/preregistration.json').read_text())
env={**old['environment'],'python':sys.version,'platform_current':platform.platform()}
for package,v in env['packages'].items():assert importlib.metadata.version(package)==v,package
p=dict(version=plan['version'],source=str(T),source_hashes=hashes,source_snapshot_sha256=sha(O/'source_snapshot.zip'),changes=changes,
 source_basis='complete343-file execution snapshot plus two-file correctness repair and seven tests; no Git HEAD substitute',
 input_hashes={str(x):sha(x) for x in (O/'inputs').iterdir()},environment=env,cells=plan['cells'],max_rounds=6000,interval=60,
 conditional_execution='first two old-scene repair cells unconditional after tests; new four only if fixed gate passed; never change failed design',
 design_sha256=sha(O/'design_manifest.json'),memory_gib=2,disk_gib=5,seed_profiles=[1,8],
 commands=dict(gate='python -X utf8 -B gate.py',cell='python -X utf8 -B run_cell.py --cell ID',pipeline='python -X utf8 -B pipeline.py'),
 script_hashes={x.name:sha(x) for x in O.glob('*.py')},label=plan['label'])
with (O/'preregistration.json').open('x',encoding='utf8') as f:json.dump(p,f,indent=2)
print('Frozen',len(hashes),sha(O/'preregistration.json'))
