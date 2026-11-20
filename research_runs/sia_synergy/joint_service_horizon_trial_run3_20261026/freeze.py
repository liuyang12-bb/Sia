"""Create once-only input and source manifest before any trial cell starts."""
import hashlib
import json
import pathlib
import shutil
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / 'research_runs/sia_synergy/joint_protected_admission_challenge_run3_20261026'
TREE = ROOT / 'sia-joint-service-horizon/sia-simulator'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


assert not (HERE / 'preregistration.json').exists()
old = json.loads((OLD / 'preregistration.json').read_text(encoding='utf-8'))
assert sha(HERE / 'scenario.json') == sha(OLD / 'scenario.json')
assert sha(HERE / 'workload.csv') == sha(OLD / 'workload.csv')
assert sha(HERE / 'offline_reference_precheck.json') == sha(OLD / 'offline_reference_precheck.json')
assert json.loads((HERE / 'offline_reference_precheck.json').read_text())['valid']
for name, expected in old['source_hashes'].items():
    assert sha(TREE / name) == expected, name

files = sorted(p for p in TREE.rglob('*.py') if '__pycache__' not in p.parts)
source_hashes = {str(p.relative_to(TREE)).replace('\\', '/'): sha(p) for p in files}
with zipfile.ZipFile(HERE / 'source_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as out:
    for p in files:
        out.write(p, p.relative_to(TREE))

old.update(tree=str(TREE), source_hashes=source_hashes,
           runs=['G112', 'G111'],
           tools_hashes={p.name: sha(p) for p in [HERE / n for n in
               ['scenario.py', 'cell.py', 'metrics_legacy_adapter.py']]},
           baseline_preregistration_sha256=sha(OLD / 'preregistration.json'),
           change='G112 service-aware scoring only; G111 comparison uses identical new source tree',
           scientific_parameters_unchanged=True)
old['input_hashes'][str(OLD / 'workload.csv')] = old['input_hashes'].pop(str(HERE / 'workload.csv'), sha(HERE / 'workload.csv'))
# Keep the historical source-input hashes; replace the copied workload path.
for path in list(old['input_hashes']):
    if pathlib.Path(path).name == 'workload.csv':
        del old['input_hashes'][path]
old['input_hashes'][str(HERE / 'workload.csv')] = sha(HERE / 'workload.csv')
(HERE / 'preregistration.json').write_text(json.dumps(old, indent=2, ensure_ascii=False), encoding='utf-8')
print('FROZEN', len(files), len(old['input_hashes']), sha(HERE / 'preregistration.json'))
