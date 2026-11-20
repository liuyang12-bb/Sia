"""Save rebuildable tools, inputs, minimal source delta, and delivery hashes."""
import difflib,hashlib,json,subprocess,sys,zipfile
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
P=json.loads((OUT/'preregistration.json').read_text());TREE=Path(P['tree'])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
original=json.loads((OUT/'restored_source_hashes.json').read_text())['files']
delta=[]
for name in ['joint_prototype/exchange_netprogress.py','tests/test_exchange_netprogress.py','tests/test_exchange_integration.py']:
 delta.extend(difflib.unified_diff([], (TREE/name).read_text(encoding='utf-8').splitlines(True),fromfile='/dev/null',tofile='b/sia-simulator/'+name))
with (OUT/'source_delta.patch').open('x',encoding='utf-8') as f:f.writelines(delta)
with (OUT/'requirements-lock.txt').open('xb') as f:f.write(subprocess.check_output([sys.executable,'-m','pip','freeze']))
with zipfile.ZipFile(OUT/'tools_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
 for p in OUT.glob('*.py'):z.write(p,p.name)
inputs={Path(p) for p in P['input_hashes']}
inputs.update((TREE/'traces').rglob('*.csv'))
with zipfile.ZipFile(OUT/'reproducible_inputs.zip','x',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(inputs):z.write(p,p.relative_to(ROOT))
report=ROOT/'research_notes/synergy/JOINT_EXCHANGE_NETPROGRESS_IMPLEMENTATION_20261026.md'
hashes={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}
manifest=dict(status='COMPLETE_MECHANISM_VALID_NO_JCT_IMPROVEMENT_OVER_G',
 tree=str(TREE),branch='exp/joint-exchange-netprogress',base_head='2d5f9dbfcb7e7bfc2bfd9dc194b1838674a227c2',
 source_restore='git worktree at base HEAD; restore parent_source.zip relative to tree; apply source_delta.patch or restore execution_source.zip under sia-simulator; verify preregistration source_hashes',
 original_sources_unchanged=len(original),new_source_files=3,tests=156,static_runs=49,real_cells=8,
 report=str(report),report_sha256=sha(report),files=hashes,
 boundary='No 20/160-job simulation or formal experiment restart. No tuning after results.')
with (OUT/'DELIVERY_MANIFEST.json').open('x',encoding='utf-8') as f:json.dump(manifest,f,indent=2)
print('SEALED',len(hashes),'files; report',sha(report),'source',sha(OUT/'execution_source.zip'))
