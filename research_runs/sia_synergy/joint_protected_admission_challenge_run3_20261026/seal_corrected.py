import difflib,hashlib,json,subprocess,sys,zipfile
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
P=json.loads((OUT/'preregistration.json').read_text());TREE=Path(P['tree'])
PARENT=OUT.parent/'joint_protected_admission_20261026'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
parent_hashes=json.loads((PARENT/'parent_hashes.json').read_text())
with zipfile.ZipFile(PARENT/'parent_source.zip') as z:
 original={r.replace('\\','/'):z.read(r.replace('\\','/')).decode('utf-8') for r in parent_hashes}
delta=[];changed=[];added=[]
for p in sorted(TREE.rglob('*.py')):
 if '__pycache__' in p.parts:continue
 name='sia-simulator/'+p.relative_to(TREE).as_posix();text=p.read_bytes().decode('utf-8')
 old=original.get(name,'')
 if old!=text:
  (changed if name in original else added).append(name)
  delta.extend(difflib.unified_diff(old.splitlines(True),text.splitlines(True),fromfile='a/'+name if name in original else '/dev/null',tofile='b/'+name))
assert changed==['sia-simulator/joint_prototype/adaptive_service.py']
with (OUT/'source_delta_minimal.patch').open('x',encoding='utf-8',newline='') as f:f.writelines(delta)
with zipfile.ZipFile(OUT/'tools_snapshot_v2.zip','x',zipfile.ZIP_DEFLATED) as z:
 for p in OUT.glob('*.py'):z.write(p,p.name)
for path,h in P['source_hashes'].items():assert sha(TREE/path)==h,path
report=ROOT/'research_notes/synergy/JOINT_PROTECTED_ADMISSION_OPPORTUNITY_REPORT_20261026.md'
manifest=dict(status='MECHANISMS_TRIGGERED_ALL_COMPLETE_JCT_REGRESSION',source_tree=str(TREE),
 branch='exp/joint-protected-admission',base_head='2d5f9dbfcb7e7bfc2bfd9dc194b1838674a227c2',
 parent_snapshot=str(PARENT/'parent_source.zip'),parent_snapshot_sha256=sha(PARENT/'parent_source.zip'),
 changed_existing=changed,added=added,source_snapshot_sha256=sha(OUT/'source_snapshot.zip'),
 rebuild='base HEAD worktree + parent_source.zip root + source_delta_minimal.patch; or source_snapshot.zip under sia-simulator; verify preregistration source hashes',
 supersedes='DELIVERY_MANIFEST.json and source_delta.patch had CRLF/LF comparison noise; no executed source or result changed',
 report=str(report),report_sha256=sha(report),tests_executions=169,distinct_tests=167,
 files={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()})
with (OUT/'DELIVERY_MANIFEST_V2.json').open('x',encoding='utf-8') as f:json.dump(manifest,f,indent=2)
print('SEALED',len(manifest['files']),'changed',changed,'added',added)
