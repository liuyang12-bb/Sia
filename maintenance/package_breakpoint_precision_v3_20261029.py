"""Immutable source/AST equivalence evidence for active numeric-correct run."""
import ast,difflib,hashlib,json,shutil,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'research_runs/sia_synergy/cc_breakpoint_study_20261029_run1';OUT=BASE/'attempt3';SRC=OUT/'source'
OLD=ROOT/'research_runs/sia_synergy/cc_resource_mixes42_20261028_run1/source'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def node(path,name):return ast.dump(next(n for n in ast.parse(path.read_text(encoding='utf8')).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name),include_attributes=False)
def main():
 p=json.loads((OUT/'preregistration.json').read_text());old=json.loads((BASE/'old_source_hashes.json').read_text())
 for n,h in p['source_hashes'].items():assert sha(SRC/n)==h
 for n,h in old.items():assert sha(OLD/n)==h
 changed=[n for n,h in old.items() if p['source_hashes'].get(n)!=h];added=sorted(set(p['source_hashes'])-set(old))
 srcfile='sia-simulator/joint_prototype/completion_continuity.py'
 equivalent={n:node(SRC/srcfile,n)==node(OLD/srcfile,n) for n in ['completion_priority','ContinuityAdmission','CompletionContinuityPolicy']}
 assert all(equivalent.values())
 same=[n for n in old if sha(SRC/n)==old[n] and any(t in n for t in ['joint_profile.py','r0_liveness.py','completion_time.py','tail_recovery.py','adaptive_service.py','protected_admission.py','job.py','/sia.py','inputs/'])]
 patch=[]
 for n in changed:
  if n.endswith('.py'):patch.extend(difflib.unified_diff((OLD/n).read_text(encoding='utf8').splitlines(True),(SRC/n).read_text(encoding='utf8').splitlines(True),fromfile='old42/'+n,tofile='new/'+n))
 for n in added:
  if n.endswith('.py'):patch.extend(difflib.unified_diff([],(SRC/n).read_text(encoding='utf8').splitlines(True),fromfile='/dev/null',tofile='new/'+n))
 with (OUT/'SOURCE_CHANGES.patch').open('x',encoding='utf8') as f:f.writelines(patch)
 with zipfile.ZipFile(OUT/'source_snapshot.zip') as z:assert z.testzip() is None
 data=dict(source_files=len(p['source_hashes']),parent42_files=len(old),changed=changed,added=added,objective_and_continuity_AST_identical=equivalent,
  response_training_admission_data_byte_identical=same,snapshot_sha256=sha(OUT/'source_snapshot.zip'),preregistration_sha256=sha(OUT/'preregistration.json'),
  source_kind='Complete isolated source copy, not a Git repository. No fabricated commit; per-file SHA256 and zip are authority.',
  numeric_scope='Same mathematical objective and constraints; reversible common cost units and optional row conditioning, independent unscaled validation. No weight/bonus/grid/model/capacity/workload change after freeze.',
  historical_bridge_caveat='New versus historical42 also includes numeric correctness repair. Do not attribute all historical changes uniquely to grid. Same-version six-method quota ablations are primary; original15 completed prefix remains separate diagnostic version.')
 with (OUT/'SOURCE_VERSION.json').open('x',encoding='utf8') as f:json.dump(data,f,ensure_ascii=False,indent=2)
 for path in ROOT.glob('maintenance/*breakpoint*20261029*.py'):
  if 'precision_v3' in path.name:shutil.copyfile(path,OUT/path.name)
 with (OUT/'REPRODUCE.md').open('x',encoding='utf8') as f:f.write('''# Reproduction

Primary output is attempt3 only; first15 cells and attempt2 bounded failures
are never mixed into the30-cell result. Same288-file source/inputs are sealed
in preregistration.json/source_snapshot.zip/SOURCE_VERSION.json. Objective and
continuity AST equality with old42 is checked; numerical representation is
corrected without changing preferences or physical constraints. Both original
solver failures and fixed16-round replay are preserved in parent directories.

Runtime source: this directory/source. Canonical legacy-test location:
D:/科研启航/Sia/cc-joint-breakpoint-precision. No Git HEAD is fabricated.
Python: workspace .venv-sia-baseline/Scripts/python.exe -X utf8 -B.
Dependencies: source/dependencies.lock.txt and preregistration packages.
Current batch entry: maintenance/run_breakpoint30_precision_v3_20261029.py batch.
Before resuming verify process status, sealed prefix hashes, memory2GiB/disk5GiB;
never overwrite a nonempty partial directory or start duplicate processes.

To reproduce from scratch use a NEW output root; update wrapper output paths
only, keep source/input hashes and scientific conditions, freeze again. Scripts
refuse reruns in existing outputs. All6000-round limits remain unchanged.
After all30: audit_breakpoint30_precision_v3_20261029.py, then
describe_breakpoint30_precision_v3_20261029.py. Audit does not run training.

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE.
''')
 print('PACKAGED_V3',len(p['source_hashes']),equivalent,sha(OUT/'preregistration.json'))
if __name__=='__main__':main()
