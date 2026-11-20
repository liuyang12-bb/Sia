"""Document reconstructible candidate-only diff; no scheduler/execution calls."""
import difflib,hashlib,json,shutil,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_breakpoint_study_20261029_run1';SRC=OUT/'source'
OLD=ROOT/'research_runs/sia_synergy/cc_resource_mixes42_20261028_run1/source'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def main():
 before=json.loads((OUT/'old_source_hashes.json').read_text());now=json.loads((OUT/'preregistration.json').read_text())['source_hashes']
 for n,h in before.items():assert sha(OLD/n)==h
 for n,h in now.items():assert sha(SRC/n)==h
 changed=[n for n,h in before.items() if now.get(n)!=h];added=sorted(set(now)-set(before));deleted=sorted(set(before)-set(now));assert not deleted
 assert set(changed)=={'source_manifest.json','sia-simulator/joint_prototype/joint_scheduler.py'}
 patch=[]
 for n in changed:
  if n.endswith('.py'):patch.extend(difflib.unified_diff((OLD/n).read_text(encoding='utf8').splitlines(True),(SRC/n).read_text(encoding='utf8').splitlines(True),fromfile='old/'+n,tofile='new/'+n))
 for n in added:
  patch.extend(difflib.unified_diff([],(SRC/n).read_text(encoding='utf8').splitlines(True),fromfile='/dev/null',tofile='new/'+n))
 with (OUT/'SOURCE_CHANGES.patch').open('x',encoding='utf8') as f:f.writelines(patch)
 with zipfile.ZipFile(OUT/'source_snapshot.zip') as z:assert z.testzip() is None
 data=dict(old_source_files=len(before),new_source_files=len(now),changed=changed,added=added,deleted=deleted,
  objective_files_byte_identical=[n for n in before if 'completion_continuity.py' in n or 'tail_recovery.py' in n or 'completion_time.py' in n or 'protected_admission.py' in n or n.endswith('/sia.py') or n.endswith('/job.py') or 'joint_profile.py' in n or 'r0_liveness.py' in n],
  no_git_repository='Both current resource-study and new canonical copy have no .git. Full284-file frozen source, patch, zip and SHA256 are authoritative, no invented Git commit.',
  snapshot_hash=sha(OUT/'source_snapshot.zip'),preregistration_hash=sha(OUT/'preregistration.json'))
 with (OUT/'SOURCE_VERSION.json').open('x',encoding='utf8') as f:json.dump(data,f,ensure_ascii=False,indent=2)
 for p in ROOT.glob('maintenance/*breakpoint*20261029*.py'):shutil.copyfile(p,OUT/p.name)
 text='''# Reproduction and scope

This is a candidate-only version copied from the sealed42-cell source. It is
not a new fairness/JCT objective and does not change R0/E-legacy, alpha, phi,
the workload, arrival sequence, capacity, service locks or startup costs.

Source authority: preregistration.json source_hashes, source_snapshot.zip,
SOURCE_CHANGES.patch, SOURCE_VERSION.json. Both source directories are source
copies without Git metadata; no commit hash is fabricated.

Canonical source used for legacy path-sensitive tests: cc-joint-breakpoint-study.
Runtime source: this directory/source. Both are byte-identical. New candidate
rule and adapter: source/breakpoint_study.py; generic default-off generation and
precommit hook: source/sia-simulator/joint_prototype/joint_scheduler.py.

Original233 tests (229old+4candidate) passed. Red missing-knot test, old run
failure and statistical-adapter fault replay are kept. small_attempt2 is the
complete8-cell gate evidence; first attempt is diagnostic, never mixed in.

Python: workspace .venv-sia-baseline/Scripts/python.exe -X utf8 -B.
Dependencies are pinned in source/dependencies.lock.txt and preregistration.
Do not blindly rerun in this output folder: all scripts refuse overwrites.
To reproduce independently create a new output root and update only output
locations in maintenance wrappers, retain all scientific settings and input
hashes, freeze it and recheck tests before executing.

Existing running batch entry: maintenance/run_breakpoint30_20261029.py batch.
Before resuming inspect CURRENT_TASK.md/execution_manifest.jsonl, verify no
duplicate runner, completed output hashes and nonempty partial directories.
Never overwrite partial or historical results. Resource gates2GiB/5GiB.

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE.
'''
 with (OUT/'REPRODUCE.md').open('x',encoding='utf8') as f:f.write(text)
 print(json.dumps(data,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
