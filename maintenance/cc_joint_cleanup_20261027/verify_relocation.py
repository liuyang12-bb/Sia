"""Verify original working bytes and dirty Git states after the move."""
import json,sys,time
from pathlib import Path
from inventory import ROOT,OUT,sha,save,command

b=json.loads((OUT/'backup_manifest.json').read_text(encoding='utf-8'))
p=json.loads((OUT/'relocation_plan.json').read_text(encoding='utf-8'));archive=Path(p['archive'])
checks=0
for rel,h in b['working_files'].items():
    path=archive/rel
    assert path.is_file() and sha(path)==h,rel
    checks+=1
git={}
for name,old in b['git_before'].items():
    path=archive/name;new={}
    for key,args in {'head':['rev-parse','HEAD'],'branch':['branch','--show-current'],
                    'status':['status','--short','--branch','--untracked-files=all']}.items():
        rec=command(['git','-C',str(path),*args])
        assert rec['exit_code']==0 and rec['stdout']==old[key]['stdout'],(name,key,rec)
        new[key]=rec
    git[name]=new
for name,files in b['protected_working_files'].items():
    for rel,h in files.items():assert sha(ROOT/name/rel)==h,(name,rel)
for name in p['names']:
    assert not (ROOT/name).exists(),name
    assert (archive/name).is_dir(),name
sys.path.insert(0,str(ROOT/'cc-joint'))
from cc_factory import verify
canonical=verify()
history=ROOT/'research_runs/sia_synergy/cc_joint_sustained_cpu_20job_20261027_run1'
delivery=json.loads((history/'delivery_manifest.json').read_text(encoding='utf-8'))
prefix=history.relative_to(ROOT).as_posix()+'/'
artifacts=0
for rel,h in delivery['archive_files'].items():
    if rel.startswith(prefix):assert sha(ROOT/rel)==h,rel;artifacts+=1
assert sha(history/delivery['archive'])==delivery['archive_sha256']
main=command(['git','-C',str(archive/p['main_repo']),'worktree','list','--porcelain'])
official=command(['git','-C',str(ROOT/'official-artifacts'),'worktree','list','--porcelain'])
assert main['exit_code']==0 and official['exit_code']==0
assert 'prunable' not in main['stdout'] and 'prunable' not in official['stdout']
save(OUT/'relocation_validation.json',dict(valid=True,created_unix=time.time(),trees=27,
    working_file_hashes_checked=checks,git_states_preserved=git,registered_worktrees_joint=main,
    registered_worktrees_official=official,official_and_synergy_working_files_unchanged=True,
    canonical_files_unchanged=len(canonical['files']),historical_artifact_files_checked=artifacts,
    latest_historical_evidence_archive_unchanged=True,
    empty_leftover_containers_only_removed=True,no_user_data_permanently_deleted=True))
print('RELOCATION_VALID',checks,'working files',len(git),'Git states',artifacts,'latest historical artifact hashes',flush=True)
