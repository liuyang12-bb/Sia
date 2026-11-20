"""Freeze every byte of the27 legacy trees before reversible relocation."""
import hashlib,json,os,subprocess,time,zipfile
from pathlib import Path
from inventory import ROOT,OUT,sha,save,command

def files(root):
    return sorted(p for p in root.rglob('*') if p.is_file())

def main():
    inventory=json.loads((OUT/'inventory.json').read_text(encoding='utf-8'))
    assert json.loads((OUT/'before/summary.json').read_text(encoding='utf-8'))['valid']
    archive=ROOT/'_archive/legacy_code_20261027'
    assert not archive.exists(),'Archive destination already exists'
    names=[r['name'] for r in inventory['folders']]
    assert len(names)==27 and all(n.startswith('sia-') and '/' not in n and '\\' not in n for n in names)
    # No symlink/reparse roots or targets outside the explicitly named project.
    for n in names:
        p=ROOT/n;assert p.resolve().parent==ROOT.resolve() and p.is_dir()
        assert not (p.lstat().st_file_attributes & 0x400),'Reparse root: '+n
    hashes={};working_hashes={};states={};bundle_records=[]
    for name in names:
        d=ROOT/name
        if (d/'.git').exists():
            states[name]={k:command(['git','-C',str(d),*a]) for k,a in {
                'head':['rev-parse','HEAD'],'branch':['branch','--show-current'],
                'status':['status','--short','--branch','--untracked-files=all']}.items()}
            for kind,args in [('unstaged',['diff','--binary']),('staged',['diff','--cached','--binary'])]:
                path=OUT/'patches'/(name+'.'+kind+'.patch');path.parent.mkdir(exist_ok=True)
                proc=subprocess.run(['git','-C',str(d),*args],capture_output=True)
                assert proc.returncode==0
                with path.open('xb') as f:f.write(proc.stdout)
        for p in files(d):
            rel=p.relative_to(ROOT).as_posix();hashes[rel]=sha(p)
            if '.git' not in p.relative_to(d).parts:working_hashes[rel]=hashes[rel]
    # All original official worktree administrative indices/HEAD/logs are also
    # preserved, because their .git files point outside the checkout folders.
    admin=ROOT/'official-artifacts/.git/worktrees'
    if admin.exists():
        for p in files(admin):hashes[p.relative_to(ROOT).as_posix()]=sha(p)
    zip_path=OUT/'legacy_trees_before_cleanup.zip'
    with zipfile.ZipFile(zip_path,'x',zipfile.ZIP_DEFLATED) as z:
        for rel,h in hashes.items():
            assert sha(ROOT/rel)==h,rel
            z.write(ROOT/rel,rel)
    with zipfile.ZipFile(zip_path) as z:
        assert len(z.namelist())==len(hashes)
        for rel,h in hashes.items():assert hashlib.sha256(z.read(rel)).hexdigest().upper()==h,rel
    protected={}
    for name in ['official-artifacts','research_sources/Synergy']:
        root=ROOT/name
        protected[name]={p.relative_to(root).as_posix():sha(p) for p in files(root) if '.git' not in p.relative_to(root).parts}
    for repo in ['official-artifacts','sia-joint-experiments']:
        path=OUT/(repo+'.bundle')
        rec=command(['git','-C',str(ROOT/repo),'bundle','create',str(path),'--all']);assert rec['exit_code']==0,rec
        verify=command(['git','-C',str(ROOT/repo),'bundle','verify',str(path)]);assert verify['exit_code']==0,verify
        bundle_records.append(dict(repo=repo,sha256=sha(path),create=rec,verify=verify))
    plan=dict(root=str(ROOT),archive=str(archive),names=names,
        linked_joint=[n for n in names if (ROOT/n/'.git').is_file() and 'sia-joint-experiments' in (ROOT/n/'.git').read_text()],
        linked_official=[n for n in names if (ROOT/n/'.git').is_file() and 'official-artifacts' in (ROOT/n/'.git').read_text()],
        snapshots=[n for n in names if not (ROOT/n/'.git').exists()],main_repo='sia-joint-experiments')
    assert set(names)==set(plan['linked_joint']+plan['linked_official']+plan['snapshots']+[plan['main_repo']])
    save(OUT/'relocation_plan.json',plan)
    save(OUT/'backup_manifest.json',dict(created_unix=time.time(),zip=zip_path.name,zip_sha256=sha(zip_path),
        zip_readback_verified=True,files=hashes,working_files=working_hashes,git_before=states,
        protected_working_files=protected,bundles=bundle_records,plan_sha256=sha(OUT/'relocation_plan.json')))
    print('BACKUP_VERIFIED',len(names),'trees',len(hashes),'files',round(zip_path.stat().st_size/2**20,2),'MiB',flush=True)

if __name__=='__main__':main()
