"""Final non-self-referential delivery seal, after raw audits and report."""
import datetime,hashlib,json,subprocess,zipfile,importlib.metadata as md,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest().upper()
def save(f,v):
 with f.open('x',encoding='utf-8') as o:json.dump(v,o,indent=2,ensure_ascii=False)
def main():
    p=json.loads((OUT/'preregistration.json').read_text(encoding='utf-8'));tree=Path(p['tree'])
    for f,h in p['hashes'].items():assert sha(Path(f))==h,f
    for f,h in p['baseline_hashes'].items():assert sha(Path(f))==h,f
    parent=json.loads((OUT/'parent_restored_hashes.json').read_text())
    for rel,h in parent.items():assert sha(ROOT/'sia-joint-budgeted-adaptation'/rel)==h,rel
    modified=[rel for rel,h in parent.items() if sha(tree/rel)!=h]
    assert not modified,modified
    added={str(f.relative_to(tree)):sha(f) for f in (tree/'sia-simulator').rglob('*.py') if '__pycache__' not in f.parts and str(f.relative_to(tree)) not in parent}
    save(OUT/'source_delta.json',dict(modified_parent_files=modified,added_sources=added,
        note='Tracked execution_audit diff versus base HEAD is inherited unchanged from G, not a new bridge modification.'))
    prior_s=json.loads((ROOT/'research_runs/sia_synergy/joint_selective_admission_20261026/attempt2/preregistration.json').read_text())
    s_hashes={f:h for f,h in prior_s['hashes'].items() if str(ROOT/'sia-joint-selective-admission') in f}
    assert s_hashes,'NO_JOINT_S_SOURCE_PREFIX_MATCH'
    for f,h in s_hashes.items():assert sha(Path(f))==h,f
    gpu={str(f.relative_to(tree/'sia-simulator')):sha(f) for f in (tree/'sia-simulator/traces').rglob('*.csv')}
    assert all(sha(ROOT/'sia-joint-budgeted-adaptation/sia-simulator'/r)==h for r,h in gpu.items())
    save(OUT/'actual_gpu_profile_hashes.json',gpu)
    with zipfile.ZipFile(OUT/'audit_tools_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in OUT.glob('*.py'):z.write(f,f.name)
    dependencies=subprocess.check_output([sys.executable,'-m','pip','freeze'])
    with (OUT/'requirements-lock.txt').open('xb') as f:f.write(dependencies)
    rows=[json.loads(line) for line in (OUT/'execution_manifest.jsonl').read_text().splitlines()]
    first_start=datetime.datetime.fromisoformat(rows[0]['finished_utc'])-datetime.timedelta(seconds=rows[0]['seconds'])
    last_finish=datetime.datetime.fromisoformat(rows[-1]['finished_utc'])
    save(OUT/'completion_record.json',dict(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        new_strategy_process_wall_seconds=sum(r['seconds'] for r in rows),new_cells=len(rows),tests=146,base_commit=p['base_commit'],
        first_child_start_utc=first_start.isoformat(),last_child_finish_utc=last_finish.isoformat(),
        batch_elapsed_seconds=(last_finish-first_start).total_seconds(),
        elapsed_through_seal_from_first_child_seconds=(datetime.datetime.now(datetime.timezone.utc)-first_start).total_seconds(),
        actual_branch=p['branch'],new_code_committed=False,source_hashes_verified=len(p['hashes']),old_artifact_hashes_verified=len(p['baseline_hashes']),
        parent_source_hashes_verified=len(parent),actual_gpu_profile_files=len(gpu),formal104='PAUSED_NOT_RESUMED',
        joint_s_source_hashes_unchanged=len(s_hashes),
        report_hashes={str(f):sha(f) for f in [OUT/'FINAL_REVIEW.md',ROOT/'research_notes/synergy/B3_VERSION_BRIDGE_REPORT.md']},
        git_status=subprocess.check_output(['git','-C',str(tree),'status','--short','--branch'],text=True),
        official_sia_status=subprocess.check_output(['git','-C',str(ROOT/'official-artifacts'),'status','--short','--branch'],text=True),
        official_synergy_status=subprocess.check_output(['git','-C',str(ROOT/'research_sources/Synergy'),'status','--short','--branch'],text=True)))
    # Evaluate all file hashes BEFORE opening the seal itself.
    hashes={str(f.relative_to(OUT)):sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and '__pycache__' not in f.parts and f.name!='DELIVERY_MANIFEST.json'}
    save(OUT/'DELIVERY_MANIFEST.json',dict(files=hashes,execution_source_sha256=sha(OUT/'execution_source.zip'),
        preregistration_sha256=sha(OUT/'preregistration.json'),interpretation=p['interpretation']))
    for rel,h in hashes.items():assert sha(OUT/rel)==h,rel
    print('SEALED',len(hashes),'FILES; WALL',sum(r['seconds'] for r in rows))
if __name__=='__main__':main()
