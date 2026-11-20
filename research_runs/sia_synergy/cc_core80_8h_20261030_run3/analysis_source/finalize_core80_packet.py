"""Verify immutable evidence, archive analysis sources, and seal the new packet."""
import datetime,hashlib,json,shutil,subprocess,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research_runs/sia_synergy/cc_core80_8h_20261030_run3'
TREE=ROOT/'cc-joint-core80-8h-v3'
REPORT=ROOT/'research_notes/synergy/CC_CORE80_8H_REPORT_20261030.md'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest().upper()

def save(path,obj):
    with path.open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)

def main():
    p=json.loads((OUT/'preregistration.json').read_text());canon=json.loads((OUT/'canonical_result_paths.json').read_text());budget=json.loads((OUT/'budget.json').read_text());frozen=json.loads((OUT/'freeze_hashes.json').read_text())
    assert time.time()<budget['deadline_unix']
    assert sha(OUT/'preregistration.json')==frozen['preregistration_sha256']
    assert sha(OUT/'frozen_source_snapshot.zip')==frozen['source_snapshot_sha256']
    assert sha(Path(canon['manifest']))==canon['manifest_sha256']
    assert all(sha(TREE/k)==v for k,v in p['source_hashes'].items())
    assert all(sha(Path(k))==v for k,v in p['input_hashes'].items())
    baseline=json.loads((OUT/'baseline_source_hashes.json').read_text())
    assert all(sha(TREE/k)==v for k,v in baseline.items())
    with zipfile.ZipFile(OUT/'frozen_source_snapshot.zip') as z:
        names={n for n in z.namelist() if not n.endswith('/')}
        assert names==set(p['source_hashes'])
        assert all(hashlib.sha256(z.read(k)).hexdigest().upper()==v for k,v in p['source_hashes'].items())
    checked=0
    for directory in (OUT,OUT/'execution_recovery1'):
        ledger=[json.loads(x) for x in (directory/'execution_manifest.jsonl').read_text().splitlines()]
        for event in ledger:
            if event['event']=='CELL_END':
                for path,h in event['output_hashes'].items():
                    assert sha(directory/path)==h,path
                    checked+=1
        if directory.name=='execution_recovery1':
            end=next(e for e in reversed(ledger) if e['event']=='BATCH_END')
    guard=json.loads((OUT/'diagnostic_guard_result.json').read_text())
    assert end['unix']<budget['formal_stop_unix'] and guard['end_unix']<budget['formal_stop_unix']
    audit=json.loads((OUT/'audit/coverage.json').read_text());assert audit['attempted']==11 and audit['violations']==0
    repos={}
    for name in ('official-artifacts','research_sources/Synergy'):
        repo=ROOT/name
        def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],encoding='utf8').strip()
        assert not git('status','--porcelain','--untracked-files=no'),name
        repos[name]=dict(head=git('rev-parse','HEAD'),branch=git('branch','--show-current'),status=git('status','--short','--branch'),tracked_changes=[])
    assert REPORT.read_bytes()==(OUT/'REPORT.md').read_bytes()
    assert not (OUT/'analysis_source').exists()
    archive=OUT/'analysis_source';archive.mkdir()
    scripts=sorted(f for f in (ROOT/'maintenance').glob('*.py') if 'core80' in f.name)
    for script in scripts:shutil.copyfile(script,archive/script.name)
    save(OUT/'analysis_source_hashes.json',{p.name:sha(p) for p in archive.iterdir()})
    save(OUT/'final_protection_verification.json',dict(source_files_verified=len(p['source_hashes']),baseline_files_verified=len(baseline),snapshot_contents_equal=True,input_files_verified=len(p['input_hashes']),
        closed_output_receipt_hashes_verified=checked,official_repositories=repos,main_session4015_exit=0,diagnostic_session85570_exit=0,
        process_scope='Managed sessions returned completion and dispatcher/guard waited on their own children; not a claim about unrelated user processes.',
        semantic='Guard exit0 means expected failure evidence captured, not a successful FULL experiment.'))
    with (OUT/'CURRENT_TASK.md').open('a',encoding='utf8') as f:
        f.write('\n\n# TERMINAL: CORE80 TASK COMPLETED AND STOPPED\nAll11core cells attempted;5complete/3failed/3compute-timeout;4optional not launched. Main4015 and diagnostic85570 ended. Read REPORT.md, audit/termination_analysis.json and final_receipt.json. No training task remains authorized in this turn; do NOT resume104/160 or retry numerical failures automatically. Source/parameters/history preserved. Wait for researcher review.\n')
    with (OUT/'REPRODUCE.md').open('a',encoding='utf8') as f:
        f.write('\n\n## Terminal outcome and analysis reconstruction\n\nMain11 were all attempted:5full80completion,3numerical/lexicographic failures,3confirmed external COMPUTE_TIMEOUT. Optional4not launched. Exact receipts: audit/termination_analysis.json; overviewCSV: audit/core80_coverage.csv. Preserve audit/raw_recomputed_metrics.json: its generic timeout-or-interruption label is refined by the separate watchdog-based classification, never by rewriting raw logs.\n\nAnalysis scripts are copied with hashes into analysis_source/. Run audit_core80_8h_20261030.py only against a NEW audit destination when reproducing; its default refuses overwrite. Then supplement_core80_results.py and report_core80_8h_20261030.py. Existing output files are immutable. Report generation time and final seal time differ; final_receipt.json records total measured wall through final packaging.\n\nOne bounded correctness reproduction after the matrix used guard_core80_failure_diagnostic.py, max5boundaries/720sec, and captured the same FULL failure. inspect_core80_solver_model.py only multiplies captured vectors by frozen constraints; it does not call a solver or train. Complete modelSHA and prefix equivalence are preserved. No tolerance/target/model repair was applied.\n')
    # All worker logs and analysis documents are now closed. Exclude only the
    # receipt/seal which are created after hashing, then add receipt's own hash.
    files={str(f.relative_to(OUT)):sha(f) for f in sorted(OUT.rglob('*')) if f.is_file()}
    now=time.time();receipt=dict(status='STOPPED_WITH_PARTIAL_MATRIX',start_unix=budget['start_unix'],final_packaging_unix=now,
        final_packaging_utc=datetime.datetime.fromtimestamp(now,datetime.timezone.utc).isoformat(),elapsed_seconds=now-budget['start_unix'],elapsed_hours=(now-budget['start_unix'])/3600,
        deadline_unix=budget['deadline_unix'],within_eight_hours=now<budget['deadline_unix'],formal_batch_end_unix=end['unix'],last_bounded_diagnostic_end_unix=guard['end_unix'],
        core_attempted=11,core_complete=5,core_failed=3,core_compute_timeout=3,optional_attempted=0,capacity_execution_violations=0,tests_passed=276,
        source_algorithm_baseline_unchanged=True,parameters_tuned=False,training_work_shortened=False,no_104_or_160_batch_started=True,
        report=str(REPORT),report_sha256=sha(REPORT),scope='SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE')
    save(OUT/'final_receipt.json',receipt);files['final_receipt.json']=sha(OUT/'final_receipt.json')
    save(OUT/'final_seal.json',dict(hash_algorithm='SHA256',files=files,external_report={str(REPORT):sha(REPORT)},source_snapshot_sha256=frozen['source_snapshot_sha256'],
                                  note='Seal excludes itself; all included output writers were closed first. No old run1/run2 or official source was overwritten.'))
    print(json.dumps(dict(receipt=receipt,sealed_files=len(files),seal_sha256=sha(OUT/'final_seal.json')),ensure_ascii=False))
if __name__=='__main__':main()
