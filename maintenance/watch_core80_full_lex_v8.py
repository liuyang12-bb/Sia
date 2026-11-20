"""Observe the one acceptance/launch pipeline, then audit/report/seal. No launch."""
import hashlib,json,subprocess,sys,time,traceback,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'research_runs/sia_synergy/cc_core80_full_lex_repair_20261031_run5'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def run(label,script):
 with (O/(label+'.log')).open('xb') as f:r=subprocess.run([sys.executable,'-X','utf8','-B',str(R/'maintenance'/script)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
 print(label,'EXIT',r.returncode,flush=True)
 if r.returncode:raise RuntimeError('POSTPROCESS_FAILURE '+label)
def main():
 last=None;deadline=time.time()+4*3600
 while True:
  if (O/'USER_STOP.json').exists():print('USER_STOP; do not launch anything',flush=True);return
  manifest=O/'execution_manifest.jsonl'
  if not manifest.exists():
   if 'Traceback (most recent call last)' in (O/'acceptance_driver.log').read_text(encoding='utf8'):raise RuntimeError('ACCEPTANCE_FAILURE_NO_FORMAL_RUN')
   if time.time()>deadline:raise RuntimeError('NO_FORMAL_LAUNCH_OBSERVED')
  else:
   ledger=[json.loads(x) for x in manifest.read_text().splitlines() if x]
   if ledger[-1]['event']=='BATCH_END':break
   ends=[x for x in ledger if x['event']=='CELL_END'];starts=[x for x in ledger if x['event']=='CELL_START'];cell=starts[-1]['cell'] if starts else None;progress={}
   if cell:
    files=sorted((O/'cells'/cell).glob('progress_*.json'))
    if files:
     try:progress=json.loads(files[-1].read_text())
     except json.JSONDecodeError:pass
   state=(len(ends),cell,progress.get('round'),progress.get('completed'))
   if state!=last:
    print('WATCH',json.dumps(dict(ended=len(ends),running=cell,round=progress.get('round'),completed=progress.get('completed'),wall=progress.get('elapsed_wall_seconds'))),flush=True);last=state
  time.sleep(30)
 p=json.loads((O/'preregistration.json').read_text());T=Path(p['source_root'])
 assert all(sha(T/n)==v for n,v in p['source_hashes'].items());assert all(sha(Path(n))==v for n,v in p['input_hashes'].items())
 for e in ledger:
  if e['event']=='CELL_END':
   for n,v in e['output_hashes'].items():assert sha(O/n)==v,('OUTPUT_CHANGED',n)
 run('raw_audit','audit_core80_full_lex_v8.py');run('final_report','report_core80_full_lex_v8.py')
 scripts=['prepare_core80_full_lex_repair.py','probe_core80_startup37.py','prepare_core80_full_lex_v8.py','supersede_core80_lex_development_v4.py','bounded_core80_full_lex_v8.py','accept_core80_full_lex_v8.py','freeze_launch_core80_full_lex_v8.py','audit_core80_full_lex_v8.py','report_core80_full_lex_v8.py','watch_core80_full_lex_v8.py']
 with zipfile.ZipFile(O/'maintenance_scripts.zip','x',zipfile.ZIP_DEFLATED) as z:
  for n in scripts:z.write(R/'maintenance'/n,'maintenance/'+n)
 with (O/'REBUILD.md').open('x',encoding='utf8') as f:f.write('Restore source_snapshot.zip to an isolated source folder, verify source_hashes.json and input_hashes.json. This is an actual329-file snapshot, not a bare Git HEAD. Source_changes.patch is against the preserved315-file base. Captured rank fixture has independent hash/provenance. Restore dependency lock/external test dependencies in the recorded layout. Run335tests, bounded FULL45/A1020/PROP12 and originalSia3round probe; reference maintenance_scripts.zip. Formal reruns require a new output directory and explicit frozen preregistration. Never reuse raw output paths or alter old histories. Objective/grid/response/work/capacity unchanged; only exact feasibility presolve/equivalent epsilon startup proofs and opt-in engineering-failure stopping. This run is now closed, no automatic next scenario.\n')
 with (O/'CURRENT_TASK.md').open('a',encoding='utf8') as f:f.write('\nBATCH ENDED; independent raw audit/report complete. Read REPORT.md,coverage.json and result_summary.json. All frozen source/input/cell-end receipts verified; no next experiment automatically authorized.\n')
 save(O/'final_hashes.json',{str(q.relative_to(O)):sha(q) for q in O.rglob('*') if q.is_file() and q.name not in ('final_hashes.json','seal_receipt.json')})
 save(O/'seal_receipt.json',dict(unix=time.time(),final_hashes_sha256=sha(O/'final_hashes.json'),source_sha256=p['source_snapshot_sha256'],source_files=len(p['source_hashes']),inputs=len(p['input_hashes']),all_cell_end_receipts_verified=True))
 print('ALL_POSTPROCESSING_COMPLETE',O/'REPORT.md',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  save(O/'postprocess_failure.json',dict(unix=time.time(),error=repr(e),traceback=traceback.format_exc()));raise
