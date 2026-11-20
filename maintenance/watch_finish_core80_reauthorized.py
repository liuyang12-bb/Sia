"""Observe the existing batch, then raw-audit/report. NEVER launch a cell."""
import hashlib,json,subprocess,sys,time,traceback,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'research_runs/sia_synergy/cc_core80_repair_full_20261031_run1';F=R/'research_runs/sia_synergy/cc_core80_repair_20261031_run3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def run(label,script):
 with (O/(label+'.log')).open('xb') as f:r=subprocess.run([sys.executable,'-X','utf8','-B',str(R/'maintenance'/script)],cwd=R,stdout=f,stderr=subprocess.STDOUT)
 print(label,'EXIT',r.returncode,flush=True)
 if r.returncode:raise RuntimeError('POSTPROCESS_FAILURE '+label)
def main():
 old=None
 while True:
  if (O/'USER_STOP.json').exists():print('USER_STOP: no postprocessing launch',flush=True);return
  ledger=[json.loads(x) for x in (O/'execution_manifest.jsonl').read_text().splitlines() if x]
  if ledger[-1]['event']=='BATCH_END':break
  end=[x for x in ledger if x['event']=='CELL_END'];start=[x for x in ledger if x['event']=='CELL_START'];cell=start[-1]['cell'] if start else None;progress={}
  if cell:
   points=sorted((O/'cells'/cell).glob('progress_*.json'))
   if points:
    try:progress=json.loads(points[-1].read_text())
    except json.JSONDecodeError:pass
  state=(len(end),cell,progress.get('round'),progress.get('completed'))
  if state!=old:
   print('WATCH',json.dumps(dict(complete_cells=len(end),running=cell,round=progress.get('round'),completed_jobs=progress.get('completed'),elapsed_wall_seconds=progress.get('elapsed_wall_seconds')),ensure_ascii=False),flush=True);old=state
  time.sleep(30)
 p=json.loads((O/'preregistration.json').read_text());assert all(sha(Path(p['source_root'])/n)==v for n,v in p['source_hashes'].items());assert all(sha(Path(n))==v for n,v in p['input_hashes'].items())
 for event in ledger:
  if event['event']=='CELL_END':
   for n,v in event['output_hashes'].items():assert sha(O/n)==v,('OUTPUT_CHANGED',n)
 run('raw_audit','audit_core80_reauthorized.py');run('final_report','report_core80_reauthorized.py')
 # Copy immutable execution/rebuild materials without modifying any source.
 for name in ('source_snapshot.zip','source_hashes.json','source_changes.patch','dependencies.lock.txt','environment.json','external_test_dependencies.zip','external_test_dependency_hashes.json','REBUILD.md'):
  if (F/name).exists():
   with (O/name).open('xb') as f:f.write((F/name).read_bytes())
 with zipfile.ZipFile(O/'analysis_scripts.zip','x',zipfile.ZIP_DEFLATED) as z:
  for name in ('launch_reauthorized_core80.py','audit_core80_reauthorized.py','report_core80_reauthorized.py','watch_finish_core80_reauthorized.py'):z.write(R/'maintenance'/name,'maintenance/'+name)
 with (O/'CURRENT_TASK.md').open('a',encoding='utf8') as f:f.write('\nBATCH ENDED; independent raw audit/report complete. Read REPORT.md and result_summary.json. Source/input/output receipts verified; final seal closes this scope. Do not launch any further scenario or repeat a cell based on JCT.\n')
 save(O/'final_hashes.json',{str(q.relative_to(O)):sha(q) for q in O.rglob('*') if q.is_file() and q.name not in ('final_hashes.json','seal_receipt.json')})
 save(O/'seal_receipt.json',dict(unix=time.time(),sha256=sha(O/'final_hashes.json'),source_sha256=p['source_snapshot_sha256'],source_files=315,input_files=38,all_cell_end_receipts_verified=True))
 print('ALL_POSTPROCESSING_COMPLETE',O/'REPORT.md',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  save(O/'postprocess_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),unix=time.time()));raise
