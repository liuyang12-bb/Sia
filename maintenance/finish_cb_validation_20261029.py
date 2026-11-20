"""Observe the already running batch, then audit; never launches simulations."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_cb_validation_20261029_run1'
start=time.time();last=-1
while True:
 path=OUT/'execution_manifest.jsonl';rows=[json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
 if len(rows)!=last:print('FINISH_OBSERVER',len(rows),'/36',flush=True);last=len(rows)
 if any(r['exit_code'] for r in rows):raise RuntimeError('Batch child failure: preserve cell and logs')
 if list(OUT.glob('safe_stop_*.json')):raise RuntimeError('Resource safe stop; no duplicate launch')
 if len(rows)==36:break
 if time.time()-start>4*3600:raise RuntimeError('Observer timeout, no results altered')
 time.sleep(2)
for name in ['audit_cb_validation_20261029.py','report_cb_validation_20261029.py']:
 with (OUT/(name+'.log')).open('x',encoding='utf8') as f:r=subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'maintenance'/name)],stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError('Postprocessing failure '+name)
print('COMPLETE_AUDITED_REPORTED_STOPPED',flush=True)
