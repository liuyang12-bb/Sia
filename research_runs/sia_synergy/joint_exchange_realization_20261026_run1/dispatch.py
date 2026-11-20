"""Four-cell repair follow-up dispatcher; no obsolete D2 admission guard."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def main():
 p=json.loads((OUT/'preregistration.json').read_text())
 with (OUT/'dispatch_manifest.json').open('x') as f:json.dump(dict(preregistration_sha256=sha(OUT/'preregistration.json'),controller_sha256=sha(Path(__file__)),reason='Copied generic batch dispatcher erroneously required D2 cells in this new four-cell list; no cell was started. Preserve execution_manifest.jsonl; use frozen cell entry point directly.',gate_evidence=dict(training_probe_sha256=sha(OUT/'training_probe_result.json'),green_tests_sha256=sha(OUT/'green.log')),order=[c['id'] for c in p['runs']]),f,indent=2)
 with (OUT/'execution_manifest_actual.jsonl').open('x') as mf:
  for cell in p['runs']:
   if psutil.virtual_memory().available<2*2**30:raise RuntimeError('MEMORY_GATE_2GiB')
   if psutil.disk_usage(str(OUT)).free<5*2**30:raise RuntimeError('DISK_GATE_5GiB')
   tick=time.time()
   with (OUT/(cell['id']+'.log')).open('x',encoding='utf-8') as log:
    res=subprocess.run([sys.executable,'-X','utf8','-B',str(OUT/'experiment.py'),'cell','--output',str(OUT),'--cell',cell['id']],cwd=OUT/'source',stdout=log,stderr=subprocess.STDOUT)
   record=dict(cell=cell['id'],exit_code=res.returncode,wall_seconds=time.time()-tick);mf.write(json.dumps(record)+'\n');mf.flush();print(record,flush=True)
   if res.returncode:raise RuntimeError('CELL_FAILED')
if __name__=='__main__':main()
