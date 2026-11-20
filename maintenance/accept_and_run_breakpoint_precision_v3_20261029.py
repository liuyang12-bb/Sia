"""Run only fixed correctness gates, then fresh complete30 frozen comparisons."""
import json,shutil,subprocess,sys,time,traceback
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_breakpoint_study_20261029_run1/attempt3'
def invoke(name,action=None):
 r=dict(memory_gib=psutil.virtual_memory().available/2**30,disk_gib=psutil.disk_usage(str(OUT)).free/2**30)
 if r['memory_gib']<2 or r['disk_gib']<5:raise RuntimeError('RESOURCE_GATE '+str(r))
 cmd=[sys.executable,'-X','utf8','-B',str(ROOT/'maintenance'/name)]+([action] if action else [])
 label=name.removesuffix('.py')+('_'+action if action else '')
 with (OUT/(label+'.log')).open('x',encoding='utf8') as f:
  start=time.time();p=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 with (OUT/'gate_execution_manifest.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(command=cmd,exit_code=p.returncode,wall_seconds=time.time()-start,resources=r))+'\n')
 print('GATE_END',label,p.returncode,flush=True)
 if p.returncode:raise RuntimeError('GATE_FAILED '+label)
def main():
 assert json.loads((OUT/'tests_result.json').read_text())['exit_code']==0
 if (OUT/'bounded_16round_summary.json').exists():
  import hashlib
  hashes=json.loads((OUT/'bounded_code_hashes.json').read_text())
  for n,h in hashes.items():assert hashlib.sha256((OUT/'source'/n).read_bytes()).hexdigest().upper()==h,n
 else:invoke('bounded_breakpoint_precision_v3_20261029.py')
 assert json.loads((OUT/'bounded_16round_summary.json').read_text())['valid']
 invoke('accept_breakpoint_precision_v3_20261029.py')
 small=OUT/'small_gates';small.mkdir(exist_ok=False)
 for n in ['offline_validation.json','mechanism_witnesses.json']:shutil.copyfile(OUT/n,small/n)
 shutil.copyfile(OUT/'tests_result.json',small/'tests_result.json')
 invoke('run_breakpoint_small_precision_v3_20261029.py','freeze')
 invoke('run_breakpoint_small_precision_v3_20261029.py','batch')
 invoke('audit_breakpoint_small_precision_v3_20261029.py')
 assert json.loads((OUT/'small_gate_audit.json').read_text())['formal30_gate_passed']
 invoke('neutral_breakpoint_precision_v3_20261029.py')
 invoke('freeze_breakpoint30_precision_v3_20261029.py')
 # Stream formal batch progress to this existing exec session.
 p=subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'maintenance/run_breakpoint30_precision_v3_20261029.py'),'batch'],cwd=ROOT)
 if p.returncode:raise RuntimeError('FORMAL_BATCH_PROCESS_FAILED')
 print('ACCEPT_AND_RUN_ENDED',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  with (OUT/'gated_stop.json').open('x',encoding='utf8') as f:json.dump(dict(error=repr(e),traceback=traceback.format_exc(),time=time.time()),f,indent=2)
  raise
