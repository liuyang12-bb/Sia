import importlib.util,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_cb_validation_20261029_ample_retry1'
spec=importlib.util.spec_from_file_location('frozenrunner',OUT/'run.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r);p=r.setup()
records=[json.loads(l) for l in (OUT/'execution_manifest.jsonl').read_text().splitlines()];assert len(records)==4 and records[-1]['exit_code']!=0
for rec in records[:3]:
 folder=OUT/'cells'/rec['cell']
 for n,h in json.loads((folder/'output_hashes.json').read_text()).items():assert r.sha(folder/n)==h
for i in [4,5]:
 mix,cap,response,method=p['cells'][i];name=r.label(mix,cap,response,method);assert not (OUT/'cells'/name).exists();resources=r.resources()
 with (OUT/(name+'.log')).open('x',encoding='utf8') as f:
  start=time.time();proc=subprocess.run([sys.executable,'-X','utf8','-B',str(OUT/'run.py'),'run','--mix',mix,'--scenario',cap,'--response',response,'--method',method],stdout=f,stderr=subprocess.STDOUT)
 with (OUT/'execution_manifest.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(index=i,cell=name,exit_code=proc.returncode,wall_seconds=time.time()-start,resources=resources))+'\n')
 print('REMAINING_CELL_ENDED',name,proc.returncode,flush=True)
for name in ['audit_cb_validation_20261029.py','report_cb_validation_20261029.py']:
 with (OUT/(name+'.log')).open('x',encoding='utf8') as f:proc=subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'maintenance'/name)],stdout=f,stderr=subprocess.STDOUT)
 if proc.returncode:raise RuntimeError('Postprocessing failed '+name)
print('ALL_ATTEMPTS_REPORTED_STOPPED',flush=True)
