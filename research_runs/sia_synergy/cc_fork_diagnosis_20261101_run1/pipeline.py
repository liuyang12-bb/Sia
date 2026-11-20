import hashlib,json,subprocess,sys,time
from pathlib import Path
O=Path(__file__).resolve().parent
def main():
 p=json.loads((O/'preregistration.json').read_text());start=time.time()
 receipt={n:hashlib.sha256((O/n).read_bytes()).hexdigest().upper() for n in ('run_fork.py','pipeline.py','raw_audit_lib.py','preregistration.json','inputs/workload.csv')}
 (O/'execution_receipt.json').write_text(json.dumps(receipt,indent=2))
 with (O/'execution_manifest.jsonl').open('x') as m:
  for cell in p['cells']:
   cmd=[sys.executable,'-X','utf8','-B',str(O/'run_fork.py'),'--cell',cell['id']]
   with (O/(cell['id']+'.log')).open('x',encoding='utf8') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
   m.write(json.dumps(dict(cell=cell['id'],exit_code=r.returncode,command=cmd,wall=time.time()-start))+'\n');m.flush()
   print(cell['id'],r.returncode,flush=True)
   if r.returncode:raise SystemExit(r.returncode)
 (O/'pipeline_complete.json').write_text(json.dumps(dict(wall=time.time()-start,cells=4)))
if __name__=='__main__':main()
