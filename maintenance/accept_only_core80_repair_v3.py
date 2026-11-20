"""Software/bounded regression ONLY. No formal launcher or training batch."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];T=R/'cc-joint-core80-repair-v3';O=R/'research_runs/sia_synergy/cc_core80_repair_20261031_run3'
def run(label,args,cwd=R):
 with (O/'tests'/(label+'.log')).open('xb') as f:r=subprocess.run([sys.executable,'-X','utf8','-B',*map(str,args)],cwd=cwd,stdout=f,stderr=subprocess.STDOUT)
 print(label,'EXIT',r.returncode,flush=True)
 if r.returncode:raise RuntimeError('TEST_FAILED '+label)
def main():
 p=O/'tests/real80_FULL_12_cbc_safety/result.json';stop=time.time()+3600
 while not p.exists():
  if time.time()>=stop:raise RuntimeError('BOUNDED_PROBE_NOT_FINALIZED')
  time.sleep(2)
 assert json.loads(p.read_text())['status']=='PASS'
 run('A10_real20',[R/'maintenance/bounded_core80_repair_v3.py','A10',20,'primal_dual'])
 run('PROP_real12',[R/'maintenance/bounded_core80_repair_v3.py','PROP',12,'primal_dual'])
 run('full_suite',['-m','unittest','discover','-s','tests','-p','test_*.py','-v'],T/'sia-simulator')
 text=(O/'tests/full_suite.log').read_text(encoding='utf8');assert 'Ran 302 tests' in text and '\nOK\n' in text
 results={}
 for name in ['official-artifacts','cc-joint-core80-repair-v3']:
  label='sia_probe_'+name;run(label,[T/'sia-simulator/check_stage1_legacy_regression.py','--probe',R/name/'sia-simulator'])
  results[name]=json.loads(next(x.removeprefix('PROBE_JSON=') for x in (O/'tests'/(label+'.log')).read_text(encoding='utf8').splitlines() if x.startswith('PROBE_JSON=')))
 assert results['official-artifacts']==results['cc-joint-core80-repair-v3']
 with (O/'tests/original_sia_exact_regression.json').open('x') as f:json.dump(dict(exact_match=True,rounds=3,states=results),f,indent=2)
 print('SOFTWARE_ACCEPTANCE_COMPLETE; formal batch remains blocked; NO LAUNCH',flush=True)
if __name__=='__main__':main()
