"""Sequential correctness acceptance only; no formal cell launcher here."""
import csv,gzip,hashlib,itertools,json,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];T=R/'cc-joint-core80-full-lex-repair-v4';O=R/'research_runs/sia_synergy/cc_core80_full_lex_repair_20261031_run1';OLD=R/'research_runs/sia_synergy/cc_core80_repair_full_20261031_run1'
sys.path[:0]=[str(T),str(T/'sia-simulator')]
def save(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def run(label,args,cwd=R):
 with (O/'tests'/(label+'.log')).open('xb') as f:a=subprocess.run([sys.executable,'-X','utf8','-B',*map(str,args)],cwd=cwd,stdout=f,stderr=subprocess.STDOUT)
 print(label,'EXIT',a.returncode,flush=True)
 if a.returncode:raise RuntimeError('ACCEPTANCE_FAILED '+label)
def clean(v):
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items() if k!='template_source'}
 if isinstance(v,list):return [clean(x) for x in v]
 return v
def frames(path,n):
 with gzip.open(path,'rt',encoding='utf8') as f:
  return [json.loads(x) for x in itertools.islice(f,n)]
def main():
 deadline=time.time()+3600
 while True:
  text=(O/'tests/full_suite_final.log').read_text(encoding='utf8')
  if 'Ran 316 tests' in text and '\nOK\n' in text:break
  if '\nFAILED (' in text or time.time()>deadline:raise RuntimeError('FULL_SUITE_NOT_ACCEPTED')
  time.sleep(3)
 run('FULL_real30',[R/'maintenance/bounded_core80_full_lex_v4.py','FULL',30,'lattice_primary'])
 run('A10_real20',[R/'maintenance/bounded_core80_full_lex_v4.py','A10',20,'lattice_primary'])
 run('PROP_real12',[R/'maintenance/bounded_core80_full_lex_v4.py','PROP',12,'lattice_primary'])
 from joint_prototype.execution_audit import audit_execution
 from joint_prototype.r0_liveness import R0Oracle
 from run_stage2c_cell import independent_rho
 work={x['name']:x for x in csv.DictReader((T/'inputs/core80_workload.csv').open(encoding='utf8'))};checks=0;locks=0;bridges=[];errors=[]
 for method,n,oldn in [('FULL',30,18),('A10',20,20),('PROP',12,12)]:
  directory=O/'tests'/f'real80_{method}_{n}_lattice_primary';rs=frames(directory/'prefix.jsonl.gz',n)
  oracle=R0Oracle(T/'inputs/registry.json',T/'inputs/models',{j:w['application'] for j,w in work.items()},'neutral' if method.startswith('A') else 'medium')
  for r in rs:
   d=r['decision_for_next_interval'];selected=d['result']['selected'];specs={x['job_id']:x for x in d['probe']['specs']};used={t+':0':[0.,0.,0.] for t in ('aws','dgx-ext','rtx')}
   assert d['result']['audit']['valid']
   for name,j in specs.items():
    if not j['checkpoint_locked'] or not j['current_config_id']:continue
    locks+=1;c=selected.get(name)
    assert c and c['config']['config_id']==j['current_config_id'] and c['placement']==dict(j['current_placement']) and (c['cpu_per_gpu'],c['ram_gb_per_gpu'])==(j['current_cpu_per_gpu'],j['current_ram_gb_per_gpu'])
   for c in selected.values():
    for node,g in c['placement'].items():
     for k,v in enumerate((g,g*c['cpu_per_gpu'],g*c['ram_gb_per_gpu'])):used[node][k]+=v
   for node,v in used.items():
    g=4 if node=='aws:0' else 8
    assert all(x<=y+1e-7 for x,y in zip(v,(g,3*g,62.5*g)))
   for e in r['execution']:
    checks+=1;assert audit_execution(e)['valid']
    a=e['resource_allocation']
    if a:
     rho=independent_rho(oracle,e['job_id'],a['config_id'],a['cpu_per_gpu'],a['ram_gb_per_gpu'])
     assert abs(rho-e['rho'])<1e-10
  cid=('dual_neutral_' if method.startswith('A') else 'dual_medium_')+method
  prior=frames(OLD/'cells'/cid/'rounds.jsonl.gz',oldn);mismatches=[]
  for i,(a,b) in enumerate(zip(prior,rs)):
   def project(x):
    z=x['decision_for_next_interval']['result'];return clean(dict(execution=x['execution'],selected=z['selected'],objective=z['scheduling_objective'],actual_goodput=z['actual_total_goodput']))
   if project(a)!=project(b):mismatches.append(i+1)
  bridges.append(dict(method=method,compared_rounds=oldn,exact_match=len(prior)==oldn and not mismatches,different_rounds=mismatches,ignored='Relocated template_source only; byte hashes preserved'))
 save(O/'tests/prefix_bridge_and_audit.json',dict(bridges=bridges,execution_checks=checks,lock_checks=locks,violations=errors))
 assert all(x['exact_match'] for x in bridges),bridges
 probes={}
 for name in ('official-artifacts','cc-joint-core80-full-lex-repair-v4'):
  label='sia_probe_'+name;run(label,[T/'sia-simulator/check_stage1_legacy_regression.py','--probe',R/name/'sia-simulator'])
  probes[name]=json.loads(next(x.removeprefix('PROBE_JSON=') for x in (O/'tests'/(label+'.log')).read_text(encoding='utf8').splitlines() if x.startswith('PROBE_JSON=')))
 assert probes['official-artifacts']==probes['cc-joint-core80-full-lex-repair-v4']
 save(O/'tests/original_sia_exact_regression.json',dict(exact_match=True,rounds=3,states=probes))
 source={str(x.relative_to(T)):hashlib.sha256(x.read_bytes()).hexdigest().upper() for x in T.rglob('*') if x.is_file() and '__pycache__' not in x.parts}
 save(O/'tests/accepted_source_hashes.json',source)
 save(O/'software_acceptance.json',dict(status='PASS',existing_tests=302,new_tests=14,total_tests=316,full_suite='tests/full_suite_final.log',full_real_prefix=30,old_failure_decision=19,original_sia_exact_rounds=3,bridges=bridges,execution_checks=checks,locks=locks,violations=0))
 print('ALL_SOFTWARE_ACCEPTED',flush=True)
if __name__=='__main__':main()
