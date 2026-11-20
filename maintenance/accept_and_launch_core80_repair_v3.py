"""Sequential acceptance then preregistered rerun; never retries science."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];T=R/'cc-joint-core80-repair-v3';O=R/'research_runs/sia_synergy/cc_core80_repair_20261031_run3'
def run(label,args,cwd=R):
 path=O/'tests'/(label+'.log')
 with path.open('xb') as f:r=subprocess.run([sys.executable,'-X','utf8','-B',*map(str,args)],cwd=cwd,stdout=f,stderr=subprocess.STDOUT)
 print(label,'EXIT',r.returncode,flush=True)
 if r.returncode:raise RuntimeError('ACCEPTANCE_FAILED '+label)
 return path
def main():
 if (O/'FORMAL_BLOCKED_AUTO_REVIEW.json').exists():raise RuntimeError('FORMAL_LAUNCH_BLOCKED: explicit renewed researcher authorization and approval review required; do not bypass')
 gate=json.loads((O/'tests/real80_FULL_12_primal_dual/result.json').read_text());assert gate['status']=='PASS' and gate['boundaries']==12
 run('A10_real20',[R/'maintenance/bounded_core80_repair_v3.py','A10',20,'dual_presolve'])
 run('PROP_real12',[R/'maintenance/bounded_core80_repair_v3.py','PROP',12,'dual_presolve'])
 log=run('full_suite',['-m','unittest','discover','-s','tests','-p','test_*.py','-v'],T/'sia-simulator')
 text=log.read_text(encoding='utf8');assert 'Ran 298 tests' in text and '\nOK\n' in text
 bridge=(R/'maintenance/bridge_core80_repair.py').read_text().replace('cc_core80_repair_20261031_run1','cc_core80_repair_20261031_run3').replace('real80_FULL_8_early_zero','real80_FULL_12_primal_dual').replace('real80_PROP_12_post_only','real80_PROP_12_dual_presolve').replace('real80_A10_20_post_only','real80_A10_20_dual_presolve')
 path=R/'maintenance/bridge_core80_repair_v3.py'
 with path.open('x',encoding='utf8') as f:f.write(bridge)
 run('historical_bridge',[path])
 results={}
 for name in ['official-artifacts','cc-joint-core80-repair-v3']:
  log=run('sia_probe_'+name,[T/'sia-simulator/check_stage1_legacy_regression.py','--probe',R/name/'sia-simulator'])
  results[name]=json.loads(next(x.removeprefix('PROBE_JSON=') for x in log.read_text(encoding='utf8').splitlines() if x.startswith('PROBE_JSON=')))
 assert results['official-artifacts']==results['cc-joint-core80-repair-v3']
 with (O/'tests/original_sia_exact_regression.json').open('x') as f:json.dump(dict(exact_match=True,rounds=3,states=results),f,indent=2)
 freeze=(R/'maintenance/freeze_core80_repair.py').read_text().replace("T=R/'cc-joint-core80-repair'","T=R/'cc-joint-core80-repair-v3'").replace('cc_core80_repair_20261031_run1','cc_core80_repair_20261031_run3').replace('Ran 289 tests','Ran 298 tests').replace('CORE80_ENGINEERING_REPAIR_CANONICAL_FIRST_EXACT_PRESOLVE','CORE80_REPAIR_V3_CERTIFIED_DUAL_REDUCED_COST')
 freeze=freeze.replace("'computation_equivalence':'Full response", "'continuous_certificate':'LP fractional primal never submitted; one fixed same-goal MILP on LP-support+existing incumbent gives only a validated upper bound, never a globally optimal result by itself. Directed80-digit Decimal dual interval lower bound, integer-cost lattice, and phase-local strict reduced-cost fixing preserve all minimizers. Existing candidate grid, original rows and MIP30sec/gap0 unchanged.',\n  'computation_equivalence':'Full response")
 path=R/'maintenance/freeze_core80_repair_v3.py'
 with path.open('x',encoding='utf8') as f:f.write(freeze)
 run('freeze_receipt',[path])
 with (O/'CURRENT_TASK.md').open('a',encoding='utf8') as f:f.write('\nALL ACCEPTANCE PASS (298tests,realFULL12,A1020,PROP12,oldprefix/originalSia). Frozen source_snapshot.zip/source_hashes.json/preregistration.json. This driver now starts supervise80.py exactly once; inspect execution_manifest.jsonl and caller exec session before any restart.11core only; source must remain unchanged.\n')
 print('FORMAL_FROZEN_LAUNCH',O,flush=True)
 child=subprocess.run([sys.executable,'-X','utf8','-B',str(T/'supervise80.py'),str(O/'preregistration.json')],cwd=T)
 print('BATCH_EXIT',child.returncode,flush=True)
 return child.returncode
if __name__=='__main__':raise SystemExit(main())
