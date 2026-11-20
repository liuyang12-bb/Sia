"""Bounded new adapter acceptance, preserving the full old regression suite."""
import contextlib,io,json,math,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_breakpoint_study_20261029_run1';SRC=OUT/'source'
sys.path[:0]=[str(SRC),str(SRC/'sia-simulator')]
from breakpoint_study import build
from resource_study import METHODS
from joint_prototype.execution_audit import audit_execution
from run_stage2c_cell import independent_rho
def main():
 start=time.time();rows=[];reference=None
 with (OUT/'adapter_acceptance.log').open('x',encoding='utf8') as log,contextlib.redirect_stdout(log):
  for method in METHODS:
   c=build(method,'R11','medium',SRC/'sia-simulator/tests/fixtures/resource_study_three.csv')
   o=c.policy.oracle;contract=(o.registry_sha256,o.provenance_digest(),o.fixed_reference,o.response_exponent)
   if reference is None:reference=contract
   assert reference==contract
   for _ in range(4):
    c.step(60);d=c.policy.last_decision;assert d['result']['audit']['valid']
    for name,col in d['result']['selected'].items():
     cp,mm=col['cpu_per_gpu'],col['ram_gb_per_gpu'];assert math.isclose(col['rho'],independent_rho(o,name,col['config']['config_id'],cp,mm),rel_tol=1e-12)
     if method in ('SIA_P','CC_P'):assert(cp,mm)==(3.,93.75)
     if method=='CC_CPU':assert mm==93.75
     if method=='CC_MEM':assert cp==3.
    for e in c.joint_round_logs[-1]['execution']:assert audit_execution(e)['valid']
    c.joint_round_logs.clear();c.logs.clear()
   assert sum(j.progress for j in c.jobs)>0
   rows.append(dict(method=method,rounds=4,valid=True,oracle_contract=contract))
   print('ADAPTER_OK',method,flush=True)
 with (OUT/'adapter_acceptance.json').open('x',encoding='utf8') as f:json.dump(dict(valid=True,cases=rows,elapsed_seconds=time.time()-start),f,indent=2)
 print('ALL_ADAPTERS_PASSED')
if __name__=='__main__':main()
