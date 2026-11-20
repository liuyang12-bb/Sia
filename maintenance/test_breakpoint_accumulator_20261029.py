"""Actual failed-round replay: legacy red, donor-contract green, bad rho red."""
import contextlib,copy,gzip,io,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_breakpoint_study_20261029_run1';SRC=OUT/'source'
sys.path[:0]=[str(SRC),str(SRC/'sia-simulator')]
from breakpoint_study import build
from run_stage2c_cell import Accumulator,independent_rho
from joint_prototype.execution_audit import audit_execution
def main():
 with gzip.open(OUT/'mechanisms/CPU_free_O/rounds.jsonl.gz','rt',encoding='utf8') as f:rounds=[json.loads(line) for line in f]
 r=rounds[-1];d=r['decision_for_next_interval'];assert d['result']['audit']['valid']
 with contextlib.redirect_stdout(io.StringIO()):c=build('CC_CPU','R01','medium',OUT/'fixtures/CPU_three.csv')
 try:Accumulator(c).consume(c,r)
 except RuntimeError as exc:assert str(exc)=='PROGRESS_FLOOR'
 else:raise AssertionError('obsolete red condition not reproduced')
 # Independent precommit/execution checks remain mandatory; only the obsolete
 # per-job guarantee is not a contract of donor-allowing CC.
 usage={}
 for name,col in d['result']['selected'].items():
  cfg=col['config'];cpu,ram=col['cpu_per_gpu'],col['ram_gb_per_gpu']
  assert abs(col['rho']-independent_rho(c.policy.oracle,name,cfg['config_id'],cpu,ram))<1e-12
  for node,g in col['placement'].items():
   v=usage.setdefault(node,[0.,0.,0.])
   for k,x in enumerate([g,g*cpu,g*ram]):v[k]+=x
 for v in usage.values():assert all(x<=y+1e-7 for x,y in zip(v,[2,12,187.5]))
 for e in r['execution']:assert audit_execution(e)['valid']
 m=Accumulator(c);m.last_decision_round=d['round'];m.consume(c,r)
 tampered=copy.deepcopy(r)
 e=next(e for e in tampered['execution'] if e['training_segments']);e['training_segments'][0]['effective_rate']*=2
 assert not audit_execution(e)['valid']
 try:
  m=Accumulator(c);m.last_decision_round=d['round'];m.consume(c,tampered)
 except RuntimeError as exc:assert str(exc)=='ONCE_ONLY'
 else:raise AssertionError('execution multiplier corruption escaped')
 with (OUT/'accumulator_red_green.json').open('x',encoding='utf8') as f:json.dump(dict(valid=True,actual_failed_round=d['round'],old_contract_reproduced='PROGRESS_FLOOR',current_contract_passed=True,tampered_effective_rate_rejected=True,capacity_rho_audited=True),f,indent=2)
 print('ACCUMULATOR_RED_GREEN_PASS')
if __name__=='__main__':main()
