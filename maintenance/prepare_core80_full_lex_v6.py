"""New immutable-version repair after the sealed startup-phase failure."""
import hashlib,json,shutil,time,psutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];OLD=R/'research_runs/sia_synergy/cc_core80_full_lex_repair_20261031_run2';T=R/'cc-joint-core80-full-lex-repair-v6';O=R/'research_runs/sia_synergy/cc_core80_full_lex_repair_20261031_run3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def main():
 assert not T.exists() and not O.exists()
 p=json.loads((OLD/'preregistration.json').read_text());base=Path(p['source_root']);assert all(sha(base/n)==v for n,v in p['source_hashes'].items())
 ledger=[json.loads(x) for x in (OLD/'execution_manifest.jsonl').read_text().splitlines()];assert ledger[-1]['event']=='BATCH_END'
 assert not [x.pid for x in psutil.process_iter() if x.name().lower()=='python.exe' and any(Path(a).name in ('run_core80.py','supervise80.py') for a in x.cmdline())]
 O.mkdir();(O/'tests').mkdir();T.mkdir()
 for n,v in p['source_hashes'].items():
  q=T/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(base/n,q);assert sha(q)==v
 for n in ('base_source_hashes.json','base_preregistration.json','preparation.json','probe_old.json','probe_repair.json','probe_lattice.json','red_literal_replay.log','red_backend_contract.log','red_backend_contract2.log','probe_primary.log','probe_primary2.log'):
  if (OLD/n).exists():shutil.copyfile(OLD/n,O/n)
 with (O/'development_base.json').open('x',encoding='utf8') as f:json.dump(dict(unix=time.time(),previous_source=str(base),previous_snapshot_sha256=p['source_snapshot_sha256'],previous_sealed_result=str(OLD),scientific_base='cc-joint-core80-repair-v3',failure='Startup second objective after36complete rounds; all7same-version resource pairs still required, no partial mean'),f,ensure_ascii=False,indent=2)
 fixture=T/'sia-simulator/tests/fixtures/core80_full_startup_step37.json.gz';shutil.copyfile(OLD/'cells/dual_medium_FULL/failure_solver_diagnostics.json.gz',fixture)
 with (O/'new_fixture_provenance.json').open('x',encoding='utf8') as f:json.dump(dict(source=str(OLD/'cells/dual_medium_FULL/failure_solver_diagnostics.json.gz'),fixture=str(fixture),sha256=sha(fixture)),f,ensure_ascii=False,indent=2)
 (O/'CURRENT_TASK.md').write_text('''# ACTIVE V6: FULL lexicographic engineering repair, no scientific changes

User authorizes fixing FULL and rerunning complete affected resource pairs; same objective/data/response/parameters, preserve history. Current source cc-joint-core80-full-lex-repair-v6, output cc_core80_full_lex_repair_20261031_run3. Scientific base315files/T3; V5 sealed319file snapshot37D0D0A205761D7C8B2614FC273C0A0945A5454812BE15811676D6786B2B2FAA.

V4 preformal development (run1) preserved; V5 formal run2 stopped at37th decision,36complete rawrounds,9completions, nooverallJCT. Oldrank19 fixed. NEW failure in secondary startup objective,200462variables,integercostgcd5,valid incumbent250,LPbound102.6124,HiGHS/CBC timeouts. Other6methods NOT RUN per gate. All old source/output/report/seals protected. Do not restart old37837/40799 sessions; ended.

V6 first offline test: apply certified primary-band infeasibility presolve already before SECONDARY, not only after startup is fixed. Existing V5 helper excludes len2; no algorithm/weights/constraints/response changes needed. If insufficient, design exact equivalent lex implementation, never accept unproved250 or relax primary/locks/time budgets. Preserve every probe. New fixture core80_full_startup_step37.json.gz.

Need red/green tests, allold318 plus newtests, realFULL prefix past37 with fixed prereg bound, originalSia/oldtrajectory regressions. Then freeze same seven Medium dual80cells FULL,SIA_P,SIA_AUX,PROP,CPU,MEM,FO; FULL first,stopengineeringfailure;2GiB/5GiB,60sec/6000rounds,30sec/zero-gap solver settings,6h/cell24h batch unchanged. No Neutral/104/160/optional/parameter search. No formal run yet. Current source is a full actual-byte copy, not bare Git HEAD.

Observer/report pipeline needs V6 adaptation, new report name CC_CORE80_FULL_LEX_REPAIR_V6_COMPARISON_20261031.md (do not overwrite V5 partial report). Source code may be edited now, before tests/freeze only. No source/data edits during formal; independent raw audit/seal at end. SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE.
''',encoding='utf8')
 print('ISOLATED_V6',T,O,flush=True)
if __name__=='__main__':main()
