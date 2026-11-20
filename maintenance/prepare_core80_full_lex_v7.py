"""Seal preformal V6 and copy actual bytes for equivalent capacity-cover cuts."""
import hashlib,json,shutil,time,zipfile,psutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];B=R/'cc-joint-core80-full-lex-repair-v6';P=R/'research_runs/sia_synergy/cc_core80_full_lex_repair_20261031_run3';T=R/'cc-joint-core80-full-lex-repair-v7';O=R/'research_runs/sia_synergy/cc_core80_full_lex_repair_20261031_run4'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def main():
 assert not T.exists() and not O.exists() and not (P/'execution_manifest.jsonl').exists()
 active=[(x.pid,x.cmdline()) for x in psutil.process_iter() if x.name().lower()=='python.exe' and any(Path(a).name in ('run_core80.py','supervise80.py','bounded_core80_full_lex_v6.py') for a in x.cmdline())];assert not active,active
 h={q.relative_to(B).as_posix():sha(q) for q in B.rglob('*') if q.is_file() and '__pycache__' not in q.parts};assert len(h)==320
 with (P/'development_source_hashes.json').open('x',encoding='utf8') as f:json.dump(h,f,indent=2)
 with zipfile.ZipFile(P/'development_source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
  for n in sorted(h):z.write(B/n,n)
 with (P/'DEVELOPMENT_CLOSED.json').open('x',encoding='utf8') as f:json.dump(dict(unix=time.time(),status='REAL_PREFIX_FAILED_AT42_NO_FORMAL_RUN',source_sha256=sha(P/'development_source_snapshot.zip'),source_files=len(h),closed_prefix=41,failure_simtime=2520,successor=str(O)),f,indent=2)
 with (P/'CURRENT_TASK.md').open('a',encoding='utf8') as f:f.write('\nCLOSED: V6 real45 gate failed at42,41complete prefix. No formal batch. Source320files archived, partial logs never repaired. Successor V7/run4; old sessions ended, do not restart.\n')
 T.mkdir();O.mkdir();(O/'tests').mkdir()
 for n,v in h.items():
  q=T/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(B/n,q);assert sha(q)==v
 for n in ('base_source_hashes.json','base_preregistration.json','preparation.json','probe_old.json','probe_repair.json','probe_lattice.json','red_literal_replay.log','red_backend_contract.log','red_backend_contract2.log','probe_primary.log','probe_primary2.log'):
  if (P/n).exists():shutil.copyfile(P/n,O/n)
 src=P/'tests/real80_FULL_45_lattice_primary/solver_failure.json.gz';dst=T/'sia-simulator/tests/fixtures/core80_full_startup_step42.json.gz';shutil.copyfile(src,dst)
 with (O/'new_fixture_provenance.json').open('x',encoding='utf8') as f:json.dump(dict(source=str(src),fixture=str(dst),sha256=sha(dst)),f,ensure_ascii=False,indent=2)
 (O/'CURRENT_TASK.md').write_text('''# ACTIVE V7: FULL exact lexicographic numerical repair

Current task: user requests fixing FULL, preserve objective/data/response/parameters/history, regression then seven full80paired resource methods. Do NOT resume104/160 or tune JCT.

Source cc-joint-core80-full-lex-repair-v7; output cc_core80_full_lex_repair_20261031_run4. Actual-byte base includes all historical source changes; no bareHEAD shortcut. Scientific base T3/315files unchanged. V4 preformal(run1) preserved; V5 sealed formal(run2) failed37 startup after36raw rounds/9events; V6 preformal(run3) full320tests passed but real45 gate failed42 startup after41complete rounds. V6source320files archived in run3/development_source_snapshot.zip. No formal run in V6. All old code/logs/reports protected.

New captured case42 has168184variables,2constraint groups(resource/job+primaryband),startup incumbent225,LPbound86.4117,gcd5. V6 certified primary pruning leaves43482; allthree30sec backends lack optimum proof. Do not promote incumbent225. Need exact equivalent strengthening, e.g capacity threshold cover inequalities derived from same GPU/CPU/RAM capacities, no new research parameters. A set of allocations each using more than (capacity+originaltol+roundoff)/(k+1) can contain at most k selected columns. Exhaustively verify no accepted binary point excluded. Optionally iterate proof reductions to fixed point only if justified; not JCT tuning. All original rows rechecked before commit.

No formal run yet. Plan red/green, existing320 tests+newcuts/captured42, bounded prefix fixed before execution past42, oldprefix41/originalSia regression, then freeze actualsource+38inputs and seven Medium dual80methods FULL,SIA_P,SIA_AUX,PROP,CPU,MEM,FO. All80fullwork/samearrival/CPU3RAM62.5perGPU/R0Elegacy/alpha.5/seed1,8/60sec6000rounds/MIP30seczero-gap,2GiB5GiB,6hcell24hbatch unchanged. FULL first,stoponengineeringfailure. No source edits during formal; no best-of result selection.

Need adapt V7 pipeline/reports after offline model validates cuts. New report name must differ from V5/V6 reports. All source versions preserved, no reset/clean/push, no official source edits. SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE.
''',encoding='utf8')
 print('V7_ISOLATED',len(h),T,O)
if __name__=='__main__':main()
