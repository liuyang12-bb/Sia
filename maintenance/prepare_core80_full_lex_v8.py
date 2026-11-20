"""Seal failed preformal V7, copy actual bytes; no scientific changes."""
import hashlib,json,shutil,time,zipfile,psutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];B=R/'cc-joint-core80-full-lex-repair-v7';P=R/'research_runs/sia_synergy/cc_core80_full_lex_repair_20261031_run4';T=R/'cc-joint-core80-full-lex-repair-v8';O=R/'research_runs/sia_synergy/cc_core80_full_lex_repair_20261031_run5'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
assert not T.exists() and not O.exists() and not (P/'execution_manifest.jsonl').exists()
active=[(p.pid,p.cmdline()) for p in psutil.process_iter() if p.name().lower()=='python.exe' and any(Path(x).name in ('run_core80.py','supervise80.py','bounded_core80_full_lex_v7.py','accept_core80_full_lex_v7.py','freeze_launch_core80_full_lex_v7.py','watch_core80_full_lex_v7.py') for x in p.cmdline())];assert not active,active
h={p.relative_to(B).as_posix():sha(p) for p in B.rglob('*') if p.is_file() and '__pycache__' not in p.parts};assert len(h)==327
with (P/'development_source_hashes.json').open('x',encoding='utf8') as f:json.dump(h,f,indent=2)
with zipfile.ZipFile(P/'development_source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
 for n in sorted(h):z.write(B/n,n)
with (P/'DEVELOPMENT_CLOSED.json').open('x',encoding='utf8') as f:json.dump(dict(status='STARTUP42_FIXED_RANK42_FAILED_NO_FORMAL_RUN',unix=time.time(),snapshot_sha256=sha(P/'development_source_snapshot.zip'),source_files=len(h),complete_prefix=41,successor=str(O)),f,ensure_ascii=False,indent=2)
with (P/'CURRENT_TASK.md').open('a',encoding='utf8') as f:f.write('\nCLOSED:331tests PASS, true42 startup solved but later per-job rank deepspeech2-10008 cost8186 unproved.41complete prefix. No formal prereg/run. All waiters stopped. V7 archived327files. Successor V8/run5; generalize SAME exact epsilon proof to integer lex objectives while retaining ALL earlier fixed rows.\n')
T.mkdir();O.mkdir();(O/'tests').mkdir()
for n,v in h.items():
 p=T/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(B/n,p);assert sha(p)==v
q=P/'tests/real80_FULL_45_lattice_primary/solver_failure.json.gz';dst=T/'sia-simulator/tests/fixtures/core80_full_rank_step42.json.gz';shutil.copyfile(q,dst)
with (O/'new_fixture_provenance.json').open('x',encoding='utf8') as f:json.dump(dict(source=str(q),fixture=str(dst),sha256=sha(dst)),f,ensure_ascii=False,indent=2)
for n in ('base_source_hashes.json','base_preregistration.json','preparation.json','probe_old.json','probe_repair.json','probe_lattice.json','red_literal_replay.log','red_backend_contract.log','red_backend_contract2.log','probe_primary.log','probe_primary2.log'):
 if (P/n).exists():shutil.copyfile(P/n,O/n)
(O/'CURRENT_TASK.md').write_text('''# ACTIVE V8 — exact FULL lex numerical repair

User asks repair FULL without changing goal/data/response/parameters, then seven SAME-VERSION complete80Medium resource pairs. No tuning,104/160/Neutral/new scenarios. Scientific315file base cc-joint-core80-repair-v3 unchanged. Protected V4/run1,V5formalrun2(failed37),V6run3(preformal failed42 startup),V7run4(331testsPASS,real45 failed42 later rank after startup proof). Never repair partial logs or delete history. All V7 processes ended; source327files archived.

Source cc-joint-core80-full-lex-repair-v8; output cc_core80_full_lex_repair_20261031_run5. Captured new42rank168184vars,4groups(base+primary+startup225+oneearliermask),deepspeech2-10008 rankinc8186,LPbound0,two30secHiGHS+existingCBC unproved. Extend equivalent epsilon budget theorem to this integer rank, retaining constraints[2:] exactly/proof-only conservative outer bounds, use dummy zero cost to build bijective GPU groups even for nonuniform current rank. Fix directed Decimal cutoff subtraction as engineering guard. No change to objective weights/bands, p/lambda, quota grid, R0/phi/clipping, seed1,8,arrival/work,batch/costs/locks or30sec/zero-gap solver settings.

First offline captured42rank and red/green tests, full existing331+new, fixed realFULL45/A1020/PROP12, prefix41 bridge and officialSia3 exact. THEN freeze actual source bytes/full snapshot+38inputs, prereg seven FULL,SIA_P,SIA_AUX,PROP,CPU,MEM,FO, independent processes serial, FULL first stop on engineeringfailure. All80same jobs,60sec6000rounds,CPU3RAM62.5perGPU,AWS4,DGX8,RTX8,Mediumalpha.5,2GiB/5GiB gate,6hcell24hbatch. No formal launch yet. Don't start V7 scripts. Prepare V8 pipeline only after offline proof succeeds. If failure preserve whole model; never promote incumbent or change scientific semantics. Final independent raw audit/events/resources/locks/rho and new comparison report; no oldcontrols filling newpairs. SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE.
''',encoding='utf8')
print('V8_COPIED',len(h),'plus rank42 fixture')
