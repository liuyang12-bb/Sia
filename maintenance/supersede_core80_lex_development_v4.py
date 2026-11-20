"""Preserve the development snapshot; stop only our preformal acceptance tree."""
import hashlib,json,shutil,time,zipfile,psutil,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];T=R/'cc-joint-core80-full-lex-repair-v4';O=R/'research_runs/sia_synergy/cc_core80_full_lex_repair_20261031_run1';N=R/'cc-joint-core80-full-lex-repair-v5';P=R/'research_runs/sia_synergy/cc_core80_full_lex_repair_20261031_run2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def main():
 assert not (O/'execution_manifest.jsonl').exists() and not (O/'preregistration.json').exists(),'Never stop a formal batch through this development helper'
 assert not N.exists() and not P.exists()
 names={'accept_core80_full_lex_v4.py','freeze_launch_core80_full_lex_v4.py','watch_core80_full_lex_v4.py','bounded_core80_full_lex_v4.py'};targets=[]
 for x in psutil.process_iter():
  if x.name().lower()=='python.exe' and any(Path(v).name in names for v in x.cmdline()):targets.append(x)
 records=[dict(pid=x.pid,ppid=x.ppid(),command=x.cmdline()) for x in targets]
 for x in reversed(targets):
  try:x.terminate()
  except psutil.NoSuchProcess:pass
 psutil.wait_procs(targets,timeout=10)
 assert not any(x.is_running() for x in targets),'An acceptance process still runs'
 h=json.loads((O/'tests/accepted_source_hashes.json').read_text());assert all(sha(T/n)==v for n,v in h.items())
 with zipfile.ZipFile(O/'development_source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
  for n in sorted(h):z.write(T/n,n)
 with (O/'DEVELOPMENT_SUPERSEDED.json').open('x',encoding='utf8') as f:json.dump(dict(unix=time.time(),state='PRE_FORMAL_DEVELOPMENT_SUPERSEDED_NOT_AN_ALGORITHM_FAILURE',formal_runs=0,processes=records,source_sha256=sha(O/'development_source_snapshot.zip'),reason='Runtime telemetry shows current-cost LPs still process full original domain before certified primary infeasibility elimination. Reorder exact presolve in a new version, with original settings and objectives unchanged. Interrupted prefix is preserved, never repaired.',successor=str(P)),f,ensure_ascii=False,indent=2)
 with (O/'CURRENT_TASK.md').open('a',encoding='utf8') as f:f.write('\nDEVELOPMENT SUPERSEDED before any formal run; all316test evidence and partialFULL prefix retained, no log repaired. Source frozen in development_source_snapshot.zip. Successor cc-joint-core80-full-lex-repair-v5 and cc_core80_full_lex_repair_20261031_run2. Do not restart this old pipeline.\n')
 N.mkdir();P.mkdir();(P/'tests').mkdir()
 for n,v in h.items():
  q=N/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(T/n,q);assert sha(q)==v
 for n in ('base_source_hashes.json','base_preregistration.json','preparation.json'):
  shutil.copyfile(O/n,P/n)
 with (P/'development_base.json').open('x',encoding='utf8') as f:json.dump(dict(previous_source=str(T),previous_source_snapshot=str(O/'development_source_snapshot.zip'),source_sha256=sha(O/'development_source_snapshot.zip'),source_files=len(h),scientific_base='cc-joint-core80-repair-v3',authorized_scope='same human FULL lex repair, seven complete resource-method pairs; no scientific/solver parameter changes'),f,ensure_ascii=False,indent=2)
 (P/'CURRENT_TASK.md').write_text((O/'CURRENT_TASK.md').read_text().split('\nRecovery sessions:')[0]+'\nACTIVE SUCCESSOR V5: only exact proof-domain ordering improvement. Prior v4 source/tests/partial prefix protected, no formal run was launched there. Start new red/green proof-order test, full318tests, realFULL30/A1020/PROP12 and originalSia/prefix bridge; then freeze seven Medium resource cells FULL,SIA_P,SIA_AUX,PROP,CPU,MEM,FO. No old sessions are recovery authority.\n',encoding='utf8')
 print('SUPERSEDED_PRE_FORMAL',records,'NEW',N,P,flush=True)
if __name__=='__main__':main()
