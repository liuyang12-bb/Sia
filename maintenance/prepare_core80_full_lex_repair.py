"""Create a byte-verified, isolated repair tree; never alter sealed history."""
import hashlib,json,shutil,time,psutil
from pathlib import Path
R=Path(__file__).resolve().parents[1]
BASE=R/'research_runs/sia_synergy/cc_core80_repair_full_20261031_run1'
T=R/'cc-joint-core80-full-lex-repair-v4'
O=R/'research_runs/sia_synergy/cc_core80_full_lex_repair_20261031_run1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(p,x):
 with p.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def main():
 assert not T.exists() and not O.exists(),'Refuse to overwrite an existing repair'
 p=json.loads((BASE/'preregistration.json').read_text(encoding='utf8'));old=Path(p['source_root'])
 for n,v in p['source_hashes'].items():assert sha(old/n)==v,('SOURCE_CHANGED',n)
 for n,v in p['input_hashes'].items():assert sha(Path(n))==v,('INPUT_CHANGED',n)
 active=[(x.pid,x.cmdline()) for x in psutil.process_iter() if x.name().lower()=='python.exe' and any(Path(a).name in ('run_core80.py','supervise80.py') for a in x.cmdline())]
 assert not active,active
 resources=dict(memory_gib=psutil.virtual_memory().available/2**30,disk_gib=psutil.disk_usage(str(R)).free/2**30)
 assert resources['memory_gib']>=2 and resources['disk_gib']>=5,resources
 O.mkdir();T.mkdir()
 for n,v in p['source_hashes'].items():
  dest=T/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old/n,dest);assert sha(dest)==v
 save(O/'base_source_hashes.json',p['source_hashes']);save(O/'base_preregistration.json',p)
 save(O/'preparation.json',dict(unix=time.time(),base=str(old),source=str(T),base_files=len(p['source_hashes']),resources=resources,base_snapshot_sha256=p['source_snapshot_sha256'],active=active,
  authorization='先定位并修复FULL的词典序求解失败，保持算法目标、数据、响应模型和参数不变，保留失败现场。补充回归测试后重跑受影响的完整配对，补齐双资源联合方案的结果，再比较它与串行FO及仅CPU、仅内存优化的收益。'))
 (O/'CURRENT_TASK.md').write_text('''# ACTIVE: FULL lexicographic engineering repair and complete resource pairs

Latest human authorization: repair FULL lexicographic solver failure, keep scientific objective/data/response/parameters unchanged, preserve failure, add regressions and rerun complete affected pairs.

Base sealed run: ../cc_core80_repair_full_20261031_run1. Base actual source cc-joint-core80-repair-v3,315files,verified by base_source_hashes.json; not a bare Git HEAD. Independent source: cc-joint-core80-full-lex-repair-v4. No official-source/history edits, reset,clean,push or tuning.

First inspect/replay the captured19th-decision131244-variable rank MIP offline. Preserve every diagnostic. Proposed fixes must be exact equivalent presolve/representation/certificates, not looser objective bands or accepting an unproved incumbent. Same solver30sec/zero-gap/retries. Verify all original rows, locks, quotas and objective hierarchy before commit.

Plan: red captured-model regression -> minimal numerical repair -> full302existing tests plus new tests, bounded FULL beyond19/originalSia/old prefixes -> freeze actual source+inputs+dependencies and seven resource cells FULL,SIA_P,SIA_AUX,PROP,CPU,MEM,FO on the same new version. FULL must pass before controls continue. All80full work/arrivals/caps/grid/R0/Elegacy/Medium/seed1,8/60sec/6000rounds remain. No Neutral rerun,optional4,104/160,parameter search. Memory2GiB,disk5GiB; independent processes serial. No formal cell launched until acceptance and prereg freeze.

Read execution manifest before any recovery; never duplicate a supervisor. Independent raw-event audit, paired completion-only comparisons, CPU/RAM interaction only if four cells complete, record clipping and compute time. Stop after report. SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE.
''',encoding='utf8')
 print('ISOLATED',T,O,resources,len(p['source_hashes']))
if __name__=='__main__':main()
