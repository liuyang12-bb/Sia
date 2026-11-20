"""New human authorization: freeze a fresh11-cell run, same315-file source."""
import hashlib,importlib.metadata,json,os,subprocess,sys,time
from pathlib import Path
import psutil
R=Path(__file__).resolve().parents[1];T=R/'cc-joint-core80-repair-v3';F=R/'research_runs/sia_synergy/cc_core80_repair_20261031_run3';O=R/'research_runs/sia_synergy/cc_core80_repair_full_20261031_run1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
 assert not O.exists(),'Refuse to reuse any existing run directory'
 p=json.loads((F/'planned_preregistration.json').read_text(encoding='utf8'));h=json.loads((F/'source_hashes.json').read_text());receipt=json.loads((F/'software_acceptance.json').read_text());assert receipt['passed']==receipt['total']==302 and not receipt['violations']
 assert len(h)==315 and all(sha(T/n)==v for n,v in h.items())
 assert sha(F/'source_snapshot.zip')=='7894EA3C16587113301D4568FAA09C2116E7EB88A171864C5B3D6AFBAD584DCC'
 assert len(p['input_hashes'])==38 and all(sha(Path(n))==v for n,v in p['input_hashes'].items())
 assert all(importlib.metadata.version(k)==v for k,v in p['environment']['packages'].items())
 active=[]
 for proc in psutil.process_iter(['cmdline']):
  args=proc.info['cmdline'] or []
  if any(Path(v).name in ('run_core80.py','supervise80.py') for v in args):active.append((proc.pid,args))
 assert not active,active
 resources=dict(memory_gib=psutil.virtual_memory().available/2**30,disk_gib=psutil.disk_usage(str(R)).free/2**30)
 assert resources['memory_gib']>=2 and resources['disk_gib']>=5,resources
 cells=p['cells'];assert len(cells)==11 and not any(c['optional'] for c in cells)
 assert [c['method'] for c in cells]==['SIA_P','FULL','SIA_AUX','PROP','CPU','MEM','FO','A00','A01','A10','A11']
 O.mkdir();(O/'cells').mkdir();now=time.time()
 authorization='确认解除本轮11格80作业补跑暂停，按冻结修复版继续。保持算法目标、数据、响应模型和参数不变，保留历史结果，完成后汇报完整对照结果。'
 save(O/'authorization.json',dict(human_instruction=authorization,recorded_unix=now,scope='Exactly frozen11core80-job cells; previous pause explicitly lifted for this scope ONLY. Not104/160 or optional4.',old_block_receipt=str(F/'FORMAL_BLOCKED_AUTO_REVIEW.json'),old_receipt_preserved=True))
 p.update(status='FROZEN_REAUTHORIZED',version='CORE80_315FILE_REPAIR_REAUTHORIZED_11',output=str(O),source_root=str(T),source_hashes=h,
  source_snapshot_path=str(F/'source_snapshot.zip'),source_snapshot_sha256=sha(F/'source_snapshot.zip'),software_acceptance_path=str(F/'software_acceptance.json'),
  frozen_unix=now,formal_stop_unix=now+24*3600,hard_deadline_unix=now+25*3600,cell_wall_limit_seconds=6*3600,
  operational_limits='Same repair protocol: common6h program compute cap percell,24h batch. These are computation watchdogs; simulated60sec/6000rounds and every solver/model parameter unchanged.',
  failure='Preserve failed model/raw logs, continue remaining fixed cells. No scientific tuning, no source edits during run; incomplete outcomes never get imputed JCT.',
  execution='Independent Python processes, serial;2GiB/5GiB before each. Every cell verifies all315source and38input hashes before construction.',
  analysis='Independent raw-event JCT/exclusive-time/resource-hour/capacity/lock/rho reconstruction. Only complete pairs get full meanJCT claims. No mixing historical versions or choosing best result.',
  warning='SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE')
 p.pop('no_launch_performed',None);p.pop('freeze_note',None)
 save(O/'preregistration.json',p);save(O/'preflight.json',dict(resources=resources,source_files_verified=315,input_files_verified=38,tests=302,duplicate_processes=active,prereg_sha256=sha(O/'preregistration.json')))
 (O/'CURRENT_TASK.md').write_text(f'''# ACTIVE: reauthorized11-cell80-job repaired comparison\n\nLatest human instruction: {authorization}\n\nFrozen source {T};315files,snapshotSHA7894EA3C16587113301D4568FAA09C2116E7EB88A171864C5B3D6AFBAD584DCC. Software acceptance302/302, FULL12/A1020/PROP12 and originalSia exact. Old blocked receipt and every history remain unchanged; this NEW human authorization lifts pause only for these11cells.\n\nUse preregistration.json and execution_manifest.jsonl here. Fixed order SIA_P,FULL,SIA_AUX,PROP,CPU,MEM,FO,A00,A01,A10,A11. Same80full declarations/arrivals/work/caps/R0-Elegacy/alpha0,.5/seed1,8/60sec/6000rounds. No source change,tuning,104,160,optional4.\n\nLaunch authority: maintenance/launch_reauthorized_core80.py invokes frozen supervise80.py exactly once. Inspect process/session and manifest before any restart; NEVER duplicate a running supervisor. Live filesystem/process reads may require escalated read-only exec due sandbox snapshots. Source/input hashes verified beforeeverycell. Safety2GiB/5GiB, common6h/cell and24h batch program watchdog, no shortened training work. Keep failures/partials, noJCTimputation.\n\nAfter BATCH_END: use adapted raw auditor for this directory and the unchanged frozen source, reconstruct completion events/exclusive times/resources/rho/locks/factor contracts; compare all11same-version cells. Deliver a NEW Chinese full comparison report, no rewrite of engineering/history reports, full hashes and wallclock. Stop after delivery or a genuine permission/scientific blocker.\n''',encoding='utf8')
 print('FROZEN_AND_REAUTHORIZED',sha(O/'preregistration.json'),resources,flush=True)
 return subprocess.run([sys.executable,'-X','utf8','-B',str(T/'supervise80.py'),str(O/'preregistration.json')],cwd=T).returncode
if __name__=='__main__':raise SystemExit(main())
