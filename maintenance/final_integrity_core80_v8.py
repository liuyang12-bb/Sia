"""Independent final sealing after the observer exits; no experiment launch.

An observer cannot hash its own still-open stdout after its final print.
Preserve that first seal; prove any discrepancy is append-only progress-log
growth, verify every scientific artifact, then add a closed-file manifest.
"""
import hashlib,json,time,shutil,psutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'research_runs/sia_synergy/cc_core80_full_lex_repair_20261031_run5'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
 assert (O/'seal_receipt.json').exists(),'Observer must finish first'
 for _ in range(60):
  active=[p.pid for p in psutil.process_iter() if p.name().lower()=='python.exe' and any(Path(x).name=='watch_core80_full_lex_v8.py' for x in p.cmdline())]
  if not active:break
  time.sleep(1)
 assert not active,'Never seal an open observer log'
 old=json.loads((O/'final_hashes.json').read_text(encoding='utf8'));mismatch=[];verified=0
 for name,expected in old.items():
  actual=sha(O/name)
  if actual==expected:verified+=1;continue
  assert name.replace('\\','/')=='watch.log',('SCIENTIFIC_ARTIFACT_CHANGED',name)
  data=(O/name).read_bytes();offset=0;prefix=None
  for line in data.splitlines(keepends=True):
   if hashlib.sha256(data[:offset]).hexdigest().upper()==expected:prefix=offset;break
   offset+=len(line)
  assert prefix is not None,'Observer log is not provably append-only'
  mismatch.append(dict(path=name,old_hash=expected,closed_hash=actual,prefix_bytes=prefix,trailer=data[prefix:].decode('utf8'),reason='Observer printed completion AFTER hashing its own redirected stdout; original seal retained.'))
 p=json.loads((O/'preregistration.json').read_text(encoding='utf8'));T=Path(p['source_root'])
 assert all(sha(T/n)==v for n,v in p['source_hashes'].items())
 assert all(sha(Path(n))==v for n,v in p['input_hashes'].items())
 ledger=[json.loads(x) for x in (O/'execution_manifest.jsonl').read_text(encoding='utf8').splitlines() if x];assert ledger[-1]['event']=='BATCH_END'
 cell_files=0
 for e in ledger:
  if e['event']!='CELL_END':continue
  for n,v in e['output_hashes'].items():assert sha(O/n)==v,('CELL_ARTIFACT_CHANGED',n);cell_files+=1
 report=R/'research_notes/synergy/CC_CORE80_FULL_LEX_INTEGER_BUDGET_COMPARISON_20261031.md';assert sha(report)==sha(O/'REPORT.md')
 result=json.loads((O/'result_summary.json').read_text(encoding='utf8'))
 save(O/'observer_seal_reconciliation.json',dict(original_seal_files=len(old),unchanged=verified,discrepancies=mismatch,scientific_artifacts_unchanged=True,cell_receipt_files=cell_files,source_files=len(p['source_hashes']),inputs=len(p['input_hashes']),external_report_sha256=sha(report),complete_cells=result['complete'],expected_cells=7))
 with (O/'INTEGRITY_ADDENDUM.md').open('x',encoding='utf8') as f:f.write('# 独立最终完整性核验\n\n原始事件、结果、源码、输入和CELL_END收据均逐文件复核。旧seal保留；其自写watch.log若在完成打印后追加，仅以原SHA匹配的完整字节前缀证明日志追加，未修改原日志。observer_seal_reconciliation.json记录全部差异。进程退出后生成final_integrity_hashes.json与final_integrity_receipt.json，作为关闭文件的最终核验入口。没有改动任何科学结果或运行源码。\n')
 shutil.copyfile(Path(__file__),O/'postseal_integrity_script.py')
 hashes={q.relative_to(O).as_posix():sha(q) for q in O.rglob('*') if q.is_file() and q.name not in ('final_integrity_hashes.json','final_integrity_receipt.json')}
 save(O/'final_integrity_hashes.json',hashes)
 save(O/'final_integrity_receipt.json',dict(unix=time.time(),files=len(hashes),manifest_sha256=sha(O/'final_integrity_hashes.json'),source_snapshot_sha256=p['source_snapshot_sha256'],source_files=len(p['source_hashes']),inputs=len(p['input_hashes']),external_report_sha256=sha(report),scientific_artifact_mismatches=0,observer_append_only_discrepancies=len(mismatch),complete_cells=result['complete'],expected_cells=7))
 print(json.dumps(dict(status='CLOSED_FILE_INTEGRITY_VERIFIED',complete=result['complete'],expected=7,files=len(hashes),source=p['source_snapshot_sha256'],scientific_artifact_mismatches=0,observer_append_only_discrepancies=len(mismatch)),ensure_ascii=False))
if __name__=='__main__':main()
