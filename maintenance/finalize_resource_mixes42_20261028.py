"""Stable post-process hashes and direct outcome interpretation, after child logs close."""
import csv,hashlib,json,shutil,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_resource_mixes42_20261028_run1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def read(n):return json.loads((OUT/n).read_text(encoding='utf8'))
def save(n,v):
 with (OUT/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
 pre=read('preregistration.json');records=[json.loads(s) for s in (OUT/'execution_manifest.jsonl').read_text().splitlines() if s.strip()]
 assert len(records)==42 and all(r['exit_code']==0 for r in records)
 assert sha(ROOT/'maintenance/run_resource_mixes42_20261028.py')==pre['runner_hash']
 assert sha(ROOT/'maintenance/audit_resource_mixes42_20261028.py')==pre['audit_script_hash']
 helper=read('audit_helper.sha256.json')
 assert sha(ROOT/helper['path'])==helper['sha256'] and sha(OUT/'audit_helper.py')==helper['sha256']
 for n,h in pre['source_hashes'].items():assert sha(OUT/'source'/n)==h and sha(ROOT/'cc-joint-resource-study'/n)==h
 for n,h in pre['input_hashes'].items():assert sha(OUT/'inputs'/n)==h
 for mix,cap,response,method in pre['cells']:
  rows=list(csv.DictReader((OUT/'inputs'/(mix+'_workload.csv')).open(encoding='utf8')))
  result=read('cells/'+'_'.join((mix,cap,response,method))+'/summary.json')
  assert len(result['jobs'])==20 and set(result['jobs'])=={r['name'] for r in rows}
  for r in rows:assert float(result['jobs'][r['name']]['arrival'])==float(r['time'])
 for name,h in read('historical_hash_manifest_files.json').items():
  mf=ROOT/name;assert sha(mf)==h
  for n,v in json.loads(mf.read_text()).items():assert sha(mf.parent/n)==v
 with zipfile.ZipFile(OUT/'source_snapshot.zip') as z:assert z.testzip() is None
 val=read('analysis/validation.json');assert val['status']=='COMPLETE42_AUDITED'
 effects=read('analysis/effect_summary.json');paired=read('analysis/paired_comparisons.json');coverage=read('analysis/coverage.json')
 oldrows=list(csv.DictReader((OUT/'source/inputs/workload.csv').open(encoding='utf8')))
 newrows=list(csv.DictReader((OUT/'inputs/M1_workload.csv').open(encoding='utf8')))
 oldmap={r['name']:r for r in oldrows};newmap={r['name']:r for r in newrows}
 changed=[n for n in oldmap.keys()&newmap.keys() if float(oldmap[n]['time'])!=float(newmap[n]['time'])]
 save('analysis/M1_vs_historical_input.json',dict(same_job_set=set(oldmap)==set(newmap),changed_arrival_jobs=sorted(changed),changed_arrival_count=len(changed)))
 def describe(k):
  v=effects[k]
  if not v['cases']:return f'{k}：无完整配对，不能比较完整JCT。'
  return f"{k}：{v['wins']}个场景改善、{v['ties']}个持平、{v['losses']}个退化；改善率中位数{v['median']*100:.2f}%，范围{v['min']*100:.2f}%至{v['max']*100:.2f}%。"
 conclusions=['## 直接结论','',f"已完成42/42格，全部20/20的格数为{sum(x['completed']==20 for x in coverage)}。下列比较以完整配对的平均JCT为准：",'']
 for k in ['CC_FULL_vs_SIA_P','CC_FULL_vs_SIA_AUX','CC_CPU_vs_CC_P','CC_MEM_vs_CC_P','CC_FULL_vs_CC_CPU','CC_FULL_vs_CC_P']:conclusions+=['- '+describe(k)]
 conclusions+=['','CPU与内存收益必须用共享CC-TAIL机制的消融判断，不能用FULL对Sia的全部改善替代。数据确实有支持域内可行的辅助资源交换及GPU数量耦合，但上述完整训练结果才决定是否兑现；存在机会与每种配额自由度都必然有利不是同一命题。',
 f'新M1与历史30格中的20作业集合一致，但有{len(changed)}个作业的受控到达时间改变，因此不能把新旧结果当同输入复跑。新矩阵内部六方法始终共享对应CSV，以组内配对判断增量。',
 '这些比例是三组嵌套受控负载、两个合成容量档下的描述性结果，不是独立多trace显著性检验；不同混合比例同时改变总训练量、任务长度及到达位置，不能把跨组差异唯一归因于敏感作业占比，核心对照是组内同任务配额消融。边界常值延拓未做其他模型的稳健性验证。',
 'SIA-AUX在各自当前状态调用同一Sia GPU/placement接口后冻结当轮意图；不同策略训练进度分叉后，不要求两条完整轨迹GPU意图相同。',
 '本轮内存检验受既有离散网格限制：已知DGX的CF/DS连续无损互补点约103.28/84.22GB不在网格，网格内125/62.5GB交换会让DS减速。该缺口早于本轮结果已记录，本轮没有加点；不能将内存配额消融无收益直接解释为内存感知无价值，也不能断言增加该点就必然降低JCT。局部进度目标、让出方完成代价及后续排队反馈仍需区分。',
 'CPU/RAM配额改变没有额外数据加载成本模型；checkpoint/rescale/migration仍使用原Sia时延。局部响应积分固定实际训练片段，不能当作合法调度反事实或完成时间收益，尤其不会重新模拟反事实完成/资源释放。',
 '资源小时和节点利用率由实际完成前的已分配GPU/CPU/RAM配额积分，不是硬件实测SM利用率、CPU busy或RSS；训练效率由统一模拟响应函数表示。',
 '本批只有CIFAR10/DeepSpeech2，50/25秒延迟通常在一个60秒周期内消耗，完整锁检查的实际触发数以raw_audit.json为准；触发数为0时不能把零违规当作覆盖。较长BERT checkpoint路径由既有229项回归覆盖。','']
 path=OUT/'REPORT.md';old=path.read_text(encoding='utf8');first,rest=old.split('\n',1)
 shutil.copyfile(path,OUT/'REPORT_tables_before_interpretation.md')
 path.write_text(first+'\n\n'+'\n'.join(conclusions)+'\n'+rest,encoding='utf8')
 start=read('cells/'+records[0]['cell']+'/start.json')['time'];end=read('CURRENT_PROGRESS.json')['time']
 final=dict(status='COMPLETE42_DELIVERED_STOPPED',cells=42,complete20_cells=sum(x['completed']==20 for x in coverage),
  tests=dict(existing=229,new=2,short_execution_cases=12),source_files_unchanged=len(pre['source_hashes']),historical30cell_outputs_unchanged=True,
  source_archive_valid=True,capacity_violations=0,checkpoint_lock_violations=0,rho_progress_violations=0,
  scientific_parameters_changed=False,formal_batch_wall_seconds=end-start,total_child_wall_seconds=sum(r['wall_seconds'] for r in records),
  delivered_unix=time.time(),warning=pre['warning'])
 save('FINAL_VALIDATION.json',final)
 with path.open('a',encoding='utf8') as f:f.write(f"\n\n最终验收：{final['status']}。正式首格开始到末格结束墙钟{end-start:.2f}秒，独立训练子进程累计{final['total_child_wall_seconds']:.2f}秒；不包含前置准备/测试和后置分析。旧30格原始输出逐文件复核未变，源码快照与当前CC-TAIL全部282文件一致。最终稳定哈希见delivery_hashes_final.json。\n")
 for p in ROOT.glob('maintenance/*resource_mixes42_20261028.py'):
  target=OUT/p.name
  if target.exists():assert sha(target)==sha(p),p
  else:shutil.copyfile(p,target)
 (OUT/'CURRENT_TASK.md').write_text('COMPLETE42_DELIVERED_STOPPED. REPORT.md, FINAL_VALIDATION.json, analysis/, delivery_hashes_final.json authoritative. All42 formal cells retained. No further run, tuning, or experiment authorized.\n',encoding='utf8')
 (OUT/'HASH_SCOPE.md').write_text('Final directory verification uses delivery_hashes_final.json, generated after audit/description logs close and report interpretation completes. Per-cell output_hashes.json protects sealed original start/log/summary files. analysis/delivery_hashes.json is an intermediate audit-stage snapshot; report, CURRENT_TASK and live audit log can subsequently change during documented report assembly, so it is not the final directory manifest. Raw cell outputs were never rewritten.\n',encoding='utf8')
 save('delivery_hashes_final.json',{p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='delivery_hashes_final.json'})
 print(json.dumps(final,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
