import collections,csv,gzip,hashlib,json,time
from pathlib import Path
from raw_audit_lib import audit
O=Path(__file__).resolve().parent
def read(n):return json.loads((O/n).read_text(encoding='utf8'))
def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
p=read('preregistration.json');g=read('gate_result.json');lines=['# 12作业混合场景：真实冷启动机会与FO/FULL配对','','SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE','','算法/响应/容量/工作量保持冻结；6CIFAR+4DeepSpeech2+2BERT，两波0/600秒，AWS4GPU/18CPU/125GB，RTX4GPU/12CPU/375GB。Medium alpha=.5，R0/E-legacy，seed_profiles(1,8)，60秒/6000轮。','',f"真实事件机会门槛：{'通过' if g['passed'] else '未通过'}。40轮冷启动前缀，事件时同状态比较，再要求实际两路径GPU差异持续≥6周期且FULL至少2周期推进；没有额外注入gradient。",'', '```json',json.dumps(g['witnesses'],ensure_ascii=False,indent=2),'```']
if not g['passed']:
 lines+=['','按冻结门槛停止，未启动完整FO/FULL。不能报告完整平均JCT。单次静态机会不等于在线可达、持续或JCT收益。事件明细见gate_result.json；所有完整前缀保留。']
else:
 work={r['name']:r for r in p['workload']};stats={};jobs={};releases={}
 for cell in p['cells']:
  method=cell['method'];stats[method],jobs[method],_=audit(cell,p,work)
  assert not stats[method]['errors'],stats[method]['errors'][:5]
  previous={};seen=set();ev=[]
  for l in gzip.open(O/'cells'/method/'rounds.jsonl.gz','rt',encoding='utf8'):
   r=json.loads(l);sel=r['decision_for_next_interval']['result']['selected']
   done={e['job_id']:e['completion_time'] for e in r['execution'] if e['completion_time'] is not None and e['job_id'] not in seen}
   if done:
    ev.append(dict(interval_end=r['interval_end'],completed=done,released_bundles={j:previous.get(j) for j in done},
      survivor_changes={j:dict(before=previous[j],after=c) for j,c in sel.items() if j in previous and c['column_id']!=previous[j]['column_id']}))
    seen.update(done)
   previous=sel
  releases[method]=ev
 save('raw_audit.json',stats);save('per_job.json',jobs);save('resource_release_events.json',releases)
 complete=all(s['completed']==12 for s in stats.values())
 saved={j:jobs['FO'][j]['jct']-jobs['FULL'][j]['jct'] for j in work} if complete else {}
 comparisons=dict(complete=complete,mean_improvement_pct=100*(1-stats['FULL']['mean']/stats['FO']['mean']) if complete else None,
  per_job_saved=saved,receiver_jct_saved=sum(max(0,v) for v in saved.values()),donor_jct_loss=-sum(min(0,v) for v in saved.values()),
  net_jct_saved=sum(saved.values()),deepspeech_net_saved=sum(v for j,v in saved.items() if j.startswith('deepspeech2')),
  improves=sum(v>0 for v in saved.values()),worsens=sum(v<0 for v in saved.values()),ties=sum(v==0 for v in saved.values()))
 save('comparison_summary.json',comparisons)
 lines+=['','## 完整结果（从原始完成事件独立复算）','','|方法|完成|平均JCT|Median|P95|P99|Makespan|','|---|---:|---:|---:|---:|---:|---:|']
 for m,s in stats.items():lines.append('|'+ '|'.join(str(v) for v in [m,s['completed'],s['mean'],s['median'],s['p95'],s['p99'],s['makespan']])+'|')
 lines+=['','```json',json.dumps(comparisons,indent=2,ensure_ascii=False),'```','','|作业|FO JCT|FULL JCT|节省s|FO等待|FULL等待|FO训练|FULL训练|FO checkpoint|FULL checkpoint|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for j in work:
  x,y=jobs['FO'][j],jobs['FULL'][j]
  lines.append('|'+ '|'.join(str(v) for v in [j,x['jct'],y['jct'],saved.get(j),x['wait'],y['wait'],x['training'],y['training'],x['checkpoint'],y['checkpoint']])+'|')
 lines+=['','## 资源、成本与审计','','|方法|GPU h|CPU h|RAM GB h|等待s|训练s|checkpoint s|重启|wasted GPU s|裁剪选中列比例|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for m,s in stats.items():
  z=s['totals'];lines.append('|'+ '|'.join(str(v) for v in [m,*[z[k] for k in ('gpu_hours','cpu_core_hours','ram_gb_hours','wait','training','checkpoint','restarts','wasted_gpu_seconds')],s['clipped_fraction']])+'|')
 lines+=['','逐节点容量、checkpoint完整锁、独立rho复算、raw×rho执行率、互斥时间与资源小时均从原始日志复算，错误数0。统计详见raw_audit.json。P95/P99仅12作业经验分位。完整均值仅在12/12配对时计算。','', '资源释放的事件顺序及所有存活任务配置变化保存在resource_release_events.json；两路径历史已经分叉，不能将全部收益归因于某一次释放。接收方收益、DeepSpeech2损失与总和必须共同解释。']
 with (O/'per_job.csv').open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(['job','FO_jct','FULL_jct','saved']);w.writerows((j,jobs['FO'][j]['jct'],jobs['FULL'][j]['jct'],saved.get(j)) for j in work)
lines+=['','## 可复现性与边界','','来源与输入hash见preregistration.json，执行脚本hash见runner_receipt.json，344文件源码和旧完整snapshot保持不变。旧结果未覆盖；未修改目标、候选、phi、裁剪或配额。该场景及R0迁移为合成机制验证，不能写成真实Saturn硬件收益。所有最终hash见artifact_hashes.json。']
(O/'REPORT.md').open('x',encoding='utf8').write('\n'.join(lines)+'\n')
save('artifact_hashes.json',{str(x.relative_to(O)):hashlib.sha256(x.read_bytes()).hexdigest().upper() for x in O.rglob('*') if x.is_file() and x.name not in ('artifact_hashes.json','pipeline.log')})
print('REPORT_READY',flush=True)
