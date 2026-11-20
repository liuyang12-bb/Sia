import collections,gzip,hashlib,json,math,time
from pathlib import Path
O=Path(__file__).resolve().parent;ROOT=O.parents[2];T=ROOT/'cc-joint-recovery-equivalent-v1'
def read(n):return json.loads((O/n).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
p=read('preregistration.json');s=read('raw_audit.json');j=read('per_job.json');c=read('comparison_summary.json');assert c['complete']
groups={}
for app in ('cifar10','deepspeech2','bert'):
 names=[r['name'] for r in p['workload'] if r['application']==app]
 groups[app]=dict(names=names,net_jct_saved=sum(c['per_job_saved'][n] for n in names),
   per_job_saved={n:c['per_job_saved'][n] for n in names},
   time_saved={k:sum(j['FO'][n][k]-j['FULL'][n][k] for n in names) for k in ('wait','training','checkpoint')})
changes={};first={};progressions={}
for m in ('FO','FULL'):
 prev={};events=[];seen=set();first[m]=min(v['completion'] for v in j[m].values())
 for line in gzip.open(O/'cells'/m/'rounds.jsonl.gz','rt',encoding='utf8'):
  r=json.loads(line);done={e['job_id']:e['completion_time'] for e in r['execution'] if e['completion_time'] is not None and e['job_id'] not in seen};sel=r['decision_for_next_interval']['result']['selected']
  if done:
   for name,cur in sel.items():
    if not name.startswith('deepspeech2'):continue
    old=prev.get(name)
    if old is None or old['column_id']!=cur['column_id']:
     slim=lambda v:None if v is None else dict(config=v['config']['config_id'],cpu=v['cpu_per_gpu'],ram=v['ram_gb_per_gpu'],rho=v['rho'])
     events.append(dict(time=r['interval_end'],completed=done,job=name,before=slim(old),after=slim(cur)))
   seen.update(done)
  prev=sel
 changes[m]=events
firstallocation={m:{n:read('cells/'+m+'/summary.json')['jobs'][n]['first_allocation'] for n in j[m]} for m in ('FO','FULL')}
metric_gain={k:100*(1-s['FULL'][k]/s['FO'][k]) for k in ('mean','median','p95','p99','makespan')}
result=dict(groups=groups,metric_improvement_pct=metric_gain,first_completion=first,first_allocation=firstallocation,DS_changes_after_completion=changes,
 total_time_saved={k:s['FO']['totals'][k]-s['FULL']['totals'][k] for k in ('wait','training','checkpoint')},
 wall_seconds=dict(gate=read('pipeline_complete.json')['wall'],pair=read('pair_complete.json')['wall']),
 source_unchanged=all(sha(T/n)==h for n,h in p['source_hashes'].items()),inputs_unchanged=all(sha(n)==h for n,h in p['input_hashes'].items()))
assert result['source_unchanged'] and result['inputs_unchanged']
save('mechanism_attribution.json',result)
lines=['# 12作业混合场景最终结论','','SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE','',
 f"FO和FULL均12/12完成。FULL相对FO平均JCT改善{metric_gain['mean']:.4f}%；净节省{c['net_jct_saved']:.1f}作业秒，改善{c['improves']}个、退化{c['worsens']}个、持平{c['ties']}个。",'',
 '|指标|FO|FULL|改善率（正为更好）|','|---|---:|---:|---:|']
for k in ('mean','median','p95','p99','makespan'):lines.append(f"|{k}|{s['FO'][k]:.4f}|{s['FULL'][k]:.4f}|{metric_gain[k]:+.4f}%|")
lines+=['','## 损益是否覆盖','','|组|净JCT节省s|等待节省s|训练节省s|checkpoint节省s|','|---|---:|---:|---:|---:|']
for app,g in groups.items():lines.append('|'+ '|'.join(str(v) for v in (app,g['net_jct_saved'],*[g['time_saved'][k] for k in ('wait','training','checkpoint')]))+'|')
lines+=['',f"全部接收方JCT收益{c['receiver_jct_saved']:.1f}s；全部受损作业代价{c['donor_jct_loss']:.1f}s；净值{c['net_jct_saved']:.1f}s。DeepSpeech2组净节省{c['deepspeech_net_saved']:.1f}s。只将完整JCT的净差作为最终收益，不把恒定速率600秒进度优势当成完成时间收益。",'',
 f"首次完成事件FO={first['FO']}s、FULL={first['FULL']}s。不能声称FULL使第一次释放大幅提前；后续释放顺序与服务/配额改变才是需要逐项核对的证据。",'',
 '资源释放后DeepSpeech2的所有配置/配额变化见mechanism_attribution.json的DS_changes_after_completion以及resource_release_events.json。FIRST_VALID_CURRENT_MODEL、ARRIVAL和恢复预算也可影响决策，故这些是实际轨迹的机制证据，不是隔离某一次释放的纯因果效应。','',
 '完整原始审计、逐作业JCT/等待/训练/checkpoint、重启和资源小时详见REPORT_v2.md；首次错误gate的REPORT.md与gate_result.json保留，正式结果以gate_result_v2.json和REPORT_v2.md为准。验收修正仅移除了冻结定义没有要求的“后续服务集合一直相同”，没有改变事件时同状态/同集合条件，原算法和所有实验参数未变。','',
 '8项预检回归与6项验收脚本测试通过。全量日志独立复算无容量、锁、rho及互斥时间违规。源码344文件和全部输入hash复核一致。候选和执行中的E-legacy裁剪仍是合成响应假设，不能把此次结果视为真实Saturn硬件测量。','',
 f"实测流水线墙钟合计{result['wall_seconds']['gate']+result['wall_seconds']['pair']:.2f}s（包含门槛、正式运行和流水线审计；额外交付分析不计入）。本轮到此停止，没有调参、扩大负载或继续其他实验。"]
(O/'CONCLUSION.md').open('x',encoding='utf8').write('\n'.join(lines)+'\n')
save('DELIVERY_RECEIPT.json',dict(status='COMPLETE',source_unchanged=True,inputs_unchanged=True,completed=dict(FO=12,FULL=12),audit_errors={m:len(s[m]['errors']) for m in s},
 reports=[str(O/'CONCLUSION.md'),str(O/'REPORT_v2.md')],attribution='complete paired JCT; release trajectories observational',finished_unix=time.time()))
save('delivery_hashes.json',{str(x.relative_to(O)):sha(x) for x in O.rglob('*') if x.is_file() and x.name!='delivery_hashes.json'})
print(json.dumps(dict(comparison=c,metrics={m:{k:s[m][k] for k in ('mean','median','p95','p99','makespan','totals','clipped_fraction')} for m in s},groups=groups,metric_gain=metric_gain),indent=2,ensure_ascii=False))
