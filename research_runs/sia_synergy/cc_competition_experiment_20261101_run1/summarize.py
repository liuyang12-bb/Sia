import csv,gzip,json,sys
from pathlib import Path
import raw_audit_lib as lib
O=Path(__file__).resolve().parent
def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
p=json.loads((O/'preregistration.json').read_text());stats={};jobs={};pairs={};extra={}
for cell in p['cells']:
 name=cell['id'];cfg=json.loads((O/('inputs/'+cell['capacity']+'.json')).read_text(encoding='utf8'))
 pp={**p,'capacities':cfg['cluster']['capacity_scenarios']['main']}
 with (O/('inputs/'+cell['workload']+'.csv')).open(encoding='utf8',newline='') as f:work={r['name']:r for r in csv.DictReader(f)}
 stats[name],jobs[name],_=lib.audit(cell,pp,work);assert not stats[name]['errors'],(name,stats[name]['errors'])
 shape=[];locks=0;serial=0;triple=0
 for l in gzip.open(O/'cells'/name/'rounds.jsonl.gz','rt',encoding='utf8'):
  r=json.loads(l);d=r['decision_for_next_interval'];a=d['result']['audit'];sel=d['result']['selected']
  triple+=sum(j.startswith('cifar10') for j in sel)>=3
  for s in d['probe']['specs']:
   if s['checkpoint_locked'] and s['current_config_id']:assert s['job_id'] in sel;locks+=1
  for z in a.get('serial_proposals',[]):assert z['fixed']=={v['job']:v['config'] for v in z['selected']};serial+=1
  shape.append(dict(time=r['interval_end'],selected={j:dict(config=c['config']['config_id'],cpu=c['cpu_per_gpu'],ram=c['ram_gb_per_gpu'],rho=c['rho']) for j,c in sel.items()}))
 extra[name]=dict(locks=locks,serial_proposals=serial,three_cifar_rounds=triple,decisions=shape)
for i in range(0,len(p['cells']),2):
 a,b=[c['id'] for c in p['cells'][i:i+2]]
 if not(stats[a]['complete'] and stats[b]['complete']):pairs[a]=dict(status='CENSORED');continue
 saved={j:jobs[a][j]['jct']-jobs[b][j]['jct'] for j in jobs[a]};total=sum(saved.values());best=max(saved,key=saved.get)
 pairs[a]=dict(full=b,mean_improvement_pct=100*(1-stats[b]['mean']/stats[a]['mean']),per_job_saved=saved,
  better=sum(v>0 for v in saved.values()),worse=sum(v<0 for v in saved.values()),equal=sum(v==0 for v in saved.values()),
  total_jct_saved=total,top_job=best,top_saved=saved[best],top_share=saved[best]/total if total else None,
  without_top_mean_saved=(total-saved[best])/11,
  time_saved={k:stats[a]['totals'][k]-stats[b]['totals'][k] for k in ('wait','training','checkpoint')})
save('raw_audit.json',stats);save('per_job.json',jobs);save('comparison_summary.json',pairs);save('decision_evidence.json',extra)
lines=['# 三敏感任务竞争场景：FO/FULL完整对照','','SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE','',
'完整12作业；前三CIFAR同时到达；冻结目标/候选/R0/E-legacy，未调参。真实门槛及源码/输入/运行命令均在本目录。',
'','|方法|完成|平均JCT|Median|P95|P99|Makespan|等待s|训练s|checkpoint s|重启|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for n,s in stats.items():
 t=s['totals'];lines.append(f"|{n}|{s['completed']}/12|{s['mean']}|{s['median']}|{s['p95']}|{s['p99']}|{s['makespan']}|{t['wait']}|{t['training']}|{t['checkpoint']}|{t['restarts']}|")
lines+=['','## FULL相对FO','']
for n,v in pairs.items():lines.append(f"- {n}: {json.dumps(v,ensure_ascii=False)}")
lines+=['','## 逐作业JCT','','|作业|'+'|'.join(stats)+'|','|---|'+'---:|'*len(stats)]
for j in next(iter(jobs.values())):lines.append('|'+j+'|'+'|'.join(str(jobs[n][j]['jct']) for n in stats)+'|')
lines+=['','## 资源与模型边界','','|方法|GPU h|CPU h|RAM GB h|wasted GPU s|裁剪比例|','|---|---:|---:|---:|---:|---:|']
for n,s in stats.items():
 t=s['totals'];lines.append(f"|{n}|{t['gpu_hours']}|{t['cpu_core_hours']}|{t['ram_gb_hours']}|{t['wasted_gpu_seconds']}|{s['clipped_fraction']}|")
lines+=['','逐完成事件独立复算JCT、互斥时间、资源小时、rho、锁、容量，错误0。所有比例/联合方法共用执行函数。P95/P99只描述12作业经验分位。求解墙钟不计入模拟JCT，沿用旧统计。E-legacy裁剪仍是合成模型假设；不代表真实Saturn性能。',
'','同状态净预测收益不能直接当作全轨迹JCT因果贡献。到达/Neutral/容量对照共同限制解释范围；逐作业退化和长尾必须保留。完成后停止，不扩大实验。']
(O/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
print('AUDIT zero errors; report saved')
