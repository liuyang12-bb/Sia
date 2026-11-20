"""Post hoc trace accounting. Does not select parameters or rerun cells."""
import gzip,json
from pathlib import Path
O=Path(__file__).resolve().parent
def read_rows(name):
 return [json.loads(l) for l in gzip.open(O/'cells'/name/'rounds.jsonl.gz','rt',encoding='utf8')]
a,b=read_rows('primary_FO'),read_rows('primary_FULL')
out={};timelines={}
for name,rs in [('FO',a),('FULL',b)]:
 entries=[];timeline=[];last=None
 for r in rs:
  c=r['decision_for_next_interval']['result']['selected'].get('cifar10-2')
  geometry=(c['config']['config_id'],c['cpu_per_gpu'],c['ram_gb_per_gpu']) if c else None
  if geometry!=last:timeline.append(dict(time=r['interval_end'],allocation=geometry));last=geometry
  if r['interval_end'] in (180,240,300,360):
   d=r['decision_for_next_interval'];sel=d['result']['selected'];j='cifar10-2';c=sel.get(j)
   ex=next(e for e in r['execution'] if e['job_id']==j)
   entries.append(dict(time=r['interval_end'],selected=c,execution=ex,job_end_state=r['job_end_state'][j],
     peers={k:dict(config=v['config']['config_id'],cpu=v['cpu_per_gpu'],ram=v['ram_gb_per_gpu']) for k,v in sel.items() if k!=j}))
 out[name]=entries
 timelines[name]=timeline
out['timelines']=timelines
with (O/'mechanism_trace.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
lines=['\n## 真实机会的兑现追踪\n',
'门槛240秒，同服务集合：FO的cifar10-2使用AWS2，每GPU4.5CPU/31.25GB；FULL使用AWS1、9CPU/62.5GB；两个DGX CIFAR配额相同。当前目标和ΣETA均改善。FO与FULL两个独立前缀在这一时刻相同，因此是同一个机会的交叉核验，不能当两次独立重复。',
'逐段执行见mechanism_trace.json；配额变化、配置切换、batch变化和checkpoint共同决定实际进度，不能只按rho比值预测JCT。']
for name,rows in out.items():
 if name=='timelines':continue
 for v in rows:
  if v['time']==300:
   e=v['execution'];lines.append(f"- {name} 240–300s cifar10-2: GPU配置={e['resource_allocation']['config_id'] if e['resource_allocation'] else None}，训练{e['training_seconds']}s，checkpoint={e['checkpoint_delay_consumed']}s，实际进度增量={e['progress_delta']}，rho={e['rho']}。")
with (O/'REPORT.md').open('a',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
with (O/'REPORT.md').open('a',encoding='utf8') as f:
 f.write('\n该收缩机会只持续240–300秒一个调度周期：300秒BERT到达后，两策略的cifar10-2均回到AWS1、4.5CPU/31.25GB；480秒均采用8CPU/41.67GB，1320秒均迁到DGX1。FO能在随后到达事件中修正GPU配置，因此不能把240秒静态剩余完成时间差一直累积到结束。完整轨迹下cifar10-2的JCT只改善21秒；checkpoint减少100秒、有效训练时间反而增加79秒。后者涉及自适应batch/epoch及资源轨迹，单靠这次局部rho不能分解出独立贡献。\n')
