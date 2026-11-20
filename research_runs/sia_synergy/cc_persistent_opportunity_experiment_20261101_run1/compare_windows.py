"""Post-run accounting only; no model fitting, optimization or simulation."""
import gzip,json
from pathlib import Path
O=Path(__file__).resolve().parent;OLD=O.parent/'cc_competition_experiment_20261101_run1'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(n,x):
 with (O/n).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
new=read(O/'raw_audit.json');old=read(OLD/'raw_audit.json');pairs=read(O/'comparison_summary.json')
comparison={}
for response,old_prefix in [('medium','primary'),('neutral','response_control')]:
 a,b=new['long_'+response+'_FO'],new['long_'+response+'_FULL'];x,y=old[old_prefix+'_FO'],old[old_prefix+'_FULL']
 comparison[response]=dict(short_fo=x['mean'],short_full=y['mean'],long_fo=a['mean'],long_full=b['mean'],
  short_improvement_pct=100*(1-y['mean']/x['mean']),long_improvement_pct=100*(1-b['mean']/a['mean']),
  algorithm_by_arrival_interaction_seconds=(a['mean']-b['mean'])-(x['mean']-y['mean']),
  scope='arrival intervention changes queues and future GPU releases; not a pure isolated CPU effect')
trace={};end={}
for method in ('FO','FULL'):
 rows=[]
 for l in gzip.open(O/'cells'/('long_medium_'+method)/'rounds.jsonl.gz','rt',encoding='utf8'):
  r=json.loads(l)
  if 240<=r['interval_end']<=1260:
   rows.append(dict(time=r['interval_end'],selected=r['decision_for_next_interval']['result']['selected'],
    execution=[e for e in r['execution'] if e['job_id'].startswith('cifar10')],job_end_state=r['job_end_state']))
  if r['interval_end']==1200:end[method]=r['job_end_state']
 trace[method]=rows
gains={j:end['FULL'][j]['progress']-end['FO'][j]['progress'] for j in end['FO'] if j.startswith('cifar10') and j in ('cifar10-0','cifar10-1','cifar10-2')}
save('window_comparison.json',comparison);save('persistent_trace.json',trace);save('progress_at_1200.json',gains)
lines=['\n## 短窗口与长窗口完整比较\n','|响应|短FO|短FULL|短改善%|长FO|长FULL|长改善%|','|---|---:|---:|---:|---:|---:|---:|']
for response,c in comparison.items():lines.append(f"|{response}|{c['short_fo']:.3f}|{c['short_full']:.3f}|{c['short_improvement_pct']:.3f}|{c['long_fo']:.3f}|{c['long_full']:.3f}|{c['long_improvement_pct']:.3f}|")
lines+=['\n240–1200秒累计实际进度差FULL−FO：'+str(gains),
'\n对照保留相同全部12任务、batch、GPU demand及训练总工作量，仅非零到达后移900秒。不同到达实验之间的均值变化不能单独归因于训练加速；应看同一到达下的FO/FULL配对、接收方及其余任务成本。Neutral同样保留GPU/辅助容量硬约束，不能简单将两档均值差当独立CPU因果贡献。',
'\n源码343文件及响应不变，两项适配/三轮精确回归测试通过；既有393项测试记录保留，本轮未再次全套运行。所有性能指标由原始完成事件和训练片段独立复算，未插补。源矩阵裁剪仍按E-legacy披露。']
with (O/'REPORT.md').open('a',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
print(comparison);print('ACTUAL PROGRESS GAIN',gains)
