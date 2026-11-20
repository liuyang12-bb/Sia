"""Independent raw-event audit and paired comparison; never runs training."""
import collections,gzip,hashlib,itertools,json,math,sys,zipfile,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_joint_tail_recovery_20261027_run1'
sys.path.insert(0,str(OUT/'source/sia-simulator'))
import numpy as np
from joint_prototype.joint_profile import load_synergy_template,CPU_COORDS,RAM_COORDS
registry={r['key']:r for r in json.loads((OUT/'source/inputs/registry.json').read_text())};templates={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(n,d):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
def rho(job,cfg,cpu,ram):
 row=registry[job.rsplit('-',1)[0]+'/'+cfg];name=row['template']
 if name not in templates:templates[name]=load_synergy_template(OUT/'source/inputs/models'/name).matrix
 def f(c,m):return float(np.interp(c,CPU_COORDS,[np.interp(m,RAM_COORDS,r) for r in templates[name]]))
 return (f(cpu/row['phi_c'],ram/row['phi_m'])/f(3/row['phi_c'],62.5/row['phi_m']))**.5
def frames(p):
 with gzip.open(p,'rt',encoding='utf-8') as f:
  for line in f:yield json.loads(line)
def geometry(a):
 if not a:return None
 return (a.get('config_id') or a['config']['config_id'],a['cpu_per_gpu'],a['ram_gb_per_gpu'],tuple(sorted(a['placement'].items())))
def trajectory(r):
 return ([(e['job_id'],e['progress_before'],e['progress_after'],e['rho'],e['waiting_seconds'],e['checkpoint_delay_consumed'],e['training_seconds'],e['completion_time'],geometry(e['resource_allocation'])) for e in r['execution']],
         sorted((j,geometry(c)) for j,c in r['decision_for_next_interval']['result']['selected'].items()))
def compare(p,q):
 count=0
 for a,b in itertools.zip_longest(frames(p),frames(q)):
  if a is None or b is None:return dict(exact=False,matching_rounds=count,first_difference='length')
  if trajectory(a)!=trajectory(b):return dict(exact=False,matching_rounds=count,first_difference=a['interval_end'])
  count+=1
 return dict(exact=True,matching_rounds=count,first_difference=None)
def audit(path,ram):
 s=json.loads((path/'summary.json').read_text());checks=collections.Counter();jobs=collections.defaultdict(lambda:collections.Counter());last=None;previous={};tail=[];frozen_geometry={}
 for r in frames(path/'rounds.jsonl.gz'):
  for e in r['execution']:
   j=jobs[e['job_id']]
   for key,field in [('wait','waiting_seconds'),('checkpoint','checkpoint_delay_consumed'),('training','training_seconds')]:j[key]+=e[field]
   assert math.isclose(e['waiting_seconds']+e['checkpoint_delay_consumed']+e['training_seconds'],e['observed_active_seconds'],abs_tol=1e-6)
   if e['checkpoint_delay_consumed']>=e['observed_active_seconds']>0:assert e['progress_delta']==0;checks['checkpoint_no_progress']+=1
   if e['allocated_seconds']>0:
    a=e['resource_allocation'];assert geometry(a)==geometry(previous[e['job_id']]);checks['cross_round_atomic_match']+=1
    expected=rho(e['job_id'],a['config_id'],a['cpu_per_gpu'],a['ram_gb_per_gpu']);assert math.isclose(e['rho'],expected,rel_tol=1e-12)
    dt=e['allocated_seconds']/3600;j['gpu_hours']+=a['gpus']*dt;j['cpu_core_hours']+=sum(u['cpu'] for u in a['node_usage'])*dt;j['ram_gb_hours']+=sum(u['ram_gb'] for u in a['node_usage'])*dt
    for z in e['training_segments']:
     assert math.isclose(z['effective_rate'],z['raw_rate_before_rho']*expected,rel_tol=1e-12)
     assert math.isclose(z['progress_delta'],z['effective_rate']*z['seconds']+z['rounding_progress_residual'],abs_tol=1e-7);checks['rho_once']+=1
   assert e['progress_delta']>=-1e-9;j['arrival']=e['arrival_time'];j['completion']=e['completion_time']
  d=r['decision_for_next_interval'];a=d['result']['audit'];selected=d['result']['selected']
  if d['round']!=last:
   last=d['round'];usage=collections.defaultdict(lambda:[0.,0.,0.]);specs={j['job_id']:j for j in d['probe']['specs']}
   for name,c in selected.items():
    for node,g in c['placement'].items():
     for k,x in enumerate([g,g*c['cpu_per_gpu'],g*c['ram_gb_per_gpu']]):usage[node][k]+=x
    q=specs[name]
    if q['checkpoint_locked']:
     assert geometry(c)==(q['current_config_id'],q['current_cpu_per_gpu'],q['current_ram_gb_per_gpu'],tuple(tuple(x) for x in q['current_placement']));checks['full_lock']+=1
    if name in frozen_geometry:assert (c['config']['config_id'],tuple(sorted(c['placement'].items())))==frozen_geometry[name]
   for u in usage.values():assert all(x<=y+1e-7 for x,y in zip(u,[2,12,ram]));checks['node_capacity']+=1
   for x in a.get('tail_recoveries',[]):
    name=x['job'];assert len(specs)==1 and not a['ready'] and not specs[name]['checkpoint_locked'];assert name not in frozen_geometry
    score=next(v for v in d['continuity_scores'] if v['column_id']==x['column_id'])
    assert math.isclose(score['estimated_completion'],x['eta_after'],abs_tol=1e-9)
    assert x['eta_after']<x['eta_before'] and score['delay']==x['actual_startup']
    assert 0<=x['extra_budget']<=x['actual_startup']<=250
    frozen_geometry[name]=(selected[name]['config']['config_id'],tuple(sorted(selected[name]['placement'].items())))
    tail.append(x)
   for name,v in a.get('service_delay_ledgers',{}).items():assert v['delay_charged']<=250+a.get('tail_grants',{}).get(name,{}).get('extra_budget',0.)+1e-9
  previous=selected
 for name,j in jobs.items():
  for key in ['wait','checkpoint','training','gpu_hours','cpu_core_hours','ram_gb_hours']:assert math.isclose(j[key],s['jobs'][name][key],abs_tol=1e-6),(name,key)
  if j['completion'] is not None:
   j['jct']=j['completion']-j['arrival'];assert math.isclose(j['jct'],j['wait']+j['checkpoint']+j['training'],abs_tol=1e-6)
 vals=[j['jct'] for j in jobs.values() if j['completion'] is not None]
 if len(vals)==20:
  for key,val in [('mean_jct',np.mean(vals)),('median_jct',np.median(vals)),('p95_jct',np.percentile(vals,95)),('p99_jct',np.percentile(vals,99))]:assert math.isclose(s[key],float(val),abs_tol=1e-6)
 return dict(valid=True,summary=s,checks=dict(checks),tail_recoveries=tail,recomputed_jobs={j:dict(v) for j,v in jobs.items()})
def main():
 p=json.loads((OUT/'preregistration.json').read_text());results={};comparisons={};prefix={};historical={}
 for ram,method in p['cells']:
  name=f'ram{ram:g}_{method}';path=OUT/'cells'/name
  if (path/'summary.json').exists():results[name]=audit(path,ram)
 for ram in [250.,187.5]:
  b=f'ram{ram:g}_JOINT_CC';n=b+'_TAIL'
  old=ROOT/'research_runs/sia_synergy'/('cc_joint_sustained_cpu_20job_20261027_run1/cells/medium' if ram==250 else 'cc_joint_queue_ram_tail_20261027_run3/cells')
  if b in results:historical[b]=compare(OUT/'cells'/b/'rounds.jsonl.gz',old/'JOINT_CC/rounds.jsonl.gz');assert historical[b]['exact']
  if b in results and n in results:
   x=results[n]['summary'];y=results[b]['summary'];prefix[n]=compare(OUT/'cells'/b/'rounds.jsonl.gz',OUT/'cells'/n/'rounds.jsonl.gz')
   events=results[n]['tail_recoveries']
   assert prefix[n]['exact'] or (events and prefix[n]['first_difference']==events[0]['time'])
   sia=json.loads((old/'SIA_P/summary.json').read_text())
   if x['completed']==y['completed']==20:
    comparisons[str(ram)]=dict(vs_old={k:x[k]/y[k]-1 for k in ['mean_jct','p95_jct','p99_jct','makespan']},vs_sia={k:x[k]/sia[k]-1 for k in ['mean_jct','p95_jct','p99_jct','makespan']},
     per_job_delta={j:x['jobs'][j]['jct']-v['jct'] for j,v in y['jobs'].items()},sia=sia,
     time_delta={k:x['totals'][k]-y['totals'][k] for k in ['wait','checkpoint','training']})
 complete=len(results)==4 and all(v['summary']['completed']==20 for v in results.values())
 save('independent_audit.json',dict(valid=True,complete=complete,results=results,comparisons=comparisons,historical_baseline_exact=historical,prefix_matches=prefix))
 lines=['# CC Joint 有界收尾恢复对照','',f'状态：{"FOUR_COMPLETE_VALID" if complete else "INCOMPLETE"}。原CC保留；新版本只增加显式收尾恢复，不改候选网格、目标、响应或负载。','',
 'SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。','',
 '## 机制和有限性','',
 '触发仅依赖当前已到达状态：活动作业只有一个、ready为空、已有有效训练、没有真实checkpoint锁、目标配置已观测、完整资源合法。比较当前几何下最佳配额的ETA与新GPU候选，目标ETA含一次真实启动成本，必须严格更小。',
 '每任务最多一次额外恢复，额度仅补足本次实际需要，不超过单次实际启动D_bar=250秒；之后GPU几何冻结，CPU/RAM仍沿原网格优化。普通路径250秒预算不变，累计延迟<=250+一次实际启动<=500秒。采用trial-copy和完整审计，求解失败不消耗许可。',
 '有限任务、单独可行配置、正速率下界及有界启动是假设；有限迁移后持续训练，保留有限队列最终完成论证。不保证ETA无误或一般在线后续到达不受影响，也未读取未来到达以判定收尾。','',
 '## 测试和实现','',
 '隔离版本cc-joint-tail-recovery；原cc-joint和277文件基线快照保留。改动四文件：adaptive_service.py仅增加默认预算钩子；tail_recovery.py为新策略；cc_factory.py新增JOINT_CC_TAIL显式入口；test_tail_recovery.py新增12测试。',
 '初始缺模块红测12项的早期10例报错；另有真实旧行为红测：耗尽预算的断言期望恢复却仍为AWS。首次绿测一例peer夹具漏填启动成本，修复夹具后12项全过；完整224项全过。原始Sia独立3轮精确回归通过。所有失败日志保留。','',
 '## 完整训练结果','', '|RAM每节点|策略|完成|平均JCT|P95|P99|Makespan|Restart|','|---:|---|---:|---:|---:|---:|---:|---:|']
 for name,v in results.items():
  s=v['summary'];lines.append(f"|{s['ram_per_node']}|{s['method']}|{s['completed']}/20|{s['mean_jct']}|{s['p95_jct']}|{s['p99_jct']}|{s['makespan']}|{s['totals']['restarts']}|")
 for ram,c in comparisons.items():
  lines+=['',f"RAM{ram}：新模式相对原CC变化（负为改善）：{json.dumps(c['vs_old'],ensure_ascii=False)}；相对同容量Sia：{json.dumps(c['vs_sia'],ensure_ascii=False)}。",f"互斥任务时间变化：{c['time_delta']}。逐作业差值：{json.dumps(c['per_job_delta'],ensure_ascii=False)}。"]
 lines+=['','## 实际恢复与归因','',json.dumps({n:v['tail_recoveries'] for n,v in results.items()},ensure_ascii=False,indent=2),'',
 '新基线逐轮GPU/配额/进度/完成/互斥时间与历史原CC精确一致；新模式首次恢复前同样逐轮一致，首次差异必须等于实际恢复时刻。由此本轮差值聚焦收尾恢复，而不是负载或候选改变。不能从预测ETA差直接推导JCT收益，最终只用真实完成事件。',
 'P95/P99在20任务中仅是描述性尾部指标。只剩一个作业时恢复不能消除其他长任务已经承受的全部损失；若尾部仍不优于Sia，应明确保留。','',
 '## 适用范围与交付','',
 '四格独立进程串行、Medium alpha=.5、60秒6000轮、同一20作业和4GPU/24CPU，RAM分别250/187.5；2GiB/5GiB门槛。无调参、重跑择优、候选新增或响应修改。Sia仅作为同配置历史参考，不改其决策。',
 'source/是完整冻结执行源码，baseline_source/保存旧版；code.diff和source_changes.json记录最小差异；preregistration.json冻结输入、源码、脚本和执行顺序；cells保存原始压缩逐轮日志、完成/删失与逐作业summary；independent_audit.json独立核验容量、完整锁、一次许可、恢复后几何、raw×rho、互斥时间和资源小时。',
 '本轮结束后停止。不自动改候选网格、加入长期欠账规则或扩大负载。']
 with (OUT/'REPORT.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
 print('FINAL',complete,{n:v['summary']['mean_jct'] for n,v in results.items()})
if __name__=='__main__':main()
