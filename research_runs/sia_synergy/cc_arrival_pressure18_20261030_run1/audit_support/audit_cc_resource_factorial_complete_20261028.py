"""Independent30-cell audit and resource-factor attribution; no simulation."""
import collections,gzip,hashlib,itertools,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_joint_resource_factorial_20261027_run1'
DEST=OUT/'completed_analysis_20261028_run1'
DEST.mkdir(exist_ok=False)
sys.path.insert(0,str(OUT/'source/sia-simulator'))
import numpy as np
from joint_prototype.joint_profile import load_synergy_template,CPU_COORDS,RAM_COORDS
REG={r['key']:r for r in json.loads((OUT/'source/inputs/registry.json').read_text())};TEMPLATES={}
def save(n,v):
 with (DEST/n).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def rho(j,c,cp,mm,exponent):
 r=REG[j.rsplit('-',1)[0]+'/'+c];t=r['template']
 if t not in TEMPLATES:TEMPLATES[t]=load_synergy_template(OUT/'source/inputs/models'/t).matrix
 def f(c,m):return float(np.interp(c,CPU_COORDS,[np.interp(m,RAM_COORDS,row) for row in TEMPLATES[t]]))
 return (f(cp/r['phi_c'],mm/r['phi_m'])/f(3/r['phi_c'],62.5/r['phi_m']))**exponent
def stream(p):
 with gzip.open(p,'rt',encoding='utf-8') as f:
  for line in f:yield json.loads(line)
def geom(a):
 if not a:return None
 return(a.get('config_id') or a['config']['config_id'],a['cpu_per_gpu'],a['ram_gb_per_gpu'],tuple(sorted(a['placement'].items())))
def audit(folder,p):
 s=json.loads((folder/'summary.json').read_text());cpu,ram=p['capacities'][s['scenario']];exponent=p['response_exponents'][s['response']]
 checks=collections.Counter();jobs=collections.defaultdict(lambda:collections.Counter());last=None;previous={};frozen={};queue=collections.Counter();nodehours=collections.defaultdict(lambda:collections.Counter());statuses=collections.Counter();clip_seconds=collections.Counter()
 for r in stream(folder/'rounds.jsonl.gz'):
  queue['waiting_job_seconds']+=sum(e['waiting_seconds'] for e in r['execution'])
  queue['active_job_seconds']+=sum(e['observed_active_seconds'] for e in r['execution'])
  for e in r['execution']:
   name=e['job_id'];j=jobs[name]
   for k,fld in [('wait','waiting_seconds'),('checkpoint','checkpoint_delay_consumed'),('training','training_seconds')]:j[k]+=e[fld]
   assert e['progress_delta']>=-1e-9
   assert math.isclose(e['observed_active_seconds'],e['waiting_seconds']+e['checkpoint_delay_consumed']+e['training_seconds'],abs_tol=1e-6)
   if e['checkpoint_delay_consumed']>=e['observed_active_seconds']>0:assert e['progress_delta']==0;checks['checkpoint_no_progress']+=1
   if e['allocated_seconds']>0:
    a=e['resource_allocation'];assert geom(a)==geom(previous[name]);checks['atomic_cross_round']+=1
    expected=rho(name,a['config_id'],a['cpu_per_gpu'],a['ram_gb_per_gpu'],exponent);assert math.isclose(expected,e['rho'],rel_tol=1e-12)
    ref=REG[name.rsplit('-',1)[0]+'/'+a['config_id']]
    for tag,cp0,mm0 in [('numerator',a['cpu_per_gpu'],a['ram_gb_per_gpu']),('denominator',3.,62.5)]:
     clip_seconds[tag+'_cpu']+=e['training_seconds']*int(not 1<=cp0/ref['phi_c']<=24)
     clip_seconds[tag+'_ram']+=e['training_seconds']*int(not 20.83<=mm0/ref['phi_m']<=250)
    dt=e['allocated_seconds']/3600;j['gpu_hours']+=a['gpus']*dt
    for u in a['node_usage']:
     j['cpu_core_hours']+=u['cpu']*dt;j['ram_gb_hours']+=u['ram_gb']*dt
     for k,fld in [('gpu','gpus'),('cpu','cpu'),('ram','ram_gb')]:nodehours[u['node_id']][k]+=u[fld]*dt
    for z in e['training_segments']:
     assert math.isclose(z['effective_rate'],z['raw_rate_before_rho']*expected,rel_tol=1e-12)
     assert math.isclose(z['progress_delta'],z['effective_rate']*z['seconds']+z['rounding_progress_residual'],abs_tol=1e-7);checks['rho_once']+=1
   j['arrival']=e['arrival_time'];j['completion']=e['completion_time'];j['restarts']=r['job_end_state'][name]['restarts'];j['wasted_gpu_seconds']=r['job_end_state'][name]['wasted_gpu_seconds']
  d=r['decision_for_next_interval'];a=d['result']['audit'];sel=d['result']['selected']
  if d['round']!=last:
   last=d['round'];specs={x['job_id']:x for x in d['probe']['specs']};usage=collections.defaultdict(lambda:[0.,0.,0.])
   statuses[d['result']['status']]+=1
   if s['method'].startswith('SIA'):
    expected={j for j,v in d['original_sia_allocations'].items() if v and v[0] is not None and v[1]}
    assert set(sel)==expected
    for j,c in sel.items():
     intent=d['original_sia_allocations'][j];assert intent[0]==c['config']['gpu_type']
     counts=collections.Counter(intent[1]);assert {f'{intent[0]}:{n}':v for n,v in counts.items()}==c['placement'];checks['sia_fixed_intent']+=1
   for j,c in sel.items():
    cp,mm=c['cpu_per_gpu'],c['ram_gb_per_gpu'];expected=rho(j,c['config']['config_id'],cp,mm,exponent);assert math.isclose(expected,c['rho'],rel_tol=1e-12)
    assert c['config'] in specs[j]['configs'] and c['query_round']==specs[j]['query_round']
    assert sum(c['placement'].values())==c['config']['gpus'] and len(c['placement'])==c['config']['num_nodes']
    assert all(n in {'aws:0','dgx-ext:0'} and n.rsplit(':',1)[0]==c['config']['gpu_type'] for n in c['placement'])
    ref=REG[j.rsplit('-',1)[0]+'/'+c['config']['config_id']]
    assert c['config']['cpu_demand_scale']==ref['phi_c'] and c['config']['memory_demand_scale']==ref['phi_m']
    assert cp>=1 and mm>=20.83
    if s['method'] not in ['SIA_P','CC_P']:
     allowed_cpu={1.,2.,3.,4.,6.,8.,9.,12.,24.,cpu/2}
     allowed_ram={20.83,62.5,125.,187.5,250.,ram/2}
     assert cp in allowed_cpu and mm in allowed_ram
    if exponent==0:assert c['rho']==1.
    if s['method'] in ['SIA_P','CC_P']:assert(cp,mm)==(cpu/2,ram/2)
    if s['method']=='CC_CPU':assert mm==ram/2
    if s['method']=='CC_MEM':assert cp==cpu/2
    if j in frozen:assert(c['config']['config_id'],tuple(sorted(c['placement'].items())))==frozen[j]
    q=specs[j]
    if q['checkpoint_locked']:
     assert geom(c)==(q['current_config_id'],q['current_cpu_per_gpu'],q['current_ram_gb_per_gpu'],tuple(tuple(x) for x in q['current_placement']));checks['full_lock']+=1
    for n,g in c['placement'].items():
     for k,x in enumerate([g,g*cp,g*mm]):usage[n][k]+=x
   for v in usage.values():assert all(x<=y+1e-7 for x,y in zip(v,[2,cpu,ram]));checks['capacity']+=1
   for x in a.get('tail_recoveries',[]):
    j=x['job'];assert len(specs)==1 and not a['ready'] and not specs[j]['checkpoint_locked'] and j not in frozen
    assert x['eta_after']<x['eta_before'] and 0<=x['extra_budget']<=x['actual_startup']<=250;checks['tail_recovery']+=1
    frozen[j]=(sel[j]['config']['config_id'],tuple(sorted(sel[j]['placement'].items())))
   for j,v in a.get('service_delay_ledgers',{}).items():assert v['delay_charged']<=250+a.get('tail_grants',{}).get(j,{}).get('extra_budget',0.)+1e-9
  previous=sel
 vals=[]
 tau=6000*60-2400
 restricted=[]
 for name,j in jobs.items():
  for k in ['wait','checkpoint','training','gpu_hours','cpu_core_hours','ram_gb_hours','restarts','wasted_gpu_seconds']:assert math.isclose(j[k],s['jobs'][name][k],abs_tol=1e-6),(name,k)
  if j['completion'] is not None:
   j['jct']=j['completion']-j['arrival'];assert math.isclose(j['jct'],j['wait']+j['checkpoint']+j['training'],abs_tol=1e-6);vals.append(j['jct'])
   restricted.append(min(j['jct'],tau))
  else:
   assert s['simulated_seconds']-j['arrival']>=tau
   restricted.append(tau)
 if len(vals)==20:
  for k,x in [('mean_jct',np.mean(vals)),('median_jct',np.median(vals)),('p95_jct',np.percentile(vals,95)),('p99_jct',np.percentile(vals,99))]:assert math.isclose(s[k],float(x),abs_tol=1e-6)
 assert dict(statuses)==s['solver_status']
 return dict(valid=True,summary=s,checks=dict(checks),common_followup_rmst_tau=tau,common_followup_sample_rmst=float(np.mean(restricted)),queue_integrals=dict(queue),clipped_productive_job_seconds=dict(clip_seconds),node_resource_hours={n:dict(v) for n,v in nodehours.items()},jobs={j:dict(v) for j,v in jobs.items()})
def neutral_equal(p,q):
 for a,b in itertools.zip_longest(stream(p),stream(q)):
  if a is None or b is None:return False
  def signature(r):return[(e['job_id'],e['progress_after'],e['completion_time'],e['cluster'],e['placement']) for e in r['execution']]
  if signature(a)!=signature(b):return False
 return True
def main():
 p=json.loads((OUT/'preregistration.json').read_text());results={};comparisons={};interactions={};coverage=[]
 for scenario,response,method in p['cells']:
  name=f'{scenario}_{response}_{method}';path=OUT/'cells'/name;exists=(path/'summary.json').exists()
  coverage.append(dict(cell=name,state='AVAILABLE' if exists else 'NOT_COMPLETE'))
  if exists:
   results[name]=audit(path,p);print('AUDITED',name,flush=True)
 for scenario in p['capacities']:
  ns={m:f'{scenario}_medium_{m}' for m in p['methods']}
  if not all(n in results and results[n]['summary']['completed']==20 for n in ns.values()):continue
  for left,right in [('CC_P','SIA_P'),('CC_FULL','SIA_P'),('CC_FULL','SIA_AUX'),('CC_CPU','CC_P'),('CC_MEM','CC_P'),('CC_FULL','CC_CPU'),('CC_FULL','CC_MEM'),('CC_FULL','CC_P')]:
   x,y=results[ns[left]]['summary'],results[ns[right]]['summary'];d={j:x['jobs'][j]['jct']-v['jct'] for j,v in y['jobs'].items()}
   comparisons[scenario+'/'+left+'_vs_'+right]=dict(improvement=1-x['mean_jct']/y['mean_jct'],mean_delta=x['mean_jct']-y['mean_jct'],improved=sum(v<0 for v in d.values()),worsened=sum(v>0 for v in d.values()),per_job_delta=d,
    time_delta={k:x['totals'][k]-y['totals'][k] for k in ['wait','checkpoint','training']},p95_change=x['p95_jct']/y['p95_jct']-1,p99_change=x['p99_jct']/y['p99_jct']-1,makespan_change=x['makespan']/y['makespan']-1)
  means={m:results[n]['summary']['mean_jct'] for m,n in ns.items()}
  interactions[scenario]=means['CC_FULL']-means['CC_CPU']-means['CC_MEM']+means['CC_P']
 neutral={}
 base=OUT/'cells/R11_neutral_CC_P/rounds.jsonl.gz'
 if base.exists():
  for m in ['CC_CPU','CC_MEM','CC_FULL']:
   q=OUT/f'cells/R11_neutral_{m}/rounds.jsonl.gz'
   if q.exists() and(OUT/f'cells/R11_neutral_{m}/summary.json').exists():neutral[m]=neutral_equal(base,q)
 if neutral:assert all(neutral.values())
 save('independent_audit_and_comparison.json',dict(status='THIRTY_AUDITED' if len(results)==30 else 'PARTIAL',results=results,comparisons=comparisons,quota_interaction_seconds=interactions,neutral_cc_trajectories_equal=neutral,coverage=coverage))
 lines=['# CC Joint CPU 内存六方法压力实验','',f'运行摘要可用{len(results)}/30，全部已读取格均通过独立原始日志核验。','',
 'SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。','',
 '## 冻结语义与机会预检','',
 '保留CC-TAIL目标、准入/持续服务/一次尾恢复、GPU候选、原CPU/RAM网格、R0/E-legacy响应和原20作业。只改变四档预注册CPU/RAM容量和四CC配额自由度；Sia基线不加入L2。SIA-AUX固定Sia GPU和放置意图，用原ECT辅助优化，不是官方Synergy-TUNE。',
 'R00每节点12CPU/250GB，R10为6CPU/250GB，R01为12CPU/187.5GB，R11为6CPU/187.5GB；GPU均为AWS/DGX各一节点各2GPU。24格Medium alpha=.5，R11另6格Neutral；所有60秒/6000轮、seed_profiles(1,8)、同训练工作量。',
 '20作业为既有18个CIFAR与2个DeepSpeech真实Saturn记录及受控到达，GPU需求/batch/完整训练工作量保留；CIFAR借用alexnet_1.py、DeepSpeech借用deepspeech_1.py。矩阵单元来自Synergy，应用关联、跨硬件phi/R0、归一化与乘法执行均为合成假设，敏感性标签不是Saturn实测。',
 'G0按结构取各GPU类型首个非checkpoint且CF/DS各1GPU、已画像的历史状态。计入双方损失，以声明总工作量归一推进；CPU压力和RAM压力均存在原网格支持域内正净推进。并不保证剩余ETA之和或JCT改善；部分见证剩余ETA之和增加。',
 'R01 DGX固定CPU6/6时，RAM125/62.5对比93.75/93.75的预测净归一推进增5.67%，计入DS减速；R11对应内存增量见证4.35%。已知连续配额103.28/84.22的无损donor机会不在网格，但不是全部关键机会只能网格外，故按既定gate继续，不插入该点。全域裁剪未删配置，计数见full_domain_coverage.json及各summary。','',
 '## 结果','', '|场景|响应|方法|完成|平均JCT|P95|P99|Makespan|Restart|','|---|---|---|---:|---:|---:|---:|---:|---:|']
 for n,x in results.items():
  s=x['summary'];lines.append(f"|{s['scenario']}|{s['response']}|{s['method']}|{s['completed']}/20|{s['mean_jct']}|{s['p95_jct']}|{s['p99_jct']}|{s['makespan']}|{s['totals']['restarts']}|")
 lines+=['','## 配额与整体对照','', '|场景与对照|平均JCT改善率|改善/退化任务|P95变化|P99变化|Makespan变化|','|---|---:|---:|---:|---:|---:|']
 for n,x in comparisons.items():lines.append(f"|{n}|{100*x['improvement']:.2f}%|{x['improved']}/{x['worsened']}|{100*x['p95_change']:.2f}%|{100*x['p99_change']:.2f}%|{100*x['makespan_change']:.2f}%|")
 lines+=['','正改善率表示均值更低；尾部变化正数表示恶化。配额交互I=J_FULL−J_CPU−J_MEM+J_P，负值表示相对于可加改善的额外互补，正值表示重叠或抵消。各场景I（秒）：'+json.dumps(interactions),'',
 '## 归因与边界','',
 'CC-P相对Sia同时包含目标、准入、GPU配置与放置差异，不能把全部收益归CPU/RAM。四CC消融才用于辅助配额自由度的闭环归因，其中还包含随后的GPU/队列反馈。SIA-AUX目标不同且无L2，是外部串行基线，不是严格四因子单元。',
 '同一20作业在四资源环境重复使用，不是4个独立workload或80次独立样本；不做虚假的显著性推断。P95/P99只有20样本，仅描述。完整均值仅对20/20完成的配对比较；删失保留完成率、censor_time与RMST。','',
 '## 正确性与复现','',
 '224既有测试和5六方法适配测试通过；后者为3作业5轮短链，Neutral四CC一致、同状态候选P⊆CPU/MEM⊆FULL、比例契约及容量验证。初稿适配模块缺失红测和首轮过慢测试中断保留；改用3作业测试夹具，不改变正式20作业。',
 '内核父279文件逐字节保留；新增resource_study.py只限制资源自由度，没有新增CPU/RAM点。source/是完整冻结版本，preregistration.json保存30格顺序、输入/源码/脚本hash和依赖。独立审计重算完成事件、互斥时间、资源小时、容量、Sia意图、完整锁、一次尾恢复和raw×rho，并核验Neutral四CC进度/GPU/完成轨迹。',
 '原始压缩轮次、执行日志、完成/删失和逐作业summary均在cells；independent_audit_and_comparison.json给逐任务差值、资源/时间分解、交互和覆盖。达到安全门槛停止时保留前缀，不覆盖部分结果。未扩大160作业、未调参、未推送。']
 with (DEST/'REPORT.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
 print('FINAL',len(results),neutral,interactions)
if __name__=='__main__':main()
