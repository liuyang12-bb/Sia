"""Independent raw-event validation and final report; no training."""
import collections,gzip,hashlib,json,math,sys,time,platform,importlib.metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_joint_queue_ram_tail_20261027_run3'
OLD=ROOT/'research_runs/sia_synergy/cc_joint_sustained_cpu_20job_20261027_run1'
sys.path.insert(0,str(OUT/'source/sia-simulator'))
import numpy as np
from joint_prototype.joint_profile import load_synergy_template,CPU_COORDS,RAM_COORDS
registry={r['key']:r for r in json.loads((OUT/'source/inputs/registry.json').read_text())};matrices={}
def save(name,data):
 with (OUT/name).open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2)
def rho(job,cfg,cpu,ram):
 app=job.rsplit('-',1)[0];r=registry[f'{app}/{cfg}'];t=r['template']
 if t not in matrices:matrices[t]=load_synergy_template(OUT/'source/inputs/models'/t).matrix
 def interp(c,m):return float(np.interp(c,CPU_COORDS,[np.interp(m,RAM_COORDS,row) for row in matrices[t]]))
 return (interp(cpu/r['phi_c'],ram/r['phi_m'])/interp(3/r['phi_c'],62.5/r['phi_m']))**.5
def audit(method):
 path=OUT/'cells'/method;s=json.loads((path/'summary.json').read_text());jobs=collections.defaultdict(lambda:collections.Counter());checks=collections.Counter();last=None
 with gzip.open(path/'rounds.jsonl.gz','rt',encoding='utf-8') as f:
  for line in f:
   r=json.loads(line);d=r['decision_for_next_interval'];sel=d['result']['selected'];specs={x['job_id']:x for x in d['probe']['specs']}
   if d['round']!=last:
    last=d['round'];usage=collections.defaultdict(lambda:[0.,0.,0.])
    for name,x in sel.items():
     cfg=x['config'];cpu,ram=x['cpu_per_gpu'],x['ram_gb_per_gpu'];rr=rho(name,cfg['config_id'],cpu,ram);assert math.isclose(rr,x['rho'],rel_tol=1e-12)
     if method in ['SIA_P','JOINT_CC_P']:assert(cpu,ram)==(6,93.75)
     if method=='JOINT_CC_CPU':assert ram==93.75
     q=specs[name]
     if q['checkpoint_locked']:
      assert cfg['config_id']==q['current_config_id'] and x['placement']==dict(q['current_placement'])
      assert(cpu,ram)==(q['current_cpu_per_gpu'],q['current_ram_gb_per_gpu']);checks['full_lock']+=1
     for n,g in x['placement'].items():
      for i,v in enumerate([g,g*cpu,g*ram]):usage[n][i]+=v
    for v in usage.values():assert all(x<=y+1e-7 for x,y in zip(v,[2,12,187.5]));checks['node_capacity']+=1
   for e in r['execution']:
    name=e['job_id'];j=jobs[name]
    for k,fld in [('wait','waiting_seconds'),('checkpoint','checkpoint_delay_consumed'),('training','training_seconds')]:j[k]+=e[fld]
    assert e['progress_delta']>=-1e-9
    assert math.isclose(e['waiting_seconds']+e['checkpoint_delay_consumed']+e['training_seconds'],e['observed_active_seconds'],abs_tol=1e-6)
    if e['checkpoint_delay_consumed']>=e['observed_active_seconds']>0:assert e['progress_delta']==0;checks['checkpoint_no_training']+=1
    a=e['resource_allocation']
    if a:
     rr=rho(name,a['config_id'],a['cpu_per_gpu'],a['ram_gb_per_gpu']);assert math.isclose(rr,e['rho'],rel_tol=1e-12)
     dt=e['allocated_seconds']/3600;j['gpu_hours']+=dt*a['gpus'];j['cpu_core_hours']+=dt*sum(x['cpu'] for x in a['node_usage']);j['ram_gb_hours']+=dt*sum(x['ram_gb'] for x in a['node_usage'])
     for z in e['training_segments']:
      assert math.isclose(z['effective_rate'],z['raw_rate_before_rho']*rr,rel_tol=1e-12)
      assert math.isclose(z['progress_delta'],z['effective_rate']*z['seconds']+z['rounding_progress_residual'],abs_tol=1e-7);checks['raw_times_rho_once']+=1
    j['arrival']=e['arrival_time'];j['completion']=e['completion_time']
 vals=[]
 for name,j in jobs.items():
  for k in ['wait','checkpoint','training','gpu_hours','cpu_core_hours','ram_gb_hours']:assert math.isclose(j[k],s['jobs'][name][k],abs_tol=1e-6), (name,k)
  if j['completion'] is not None:
   j['jct']=j['completion']-j['arrival'];vals.append(j['jct']);assert math.isclose(j['jct'],j['wait']+j['checkpoint']+j['training'],abs_tol=1e-6)
 if len(vals)==20:assert math.isclose(float(np.mean(vals)),s['mean_jct'],abs_tol=1e-7)
 return dict(valid=True,checks=dict(checks),completed=len(vals),summary=s,recomputed_jobs={k:dict(v) for k,v in jobs.items()})
def main():
 p=json.loads((OUT/'preregistration.json').read_text());results={}
 save('analysis_environment.json',dict(python=sys.version,platform=platform.platform(),packages={n:importlib.metadata.version(n) for n in ['numpy','scipy','cvxpy','cylp','pandas','psutil']}))
 for method in p['methods']:
  if (OUT/'cells'/method/'summary.json').exists():results[method]=audit(method)
 complete=len(results)==5 and all(v['completed']==20 for v in results.values())
 comparisons={}
 if complete:
  for a,b in [('JOINT_CC','SIA_P'),('JOINT_CC','SIA_ECT_AUX'),('JOINT_CC','JOINT_CC_P'),('JOINT_CC','JOINT_CC_CPU'),('JOINT_CC_CPU','JOINT_CC_P')]:
   x,y=results[a]['summary'],results[b]['summary'];ds={j:x['jobs'][j]['jct']-v['jct'] for j,v in y['jobs'].items()}
   comparisons[a+'_vs_'+b]=dict(mean_delta=x['mean_jct']-y['mean_jct'],improvement=1-x['mean_jct']/y['mean_jct'],improved=sum(v<0 for v in ds.values()),worsened=sum(v>0 for v in ds.values()),per_job_delta=ds,
    p95_change=x['p95_jct']/y['p95_jct']-1,p99_change=x['p99_jct']/y['p99_jct']-1,makespan_change=x['makespan']/y['makespan']-1)
 save('validation_and_comparison.json',dict(status='FIVE_COMPLETE_VALID' if complete else 'PARTIAL_NOT_PAIRED',results=results,comparisons=comparisons))
 lines=['# CC Joint 排队收益与内存压力实验','', '本轮不修改调度算法。旧20任务轨迹只读分析；新场景仅将每节点RAM从250降为187.5GB，CPU/GPU/任务/响应/观察窗均保持不变。','',
 'SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。','',
 '## 排队优势的具体来源','',
 '原Medium中累计JCT减少67464任务秒：首次分配前等待减少73560秒、后续等待减少2760秒、checkpoint减少575秒、训练任务墙钟增加9431秒。96.38%的等待节省是初始排队，不是抢占后恢复变少。',
 'CF3首次分配由7020提前到1080秒；其JCT从8540降到1604秒。CF20首次分配从10740提前到6420秒。CC-P无需CPU自由度已经显著减少等待。',
 '原始运行以周期边界并发任务数计，Sia4任务并发600秒、CC-P8520秒、CC7620秒。Sia在t840存在两个DeepSpeech各占2GPU、CIFAR等待的完整见证；CC中未观察到该结构。区间内完成会改变精确并发时长，因此这些是边界占用分类而非精确连续并发积分。',
 '解释：完成导向的配置选择使更多作业以较小GPU配置共驻，连续服务减少抖动，较短任务完成并释放资源。FIFO与可选准入也改变任务集合。整体JCT变化仍不能唯一分解为目标、准入或放置的各自因果贡献；不能称Sia通用缺陷或未经文献核验的首次创新。可提炼为异构联合调度中的完成释放与服务连续性机制。','',
 '## 长尾不是主要在排队','',
 'DS8的JCT由6963增至12916秒，其中等待增加60、checkpoint增加100、训练增加5793秒；它t120已获服务。Sia路径大量使用两GPU和DGX，CC前期长期使用单GPU并让出CPU。',
 't9960只剩DS8：当前AWS2预计剩余2627.17秒；同状态DGX2候选含25秒迁移预计1324.72秒，合法资源空闲，但生命周期启动预算250秒已耗尽，候选被SERVICE_DELAY_BUDGET挡住。事件窗口也限制非事件轮调整。此为局部预测机会，不是已执行反事实JCT。完整见证保存在tail_budget_witness.json。',
 '最终Sia由CF22在11907秒完成，CC由DS8在12916秒完成；CC最高两个JCT为DS8和DS9，解释P95/P99/makespan与均值冲突。','',
 '## 预注册的内存机会与代价','',
 '保持原应用映射与R0。187.5GB是两个现有网格点125+62.5之和，不根据JCT选择。DGX同CPU/GPU下，CF RAM93.75→125速度增6.56%，DS93.75→62.5速度降15.32%；均在源矩阵支持域。因此不存在已证明的无损donor，更不能预设JCT获胜。',
 '额外JOINT_CC_CPU只允许CPU优化、RAM固定93.75；完整CC对CPU-only差值是本场景内存自由度的闭环增量，含后续配置/队列反馈。不是固定GPU直接内存加速率。',
 '另有重要候选覆盖缺口：原矩阵RAM平台点62.5乘固定phi_m，给出DGX的CF103.2809GB、DS84.2191GB，总187.5GB；CPU各6。该物理合法组合让CF增速6.56%、DS不减速，分子/分母均在支持域；但两个配额均不在当前网格内。本轮没有将它加入运行，也没有按结果改网格。memory_candidate_gap.json给出全部查询。由此不能把内存失败全归因场景：候选表达能力也是明确限制。','',
 '## 新场景结果','',f'状态：{"五格20/20完成且独立审计通过" if complete else "未形成完整配对，不能报告五策略总体比较"}。','',
 '|策略|完成|平均JCT|Median|P95|P99|Makespan|Restart|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for m,v in results.items():
  s=v['summary'];lines.append(f"|{m}|{s['completed']}/20|{s['mean_jct']}|{s['median_jct']}|{s['p95_jct']}|{s['p99_jct']}|{s['makespan']}|{s['totals']['restarts']}|")
 for name,v in comparisons.items():lines.append(f"\n{name}：平均JCT改善{100*v['improvement']:.2f}%，改善/退化作业{v['improved']}/{v['worsened']}；P95/P99/Makespan变化{100*v['p95_change']:.2f}%/{100*v['p99_change']:.2f}%/{100*v['makespan_change']:.2f}%（正为恶化）。")
 lines+=['','|策略|等待任务秒|训练任务秒|Checkpoint任务秒|GPU-h|CPU单位-h|RAM-GB-h|非比例RAM列数|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for m,v in results.items():
  s=v['summary'];t=s['totals'];lines.append(f"|{m}|{t['wait']}|{t['training']}|{t['checkpoint']}|{t['gpu_hours']:.3f}|{t['cpu_core_hours']:.3f}|{t['ram_gb_hours']:.3f}|{s['checks'].get('nonproportional_ram',0)}|")
 if complete:
  lines+=['','### 逐作业 JCT','', '|Job|'+'|'.join(p['methods'])+'|','|---|'+'---:|'*5]
  for j in results['SIA_P']['summary']['jobs']:
   lines.append('|'+j+'|'+'|'.join(str(results[m]['summary']['jobs'][j]['jct']) for m in p['methods'])+'|')
 lines+=['','## 保留平均优势并缓解尾部的最小后续设计','',
 '优先验证收尾恢复，而不是继续调倒数评分权重：仅在队列已空、只剩一个未完成作业、无真实checkpoint锁、存在已观测且容量合法的更快配置时，评估含真实迁移成本的严格净完成时间改善。它不改变前面20作业的竞争/准入路径，但必须先解决生命周期预算的活性证明，不能直接放松预算。可用一次性有界收尾许可或单调剩余完成势能规则；预测误差须做保守门槛和失败回退。',
 '第二优先是累计让出方进度欠账保护，而不是每轮禁止任何减速：在相同已观测GPU/batch下记录比例执行参照与实际执行的归一工作差，达到预注册界限时优先偿还。保留短时有利交换，约束长期牺牲。此方案改变调度语义，必须独立新版本、红绿测试和原CC配对，不能由本轮结果临时选阈值。',
 '先做1–3任务预算耗尽且快GPU空闲的收尾测试、长期donor测试和checkpoint/原子回退测试，再跑原20任务共同冻结观察窗。成功门槛应同时要求平均JCT不显著丢失、DS8/DS9退化下降与资源/锁审计通过；不保证一定达到。','',
 '## 验证与限制','',
 '212既有测试通过（日志在run2/existing_tests.log），7新增离线验收通过（offline_tests_final.log）。每格独立进程、2GiB/5GiB门槛、60秒/6000轮；无重跑挑选、无算法修改。新增结果的完成事件、互斥时间、资源小时、逐节点容量、完整锁和raw×rho由独立读日志路径核验，详情validation_and_comparison.json。',
 'run1导入失败和run2无损donor假设被离线预检否定的现场保留，均未启动训练。当前新场景只有一组受控负载，不是20次独立重复；完整候选仍有E-legacy裁剪，计数在summary.checks，不能冒称模型真实性验证。辅助配额调整额外开销未建模。',
 '原算法源码快照在source，参数/全文件SHA在preregistration.json；原始轮次在cells/<method>/rounds.jsonl.gz，逐作业数据及资源小时在summary.json和validation_and_comparison.json。']
 with (OUT/'REPORT.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
 save('delivery_hashes.json',{p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest().upper() for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts})
 print('FINAL',complete,{m:v['summary']['mean_jct'] for m,v in results.items()})
if __name__=='__main__':main()
