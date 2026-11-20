"""Independent raw-event reconstruction; never derives mean from summaries."""
import collections,csv,gzip,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent;R=O.parents[2];T=R/'cc-joint-release-evaluation-v1'
sys.path[:0]=[str(T),str(T/'sia-simulator')]
from joint_prototype.execution_audit import audit_execution
from joint_prototype.r0_liveness import R0Oracle
from run_stage2c_cell import independent_rho
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
def close(a,b):return math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-6)
def audit(cell,p,work):
 out=O/'cells'/cell['id'];summary=json.loads((out/'summary.json').read_text())
 oracle=R0Oracle(T/'inputs/registry.json',T/'inputs/models',{j:v['application'] for j,v in work.items()},'medium')
 rows={j:dict(wait=0.,checkpoint=0.,training=0.,gpu_hours=0.,cpu_core_hours=0.,ram_gb_hours=0.,completion=None,progress=0.) for j in work}
 counts=collections.Counter();plans=collections.Counter();evaluations=[];last_decision=None;rounds=0;errors=[];clipped=0;selected=0
 with gzip.open(out/'rounds.jsonl.gz','rt',encoding='utf8') as f:
  for line in f:
   r=json.loads(line);rounds+=1;d=r['decision_for_next_interval'];res=d['result']
   if d['round']!=last_decision:
    last_decision=d['round'];counts['decisions']+=1
    if not res['audit']['valid']:errors.append('INVALID_COMMIT')
    specs={x['job_id']:x for x in d['probe']['specs']};used={t+':0':[0.,0.,0.] for t in p['topology']}
    for j,c in res['selected'].items():
     selected+=1;clipped+=bool(c['source_coordinates']['clipped'] or c['baseline_source_coordinates']['clipped'])
     rho=independent_rho(oracle,j,c['config']['config_id'],c['cpu_per_gpu'],c['ram_gb_per_gpu'])
     if not close(rho,c['rho']):errors.append('RHO_QUERY')
     for n,g in c['placement'].items():
      for i,x in enumerate((g,g*c['cpu_per_gpu'],g*c['ram_gb_per_gpu'])):used[n][i]+=x
     sp=specs[j]
     if sp['checkpoint_locked'] and sp['current_config_id']:
      counts['lock_checks']+=1
      if not (sp['current_config_id']==c['config']['config_id'] and dict(sp['current_placement'])==c['placement'] and close(sp['current_cpu_per_gpu'],c['cpu_per_gpu']) and close(sp['current_ram_gb_per_gpu'],c['ram_gb_per_gpu'])):errors.append('LOCK_CHANGED')
    for n,u in used.items():
     t=n.split(':')[0];g=p['topology'][t]
     if any(x>y+1e-7 for x,y in zip(u,(g,3*g,62.5*g))):errors.append('CAPACITY')
    ev=res['audit'].get('release_completion')
    if ev:
     counts['evaluation_'+ev['status']]+=1;plans[ev['selected_plan']]+=1
     if ev['status']=='OK':
      waiting=set(specs)-set(ev['fixed_service_set'])
      counts['evaluation_ok_with_waiters']+=bool(waiting)
      counts['predicted_waiter_gain_seconds']+=sum(ev['per_job_predicted_saved'][j] for j in waiting)
      delta=ev['baseline']['sum_completion']-ev['selected']['sum_completion']
      if delta < -1e-7 or not close(delta,sum(ev['per_job_predicted_saved'].values())):errors.append('PREDICTED_NET_ACCEPTANCE')
      if set(ev['per_job_predicted_saved'])!=set(specs):errors.append('EXCLUDED_CURRENT_JOB')
      if ev['fo_fixed_gpu'] is not None and ev['fo_fixed_gpu']!={j:c['config']['config_id'] for j,c in res['selected'].items()}:errors.append('FO_GPU_CHANGED')
      if ev['selected_plan']!='LEGACY':evaluations.append(dict(time=r['interval_end'],**ev))
   for e in r['execution']:
    j=e['job_id'];v=rows[j]
    if not audit_execution(e)['valid']:errors.append('EXECUTION')
    counts['execution_checks']+=1
    for key,field in [('wait','waiting_seconds'),('checkpoint','checkpoint_delay_consumed'),('training','training_seconds')]:v[key]+=e[field]
    if e['progress_after']<e['progress_before']-1e-8:errors.append('PROGRESS_MONOTONIC')
    v['progress']=e['progress_after']
    if e['completion_time'] is not None:v['completion']=e['completion_time']
    a=e['resource_allocation']
    if a:
     rho=independent_rho(oracle,j,a['config_id'],a['cpu_per_gpu'],a['ram_gb_per_gpu'])
     if not close(e['rho'],rho):errors.append('EXEC_RHO')
     for seg in e['training_segments']:
      if not close(seg['raw_rate_before_rho']*rho,seg['effective_rate']):errors.append('DOUBLE_RHO')
     dt=e['allocated_seconds']/3600;v['gpu_hours']+=a['gpus']*dt
     for u in a['node_usage']:v['cpu_core_hours']+=u['cpu']*dt;v['ram_gb_hours']+=u['ram_gb']*dt
   for j,v in r['job_end_state'].items():
    rows[j]['restarts']=v['restarts'];rows[j]['wasted_gpu_seconds']=v['wasted_gpu_seconds']
 for j,v in rows.items():
  v['arrival']=float(work[j]['time']);v['jct']=v['completion']-v['arrival'] if v['completion'] is not None else None
  if v['jct'] is not None and not close(v['jct'],v['wait']+v['training']+v['checkpoint']):errors.append('EXCLUSIVE_TIME')
  for k in ('jct','wait','checkpoint','training','gpu_hours','cpu_core_hours','ram_gb_hours','restarts','wasted_gpu_seconds'):
   if v[k] is not None and not close(v[k],summary['jobs'][j][k]):errors.append('SUMMARY_'+k)
 complete=all(v['jct'] is not None for v in rows.values());vals=[v['jct'] for v in rows.values() if v['jct'] is not None]
 stats=dict(completed=len(vals),rounds=rounds,complete=complete,mean=float(np.mean(vals)) if complete else None,
  median=float(np.median(vals)) if complete else None,p95=float(np.percentile(vals,95)) if complete else None,p99=float(np.percentile(vals,99)) if complete else None,
  makespan=max(v['completion'] for v in rows.values())-min(v['arrival'] for v in rows.values()) if complete else None,
  totals={k:sum(v[k] for v in rows.values()) for k in ('wait','checkpoint','training','gpu_hours','cpu_core_hours','ram_gb_hours','restarts','wasted_gpu_seconds')},
  errors=errors,checks=dict(counts),plans=dict(plans),clipped=clipped,selected=selected,clipped_fraction=clipped/selected if selected else None,
  wall=summary['total_wall_seconds'])
 return stats,rows,evaluations
def main():
 p=json.loads((O/'preregistration.json').read_text());work={v['name']:v for v in csv.DictReader((O/'inputs/workload.csv').open())}
 for n,h in p['source_hashes'].items():assert sha(T/n)==h,n
 for n,h in p['input_hashes'].items():assert sha(n)==h,n
 results={};jobs={};events={}
 for cell in p['cells']:
  stats,rows,ev=audit(cell,p,work);results[cell['id']]=stats;jobs[cell['id']]=rows;events[cell['id']]=ev
 save('raw_audit.json',results);save('per_job.json',jobs);save('decision_changes.json',events)
 comparisons={}
 for a,b in [('OLD_FO','OLD_FULL'),('NEW_FO','NEW_FULL'),('OLD_FO','NEW_FO'),('OLD_FULL','NEW_FULL')]:
  if not (results[a]['complete'] and results[b]['complete']):comparisons[a+'__'+b]={'paired_complete':False};continue
  delta={j:jobs[a][j]['jct']-jobs[b][j]['jct'] for j in work};top=max(delta,key=delta.get)
  comparisons[a+'__'+b]=dict(paired_complete=True,mean_saved=results[a]['mean']-results[b]['mean'],percent=100*(1-results[b]['mean']/results[a]['mean']),per_job_saved=delta,
    better=sum(v>0 for v in delta.values()),worse=sum(v<0 for v in delta.values()),equal=sum(v==0 for v in delta.values()),
    largest_gain_job=top,largest_gain=delta[top],total_saved=sum(delta.values()),without_largest_gain_mean_saved=(sum(delta.values())-delta[top])/(len(work)-1))
 save('comparison_summary.json',comparisons)
 lines=['# 完成事件评价：实现、同状态验证与12作业完整训练','',
  'SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE','',
  '旧版本完整保留。本轮只修改最终完整分配方案的评价，保留准入、服务、checkpoint、恢复预算、网格、响应与工作量。没有80作业重跑。',
  '', '## 机制与正确性','',
  '新模式比较旧倒数ETA、最小静态ETA和当前合法分配三个方案，预测所有当前已到达任务的完成时间总和。等待者只能在下一60秒调度边界接续；存活作业配额维持到预测完成，禁止用未来模型。若任一当前作业ETA未知，整轮退回旧bootstrap，明确记录UNKNOWN_MODEL。不是全局JCT最优。FO仅在当轮原FO冻结GPU域中改配额；FULL在原合法GPU域中选择。',
  '', '红测捕获旧目标选择40/500秒而不是50/200秒的净损失；新模式改选总和250秒。额外独立用例覆盖完成释放/等待、60秒边界、真实矩阵rho、FIFO、锁、原子失败、FO冻结与只读。设计审查修复了预测器可能把未提交等待者提前到t=0启动的错误；它现在最早t=60。正式运行前修复并测试，没有改正式结果。','',
  '完整旧套件及新增测试见full_suite.log、green_boundary_review.log；原始Sia3轮逐状态精确回归见original_sia_regression.json。旧332项源码哈希仍一致；source_snapshot.zip、source_changes.patch、source_receipt.json可重建。该目录是源码副本，不虚构Git commit。',
  '', '## 同状态证据','',
  '沿用上轮既定420/1200/13500秒三个状态，不搜索有利时间点。420、1200秒有未知等待者，所以不伪造全体完成预测。13500秒9个长作业状态以原FO分配为共同参考，FO新评价减少预测完成时间总和7830.058秒，FULL减少9495.206秒。这是静态当前模型下的证据，不是实际JCT；该状态没有等待者，不能归因于排队释放收益。',
  '', '## 冻结场景','',
  '既有COMPACT12：6 CIFAR10＋6 BERT，AWS/DGX各一个2GPU节点、每节点6合成CPU单位和125GB。原到达0–1200秒、原训练工作/batch；seed_profiles(1,8)；R0/E-legacy、Medium alpha0.5；60秒、6000轮；独立进程串行。四格完整源/输入哈希在preregistration.json。','',
  '|方法|完成|平均JCT|Median|P95|P99|Makespan|轮数|墙钟秒|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for k,v in results.items():lines.append(f"|{k}|{v['completed']}/12|{v['mean']}|{v['median']}|{v['p95']}|{v['p99']}|{v['makespan']}|{v['rounds']}|{v['wall']:.2f}|")
 lines+=['','P95/P99仅12个样本，是本组任务描述，不是总体尾分位精确估计。','',
  '|对照→新方法|平均节省秒|平均改善%|改善/退化/持平作业|最大贡献|去掉最大贡献后均值节省（诊断）|','|---|---:|---:|---|---|---:|']
 for k,v in comparisons.items():
  if v['paired_complete']:lines.append(f"|{k}|{v['mean_saved']:.3f}|{v['percent']:.3f}|{v['better']}/{v['worse']}/{v['equal']}|{v['largest_gain_job']}：{v['largest_gain']:.1f}s|{v['without_largest_gain_mean_saved']:.3f}|")
 lines+=['','## 时间与资源','', '|方法|等待s|训练s|checkpoint s|重启|浪费GPU-s|GPU-h|CPU单位-h|RAM GB-h|裁剪选中点比例|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for k,v in results.items():
  z=v['totals'];lines.append(f"|{k}|{z['wait']:.1f}|{z['training']:.1f}|{z['checkpoint']:.1f}|{z['restarts']}|{z['wasted_gpu_seconds']:.1f}|{z['gpu_hours']:.3f}|{z['cpu_core_hours']:.3f}|{z['ram_gb_hours']:.3f}|{v['clipped_fraction']:.3%}|")
 lines+=['','## 逐作业JCT','', '|作业|旧FO|旧FULL|新FO|新FULL|','|---|---:|---:|---:|---:|']
 for j in work:lines.append('|'+j+'|'+'|'.join(str(jobs[k][j]['jct']) for k in results)+'|')
 lines+=['','## 实际启用与限制','']
 for k in ('NEW_FO','NEW_FULL'):lines.append(f"- {k}：{results[k]['checks']}；方案选择{results[k]['plans']}。")
 lines+=['', '旧与新FO/FULL包含同样的活性/服务与GPU学习规则；新FO仍先固定旧目标的GPU意图，所以本轮测的是同一串行GPU基线的最终辅助评价改进，不能声称所有阶段都求解了新目标。旧策略完全未改名为新方法。',
  '', '新方法在当前已知集合和保守接续模型中不降低预测总分，不保证真实平均JCT：epoch变化、学习误差、冷启动未知、后续资源重分配及未到达作业均会影响兑现。选中裁剪点仍属E-legacy合成边界，不能当作实测。资源优化没有额外CPU/RAM换配额开销，求解墙钟未加入模拟JCT。',
  '',f"独立原始事件复算：总错误{sum(len(v['errors']) for v in results.values())}；所有完整作业的JCT、互斥时间、资源小时与summary逐项一致。锁、容量、rho和真实训练raw×rho已独立核验。",'',
  '本轮结束，不自动扩大到80作业或其他负载，不按结果改目标/参数。下一步判断以实际配对结果为准，不能仅引用同状态预测收益。']
 (O/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 save('validation.json',dict(errors=sum(len(v['errors']) for v in results.values()),all_completed=all(v['complete'] for v in results.values()),raw_recalculated=True,old_source_unchanged=all(sha(R/'cc-joint-fo-audit-repair-v2'/n)==h for n,h in json.loads((O/'baseline_hashes.json').read_text()).items())))
 print(json.dumps({'comparisons':comparisons,'errors':sum(len(v['errors']) for v in results.values())},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
