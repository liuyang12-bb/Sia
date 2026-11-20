"""Post-run analysis of persisted events only; never repairs logs or runs jobs."""
import gzip,hashlib,json,sys
from pathlib import Path
O=Path(__file__).resolve().parent
T=O.parents[2]/'cc-joint-recovery-equivalent-v1';sys.path[:0]=[str(T),str(T/'sia-simulator')]
from joint_prototype.execution_audit import audit_execution
from joint_prototype.r0_liveness import R0Oracle
from run_stage2c_cell import independent_rho
import math
def save(n,v):
 with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def footprint(c):return {k:c[k] for k in ['config','placement','cpu_per_gpu','ram_gb_per_gpu','rho']}
gate=json.loads((O/'gate_result.json').read_text()) if (O/'gate_result.json').exists() else None
release={};prefix={};trajectories={};audit=json.loads((O/'raw_audit.json').read_text()) if (O/'raw_audit.json').exists() else {};checks=0;errors=[]
oracle=R0Oracle(T/'inputs/registry.json',T/'inputs/models',{f'cifar10-{i}':'cifar10' for i in range(6)},'medium')
for owner in ('FO','FULL'):
 p=O/f'gate_{owner}.jsonl.gz'
 if not p.exists():continue
 totals={};first_divergences=[];trajectories[owner]={}
 for line in gzip.open(p,'rt',encoding='utf8'):
  r=json.loads(line)
  aa=r['actual']['result']['audit']
  trajectories[owner][r['time']]={'configs':{j:c['config']['config_id'] for j,c in r['actual']['result']['selected'].items()},'progress':{e['job_id']:e['progress_delta'] for e in r['execution']},'completed':{e['job_id']:e['completion_time'] for e in r['execution'] if e['completion_time'] is not None},'geometry_event_reasons':aa.get('geometry_event_reasons'), 'candidate_filters':{j:{k:v.get(k) for k in ['eligible_count','rejection_counts','delay_charged','geometry_event_reasons']} for j,v in aa.get('adaptive_candidate_filters',{}).items()}}
  for method,d in r['decisions'].items():
   used={n:[0.,0.,0.] for n in ['aws:0','rtx:0']};specs={x['job_id']:x for x in d['probe']['specs']};sel=d['result']['selected']
   for j,c in sel.items():
    rho=independent_rho(oracle,j,c['config']['config_id'],c['cpu_per_gpu'],c['ram_gb_per_gpu'])
    if not math.isclose(rho,c['rho'],rel_tol=1e-9,abs_tol=1e-7):errors.append([owner,r['time'],method,j,'RHO'])
    for n,g in c['placement'].items():
     for i,v in enumerate([g,g*c['cpu_per_gpu'],g*c['ram_gb_per_gpu']]):used[n][i]+=v
   for n,lim in [('aws:0',[4,18,125]),('rtx:0',[4,12,375])]:
    if any(v>b+1e-7 for v,b in zip(used[n],lim)):errors.append([owner,r['time'],method,'CAPACITY'])
   for j,s in specs.items():
    if s['checkpoint_locked'] and s['current_config_id']:
     c=sel.get(j)
     if not c or c['config']['config_id']!=s['current_config_id'] or c['placement']!=dict(s['current_placement']) or abs(c['cpu_per_gpu']-s['current_cpu_per_gpu'])>1e-7 or abs(c['ram_gb_per_gpu']-s['current_ram_gb_per_gpu'])>1e-7:errors.append([owner,r['time'],method,j,'LOCK'])
  if r['event']['structural'] and len(first_divergences)<3:first_divergences.append({'time':r['time'],'event':r['event'],'selected':{m:{j:footprint(c) for j,c in d['result']['selected'].items()} for m,d in r['decisions'].items()}})
  for e in r['execution']:
   checks+=1
   if not audit_execution(e)['valid']:errors.append([owner,r['time'],e['job_id'],'EXECUTION'])
   t=totals.setdefault(e['job_id'],dict(progress=0.,wait=0.,training=0.,checkpoint=0.))
   for k,s in [('progress','progress_delta'),('wait','waiting_seconds'),('training','training_seconds'),('checkpoint','checkpoint_delay_consumed')]:t[k]+=e[s]
  del r,d,aa
 prefix[owner]=dict(totals=totals,first_divergences=first_divergences)
save('gate_evidence.json',prefix)
actual_differences=[];best=cur=0
for t in sorted(set(trajectories.get('FO',{}))&set(trajectories.get('FULL',{}))):
 a,b=[trajectories[m][t] for m in ('FO','FULL')];changed=[j for j in set(a['configs'])&set(b['configs']) if a['configs'][j]!=b['configs'][j]]
 same=set(a['configs'])==set(b['configs']);cur=cur+1 if same and changed else 0;best=max(best,cur)
 actual_differences.append(dict(time=t,same_service_set=same,changed=changed,fo=a,full=b))
save('actual_prefix_comparison.json',dict(longest_gpu_difference_intervals=best,rows=actual_differences,scope='diverged actual prefixes, not same-state causal counterfactual; not complete JCT'))
save('gate_independent_audit.json',dict(execution_checks=checks,errors=errors,valid=not errors));assert not errors,errors[:10]
for name,stat in audit.items():
 timeline=[];seen=set();prev={}
 for line in gzip.open(O/'cells'/name/'rounds.jsonl.gz','rt',encoding='utf8'):
  r=json.loads(line);d=r['decision_for_next_interval'];current=d['result']['selected'];done={e['job_id']:e['completion_time'] for e in r['execution'] if e['completion_time'] is not None and e['job_id'] not in seen}
  if done:
   timeline.append(dict(interval_start=r['interval_start'],interval_end=r['interval_end'],completed=done,released_previous_bundles={j:footprint(prev[j]) for j in done if j in prev},
     next_allocations={j:footprint(c) for j,c in current.items()},survivor_changes={j:dict(before=footprint(prev[j]),after=footprint(c)) for j,c in current.items() if j in prev and c['column_id']!=prev[j]['column_id']}))
   seen.update(done)
  prev=current
 release[name]=timeline
save('resource_release_events.json',release)
lines=['# FULL 与 FO GPU配置自由度实验','', 'SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE','',
 '本轮固定6作业/8GPU主场景及8作业/8GPU对照。AWS4GPU/18合成CPU/125GB；RTX4GPU/12合成CPU/375GB。真实应用训练模型和工作量保留，所有作业同时到达。算法和响应未修改。', '']
if gate:
 lines+=['## 机会门槛', '', '|实际前缀|轮数|连续GPU差异且目标提高|其中比较点均受支持的最长连续轮数|通过|','|---|---:|---:|---:|---|']
 for owner,v in gate['owners'].items():lines.append(f"|{owner}|{v['rounds']}|{v['structural_longest']}|{v['supported_longest']}|{v['passed']}|")
 lines+=['','门槛为同一真实状态、同一服务集合、有效原子解、不同GPU配置且共享目标严格提高，至少连续6轮；变化任务的选中响应分子/分母均须受支持。前缀之间的累计进度是不同轨迹结果，不作为同状态纯因果效应。','']
 if not gate['passed']:lines+=['**门槛未通过，按预注册停止，六格完整训练未启动。不能报告本轮总体JCT或以旧状态离线22.61%目标增益替代。**','']
if audit:
 lines+=['## 完整结果','', '|格|完成|平均JCT|Median|P95|P99|Makespan|独立审计错误|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for n,s in audit.items():lines.append('|'+ '|'.join(str(x) for x in [n,s['completed'],s['mean'],s['median'],s['p95'],s['p99'],s['makespan'],len(s['errors'])])+'|')
 pairs=json.loads((O/'comparison_summary.json').read_text())
 lines+=['','## FO/FULL完整配对','', '```json',json.dumps(pairs,ensure_ascii=False,indent=2),'```','', '逐作业和互斥时间来自per_job.json，资源释放事件来自resource_release_events.json。提前完成后的配额变化是实际轨迹证据；两方法此前状态分叉，不能把全部JCT差值归因于某一次释放。仅全部作业完成时报告完整JCT均值。']
lines+=['','## 版本与证据','', 'preregistration.json、runner_receipt.json与source_snapshot.zip冻结输入、执行脚本和344文件实际源码；full_tests.log与回归日志为测试证据。gate_FO/FULL.jsonl.gz保留同状态决策及真实推进；历史结果未覆盖。', '', 'E-legacy为合成边界假设，算法正常并不验证跨硬件迁移真实性；机会门槛单独报告发生裁剪的变化任务。']
(O/'REPORT.md').open('x',encoding='utf8').write('\n'.join(lines)+'\n')
save('artifact_hashes.json',{str(p.relative_to(O)):hashlib.sha256(p.read_bytes()).hexdigest().upper() for p in O.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'})
print('Analysis written',len(audit),'full cells')
