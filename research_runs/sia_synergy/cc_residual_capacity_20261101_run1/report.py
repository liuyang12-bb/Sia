import csv,gzip,json,math,re,sys,hashlib
from pathlib import Path
import raw_audit_lib as lib
O=Path(__file__).resolve().parent;R=O.parents[2];T=R/'cc-joint-residual-capacity-v1';OLD=R/'cc-joint-uncertainty-v4'
def save(n,x):
 with (O/n).open('x',encoding='utf8') as f:json.dump(x,f,indent=2,ensure_ascii=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
p=json.loads((O/'preregistration.json').read_text());work={x['name']:x for x in csv.DictReader((O/'inputs/workload.csv').open())}
stats={};jobs={};events={};extra={}
for cell in p['cells']:
 name=cell['id'];stats[name],jobs[name],events[name]=lib.audit(cell,p,work);assert not stats[name]['errors']
 errors=[];newpoints=[];serial=0;last=None
 for line in gzip.open(O/'cells'/name/'rounds.jsonl.gz','rt',encoding='utf8'):
  r=json.loads(line);d=r['decision_for_next_interval'];a=d['result']['audit']
  if d['round']==last:continue
  last=d['round'];sel=d['result']['selected'];spec={s['job_id']:s for s in d['probe']['specs']}
  for j,s in spec.items():
   if s['checkpoint_locked'] and s['current_config_id'] and j not in sel:errors.append('LOCK_UNSELECTED')
  for z in a.get('serial_proposals',[]):
   serial+=1
   if z['fixed']!={x['job']:x['config'] for x in z['selected']}:errors.append('FO_FIXED_GPU')
  for j,c in sel.items():
   if abs(c['cpu_per_gpu']-5.)<1e-9 or abs(c['ram_gb_per_gpu']-104.17)<1e-9:
    newpoints.append(dict(time=r['interval_end'],job=j,config=c['config']['config_id'],cpu=c['cpu_per_gpu'],ram=c['ram_gb_per_gpu'],rho=c['rho']))
 assert not errors
 extra[name]=dict(errors=errors,serial_proposals_checked=serial,residual_specific_selected=newpoints)
comparisons={}
for a,b in [('OLD_FO','NEW_FO'),('OLD_FULL','NEW_FULL'),('OLD_FO','OLD_FULL'),('NEW_FO','NEW_FULL')]:
 complete=stats[a]['complete'] and stats[b]['complete']
 if not complete:comparisons[a+'__'+b]=dict(complete=False);continue
 delta={j:jobs[a][j]['jct']-jobs[b][j]['jct'] for j in work}
 comparisons[a+'__'+b]=dict(complete=True,mean_saved=stats[a]['mean']-stats[b]['mean'],percent=100*(1-stats[b]['mean']/stats[a]['mean']),per_job_saved=delta,better=sum(v>0 for v in delta.values()),worse=sum(v<0 for v in delta.values()),equal=sum(v==0 for v in delta.values()))
save('raw_audit.json',stats);save('per_job.json',jobs);save('decision_changes.json',events);save('extra_validation.json',extra);save('comparison_summary.json',comparisons)
legacy=json.loads((R/'research_runs/sia_synergy/cc_uncertainty_20261101_run1/per_job.json').read_text())
matches={n:all(jobs[n][j]['jct']==legacy[n][j]['jct'] and jobs[n][j]['restarts']==legacy[n][j]['restarts'] for j in work) for n in ('OLD_FO','OLD_FULL')};assert all(matches.values())
base=json.loads((O/'baseline_hashes.json').read_text());source=json.loads((O/'source_hashes.json').read_text())
assert all(sha(OLD/n)==h for n,h in base.items());assert all(sha(T/n)==h for n,h in source.items())
oldout=R/'research_runs/sia_synergy/cc_uncertainty_20261101_run1';seal=json.loads((oldout/'artifact_hashes.json').read_text());assert all(sha(oldout/n)==h for n,h in seal.items())
save('validation.json',dict(status='PASS' if all(s['complete'] for s in stats.values()) else 'INCOMPLETE',raw_errors=0,old_baseline_jct_restart_match=matches,source_hashes_match=True,old_source_and_artifacts_unchanged=True))
lines=['# 统一执行下限剩余容量候选：COMPACT12四格对照','','SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE','','## 规则与边界','',
'基于原FULL/FO实际源码完整340文件副本，新增residual_capacity.py及对应测试，不改任何原有源码文件。默认build行为保持原样，NEW两格显式安装新provider。V4预测纠偏未启用，唯一变量是候选表达。',
'',r'每维候选集合为原网格与 $q_k=\min_n[(C_n-k f)/g_n]$ 的并集，$k=0,\ldots,\min_n(G_n-g_n)$。$f$ 是该维既有执行下限（CPU=1、RAM=20.83），$g_n$是本候选占用GPU数，$G_n,C_n$是节点容量。枚举所有可能共驻GPU数，一次生成，不递归、不识别任务名称或读取未来。CPU/RAM分别生成，再由完整联合容量约束过滤。多节点取最紧上界，配额一致。',
'','这只补齐执行下限所对应的剩余容量点，不宣称覆盖所有连续配额最优解。FO仍先在比例域选服务集合与GPU，再优化辅助配额；FULL联合选配置。目标、FIFO/持续服务、checkpoint/恢复预算、batch、rho与执行物理规则均不变。',
'','## 同状态与测试','',
'红测red_behavior.log把新测试接入旧provider，5 CPU缺失断言失败；初始模块缺失的red.log也保留。新增4个实质候选测试通过，green.log还含复用的4项旧断点测试。完整测试结果见full_suite.log。',
'','180、420、4800、5820四个原FULL状态通过当前候选/比例恢复点构建及完整容量/锁审计。420秒两个5 CPU点旧网格拒绝、新网格接受，其他任务配额不变，rho分别提升7.344%/5.021%。最初同状态工具未加入实际流程保留的比例恢复点而在5820失败，same_state.log保留；工具补齐同一真实候选来源后通过same_state2.log，未放松校验。',
'','原始Sia、旧FO、旧FULL各3轮逐状态精确回归通过。完整旧基线由本轮OLD两格再次核对逐作业JCT/restart。',
'','## 完整训练结果','',
'相同12作业、AWS/DGX各2GPU/6合成CPU/125GB、R0/E-legacy Medium alpha=.5、seed_profiles(1,8)、60秒/6000轮。独立进程串行，启动前2GiB/5GiB。四格使用同一源码快照及输入，仅新provider开关不同。',
'','|方法|完成|平均JCT|Median|P95|P99|Makespan|轮数|墙钟秒|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for n,s in stats.items():lines.append(f"|{n}|{s['completed']}/12|{s['mean']}|{s['median']}|{s['p95']}|{s['p99']}|{s['makespan']}|{s['rounds']}|{s['wall']:.3f}|")
lines+=['','P95/P99只是12作业经验分位数，不是总体高精度尾延迟估计。','','|比较（后者对前者）|平均JCT改善%|节省秒/作业|改善/退化/持平|','|---|---:|---:|---|']
for pair,z in comparisons.items():lines.append(f"|{pair}|{z.get('percent')}|{z.get('mean_saved')}|{z.get('better')}/{z.get('worse')}/{z.get('equal')}|")
lines+=['','## 逐作业结果','', '|作业|旧FO|新FO|旧FULL|新FULL|','|---|---:|---:|---:|---:|']
for j in work:lines.append('|'+j+'|'+'|'.join(str(jobs[n][j]['jct']) for n in ('OLD_FO','NEW_FO','OLD_FULL','NEW_FULL'))+'|')
lines+=['','## 时间与资源','', '|方法|等待s|训练s|checkpoint s|restart|wasted GPU-s|GPU-h|CPU单位-h|RAM GB-h|裁剪点比例|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for n,s in stats.items():
 t=s['totals'];lines.append(f"|{n}|{t['wait']}|{t['training']}|{t['checkpoint']}|{t['restarts']}|{t['wasted_gpu_seconds']}|{t['gpu_hours']:.5f}|{t['cpu_core_hours']:.5f}|{t['ram_gb_hours']:.5f}|{s['clipped_fraction']:.3%}|")
lines+=['','## 结果归因和限制','']
for old,new in [('OLD_FO','NEW_FO'),('OLD_FULL','NEW_FULL')]:
 z=comparisons[old+'__'+new];benefit=sorted(z['per_job_saved'].items(),key=lambda x:x[1],reverse=True);net=sum(z['per_job_saved'].values());top=benefit[0]
 lines.append(f"- {new}相对{old}：净总JCT节省{net:.3f}s，最大受益{top[0]}={top[1]:.3f}s，逐作业见上表。")
 if net:lines.append(f"  最大单任务占净收益{100*top[1]/net:.3f}%；该比值可超过100%，因为其他任务可能退化。")
 t0=stats[old]['totals'];t1=stats[new]['totals'];lines.append(f"  时间分解：等待节省{t0['wait']-t1['wait']:.3f}s，训练节省{t0['training']-t1['training']:.3f}s，checkpoint节省{t0['checkpoint']-t1['checkpoint']:.3f}s。这是完整轨迹的会计分解，不单独证明直接CPU加速贡献。")
 lines.append(f"  新特征配额（5 CPU或104.17GB）被选中的作业轮数{len(extra[new]['residual_specific_selected'])}；完整事件在extra_validation.json。该计数只追踪本场景的新标志点，不能当作所有新增候选总量。")
lines+=['','新旧差异可以归因于统一候选表达改变及其后续调度效应；不能全部说成任务直接训练加速。FO也获得同样候选，FULL相对FO的增量应另外看，不能把共享收益全归联合GPU优化。',
'','接收方5 CPU响应坐标在源矩阵范围内，但共驻BERT低配额及部分参考点仍依赖E-legacy裁剪。完整对照不验证真实Saturn硬件性能。求解墙钟不计入模拟JCT，墙钟另列；没有改变历史统计口径。',
'','## 审计与复现','',
'逐完成事件重算JCT、互斥等待/训练/checkpoint、直接raw×rho、进度、容量、锁、资源小时和restart，错误0。额外检查locked不得漏选、FO各串行提案严格固定GPU。源码、输入、依赖、命令、日志、最小diff和完整zip均保存。本轮不运行80作业，不根据结果调参。',
'','完成后停止，等待审查。']
(O/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps(dict(stats={n:{k:s[k] for k in ('mean','p95','p99','makespan','completed')} for n,s in stats.items()},comparisons=comparisons,errors=0),ensure_ascii=False,indent=2))
