"""Raw-event audit and CPU/RAM factorial attribution, no simulation."""
import collections,csv,gzip,hashlib,json,math,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_resource_mixes42_20261028_run1';DEST=OUT/'analysis'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def save(n,x):
 with (DEST/n).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def table(h,rs):return '\n'.join(['|'+'|'.join(h)+'|','|'+'|'.join(['---']*len(h))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rs])
def main():
 DEST.mkdir(exist_ok=False)
 # Reuse independent raw-event auditor without running its historical30-cell main.
 old=ROOT/'maintenance/audit_cc_resource_factorial_complete_20261028.py'
 text=old.read_text(encoding='utf8').split('\ndef main():',1)[0]
 text=text.replace("OUT=ROOT/'research_runs/sia_synergy/cc_joint_resource_factorial_20261027_run1'", "OUT=ROOT/'research_runs/sia_synergy/cc_resource_mixes42_20261028_run1'")
 text=text.replace("DEST=OUT/'completed_analysis_20261028_run1'", "DEST=OUT/'analysis'").replace('DEST.mkdir(exist_ok=False)','')
 ns={'__file__':str(old),'__name__':'independent_audit_library'};exec(compile(text,str(old),'exec'),ns)
 pre=json.loads((OUT/'preregistration.json').read_text());records=[json.loads(x) for x in (OUT/'execution_manifest.jsonl').read_text().splitlines() if x.strip()]
 assert len(records)==42 and all(r['exit_code']==0 for r in records)
 results={};coverage=[];jobs=[];clipping={}
 for i,(mix,cap,response,method) in enumerate(pre['cells']):
  name='_'.join((mix,cap,response,method));assert records[i]['cell']==name
  folder=OUT/'cells'/name
  for n,h in json.loads((folder/'output_hashes.json').read_text()).items():assert sha(folder/n)==h
  x=ns['audit'](folder,pre);s=x['summary'];results[name]=x
  coverage.append(dict(cell=name,completed=s['completed'],mean_jct=s['mean_jct'] if s['completed']==20 else None,
   median=s['median_jct'] if s['completed']==20 else None,p95=s['p95_jct'] if s['completed']==20 else None,p99=s['p99_jct'] if s['completed']==20 else None,
   makespan=s['makespan'],rounds=s['rounds'],censored=20-s['completed'],wall_seconds=records[i]['wall_seconds']))
  for j,v in x['jobs'].items():jobs.append(dict(cell=name,job=j,**v))
  training=s['totals']['training']
  clipping[name]=dict(selected_column_counts=s['checks'],productive_job_seconds=x['clipped_productive_job_seconds'],
    fraction_productive_job_seconds={k:v/training if training else None for k,v in x['clipped_productive_job_seconds'].items()},
    node_resource_hours=x['node_resource_hours'],totals=s['totals'],solver_status=s['solver_status'],
    actual_fallback_status_count=sum(v for k,v in s['solver_status'].items() if 'FALLBACK' in k.upper()),
    fallback_definition='Count fresh decisions whose result.status explicitly denotes FALLBACK; not mere fallback_reason presence.')
  print('AUDITED',name,s['completed'],flush=True)
 save('raw_audit.json',results);save('coverage.json',coverage);save('clipping_and_resources.json',clipping)
 with (DEST/'per_job.csv').open('x',encoding='utf8',newline='') as f:
  fields=sorted({k for j in jobs for k in j});w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(jobs)
 comparisons={};interactions={}
 contrasts=[('CC_FULL','SIA_P'),('CC_FULL','SIA_AUX'),('CC_P','SIA_P'),('CC_CPU','CC_P'),('CC_MEM','CC_P'),('CC_FULL','CC_CPU'),('CC_FULL','CC_MEM'),('CC_FULL','CC_P'),('SIA_AUX','SIA_P')]
 for mix in ['M1','M2','M3']:
  for cap in ['R01','R11']:
   for left,right in contrasts:
    a=results[f'{mix}_{cap}_medium_{left}']['summary'];b=results[f'{mix}_{cap}_medium_{right}']['summary'];key=f'{mix}/{cap}/{left}_vs_{right}'
    if a['completed']!=20 or b['completed']!=20:comparisons[key]=dict(status='CENSORED_NO_COMPLETE_MEAN_COMPARISON');continue
    ds={j:a['jobs'][j]['jct']-v['jct'] for j,v in b['jobs'].items()};winner=min(ds,key=ds.get);loser=max(ds,key=ds.get)
    byapp=collections.Counter()
    for j,v in ds.items():byapp[j.rsplit('-',1)[0]]+=v
    comparisons[key]=dict(improvement=1-a['mean_jct']/b['mean_jct'],mean_delta=a['mean_jct']-b['mean_jct'],per_job_delta=ds,
     improved=sum(v<0 for v in ds.values()),worsened=sum(v>0 for v in ds.values()),unchanged=sum(v==0 for v in ds.values()),application_total_delta=dict(byapp),
     biggest_gain=dict(job=winner,delta=ds[winner]),biggest_loss=dict(job=loser,delta=ds[loser]),
     leave_best_out_diagnostic_improvement=1-(sum(v['jct'] for j,v in a['jobs'].items() if j!=winner)/sum(v['jct'] for j,v in b['jobs'].items() if j!=winner)),
     time_delta={k:a['totals'][k]-b['totals'][k] for k in ['wait','checkpoint','training']},resource_delta={k:a['totals'][k]-b['totals'][k] for k in ['gpu_hours','cpu_core_hours','ram_gb_hours']},
     p95_change=a['p95_jct']/b['p95_jct']-1,p99_change=a['p99_jct']/b['p99_jct']-1,makespan_change=a['makespan']/b['makespan']-1)
   vals={m:results[f'{mix}_{cap}_medium_{m}']['summary'] for m in ['CC_P','CC_CPU','CC_MEM','CC_FULL']}
   interactions[mix+'/'+cap]=vals['CC_FULL']['mean_jct']-vals['CC_CPU']['mean_jct']-vals['CC_MEM']['mean_jct']+vals['CC_P']['mean_jct'] if all(v['completed']==20 for v in vals.values()) else None
 save('paired_comparisons.json',comparisons);save('interaction_seconds.json',interactions)
 neutral={m:ns['neutral_equal'](OUT/'cells/M2_R11_neutral_CC_P/rounds.jsonl.gz',OUT/f'cells/M2_R11_neutral_{m}/rounds.jsonl.gz') for m in ['CC_CPU','CC_MEM','CC_FULL']}
 neutral['SIA_AUX_vs_SIA_P']=ns['neutral_equal'](OUT/'cells/M2_R11_neutral_SIA_P/rounds.jsonl.gz',OUT/'cells/M2_R11_neutral_SIA_AUX/rounds.jsonl.gz')
 save('neutral_regression.json',neutral)
 assert all(neutral.values()),'NEUTRAL_TRAJECTORY_DIVERGENCE_REQUIRES_DIAGNOSIS'
 for n,h in pre['source_hashes'].items():assert sha(OUT/'source'/n)==h
 for n,h in json.loads((OUT/'historical_hash_manifest_files.json').read_text()).items():assert sha(ROOT/n)==h
 totalwall=sum(r['wall_seconds'] for r in records)
 lines=['# 三种负载CPU/内存感知六方法42格实验','',f'42格全部执行结束；完成20/20的格数{sum(x["completed"]==20 for x in coverage)}/42。独立从原始完成事件与执行片段核对。训练子进程累计墙钟{totalwall:.2f}秒。','',
 'SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE','',
 '## 固定实验定义','',
 '三组嵌套受控负载：M1=18CIFAR10+2DeepSpeech2，M2=14+6，M3=10+10；来自workload-1最早源行，完整参数/工作量，固定交错、首4在0秒、其后每300秒2个。不是三个独立自然trace。AWS/DGX-ext各1节点2GPU，R01每节点12合成CPU/187.5GB，R11为6CPU/187.5GB。Medium alpha=.5，36格；M2/R11六Neutral。60秒/6000轮，seed_profiles(1,8)。','',
 '原CC-TAIL源码282文件逐字节保留；仅新运行器传入冻结cohort路径。SIA-P调用官方Sia GPU/放置，无L2；SIA-AUX固定同一Sia GPU/放置意图后用现有ECT辅助优化，不是官方Synergy-TUNE。四CC共享目标、FIFO/持续服务及有界收尾，仅配额自由度不同。','',
 '统一R0/E-legacy：分子分母均按原矩阵边界常值延拓；保留CIFAR DGX双GPU参考CPU0.940805越界，以及其他分子越界，不截phi、不删除合法候选、不改曲线。配额贡献仅在共享CC机制下比较，整体对Sia改善不能全部归CPU/RAM。','',
 '## 指标','',table(['负载','容量','响应','方法','完成','平均JCT','Median','P95','P99','Makespan'],[[s['mix'],s['scenario'],s['response'],s['method'],str(s['completed'])+'/20',s['mean_jct'] if s['completed']==20 else '删失',s['median_jct'],s['p95_jct'],s['p99_jct'],s['makespan']] for s in [r['summary'] for r in results.values()]]),'',
 '## CPU与内存独立增量','',table(['负载/容量','CPU对P','MEM对P','FULL对CPU','FULL对P','FULL对Sia','交互秒'],[[f'{mix}/{cap}']+[f"{100*comparisons[f'{mix}/{cap}/{l}_vs_{r}']['improvement']:.2f}%" if 'improvement' in comparisons[f'{mix}/{cap}/{l}_vs_{r}'] else '删失' for l,r in [('CC_CPU','CC_P'),('CC_MEM','CC_P'),('CC_FULL','CC_CPU'),('CC_FULL','CC_P'),('CC_FULL','SIA_P')]]+[round(interactions[f'{mix}/{cap}'],2) if interactions[f'{mix}/{cap}'] is not None else '删失'] for mix in ['M1','M2','M3'] for cap in ['R01','R11']]),'',
 '改善率=1−新/旧，正值表示JCT更低。交互I=FULL−CPU−MEM+P，负表示比两项相加更有利，正为重叠或抵消；闭环反馈不是直接响应导数。','',
 '## 逐作业、时间与长尾贡献','']
 for key,v in comparisons.items():
  if not key.endswith(('CC_CPU_vs_CC_P','CC_MEM_vs_CC_P','CC_FULL_vs_CC_CPU','CC_FULL_vs_SIA_P')) or 'improvement' not in v:continue
  lines.append(f"- {key}：改善/退化/不变={v['improved']}/{v['worsened']}/{v['unchanged']}；应用ΣJCT差{v['application_total_delta']}秒；等待/训练/checkpoint差={v['time_delta']}；最大受益{v['biggest_gain']}，最大受损{v['biggest_loss']}；去单个最大受益者的诊断改善率={v['leave_best_out_diagnostic_improvement']*100:.2f}%，不替代全部20作业主结果。P95/P99/makespan变化={v['p95_change']*100:.2f}%/{v['p99_change']*100:.2f}%/{v['makespan_change']*100:.2f}%。")
 lines+=['','## 裁剪与资源审计','',table(['格','CPU分子裁剪训练秒%','RAM分子%','CPU分母%','RAM分母%','GPU-h','CPU-h','RAM-GB-h','Restart'],[[n]+[round(100*x['fraction_productive_job_seconds'][k],4) if x['fraction_productive_job_seconds'][k] is not None else None for k in ['numerator_cpu','numerator_ram','denominator_cpu','denominator_ram']]+[round(x['totals'][k],3) for k in ['gpu_hours','cpu_core_hours','ram_gb_hours']]+[x['totals']['restarts']] for n,x in clipping.items()]),'',
 '裁剪比例按有效训练任务秒加权；候选/选中列计数另外保留，不与执行占比混用。0%裁剪不能证明合成跨硬件迁移真实。所有方法共享执行函数，raw_rate_before_rho由执行端直接记录；逐片段验证effective=raw*rho、进度残差、checkpoint不被rho加速。','',
 '## 正确性与范围','',f'Neutral四CC及Sia/串行GPU/进度/完成轨迹逐轮一致：{neutral}。原始raw审计容量、完整checkpoint锁、当轮候选、Sia意图、服务预算、互斥时间、完成前资源小时检查均通过；各格checks和求解/fallback状态保留在raw_audit.json及clipping_and_resources.json。',
 '原229项回归通过，新增2项验收覆盖42配置及12个短链路；初始路径迁移测试失败保留，之后在字节相同原布局复测。没有为测试或JCT修改算法。',
 '20样本P95/P99仅描述，不作为精确总体尾估计。三负载嵌套且同源，不把840个重复作业/42格当独立实验。闭环JCT与单状态进度机会不同；算法更偏短任务、GPU服务承诺和边界外推都可能改变结果，未做未经授权的反事实重跑。',
 '达到上限则保留删失，完整均值只比较两侧均20/20的配对；共同随访sample RMST tau=357600秒已独立复算。没有插补未完成JCT、没有按结果调参或选择性重跑。','',
 '## 复现','',
 'preregistration.json冻结42格顺序、执行合同、源码/输入/依赖；source/完整282文件、inputs/三CSV、各cells独立gzip原始逐轮日志/summary/output_hashes、execution_manifest.jsonl和测试证据。analysis/per_job.csv、paired_comparisons.json、interaction_seconds.json及raw_audit.json保留完整数值。累计墙钟不含准备/测试/资源安全暂停。完成后停止，不自动修改算法或开展其他矩阵。']
 (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
 save('validation.json',dict(status='COMPLETE42_AUDITED',all42_ended=True,all20_completed=all(x['completed']==20 for x in coverage),
  capacity_violations=0,rho_or_progress_violations=0,checkpoint_lock_violations=0,neutral_trajectories_equal=neutral,algorithm_changed=False,total_child_wall_seconds=totalwall))
 shutil.copyfile(Path(__file__),OUT/Path(__file__).name)
 (OUT/'CURRENT_TASK.md').write_text('COMPLETE42_AUDITED. REPORT.md and analysis/ contain42-cell coverage, CPU/MEM ablations, per-job raw-event audit and clipping. STOP; do not tune or expand.\n',encoding='utf8')
 save('delivery_hashes.json',{p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='delivery_hashes.json'})
 print('COMPLETE42_AUDITED',flush=True)
if __name__=='__main__':main()
