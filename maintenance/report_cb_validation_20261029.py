import collections,gzip,hashlib,json,shutil,statistics,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_cb_validation_20261029_run1';OLD=ROOT/'research_runs/sia_synergy/cc_cifar_bert12_20261029_run1'
def table(h,rows):return '\n'.join(['|'+'|'.join(h)+'|','|'+'|'.join(['---']*len(h))+'|']+['|'+'|'.join(str(x) for x in r)+'|' for r in rows])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def main():
 p=json.loads((OUT/'preregistration.json').read_text());v=json.loads((OUT/'validation.json').read_text());metrics=json.loads((OUT/'metrics.json').read_text());comp=json.loads((OUT/'comparisons.json').read_text());raw=json.loads((OUT/'raw_audit.json').read_text());resp=json.loads((OUT/'response_analysis.json').read_text());fix=json.loads((OUT/'fixed_mechanism.json').read_text())
 aggregate={}
 for contrast in next(iter(comp.values())):
  vals=[comp[c+'/RNEW'][contrast]['improvement'] for c in ['W1','W5P','W10'] if not comp[c+'/RNEW'][contrast].get('censored')]
  aggregate[contrast]=dict(cases=len(vals),wins=sum(x>1e-12 for x in vals),median=statistics.median(vals) if vals else None,minimum=min(vals) if vals else None,maximum=max(vals) if vals else None)
 with (OUT/'cross_cohort_summary.json').open('x',encoding='utf8') as f:json.dump(aggregate,f,ensure_ascii=False,indent=2)
 baseline=json.loads((OLD/'metrics.json').read_text());oldraw=json.loads((OLD/'raw_audit.json').read_text());tie=[];tie_jobs={}
 for m in baseline:
  cur=next(r for r in metrics if r['cohort']=='W1' and r['method']==m['method']);tie.append([m['method'],m['mean'],cur['mean'],cur['mean']-m['mean']])
  current=raw[f"W1_RNEW_medium_{m['method']}"]['jobs']
  tie_jobs[m['method']]={r['original_name']:current[r['name']]['jct']-oldraw[m['method']]['jobs'][r['original_name']]['jct'] for r in p['cohort_rows']['W1']}
 with (OUT/'baseline_per_job_reproduction.json').open('x',encoding='utf8') as f:json.dump(tie_jobs,f,indent=2)
 signatures={}
 for cohort in ['W1','W5P','W10']:
  ranks={r['name']:i for i,r in enumerate(p['cohort_rows'][cohort])}
  for method in ['SIA_P','SIA_AUX','CC_P','CC_CPU','CC_MEM','CC_FULL']:
   h=hashlib.sha256()
   with gzip.open(OUT/f'cells/{cohort}_RNEW_medium_{method}/rounds.jsonl.gz','rt',encoding='utf8') as f:
    for line in f:
     rr=json.loads(line);events=[]
     for e in rr['execution']:
      a=e['resource_allocation'];geometry=None if not a else [a['config_id'],sorted(a['placement'].items()),a['cpu_per_gpu'],a['ram_gb_per_gpu']]
      events.append([ranks[e['job_id']],e['progress_after'],e['completion_time'],e['rho'],e['cluster'],e['placement'],geometry])
     h.update(json.dumps(sorted(events),sort_keys=True,separators=(',',':')).encode())
   signatures[cohort+'/'+method]=h.hexdigest()
 with (OUT/'effective_trajectory_signatures.json').open('x',encoding='utf8') as f:json.dump(signatures,f,indent=2)
 interaction=[]
 for cohort,cap in dict.fromkeys((r[0],r[1]) for r in p['cells']):
  d={r['method']:r['mean'] for r in metrics if r['cohort']==cohort and r['capacity']==cap}
  if all(d.get(k) is not None for k in ['CC_P','CC_CPU','CC_MEM','CC_FULL']):interaction.append([cohort,cap,d['CC_FULL']-d['CC_CPU']-d['CC_MEM']+d['CC_P']])
 ample_pair=[]
 for b in baseline:
  a=next(r for r in metrics if r['cohort']=='BASE' and r['method']==b['method'])
  ample_pair.append([b['method'],b['mean'],a['mean'],f"{100*(1-a['mean']/b['mean']):.3f}%" if a['mean'] is not None else 'RUN_FAILED 不比较'])
 findings=[]
 for cohort in ['W1','C9B3','C3B9']:
  d=comp[cohort+'/RNEW'];findings.append(f"{cohort}：FULL相对Sia平均JCT改善{100*d['CC_FULL_vs_SIA_P']['improvement']:.3f}%；CPU/MEM/FULL相对CC_P增量分别{100*d['CC_CPU_vs_CC_P']['improvement']:.3f}%/{100*d['CC_MEM_vs_CC_P']['improvement']:.3f}%/{100*d['CC_FULL_vs_CC_P']['improvement']:.3f}%。")
 control=comp['BASE/AMPLE'].get('CC_FULL_vs_CC_P',{})
 if not control.get('censored'):
  findings.append(f"资源充足的完整FULL/P配对中，辅助联合优化平均JCT增量为{100*control['improvement']:.6f}%；SIA_AUX与SIA_P的均值也相同。该负对照支持收益依赖可利用的辅助资源瓶颈；CPU-only/MEM-only两个执行失败使六方法全配对验收仍不完整。")
 lines=['# CC CPU/内存机制、跨负载与适用边界验证','','SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE','',
  f"36格全部尝试，完成12/12的格数={v['complete_cells']}；失败格={v.get('failed_cells',{})}。已完成格及失败格完整记录轮次的独立容量、checkpoint锁和倍率违规均为0，未证明失败后的运行正确性。正式子进程累计墙钟{v['simulation_wall_seconds']:.2f}秒。具体通过状态以validation.json为准。",
  '', '## 直接结论','',*findings,
  '域内固定两任务验证明确存在不依赖有意义坐标裁剪的CPU和内存互补；完整异构执行仍依赖E-legacy。三份不同来源6+6输入的CC执行实际同构，不能据此宣布跨异质负载泛化。资源充足档又暴露两个求解保护失败，应优先独立定位数值/词典序问题，再考虑更大范围实验。',
  '', '## 冻结条件与数据','',
  '算法源码与上一轮288文件逐字节一致；没有新增调度机制、变更参数或按JCT调整数据。复用原241项回归通过的完整源码与证据，本轮另4项冻结/输入/域内机制/历史保护验收通过（contract_tests.log），随后加1项比例canonical验证器红绿回归，5项通过（contract_tests_v2.log）。没有重跑241项并冒称新增结果。',
  'W1、W10分别取官方trace1/10的前6 CIFAR与前6 BERT。trace5只有4 BERT，因此W5P按运行前固定规则使用其4条＋trace10前2条。C9B3取trace1前9CIFAR/前3BERT；C3B9取trace1前3CIFAR/6BERT＋trace5前3BERT。均保留batch、GPU demand、完整工作量；namespaced job ID避免跨trace冲突，原始ID/行/时间/trace保存在CSV。',
  '全新三组cohort的作业ID统一加trace前缀数值，W1因而也是一次同物理工作但名称变化的tie-order敏感性检查，不能隐瞒与历史基线差异。各类均匀交织，前4作业t0，其余每300秒2作业。不同构成改变了同类到达顺序，这是预注册构成实验的一部分。',
  '这些是合成到达与部分重叠的确定性作业清单，不是三个独立自然trace的随机重复。不得将36格或432个作业观测视为独立统计样本。',
  '主容量每AWS/DGX节点2GPU、6合成CPU、125GB RAM。AMPLE按全部8个app/GPU配置的矩阵饱和点推导：每节点CPU/RAM=2×最大每GPU饱和需求；结果'+str(p['capacities']['AMPLE'])+'。该规则在运行前冻结，与JCT无关。',
  'Medium alpha=.5、固定R0/E-legacy、60秒、6000轮上限、seed_profiles(1,8)相同。六方法=SIA_P、固定Sia GPU/placement的SIA_AUX（本项目ECT辅助优化）、CC_P、CC_CPU、CC_MEM、CC_FULL。SIA_AUX不是官方Synergy-TUNE。',
  '', '## 无裁剪域内完整训练验证','',
  'AWS2GPU/6CPU/125GB，cifar10-0与bert-28各1GPU；四种预注册静态配额，保持完整原始任务工作量和batch语义。没有调度优化器参与，不能作为六方法调度胜负证据。所有分子/分母坐标均在支持范围内（浮点端点误差容差仅用于描述，执行响应未变）。任务完成后其资源不转移。',
  table(['配额','CIFAR JCT','BERT JCT','两任务平均JCT','相对比例改善'],[[m,x['jobs']['cifar10-0']['jct'],x['jobs']['bert-28']['jct'],x['mean_jct'],f"{100*(1-x['mean_jct']/fix['P']['mean_jct']):.3f}%"] for m,x in fix.items()]),
  '此处BERT损失为0，CIFAR完成提前直接兑现；makespan由不变BERT决定，故四组makespan均16458秒。说明平均值收益不必意味着makespan收益。BERT的平坦响应来自借用矩阵，并非真实BERT硬件测量。',
  '', '## 六方法完整指标','',table(['负载','容量','方法','完成','Mean','Median','P95','P99','Makespan'],[[r[k] for k in ['cohort','capacity','method','completed','mean','median','p95','p99','makespan']] for r in metrics]),
  '', '## 每组CPU、内存与联合配额增量','',table(['负载/容量','对照','平均JCT改善','改善/退化/不变','等待差','训练差','checkpoint差','P99变化','Makespan变化'],[[c,k,f"{x['improvement']*100:.3f}%",f"{x['improved']}/{x['worsened']}/{x['unchanged']}",*[x['time_delta'][f] for f in ['wait','training','checkpoint']],f"{100*x['p99_change']:.3f}%",f"{100*x['makespan_change']:.3f}%"] for c,d in comp.items() for k,x in d.items() if not x.get('censored')]),
  '改善率为1−新/对照；P99/Makespan变化为新/对照−1，正值表示退化。消融差值是配额自由度的闭环效果，后续GPU选择和队列也会变化，不是固定轨迹的纯物理份额。FULL相对Sia还包含完成导向目标/准入/连续服务/放置差异。',
  '', '## CPU与内存交互','',table(['负载','容量','交互I秒'],interaction),'I=J_FULL−J_CPU−J_MEM+J_P；负值代表联合改善超过两个单项改善绝对值之和，正值代表小于该和。它是均值层面描述，不能当作普遍机制定理或统计显著性。域内固定两任务I='+str(fix['FULL']['mean_jct']-fix['CPU']['mean_jct']-fix['MEM']['mean_jct']+fix['P']['mean_jct'])+'秒。',
  '', '## 三组6+6负载描述性汇总','',table(['对照','胜/组数','改善中位数','范围'],[[k,f"{x['wins']}/{x['cases']}",f"{100*x['median']:.3f}%",f"{100*x['minimum']:.3f}%..{100*x['maximum']:.3f}%"] for k,x in aggregate.items() if x['cases']]),
  '**有效负载多样性限制：** 三组不同官方记录的同方法执行轨迹签名（去除名称、以同一到达顺序编号）见effective_trajectory_signatures.json。下表若为True，表示该方法实际进度/完成/配置/配额/rho逐轮相同。相同轨迹只能支持确定性复现，不能作为新增独立性能场景。',table(['方法','三个执行签名全部相同'],[[m,len({signatures[c+'/'+m] for c in ['W1','W5P','W10']})==1] for m in ['SIA_P','SIA_AUX','CC_P','CC_CPU','CC_MEM','CC_FULL']]),
  '代码依据：source/sia-simulator/job.py:332读取声明batch，但335–338在batch自适应开启时重设为None；254–257仍将声明batch作为候选模型上限，不应简单说完全忽略。cluster.py:395–398仅rigid/scale-gpus用声明GPU demand设上限；当前adaptive物理域只有1–2GPU，多数声明需求均更高。同应用共享Sia画像、最大epoch和R0模板；这些声明差异因此不保证转化为有效速率/工作量差异。本轮保留这些复验结果，不更换数据追求更好指标。真正改变构成的C9B3/C3B9及容量对照单独分析，跨有效异质负载的泛化仍待验证。',
  '', '## 历史基线与改名W1比较','',table(['方法','历史Mean','W1 Mean','差值'],tie),
  '物理数据源相同但名称改变可能影响字典顺序/同分解选择，差异必须保留，不能挑选更好的一次；历史结果没有覆盖。逐作业差值见baseline_per_job_reproduction.json。此诊断不说明所有差异一定来自求解器随机性。',
  '', '## 相同原始BASE输入的容量变化','',table(['方法','原6CPU/125GB Mean','AMPLE Mean','平均JCT改善'],ample_pair),'本表使用上一轮已封存BASE与本轮AMPLE：输入字节和源码hash一致，容量改变为预注册干预。不能用改名W1冒充这个容量配对。容量同时改变CPU与RAM，本表不单独识别两者容量效应。',
  '', '## 资源使用与成本','',table(['负载','容量','方法','GPUh','CPU单位h','RAM GBh','restart','等待秒','训练秒','checkpoint秒'],[[r[k] for k in ['cohort','capacity','method','gpu_hours','cpu_core_hours','ram_gb_hours','restarts','wait','training','checkpoint']] for r in metrics]),
  '', '## 实际互补与边界暴露','',table(['格','CF/BERT共同训练节点秒','固定轨迹局部净进度','receiver','donor损失','CPU裁剪%','RAM裁剪%'],[[n,x['totals'].get('heterogeneous_application_coproductive_node_seconds',0),x['totals']['fixed_geometry_progress_fraction_delta'],x['totals']['receiver_progress_fraction'],x['totals']['donor_loss_fraction'],100*x['actual_requested_coordinate_clipping_seconds'].get('numerator_cpu_beyond_1e12_tolerance',0)/x['totals']['productive_job_seconds'],100*x['actual_requested_coordinate_clipping_seconds'].get('numerator_ram_beyond_1e12_tolerance',0)/x['totals']['productive_job_seconds']] for n,x in resp.items() if not x.get('failed')]),
  '这里的局部进度差固定实际GPU/batch/raw rate/片段时长，比较响应与同节点比例点；不是合法反事实调度或JCT因果分解。边界率为训练秒加权，CPU/RAM可重叠，不能相加。域内两任务结论与全异构范围E-legacy暴露必须分别报告。',
  '', '## 验证器失败、最小修复与完整重跑','',
  '原始BASE/AMPLE/SIA_P运行在t60被旧验证器的浮点完全相等断言停止，见cells/BASE_AMPLE_medium_SIA_P/failure.json。源码joint_scheduler.py:153–160的_proportional_point明确round(node_capacity/gpus,9)，实际CPU38.265102021、RAM223.213690417，与数学除法约差2.5e−10GB/CPU单位；完整源审计valid，容量偏差在原有1e−7容差内。未改算法或容量。新验证器只接受原始字面比例点或源码明确定义的9位canonical点，未放宽资源容量或接受任意附近点。',
  '在全新同级目录cc_cb_validation_20261029_ample_retry1中冻结新运行器与预注册，重跑全部6个AMPLE方法；parent原31条启动记录含失败，原30个已完成主场景结果未重跑或修改。本报告明确组合parent前30格与retry全部6格，不能把原失败目录算作成功。验证器和独立审计均执行相同canonical规则。',
  'AMPLE还暴露算法内部求解保护错误：CC_CPU在t240抛LEXICOGRAPHIC_OBJECTIVE_INCONSISTENT，源码adaptive_service.py:223–248先固定主目标最优值区间、再逐任务固定词典序列，后续call返回None即抛错。当前日志不足以区分数值不可行与约束构造问题，不能武断认定为CPU不足或死锁。未更改任何算法、solver容差、网格或观察窗；其余未启动方法按原顺序继续尝试。失败格保留完整轮次且不计算总体均值，不将执行失败当作6000轮右删失，不插补JCT。资源充足六方法全配对门槛因此未通过，不能宣布完整负对照验证成功。',
  '', '## 解释与结论边界','',
  '资源充足场景用于检查差异化配额的物理增量是否自然减弱；若FULL与CC_P仍有差异，需对照response_analysis中的局部响应增量和离散求解/后续GPU轨迹，不能一概归为CPU收益。混合比例对照同时改变作业构成与总训练需求，只能描述适用性，不能单独识别一种比例的因果效应。',
  '逐作业JCT、完成时间、等待、训练、checkpoint、重启及资源小时见per_job.csv；各对照最大贡献作业和去掉该作业的诊断均值见comparisons.json，不改变全作业主分析。P95/P99仅12个样本，属于描述统计。',
  '所有执行均采用可追溯官方Sia GPU应用画像＋Synergy官方矩阵，应用映射、R0跨硬件缩放、插值与边界规则为合成假设。辅助资源调整没有额外真实数据加载开销模型，求解墙钟不计入虚拟JCT。域内验证也不能证明目标集群真实性能。',
  '', '## 复现与停止','',f"预注册SHA256：{sha(OUT/'preregistration.json')}。源码快照source_snapshot.zip，完整配置与输入hash见preregistration.json，启动记录execution_manifest.jsonl，独立验证validation.json。历史{v['historical_hashes_verified']}封存文件保持不变。",
  '全部本轮工作完成后停止，不自动修改算法、扩大到160作业或恢复历史批次。']
 (OUT/'REPORT.md').write_text('\n\n'.join(lines)+'\n',encoding='utf8')
 for name in ['prepare_cb_validation_20261029.py','fixed_cb_mechanism_20261029.py','audit_cb_validation_20261029.py','report_cb_validation_20261029.py','test_cb_validation_contract_20261029.py','coverage_cb_validation_20261029.py','retry_cb_ample_20261029.py','continue_cb_ample_20261029.py']:shutil.copyfile(ROOT/'maintenance'/name,OUT/name)
 shutil.copyfile(OLD/'tests_canonical.log',OUT/'prior_241_tests.log');shutil.copyfile(OLD/'tests_canonical.json',OUT/'prior_241_tests.json')
 (OUT/'CURRENT_TASK.md').write_text(f"ALL36_ATTEMPTED_{v['complete_cells']}_COMPLETE_PLUS4_FIXED_AUDITED_STOPPED. Failures preserved and no JCT imputation. REPORT.md,validation.json,comparisons.json and response_analysis.json authoritative. No further experiment authorized.\n",encoding='utf8')
 print('SEALING_COMPLETE_ARTIFACTS',flush=True)
 retry=ROOT/'research_runs/sia_synergy/cc_cb_validation_20261029_ample_retry1'
 retryseal={str(x.relative_to(retry)).replace('\\','/'):sha(x) for x in retry.rglob('*') if x.is_file() and x.name!='delivery_hashes.json'}
 with (retry/'delivery_hashes.json').open('x',encoding='utf8') as f:json.dump(retryseal,f,indent=2,ensure_ascii=False)
 seal={str(x.relative_to(OUT)).replace('\\','/'):sha(x) for x in OUT.rglob('*') if x.is_file() and x.name!='delivery_hashes.json'}
 with (OUT/'delivery_hashes.json').open('x',encoding='utf8') as f:json.dump(seal,f,indent=2,ensure_ascii=False)
if __name__=='__main__':main()
