"""Summarize the frozen factorial without further simulation or tuning."""
import difflib,hashlib,itertools,json,math,shutil,statistics,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_queue_mechanism_ablation_20261028_run1';A=OUT/'analysis'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(p,v):
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
def main():
 r=load(A/'raw_audit.json');comp=load(A/'paired_comparisons.json');fac=load(A/'factorial_attribution.json');p=load(OUT/'preregistration.json');val=load(A/'validation.json')
 complete=all(x['summary']['completed']==20 for x in r.values())
 per_metric={}
 if 'shapley_mean_jct_seconds_saved' in fac:
  for metric in ['initial_wait','later_wait','training','checkpoint']:
   y={z:r['Q'+''.join(map(str,z))]['totals'][metric]/20 for z in itertools.product([0,1],repeat=3)};s={}
   for i,n in enumerate('ABC'):
    result=0.
    for z in itertools.product([0,1],repeat=3):
     if z[i]:continue
     zz=tuple(1 if k==i else v for k,v in enumerate(z));k=sum(z)
     result+=math.factorial(k)*math.factorial(2-k)/6*(y[z]-y[zz])
    s[n]=result
   per_metric[metric]=s
  for n in 'ABC':assert math.isclose(sum(v[n] for v in per_metric.values()),fac['shapley_mean_jct_seconds_saved'][n],abs_tol=1e-7)
 save(A/'factor_time_attribution.json',per_metric)
 y={z:r['Q'+''.join(map(str,z))]['summary']['mean_jct'] for z in itertools.product([0,1],repeat=3)}
 triple=sum((-1)**(3-sum(z))*v for z,v in y.items()) if complete else None
 save(A/'three_way_interaction.json',dict(mean_jct_seconds=triple,definition='sum((-1)^(3-|z|) * Y_z), negative denotes additional reduction beyond lower-order terms'))
 lines=['# CC-TAIL排队机制独立消融：完成导向配置、准入与持续服务','',
 '本报告使用新隔离源码和新日志，原CC-TAIL、官方源、已有30格均未修改。没有CPU/RAM响应调整，没有P99优化，没有参数搜索。',
 '**SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE**','',
 '## 1. 实验设计与可识别范围','',
 '同一受控20作业：18 CIFAR10、2 DeepSpeech2，真实Saturn训练参数及工作量，原受控到达序列不变；AWS/DGX-ext各一节点、各2 GPU/6合成CPU/187.5GB。统一Neutral alpha=0、rho=1、比例每GPU 3CPU/93.75GB；60秒、6000轮。所有响应数据、R0 registry与画像初始化seed_profiles(1,8)相同。',
 '选择Neutral与比例配额是为了隔离排队机制，不以CPU差异化加速解释本轮JCT。每GPU配额恰为节点容量/2且均超过合法floor，故辅助容量不额外排除合法GPU组合；所有方法享有相同硬件。本轮CPU小时=3×GPU小时、RAM小时=93.75×GPU小时，资源小时变化不能称为差异化配额贡献。结果不直接代表Medium、更大GPU集群或其他负载。','',
 '|因素|0|1|','|---|---|---|',
 '|A 配置/任务评分|Sia归一化Goodput公平效用及原restart proxy|CC倒数剩余ETA，含真实启动损失|',
 '|B 准入|服务承诺之外目标驱动选择，不强制FIFO队首或排空|旧CC的FIFO队首强制，阻挡时排空，其他任务可选|',
 '|C 持续服务|300秒有效训练段到期重新入队|保持必要服务至真实完成|','',
 '共同边界：完整checkpoint锁、资源容量、原子提交、非空启动保障、原候选/未知模型比例bootstrap、生命周期250秒启动预算、held作业的相同事件门槛和原一次有界TAIL恢复。C0不重置预算；若服务到期后已无再启动预算，则保持当前服务并记录安全保护次数。C0不是无checkpoint/无服务保护的任意抢占。B0不声称个体防饥饿，删失不补JCT。',
 'C的准确解释：服务段到期会使作业退出held集合，除允许重新排队，还会解除只对held生效的配置事件/来源过滤。因此C衡量的是现有持续服务机制包（含其配置限制），不能把它的差值全部解释为抢占或checkpoint代价。原始候选网格不变，不等于动态允许集合不变；后续如需分离这两项，应另做共同eligibility的更细消融，本轮不根据结果修改。',
 'A0不是官方Sia分配器：它只复用Sia评分，仍使用共同联合候选与节点约束；官方Sia另作SIA_P外部基线。A的差异同时包含任务价值/配置评分、未知模型效用和重启估价，不能把A效应狭义解释为GPU数量一项。',
 'u0=λ_n−clip(round(max(Gnorm·restart_factor,epsilon)^p,2))，p=-0.5、λ_n=1.1；u1=(1/(D+W/r))/max_candidate_priority。未知ETA的u1=0，不伪造性能；W仅来自公开固定epoch工作量减当前progress。','',
 'Qabc表示三个开关；8格全因子加SIA_P，共9格，执行顺序与规则在运行前冻结。原始差异包含交互，不能把三个留一改善率相加。','',
 '## 2. 版本与测试','',
 '父版为上一轮30格执行source/的完整282文件，新树cc-joint-queue-ablation复制后逐文件验证。原282文件未改，仅新增queue_factorial.py与test_queue_factorial.py。新模块保留原控制器，借助独立核心实现B/C开关；全部开启走原控制器核心。版本以完整source/快照及preregistration.json/source_hashes为准，不冒用裸Git HEAD。',
 '先记录新增模块不存在的红测（10项均捕获缺失实现）；实现后10项通过；完整239项通过（旧224+此前资源适配5+本轮10），包含真实unseeded BERT完整调度与旧Sia有界回归。另用未改动旧控制器替换新控制器，FIFO关闭/300秒服务段两个新契约均按预期失败，证明开关测试能识别旧行为；初次红测驱动的路径错误日志也保留，没有覆盖。红绿日志均保留。','',
 f"Q111与旧CC-P Neutral逐轮精确回归：{val['historical_regression']['Q111']['exact']}，{val['historical_regression']['Q111']['rounds']}轮；SIA_P与旧Sia适配基线：{val['historical_regression']['SIA_P']['exact']}，{val['historical_regression']['SIA_P']['rounds']}轮。比较包含分配、progress、restart、完成事件，不比较求解耗时。",'',
 '## 3. 九格原始指标','',
 '|方法|完成|平均JCT(s)|Median|P95|P99|Makespan|总等待任务秒|有效训练任务秒|Checkpoint任务秒|Restart|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for name in p['cells']:
  s=r[name]['summary'];t=r[name]['totals'];fmt=lambda x:'—' if x is None else f'{x:.2f}'
  lines.append(f"|{name}|{s['completed']}/20|{fmt(s['mean_jct'])}|{fmt(s['median_jct'])}|{fmt(s['p95_jct'])}|{fmt(s['p99_jct'])}|{fmt(s['makespan'])}|{t['wait']:.0f}|{t['training']:.0f}|{t['checkpoint']:.0f}|{t['restarts']:.0f}|")
 lines+=['','各cohort仅20作业，P95/P99为样本插值，不作为高精度总体尾延迟估计。本轮只记录尾部，不据此改策略。','',
 '## 4. 三项机制的条件贡献','',
 '下表是在其余两项开启时，移除一项的对照；改善为1−Q111/移除版，负数表示开启该因素反而更差。差值均为Q111减移除版。','',
 '|开启的因素|移除版|平均JCT差(s)|Q111改善率|首次等待总差|后续等待总差|训练总差|Checkpoint总差|','|---|---|---:|---:|---:|---:|---:|---:|']
 for factor,off in [('A完成导向评分','Q011'),('B FIFO准入','Q101'),('C持续至完成','Q110')]:
  v=comp['Q111_vs_'+off]
  if not v['comparable_full_mean']:lines.append(f'|{factor}|{off}|删失，不能比较完整均值|—|—|—|—|—|');continue
  d=v['decomposition'];lines.append(f"|{factor}|{off}|{v['mean_delta']:+.2f}|{v['improvement']*100:+.2f}%|{d['initial_wait']:+.0f}|{d['later_wait']:+.0f}|{d['training']:+.0f}|{d['checkpoint']:+.0f}|")
 if all(comp['Q111_vs_'+n]['comparable_full_mean'] for n in ['Q011','Q101','Q110']):
  ca,cb,cc=[comp['Q111_vs_'+n] for n in ['Q011','Q101','Q110']]
  lines+=['',f"直接判断：在另两项保持开启时，开启A使均值{'下降' if ca['mean_delta']<0 else '增加'}{abs(ca['mean_delta']):.2f}秒，开启B使均值{'下降' if cb['mean_delta']<0 else '增加'}{abs(cb['mean_delta']):.2f}秒，开启C使均值{'下降' if cc['mean_delta']<0 else '增加'}{abs(cc['mean_delta']):.2f}秒。这是本负载的条件贡献，不能预先认定三个开关都带来收益。"]
  if ca['mean_delta']>0:
   lines +=[f"尤其，Q011保留共同联合配置/节点约束、FIFO和持续服务，只换回Sia式公平评分，其均值{r['Q011']['summary']['mean_jct']:.2f}秒，比现CC-TAIL更低。本结果不支持把CC对Sia的优势主要归为新完成评分。当前版本保留，不因单cohort更好就自动替换为Q011。"]
 if 'main_effects' in fac:
  lines+=['','### 4.1 全因子平均效应与交互','',
   '主效应是该因素从0切为1、对其他两因素四种组合等权平均的JCT差；负值有利。它与全开版本的留一差异可能不同，反映交互而非统计错误。','',
   '|因素|平均JCT主效应(s)|Shapley节省(s/作业)|首次等待贡献|后续等待贡献|训练贡献|Checkpoint贡献|','|---|---:|---:|---:|---:|---:|---:|']
  for n in 'ABC':lines.append(f"|{n}|{fac['main_effects'][n]['mean_delta']:+.2f}|{fac['shapley_mean_jct_seconds_saved'][n]:+.2f}|{per_metric['initial_wait'][n]:+.2f}|{per_metric['later_wait'][n]:+.2f}|{per_metric['training'][n]:+.2f}|{per_metric['checkpoint'][n]:+.2f}|")
  lines+=['',f"Shapley三项相加严格等于Q000→Q111的平均JCT节省：{fac['baseline000']-fac['full111']:+.2f}秒。它只分解8格之间的差，不能把Sia→Q000的剩余差异归给这三因素。分解包含交互的平均分摊，不能声称每项在所有状态都有同等收益。",'']
  for n,v in fac['pair_interactions'].items():lines.append(f"- {n}二阶交互平均：{v['average']:+.2f}秒；第三因素0/1时分别{v['by_third_factor'][0]:+.2f}/{v['by_third_factor'][1]:+.2f}。正号表示JCT降低量小于简单相加。")
  lines.append(f'- ABC三阶交互：{triple:+.2f}秒，见three_way_interaction.json；进一步说明不能用三个留一比例相加解释整体。')
 lines+=['','### 4.2 评分与冷启动的同状态证据','',
  'Q111与Q011的首轮进入状态、JobSpec与原始候选ID逐项相同，B/C也相同。t=60秒，四个已到达作业的ETA均未知：Q111的效用为0，只强制队首CF0，其他零分任务会增加次级启动成本，因此只启动1个单GPU任务；Q011沿用Sia冷启动Goodput估计，效用各为0.1，选择四个单GPU任务，恰好占满两节点。',
  '这是一项已发生、可复算的初始欠填充证据，不能据此把全部后续均值差归给第一轮。后续在线模型、GPU与完成顺序分叉，完成目标的已知模型排序作用仍与初始化效用一起包含在A评分包中。见identical_initial_state_scoring.json。',
  '原Sia在相同首轮也启动了4个单GPU任务，虽然GPU类型放置不同。因此初始零效用欠填充属于CC评分/准入组合的问题，不能倒过来称为Sia的冷启动缺陷；Sia后续排队和服务中断的损失需要用其完整时间线解释。',
  '因此，本轮若完成导向评分没有增益，就应直接承认；不能因为当前CC整体优于Sia，反推新评分必然是收益来源。也不能在看到这一结果后修改bootstrap再重跑挑选更好结果。']
 lines+=['','## 5. 与Sia的关系：不能把全部差异都叫完成优先','']
 for name in ['Q000','Q111']:
  v=comp[name+'_vs_SIA_P']
  if v['comparable_full_mean']:
   d=v['decomposition'];lines.append(f"- {name}对Sia：平均JCT改善{v['improvement']*100:.2f}%，{v['improved']}作业改善/{v['worsened']}退化/{v['unchanged']}不变；Σ等待差{d['wait']:+.0f}秒（首次{d['initial_wait']:+.0f}、后续{d['later_wait']:+.0f}），Σ训练差{d['training']:+.0f}、Σcheckpoint差{d['checkpoint']:+.0f}。")
 lines+=['','Sia与Q000仍有候选展开/节点约束、任务选择、服务保护、共同非空安全、预算和Tie规则等差异。因此三个因素消融只能检验预注册开关，不能把相对Sia的全部优势机械分摊或证明Sia普遍存在缺陷。',
 '尤其要区分FIFO的排队顺序作用与冷启动准入作用：完成时间模型未知时A1效用为0，A0继续使用Sia已有冷启动Goodput估计；共同bootstrap候选域相同，但分值不同。这是预先定义的A评分包内部差异。若B在A1下收益远大于A0，则至少部分是在弥补A1零分新任务不易被选择的问题，不能把该收益包装为FIFO本身的普适创新。要进一步分离物理GPU配置评分与冷启动价值，需后续另行冻结对称bootstrap消融；本轮不追着结果改变定义。','',
 '## 6. 是否真的改变排队、并发与中断','',
 '|方法|4作业同时占GPU秒|4作业同时有效训练秒|训练中平均每任务GPU|初次等待任务秒|后续等待任务秒|可选任务选中×轮|服务段重新入队|预算安全保持×轮|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for name in p['cells']:
  v=r[name];gpu_training=sum(j.get('one_gpu_training_seconds',0)+2*j.get('two_gpu_training_seconds',0) for j in v['jobs'].values())/v['totals']['training'];lines.append(f"|{name}|{v['allocated_concurrency']['histogram_seconds'].get('4',0):.0f}|{v['productive_concurrency']['histogram_seconds'].get('4',0):.0f}|{gpu_training:.3f}|{v['totals']['initial_wait']:.0f}|{v['totals']['later_wait']:.0f}|{v['states']['optional_selected']}|{v['events'].get('REQUEUE_PRODUCTIVE_QUANTUM',0)}|{v['states']['budget_safety_holds']}|")
 lines+=['','预算安全保持是节点/作业每轮计数，不等于独立事件；服务段到期重新入队也不必然产生物理抢占，若同轮选回相同配置可零重启，必须结合restart、checkpoint和后续等待判断。并发按周期内真实完成边界积分，不以周期起点的分配数量近似。','',
 '## 7. 逐作业与资源指标','',
 '|作业|Sia JCT|Q111 JCT|差(s)|首次等待差|后续等待差|训练差|Checkpoint差|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for j in r['SIA_P']['jobs']:
  x,y=r['Q111']['jobs'][j],r['SIA_P']['jobs'][j]
  f=lambda k:x[k]-y[k]
  if x['jct'] is not None and y['jct'] is not None:lines.append(f"|{j}|{y['jct']:.0f}|{x['jct']:.0f}|{f('jct'):+.0f}|{f('initial_wait'):+.0f}|{f('later_wait'):+.0f}|{f('training'):+.0f}|{f('checkpoint'):+.0f}|")
 lines+=['','全部9方法×20作业的原始完成、等待、重启、GPU/CPU/RAM小时见raw_audit.json/jobs，成对逐作业差见paired_comparisons.json；未删除任何长作业。','',
 '|方法|GPU-h|CPU单位-h|RAM-GB-h|Wasted GPU-s|求解累计墙钟(s)|','|---|---:|---:|---:|---:|---:|']
 for n in p['cells']:
  t=r[n]['totals'];lines.append(f"|{n}|{t['gpu_hours']:.3f}|{t['cpu_core_hours']:.3f}|{t['ram_gb_hours']:.3f}|{t['wasted_gpu_seconds']:.0f}|{r[n]['summary']['solve_total_seconds']:.3f}|")
 transitions=load(A/'active_to_wait_transitions.json')
 lines+=['','### 7.1 持续服务减少的是哪一种等待','',
  'A1/B1固定时，C1相对C0：首次等待增加77520任务秒，但后续回队等待减少100080，checkpoint减少2600，有效训练增加552；净JCT节省24608任务秒，即1230.40秒/作业、14.04%。因此不能说持续服务主要使任务更早首次启动，它主要避免中途反复回队。',
  'A0/B1固定时，C1同样改善均值17.71%。但B0下C1退化，说明持续服务不是无条件优势：它还会使任务长期受held配置事件门槛约束。首个同状态证据在t=7140：Q100到期后允许DS9从AWS2换DGX2；Q101因无ARRIVAL/COMPLETION/FIRST_MODEL事件拒绝这一已生成目标，虽然启动预算尚足。见continuity_first_geometry_divergence.json。这是配置限制的代价，不能错误归为CPU数据或checkpoint损失。','',
  '|方法|活动分配转等待次数|','|---|---:|']
 for n in p['cells']:lines.append(f"|{n}|{transitions[n]['active_to_wait_transitions']}|")
 lines+=['','active_to_wait_transitions.json保留每个真实转等待事件的时间、原配置、仍活动作业和新分配。完成任务退出不算抢占，checkpoint中的转等待一律非法。',
  '从模块角度，最有证据支持的是有序启动准入与有效服务连续性，二者合作减少初次和中途排队；现有完成评分不是本负载已证实的额外优势。下一阶段应先对称化冷启动评价，并把“必须被服务”与“何时允许合法改配置”拆开验证，而不是立刻调权重或CPU响应。本轮仅提出，不实施；也不把Q011单cohort较好直接当成部署结论。']
 lines+=['','## 8. 正确性、复现和局限','',
 '独立审计从原始逐轮完成事件重算JCT、分位数、初次/后续等待、队列面积和未完成面积；核对JCT=wait+training+checkpoint，训练段直接raw_rate_before_rho×rho，不通过effective/rho反推raw。按上一轮决策核对本轮实际分配，完整checkpoint锁、合法配置、三资源容量及单次rho均检查。资源计时排除周期中途完成后的时间。',
 f"九格完成情况：{'; '.join(n+':'+str(v['summary']['completed'])+'/20' for n,v in r.items())}。独立容量/锁/进度/倍率违规数为0。全部旧父文件{val['parent_files_unchanged']}个未变；所有输出文件hash校验通过。",'',
 '结论适用范围：单一受控20作业、四GPU、Neutral/比例资源、当前在线性能估计与候选。单cohort的确定性8格是机制对照，不是8次独立随机重复，不提供总体显著性检验。C0的预算安全保护与共用TAIL会缩小持续服务开关差异，报告不能外推到无限预算的任意抢占。A包含冷启动及重启评分差异，不声称仅GPU扩展效率解释全部。',
 '日志有继承标签问题：C0仍可能带旧父类periodic_service_expiry=False/nonpreemptive_completion=True，以及OPTIMAL_COMPLETION_CONTINUITY状态名；A0的旧probe.contract.objective也可能仍为完成评分标签。实际开关以新queue_factors、最终shared_contract.objective/service、audit.service_rule及真实重新入队/候选效用事件为准，不能用旧标签推断本轮机制。独立审计在effective_semantics逐格计数并规范化解释，保留原始日志，不修改运行中冻结源、不重写历史记录。此问题影响元数据解读，不改变已执行的分配、评分或进度；下一版应修正标签并增加元数据一致性测试。',
 '本负载应用的单次启动25/50秒短于60秒周期，正式格可能没有跨周期checkpoint锁；不能以正式格零违规替代锁覆盖，真实120秒unseeded BERT链路和8格控制器故障/锁测试提供独立证据。',
 '当前未实现新的长尾保护、CPU/RAM优化或用于性能部署的新活性机制；B0/C0仅是隔离的消融开关及必要非破坏性安全边界。若有因子无收益或退化，保留结果，不做调参后挑选。下一阶段只能据本轮效果提出独立模块边界和新的预注册多负载验证，不能自动扩大实验。','',
 '## 9. 复核与停止','',
 '预注册preregistration.json/DESIGN.md，执行execution_manifest.jsonl；逐格cells/*/rounds.jsonl.gz和summary.json；红绿red_missing_module.log/green1.log/full_tests.log；精确历史回归Q111_historical_regression.json/SIA_P_historical_regression.json；独立审计analysis/validation.json、raw_audit.json、paired_comparisons.json、factorial_attribution.json与factor_time_attribution.json。',
 '新源码source/；完整包rebuild_source.zip与source/内依赖锁；离线脚本见rebuild_scripts/。本轮完成即停止，不自动改算法或启动下一轮。']
 log=[json.loads(x) for x in (OUT/'execution_manifest.jsonl').read_text().splitlines()]
 lines.insert(6,f"正式9格串行实测累计子进程墙钟：{sum(x['wall_seconds'] for x in log):.2f}秒；不含准备、测试和离线分析。")
 with (OUT/'REPORT.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
 scripts=OUT/'rebuild_scripts';scripts.mkdir(exist_ok=False)
 for n in ['prepare_queue_ablation_20261028.py','implement_queue_ablation_20261028.py','run_queue_ablation_20261028.py','audit_queue_ablation_20261028.py','queue_interruptions_20261028.py','report_queue_ablation_20261028.py']:shutil.copy2(ROOT/'maintenance'/n,scripts/n)
 parent=load(OUT/'parent_source_hashes.json');new_files=sorted(set(p['source_hashes'])-set(parent))
 save(OUT/'source_changes.json',dict(unchanged_parent_files=len(parent),changed_parent_files=[],added_source_files=new_files,
  note='Only isolated opt-in module and tests added; old controllers retained byte-for-byte.'))
 with (OUT/'source_additions.patch').open('x',encoding='utf-8') as f:
  for n in new_files:f.writelines(difflib.unified_diff([], (OUT/'source'/n).read_text(encoding='utf-8').splitlines(True),fromfile='/dev/null',tofile='b/'+n))
 (OUT/'REPRODUCE.md').write_text('''# 可重建与复跑\n\n1. 保留本目录所有旧日志不动。rebuild_source.zip含完整284文件源码、GPU数据/CPU矩阵、输入与dependencies.lock.txt，逐文件哈希为preregistration.json/source_hashes。source_additions.patch只记录本轮新增模块/测试，父282文件未改。\n2. 在全新项目根目录还原source为cc-joint-queue-ablation，按原依赖锁创建独立环境；测试路径依赖原项目层级及官方Synergy矩阵，须保留research_sources/Synergy/simulator/models的相同只读内容；本轮没有安装/升级任何依赖。\n3. 复制rebuild_scripts到新根目录maintenance，复制本轮preregistration.json及source到同名research_runs/sia_synergy/cc_queue_mechanism_ablation_20261028_run1。不得复制旧cells到目标，否则独占创建拒绝运行；必须使用不同根目录，不能覆盖本项目证据。\n4. 使用相同Python/依赖依次调用run_queue_ablation_20261028.py cell --name Q111/SIA_P/Q000/Q001/Q010/Q011/Q100/Q101/Q110；每格独立进程，启动前2GiB内存/5GiB磁盘检查。脚本verify不允许随意变更冻结源或依赖。\n5. 历史逐轮回归需要原30格Q111/Sia参考日志，路径和hash见两个historical_regression.json。与新训练独立的复核入口audit_queue_ablation_20261028.py从raw完成事件与训练片段复算；应放在新的输出根目录，analysis独占创建。\n\n所有复跑是操作说明，本轮交付后不自动执行。\n''',encoding='utf-8')
 with zipfile.ZipFile(OUT/'rebuild_source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
  for f in (OUT/'source').rglob('*'):
   if f.is_file():z.write(f,'source/'+f.relative_to(OUT/'source').as_posix())
 with zipfile.ZipFile(OUT/'rebuild_source.zip') as z:assert z.testzip() is None
 current=OUT/'CURRENT_TASK.md';old=current.read_text(encoding='utf-8');current.write_text('最终状态：COMPLETE_DIAGNOSED；9格已结束，独立审计通过，报告REPORT.md。停止，不自动修改算法或启动后续实验。\n\n'+old,encoding='utf-8')
 save(OUT/'delivery_hashes.json',{f.relative_to(OUT).as_posix():sha(f) for f in OUT.rglob('*') if f.is_file() and f.name!='delivery_hashes.json'})
 print('REPORT_AND_REBUILD_PACKAGE_COMPLETE',OUT)
if __name__=='__main__':main()
