"""Deliver only frozen attempt3; never mix failed versions or run training."""
import collections,csv,hashlib,json,shutil,time,zipfile,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'research_runs/sia_synergy/cc_breakpoint_study_20261029_run1';OUT=BASE/'attempt3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def read(n):return json.loads((OUT/n).read_text(encoding='utf8'))
def save(n,x):
 with (OUT/n).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def table(h,rows):return '\n'.join(['|'+'|'.join(h)+'|','|'+'|'.join(['---']*len(h))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])
def percent(x):return f'{100*x:.3f}%'
def main():
 p=read('preregistration.json');v=read('analysis/validation.json');a=read('analysis/raw_audit.json');c=read('analysis/paired_comparisons.json');effects=read('analysis/effect_summary.json');response=read('analysis/execution_response_accounting.json')
 assert v['cells']==30 and v['source_hashes_valid'] and all(v['neutral'].values())
 for n,h in p['source_hashes'].items():assert sha(OUT/'source'/n)==h
 version=read('SOURCE_VERSION.json')
 assert sha(OUT/'source_snapshot.zip')==version['snapshot_sha256']
 with zipfile.ZipFile(OUT/'source_snapshot.zip') as package:assert package.testzip() is None
 old=ROOT/'research_runs/sia_synergy/cc_resource_mixes42_20261028_run1'
 oldhash=json.loads((old/'delivery_hashes_final.json').read_text());assert all(sha(old/n)==h for n,h in oldhash.items())
 rows=[]
 for name,r in a.items():
  s=r['summary'];row=dict(cell=name,completed=s['completed'],mean_jct=s['mean_jct'],median_jct=s['median_jct'],p95_jct=s['p95_jct'],p99_jct=s['p99_jct'],makespan=s['makespan'],rounds=s['rounds'],**s['totals'])
  row['solve_total_seconds']=s['solve_total_seconds'];row['solve_p99_seconds']=s['solve_p99_seconds'];row['process_wall_seconds']=s['total_wall_seconds']
  rd=response[name];den=rd['totals']['productive_job_seconds']
  for tag in ['numerator_cpu','numerator_ram','denominator_cpu','denominator_ram']:
   row['clip_fraction_'+tag]=rd['actual_requested_coordinate_clipping_seconds'].get(tag+'_beyond_1e12_tolerance',0)/den
  row['fallback_count']=sum(value for key,value in s['solver_status'].items() if 'FALLBACK' in key.upper());rows.append(row)
 with (OUT/'ALL_30_CELLS.csv').open('x',encoding='utf-8-sig',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 shutil.copyfile(OUT/'analysis/per_job.csv',OUT/'ALL_JOB_RESULTS.csv')
 primary=[];decomposition=[];influence=[]
 for mix in ['M1','M2','M3']:
  keys=[f'{mix}/R11/{x}_vs_{y}' for x,y in [('CC_CPU','CC_P'),('CC_MEM','CC_P'),('CC_FULL','CC_CPU'),('CC_FULL','CC_P'),('CC_FULL','SIA_P')]]
  if not all('improvement' in c[k] for k in keys):primary.append([mix,'存在删失，完整均值不比较']);continue
  primary.append([mix]+[percent(c[k]['improvement']) for k in keys])
  for label,key in [('CPU',keys[0]),('MEM',keys[1]),('FULL相对CPU',keys[2])]:
   x=c[key];apps=collections.Counter()
   for n,d in x['per_job_delta'].items():apps[n.rsplit('-',1)[0]]+=d
   decomposition.append([mix,label,sum(x['per_job_delta'].values()),x['time_delta']['wait'],x['time_delta']['training'],x['time_delta']['checkpoint'],dict(apps)])
   saving=-sum(x['per_job_delta'].values());share=x['biggest_gain_seconds']/saving if saving>0 else None
   influence.append([mix,label,x['improved'],x['worsened'],x['unchanged'],x['biggest_gain_job'],x['biggest_gain_seconds'],percent(share) if share is not None else '净退化',percent(x['leave_best_out_diagnostic'])])
 main_table=table(['低CPU负载','CPU对P','MEM对P','FULL对CPU','FULL对P','FULL对Sia'],primary)
 equal_means={method:statistics.mean(a[f'{mix}_R11_medium_{method}']['summary']['mean_jct'] for mix in ['M1','M2','M3']) for method in ['SIA_P','SIA_AUX','CC_P','CC_CPU','CC_MEM','CC_FULL']}
 equal_effects={f'{left}_vs_{right}':1-equal_means[left]/equal_means[right] for left,right in [('CC_CPU','CC_P'),('CC_MEM','CC_P'),('CC_FULL','CC_CPU'),('CC_FULL','CC_P'),('CC_FULL','SIA_P')]}
 high=[]
 for left,right in [('CC_CPU','CC_P'),('CC_MEM','CC_P'),('CC_FULL','CC_P'),('CC_FULL','CC_CPU')]:
  x=c[f'M2/R01/{left}_vs_{right}'];high.append([left+'对'+right,percent(x['improvement']),x['improved'],x['worsened'],x['time_delta']['wait'],x['time_delta']['training'],percent(x['p99_change']),percent(x['makespan_change'])])
 overhead=[]
 for method in ['SIA_P','SIA_AUX','CC_P','CC_CPU','CC_MEM','CC_FULL']:
  selected=[x for x in rows if x['cell'].endswith('_'+method)]
  overhead.append([method,round(sum(x['process_wall_seconds'] for x in selected),2),round(sum(x['solve_total_seconds'] for x in selected),2),round(max(x['solve_p99_seconds'] for x in selected),4)])
 clip_table=table(['低CPU Medium格','CPU分子%','RAM分子%','CPU分母%','RAM分母%'],[[x['cell']]+[round(100*x['clip_fraction_'+tag],4) for tag in ['numerator_cpu','numerator_ram','denominator_cpu','denominator_ram']] for x in rows if '_R11_medium_' in x['cell']])
 summary=dict(primary_unit='three nested preregistered20-job cohorts, not independent natural traces',
  cpu_consistent=effects.get('CC_CPU_vs_CC_P',{}).get('wins')==3,
  memory_consistent=effects.get('CC_MEM_vs_CC_P',{}).get('wins')==3,
  conditional_memory_consistent=effects.get('CC_FULL_vs_CC_CPU',{}).get('wins')==3,effects=effects,
  all30_complete=v['complete_cells']==30,old547_files_verified=len(oldhash),wall_seconds=v['wall_seconds'],snapshot_crc_valid=True,snapshot_sha256=version['snapshot_sha256'],
  scientific_objective_changed=False,candidate_representation_changed=True,numeric_units_repaired=True,
  low_cpu_equal_cohort_mean_jct=equal_means,low_cpu_ratio_of_equal_means=equal_effects,
  warning='SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE')
 save('DELIVERY_SUMMARY.json',summary)
 lines=['# 结论与机制解释','','SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE','',main_table,'',
 '## 冻结设计与实现范围','',
 '问题：旧粗网格可能错过源矩阵支持的互补配额。本轮预先要求CPU和RAM各自存在域内的可落地见证，再观察差异化配额能否在完整训练中降低JCT；没有预设FULL必须优于单独优化。','',
 'M1/M2/M3分别为18/2、14/6、10/10个CIFAR10/DeepSpeech2作业，完整训练工作量不变。各4个作业在0秒到达，之后每300秒到达2个直到2400秒；每节点2张GPU，AWS和DGX-ext各一个节点。R11每节点6个合成CPU单位/187.5GB RAM，R01为12/187.5；60秒调度周期、6000轮上限、seed_profiles(1,8)。三组R11 Medium六方法18格，M2/R01 Medium六格，M2/R11 Neutral六格，共30格。','',
 'SIA-P为实际Sia GPU/placement决策加比例执行；SIA-AUX冻结相同Sia GPU/placement意图后使用原辅助ECT优化器，并非Synergy-TUNE。四种CC同用CC-TAIL：P固定双轴比例，CPU只放开CPU，MEM只放开RAM，FULL放开双轴。目标/准入/服务保护/有界收尾完全共用。','',
 '候选构造保留旧网格和比例点，把Synergy矩阵各轴斜率变化点及端点k逆映射为CPU=phi_c×k、RAM=phi_m×k；再加入兼容单节点peer的预算补点(C−g_peer×q_peer)/g_self。只保留当前合法配额与容量内点。peer来自固定公开registry，未读取未来到达或完成时间；候选资格在原子提交及fallback前按当前策略重新审计。','',
 '源码目录不是Git仓库，没有虚构commit。SOURCE_VERSION.json、preregistration.json、逐文件SHA256、SOURCE_CHANGES.patch和source_snapshot.zip共同固定可重建版本；依赖锁包含于快照。','',
 '改善率=1−新方法平均JCT/对照平均JCT。三组低CPU一致性只适用于这些已知冻结cohort；它们嵌套且来自同一trace，不作独立样本显著性检验。','',
 '三组cohort等权平均JCT（秒）：'+str(equal_means)+'。按等权均值之比计算的改善率：'+str({k:percent(v) for k,v in equal_effects.items()})+'。这与逐cohort改善率中位数不同，后者见effect_summary.json。','',
 '## 较高CPU容量对照','',table(['M2/R01对照','平均JCT改善','改善作业','退化作业','等待差秒','训练差秒','P99变化','Makespan变化'],high),'',
 'CPU容量加倍后的CPU单独优化出现平均JCT退化，不能宣称候选自由度越多JCT必然越低。此处使用同一冻结目标与参数，结果完整保留；没有据此改目标或再次调参。内存优化单独收益与CPU开启后的条件增量应分别解读。','',
 '## 逐作业与时间归因','',table(['负载','对照','ΣJCT差秒','等待差','训练差','checkpoint差','应用差'],decomposition),'',
 '负时间差为节省。CPU或RAM自由度的闭环效果包含其后GPU选择、任务提前释放和队列变化，不能全部解释为固定GPU训练提速。固定GPU/完整真实Job.step的小型证据单独见small_gate_audit.json，CPU与RAM均有接收方加速且对应固定见证的让出方无减速。三作业自由调度仍有个别作业退化，全部保留。','',
 'CPU严格支持见证位于R01/AWS单GPU：CF/DS比例CPU各6，替代为6.403597834/4.363200978，RAM仍各93.75GB；CF同状态rho从1.843770214到1.906118055，DS保持1.370511790。RAM见证位于R11/DGX单GPU：CPU各3，RAM从各93.75改为103.280882817/79.071906749；CF rho从1.243060367到1.308234837，DS保持1.165006640。两类见证分子、参考分母均在源数据域内，整体资源不超配。CPU见证并不是R11域内证明，不能把主矩阵所有CPU收益都归为无裁剪机制。','',
 '固定GPU完整训练：CPU平均JCT9960→9907.5秒（0.527%），RAM5176.5→5150秒（0.512%）；自由三作业CPU改善1.641%，RAM改善0.191%。这些值来自独立小型完成事件，静态瞬时ETA不等于整段训练/JCT预测。','',
 '## 高影响作业诊断','',table(['负载','对照','改善数','退化数','不变数','最大受益作业','其节省JCT秒','占净节省','去掉该作业后改善'],influence),'',
 '最大受益作业占净节省可能超过100%，表示其他作业抵消部分收益。去掉作业仅为预定影响诊断，不替代包含20作业的主结果，也不说明这些逐作业观测是独立重复。','',
 '## 条件增量及整体收益的边界','',
 'CC-P本身在GPU候选评分中使用rho；CPU/MEM/FULL消融衡量差异化配额自由度，而不是从完全不感知CPU到感知CPU的全部作用。CC-FULL对Sia还包含完成导向评分、FIFO/持续服务、GPU配置与放置差异；相对Sia的总改善不能全部归CPU/RAM。SIA-AUX固定Sia GPU/placement后按原ECT辅助优化，不是原生Synergy-TUNE，也与CC的准入/目标不同。','',
 '内存单独优化有效不自动意味着CPU优化后还有等量额外收益。请以FULL对CPU的逐场景差值为准；CPU与RAM可能作用于相同接收任务、触及同一速度平台或改变其GPU选择。局部响应分解和实际CF/DS共同训练时间见execution_response_accounting.json，不能当作完整合法反事实调度或JCT因果分解。','',
 '保留的完成优先评分为当前剩余ETA倒数（计入原重配置估计），不等同于全局平均JCT最优。更细候选可能改善当轮分数却让部分长作业延后；较高CPU下的负结果与这种局限相容，但本轮没有用改目标的隔离对照证明唯一因果。','',
 '## 数值失败与版本纪律','',
 '原始本轮15格完成，第16格M3/CC_CPU在t960停止：HiGHS次级结果违反原主目标带约2.67e-7。仅做可逆行缩放后，真实链路又发现主目标“最优值”比已知合法分配低约2.13e-7。独立验证该分配满足原准入/容量，证明是数值精度问题。最终版本把非整数目标统一换算为从既有1e-9半带导出的二幂单位，并保留必要的可逆窄行条件化；目标相对系数、排序、物理约束和求解器选项不变，每次仍验证原始未缩放行。','',
 '原15格、原始失败日志、16轮复现、237项中异常前缀兼容失败及修复、所有红测/失败尝试均保留。attempt2未启动正式矩阵；本报告主结果只使用attempt3重新运行的30格，不挑选各版本最好结果。','',
 '历史42格桥接还包含本次数值正确性修复，因此不能把所有新旧变化唯一归因于网格。主结论来自同一attempt3版本的配额消融。首版已完成的两个完整六方法配对仍是独立保留的旧数值后端下候选扩充证据，不混入三cohort主汇总。','',
 '## 模型与外推限制','',
 '响应仍为固定R0＋E-legacy，alpha=.5；未改变映射、phi、CPU/RAM容量、到达、训练工作量或观察窗。主要静态互补见证分子分母均在源矩阵域内，但全候选域保留CIFAR/DGX双GPU等参考分母缺口；完整训练仍依赖合成边界假设。裁剪比例不能当作收益贡献比例，更不能称迁移模型已真实验证。','',
 '诊断同时报告精确请求坐标越界，以及超过1e-12浮点舍入范围的越界，后者只用于解释端点浮点暴露，未改变执行裁剪。CPU单位为合成可分配单位；RAM-GB小时和CPU/GPU小时为已分配配额积分，不是实测CPU忙碌率、RSS或SM利用率。','',
 '以下为实际训练秒加权、超过1e-12舍入范围的请求坐标越界；各维度可重叠，不能相加解释为总收益份额。精确比较口径另存execution_response_accounting.json，执行时仍为原严格裁剪规则。','',clip_table,'',
 'P95/P99每场景仅20作业，精度有限。辅助资源变化的额外加载开销未建模；调度求解墙钟不计入Sia虚拟训练时间，求解/运行墙钟另列。当前矩阵只有两个合成迁移模板，不证明其他应用或真实集群有同样收益。','',
 '## 计算开销','',table(['方法（各5格）','进程墙钟总秒','调度求解总秒','各格最大P99求解秒'],overhead),'',
 '候选扩展可能显著增加计算成本，不能只报告虚拟JCT。以上墙钟受本机负载影响，非独立硬件基准；资源与求解明细见ALL_30_CELLS.csv。','',
 'Neutral全程rho精确为1；轨迹回归比较逐轮job ID、GPU类型/placement、训练进度和完成时间，不要求无性能作用时的CPU/RAM配额逐字相同。','',
 '## 交付与停止','',f"30格完成状态：{v['complete_cells']}/30；正式子进程累计墙钟{v['wall_seconds']:.2f}秒。241项全套测试、实际16轮失败链路、八格小型训练与Neutral短链通过。旧42格{len(oldhash)}份封存文件hash核验通过。",'',
 '完成本轮后停止。若CPU/MEM未在三组一致改善或FULL对CPU不稳定，如实报告，不自动调权重、换模板、加容量档、扩到104/160或新trace。下一步是否值得独立trace复验由研究者审查。']
 with (OUT/'analysis/INTERPRETATION.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
 report=(OUT/'REPORT.md').read_text(encoding='utf8')
 # Put the direct result first; preserve the machine-generated detail below.
 with (OUT/'FINAL_REPORT.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n\n---\n\n'+report)
 with (OUT/'CURRENT_TASK.md').open('a',encoding='utf8') as f:f.write('\nCOMPLETE30_DELIVERED_STOPPED. FINAL_REPORT.md +analysis/raw_audit.json authoritative. No running formal batch, no further experiment authorized.\n')
 with (BASE/'REPORT.md').open('x',encoding='utf8') as f:f.write('# 本轮实验版本索引\n\n正式主结果在[attempt3/FINAL_REPORT.md](attempt3/FINAL_REPORT.md)。\n\n首版15格完成+第16格数值失败、attempt2有界门槛失败均保留，禁止混用。attempt3使用相同数学目标与数据/响应，完成数值表示正确性修复后统一重跑30格。原42格及所有历史报告未覆盖。\n')
 with (BASE/'CURRENT_TASK.md').open('a',encoding='utf8') as f:f.write('\nFINAL: use attempt3/FINAL_REPORT.md, complete30 delivered and stopped. Parent15 are historical diagnostic, not primary. No restart/next experiment.\n')
 for script in ROOT.glob('maintenance/*breakpoint*20261029*.py'):
  if 'precision_v3' in script.name:
   target=OUT/script.name
   if not target.exists():shutil.copyfile(script,target)
 save('FINAL_VALIDATION.json',dict(v,delivery_complete=True,old_results_hashes_verified=len(oldhash),scientific_changes='grid extension plus exact mathematical numerical representation; no target/model/parameter search'))
 hashes={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name!='delivery_hashes_final.json'}
 save('delivery_hashes_final.json',hashes)
 assert all(sha(OUT/n)==h for n,h in hashes.items())
 print('DELIVERED',summary,'hashes',len(hashes))
if __name__=='__main__':main()
