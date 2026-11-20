"""Offline proposals only; never constructs a Cluster or advances training."""
import collections,csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_ample_lex_repair_20261030_run1'
def main():
 rows=list(csv.DictReader((OUT/'inputs/BASE_workload.csv').open()));plans={}
 for name,times in [('compact',[int(r['time']) for r in rows]),('burst',[0]*12),('spread',[i*300 for i in range(12)])]:
  jobs=[dict(r,time=t) for r,t in zip(rows,times)]
  plans[name]=dict(jobs=jobs,arrivals=times,arrival_span=max(times)-min(times),submitted_counts={t:sum(x<=t for x in times) for t in [0,300,600,1200,3300]},sha256=hashlib.sha256(json.dumps(jobs,sort_keys=True).encode()).hexdigest().upper())
 assert len({tuple(v['arrivals']) for v in plans.values()})==3
 assert len({v['submitted_counts'][0] for v in plans.values()})==3
 reg={r['key']:r for r in json.loads((OUT/'source/inputs/registry.json').read_text())};coverage=[]
 from joint_prototype.joint_profile import load_synergy_template
 for app in ['cifar10','bert','deepspeech2']:
  for typ in ['aws','dgx-ext']:
   for g in [1,2]:
    key=f'{app}/{typ}:1:{g}';r=reg.get(key)
    if r is None:coverage.append(dict(key=key,missing=True));continue
    a=load_synergy_template(OUT/'source/inputs/models'/r['template']).matrix
    cc,mm=3/r['phi_c'],62.5/r['phi_m']
    coverage.append(dict(key=key,template=r['template'],phi_c=r['phi_c'],phi_m=r['phi_m'],reference_coordinates=[cc,mm],reference_supported=1<=cc<=24 and 20.83<=mm<=250,source_matrix_min=float(a.min()),source_matrix_max=float(a.max()),source_cpu_response_nonflat=bool((a[0,:]!=a[-1,:]).any()),source_ram_response_nonflat=bool((a[:,0]!=a[:,-1]).any())))
 source=ROOT/'official-artifacts/sia-simulator/workloads/saturn/workload-1.csv';original=list(csv.DictReader(source.open()));mix=[]
 for app,n in [('cifar10',6),('bert',3),('deepspeech2',3)]:
  for i,r in [(i,r) for i,r in enumerate(original) if r['application']==app][:n]:mix.append(dict(r,original_source_row_index=i,original_time=r['time']))
 with (OUT/'proposed_workloads.json').open('x',encoding='utf8') as f:json.dump(dict(status='DESIGN_ONLY_NOT_EXECUTED',plans=plans,optional_three_application_real_records=mix,optional_profile_coverage=coverage),f,indent=2,ensure_ascii=False)
 text='''# 下一轮真正有差异的负载设计（仅设计，未运行）

三份旧trace的CC轨迹同构，原因是应用画像/训练预算相同，batch自适应和1–2GPU物理域使很多声明字段差异失效。因此不继续靠换行号或CSV duration字段增加样本。

推荐先只改变真实到达压力，保持同一6 CIFAR＋6 BERT、原始完整训练工作量、模型、配额网格与R0/E-legacy不变。使用已验收的受限RNEW：AWS/DGX各1节点、各2GPU/6合成CPU单位/125GB RAM，Medium alpha=.5、seed_profiles(1,8)、60秒调度/6000轮；不重新搜索容量。AMPLE是资源饱和负对照，不作为检验配额加速的主容量：

|负载|真实arrival规则|t0已到达作业|最大arrival|
|---|---|---|---|
|compact|前4个t0，随后每300秒2个|4|1200s|
|burst|12个全部t0|12|0s|
|spread|第i个i×300秒，i=0…11|1|3300s|

参数源：既有60秒调度周期，300秒为既有到达节拍（5个周期），不是根据本轮JCT选择。相同作业顺序CF/BERT交织；所有方法使用每个负载完全相同输入与容量。三个t0实际活动集合数量不同（4/12/1），直接进入真实Job.submission_time及调度约束，因而不再是只改名字的伪复验。已到达数不等于实际并发数，更不能代替实际互补机会证据。

建议下一次获准后先对burst做60轮上限的有界机会gate：实际CF+BERT同节点有效共训至少60秒；固定实际GPU/batch下记录receiver增益、donor损失和全服务集合净收益；源坐标/容量/锁必须可审计。阈值是一个调度周期；如果未触发，只报告未触发，不能调整到达间隔直到有利。通过后才冻结同版本六方法的完整完成对照。spread用于验证低并发或无互补时收益是否自然减弱，不要求CC获胜。

三种负载×六方法的18格仅是后续建议，尚未授权或启动；本轮不创建可自动启动的批次。compact也需要用修复后同一版本重新配对，不能把旧版结果混进新主分析。

若要进一步改变有效训练模型，而非只有队列压力，候选方案为6 CIFAR＋3 BERT＋3 DeepSpeech2，均取官方trace1各类最早记录，保留完整原始训练工作与Sia的batch自适应能力。DeepSpeech2具有不同的实际Sia应用画像和训练预算，也使用既有非完全平坦的官方响应模板；这比同类换记录更有实质区别。具体记录和12个app/config参考点覆盖见proposed_workloads.json。

三应用方案尚未通过机会gate：必须先完成全部合法GPU配置与CPU/RAM候选覆盖、分子/分母边界曝光、接收方和让出方代价，以及实际共驻机会预检。源矩阵非平坦不意味着目标硬件真实性能已知，也不保证同一节点预算内存在净收益。不得补phi=1、删作业、添加新边界规则或改训练工作量。不同应用progress单位不能直接相加当作统一工作量；需用已有执行率和声明剩余工作估计完成时间。

主要指标：完整配对平均JCT、逐作业改善/退化、等待/训练/checkpoint分解、P95/P99/makespan、重启、配额小时、调度开销及裁剪比例。按有效负载比较，不把相同模板作业当作独立重复；计算量和真实CPU/RAM加载开销仍需单独说明。

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE
'''
 (OUT/'NEXT_WORKLOAD_DESIGN.md').write_text(text,encoding='utf8')
 print('DISTINCT_WORKLOAD_DESIGNS_ONLY',flush=True)
if __name__=='__main__':main()
