"""Read-only interpretation after all18 cells; no simulation or tuning."""
import collections,gzip,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_arrival_pressure18_20261030_run1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def table(h,rows):return '\n'.join(['|'+'|'.join(h)+'|','|'+'|'.join(['---']*len(h))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])
def main():
 m=json.loads((OUT/'metrics.json').read_text());c=json.loads((OUT/'comparisons.json').read_text());raw=json.loads((OUT/'raw_audit.json').read_text());v=json.loads((OUT/'validation.json').read_text());response=json.loads((OUT/'response_analysis.json').read_text());assert v['complete_cells']==18
 tail={}
 for method in ['SIA_P','CC_FULL']:
  seconds=collections.Counter();transitions=[];last=None;budget=[]
  with gzip.open(OUT/f'cells/SPREAD_RNEW_medium_{method}/rounds.jsonl.gz','rt',encoding='utf8') as f:
   for line in f:
    event=json.loads(line);d=event['decision_for_next_interval'];audit=d['result']['audit']
    for e in event['execution']:
     if e['job_id']=='bert-37' and e['resource_allocation']:seconds[e['resource_allocation']['config_id']]+=e['training_seconds']
    col=d['result']['selected'].get('bert-37');ledger=audit.get('service_delay_ledgers',{}).get('bert-37');filt=audit.get('adaptive_candidate_filters',{}).get('bert-37',{}).get('rejection_counts',{})
    if col and col['config']['config_id']!=last:
     last=col['config']['config_id'];transitions.append(dict(round=d['round'],time=(d['round']+1)*60,config=last,delay_ledger=ledger,tail_recoveries=audit.get('tail_recoveries',[])))
    if filt.get('SERVICE_DELAY_BUDGET'):budget.append(dict(round=d['round'],rejected=filt['SERVICE_DELAY_BUDGET'],ledger=ledger))
  tail[method]=dict(job=raw[f'SPREAD_RNEW_medium_{method}']['jobs']['bert-37'],gpu_training_seconds=dict(seconds),transitions=transitions,budget_filter_rounds=budget)
 with (OUT/'tail_bert37_evidence.json').open('x',encoding='utf8') as f:json.dump(tail,f,indent=2,ensure_ascii=False)
 methods=['SIA_P','SIA_AUX','CC_P','CC_CPU','CC_MEM','CC_FULL'];cohorts=['BURST','COMPACT','SPREAD'];lookup={(r['cohort'],r['method']):r for r in m}
 lines=['# 三种到达压力18格：结果解读','SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE',
 '18/18格全部完成12/12。完整审计未见容量、checkpoint锁、倍率重复应用违规；源算法289文件与已验收修复版逐文件相同。没有按结果调整数据/目标/响应/资源/参数。',
 '## 六方法平均JCT（秒）',table(['方法','同时到达BURST','紧凑COMPACT','分散SPREAD'],[[method]+[f"{lookup[co,method]['mean']:.2f}" for co in cohorts] for method in methods]),
 '## 主要配对效果',table(['负载','FULL对Sia','FULL对串行AUX','CPU对CC-P','MEM对CC-P','FULL对CC-P','FULL对CC-P绝对平均省时'],[[co]+[f"{100*c[co][k]['improvement']:.2f}%" for k in ['CC_FULL_vs_SIA_P','CC_FULL_vs_SIA_AUX','CC_CPU_vs_CC_P','CC_MEM_vs_CC_P','CC_FULL_vs_CC_P']]+[f"{-c[co]['CC_FULL_vs_CC_P']['mean_difference']:.2f}s"] for co in cohorts]),
 '最有辨识力的结果：CC-P在BURST和SPREAD分别比Sia差0.46%和8.42%，开放完整CPU/RAM配额后则分别比Sia好18.34%和18.63%。在本合成模型中，优势不能只归于CC排队机制；辅助配额自由度在相同CC目标/准入/候选GPU/执行规则下具有明确闭环增量。该增量包含后续GPU配置与队列反馈，并非固定GPU纯物理加速份额。',
 '## 机会确实触发，但范围须限定',
 'BURST的固定60轮预检中，FULL有4150节点秒的CF加速/BERT不减速共驻（总共驻4240节点秒）；全执行集合净归一化进度为正，同状态ETA核对为正。只有60节点秒的共驻点分子/分母均在源矩阵支持范围内，其余不能冒充域内实测。矩阵查询/网格/registry的41份相关文件与旧覆盖预检逐字节一致，8个物理app/config中5个比例参考点域内、3个有既有边界缺口。',
 'CPU/MEM少量轮次的第一训练段为0秒epoch量化边界，补充按首个正时长段复核ETA（gate_eta_boundary_recheck.json）；FULL判定指标没有该边界，结果完全不变。原始gate结果保留，重算只影响辅助诊断，不改变任何模拟、门槛或JCT。ETA状态求和不是实际节省JCT秒数。',
 '## 到达压力与分母效应',
 '四种CC方法在BURST和COMPACT的GPU/进度/rho/完成轨迹逐轮完全相同，BURST每作业JCT恰好增加其原compact到达时间，平均增加500秒。这说明compact任务已在实际服务机会出现前到达，集中到达没有额外改变CC服务路径。不能把两组当作两种独立执行机制复验。Sia/SIA_AUX在这两组的实际路径发生变化；SPREAD六方法实际路径均变化。',
 'FULL对CC-P的BURST/COMPACT相对改善18.72%/20.84%不同，但绝对平均省时均为919.83秒。SPREAD相对改善更高24.94%，绝对平均省时909.83秒，未超过compact。故不能仅凭百分比宣称分散到达产生了更大的物理优化机会。',
 '## 时间分解与尾部',table(['负载','FULL−Sia总等待秒','总训练秒','总checkpoint秒','改善/退化/不变作业','P95变化','P99变化','Makespan变化'],[[co]+[c[co]['CC_FULL_vs_SIA_P']['time_delta'][k] for k in ['wait','training','checkpoint']]+[f"{c[co]['CC_FULL_vs_SIA_P']['improved']}/{c[co]['CC_FULL_vs_SIA_P']['worsened']}/{c[co]['CC_FULL_vs_SIA_P']['unchanged']}"]+[f"{100*c[co]['CC_FULL_vs_SIA_P'][k]:.2f}%" for k in ['p95_change','p99_change','makespan_change']] for co in cohorts]),
 'BURST/COMPACT的平均收益主要由等待与checkpoint减少兑现，合计实际训练时间反而增加；SPREAD三项时间都下降。因此不能宣称每个任务或总训练过程都更快。去掉单个最大获益作业的诊断平均改善仍为BURST14.46%、COMPACT20.00%、SPREAD13.23%；不作为主结果、不删除任何作业。',
 'SPREAD的P99恶化23.69%主要由bert-37驱动。其JCT从3117变6768秒：等待减少1140秒，checkpoint仍为360秒，训练从1617增至6408秒（+4791），净退化3651秒。Sia主要在DGX单GPU训练1557秒；FULL主要在AWS双GPU训练6180秒，最后才转DGX双GPU168秒。BERT模板rho恒为1，故该退化不是CPU/RAM让出造成的直接减速，而是GPU/执行路径差异。',
 '进一步日志证据：FULL在t900以AWS单GPU启动，预算记120；t1080扩容至AWS双GPU后累计240/250，后续大量GPU候选因SERVICE_DELAY_BUDGET被过滤；直到t7380只剩该作业，收尾恢复授予额外110预算，才迁至DGX双GPU，累计startup360。预算限制与初期配置路径确实限制了后续修正自由度。但没有独立证明更早迁移在全局资源与其他作业代价下必然更优，不能只凭单任务吞吐下结论。本轮不修改策略或预算。',
 '## 结论边界与下一步',
 '当前数据支持：CPU、内存以及联合自由配额有可解释的模型内JCT增量；完整方法在三种到达条件下均优于Sia与串行AUX。不能据此证明真实Saturn集群收益、普遍GPU硬件收益或全局JCT最优。旧E-legacy裁剪、借用平坦BERT模板、零额外辅助配额加载开销和小规模2节点/12作业限制保留。下一步应先审查bert-37预算/初期配置路径，以及增加有效应用性能差异的画像覆盖方案；本轮不自动调参或启动新实验。',
 '全部原始结果、逐作业、资源小时、边界曝光与source/input hashes见REPORT.md、metrics.json、per_job.csv、response_analysis.json、preregistration.json、validation.json。最终封存以delivery_hashes_final.json为准。']
 (OUT/'RESULTS_SUMMARY.md').write_text('\n\n'.join(lines)+'\n',encoding='utf8')
 notes='''# 编排与审计补充

初次启动前可用内存1.670GiB，未构造任何模拟器。短时复查恢复到2.058GiB后开始首个Sia有界gate并完成。离线分析曾在启动器中驻留NumPy/应用画像，下一方法前降至1.958GiB而停止；随后改为独立分析进程。另一次CC-CPU子进程在构造模拟器前测得1.999920GiB而退出，未创建case目录，日志保留。

最终启动器在既有2GiB强制门槛之上等待0.125GiB启动余量，以避免父/子进程启动瞬间的边界抖动；子进程仍独立检查原2GiB/5GiB门槛。完整Sia gate经output_hashes核验后复用，没有重复执行或重写完成日志。所有策略串行、独立Python进程，无用户程序被关闭、无内存门槛降低、无虚拟CPU/RAM/模型/算法参数改变。

正式18格启动时最小可用内存2.529GiB。暂停/等待只增加真实墙钟时间，不推进模拟时间。全部正式case完整，失败启动只发生于gate前资源检查，未补写中断JSONL。

gate_eta_boundary_recheck.json单独记录零时长epoch边界的ETA诊断复核；FULL正式门槛判定完全不变，主JCT数据不涉及此诊断公式。

初始delivery_hashes.json由审计子进程生成，其中gate_resume3_console.log在子进程返回后被父启动器正常追加ALL18_AUDITED_STOPPED，造成唯一活动控制台hash变化。所有源码/输入/原始结果hash一致。所有进程退出后再生成delivery_hashes_final.json，作为最终完整封存；不重写初始manifest。
'''
 (OUT/'ORCHESTRATION_NOTES.md').write_text(notes,encoding='utf8')
 support=OUT/'audit_support';support.mkdir(exist_ok=False)
 for name in ['audit_breakpoint30_precision_v3_20261029.py','audit_cc_resource_factorial_complete_20261028.py']:shutil.copyfile(ROOT/'maintenance'/name,support/name)
 for name in ['interpret_arrival18_20261030.py','recheck_arrival_eta_boundary_20261030.py']:shutil.copyfile(ROOT/'maintenance'/name,OUT/name)
 original=json.loads((OUT/'delivery_hashes.json').read_text());bad=[n for n,h in original.items() if sha(OUT/n)!=h];assert bad==['gate_resume3_console.log'],bad
 with (OUT/'intermediate_seal_review.json').open('x',encoding='utf8') as f:json.dump(dict(changed=bad,expected_parent_final_console_append=True,source_input_result_unchanged=True),f,indent=2)
 final={f.relative_to(OUT).as_posix():sha(f) for f in OUT.rglob('*') if f.is_file() and f.name!='delivery_hashes_final.json'}
 with (OUT/'delivery_hashes_final.json').open('x',encoding='utf8') as f:json.dump(final,f,indent=2,ensure_ascii=False)
 assert all(sha(OUT/n)==h for n,h in final.items());print('FINAL_SEAL_VERIFIED',len(final),flush=True)
if __name__=='__main__':main()
