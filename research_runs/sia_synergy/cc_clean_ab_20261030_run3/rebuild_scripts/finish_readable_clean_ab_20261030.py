"""Non-destructive final interpretation and post-writer-close integrity seal."""
import hashlib,json,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_clean_ab_20261030_run3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def save(n,v):
 with (OUT/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
def main():
 m=json.loads((OUT/'metrics.json').read_text());cs=json.loads((OUT/'comparisons.json').read_text());rr=json.loads((OUT/'response_realization.json').read_text())['results'];con=json.loads((OUT/'service_concurrency.json').read_text())
 details={};paragraphs=['## 重要补充：长尾、真实并发及数据支持域','']
 for w in ('COMPACT','SPREAD'):
  c=cs[w+'/B_JO_vs_B_FO'];j=c['largest_beneficiary'];a=m[w+'_B_JO'];b=m[w+'_B_FO'];others=[x for x in a['jobs'] if x!=j]
  aj=sum(a['jobs'][x]['jct'] for x in others)/len(others);bj=sum(b['jobs'][x]['jct'] for x in others)/len(others)
  details[w]=dict(largest_job=j,joint_jct=a['jobs'][j]['jct'],serial_jct=b['jobs'][j]['jct'],largest_saved_seconds=-c['per_job_delta'][j],total_saved_seconds=-12*c['mean_delta'],
    leave_one_out_joint_mean=aj,leave_one_out_serial_mean=bj,leave_one_out_improvement=1-aj/bj,leave_one_out_delta=aj-bj,
    interpretation='Diagnostic only; all12 jobs retained in primary result')
  paragraphs.append(f"{w}联合相对串行的增量：最大受益作业{j}的JCT从{b['jobs'][j]['jct']:.0f}降到{a['jobs'][j]['jct']:.0f}秒，节省{-c['per_job_delta'][j]:.0f}任务秒；全12作业净节省{-12*c['mean_delta']:.0f}。诊断性去掉该作业后，JO/FO均值为{aj:.2f}/{bj:.2f}秒，相对改善{100*(1-aj/bj):.2f}%。主结果仍包含全部作业，不删除长尾。")
 paragraphs+=['',
  '因此：紧凑到达下去掉最大受益作业后联合增量仍为正；分散到达下额外联合增量依赖bert-28，其余11作业诊断性均值反而退化。这不改变全12作业主结果的8.53%改善，但不支持“联合相对串行收益普遍分布于多数作业”。资源优化相对比例的总体收益，与联合相对串行的较小增量，应分开讨论。','',
  f"A组真实并发：A11四作业同时有效训练{con['A11']['training_concurrency_seconds'].get('4',0):.0f}秒，Sia为{con['A_SIA']['training_concurrency_seconds'].get('4',0):.0f}秒；有效训练中每任务平均GPU分别为{con['A11']['average_gpus_per_productively_training_job']:.3f}/{con['A_SIA']['average_gpus_per_productively_training_job']:.3f}。这支持更多任务以较小GPU配置稳定并发、减少排队的解释，并非所有任务都训练更快。完整并发积分见service_concurrency.json。",'']
 for w in ('COMPACT','SPREAD'):
  q=rr[w+'_B_JO']['totals'];paragraphs.append(f"{w} JO实际混合应用共驻有效训练{q.get('mixed_coproductive_node_seconds',0):.0f}节点秒，其中receiver更快且donor不慢{q.get('receiver_faster_donor_not_slower_node_seconds',0):.0f}节点秒；分子/分母均在支持域内的混合共驻仅{q.get('supported_mixed_coproductive_node_seconds',0):.0f}节点秒，域内且受益的对应计数为{q.get('supported_receiver_faster_donor_not_slower_node_seconds',0):.0f}。因此本轮大规模互补兑现主要依赖固定E-legacy边界假设，不能当作源矩阵域内或真实目标硬件的收益证据。")
 paragraphs+=['',
  '同执行轨迹固定GPU/batch/时长的响应核对已计入receiver收益和donor损失（见response_realization.json）；它不是另一个可行完整调度，也不能把归一化进度差换算为节省的JCT。既有域内固定GPU两作业完整训练是另一项独立机制证据，不能据此消除本轮整体实验的边界不确定性。','',
  '本轮能够识别固定模型下的服务组织和资源联合决策贡献，但不构成目标硬件CPU/RAM画像已校准的证明。下一阶段优先补数据支持域与非平坦donor稳健性，而非把这些小场景百分比外推成普遍收益；本轮不实施下一阶段。','']
 save('final_interpretation.json',details)
 text=(OUT/'REPORT.md').read_text(encoding='utf8')
 # Keep the first sealed rendering intact; create an improved FINAL version.
 header='|负载/比较|平均JCT差秒|改善率|改善/退化/持平作业|最大受益作业|去掉最大受益后的诊断均值差秒|\n|---|---|---|---|---|---|\n'
 text=text.replace('|SPREAD/B_FO_vs_B_FP|','\n'+header+'|SPREAD/B_FO_vs_B_FP|',1)
 idx=text.index('## 1. 本轮范围与控制变量');text=text[:idx]+'\n'.join(paragraphs)+'\n'+text[idx:]
 text+='\n\n## 最终交付入口\n\n原REPORT.md保留为首次封存版；本REPORT_FINAL.md修正跨负载表格排版，并补充长尾增量和实际支持域解释，不修改任何运行数据。最终完整性以delivery_hashes_final_v2.json为准。初版seal在最终console写入结束前生成，唯一后续变化为finalize_console.log；旧seal和该日志均保留，v2在所有写入完成后重封存。\n'
 with (OUT/'REPORT_FINAL.md').open('x',encoding='utf8') as f:f.write(text)
 note=ROOT/'research_notes/synergy/CC_CLEAN_ABLATION_AB_C_20261030_FINAL.md'
 with note.open('x',encoding='utf8') as f:f.write(text)
 (OUT/'FINAL_DELIVERY.md').write_text('''# COMPLETE — 15 formal cells, raw audited, stopped

Read REPORT_FINAL.md and C_RESPONSE_PREFLIGHT.md. Main source and all scientific parameters are unchanged since frozen run3. Raw summaries, completion events and logs are immutable. Final integrity authority: delivery_hashes_final_v2.json, generated after final writer closure. OldREPORT and v1 seal are preserved. No new experiment is authorized by this index.
''',encoding='utf8')
 shutil.copy2(Path(__file__),OUT/'rebuild_scripts'/Path(__file__).name)
 prior=json.loads((OUT/'delivery_hashes_final.json').read_text());bad=[n for n,h in prior['files'].items() if sha(OUT/n)!=h]
 assert bad==['finalize_console.log'],bad
 save('seal_v1_postclose_review.json',dict(changed_after_initial_seal=bad,reason='Final console completion line written by the sealer after v1 hashing. Source/inputs/raw logs/metrics unchanged.',v1_sha256=sha(OUT/'delivery_hashes_final.json')))
 files={f.relative_to(OUT).as_posix():sha(f) for f in OUT.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.name!='delivery_hashes_final_v2.json'}
 save('delivery_hashes_final_v2.json',dict(files=files,external_files={str(note):sha(note)},created=time.time(),all_writers_closed=True,parent_seal_sha256=sha(OUT/'delivery_hashes_final.json'),formal_simulation_wall_seconds=prior['formal_simulation_wall_seconds']))
 for n,h in files.items():assert sha(OUT/n)==h,n
 assert sha(note)==json.loads((OUT/'delivery_hashes_final_v2.json').read_text())['external_files'][str(note)]
 print('FINAL_READABLE_SEALED',len(files),note,flush=True)
if __name__=='__main__':main()
