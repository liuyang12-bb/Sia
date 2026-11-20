"""Generate a new independent report; never edit source or historical output."""
import json,hashlib,math
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2];T=R/'cc-joint-core80-full-lex-repair-v8'
S=R/'research_runs/sia_synergy/cc_core80_full_lex_repair_20261031_run5'
def read(n):return json.loads((O/n).read_text(encoding='utf8'))
def link(rel,line):return f'[{rel}:{line}]({(T/rel).as_posix()}:{line})'
def f(x):return f'{x:,.2f}'
def main():
 s=read('summary.json');m=read('method_mechanisms.json');states=read('structural_states.json');late=read('late_budget_states.json');jobs=read('per_job_comparison.json');trajectories=read('job_trajectories.json')
 raw=json.loads((S/'audit/raw_recomputed_metrics.json').read_text(encoding='utf8'))
 a=[];w=a.append
 w('# 80作业 FULL 与串行 FO 的收益抵消及场景适用性诊断\n')
 w('**结论：场景提供了CPU/内存优化机会，FO已经利用了其中大量机会；FULL的额外联合选择也产生了收益，但被部分GPU配置路径、持续服务与生命周期启动预算的代价抵消。不能把1.09%全部归于数据不够敏感，也不能据此否定联合调度。尚未闭合的是在相同状态下量化“必须改变GPU意图才能获得”的机会，并验证能否持续兑现。**\n')
 w('SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE。此轮仅读取封存日志、核验源码、计算局部响应，不运行训练仿真或求解批次，不修改算法、参数或旧报告。\n')
 w('## 两种问题需要分别检验\n')
 w('辅助资源问题：给定GPU配置，比例CPU/RAM浪费了一些任务用不完的资源，同时使敏感任务无法达到更高速度。CPU/MEM/FULL对PROP能够检验这一点。\n')
 w('联合决策问题：比例点上看起来较好的GPU配置，在重新分配CPU/RAM后不再是最好的可行选择；固定该GPU意图的FO即使把辅助资源分配到最优，也无法达到另一个GPU/放置/服务组合的效果。FULL对FO检验这个更强的问题。仅有敏感任务、排队或CPU不足，都不足以推出FULL必然明显优于FO。\n')
 w('设相同状态、相同强制服务集合和候选资格下，c为GPU/placement选择、q为辅助配额，U(c,q)为当前目标。串行先求cP=argmax U(c,qP)，再求qFO=argmax U(cP,q)；联合直接求(cJ,qJ)=argmax U(c,q)。FO最终解属于联合域时，单状态U(cJ,qJ)≥U(cP,qFO)。严格优势要求联合最优不能由cP下的配额调整实现；闭环JCT没有这个包含关系保证，因为下一轮状态、模型和约束历史会分叉。\n')
 w('## 现有代码保留了哪些自由度\n')
 w('- FO的两阶段求解：'+link('sia-simulator/joint_prototype/clean_ablation.py',69)+'。先用比例候选求服务/GPU提案，再冻结所选作业及GPU类型、数量、节点数，重新优化辅助资源与placement。它不是原Sia MIP，也不是仅做比例分配的弱基线。\n')
 w('- FULL与FO共同目标来自'+link('sia-simulator/joint_prototype/completion_continuity.py',225)+'：每候选得分 u=1/[d+W/(r×rho)]，再除以共同正数作数值归一化。W是声明剩余进度，r来自当前已学习GPU模型与实际batch规则，d为剩余checkpoint或真实目标启动成本。'+link('sia-simulator/joint_prototype/completion_time.py',31)+'并未读取未来完成时间。\n')
 w('- 这是倒数ETA之和代理，不是最小化全体实际JCT，也不是最小化预计ETA之和。忽略d时u=r×rho/W，边际训练率权重约为1/W；它会倾向给较接近完成的任务资源，不能自动计入被减速长作业后续延迟。纯数学例子：ETA(1,100)的倒数和大于(3,4)，但ETA总和101远大于7；此例只说明目标不等价，不是本次仿真的反事实。\n')
 w('- 准入与持续服务：'+link('sia-simulator/joint_prototype/clean_ablation.py',166)+'、'+link('sia-simulator/joint_prototype/clean_ablation.py',292)+'。已服务者保持强制选中；队首可行才进入必选集合，不能共存就排空；新任务未知ETA仅允许比例bootstrap且效用为0（'+link('sia-simulator/joint_prototype/completion_time.py',112)+'）。次级目标最小化真实启动，因此不必额外接收一个零效用、正启动成本的新任务。\n')
 for name in ('FULL','FO'):
  x=m[name];w(f'  - {name}：{x["decisions"]}次真实决策，optional_selected合计{x["optional_selected"]}作业轮，{x["drain_rounds"]}次排空。实际没有通过optional自由选择额外作业；服务集合由已有承诺/FIFO队首主导。源码存在可选机制，不等于本轮实际利用了它。')
 w('\n- GPU调整受事件、已观测profile、前次有效训练及累计250秒启动预算约束，见'+link('sia-simulator/joint_prototype/clean_ablation.py',300)+'。一次尾部恢复只在剩余一个已到达作业时提供，见同文件325行。完整checkpoint锁保留，不属于待删除的限制。\n')
 w(f'FULL/FO有{s["initial_gpu_differences"]}/80个作业首次GPU配置不同、{s["final_gpu_differences"]}/80个最后GPU配置不同。两者不是同一策略；但历史约束会限制后续纠偏。过滤计数按候选列轮或作业轮计，不当作独立丢失机会数量：FULL的启动预算过滤涉及{m["FULL"]["rejection_job_round_counts"].get("SERVICE_DELAY_BUDGET")}作业轮，FO为{m["FO"]["rejection_job_round_counts"].get("SERVICE_DELAY_BUDGET")}。\n')
 w('## 平均JCT收益为什么只剩百分之一左右\n\n差值统一为FO−FULL，正值表示FULL节省。\n\n|应用|数量|JCT总节省秒|等待节省秒|checkpoint节省秒|训练节省秒|\n|---|---:|---:|---:|---:|---:|')
 for app,x in s['application_decomposition'].items():w('|'+app+'|'+str(x['count'])+'|'+'|'.join(f(x[k]) for k in ('jct_saved','wait_saved','checkpoint_saved','training_saved'))+'|')
 total={k:sum(x[k] for x in s['application_decomposition'].values()) for k in ('jct_saved','wait_saved','checkpoint_saved','training_saved')}
 w('|总计|80|'+'|'.join(f(total[k]) for k in ('jct_saved','wait_saved','checkpoint_saved','training_saved'))+'|')
 w(f'\n21,240秒等待节省，被14,848秒额外训练和650秒额外checkpoint抵消，净省{f(total["jct_saved"])}秒，即每作业71.775秒。CIFAR净省10,704秒、BERT净省6,232秒，DeepSpeech2净损失11,194秒。DeepSpeech2虽少等4,020秒，训练却多15,164秒；这是主要抵消来源。\n')
 w('FULL相对FO：61作业改善、18退化、1持平，逐作业改善的中位数243.5秒。多数作业小幅改善与少数作业大幅退化并存，不是没有改善，也不是全体普遍更快。最大受益bert-10019节省6818秒，比全部净节省5742秒还多；诊断性去掉它后均值退化0.207%，主分析仍保留全部80作业。\n')
 w('## FO 已经利用的资源机会\n')
 x=m['FO'];w(f'FO共{x["serial_rounds"]}次同状态串行优化，{x["serial_positive_utility_rounds"]}次（{100*x["serial_positive_utility_rounds"]/x["serial_rounds"]:.2f}%）辅助阶段提高当前目标；{x["serial_changed_quota_jobs"]}个作业轮改变比例配额，GPU意图违规{x["serial_gpu_violations"]}。这些是同一状态中的proposal→selected真实记录，不涉及跨策略重建。\n')
 ex=x['serial_examples'][0]
 w('例如t=180秒的DGX节点：同样两个CIFAR作业各4GPU，提案各3CPU/62.5GB每GPU；辅助阶段改成receiver每GPU5CPU/104.17GB、donor每GPU1CPU/20.83GB，总资源仍为8GPU/24CPU/500GB。receiver的rho=1.7287，donor的rho=0.7546，已将让出方损失计入同一当前目标，目标从0.8786提高至1.0994。FO已经具备差异化CPU/RAM和有损但净目标有利的交换，不只是依赖平坦BERT无损捐赠。\n')
 w('|方法/应用|实际训练任务秒|训练秒加权rho|rho>1训练秒|rho<1训练秒|\n|---|---:|---:|---:|---:|')
 for name in ('FO','FULL'):
  for app,v in m[name]['application_execution'].items():w(f'|{name}/{app}|{f(v["training"])}|{v["time_weighted_rho"]:.4f}|{f(v["accelerated_seconds"])}|{f(v["slowed_seconds"])}|')
 w('rho加权均值只能描述各自轨迹，不能直接比较绝对训练速度：GPU、batch、epoch和任务组成随时间不同。FULL的DeepSpeech平均rho稍高，实际训练时间仍更长，表明辅助倍率没有弥补GPU配置路径等差异。\n')
 w('FO同节点差异化配额、receiver加速且存在低比例peer的重叠训练累计328598作业对秒；FULL为312205。完全在原始分子和分母支持域内的对应数量仅4951和3357作业对秒，不能用大量裁剪片段证明目标硬件真实性。局部相同比例参考的净等效训练秒FO=52443.38、FULL=63132.17，只是固定已执行片段的响应核对，不是合法闭环反事实或JCT节省。\n')
 w('## 未兑现的联合配置收益和恢复限制\n\n|作业|FULL JCT|FO JCT|FULL训练|FO训练|直接观察|\n|---|---:|---:|---:|---:|---|')
 notes={'bert-10009':'FULL AWS1→RTX1，FO RTX1→DGX2；两边rho一直1、累计启动240秒。',
        'bert-10019':'FULL AWS1→DGX1，FO AWS1→RTX1；两边rho一直1、累计启动240秒。',
        'deepspeech2-10043':'FULL在5220秒耗尽250秒预算并停留AWS1；FO最终停留RTX1。',
        'deepspeech2-10048':'FULL在6060秒耗尽预算停留AWS1；20820秒仅剩它时才获一次DGX8尾部恢复。',
        'deepspeech2-10068':'FULL最终DGX1；FO能调整到DGX2→DGX4。'}
 for j,note in notes.items():
  row=next(z for z in jobs if z['job']==j);w('|'+j+'|'+'|'.join(f(row[k]) for k in ('jct_FULL','jct_FO','training_FULL','training_FO'))+'|'+note+'|')
 w('\nbert-10009在FULL的RTX1上有效训练7895秒，另有AWS1的60秒；FO在DGX2上748秒、RTX1上60秒。两个策略均已消耗240秒，而再迁移需120秒，会超过250秒预算。BERT模板为全1矩阵，因此不能把这7147秒差距归于CPU/RAM响应。这里证明的是配置路径与历史约束会长期保留事后较慢的选择；原候选ETA仅保存hash，不能宣称当时模型已经准确预见全部损失。\n')
 w('deepspeech2-10043在FULL的AWS1上训练16477秒，训练加权rho1.292；FO主要在RTX1上8250秒，加权rho1.122。FULL给了更好的相对辅助倍率，仍多训练4687秒。deepspeech2-10048在AWS/RTX间多次迁移、累计250秒预算后长期停留AWS1，直到只有一个作业剩余才恢复。直接额外checkpoint总计仅650秒；更大的风险是花完预算后禁止未来GPU纠偏，不应将所有损失都归给启动倒计时本身。\n')
 w('## 少量结构状态核验\n\n先固定三类条件，按时间取每种方法最早匹配状态；“全到达后至少10个作业排队且某节点辅助资源空闲超过25%”两种方法均未命中，未人为替换成更有利状态。其余四个早期状态如下。局部剩余预算上界是单作业物理响应诊断，未校验是否属于最终网格及bootstrap资格，不能作为漏选有效候选的证明。\n\n|方法|结构条件|模拟秒|已选任务|等待任务|CPU使用 AWS/DGX/RTX|\n|---|---|---:|---:|---:|---|')
 for x in states:w(f'|{x["method"]}|{x["criterion"]}|{x["time"]}|{len(x["selected"])}|{len(x["waiting"])}|'+ '/'.join(f(x['node_totals'][n]['cpu']) for n in ('aws:0','dgx-ext:0','rtx:0'))+'|')
 w('\n另对两种方法各取最早的“ready为空、仍有多个作业、DeepSpeech预算耗尽、其他分配固定时存在更高当前GPU模型×rho的比例候选”状态。这是为解释已观察损失的事后结构诊断，不是预注册性能实验。\n')
 for x in late:
  best=x['alternatives'][0];w(f'- {x["method"]} t={x["time"]}、{x["job"]}，当前{x["current"]["config"]["config_id"]}，累计启动{x["delay_ledger"]["delay_charged"]}秒；其余资源固定，比例点{best["config"]}在节点容量内，当前Sia归一化Goodput×rho代理为现配置的{best["proxy_ratio_to_current"]:.3f}倍，但生命周期预算阻止切换。不是实际ETA或保证JCT收益，详见late_budget_states.json。')
 if not late:w('该严格物理局部条件没有命中；不能据此推断重新交换多个peer资源也没有机会。')
 w('\n上述两个晚期状态的当前点、替代点，其分子和分母坐标均在原始矩阵域内，替代配置均有当前direct_profile；已独立核对节点余量容纳比例配置。这说明恢复受限确实可能挡住数据支持的物理方案，而非只有裁剪制造的空想收益。该归一化GPU模型代理仍不能替代真实batch/梯度下的物理ETA，也不是完整训练的JCT证明。\n')
 w('逐候选物理ETA、atomic batch与梯度参数的完整可重建状态未写入本轮日志，只保留候选计数/hash；因此没有伪造相同状态的FULL、FO最优目标差，更没有把同一时刻两条已分叉轨迹当作同状态。\n')
 w('## 已掌握的适用场景及尚缺的证据\n\n|问题|已有场景和结果|能支持什么|不能支持什么|\n|---|---|---|---|\n|CPU差异化|6 CIFAR+6 BERT、4GPU，三种到达下CPU对比例平均JCT改善约5.18%–8.11%；80混合对比例12.44%|敏感receiver与低敏感donor共驻、合法配额点能转为JCT收益|CPU越紧FULL就必然优于FO|\n|RAM差异化|同12任务场景MEM对比例约9.30%–10.95%；80混合10.21%|RAM限制边际速度且能从低敏感peer重分配时有效|所有应用混合都有效；早期CF/DS粗网格和部分更密网格收益很小或退化|\n|双资源|80场景FULL对比例22.71%、对CPU11.73%、对MEM13.92%|双轴开放在此模型中确有增量，交互约额外5.44秒/作业，接近相加|双资源与GPU联合选择必然显著超过串行两轴优化|\n|联合超过串行|12任务COMPACT/SPREAD为6.18%/8.53%，80为1.09%|存在条件收益|稳定普遍显著；SPREAD和80增量都有高影响作业|\n')
 w('历史12任务是6CIFAR+6BERT、AWS/DGX各2GPU；80任务是48CIFAR+16DeepSpeech+16BERT、AWS4/DGX8/RTX8。每GPU资源供给同为3CPU/62.5GB；不能把差异直接说成80作业CPU更充足。BERT比例从50%降到20%，新增占大部分训练时间的DeepSpeech，以及更宽节点、第三种GPU类型、不同到达和长度共同改变了竞争。跨场景没有单因子控制，尚不能给这些变化分配因果份额。\n')
 w('已有12任务报告：[到达压力18格]('+str((R/'research_runs/sia_synergy/cc_arrival_pressure18_20261030_run1/REPORT.md').as_posix())+')；[干净FO消融]('+str((R/'research_notes/synergy/CC_CLEAN_ABLATION_AB_C_20261030_FINAL.md').as_posix())+')。BERT平坦、应用映射、固定phi、常值边界延拓均是合成假设，资源感知有效性不等于实测画像已校准。\n')
 w('## 问题优先级与最小后续方案\n\n|级别|问题及证据|最小后续处理建议|\n|---|---|---|\n|High|FULL减少排队却让DeepSpeech多训练15164秒；倒数ETA目标与全程JCT不等价|先记录同状态GPU/配额交换的全服务集合预测得失和执行误差，区分真实资源取舍与模型误判；不直接调权重|\n|High|bert-10009与10019出现相反的快慢GPU路径；250秒预算禁止纠偏，尾部恢复只剩一作业才开放|先用已记录慢配置检查“未完成多个任务但有空闲资源”的恢复窗口，设计保留checkpoint和有限切换开销的隔离诊断；本轮不修改预算或恢复规则|\n|High|FULL/FO optional入选均0，联合优化实际没有额外任务选择|保留FIFO防饥饿与持续有效服务保护，先隔离评估未知ETA零bootstrap/次级启动偏好对自由任务选择的限制；不简单删L2|\n|Medium|CPU/内存互补已经让FO受益，现有门槛没有量化联合对串行的机会缺口|新增独立同状态机会门槛，比较最优FO固定GPU后的上界与合法联合方案，而非只比较比例配置|\n|Medium|候选物理ETA只保留hash，无法精确重建当时评分与替代配置|后续只增加可重建候选/模型状态日志，不改算法，再做少量同状态审计|\n|Medium|大量真实交换依赖E-legacy裁剪|把完全域内、分子越界、分母越界收益分别报告；不靠修改phi/曲线制造JO优势|\n')
 w('若隔离检验生命周期预算与多作业尾部恢复，FULL和FO必须使用相同修复，而不能只给FULL更宽松的恢复权。本轮只提出这一方案，不增加恢复次数或改变启动预算。\n')
 w('下一轮应先完成静态门槛，不立即扩大训练矩阵：设一个“FO已能解决”的负对照，再分别寻找CPU主导、RAM主导、双资源耦合的配置排序反转见证。每个见证必须满足相同知识、服务承诺、checkpoint和物理预算；FO固定GPU意图后已优化辅助配额的结果仍被合法联合方案超过，计入全部donor损失与重配置成本，且机会有持续时间支撑。找不到就报告该负载不适合识别额外联合价值，不把单任务加CPU可提速当作通过。\n')
 w('门槛确认后才预注册持续竞争负载和完整训练比较。核心保留Sia比例、Sia辅助优化、共同控制器FO、FULL及CPU/MEM/比例消融；同一负载、模型、算法版本、资源预算。先对新观测到的路径错误和恢复限制做单因子隔离，不同时换数据又改目标后宣称全部来自联合资源。实际JCT允许持平或退化，不按JCT挑模板、种子或配额。\n')
 w('## 证据与可复核性\n')
 w('analyze.py顺序读取FO/FULL封存gzip，生成逐作业分解、完整配置/配额变化轨迹和四个先定条件状态。budget_probe.py只做两个事后结构状态的物理可行性与归一化GPU模型代理核对，没有构造Cluster或推进Job。全部329份实际源码hash与原冻结清单一致；原始输入与两份逐轮日志hash与最终完整性清单一致。provenance.json列出脚本/输入hash。\n')
 w('validation.json核对80份逐作业差值、互斥时间、总训练时间、302次FO GPU意图、6个结构状态、329份源码及5项封存输入hash，结果PASS；不是新增算法测试或新训练实验。\n')
 w('原报告和执行数据保持不变。仍未做的严格反事实包括：同状态完整FULL-vs-FO目标重求、修复启动预算后的完整JCT、策略差异的单因子因果份额；本报告不补造这些结果。此轮停止，不修改算法或恢复正式实验。\n')
 with (O/'REPORT.md').open('x',encoding='utf8') as f0:f0.write('\n'.join(a))
 paths=[p for p in O.iterdir() if p.is_file() and p.name!='diagnostic_hashes.json']
 hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest().upper() for p in paths}
 with (O/'diagnostic_hashes.json').open('x',encoding='utf8') as z:json.dump(hashes,z,ensure_ascii=False,indent=2)
 print('REPORT_READY',O/'REPORT.md',flush=True)
if __name__=='__main__':main()
