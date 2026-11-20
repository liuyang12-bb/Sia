"""Render new offline-preflight evidence, never edit historical artifacts."""
import hashlib,importlib.metadata,json,sys,time
from pathlib import Path
O=Path(__file__).resolve().parent;ROOT=O.parents[2];TREE=ROOT/'cc-joint-recovery-equivalent-v1'
def read(n):return json.loads((O/n).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def write(n,v):
    with (O/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
P=read('preregistration.json');S=read('reconciled_preflight.json');Q=read('conditional_opportunities.json');C=read('coverage.json')
archive=O.parent/'cc_full_fo_event_gate_experiment_20261101_run1/source_snapshot.zip'
assert sha(archive)==P['source_snapshot_sha256']
versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','cvxpy','cylp')}
write('environment.json',dict(python=sys.version,executable=sys.executable,versions=versions,simulator_imported=False,
    source_is_git_worktree=False,source_snapshot_path=str(archive),source_snapshot_sha256=sha(archive)))
lines=[]
def a(s=''):lines.append(s)
def table(headers,rows):
    a('|'+ '|'.join(headers)+'|');a('|'+'|'.join(['---']*len(headers))+'|')
    for r in rows:a('|'+'|'.join(str(v) for v in r)+'|')
a('# 12 作业混合场景：FO/FULL 离线机会预检')
a();a('**SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE**')
a();a('## 1. 结论与本轮边界')
a();a('结论：**候选覆盖、物理容量和条件性联合选择机会通过；值得开展这个冻结的 12 作业小型 FO/FULL 验证，但真实在线持续性仍未验收。** 不将它称为已经保证 FULL 大幅降低 JCT 的场景。下一步若获授权，应先检查短前缀的真实事件选择和后续执行，满足预注册机会条件后再读完整 JCT；本轮到此停止。')
a();a('最直接的机会是 GPU 与辅助资源被分隔：在同一参考状态下，真实 FO 已优化 CPU/RAM，但 AWS 的 4 张 GPU 被 BERT 占用、14 个 CPU 单位闲置；RTX 的 CPU 几乎用满，CIFAR 仍争用 CPU。FULL 改变 BERT/CIFAR/DeepSpeech2 的 GPU 配置，将闲置的 AWS CPU 用于 CIFAR。这个机会不是“FO 没有做辅助资源优化”。')
a();a('反面证据也保留：FULL 会牺牲 DeepSpeech2 的速度，恒定配置 ETA 总和变差。参考模型下的净进度优势不是净 JCT 优势；需要实际完成后的资源释放补偿受损作业。BERT 在当前模板中完全不敏感，不能冒称内存敏感任务。')
a();a('本轮没有调用 Job.step 或 Cluster.step，没有训练推进、完成事件或 JCT。只进行了全域候选枚举、6 次小型静态 MIP、4 次真实控制器的单次 optimize_policy 调用及 8 项离线核对。所有调用在独立诊断进程中进行；不修改调度源码或正式初始化。')
a();a('## 2. 冻结场景与完整清单')
a();a('选择规则在任何新求解前冻结：官方 workload-1.csv 中按原行序取前 6 个 CIFAR、前 4 个 DeepSpeech2、前 2 个 BERT。两波均按 CIFAR、DeepSpeech2、BERT、CIFAR、DeepSpeech2、CIFAR 排列。只把到达时间设为 0/600 秒；原始 GPU demand、batch、应用与完整训练工作量保留。新到达时序为受控合成设定，不是原 Saturn 到达序列。')
a();table(['序号','作业','新到达 s','原到达 s','原 GPU demand','原 batch','源 CSV 行'],[[i+1,r['name'],r['time'],r['original_arrival'],r['num_replicas'],r['batch_size'],r['source_line']] for i,r in enumerate(P['workload'])])
a();table(['节点','GPU','合成 CPU 单位','RAM GB','比例点 CPU/GPU、GB/GPU'],[['aws:0',4,18,125,'4.5、31.25'],['rtx:0',4,12,375,'3、93.75']])
a();a('容量与上一轮 6-CIFAR 场景完全相同，避免同时变动容量。每类型合法 GPU 数为 1/2/4，单节点；GPU demand 是原始任务声明，策略仍允许合法弹性配置。12 作业超过总计 8 GPU：若第二波到达时第一波仍存活，则至少 4 作业不能同时获得至少 1 GPU。该条件说明竞争设计，并不代替真实队列日志。')
a();a('保留 Medium alpha=0.5、固定 R0/E-legacy、60 秒周期、seed_profiles(1,8)、原资源网格及既有 breakpoint/residual 候选。配置保留 6000 轮上限，但本轮没有启动任何轮次。工作量为 CIFAR 100 epoch、DeepSpeech2 80 epoch、BERT 2 epoch，来自原应用声明，不为实验缩短。')
a();a('## 3. 响应模型与互补机制')
a();table(['数量','应用→模板','当前矩阵性质','场景角色'],[[6,'CIFAR→alexnet_1.py','CPU 增益明显；RAM 低于 62.5 源坐标时有显著惩罚，高 CPU 时仍有部分 RAM 增益','主要 CPU/RAM 接收方'],[4,'DeepSpeech2→deepspeech_1.py','源 CPU≥3 且 RAM≥62.5 后饱和；低于阈值仍会减速','较早饱和，可能互补；不是任意配额都可无损让出'],[2,'BERT→bert_1.py','9×5 单元全为 1','CPU/RAM 不敏感；仍有异构 GPU 配置差异']])
a();a('三种映射均为合成跨硬件迁移。矩阵数值来自 Synergy 源码；没有证据证明本场景中的真实 CIFAR/DeepSpeech2/BERT 在目标硬件上具有这些响应。当前映射也不包含独立、平滑的纯内存敏感模板，因此不能用本轮单独证明一般内存优化有效。')
a();a('令 M 为矩阵双线性插值函数（E-legacy 先把坐标截到数据域），每 GPU 配额为 c,m，则 rho=[M(c/phi_c,m/phi_m)/M(3/phi_c,62.5/phi_m)]^0.5。phi 来自固定 registry，不随本轮轨迹变化；CPU 源域 [1,24]、RAM 源域 [20.83,250]。比例配额随节点容量变化，不保证 rho=1。候选执行率为当前 Sia 模型预测的进度率 × rho；没有用 Synergy 的绝对 iteration time 替代 Sia。')
a();table(['应用/配置','phi_c','phi_m','比例 rho','仅增 CPU 可达 rho','仅增 RAM 可达 rho','联合点可达 rho'],[[app+'/'+r['config'],f"{r['phi_c']:.4f}",f"{r['phi_m']:.4f}",f"{r['proportional_rho']:.4f}",f"{r['cpu_only_best']:.4f}",f"{r['memory_only_best']:.4f}",f"{r['best_rho']:.4f}"] for app,rs in S['classification'].items() for r in rs if r['config'].endswith(':1')])
a();a('以上单任务最大值仅验证节点内候选可行，不等于多个作业可以同时取最大值。下面的联合解才计入整个服务集合的预算。覆盖原始明细见 coverage.json 和 reconciled_preflight.json。')
a();a('## 4. 数据域、边界与候选覆盖')
cov=S['coverage'];total=cov['full12_candidate_rows']
a();a(f"全部 12×6=72 作业/GPU 配置对有 R0 和正的离线参考预测率，缺失 0；每个配置比例点物理可行。计入重复作业后的候选记录 {total} 条，其中分子裁剪 {cov['numerator_clipped_rows']}（{100*cov['numerator_clipped_rows']/total:.2f}%），分母裁剪 {cov['denominator_clipped_rows']}（{100*cov['denominator_clipped_rows']/total:.2f}%），分子与分母都受支持 {cov['fully_supported_rows']}（{100*cov['fully_supported_rows']/total:.2f}%）。两类裁剪可重叠，不相加当作总越界率；这是候选级计数，不是运行时间加权裁剪率。")
a();a('分母越界只涉及两个 BERT 的 RTX:1:1：3/phi_c≈0.99155<1；比例点分子同样越界。BERT 矩阵恒为 1，数值结果仍为 rho=1，但仍记录越界。CIFAR/DeepSpeech2 的六种 GPU 配置参考分母均在域内。低资源候选仍大量分子越界，不得将 clipped donor 宣称为实测的无损让出方。')
a();a('两波在本离线参考模型中的应用、合法配置、phi、模板、剩余工作与预测率逐项相等，核验后复用同一六任务静态求解。没有把第二波算成独立实验重复，也没有模拟第二波的真实在线状态。')
a();a('## 5. 同状态 FO/FULL：真实调度入口的单次选择')
a();a('先按原初始化做真实冷启动调用：FO/FULL 都只启动 cifar10-0，使用 AWS 1 GPU、4.5 CPU、31.25 GB，目标为 0。seed_profiles(1,8) 只给吞吐画像，未观测 gradient 时 ETA 仍未知。这轮调用没有产生训练，不能冒称已经获得在线 profile 或混合共驻。')
a();a('另一个明确标记的参考状态：一次性为已到达的首波作业设置原应用 epoch 0、init_batch 对应的离线 gradient，随后通过真实 optimize_policy 调用。该状态只用于 disposable 离线对象，**不是获批的实验初始化，也未注入正式训练**。两种方法选择相同 5 作业、等待同一个 deepspeech2-9，故下面差异不由本次服务集合变化造成；实际完整执行的集合可能继续分叉。')
one=S['one_shot']['reference_E0'];a();table(['方法','作业','GPU配置','CPU/GPU','RAM/GPU GB','rho','预测进度率','初始延迟 s','分子/分母裁剪'],[[m,v['job'],v['config'],f"{v['cpu_per_gpu']:.4f}",f"{v['ram_per_gpu']:.4f}",f"{v['rho']:.4f}",f"{v['rate']:.4f}",v['startup'],f"{v['num_clipped']}/{v['den_clipped']}"] for m in ('FO','FULL') for v in one[m]['selected']])
a();table(['方法/节点','GPU已用','CPU已用/容量','RAM已用/容量'],[[m+'/'+n,v['gpus'],f"{v['cpu']:.4f}/"+('18' if n=='aws:0' else '12'),f"{v['ram_gb']:.4f}/"+('125' if n=='aws:0' else '375')] for m in ('FO','FULL') for n,v in one[m]['nodes'].items()])
a();a('机制：FO 已把 BERT 的辅助资源压到下限，但 BERT 的 AWS 4-GPU 意图被冻结；AWS 空闲的 14 CPU 无法跨节点交给 RTX CIFAR。FULL 把 BERT 改到 RTX 2-GPU，把 cifar10-0 放到 AWS 1-GPU/17 CPU，并将 DeepSpeech2-8 转到 AWS 1-GPU。FULL 不增加硬件：AWS 的 CPU/RAM 恰好用尽，剩余 GPU 也不能绕过辅助资源约束。FO 后续合法事件仍可能重选 GPU，不能声称永远被固定。')
a();a(f"真实选择的共同主目标（sum 1/ETA）提高 {one['objective_gain_pct']:.4f}%。保持所选 bundle 与预测率不变并扣除实际 startup，360 秒净归一化工作进度提高 {one['held_bundle_conditional_windows']['360']['relative_pct']:.4f}%，600 秒提高 {one['held_bundle_conditional_windows']['600']['relative_pct']:.4f}%。这不是 JCT 改善率。")
a();table(['作业','600秒完成工作比例差 FULL−FO'],[[k,f'{v:+.8f}'] for k,v in one['held_bundle_conditional_windows']['600']['per_job'].items()])
w=one['held_bundle_conditional_windows']['600']
a();a(f"600 秒接收方增加 {w['receivers']:.8f}，让出方损失 {w['donor_loss']:.8f}，净值 {w['net']:.8f} 个任务完整工作量的等价份额。原始进度单位跨应用不可直接相加，故这里只加各任务的完成比例。未把 donor 损失遗漏。")
a();a(f"重要反证：如果永远保持当前配置/当前速率，五个已服务作业 ETA 总和 FO={one['constant_rate_sum_eta']['FO']:.2f}s、FULL={one['constant_rate_sum_eta']['FULL']:.2f}s，变差；其中 DeepSpeech2-8 从约 18080s 延长到 50989s。预计第一个释放仅从 {one['earliest_constant_rate_completion']['FO']:.2f}s 到 {one['earliest_constant_rate_completion']['FULL']:.2f}s。这个恒定速率外推不是实际 JCT，但说明“后续资源释放能否补偿长作业”仍是必要验收，不能由主目标或净进度替代。")
a();a('## 6. 持续性预检：条件通过，实际执行尚待验证')
a();a('预注册了两个参考状态 E0、E25。E25 使用 floor(0.25×max_epochs)：CIFAR epoch25、DeepSpeech2 epoch20、BERT epoch0（BERT 总共2 epoch）。它们不是从一次运行连续采样的两个时刻，不能把它们之间称为实际持续收益。')
a();a('对同一固定六任务服务集合，FO 先用比例点选 GPU，再固定 GPU 优化 CPU/RAM；FULL 联合求解。使用相同节点约束、已存在候选、当前算法主目标。每个静态 MIP 均 status=0、gap=0，独立容量核对通过；只复现主资源子问题，不宣称复现控制器的 FIFO、历史锁、恢复预算和全部并列解规则。')
a();table(['参考状态','FULL目标提高','360秒净归一化进度提高','600秒净归一化进度提高','恒定配置 ETA 总和变化'],[[s,f"{Q[s]['objective_gain_pct']:.2f}%",f"{Q[s]['windows']['360']['relative_to_fo_pct']:.2f}%",f"{Q[s]['windows']['600']['relative_to_fo_pct']:.2f}%",f"恶化 {-Q[s]['static_sum_eta_gain_pct']:.2f}%"] for s in ('E0','E25')])
a();a('两个保持窗口都没有预测完成，因此窗口中没有凭空重复利用“已释放资源”；初始 startup 使用 CF50s、DS25s、BERT120s。这里是从空状态初始分配，不是一次旧配置→新配置的迁移，不能声称已经核验真实运行中的重配置成本、锁或恢复预算。模型在600秒内保持不变是条件，非由日志验证的事实。')
a();table(['验收项','本轮状态'],[['12作业/72配置/原工作量/同容量','通过'],['FO已优化CPU/RAM后仍有联合GPU选择差异','参考同状态通过，真实控制器单次选择证据'],['净收益计算包含受损任务','通过；净进度正，但恒定ETA总和负面'],['360/600秒条件性保持收益','通过（静态模型条件），不是实际执行'],['全部关键点都在源矩阵内','未通过：保留并披露E-legacy裁剪'],['冷启动到混合共驻的可达性','未验证'],['GPU差异在真实事件后持续执行','未验证'],['提前完成/资源释放补偿长任务，降低完整JCT','未验证，未运行训练']])
a();a('推荐仅进入这个冻结场景的**小型机制验证**，不扩大负载或挑选更有利模板。下一轮若授权运行：先沿真实冷启动前缀，在 ARRIVAL/COMPLETION/FIRST_VALID_CURRENT_MODEL 等合法事件做同状态 FO/FULL 比较，检查相同服务集合时的几何差异与全体损益，再看该差异是否持续至少6个周期且至少2个周期有真实训练进度。不能要求每周期都重新选出同一差异，也不能为了过门槛额外 seed gradient。若未触发，保留结果并停止，不改参数。')
a();a('只有实际机会门槛成立才评价完整12/12 JCT；两策略统一60秒/6000轮，保留删失及损失作业。重点应是首次释放提前量、释放后DeepSpeech2是否恢复、全部任务JCT之和，不能只解释CIFAR加速。该建议不构成本轮启动训练的操作。')
a();a('## 7. 保护、验证与复算入口')
a();a('源树是完整源码快照，不是 Git 工作树，git rev-parse/status 返回“not a git repository”；因此不虚构分支/HEAD。版本身份采用上一轮 source_snapshot.zip 与344个逐文件SHA-256，前后完全一致；官方源和历史结果没有写入。没有创建新策略、改目标、改响应、增候选、删工作量或调参。')
a();a(f"预注册 SHA-256：`{sha(O/'preregistration.json')}`。源码 ZIP：`{sha(archive)}`。R0 registry：`{sha(TREE/'inputs/registry.json')}`。")
a();table(['输入','SHA-256'],[[Path(k).name,v] for k,v in P['input_hashes'].items()])
a();a('初始资源预检：可用内存3.173GiB、磁盘7.571GiB，高于2/5GiB门槛。后续独立静态/单次分配进程也调用同一gate；没有完整实验进程。静态求解30秒、mip_rel_gap=0仅是离线审计求解器，不替换真实策略求解参数。')
a();a('8项独立离线核对全部通过：来源与输入哈希、原始工作量字段/到达变换、72配置及比例容量、从源矩阵独立复算rho、10个静态或单次分配的容量、FO GPU冻结、单次调用零时间推进/共同执行契约、receiver−donor算术。没有以这8项替代完整训练回归。')
a();a('保留的诊断失败：第一次直接optimize时Job尚无空训练片段账本，报joint_step_segments缺失；只在one_shot_v2.py的一次性诊断对象设空列表，未执行Job.step、未改源代码。summarize第一版误读取query不存在的phi字段，随后改从固定registry读取。原脚本、失败日志均保留。测试首轮只有文件未关闭ResourceWarning，第二轮使用with读文件后8项无警告通过。')
a();a('源码证据（相对上述源树）：core80.py:116–124（FO/FULL安装与共同恢复）；clean_ablation.py:75–96（FO先比例GPU提案，再固定GPU优化辅助资源）、:172–191（共同服务集合形成）、:319–331（checkpoint/事件/预算）；completion_continuity.py:18–22、:226–243（倒数ETA主目标）；completion_time.py:31–62（真实Sia模型与batch预测）、:112–117（冷启动未知ETA）；joint_profile.py:179–199、:255–276（插值、边界和一次rho）；r0_liveness.py:24–56（固定registry和模板核验）。')
a();a('本目录可复算脚本：freeze.py（输入冻结）、precheck.py（全域/静态）、one_shot_v2.py（--method FO/FULL --state cold/reference_E0，只分配不训练）、summarize.py、verify.py。脚本拒绝覆盖已存在结果；如需重新执行，应复制到全新的诊断输出目录，不删除已有文件。本轮真实分配四次均有效，selected、quota、rho、节点使用量保存在 oneshot_v2_*.json。')
a();a('完整结果入口：reconciled_preflight.json、conditional_opportunities.json、coverage.json、reference_models.json、offline_checks_v2.log、artifact_hashes.json。无平均JCT、P99、makespan结果可供报告。')
with (O/'REPORT.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
write('COMPLETION_RECEIPT.json',dict(status='OFFLINE_PREFLIGHT_COMPLETE_CONDITIONAL_OPPORTUNITY',
    training_steps=0,cluster_steps=0,static_solves=6,allocation_only_calls=4,offline_tests_passed=8,
    source_files_unchanged=all(sha(TREE/n)==h for n,h in P['source_hashes'].items()),
    inputs_unchanged=all(sha(n)==h for n,h in P['input_hashes'].items()),
    online_persistence_verified=False,jct_claim=None,authorized_next_action='STOP_AND_REPORT_NO_TRAINING_AUTHORIZED',
    recommendation='Worth a bounded actual-prefix opportunity check before evaluating this frozen 12-job pair; conditional progress evidence, not JCT guarantee.',
    report=str(O/'REPORT.md'),completed_unix=time.time()))
write('artifact_hashes.json',{str(p.relative_to(O)):sha(p) for p in sorted(O.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'})
print(O/'REPORT.md')
