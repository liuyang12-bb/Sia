# 论文机制形式化与版本核验

审阅日期：2026-11-01。只读核对源码与已有报告；未运行仿真、未修改调度算法。本文服务于初稿，不能把公式近似解释为最优性证明。

## 1. 推荐主版本及证据分层

建议论文以 `cc-joint-core80-full-lex-repair-v8` 的完成版80作业七方法实验作为主系统版本，简称 **CC-Joint（FULL）**。它的实际执行源码是329文件快照，而非一个裸Git HEAD；`source_snapshot.zip` SHA256为 `B0DCAEB4A5A205B6CA27893F3ACFE1F43C3F850CA8AF6E5BC79B15029518D6EB`。原报告为 `research_runs/sia_synergy/cc_core80_full_lex_repair_20261031_run5/REPORT.md`。这不是最新所有修复的混合体。

|证据|实现与用途|不能混同|
|---|---|---|
|80作业7方法完整配对|v8主实现；同源码、同数据、同响应|可用于主表；不能替换成最新恢复等效源码仍沿用旧数值|
|Clean A组20作业|同候选资格、原Sia式公平评分；切换有序/持续服务因素|A11不是主系统的完成优先目标；35.56%改善不能全部归因完成优先|
|Clean B组12作业|共同完成优先目标/有序持续服务，固定/联合GPU × 比例/优化辅助资源|与80作业配置和版本不同；属于机制佐证而非同一大矩阵|
|最新mixed12 FO/FULL|`cc-joint-recovery-equivalent-v1` 344文件，恢复许可按GPU几何闭包|最新0.336%均值改善、尾部退化必须保留；不得解释成v8所有数据的重现|

最新恢复等效版与v8在 `clean_ablation.py::shared_eligible` 有关键差异：v8:329–351只添加最佳单列；最新:345–370将该恢复GPU几何下所有当前合法、严格支付切换成本的辅助配额列闭包纳入，从而给FO比例提案相同恢复许可。该改变没有在v8的80作业结果中验证。

## 2. 问题定位与不得使用的表述

可写：Sia以异构、弹性GPU配置上的Goodput公平效用进行周期调度，不能直接等价为平均JCT最小化；在本研究受控负载中，服务中断、配置改变及等待累积存在改进空间。CC-Joint通过有序准入、持续有效服务和完成导向资源配置改善该情形。

不可写：Sia“不考虑重配置成本”“完全没有checkpoint保护”“普遍存在错误”。官方 `official-artifacts/sia-simulator/sia.py:720–747` 已计算重配置/迁移折损，`:769–798` 收集checkpoint不可抢占作业，`:1011–1029` 通过等式强制选中原GPU配置。

也不可写：本实现直接移植了Synergy-TUNE。FULL使用官方矩阵启发的响应Oracle及本项目有限列MIP，未调用官方TUNE递归分配流程。官方核心在 `research_sources/Synergy/simulator/resources/cluster.py:575` `allocate_synergy_tune`，`:589` `_tune`，`:698` `_reallocate_peer`，`:733` `_make_fair_share`；固定GPU需求与辅助资源回退是不同的算法。FO也不是官方Synergy或原始Sia。

## 3. 数据结构与候选域

令作业集合为 \(\mathcal J_t\)，节点为 \(n\in\mathcal N\)，容量 \((G_n,C_n,M_n)\)。一个列 \(k\in\mathcal K_j(t)\) 包含GPU类型 \(v\)、数量 \(g\)、节点数、节点GPU向量 \(g_{jkn}\)、每GPU CPU配额 \(c_{jk}\)、RAM配额 \(m_{jk}\)、对应batch/累积步数和Oracle版本。

GPU域沿用Sia合法配置与JobInfo当前min/max replica、scale-unit、rigid限制。`joint_integration.py::_raw_candidates:114–212` 调用Sia的 `_compute_goodputs_v2`，保留direct/projected/unprofiled来源；先按最小非零Goodput和scale-unit归一化。**主版本并非直接把这个归一化值当ETA速率**，而是在完成评分时重新用真实当前模型与batch计算progress/s。

候选生成：`joint_scheduler.py::enumerate_placements:134`，`build_joint_columns:312–383`，`build_proportional_columns:385–453`。所有单节点候选受物理GPU、CPU、RAM容量以及每GPU最低1 CPU/20.83GB限制。配置跨节点时需同一每GPU配额；但本轮主实验是每类型一个节点，`physical_domain.py:11–28` 限制为单节点合法GPU域，**不能宣称已完成大规模跨节点placement评估**。

主实验原网格CPU为{1,2,3,4,6,9,12,24}，实际安装时加8；RAM为{20.83,62.5,125,187.5,250}。此外 `breakpoint_study.py::resource_points:21–58` 加入源矩阵斜率变化点的反向映射，以及一个同类型潜在peer的容量余量点；peer来自完整固定离线应用registry而不是未来队列。正数且不低于floor，再由逐节点容量过滤。这个增强已在v8主版本存在，不能只写“固定8×5网格”。候选规模可达20余万列，非轻量闭式调度。

## 4. 统一响应与训练速率

R0 registry含5应用×11单节点GPU配置=55项，固定hash `78F676E7695CBE09AA9A3DB71F539FE2A2CEF1DBC243E57CC064E4E294DCF1C9`。参见 `r0_liveness.py:17–51`。映射：CIFAR→alexnet_1，DeepSpeech2→deepspeech_1，BERT→bert_1，ImageNet→res50_1，YOLOv3→mobilenet_1。**这些应用映射是合成迁移；BERT矩阵为常数1，不是内存敏感BERT实测结论。**

对应用a/config q，离线registry给出同一参考microbatch、placement-source优先中位step time的每GPU吞吐 \(T^0_{a,q}/g_q\)，以及该应用完整GPU域的最小每GPU吞吐 \(T^{\min}_a\)。令 \(r_{a,q}=(T^0_{a,q}/g_q)/T^{\min}_a\)，主registry中的 \(\phi_c=\operatorname{clip}(r_{a,q},0.5,4),\phi_m=\operatorname{clip}(\sqrt{r_{a,q}},0.5,2)\)。55项实查有4项BERT/DGX配置触及上界4/2，其余未被上界改变；这是需求缩放的合成上界，与查询坐标裁剪不同。这只规定合成CPU/RAM需求相对缩放；不向在线Sia性能模型灌入未来真实运行结果。registry记录完整source_rows/hash/line/microbatch/aggregation。

设 \(\widetilde M_a\) 是官方Synergy模板经边界裁剪后双线性插值的响应，则固定参考点3 CPU/62.5GB下：

\[
\rho_{j,q}(c,m)=\left[\frac{\widetilde M_{a(j)}(c/\phi_c(q),m/\phi_m(q))}{\widetilde M_{a(j)}(3/\phi_c(q),62.5/\phi_m(q))}\right]^{\alpha},\quad\alpha\in\{0,0.5\}.
\]

`joint_profile.py::_bilinear:178–204` 与 `query:252–340`。裁剪CPU至[1,24]，RAM至[20.83,250]。该行为不能称为目标硬件测量支持的外推；分母和分子裁剪都必须统计。主80场景比例点恰为3/62.5，因此rho=1；其他容量场景比例点不一定等于固定参考。CPU/RAM-only调整在执行模型里不附加迁移/加载开销，这是必须披露的假设。

实际Job训练片段：
\[
r^{\mathrm{exec}}_{j,q}=\frac{\mathrm{gain}}{t_{\mathrm{step}}+a\,(t_{\mathrm{step}}-t_{\mathrm{sync}})}\,(1-\mathrm{interference})\,\kappa_j\,\rho_{j,q}(c,m).
\]

`job.py:484–497` 先直接记录raw_rate_before_rho再乘一次rho；`:431–442` 先扣除checkpoint/restore倒计时，延迟不乘rho；`:504–549` 执行epoch/完成边界与独立残差日志。受原模拟器整数秒epoch取整影响，不能用全轮简单等式替代边界审计。

## 5. 完成导向目标

声明总工作量 \(W_j=\mathrm{get\_progress}(\mathrm{max\_epochs})\)，当前剩余 \(w_j(t)=\max(0,W_j-p_j(t))\)。`completion_time.py::declare_work:66–76` 拒绝未声明收敛工作量；不读取未来完成事件。`predict_job:29–63` 使用当前学习的Goodput模型、真实 `update_local_bsz` 选择batch/累积步数，得到

\[
\widehat r_{jq}^{\mathrm{raw}}=G_j(q,b_{jq},a_{jq})\,\kappa_j/b_j^{\mathrm{init}},\quad
\widehat T_{jk}=d_{jk}+\frac{w_j(t)}{\widehat r^{\mathrm{raw}}_{jq}\rho_{jq}(c_{jk},m_{jk})}.
\]

同GPU几何用剩余 `job.rescale_time`，更换GPU类型/数量/placement用目标应用真实startup代价，`score_columns:79–129`。未知性能时ETA未知、主权重0，**仅保留相同比例bootstrap点与真实锁**，不把冷启动占位Goodput1当测得训练率。PMP ETA单位当前未验证，保留合法候选但ETA未知；主实验应用属于已验证范围。

`completion_continuity.py:18–22,226–246` 定义 \(u_{jk}=1/\widehat T_{jk}\)，同轮除以最大权重 \(s_t>0\) 数值归一化。主问题为

\[
\max_{x}\sum_{j,k}\frac{u_{jk}}{s_t}x_{jk},\quad x_{jk}\in\{0,1\}.
\]

这是**当前速率恒定假设下的完成优先代理**，不是 \(\min\sum C_j\) 的等价转换；不预测未来到达、全局队列资源释放或donor未来恢复。长作业的较小倒数权重、逐轮估计误差和锁定均可能导致尾部退化。可以称“以降低JCT为目标的启发式完成优先评分”，不可写“最优平均JCT”。

## 6. 约束、有序准入与持续服务

每作业最多一个列；每节点三资源容量：

\[
\sum_{k\in\mathcal K_j}x_{jk}\le1;\quad
\sum_{j,k}g_{jkn}x_{jk}\le G_n;\quad
\sum_{j,k}g_{jkn}c_{jk}x_{jk}\le C_n;\quad
\sum_{j,k}g_{jkn}m_{jk}x_{jk}\le M_n.
\]

真实checkpoint锁 \(j\in L_t\) 仅保留原GPU类型/数量、完整节点placement、CPU/RAM配额的当前重查询列 \(k_j^{old}\)，并强制 \(x_{j,k_j^{old}}=1\)。对应 `joint_scheduler.py::audit:667–740`、`clean_ablation.py::shared_eligible:298–327`。锁不能用服务承诺代替：服务承诺保证选中，但在合法事件/预算下仍可变GPU几何和辅助配额。

设 \(H_t\) 为尚未完成的服务承诺集合。所有 \(j\in H_t\) 必须 \(\sum_kx_{jk}=1\)，持续到实际完成，不每300秒强制退场。启动顺序按真实arrival/source_row/job_id，`clean_ablation.py:25–30`。若队首h独立不可行，报错而非跳过；若 \(H_t\cup\{h\}\) 不可行，只服务 \(H_t\) 排空，不再给后续作业新承诺；否则必须加入h，其他已到达未承诺作业仍可由MIP选择。即有序队首保护不等于逐个严格FIFO独占，也不等于前置固定全部服务集合。

核心分支 `clean_ablation.py:172–198`。活动服务不能随目标变差被随意取消；可选作业的选择仍优化目标。对空承诺/无强制队首的消融模式另加 \(\sum x\ge1\)，见`:111–116`，只防全空，不提供个体无饥饿保证。

同一已启动历史的候选过滤独立于当前是否held，`:298–351`：切换几何需自上次几何变化后获得真正productive service、目标为direct/projected模型、发生arrival/completion/首次模型事件、累计启动费用不超过250秒。辅助配额变化不新增费用。v8没有每轮“不能低于比例速度”的progress-floor，允许合法donor降速；`completion_time.py:149–156`明确移除floor与旧fair目标，`exchange_netprogress.py::configure_scheduler:29`。

持续服务活性是**条件性**：每个作业存在单独合法配置、队列有限/每个作业前面的等待者有限、已承诺作业真实速率有正下界、有限声明工作和有限累计启动损失、求解成功且未来资源不被撤销，则held逐步完成、blocked head终能被接纳。不能声称面对任意故障、零速率或无限负载均保证有限等待。监测器 `completion_continuity.py:169–182` 在250+60秒无productive增量时报错，这是验证保护而非能让不满足假设的训练自动完成。

## 7. 词典序与有界收尾恢复

每次可行性检查只判断资源与强制集合；真正选择时：

1. 最大化完成优先主效用；
2. 保持主值±1e-9带内，最小化本轮真实startup总秒数，`bundle_adaptation.py::_secondary_tie:88–106`；
3. 按arrival/source_row/job_id对optional作业进行presence优先，再按稳定候选顺序最小整数rank，逐级固定，`clean_ablation.py:140–168`。

v8工程实现使用可证明等价的列消元、单位缩放、整数格点和预算界，不能解释成新增科学目标。仅最优且通过原始约束的解被提交；无证明/超时不是成功，失败日志保留。`numeric_repair.py::canonical_bundle:13` 与 `engineering_repair.py::repair_context:8` 是实际正式运行覆盖入口。

收尾恢复只在当前已到达活动作业仅剩1个、ready空/只有自身、无checkpoint、已有productive service、目标profile已知、真实切换后ETA严格小于当前最优ETA时，提供一次额外GPU恢复机会。额外预算 \(\max(0,B_j+d-250)\le250\)，最多一次；使用后GPU几何冻结，辅助配额仍可优化。`tail_recovery.py:12–84` 与Clean公共资格版本 `clean_ablation.py:328–351`。**它不是所有长作业的通用抢救机制，无法挽回竞争期间已经累积的慢速代价。**

## 8. 方法对照与真实伪代码

|方法|任务/GPU决定|辅助资源|服务规则|主评分|
|---|---|---|---|---|
|SIA_P|真正 `SiaPolicy.optimize` 与官方placement|比例|官方Sia，无CC/L2|原Goodput公平目标与restart折损|
|SIA_AUX|同官方完整GPU/placement意图冻结|自研固定集合CPU/RAM优化|同官方Sia|固定集合的旧ETA-saving，等价该集合最小ETA和；不是FULL倒数目标|
|FO|先在比例列上按CC目标决定集合/GPU，再冻结config_id|第二阶段优化辅助配额和placement|CC有序+持续+tail|同FULL倒数ETA|
|FULL|同一个列选择任务、GPU、配额、placement|CPU/RAM开放|CC有序+持续+tail|倒数ETA|
|PROP/CPU/MEM|同FULLGPU域、服务规则|比例/仅CPU自由/仅RAM自由|同FULL|同FULL完整域评分后过滤|

注意SIA_AUX调用 `completion_time.py::CompletionAuxBase:169–195` 与 `fixed_set_solve`；其辅助评分不是FULL倒数ETA，因此FULL vs SIA_AUX并非只差GPU选择的单因素消融。FO才是较干净的串行机制对照（但v8单列tail恢复边界后来已修正）。

```text
每轮t（只读当前已到达且未完成Job）:
  由Sia当前状态枚举合法GPU/config；生成placement和辅助资源列
  用固定R0/模板查询rho；从当前学习模型+实际batch计算ETA并评分
  用当前真实checkpoint、历史模型来源/事件/预算过滤列
  将完成承诺移出；新作业依真实arrival及稳定源行入队
  held = 尚未完成服务承诺；验证完整locked配置
  若队首单独无可行列: 显式失败
  若held+队首不可行: drain，仅held；否则held+队首强制，其他optional
  FULL: 联合词典序MIP（效用→真实startup→确定性presence/rank）
  FO: 同目标先在比例域求解，然后冻结服务集合与GPU config做辅助MIP
  对全结果独立重验容量、锁、当前候选/Oracle版本
  原子提交allocation与服务账本；任一失败不部分提交
  先消耗真实checkpoint/restore秒数，再以raw_rate×rho推进训练
  记录等待/延迟/训练互斥秒数、完成事件、资源小时、边界裁剪
```

## 9. 论文应披露的性能和外推限制

- 有序持续服务对原Sia缺陷的论证限定到受控负载，不能由一个队列时间分解推出普遍结论。
- 80作业FULL相对Sia均值下降29.620%，但P99/makespan分别恶化18.289%/14.499%；FULL相对FO仅改善1.086%，应报告，不以“普遍胜串行”作贡献。
- FULL选择开销累计7289.254秒、P99=90.770秒；Sia选择总15.407秒、P99=.091秒。**当前离散仿真没有把计算墙钟加入作业时间，P99超过60秒调度周期，不能直接主张在线可部署。**需要工程优化/更大调度周期的独立研究，但本轮不修改。
- 80作业七方法335项工程/语义测试、零原始日志审计违规，是正确性证据，不是响应迁移正确或性能普适性的证明。
- 业务标签、合成CPU单位与R0/phi/clip均非Saturn联合硬件测量。矩阵coverage完整不等于每个查询都在实测域内；需报告分子/分母裁剪。
- 主实验有profile seed(1,8)，所有方法相同，不能宣称纯无画像冷启动的主结果。
- 当前保留Sia异构、弹性、batch自适应和Goodput建模能力；具体实验只验证三应用和单节点候选。预测当前速率近似受epoch/梯度统计变化影响，未证明未来剩余速率常数。

## 10. 建议稿件机制命名

主方法用“CC-Joint：有序启动与持续有效服务驱动的异构多资源联合调度”。CC-TAIL可作为实现中的有界收尾扩展名称。两条贡献分开：

1. 面向资源竞争，使用真实完成/有效服务账本组织启动与连续执行，减少已有受控场景的反复等待与无效切换；通过清洁A消融检验，不宣称这是Sia未见过的所有思想。
2. 将Sia合法异构弹性GPU模型与Synergy启发的CPU/RAM响应纳入同一候选预算，进行完成优先的差异化配置；通过PROP/CPU/MEM/FULL及FO比较区分辅助优化与串行替代。

第三项可列“可复现实现与有审计的仿真证据”，不要另造全局JCT最优、零开销或普遍异构多节点理论保证。

