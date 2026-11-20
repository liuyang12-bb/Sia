# Stage2B 离线覆盖与 Neutral 小型活性验收

## 1 结论与边界

Stage2B 的 R0 离线参考域闭合，独立 Neutral L2 四单元已实现并通过小型机制测试。最终开发验收为 **88/88：原有 71 项原文保留，新增 17 项**。原始 Sia 一作业三轮状态回归与 Stage1.1、历史保存状态一致。旧 B3-PROP 全空吸收态测试仍通过，未把历史模式偷偷改为 L2。

这不是 20-job 准入或性能胜负结论。新模式显式限制最多三个同时活动作业；没有运行 20-job、6000 轮、九 cohort、160-job、参数搜索或 Medium 性能实验。三作业 fixture 在固定 30 轮中有未完成作业，未插补其 JCT，未延长窗口。它检验首次/重复服务与时间、容量语义，不宣称证明所有真实 workload 最终完成。

**重要数据结论：E-strict 下，当前容量可行的 50 个 proportional 配置全部因分子越界而不可查询；非 Neutral 四单元不能直接采用该方案。** E-legacy/E-linear 均只作离线计算，未接入执行。新模式执行 rho 严格为 1。

## 2 隔离与可重建版本

- 基线：`216573cb26f3136520ce9d4f609623b4c2c75df1`。
- 新分支：`exp/b3-stage2b-neutral`。
- 新工作树：`D:\科研启航\Sia\sia-joint-stage2b-neutral`。
- 本地 commit：`bd6105a37c6662336236a135eb7ad610ca6e16de`，未推送。
- 新产物：[stage2b_neutral](D:/科研启航/Sia/research_runs/sia_synergy/stage2b_neutral)。

初始快照保存 787 个实际文件字节、完整 bundle、空 tracked/index patch 和依赖锁；新树逐文件恢复验证一致。最终源码快照为 791 文件。3432 个受保护文件复查无变化，包括官方仓库、Synergy、Stage1.1、medium 冻结树、历史报告与 2×2 run1/run2。787 个基线源码文件在新树中也全部字节不变。

commit 仅增加四文件、762 行，没有改历史生产模块或旧测试。Windows 换行/stat-only 状态仅在字节哈希和 canonical diff 均一致后，按明确文件名单刷新索引；没有 reset、clean、add . 或 add -A。

| 新文件（sia-simulator 下） | 职责 |
|---|---|
| [joint_prototype/neutral_liveness.py](D:/科研启航/Sia/sia-joint-stage2b-neutral/sia-simulator/joint_prototype/neutral_liveness.py) | L2 状态事务、三资源 MIP、四单元过滤、真实执行观测 hook |
| [joint_prototype/stage2b_neutral_config.json](D:/科研启航/Sia/sia-joint-stage2b-neutral/sia-simulator/joint_prototype/stage2b_neutral_config.json) | Neutral 范围、Q/D_bar、目标/求解参数、预先固定上限 |
| [stage2b_preflight.py](D:/科研启航/Sia/sia-joint-stage2b-neutral/sia-simulator/stage2b_preflight.py) | 只读 R0 registry、边界/容量/映射/延迟审计 |
| [tests/test_stage2b_neutral.py](D:/科研启航/Sia/sia-joint-stage2b-neutral/sia-simulator/tests/test_stage2b_neutral.py) | 17 个新增测试，1–3 作业 fixture；四作业输入仅用于“不启动训练即拒绝”的范围检查 |

## 3 G0 离线数据

最终数据目录为 [g0_verified](D:/科研启航/Sia/research_runs/sia_synergy/stage2b_neutral/g0_verified)。`g0_initial` 和 `g0_final` 是保留的中间预检，不替换历史数据。离线脚本不构造 Job/Cluster、不拟合在线性能、不运行训练；只读取 Application 元数据、CSV 和源矩阵。

### 3.1 R0 固定参考

域为五应用×AWS 1/2/4 GPU、DGX-ext 和 RTX 1/2/4/8 GPU，单节点 DP，共 55 配置。完整域保留即使某配置最终受 CPU 排除，不能用已选列缩小归一化分母域。

每应用取所有配置直接记录的 microbatch 交集最小值；相同几何优先 placements，缺失才取 scalability；同源重复记录取 step_time 中位数。每项记录文件、SHA-256、CSV 物理行号（表头为第 1 行）、全部参与行、batch、聚合方法与域 hash。

| 应用 | b_ref | phi_c 范围 | phi_m 范围 |
|---|---:|---:|---:|
| BERT | 4 | 1–4 | 1–2 |
| CIFAR10 | 32 | 1–3.188759 | 1–1.785710 |
| DeepSpeech2 | 10 | 1–2.267877 | 1–1.505947 |
| ImageNet | 20 | 1–2.227376 | 1–1.492440 |
| YOLOv3 | 4 | 1–2.740012 | 1–1.655298 |

计算为 `T0=GPU*b_ref/median(step_time)`、`r0=(T0/GPU)/min_D(T0/GPU)`，`phi_c=clip(r0,.5,4)`、`phi_m=clip(sqrt(r0),.5,2)`。55/55 直接覆盖、零缺失、无插值；phi=1 是计算结果而非补值。五应用的最慢 per-GPU 参考都是 RTX 8-GPU 几何；该点仍属于完整参考域，但受当前 6 CPU 的 floor 限制，不能整配置执行。

`registry.json` 与 `source_support.json` 提供完整复算路径。没有使用任何策略轨迹、未来 epoch、完成时间或获胜策略的画像；CPU 需求按吞吐、RAM 按平方根缩放仍是 synthetic cross-hardware transfer，不是测得的联合需求。

### 3.2 全域与作业覆盖

读取官方 workload-1/5/10 各 160 行，共 480 个真实记录，只做映射和静态可行性预检，不训练这些作业。映射保持 CIFAR10→alexnet、DeepSpeech2→deepspeech、ImageNet→res50、BERT→bert、YOLOv3→mobilenet；每行记录 trace hash、源行号和原始字段。训练静态域按原 adaptive DP/batch-tuning 语义核对 `GPU*min_local_bsz <= max_batch_size`，不把 workload demand 错当 rigid 限制，也不把在线尚未画像上限当离线域。

PMP、新应用、额外多节点不在这份离线 phi 覆盖声明内；gang 单元测试使用人工几何和 Neutral，不借它声称多节点响应数据已经覆盖。

### 3.3 三种边界模型的离线比较

固定 reference=(3 CPU/GPU,62.5 GB/GPU)，分子 `n=(CPU/phi_c,RAM/phi_m)`，分母 `d=(3/phi_c,62.5/phi_m)`。离线 rho=`sqrt(R(n)/R(d))`，没有改变 alpha=0.5。

- E-legacy：原双线性插值和坐标 clipping。
- E-linear：同一 clip 点的响应乘 `min(1,c)*min(1,m/20.83)`；分子分母都使用同一延拓。
- E-strict：任一坐标超出 CPU[1,24]、RAM[20.83,250] 则不可定义；分母非法导致整配置无法归一化。若 proportional 基准非法，优化候选也不能证明 progress floor。

| 统计层级 | 配置数 | 配额点 | E-legacy 已定义 / 过 floor | E-linear 已定义 / 过 floor | E-strict 已定义 / 过 floor |
|---|---:|---:|---:|---:|---:|
| 完整原始网格 | 55 | 2200 | 2200 / 1761 | 2200 / 1798 | 1380 / 168 |
| 容量可行 | 50 | 680 | 680 / 493 | 680 / 518 | 256 / 0 |
| 容量可行比例点 | 50 | 50 | 50 / 50 | 50 / 50 | 0 / 0 |

| 越界类型 | 全网格：点 / 配置 | 容量可行：点 / 配置 | 可行比例点：点 / 配置 |
|---|---:|---:|---:|
| 分子 CPU 下界 | 375 / 50 | 330 / 50 | 50 / 50 |
| 分子 RAM 下界 | 400 / 50 | 140 / 50 | 0 / 0 |
| 分母 CPU 下界 | 240 / 6 | 103 / 6 | 6 / 6 |
| 分母 RAM 下界 | 0 / 0 | 0 / 0 | 0 / 0 |

所有上界越界为零。CPU/RAM 越界计数可能重叠，不相加当作唯一失败总数。12 个官方源矩阵的 min/max、shape/hash 保存为 `template_summary.json`；逐点结果在 `boundary_queries.json`。

因此全部 480 个映射作业在 E-strict 下失去可执行的 proportional 配置；不能推进该模型的非 Neutral 四单元。E-linear/legacy 的“可定义”也不代表实测有效。本轮没有选择并部署其中任何非 Neutral 模型。

## 4 延迟上界与状态机保证

核查 [Job.reallocate/migrate](D:/科研启航/Sia/sia-joint-stage2b-neutral/sia-simulator/job.py:553)、[Cluster.optimize_policy](D:/科研启航/Sia/sia-joint-stage2b-neutral/sia-simulator/cluster.py:159)、[Application 延迟表](D:/科研启航/Sia/sia-joint-stage2b-neutral/sia-simulator/applications.py:62)及所有 rescale_time 赋值：初始 0；重分配或迁移重置为目标应用延迟；Job.step 先扣延迟。一次配置变更调用迁移或重分配之一，不叠加两次费用。epoch 不重置倒计时。

五应用各 GPU 类型延迟分别为：CIFAR10=50、DeepSpeech2=25、BERT=120、YOLOv3=80、ImageNet=250 秒。故 `D_bar=250`，`Q=60*ceil(250/60)=300` 秒。这里是实际执行延迟，不是修改 Sia 目标中的 restart penalty。

先于实现保存的 [L2_STATE_MACHINE_PROOF.md](D:/科研启航/Sia/research_runs/sia_synergy/stage2b_neutral/L2_STATE_MACHINE_PROOF.md)从状态转移推导：服务完整几何不变，延迟只付一次；非完成服务累计 Q 后下一边界退出，保守 `L=D_bar+Q+Δ=610`。排空禁止续签，现有承诺≤L退出；每个 ready 前驱签发一次即离开队首、退队尾不能再次越过目标作业。在有限 N、单独可行、正执行率、无永久求解失败等条件下，采用 `W_alloc=(N+1)L`；首次有效训练的保守离散观察界为 `Δ+(N+1)L+D_bar+Δ`，N=3 时 2810 秒。

8 个 0/250 秒延迟角点 fixture 检查真实状态机不变量及首次服务；这不是穷举所有到达轨迹。观察到首次训练后才计服务，checkpoint 不计。原先误要求三个最坏延迟作业在 30 轮内均完成首个 Q 的断言失败已保留：所需边界可达 1800 秒，而状态角点最后观察为 1740 秒。**没有延长窗口或降低 Q**；该测试限定为首次服务界，未满 Q 的数值保留。重复完整 Q 由另一个固定 16 轮 fixture 验证。

零率、不可行承诺、无单独可行候选与求解器未决均显式停止正常保证路径。不声称无限到达稳定性，不根据短测试推算未知完成时间。

## 5 独立 Neutral 四单元

共享父候选来自 Stage1.1 `_raw_candidates`，在过滤前归一化；direct/projected/unprofiled 来源不改。最后目标保持原 B3 的 p=-0.5、lambda_n=1.1、restart factor、round/clip。没有 allocation bonus；优先队列决定服务集合后再优化该效用。

| 单元 | GPU 集合 | CPU/RAM | placement / 准入 / 执行 |
|---|---|---|---|
| U-FP | GPU-only 单提案 | 比例点与原最低 floor | 四单元共同优化 placement、L2、rho=1 |
| U-FO | 同一提案规则 | 原离散网格含比例点 | 同上 |
| U-JP | 全部当前合法候选 | 比例点与 floor | 同上 |
| U-JO | 全部当前合法候选 | 原离散网格含比例点 | 同上 |

F 在单独 proportional 可行配置中最大化 Sia normalized Goodput×restart factor，并列按 config ID。它不是原 Sia MIP，也不读取 CPU/RAM 收益排名。承诺期间保持配置，服务到期重新提案。新 O 枚举不调用会删除 Neutral 比例点的 Pareto 剪枝，保证同状态 `FP⊆FO, FP⊆JP, FO⊆JO, JP⊆JO`；旧剪枝完全未改。

L2 按真实 arrival/source row/job key FIFO；新到先入队，已完成 Q 的作业按签发序号退到队尾。checkpoint 和服务承诺分别记录；每个被保护作业等式选中、保持完整几何与配额。队首不共存时排空；可共存则按 FIFO 逐个可行性准入，再固定服务向量优化效用。并列结果使用相同词典序末级规则，不改变主效用权重。

求解器 status=2 才视作不可行证明；超时、无可用解、伪造非整数/超配返回拒绝。Stage2B 采取保守的**显式停止**，不新增继续运行的 fallback；失败不提交分配或队列事务。历史安全 fallback 保留不动。任何成功结果均重新经过 Stage1.1 当前列、query、锁和三资源容量审计。

## 6 红绿证据与小场景结果

| 日志 | 结果 | 解释 |
|---|---|---|
| red.log / red_test_source.py | 11 项，6 failure、5 error | 5 项实际旧行为反例失败；其余是新增接口尚不存在，未冒称全是算法缺陷 |
| green1.log | 10/11 通过 | 三作业 fixture 的 1-GPU 节点与原 Sia 4/8 配置表不兼容；停止证据保留 |
| green2.log | 11/11 | 恢复原 4/8/8 GPU 拓扑；CPU、输入、Q 和 30 轮上限不改 |
| boundary_red.log | 15 项，2 failure | 捕获到期 F 提案未刷新；另为上文 30 轮未满 Q 的错误测试断言 |
| green_full.log | 86 项，1 failure | F 到期已修复，剩余有限观察窗断言；保留原输出 |
| final_tests.log | **88/88** | 原71＋新17；22.432 秒测试时间、26.649 秒墙钟；开始可用内存3.163 GiB |
| legacy_regression/comparison.json | 全等 | 官方 Sia、Stage1.1、新树及归档历史：一作业三轮 |

新增测试覆盖负效用非空/FIFO、服务重排、gang 排空、checkpoint 与普通服务区分、无候选、零率、服务中 solver failure、非法部分解、不可行承诺、到期重提案、四单元同状态包含、真实未画像 BERT、边界枚举和范围限制。旧 PMP 测试仍出现既有 autograd log RuntimeWarning，不将其隐藏为新收益。

### 6.1 状态机人工场景

- 负效用：所有列 utility=-100，旧真实求解器选择空；L2 在两作业中选真实先到的 z，不按名称选 a。只覆盖目标系数反例，没有伪造真实训练 Goodput。
- 重复服务：两作业一份资源、Q300、固定16轮；0秒first、300秒second、600秒first，验证队尾重排。
- gang：两个2-GPU人工节点、两个1-GPU作业、60秒到达的4-GPU双节点作业；固定12轮，触发 DRAIN，300秒 gang 获得完整放置。该场景不是离线 phi 多节点覆盖证明。
- 真实 checkpoint/故障：完整保留 config、nodes、quota；未知/超配结果和 timeout 不提交。故障是注入求解结果或执行观测，不改真实应用 Goodput 以制造成功。

### 6.2 真实未画像 BERT 完整链路

arrival=17，batch=384、demand=1；固定 step(60,60,60,1)。四新单元均在60秒首次分配，180秒首次有效训练，waiting=43、checkpoint=120、productive=1、progress=0.90430728135，restart=0。四次均经过真实 optimize_policy，没有 seed 或 mock 下一次调度。与历史 Sia 选何种 GPU 不要求相同；新服务承诺此时尚未到 Q，不允许为了新画像立即重配置。

### 6.3 真实三作业 30 轮

三个未 seed BERT，arrival=0/17/34，batch=384、demand=1；GPU4/8/8，每类型一个节点，人工竞争 fixture 各节点CPU=1、RAM=62.5。模板沿用旧测试的res18，Neutral下无响应收益；不是修改正式冻结集群或真实Saturn记录。所有单元相同输入，1800秒停止。

| 单元 | 完成 | 三作业 productive 秒 | 三作业 waiting 秒 | restart 总数 | checkpoint 总秒 | DRAIN 轮 |
|---|---:|---|---|---:|---:|---:|
| U-FP | 0/3 | 900 / 600 / 300 | 540 / 943 / 1286 | 4 | 780 | 25 |
| U-FO | 0/3 | 900 / 600 / 300 | 540 / 943 / 1286 | 4 | 780 | 25 |
| U-JP | 1/3 | 1620 / 1591 / 1620 | 60 / 43 / 26 | 0 | 360 | 0 |
| U-JO | 1/3 | 1620 / 1591 / 1620 | 60 / 43 / 26 | 0 | 360 | 0 |

FP/FO 最终 progress=5611.480249 / 2810.361347 / 271.292184；JP/JO=1464.977796 / 14774 / 3020.701883，bert-1 在1771秒完成。固定提案争用同类型时确实产生排空和等待；联合候选可选择其他GPU。这只是受限机制实例，不是平均JCT优越性结论；例如bert-0在联合模式下反而进度更少。

资源小时和逐作业重启、首服务时间、wasted GPU-seconds、完整每轮决策见 [mechanism_summary.json](D:/科研启航/Sia/research_runs/sia_synergy/stage2b_neutral/mechanism_summary.json)及 `final_tests_evidence/real_*.json`。本组FP/FO GPU-hours=0.716667、CPU-hours=0.716667、RAM-hours=14.928167；JP/JO分别为1.441944、1.441944、30.035703。所有实际执行/资源审计零违规、rho=1；完成后的29秒不计资源使用。较高资源小时不是自动更优。

## 7 复现与停止点

恢复说明为 [REBUILD.md](D:/科研启航/Sia/research_runs/sia_synergy/stage2b_neutral/REBUILD.md)。完整文件 hash 在 `repair_manifest.json`、最终证据索引在 `delivery_manifest.json`；依赖锁、base/新 bundle、实际字节 ZIP、patch 和官方矩阵快照全部保存。

| 产物 | SHA-256 |
|---|---|
| stage2b_source_snapshot.zip | BA8655A7D0E480BF137315E912DB9F25CCEF43FF16FED45377D371250D21F107 |
| stage2b.bundle | DF42DA234B7EB41BE084DC42A2B7367D66AF117E64D6EAF362043086DD97CDAC |
| stage2b.patch | C58300621B0D1966AE4360D046DEEF0E9AA79DB79EFE3326416C0F49114DBA8F |

本阶段只证明有限小场景的工程行为。后续仍需批准解除三作业范围限制、冻结单 cohort 观察/删失准则，以及选择非 Neutral 边界模型。E-strict已明确不能支持当前proportional域；不能自动补phi、删作业、改容量或改用E-linear训练。等待研究者审查，不自动进入20-job或Medium实验。
