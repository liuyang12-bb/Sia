# Stage2A 活性机制与公平消融设计

状态：设计评审稿，尚未批准实施。

基线：`216573cb26f3136520ce9d4f609623b4c2c75df1`，工作树 `D:\科研启航\Sia\sia-joint-correctness-stage1p1`，分支 `exp/b3-correctness-stage1p1`。检查时 HEAD 一致、工作树干净。

## 1 推荐结论与研究边界

推荐组合为：**统一的队列准入与有效训练服务段保护、新命名的四单元候选消融、策略无关的离线吞吐参考 phi**。响应边界首选供评审的“下界线性延拓、上界饱和”显式合成模型；若不批准延拓，则采用严格支持域并停止不完整 cohort，不自动删作业或填 phi。

这不是给旧 B3-PROP 加一个补丁后保留旧 2×2 结果。新的准入会限制现有公平效用的自由度，固定 GPU 单元也不再等同于原 Sia MIP；因此新四单元命名为 `U-FP / U-FO / U-JP / U-JO`。历史 B1/B4/B3/B3-PROP、B5 和 SY adapted 保留为历史证据或外部基线，不混入新因子分析。

目标是识别：在**同一活性保障和执行模型**下，扩大 GPU 配置候选集、扩大 CPU/RAM 配额候选集分别产生什么效果。placement 在四单元中共同优化，不再把 placement 优化暗中捆绑到辅助资源因子。

本轮只新增本文；没有修改源码、创建工作树/commit、生成新 phi、运行拟合、测试或训练仿真。下述规则、数值推导和测试都是待批准设计。模拟证据仍属于 synthetic cross-hardware transfer，不是真实 Saturn 集群测量。

## 2 当前证据与必须保留的修复

| 事实 | 源码或现有证据 | 设计含义 |
|---|---|---|
| B3/B3-PROP 对有效 Goodput 作负指数效用变换 | [joint_integration.py:394](D:/科研启航/Sia/sia-joint-correctness-stage1p1/sia-simulator/joint_prototype/joint_integration.py:394) | 不能把“有合法列”等同于“会分配” |
| 未锁定作业只有至多一列约束，空集可最优 | [joint_scheduler.py:514](D:/科研启航/Sia/sia-joint-correctness-stage1p1/sia-simulator/joint_prototype/joint_scheduler.py:514)、[solve:580](D:/科研启航/Sia/sia-joint-correctness-stage1p1/sia-simulator/joint_prototype/joint_scheduler.py:580) | 活性必须有独立语义，不能依赖效用刚好为正 |
| Stage1.1 已修复真实 checkpoint 等式保护、解锁资格、共享 replica 边界与当前轮 fallback | [Stage1.1 报告](D:/科研启航/Sia/research_notes/synergy/B3_CORRECTNESS_STAGE1P1_REPORT.md) | 新活性层不得回退到信任 candidate_kind 或陈旧列 |
| 动态 phi 取当前候选的最小 per-GPU Goodput 作分母 | [joint_integration.py:186](D:/科研启航/Sia/sia-joint-correctness-stage1p1/sia-simulator/joint_prototype/joint_integration.py:186) | 候选范围、在线画像和策略轨迹会改变响应；仅冻结指数不足够 |
| 坐标 clip 同时作用于分子和分母 | [joint_profile.py:179](D:/科研启航/Sia/sia-joint-correctness-stage1p1/sia-simulator/joint_prototype/joint_profile.py:179)、[query:239](D:/科研启航/Sia/sia-joint-correctness-stage1p1/sia-simulator/joint_prototype/joint_profile.py:239) | 两种越界必须单独统计、单独定义 |
| Job.step 先扣 rescale_time，再记录 raw rate 并乘 rho | [job.py:419](D:/科研启航/Sia/sia-joint-correctness-stage1p1/sia-simulator/job.py:419) | 服务计时必须使用 productive segment，不能用含 checkpoint 的 attained_service |
| 旧 B3-PROP 6000 轮全部空分配；旧 B3−B4 总 JCT 节省约 90.08% 来自 deepspeech2-159 | [旧 2×2 报告第 7、9 节](D:/科研启航/Sia/research_notes/synergy/B3_W10_LATE_2X2_ABLATION_REPORT.md) | 旧矩阵不能识别完整主效应/交互；长尾贡献必须预注册报告 |

现有 71 项通过是 Stage1.1 的历史验收结果，不是新机制已通过测试。旧报告称 B4−B1、B3−B4 为简单效应时，只能理解为历史策略间的描述性差异；准入/目标不同使其不构成干净的因果因子效应。

## 3 统一记号和活性保证层级

作业 j 的到达为 a_j；当前状态为 s_j(t)。GPU 配置 c 包含类型、GPU 数量、节点数和 scale-unit 合法性；节点放置为 π；每 GPU CPU/RAM 配额为 q=(q_c,q_m)。列 k=(j,c,π,q)。x_k∈{0,1}，z_j=Σ_{k属于j}x_k。

`g_jc(t)` 是 Sia 当前 direct/projected/unprofiled-fallback 候选值；`s_jc(t)` 是按**同一完整父候选集**归一化后的值。调度预测值与执行率不是同一个量：

```text
predicted_effective = s_jc(t) * rho(j,c,q)
executed_rate       = raw_rate_before_rho(Job.step, current batch/epoch) * rho(j,c,q)
```

继续使用 Stage1.1 B3 的效用公式作为最后一级目标，而不是声称移植了原 Sia 全部 MIP：

```text
v_jk = max(s_jc * rho_jk * restart_factor_jc, ZERO_ALLOC_GAIN)
u_jk = sign(p) * (min(round(v_jk ** p, 2), goodput_clip) - lambda_n)
p = -0.5, lambda_n = 1.1；不新增 bonus
```

忽略取整时 u>0 要求 v>(1/1.1)^2≈0.82645。若所有列 u≤0，且没有真实锁，空集目标 0 可以最优。配置“合法但未画像”必须与物理不可行区分；保留 Sia 最小冷启动候选，不为打破空集伪造 Goodput。

四层不同保证：

1. **系统非空性**：存在可分配作业时，至少服务一个作业。
2. **逐作业首次有效训练**：每个满足条件的作业有限时间内 progress 真正增加；首次 allocation 不够。
3. **重复有效服务**：未完成作业反复得到正的训练时段，但不是所有作业一直连续占有资源。
4. **最终完成**：有限工作量被处理完。还需执行率正下界、状态不回滚和足够长期资源，不由 OPTIMAL 或一次非空推出。

## 4 活性方案比较

| 方案 | 打破全空 | 个别饥饿和重启风险 | 可证明的保证 | 对公平目标与研究问题的改变 |
|---|---|---|---|---|
| L1 可行时 Σx≥1 | 是，前提是同时满足现有锁后确有可行解 | 可永远选择同一高效用作业；锁结束即换配置可导致反复启动而无训练；大 gang 可被碎片永久阻塞 | 仅系统非空性；不保证逐作业首次训练或完成 | 在原目标上增加小约束；可作反例/弱活性对照，不宜作为主机制 |
| L2 队列词典序准入＋有效训练服务段＋必要时排空 | 是，不依赖效用符号 | 限定每次保护时长，队首不能被后到者超越；排空处理碎片。仍受不可行作业、零执行率和求解器永久失败限制 | 在第 5 节假设下，逐作业首次训练、重复服务及有限 cohort 最终完成 | 公平效用变成“准入与服务承诺内的条件优化”；不是原无约束公平最优 |
| L3 proportional-relative admission 或首次训练前 rho_sched=1 | 冷启动占位值足够且无其他阻塞时可能 | 解锁后绝对效用仍可非正；配置切换、碎片、较弱作业饥饿仍存在 | 单靠归一化/对称 bootstrap 不保证首次有效训练；加保护才能得到局部保证 | 改变调度效用的性能标尺，混入第三个因素；执行仍用绝对 rho 时预测/执行也分离 |
| L4 FIFO 准入后整份分配保护到完成 | 是 | 有限作业、每个单独可行且保持正执行率时可完成；长作业造成严重队首阻塞 | 条件成立时首次训练和最终完成；无限到达下不保证队列稳定 | 取消运行期弹性重配置，明显改变研究问题；可作验证参照，不推荐主实验 |

不推荐“最大服务作业数优先”作为 L2 的第一目标：一个需要整节点的作业可能持续输给两个小作业，最大基数本身不是逐作业公平性。也不选择增大 lambda_n 或 bonus：这只能移动空集阈值，不能解决锁结束后的重启循环或碎片饥饿。

**推荐 L2**，因为可把活性保证与资源效用分开，且允许服务段之间继续弹性选择。代价是显式增加服务优先权和可能的排空空闲；必须报告这些代价，不能将其藏在 B3 的资源优化收益中。

## 5 L2 的精确定义

### 5.1 服务段与优先队列

采用同一全局训练服务量 Q，单位为**productive wall-clock seconds**，不按 GPU 数量或 rho 加权，不包含等待/checkpoint/restore。此量不使用 `Job.attained_service`，因为原字段包含延迟且按 GPU 数计权。

供批准的主实验 Q 规则：`Q=Δ*max(1,ceil(D_bar/Δ))`，Δ=60 秒；D_bar 是依据当前数据域所有 application 的实际 startup/rescale/migration 代码路径预检得到的有限上界，不是从 JCT 或获胜策略日志估计。只有上界闭合后才写入冻结配置。若 D_bar=250 秒，则公式给 Q=300 秒；这是公式示例，不声称已完成本轮 D_bar 预检。采用共同 Q 避免给不同策略/应用暗加不同服务权重，也避免固定 60 秒训练却频繁付出数百秒启动成本。

作业到达后进入 ready FIFO，按真实 arrival、原始源行顺序、稳定 job key 排序。未完成作业获得 Q 有效训练后，在下一调度边界释放服务承诺并排到队尾。到达与释放同一时刻的顺序统一为：原始源行顺序的新到达在前，释放者按上次服务签发序号在后；规则四单元一致。未到达作业不进入排序或资源决策。

实际调度每 Δ 秒进行，因此一次服务可能有不足 Δ 秒的超额训练。承诺期间保留 config、完整 placement 和 CPU/RAM 配额；不借 epoch 边界重置服务量。原 Sia batch/epoch 更新继续执行。完成立即结束服务；服务量未达到 Q 不等于作业未完成。

服务段是本项目新的活性层，不是 Synergy-TUNE 或 Sia 的原机制，也不是本轮已经实现的功能。

### 5.2 硬约束与排空

L(t) 是当前真实 checkpoint-locked 集合；H(t) 是未完成服务段集合。对 j∈L∪H：

```text
Σ_{k属于j} x_k = 1
x_k = 0，若 (c,π,q)_k != 当前已提交的完整分配
```

两种保护原因分别记录，不把服务段伪装成 checkpoint。列的 query 必须重算当前 raw Goodput、版本和固定响应；Stage1.1 的当前资格检查仍适用。若当前完整分配真的违反硬容量/训练合法性，进入 `INFEASIBLE_COMMITMENT` 并停止，不能丢弃锁、用旧 rho 或把审计改为 valid。

对 ready 队首 h：

- 若 h 在空集群中也没有该单元合法候选，标记 `NO_SOLO_FEASIBLE_CANDIDATE`，区分数据缺失、GPU/scale-unit、CPU/RAM floor 和模型支持域；单 cohort gate 停止，不跳过作业假装成功。
- 若 H∪L 与 h 可共同放置，强制 z_h=1；按队列次序依次尝试加入后续作业。每次保留此前已接纳集合，只做可行性判断，不以效用或最大基数抢占队首。
- 若 h 单独可行但不能与当前承诺共同放置，进入 `DRAIN_FOR_HEAD`：不签发/续签任何新服务段；让已有承诺完成其余 Q 或 checkpoint 后的服务。必要时全局排空，再服务队首。该保守规则避免无限以“小作业回填”延迟大 gang。
- 求解器超时/无可用解不是不可行证明。记 `SOLVER_UNDECIDED`，仅可保留通过当前完整审计的既有分配；冻结队列与服务签发事务，保留已发生的真实训练计数。此类回退不承担活性保证；验证超时预算不通过时停止 gate。

确定准入集合 S 后固定 z_j（S∪H∪L 为 1，其余为 0），最后最大化 Σu_jk x_k。p、lambda_n 和 restart_factor 不变；此时 lambda_n 对固定服务集合只是常数项。不得用它重新决定是否服务，也不引入额外 allocation bonus。完全并列解以稳定列键作相同的词典序末级判定，不用随意 epsilon 改写主目标。

```text
account actual execution; complete/requeue exhausted service segments
build current legal parent candidates; validate real locks and commitments
construct the selected factor cell by restricting candidates only
if head has no solo-feasible column: explicit data/config failure
elif commitments + head provably infeasible: DRAIN_FOR_HEAD
else:
    lexicographically admit feasible jobs in ready order
    freeze admission vector; maximize the unchanged B3 utility
rebuild current queries; audit whole GPU/CPU/RAM map and locks
atomically commit allocation + queue/service decisions, or commit nothing
```

### 5.3 保证、证明思路和不保证的情况

条件：有限初始/最终到达 cohort；每作业在每次服务时至少有一个保持合法的单独可行列；服务段内节点可用和容量稳定；checkpoint/restore 总延迟有统一上界 D_bar；有限训练工作量、有限 epoch、进度不因抢占回滚；合法列执行率在未完成区间有正下界 ε_j；求解器/可行性证明不会永久失败。硬约束不能由“平均负载不高”替代。

在这些条件下，已有服务段在不超过 `L=D_bar+Q+2Δ` 的保守时间内退出或完成，且排空期间不续签，所以碎片阻塞不能无限延续。队列中排在 j 前面的作业每次服务后到队尾，新到达者不能插到 j 前面；有限队列使 j 有限时间获得服务。一次未提前完成的服务带来至少 Q 的有效训练时间，且执行进度增加。有限工作量与正执行率下，重复服务最终完成。

对最多 N 个未完成作业，可采用宽松 watchdog 上界 `B=(2N+2)L` 检查相邻服务机会间隔；这只适用于上述承诺/求解假设和共同 Q 规则，实施时须以状态机证明和小规模枚举复核。完成界还依赖未知工作量/ε_j，**调度器不得读取它们作排序**。Sia epoch 秒数取整残差须按 Stage1.1 规则单独审计，不能用一次乘法等式忽略它。

持续无限到达时，局部有限到达、每次入队前只有有限个先行者等条件可支持个体“最终获得一次服务”的论证，但这里不宣称有统一等待上界、稳定队长或稳态尾延迟。仅“负载在多资源容量域内”也不充分，因为重启与保守排空会损失可用服务能力。本阶段正式保证范围限定有限 cohort。

零/负执行率、模型缺失、不可行 gang、无限/重复重置 checkpoint、节点故障、旧配额在服务段中失效及永久 solver failure 都打破前提，应显式失败，不能靠 lease 永远保留一个不训练作业。记录 `productive_delta=0` 的原因；仅资源等待可以持续至对应承诺界，不能把 delay 到期后反复零进度当成正常活性。

## 6 新四单元的公平消融

### 6.1 两个因素的精确定义

令 C_j(t) 为按 Stage1.1 实际 Sia 规则生成的**共同父 GPU 候选集**。先在所有策略外部统一生成 s_jc 及固定 phi；不得因 F 只剩一列就重新归一化 s 或 phi。

F 的含义是“本轮一个 GPU-only 提案随后冻结”，不是整段训练固定需求。独立、确定性的提案规则为：在 C_j 中，先剔除空集群 proportional 分配也不可行或响应数据未定义的配置，再选择 `argmax s_jc(t)*restart_factor_jc(t)`；并列按稳定 config ID。只检查静态单作业可行性，不读取辅助响应收益来排名、不考虑其他任务的剩余容量。未画像时沿用原 Sia cold-start 候选；若集合为空则显式报错。提案只定 GPU type/count/node-count，不定 placement。

这是一种 **Sia Goodput 启发的单提案控制策略**，不是运行原 Sia MIP。原 Sia MIP同时决定 admission 和全局 GPU 竞争，若把它的空分配/placement 继承到一个单元，就再次引入目标、准入和 placement 混杂。真实 lock/服务承诺期间所有单元均继承已承诺 c，不重新提案。

P 对每个 c,π 只允许：

```text
qP_c = max(node_CPU/node_GPU, minimum_CPU_per_GPU)
qP_m = max(node_RAM/node_GPU, minimum_RAM_per_GPU)
```

多节点配置要求各节点资源比例一致并遵循现有整节点放置规则。O 使用现有离散 CPU/RAM 网格并含 qP，保持相同最低资源点和同配置的 `rho(q) >= rho(qP)` progress floor；所有查询使用同一响应模型。P 与 O 都优化 placement，不允许 P 沿用一个更差的固定 Sia placement。

| 单元 | 因素 A GPU 集合 | 因素 B 辅助配额集合 | placement |
|---|---|---|---|
| U-FP | 本轮提案 {c_j^0} | {qP(c,π)} | 同一合法枚举与优化 |
| U-FO | 本轮提案 {c_j^0} | 当前辅助网格＋qP，受共同 floor 约束 | 同上 |
| U-JP | C_j(t) | {qP(c,π)} | 同上 |
| U-JO | C_j(t) | 当前辅助网格＋qP，受共同 floor 约束 | 同上 |

同一输入状态下应满足列集合包含关系 FP⊆FO、FP⊆JP、FO⊆JO、JP⊆JO（按几何/配额识别，不按列名）。只在各自合法集内优化；不声称扩大列集能保证整个 JCT 轨迹改善。

### 6.2 共享项与预期差异

| 项目 | 四单元共同规则 | 唯一允许的差异或解释 |
|---|---|---|
| 主目标 | 第 3 节 u，完全相同 p、lambda_n、restart、取整/clip | 可选列不同导致最优解不同 |
| 准入 | L2 相同 ready 顺序、feasibility 判定和服务承诺 | 可行域不同会改变实际服务集合，这是因素的中介结果，不要求跨策略同一作业同时获准 |
| 活性 | 相同 Q 公式、FIFO requeue、排空、锁/失败状态机 | 不只给 JP 加保活 |
| GPU 候选 | 同一 C_j 生成、direct/projected 标记与父集归一化 | F 过滤为一个提案，J 保留所有候选 |
| CPU/RAM | 相同容量、minimum floor、profile、rho、无动态 phi | P 唯一比例点，O 多个合法点 |
| placement | 同一枚举、逐节点约束、优化与完整锁 | 本轮不估计“有无 placement 优化”效应 |
| 执行 | 同一 Job.step、总工作量、arrival/batch/epoch/完成事件、rho 单次作用 | 不以候选预测 Goodput 代替实际执行率 |
| 重配置/费用 | 原 Sia 时间推进；服务承诺内不重配置 | 未建模的纯辅助配额变化成本仍明确记为合成零额外成本，不能声称真实零成本 |
| fallback | 当前资格、当前 raw/query/version、整图审计；冻结事务 | 无过期 locked 列，无部分提交；solver failure 不当作可行性结论 |

相同代码快照不意味着运行到第 k 轮时四策略仍有相同 profile/progress；由既往分配产生的在线画像分叉是总策略效应。另用**同一状态快照的一轮四单元测试**验证只有候选过滤不同，区分实现混杂与路径中介。

### 6.3 备选对照定义

若研究者坚持“固定端必须就是原 Sia MIP”，可保留 `SiaIntent-P / SiaIntent-O` 与 `Joint-P / Joint-O`，但只能称为四策略对照，不称正交 2×2：原 Sia 的全局 admission/候选选择仍属于处理的一部分。另一备选是提前按 workload demand 和类型映射固定整程 GPU；这样消融同时移除异构与弹性选择，问题更粗，也不能称原 Sia。

推荐 U 四单元；B1/B4 保留外部基线。B5/SY adapted 单列，注明原有 TUNE demand→fair→peer downgrade 及固定 GPU 适配规则，不宣称官方未经修改的 Synergy。若未来在新执行模型下重跑外部基线，用 `H-B1@E2 / H-B4@E2 / H-B5@E2 / SY-adapted@E2` 命名，不覆盖历史数据、不暗中给它们添加 L2；其删失单独报告。

## 7 策略无关的 phi 数据方案

### 7.1 已有资料能支持什么

[响应数据预检](D:/科研启航/Sia/research_notes/synergy/B3_RESPONSE_DATA_PREFLIGHT.md)及[机器可读覆盖](D:/科研启航/Sia/research_runs/sia_synergy/correctness_stage1/response_preflight_final/response_preflight.json)给出：5 应用×3 GPU 类型共 30 个 placement/scalability 源文件；AWS 单节点 1/2/4 GPU、DGX-ext/RTX 1/2/4/8 GPU 的 **55 个应用/类型/几何组合有记录**。这不是 55 个组合在统一 batch/epoch 下全部可校准的证明。

Synergy 有 12 个 9×5 正矩阵；当前映射 CIFAR10→alexnet、DeepSpeech2→deepspeech、ImageNet→res50、BERT→bert、YOLOv3→mobilenet 不变。轴为 CPU `[1,2,3,4,5,6,9,12,24]`，RAM `[20.83,62.5,125,187.5,250]`。原矩阵是数据来源；这些跨应用/跨硬件映射不是 Saturn 实测。

历史 selected 日志只给 B1/B4 各 53、B3 63 个 job/config 的 phi；各自还有 167/157 个枚举 ID 无值，B3-PROP 没有 selected phi。不能从这些轨迹挑某个“代表值”填表。Stage1.1 还证明补全 GPU 候选会改变现有 phi 分母，因此“取第一轮”同样不能解决所有配置覆盖。

### 7.2 推荐参考状态 R0

推荐把 phi 的依据从**在线统计 Goodput**改为**固定离线参考吞吐**，并单独版本化为 E2。理由是 CPU 输入速率需求的合成假设不必混入训练梯度效率，且原料只需已存在的 step_time，不使用未来训练 epoch、完成时长或任何策略轨迹。

定义不可随策略改变的数据域 D：本轮冻结拓扑下全部物理/训练几何合法的配置，包含之后可能成为 online candidate 的配置。D 在查看新结果前冻结，不因本轮只有 1-GPU 在线候选而缩小，也不按某策略选中频率加权。当前单节点 DP 域预计对应上述 55 几何；PMP、额外多节点或新应用必须另建域，不能自动借用覆盖声明。

对每个应用 m，令 A_m 为 D 内所有几何在各 GPU 上**均有直接记录**的 local microbatch 值交集。选 `b_ref(m)=min A_m`，accumulation=0，epoch/梯度状态标为“不适用，吞吐参考”。相同 microbatch 保证比较的是固定输入粒度，不随作业实际 batch 调整；它只是生成固定需求倍率的校准条件，不改变 Job 的训练 batch。

读取每个几何在 b_ref 的 step_time；同一来源的重复行按预注册中位数聚合，优先 placements，同几何缺失时才检查 scalability，保留全部行索引。不得根据哪个来源给出更快吞吐选来源。若 A_m 为空或存在非正/非法 step_time，标记缺失，停止数据 gate；本轮未计算交集，所以不能声称该规则已覆盖全部 55 项。

```text
T0(m,c) = GPUs(c) * b_ref(m) / step_time(m,c,b_ref)
r0(m,c) = [T0(m,c)/GPUs(c)] / min_{d in D_m}[T0(m,d)/GPUs(d)]
phi_c(m,c) = clip(r0, 0.5, 4)
phi_m(m,c) = clip(sqrt(r0), 0.5, 2)
```

保留现有倍率上下界，不再在线重算。r0≥1；最慢参考组合得到 phi=1 是公式结果，**不是缺失默认值**。需求随每 GPU 参考吞吐变化、RAM 随其平方根变化依然是合成假设，数据不能证明真实 CPU/RAM 需求如此缩放。

直接记录交集不足时的备选 R1：预注册公共 local-batch 支持区间、只做区间内插值，复用 [Application.get_throughput:248](D:/科研启航/Sia/sia-joint-correctness-stage1p1/sia-simulator/applications.py:248) 的已知几何路径并显式记录插值来源；区间外不外推。再备选 R2 是固定 batch＋固定离线 epoch/梯度状态的 Goodput 标定，但引入更多资料和“离线先验”假设，不推荐作为最小实现。R0 失败时不能自动切 R1/R2，需单独数据审批。

### 7.3 完整覆盖表与缺失规则

将 `reference_state_id / D_hash / app / gpu_type / nodes / gpus / placement_shape / b_ref / accumulation / source files+hashes+rows / T0 / denominator domain+value / r0 / phi_c / phi_m / quality / missing_reason` 存入不可变 registry。job key 用 trace hash＋源行号，避免不同 trace 的同名 ID 混淆；job→application/template 映射另表冻结。

生成覆盖顺序为：源文件→固定参考 batch→全部 D 配置→各作业静态训练限制→所有可能 quota/placement→当前四单元过滤。报告源行直接覆盖、插值覆盖、缺失分别多少；不得只报告选中列覆盖。任何待运行作业缺全部候选、任一可能被查询的 registry key 缺失，均停止 cohort gate；不删作业、不缩 GPU 拓扑、不用其他策略轨迹填洞。

Sia 的在线 Goodput 仍按原语义更新，registry 不反向 seed 在线 profile；调度器只查询已到达作业的响应。离线知识提前声明为共同模拟环境先验，不等于获得未来到达、未来实际工作量或最优运行轨迹。

## 8 分子与分母边界的模型选择

保留 fixed-reference 配额 `(3 CPU/GPU,62.5 GB/GPU)` 和 Medium α=0.5；若 registry 固定，当前公式为：

```text
numerator coordinate   n = (q_c/phi_c, q_m/phi_m)
denominator coordinate d = (3/phi_c, 62.5/phi_m)
rho = [R(n) / R(d)]^0.5
```

参考点 q=(3,62.5) 必须严格 rho=1，分子分母必须使用同一边界算子。基准配额是校准参考，不要求每个 GPU 配置都能在当前容量下同时获得它；不能把不可容纳的参考配置说成真实测量。

| 边界方案 | 分子越界 | 分母越界 | 完整性与代价 |
|---|---|---|---|
| E-strict | 此 quota 不可查询/不可作为列 | 整个 config 的归一化不可定义 | 最少外推假设，但可能丢光某作业候选；必须停止而非自动删作业 |
| E-legacy | clip 到边界 | 同样 clip | 与旧定义接近，但容量不足可能落入边界平台；仅可作独立历史兼容模型，不等于真实低资源行为 |
| E-linear 推荐供审批 | 下界显式线性衰减，上界饱和 | 使用完全同一延拓计算；正基准且有越界标记 | 定义域完整，但延拓特别是分母是新的合成校准，不能称源矩阵支持的测量 |

令源矩阵域 Ω=[1,24]×[20.83,250]，R_Ω 为原双线性插值。推荐 E-linear 的明确公式：

```text
R_ext(c,m) = R_Ω(clip_Ω(c,m)) * min(1,c/1) * min(1,m/20.83), c,m >= 0
```

域内不改矩阵；高于上界假设饱和；低于任一下界按该资源比例衰减，两者同时不足则相乘。幂 1 只是连续、零资源处为零的一阶合成闭合规则，不是拟合所得指数；α 仍为 0.5。它可避免把越来越少的 CPU 全映射到同一个正吞吐底部平台，但不能证明真实硬件严格线性或全域边际收益递减；原始矩阵的非单调点不做暗中平滑。

原来的需求倍率 clip 与这里的坐标边界是两件事。仍记录每个查询的原始 n、d、源矩阵 clip 坐标以及下界衰减因子；不能通过只记录 clip 后坐标让越界统计变成 0。E-linear 不是“禁止所有 clipping”，而是用明确模型替代旧下界常数延伸。

已知边界事实：现有三类节点比例执行点 CPU=1，phi_c∈[1,4]，故比例分子 CPU=1/phi_c，严格域内当且仅当 phi_c=1；分母 CPU=3/phi_c，phi_c>3 时越界。旧条件状态中 B1/B4 的 300 个比例查询状态有 280 个会被 strict 删除，B3 的 324 个有 304 个会被删除。这些是历史观测子集，不能拿来预测新固定 registry 的精确覆盖比例。

CPU/RAM × numerator/denominator × 下界/上界须分别按“所有原始 quota 点、容量可行列、progress-floor 后列、唯一 job/config、实际执行 job-round”统计。历史 RAM 越界为 0 不保证完整新表为 0，例如 q_m=20.83、phi_m>1 会低于源 RAM 下界。

特别限制：恒 1 模板只说明源域内不敏感；E-linear 在域外仍赋予资源不足惩罚，不能再把这些作业称为全域 CPU-insensitive。无敏感性机制测试应另用明确的 neutral rho=1 对照，不能据此更换主实验模板。

数据 gate 需达到 **100% 模型定义覆盖、0 未标记越界、0 非正分母、0 无解释缺失**，并同时公布真实源域覆盖率；这不是要求所有点都有实测支持。批准 E-linear 后越界率本身不被事后拿来调参；若研究者不接受基准点外推，选择 E-strict 并承认可能无法开展完整现有 cohort。

## 9 最小实现范围与验收测试规划

建议后续新分支 `exp/b3-stage2-unified-liveness`，从本设计基线 commit 建独立工作树，完整快照、patch、依赖锁与新 registry 一并冻结。下列是预计修改范围，不是本轮改动：

| 文件或新模块 | 最小职责 |
|---|---|
| 新 `service_admission.py` | ready 顺序、服务计数、DRAIN、词典序可行性与事务状态 |
| joint_scheduler.py | 真实 lock＋service commitment 等式、固定 admission vector、共用效用与当前资格审计 |
| joint_integration.py | 共用父候选生成、F 提案过滤、四个新 mode、完整提交/失败日志 |
| 新 `reference_profile_registry.py` 与离线生成器 | 只读原始 profile 生成固定表；禁止导入 Job.step/Cluster.step |
| joint_profile.py | 固定 phi registry、版本化边界算子和分子/分母记录；E1 历史路径保留 |
| JointCluster 与 execution_audit | 消费既有 productive segments 计数；不重写训练推进；检查承诺和回退 |
| 新验证器/分析器与测试 | 候选差异白名单、覆盖/活性证书、删失、2×2 汇总 |

先写红测再实施；现有 71 项全部保留。建议精确 fixture：

| 测试文件 | 输入 | 必须得到的输出 |
|---|---|---|
| test_service_admission.py | 1 节点 1 GPU，两个合法列各 u=-1，均到达 0 | 队首被服务，不允许 OPTIMAL 空集；无需修改效用 |
| 同上 | 两作业共享 1 GPU，D=120、Q=60 的测试实例，Δ=60 | 120 秒延迟不算服务量；至少完成 60 秒有效训练才轮换，另一作业随后获得服务 |
| 同上 | 2 GPU 节点；大作业需 2、小作业需 1，不断有限地到达小作业 | 大作业成为队首后排空，不继续签发妨碍它的服务；分别演示 L1 的饥饿反例 |
| 同上 | gang 要 2 个整节点，已有两个剩余服务段不同步 | 保护已有承诺；全部退出后 gang 获准；无碎片无限回填 |
| 同上 | 真实锁 u<0、另一列更高效用；未知 kind/过期 query | 真锁完整保留且选中；伪造列拒绝 |
| 同上 | 单作业 CPU floor 大于节点容量，或执行率 0 | 明确 UNSCHEDULABLE/执行错误，不无限等待、不补值 |
| test_unified_factor_contract.py | 同一真实 JobInfo 快照运行四个候选过滤 | 目标、Q、准入、归一化、placement、oracle hash 全同；仅两项过滤不同，满足包含关系 |
| 同上 | 候选 raw 为 2 与 3；只有资源联合选择能容纳更多作业 | F 在资源收益之前固定提案；J 可另选；不能把 Sia MIP no-allocation 当 F 提案 |
| test_fixed_reference_registry.py | 同一 app/config，打乱 job/策略/候选顺序和在线 profile 值 | phi 字节一致；同一 config/quota 的 rho 不变 |
| 同上 | 删除一个源文件/固定 batch 单元 | 标记具体缺失并停止；不得 phi=1 或偷用旧轨迹 |
| test_response_boundaries.py | n_CPU=0.5、d_CPU=0.75，RAM 域内；以及只分母越界 | E-linear 使用显式 0.5/0.75 因子；E-strict 分别拒绝；两类 flag 不混合 |
| 同上 | q=(3,62.5)，含 phi_c=4 的样例 | rho=1；分母越界仍须标记，不能因比值为 1 忽略 |
| test_service_execution.py | 相同 Job、rho=1 和 k，跨 checkpoint/epoch/完成边界 | delay 不加速；独立 raw 积分与 progress 一致；服务只累计 productive seconds |
| 同上 | 服务段中 solver x=None、真实锁、解锁后非法旧 quota | 只保留当前可验证完整分配；无陈旧 rho/过期资格，失败不伪造服务完成 |
| test_factor_provenance.py | 修改一个单元的 Q、目标或响应 registry | verifier 拒绝整组；不能仅警告继续分析 |

D=120、Q=60 等仅是测试 fixture，不是以测试结果替换主实验 Q 规则。真实集成测试继续包含 Stage1.1 unseeded bert-mid 完整 optimize_policy 链路，不得为了新准入通过而 mock 下一次调度。

## 10 统计定义和效应分析

在 [arrival,completion) 上按实际事件分段：waiting=未分配；startup/checkpoint=已分配且延迟倒计时；productive=已分配且真正训练；stall=已分配但前两者之外无进度。四者互斥，JCT 为其和。正常运行 stall 应为 0；异常 stall 不能混入 productive。保留首次 allocation 与首次 productive 时间。完成后的周期残余不计 JCT 或实际资源小时；若资源直到下一轮才可重调度，另报调度保留空闲，不混作训练使用。

GPU/CPU/RAM 小时按分段配额×实际持有时长积分，checkpoint 占用计入，完成后不计。报告 lease-start、renew/requeue、DRAIN 时长/空闲资源、restart、真实 delay、wasted GPU-seconds、solver/fallback 各状态，区分预测归一化 Goodput 与执行 raw rate。

设四个单元完整平均 JCT 为 μ00=FP、μ01=FO、μ10=JP、μ11=JO，节省为正：

```text
GPU simple effects: μ00-μ10, μ01-μ11
Aux simple effects: μ00-μ01, μ10-μ11
GPU main effect:    [(μ00-μ10)+(μ01-μ11)]/2
Aux main effect:    [(μ00-μ01)+(μ10-μ11)]/2
Interaction gain:   μ10+μ01-μ11-μ00
```

交互为正表示辅助优化在联合 GPU 候选下节省更多 JCT。每项同时给 job-seconds、相对各自对照的百分比、paired per-job difference、改善/退化/相同数量；不把不同分母的百分比直接当绝对交互。

删失报告 `(observed_time,completed,censor_time)` 和固定观察窗完成率，不插补∞或预计 JCT。对同一 cohort 选 `τ=T_max-max_j arrival_j`，保证所有作业至少可观察 τ 的 job-age；可计算经验 RMST=`mean(min(JCT_j,τ))`，未完成但确实观察到 τ 的作业贡献 τ，这是截断估计不是填补完整 JCT。异常中止不是行政删失，不能照此填 τ。若 τ 不共同可观察则另行预注册生存估计，不自动混用。

四单元不全完成时不能报告完整 JCT 因子效应，只报告预注册 RMST/完成率因子差异，并停止扩展 gate。完整时报告 median、P95/P99、makespan 与资源小时的冲突；每组仅 20 作业，尾部只是描述性样本指标。

预先跟踪 deepspeech2-159 及每 cohort 对配对总差值贡献最大的作业。报告其贡献占比、去掉该单作业后的诊断性均值，但全作业结果始终是主分析。九 cohort 以 cohort 为等权比较单位，给胜负数、中位改善率、范围/最差组，不把 180 个相关作业当成 180 次独立实验。

## 11 分级准入与停止标准

所有 gate 必须先冻结配置和完整源码/数据快照。本设计不授权运行任何 gate。

| Gate | 工作范围 | 通过条件 | 停止条件 |
|---|---|---|---|
| G0 数据与语义闭合 | 不训练：生成 registry、完整 quota 网格预检、确定 D_bar/Q、四单元契约 | 研究者批准下述三项决定；可能查询 key 全覆盖；源/外推分别标注；零无解释缺失/非正分母；所有作业至少一个单独可行配置；候选/依赖哈希冻结 | 参考 batch 交集不闭合、任一作业失去全部候选、需要新外推或新增资源、无法界定延迟上界 |
| G1 小型机制 | 原 71 项＋新增红绿；1–3 作业有限场景，真实 Sia 链路；neutral 与选定 E2 Medium | 所有合法 fixture 在事先推导服务界内首次训练并重复服务/完成；锁/容量/原子性/单次 rho/时间划分全通过；反例确实区分 L1/L2；原始 Sia 有界回归不变 | 任意非法提交、零进度循环、无法解释的超界、被 test mock 掩盖的真实调度失败 |
| G2 单 20 作业 | 固定 w10_late 140–159；先 neutral 四单元再 E2 Medium 四单元，各独立进程；Δ=60、同一 6000 轮窗、原容量/workload | 四单元在两响应设置均 20/20；无 round limit；逐节点容量/锁/资格/时间 audit 零违规；所有有效等待有原因；正常运行无未解释 solver failure/fallback；哈希/目标契约一致 | 删失或 solver failure 保留原证据，只报预注册截断量；不延长窗口、不调 Q/phi/映射直到“通过” |
| G3 九 cohort 复验 | 获批后固定既有 w1/w5/w10 early/middle/late 规则；同版本、同响应模式下 9×4；外部基线另矩阵 | 每 cohort 四单元完整且所有审计通过；跨 cohort 等权、负结果和长尾影响完整报告 | 任何新增数据缺失或语义错误停止后续策略；单 cohort 无收益不是失败，不据收益选择继续哪些组 |

每个策略前可用内存必须≥2 GiB；不足则安全停止，不降低门槛。观察窗不足与死锁必须区分，但两者都不满足完整 JCT gate。若需改变窗口或服务量，必须另开版本、事先对所有单元统一冻结，不修改已完成结果。

G2 通过只允许请求 G3 审查，不自动启动；G3 不以 JO 获胜为准入条件。不做参数搜索，不运行 160-job，不自动增加 Low/High。

## 12 历史兼容性和需批准的最少决定

历史 run1/run2、Neutral/Medium、Stage1/1.1 全部保留。新活性约束、F 提案和静态 phi/边界算子都会改变语义，不能把旧 B1/B4/B3 数值替换到新矩阵里；也不能将 E1 与 E2 的收益差全部解释为算法进步。若未来要隔离“模型变更效应”，需要同一新四单元在两种预注册模型下另作配对设计，而不是拼接历史结果。

下一阶段只需决定三件事：

1. **活性机制**：是否批准 L2 的共同 productive 服务段、Q 推导规则、FIFO 重排和必要全局排空；接受其条件公平目标与有限 cohort 保证范围。
2. **GPU 对照定义**：是否批准 U-FP/FO 的 GPU-only 单提案控制与四单元共同 placement；若坚持原 Sia MIP，改称策略对照而不是干净 2×2。
3. **响应边界模型**：是否批准 R0 固定离线吞吐 phi 与 E-linear 明确合成延拓；或选 E-strict，并接受覆盖不足即停止。R0 缺数据时是否另行审批 R1 内插，不能默认切换。

在这些决定及 G0 输出获批前，不进入算法实施或新训练实验。
