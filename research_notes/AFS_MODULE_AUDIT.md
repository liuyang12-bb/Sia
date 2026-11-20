# AFS-P 模块审计与 Sia 异构适配设计

- 审计日期：2026-10-23（Asia/Shanghai）
- 研究基准：Sia SOSP 2023 Artifact，commit `aaf7a0de7b6cb5eab06ea020467ede5d2fd18e70`
- AFS 论文：Changho Hwang et al., *Elastic Resource Sharing for Distributed Deep Learning*, NSDI 2021, pp. 721–739
- 本地论文：`research_sources/AFS/nsdi21-hwang.pdf`
- PDF SHA-256：`9CA51FCD1FB65C318C0936C5BE4CD465B63C3CEC0EA040C3FD0DC7A31288BB65`

## 1. 基线保护记录

阶段 0 检查结果如下。

| 项目 | 结果 |
|---|---|
| `official-artifacts/` HEAD | `aaf7a0de7b6cb5eab06ea020467ede5d2fd18e70` |
| 原始仓库状态 | `## main...origin/main`，无工作区修改 |
| LAS worktree | `sia-experiments/`，分支 `exp/sia-las-weighted`，仅有上一轮预期的 `sia.py`、`multi_simulator.py`、`sia_las.py` 和测试修改 |
| LAS 产物 | `research_runs/sia_las_weighted/` 与 `research_notes/SIA_LAS_EXPERIMENT_REPORT.md` 均存在 |
| AFS worktree | 从原始 commit 新建 `sia-afs-experiments/`，分支 `exp/sia-afs` |
| Python | `.venv-sia-baseline/Scripts/python.exe`，Python 3.10.10 |
| 官方 trace | `workload-1.csv`、`workload-5.csv`、`workload-10.csv` 均存在且各含 160 个作业 |
| CPU / 内存 / 磁盘 | 8 物理核、16 逻辑核；总内存 13.69 GiB；检查时可用内存 1.50 GiB；D 盘可用 29.97 GiB |

为按 PDF 审阅流程抽取和渲染原论文，在既有隔离环境中安装了仅用于审计的 `pypdf==6.21.2`、`PyMuPDF==1.28.2` 及其依赖 `typing_extensions==4.16.0`；未修改系统 Python。已将 PDF 第 5、6、7、18 页渲染到临时目录 `tmp/pdfs/afs/` 并视觉核对公式、Algorithm 1 与 Appendix D。

检查时可用内存低于任务规定的 2 GiB 门槛，因此本阶段不会启动完整 trace 模拟；实现、单元测试和小型 smoke 不代表对长实验安全门槛的放宽。

## 2. AFS 的问题定义与核心假设

### 2.1 输入、输出与触发时机

论文 Section 3.1（p. 724）把问题定义为：在具有 `M` 个同构 GPU 的集群中，对当前和未来到达的 BSP 训练作业决定每个 churn event 后的 GPU 份额向量

`R_u = {r_1,u, r_2,u, ..., r_n,u}`。

基本调度事件是作业到达和完成；脚注说明为简洁起见省略 eviction timeout 等其他事件。AFS-P 的 processor-sharing 时间片到期会额外触发重调度（p. 726）。输出是每个活动作业的 GPU 数，份额为正的作业一直运行到下一事件。

论文的简化模型假设 GPU 计算/显存能力相同、访问网络延迟相同，并允许作业在运行中使用任意数量的 GPU。它不读取未来到达或真实剩余长度；AFS 的关键假设反而是：除即将完成者外，现有作业未来获得的资源份额与当前相同，即使有作业完成释放资源，也不乐观地假设这些资源会全部交给剩余作业（Section 3.1–3.2，pp. 724–725）。

### 2.2 两作业推导与式 (3)

Section 3.2（p. 725）从已知作业长度的两个作业推导。令 `p_j(r)` 为作业 `j` 使用 `r` 个 GPU 的训练吞吐量，`p'_j = p_j(r+1)`。在 `j_a` 为较短作业、`j_b` 为较长作业时，额外 GPU 应优先给 `j_b` 的 AFS 判定是论文式 (3)：

```text
(p'_b - p_b) / p'_b  >  (p'_a - p_a) / p_a
```

左侧用新增 GPU 后吞吐量作分母，右侧用新增前吞吐量作分母；实现时不能把它误写成普通的“最大边际吞吐量”或两个同分母比率。

式 (3) 来自 AFS 的未来份额假设：在两作业分析的未来时段中，以当前份额对应的吞吐量替代“完成者释放资源后另一作业独占资源”的乐观吞吐量。其结果会在竞争高时抑制短但扩展效率差的作业独占资源。

## 3. AFS-L Algorithm 1

论文只给出 AFS-L 的正式伪代码（p. 726），未给出一份独立的 AFS-P Algorithm 2。以下为不逐字复制论文的等价结构：

```text
TopPriority(J):
    从任一作业开始作为当前胜者 j*
    依次与其他作业 j 两两比较：
        若两者当前份额均为 0：比较单 GPU 下的完成长度，短者胜
        否则：按当前份额下的完成长度，把短者记为 ja、长者记为 jb
              若式 (3) 成立则 jb 胜，否则 ja 胜
    返回 j*

AFS-L(J, M):
    把每个作业的临时份额 cnt 清零
    重复 M 次：
        j* = TopPriority(J)
        j*.cnt += 1
    按 cnt 分配 GPU
```

输入是活动作业集合、每个作业的长度函数/吞吐量函数以及 GPU 总数 `M`；输出是每作业整数 GPU 份额。复杂度为 `O(Mn)`。

AFS-L 是两阶段算法：

1. 先假设所有作业单 GPU 运行，按单 GPU 作业长度从短到长各分配一个 GPU；若 `M` 小于作业数，到此停止。
2. 若仍有 GPU，再逐个用式 (3) 同时考虑长度和资源效率。

当某作业当前份额为 0 时，其当前完成长度视为无穷。论文 Appendix C（pp. 736–737）证明已知长度版本的两两优先关系可传递。

## 4. AFS-P：未知作业长度时的实际规则

### 4.1 修改后的 TopPriority

Section 3.3（p. 726）明确说，在不知道作业长度时，对同一对作业把式 (3) 的两种长度次序都计算一次：

- 假设 `j_a = j*`, `j_b = j`；
- 再假设 `j_a = j`, `j_b = j*`。

若任一方向成立，相应的 `j_b` 优先；论文说明两种方向不可能同时成立。若两个方向都不成立，真实优先级依赖未知的完成顺序，因此随机选一个。这里的随机选择是论文规则，不应悄悄改成按名称、年龄或未来长度排序。

### 4.2 Processor-sharing 状态

AFS-P 用 processor sharing 模拟 AFS-L 第一阶段的类 SRTF 行为：

- 每个作业到达时计数器为 0；
- 作业每被调度执行一个 unit time，计数器增加 1；
- 当活动作业少于 `M` 时，忽略这些计数器，保证每个作业至少一个 GPU，然后用修改后的 `TopPriority` 分配剩余 GPU；
- 当作业数超过 `M` 时，算法停留在第一阶段，只给计数器最小的作业各一个 GPU；运行中的作业在用满一个 unit 前不被抢占，用满后才被驱逐并触发全局重调度；
- 论文所有 full-scale traces 的 unit time 为 2 小时，即 7200 秒，并要求工作负载时间尺度变化时相应缩放（p. 726，脚注 11）。

`n = M` 时所有作业恰好各得一个 GPU，计数器不会改变可运行集合；实现可归入第一阶段而不改变结果。

计数器是“被调度的时间”，不是 GPU-seconds、提交年龄或真实剩余时长。多 GPU 阶段也不能把 `GPU 数 × 时间` 当成该计数器。

## 5. 吞吐量、可分割资源和重配置假设

- 论文以训练吞吐量 `p_j(r)` 为份额判定依据，并假定 BSP，使系统吞吐量与算法训练吞吐量一致（Appendix D，p. 737）。
- AFS-P 需要任意 GPU 数下的 throughput scalability。CoDDL 的 Section 4.5（pp. 728–729）采用“先高估、后重调”：新作业先假设线性扩展且初始份额不超过公平份额，随后实际测量不超过当前份额的各可分配点；超过已测范围时从最后一点线性外推，收到真实吞吐报告后再次调整。
- AFS 的算法模型允许任意整数 GPU 份额。CoDDL 实际系统再做 network packing：每台 `2^k` 个 GPU 时，把份额规整为 `0`、小于节点大小的二次幂或节点大小的整数倍（Section 4.4，p. 728；Appendix E，pp. 737–738），并尽量减小与原份额的差。
- CoDDL 依赖低开销弹性调整：缩容可在毫秒级完成，扩容初始化与旧份额执行重叠，并可取消过时的重配置。Sia 模拟器则会显式收取 checkpoint/restore 与迁移代价；这是一项不能忽略的系统假设差异。
- 原文在份额算法中没有 Sia 的重启折扣目标项，也没有异构 GPU 类型选择。

## 6. 公开源码核查

检查了 USENIX 论文页、论文正文/参考文献、KAIST 项目介绍、作者/实验室公开 GitHub 检索结果。论文没有 artifact/source-code 链接，USENIX 页面只提供论文、幻灯片和演讲；截至本次审计，没有找到可可靠归属于作者的 CoDDL/AFS 公开实现。

因此后续实现只能依据论文的式 (3)、Algorithm 1、AFS-P 修改段、Section 4.4–4.5 与附录完成。报告必须称为“论文驱动的 AFS-P 参考实现及 Sia 异构适配”，不能称为官方源码移植。

## 7. Sia 接口与语义核查

### 7.1 可直接复用的状态

- `cluster.py:get_sia_multi_job_info()` 提供每种 GPU 类型的 `speedup_fn`、`max_replicas`、`scale_unit`、一 GPU throughput/跨类型 ratio、checkpoint 状态和既有累计状态。
- `job.py:Job.step()` 中 `execution_time` 只在真正训练推进时增加，不含等待和 checkpoint/restore 延迟；它比 `used_gpu_seconds` 更符合 AFS-P 的“被调度时间”计数器。实现只需把该现有值附到 `JobInfo`，不新增第二套计时。
- `SiaPolicy._compute_goodputs_v2()` 能在已有模型或 `--project_throughputs` 下计算具体 GPU 类型与配置的 Goodput。`SpeedupFunction.get_goodput()` 默认返回绝对 Goodput，而不是仅返回 speedup。
- `SiaPolicy.populate_valid_configs()` 已定义每类节点合法的 `1/2/4/8...` 配置；`alloc_to_placement()` 可把 `(num_nodes, num_gpus)` 转换为合法节点列表。

### 7.2 Goodput 与 AFS throughput 的关系

Sia 的实际作业推进由 `goodput = statistical_gain / iteration_time` 决定。若 AFS 适配仍用原始 examples/s，它会忽略批量大小改变造成的统计效率，优化对象将与模拟器 JCT 推进不一致。因此异构适配用 Sia Goodput 作为论文 `p_j(r)` 的进度速率代理。这是有明确语义依据的替换，不是给 Sia 目标函数乘权重。

冷启动时尚无 Goodput 模型。按照论文 Section 4.5，适配先用一 GPU观测值作线性外推；当 profile 从无到有或 profile 点数变化时，将“吞吐量报告”视为额外重算事件。它不读取未来到达、剩余 epoch 或真实完成时间。

## 8. 两层适配算法

### 8.1 A：同构 AFS-P 核心

实现一个不依赖 GPU 类型和物理节点的纯函数，输入：

- 作业集合；
- 每个作业的整数份额 Goodput 曲线 `p_j(0..k_max)`；
- 总 GPU 数 `M`；
- processor-sharing 累计执行时间、上次被派发时的计数器；
- 2 小时时间片和固定随机种子。

输出虚拟整数份额 `s_j`。`n < M` 时先给每个可运行作业一个 GPU，再逐 GPU执行修改后的双向式 (3)；`n >= M` 时按论文的非抢占 processor-sharing 规则选择至多 `M` 个作业，每个一个 GPU。

这一层将用单类型曲线做单元测试，是尽量忠实于论文的 AFS-P 参考部分。

### 8.2 B：Sia 异构映射

主策略 `--policy=sia_afs` 在总物理 GPU 数形成的虚拟池上先运行 AFS-P 核心，再把 `s_j` 映射为 Sia 合法配置：

1. 对每个作业、GPU 类型和合法配置计算可观测/可投影的 Goodput；未建模区间按论文冷启动规则线性外推。
2. 通过整数规划为每个目标份额为正的作业选择至多一个 `(GPU 类型, 节点数, GPU 数)` 配置。
3. 硬约束每类 GPU 容量、每作业单一配置、scale unit、max replicas，以及 checkpoint/restore 中作业保持原配置。
4. 主目标最小化合法配置 GPU 数与 AFS-P 虚拟份额的绝对偏差；整数偏差相同才优先更充分利用不超过虚拟总份额的 GPU，再按原论文 Section 4.4 的顺序优先复用既有 GPU 类型/规模，最后用 Goodput 作异构候选的确定性 tie-break。
5. 最后调用 Sia 原 placement 例程映射到物理节点；实际重配仍由 `Job.reallocate()` / `Job.migrate()` 收取原始 checkpoint/restart 成本。

这类似原论文“先算份额、再做 network-packing regulation”的分层，但下列部分是本实验新增，论文没有给出：

- 把三个异构 GPU 池合成虚拟总容量；
- 用各类型可用 Goodput 的包络构造虚拟份额曲线；
- 用整数规划同时选择 GPU 类型和合法 Sia 配置；
- checkpoint 中配置硬保持；
- 对 Sia 在线 profile 报告增加重算事件。

异构映射不会把 `policy_p_val`、`mip_lambda_a` 或 `mip_lambda_n` 混入 AFS-P 份额公式；这些是 Sia 目标参数，不属于 AFS。

## 9. 可实现性与停止判断

论文足以确定以下核心：两阶段结构、式 (3) 的精确不对称比率、未知长度时的双向判断/随机不确定项、processor-sharing 计数与非抢占时间片、2 小时默认 unit，以及份额后合法化的分层方式。因此可以实现一个可审计的 AFS-P 核心。

论文没有定义异构 GPU 比较和配置分配。该缺口不会被伪装成原算法；它由第 8.2 节明确的新映射层补足。若测试发现映射不能同时满足容量、合法配置、checkpoint 保持或无未来信息约束，则停止正式实验并把结论标为 `INVALID`。

## 10. 审计后的实现与验证状态

论文审计确认核心规则足以实现，因此进入独立策略开发。最终位置如下：

- `sia-afs-experiments/sia-simulator/sia_afs.py:28-172`：scheduled-wall-time 计数器、论文式 (3)、长度未知时的双向比较/随机选择，以及同构 AFS-P 参考分配器；
- `sia-afs-experiments/sia-simulator/sia_afs.py:175-313`：`SiaAFSPolicy`、事件状态、冷启动公平份额与异构虚拟曲线；
- `sia-afs-experiments/sia-simulator/sia_afs.py:315-445`：合法候选和异构映射 MIP；
- `sia-afs-experiments/sia-simulator/sia_afs.py:447-598`：物理 placement、事件式优化与结构化审计记录；
- `sia-afs-experiments/sia-simulator/cluster.py:396-400`：把既有 `execution_time` 和 profile 点数只读附加到策略输入；
- `sia-afs-experiments/sia-simulator/multi_simulator.py:38-49,145-170`：独立 `--policy=sia_afs` 入口和 AFS 参数。

13 项测试全部通过，覆盖同构核心、`n >= M`、非抢占时间片、冷启动、异构合法配置、容量、无法映射、checkpoint 保持和事件更新。原始 `--policy=sia` 的 20 作业 JSONL 与已有基线字节一致。

三条正式 Saturn trace 均完成 160/160；共 1551 次实际异构映射全部返回 `optimal`，没有未映射作业或容量违规。因此实现没有触发本节预设的 `INVALID` 停止条件。需要限定的是：正式 trace 每轮活跃作业数都小于 64，全部状态记录均为 `efficiency`，没有在完整性能实验中触发 `n >= M` processor-sharing 分支；该分支只由测试验证。性能结论与后续判断见 `research_notes/SIA_AFS_EXPERIMENT_REPORT.md`。

