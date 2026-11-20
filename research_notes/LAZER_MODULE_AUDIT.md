# Lazer 原论文与可组合机制审计

- 审计日期：2026-10-23（Asia/Shanghai）
- 论文：Younghun Go 等，*Making Sense of Job Preemption for Distributed Deep Learning Acceleration*，DAC 2026
- 作者实验室页面：[Korea University Systems Software Lab](https://ss.korea.ac.kr/publication/making-sense-of-job-preemption-for-distributed-deep-learning-acceleration-2026/)
- 论文 PDF：[作者站点 PDF](https://ss.korea.ac.kr/papers/Making-Sense-of-Job-Preemption-for-Distributed-Deep-Learning-Acceleration.pdf)
- 本地核验副本：`research_sources/Lazer/Making-Sense-of-Job-Preemption-for-Distributed-Deep-Learning-Acceleration.pdf`
- 本地 PDF SHA-256：`EBED5768E8E19EE79F3294105B5125F56CA7281E9D9FB86E7A8A6E43DE387881`

## 1. 结论摘要

Lazer 不是一个“在任意调度器外层禁止短期二次抢占”的简单 Guard。论文方法由以下三部分共同组成：

1. 以剩余运行时间为优先级、按固定 GPU demand 选择被抢占 Job 集合的 scheduling orchestrator；
2. 以贝叶斯优化预测推迟时间 `X`，并用 DWQ/DRQ 与独立 deferral thread 改变调度发生的时刻；
3. 在 Job 完成时立即从 WQ 按最短剩余时间填充释放 GPU 的 completion handler。

其中，剩余运行时间、独立的 pause-and-save/load 阶段、到达/完成事件触发和固定 GPU demand 都是方法语义的一部分。把 Lazer 简化为任意 Goodput 权重或固定冷却时间，会脱离论文算法，不能称作 Lazer 移植。

对当前 Sia Artifact，可独立提取且不改变调度器的只有**观测机制**：记录重启、combined checkpoint/restore 倒计时、有效训练与浪费 GPU-seconds。Lazer 的 orchestrator、BO deferral 和 completion handler 都会改变调度时机或替换 Sia MIP 的配置选择逻辑。

## 2. 论文对 futile preemption 的严格定义

论文第 2 页式 (1) 把 Job completion time 分解为：

```text
JCT_k = W_k + L_k + T_k + P_k
```

- `W_k`：等待；
- `L_k`：从存储加载 Job/model 到 GPU；
- `T_k`：有效训练；
- `P_k`：pause-and-save。

论文第 3 页式 (2) 定义的 futile preemption 场景为：运行中的 `j_k` 被更短的 `j_{k+1}` 抢占，`j_{k+1}` 完成加载并开始训练前，更短的 `j_{k+2}` 到达并再次抢占它。条件是：

```text
T'_{k+2} < T'_{k+1} < T'_k
p_k < ΔA_{k+2} <= p_k + l_{k+1}
```

其中 `T'` 是剩余运行时间，`p_k` 是被抢占 Job 的 pause-and-save 时间，`l_{k+1}` 是新 Job 的加载时间，`ΔA` 是到达间隔。论文的“无效”不是泛指重启开销大，也不是“相邻两轮发生配置变化”；它要求更高优先级 Job 在前一个新 Job 的加载窗口内到达，并使该 Job 在完成有效训练前再次被抢占。

## 3. Lazer 的完整调度流程

### 3.1 使用的状态

论文第 3–5 页使用：

- RQ：正在运行的 Job，以及每个 Job 的剩余运行时间 `T'` 和 GPU demand；
- WQ：等待资源的 Job；
- DWQ：正在等待 deferral 到期的新 Job；
- DRQ：被选作候选、但在 deferral 期间继续运行的 Job；
- 最近一小时平均到达间隔 `ΔĀ`；
- 新 Job 的剩余运行时间和 load time；
- 候选被抢占 Job 中最大的 pause-and-save time；
- 当前空闲 GPU 数及每个 Job 固定 GPU demand。

### 3.2 Algorithm 1：到达与完成事件

论文第 4 页 Algorithm 1 可忠实转写为：

```text
on_arrival(j_new):
    J_r = scheduling_orchestrator(j_new)
    if J_r is empty:
        enqueue j_new into WQ
        return

    X = deferral_predictor(j_new, J_r)
    if X == 0:
        pause-and-save every job in J_r
        enqueue victims into WQ
        load j_new
    else:
        launch_deferral_thread(j_new, J_r, X)
        # thread: j_new -> DWQ, J_r -> DRQ; J_r keeps running; sleep X
        # then restore J_r to RQ, remove j_new from DWQ and re-evaluate
        J_r2 = scheduling_orchestrator(j_new)
        if J_r2 is empty:
            enqueue j_new into WQ
        else:
            pause-and-save every job in J_r2
            enqueue victims into WQ
            load j_new

on_completion(j_running):
    secured_gpus = GPUs released by j_running
    sort WQ by ascending remaining duration
    for j in WQ:
        if demand(j) <= secured_gpus:
            load and start j
            secured_gpus -= demand(j)
```

Deferral 期间 DWQ/DRQ 中的 Job 不参与其他新到达 Job 的候选选择，从而避免同一候选集合被并发到达事件反复重排。到期后必须重新运行 orchestrator，因为剩余时间、完成状态和资源状态已经变化。

### 3.3 Algorithm 2：抢占集合选择

论文第 4 页 Algorithm 2：

```text
scheduling_orchestrator(j_new):
    J_r = empty
    avail_gpus = 0
    sort RQ by descending remaining duration
    for j_running in RQ:
        if remaining(j_new) < remaining(j_running)
           and avail_gpus < demand(j_new):
            add j_running to J_r
            avail_gpus += demand(j_running)
    if avail_gpus < demand(j_new):
        return empty
    return J_r
```

这不是全局 Goodput 最大化：它按 SRTF 式剩余时长优先级，从最长的运行 Job 起收集“刚够”新 Job demand 的 GPU。

## 4. BO 具体优化对象和参数

论文第 4 页式 (3)–(5)：

```text
X = argmax_{X in [l,h]} P(f(X|c) = 0)
f(X|c) = |X - F|
c = {ΔĀ, T'_{j_new}, l_{j_new}, max_{k in J_r} p_k}
```

- `F`：若立即调度，新加载 Job 在开始训练前会被再次抢占的预期持续时间；
- 搜索范围：`l=0`, `h=100` 秒；
- 到达统计窗口：过去 1 小时；
- surrogate：Gaussian Process；实现采用 BoTorch `SingleTaskGP`；
- acquisition：实现采用 `LogNoisyExpectedImprovement`；
- bootstrap：Latin hypercube sampling，`M=10`；
- acquisition：5 个随机起点，对 acquisition function 做 L-BFGS；
- 进入 exploitation 的条件：GP 更新达到 100 次且选中点 EI 低于 10%；
- exploitation：直接对 GP 用 L-BFGS 求 `X`。

因此 BO 优化的是**推迟抢占的时间**，不是 GPU 份额、Goodput 权重、重启惩罚或 MIP 系数。

## 5. 调度时机变化

Lazer 对新 Job 到达和 Job 完成即时响应；`X>0` 时还在任意秒数后由独立 thread 重新触发调度。论文把 Pollux/Sia 的 60 秒周期调度导致的等待作为第二个主要问题。

这与当前 Sia 模拟器的“每 60 秒先推进所有 Job，再统一求一次 MIP”不同。要忠实实现 Lazer，必须引入事件驱动时间推进、DWQ/DRQ/WQ/RQ 状态和剩余时长优先级；它不是在现有一次 MIP 求解之后撤销若干迁移。

## 6. 可提取机制与不可独立提取部分

| 论文机制 | 能否作为 Sia 的小模块独立提取 | 原因 |
|---|---|---|
| 重启/load/有效训练观测 | 可以 | 只读状态，不改变 Sia 目标、约束或时机。 |
| 固定的“刚重启不可再迁移”约束 | 不能称为 Lazer | 论文不是固定 cooldown；`X` 是上下文 BO 预测，且受 DWQ/DRQ 语义约束。 |
| Scheduling orchestrator | 需要替换核心选择逻辑 | 使用剩余时间优先级和固定 demand，不使用 Sia Goodput MIP。 |
| BO deferral predictor | 需要新增完整状态机与事件时钟 | 依赖 `F` 在线反馈、独立 pause/load 阶段和到达事件。 |
| Completion handler | 需要改变调度时机 | Job 一完成就调度，而非等下一 60 秒边界。 |
| 仅在 MIP 中加 preemption Guard | 可以设计为新组合方法，但不是 Lazer 原方法 | 必须在求解前形成约束并重新求解；事后撤销可能导致容量不一致。 |

若未来数据达到 PASS 门槛，最小的 Sia 组合设计应保持原 Goodput 目标，在求解前把被保护 Job 的旧配置作为硬等式约束，然后重新求解整个 MIP；Guard 关闭时不加约束。该设计只能称为“受 Lazer 无效抢占问题启发的 Sia Guard”，除非也实现论文的 BO、队列状态和事件触发。

## 7. 原论文假设与 Sia 的差异

| 维度 | Lazer | 当前 Sia Artifact |
|---|---|---|
| 优先级 | 已知/可用的剩余运行时间，SRTF | Goodput + `p` fairness 的异构配置 MIP |
| 资源请求 | Job GPU demand | 每轮在 GPU 类型、节点与合法副本配置中联合选择 |
| GPU 比较 | 以 GPU 数量满足 demand；算法没有异构 Goodput 选择步骤 | 不同 GPU 类型的 throughput/Goodput 可比 |
| 执行开销 | pause-and-save 与 load 分开 | 一个 combined checkpoint/restore 倒计时 |
| 调度触发 | 到达、完成、deferral 到期 | 固定周期，默认 60 秒 |
| 抢占控制 | WQ/RQ/DWQ/DRQ + BO | 重分配 soft factor + checkpoint 期间 hard equality |

论文物理平台虽然包含多种 GPU 型号，但 Algorithm 1/2 的资源条件仍按 GPU demand 数量表达；论文没有给出把异构型号和弹性配置映射到 Goodput MIP 的算法。

## 8. 公开源码核查

截至本次审计：

- 作者实验室论文页面仅提供论文链接，没有 source-code/GitHub 链接；
- PDF 中没有代码仓库 URL；
- 以论文标题、作者、`Lazer futile preemption GitHub` 等关键词检索，未找到可验证为作者发布的实现；
- 论文只说明 Lazer 为约 10K 行 Python，BO 基于 BoTorch。

结论：**没有找到可靠的作者公开源码**。因此不能用第三方同名仓库替代论文算法，也不能声称完成源码级复现。

## 9. Evaluation 与消融边界

- 物理实验：16 块异构 GPU，Earth trace 中抽样 100 个 Job；指标含 JCT、等待、futile preemption time、GPU utilization。
- 大规模模拟：论文所述 Earth/Saturn/Uranus 全量生产 trace（数十万 Job），不是 Sia Artifact 的三个 160-Job Saturn CSV。
- Figure 8(c) 比较完整策略的 futile preemption time；Figure 8(e) 只比较 BO、random search 和 grid search 对 `f(X|c)` 的搜索效率；Figure 8(f) 比较调度延迟。
- 论文没有给出“仅 orchestrator / 仅 deferral / 仅 completion handler”的完整组件消融表。因此不能从 Figure 8(e) 推断各组件分别贡献多少 JCT 收益。

## 10. 对本任务的约束结论

后续是否实现只能由当前 Sia 数据中是否存在可避免的目标事件决定。现有事件审计显示，在三个预注册 trace 上，Sia 的 checkpoint/restore hard equality 已经阻止倒计时未结束时再次改配；因此本轮不具备实现 Lazer-inspired Guard 的证据门槛。具体统计见 `SIA_PREEMPTION_EVENT_AUDIT.md`。
