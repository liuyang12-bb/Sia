# Sia 抢占事件与 Lazer 适用性审计

- 审计日期：2026-10-23（Asia/Shanghai）
- 原始 Artifact：`official-artifacts/`，commit `aaf7a0de7b6cb5eab06ea020467ede5d2fd18e70`
- 观测 worktree：`sia-lazer-observe/`，分支 `exp/sia-lazer-observe`
- 输入：已保存的原始 Sia `workload-1/5/10.log`，没有重跑完整 trace
- 最终门槛判定：**NO-GO**

## 1. 判定摘要

三个预注册 Saturn trace 共观察到 2,100 次重启，combined checkpoint/restore 损失为 849,520 GPU-seconds（235.98 GPU-hours）。这些是实际执行模型中的重启开销，但**不能整体视为 Lazer 可避免损失**。

排除“节点 ID 变化但 placement 不变、`num_restarts` 不增”的调度日志变化后：

- checkpoint/restore 倒计时结束前再次配置变化：三条 trace 均为 0；
- 重启后尚未观察到有效训练就再次配置变化：三条 trace 均为 0；
- 恰好在倒计时结束边界、尚无进度就再次配置变化：三条 trace 均为 0。

原因与源码一致：`preserve_ckpt_allocs=True`，`rescale_time>0` 的 Job 被加入 non-preemptible 集合，并由 MIP 等式强制保持旧配置。因而当前模拟执行模型中，没有为“再加一层相同性质 Guard”留下可测的目标事件。

保存的 JSONL 无法严格计算论文式 (2) 的 Lazer futile preemption 数，因为它没有独立 pause/load 时间、剩余时长优先级或决策原因。该严格指标的**数据状态是 UNKNOWN**；但针对“是否值得实现最小 Sia Guard”的工程门槛是 **NO-GO**：源码硬约束已经覆盖 combined rescale 窗口，三条 trace 的可观察候选为零。

固定 60 秒轮次导致约 26–32 秒的平均到达/完成边界等待，确实存在；解决它需要事件驱动调度和时间推进改造，属于完整调度器替换，不是本任务允许的最小抢占保护模块。

## 2. 基线保护

执行前与最终均确认：

- `official-artifacts` 保持指定 commit，`git status --short --branch` 为 `## main...origin/main`；
- `sia-experiments` 仍仅有 LAS 的既有修改；
- `sia-afs-experiments` 仍仅有 AFS-P 的既有修改；
- 三条原始 Sia 日志只读复用，没有覆盖；
- 新代码只在 `sia-lazer-observe/`；根目录新增的只有本轮报告、统计、观测 smoke 与运行日志；
- 没有创建 `--policy` 新入口，没有修改 Sia 目标、容量约束、执行推进或历史实验。

输入日志哈希：

| Trace | 路径 | SHA-256 |
|---|---|---|
| workload-1 | `research_runs/sia_afs/full/sia/workload-1.log` | `C2EC4462ED57135770BCD33D4DB41CE91990E63DE03B2D8D6C3B29BB52DD699C` |
| workload-5 | `research_runs/sia_afs/full/sia/workload-5.log` | `1630F335C99FA7B937B30D7DCE1DCF8DC965AB969C493C744BE73BC4021B18D2` |
| workload-10 | `research_runs/sia_afs/full/sia/workload-10.log` | `E8A1713EBAD253EAEF7987720243B3D55756F00478603ADED0323F2B9B3797F8` |

## 3. 原始 Sia 抢占机制源码审计

### 3.1 restart factor：软偏好，不是硬禁止

`official-artifacts/sia-simulator/sia.py`：

- `get_restart_penalty()`，23–35 行：按应用返回策略估计的 restart penalty；
- `_compute_rigid_job_goodput_single()`，527–553 行：rigid Job 用累计 restart loss 与 age 构造 `realloc_factor`；
- adaptive 路径，721–748 行：当前配置 `sia_use_pollux_realloc_factor=True`，实际使用
  `max(age - num_restarts*penalty, 0)/(age+penalty)`；关闭该选项时才走 `used_gpu_seconds/(used+wasted+penalty)` 分支；
- `__solve_mip_rescaled()`，844–1031 行：已有配置保持原 Goodput，其他类型/配置乘 `realloc_factor`（874–900 行）。

因此 factor 只是让变更配置的候选收益变小，仍允许 MIP 在收益足够高时重分配。`migrate_factor` 被设为相同值，Artifact 没有独立的迁移代价模型。

策略估计和执行开销并不完全相同：`get_restart_penalty()` 对 CIFAR10/YOLOv3 使用 30/45 秒，而 `applications.py:56-69` 的实际 combined rescale time 是 50/80 秒。本轮不修改这一历史行为。

### 3.2 checkpoint/restore：硬保持约束

- `simulator_config.py:27`：`preserve_ckpt_allocs=True`；
- `cluster.py:get_sia_multi_job_info()`，373–432 行；425 行令 `scaling_underway = (job.rescale_time > 0)`；
- `sia.py:763-785`：把 scaling Job 收入 rigid/adaptive non-preemptible 集合；
- `sia.py:1011-1028`：MIP 对这些 Job 加 `X[旧配置] == 1` 等式。

这与一个候选 Guard 的区别是：原约束只在 combined checkpoint/restore 倒计时为正时生效；若另加 Guard，必须保护倒计时结束后的额外窗口，并会排除一部分 MIP 配置。正确实现只能在求解前加入等式并重新求解，不能先求解再事后撤销，否则释放/占用资源可能不一致。

### 3.3 哪些变化触发执行事件

`cluster.py:optimize_policy()`，160–225 行：

- 分配映射变化且 GPU 类型变化：调用 `Job.migrate()`；
- 同 GPU 类型下分配变化：调用 `Job.reallocate()`；
- `Job.reallocate()` 内部只有 placement tuple 真正变化时才重置 rescale time 并增加 restart；
- 节点 ID 改变但压缩后的 placement 相同，可能记录 rescale event，但不会增加 `num_restarts`，不得当作重启。

`job.py:524-580`：

- 第一次非空分配把 `num_restarts` 从 `None` 设为 0；以后真实非空重分配或迁移才加 1；
- deallocation 只清空 placement，不增加 restart；后续 resume 会产生 restart；
- migrate 与 GPU 数量变化都汇总到 `num_restarts`。

`num_migrations` 在 `job.py:86` 初始化后没有任何递增赋值，不能直接用作类型迁移统计。本轮从相邻重启配置的 `cluster` 与 GPU 数量回溯分类。

### 3.4 开销与有效训练

`job.py:414-518` 的 `Job.step()`：

1. `delay=min(rescale_time, seconds)`；
2. 这段时间增加 `attained_service` 和 `wasted_gpu_seconds`，不增加训练进度；
3. 倒计时归零后，本轮剩余秒数才增加 `used_gpu_seconds`、`execution_time` 与训练进度。

`applications.py:56-69` 的 combined checkpoint+restore 时间：CIFAR10 50 秒、DeepSpeech2 25 秒、BERT 120 秒、YOLOv3 80 秒、ImageNet 250 秒、NCF 15 秒、默认 30 秒。模拟器没有把 pause-and-save 与 load 分开，故不能直接验证 Lazer 式 (2) 的 load-window 条件。

### 3.5 固定 60 秒轮次

- `multi_simulator.py:65-68,152-153`：默认 `interval=60`，循环调用 `cluster.step(interval)`；
- `cluster.py:117-158`：先把所有 Job 推进整轮，再增加集群时间，然后调用策略；
- 到达与完成不会在精确事件时间立即触发 MIP，新 Job 和释放资源在下一个边界被策略看见。

### 3.6 原日志能力与缺口

`cluster.py:step_log()`，306–338 行只记录 progress、累计 restart、cluster、allocation、placement、submission/completion 等。`output_logs()`，513–517 行只写 `self.logs`。

缺失项：

- 当前 `rescale_time`；
- `used_gpu_seconds` / `wasted_gpu_seconds` / `execution_time`；
- 本轮 `reallocate()` / `migrate()` 事件；
- pause/save 与 load 的独立起止；
- 剩余运行时间、优先级和触发此次决策的到达 Job；
- `start_time` 虽在 schema 中，但源码没有赋值，日志中为 `null`。

因此，保存日志能精确还原 60 秒边界的 allocation、progress 与累计 restart，能够依据固定应用开销重建 Sia 模型中的 combined rescale；但不能把任何粗粒度候选宣称为论文严格 futile preemption。

## 4. 统计方法

仓库外脚本：`research_notes/audit_sia_preemption_events.py`。

- 以 `num_restarts` 增量识别真实 restart episode；
- 初始启动与 restart 分开；
- 相邻记录的节点 ID 变化只有在 restart 增量时才开始新的 load episode；
- 使用 `applications.py` 的固定 combined rescale time 重建浪费 GPU-seconds；
- 以 progress 差判断“是否观察到有效训练”；
- 到达/完成边界等待由 trace 的精确 submission/completion time 与日志 timestamp 计算；
- 严格 Lazer 计数固定输出 `null / UNKNOWN_FROM_SAVED_JSONL`。

每个 Job 的累计 restart、每次 allocation change、每个 restart episode、到下一变化/完成的时间、progress 差和重建开销均保存在：

- `research_notes/run_logs/sia_preemption_workload1.events.json`
- `research_notes/run_logs/sia_preemption_workload5.events.json`
- `research_notes/run_logs/sia_preemption_workload10.events.json`
- 汇总：`research_notes/run_logs/sia_preemption_event_summary.json`

三条 trace 的“最终 restart 总和”与“识别到的 restart episode 数”分别为 659=659、692=692、749=749，内部一致。

## 5. 三条 trace 的真实事件统计

### 5.1 重启与损失

| 指标 | workload-1 | workload-5 | workload-10 |
|---|---:|---:|---:|
| Job 数 | 160 | 160 | 160 |
| restart 总数 | 659 | 692 | 749 |
| restart/Job 均值 | 4.11875 | 4.32500 | 4.68125 |
| restart/Job 中位数 | 3 | 3 | 4 |
| restart/Job P95 / P99 / 最大 | 10 / 16 / 25 | 12 / 18 / 23 | 11 / 19 / 23 |
| restart combined 开销 (GPU-s) | 262,590 | 285,065 | 301,865 |
| restart combined 开销 (GPU-h) | 72.9417 | 79.1847 | 83.8514 |
| restart 后有效训练秒数中位数 | 130 | 130 | 130 |
| restart 后有效训练秒数最小值 | 1 | 1 | 3 |
| 倒计时结束前再次改配 | 0 | 0 | 0 |
| 无有效训练即再次改配 | 0 | 0 | 0 |
| 倒计时边界无进度即改配 | 0 | 0 | 0 |

“restart 后有效训练秒数”按 episode 持续时间减固定 rescale time 重建；最小值仍大于 0，与 hard protection 的预期一致。

### 5.2 GPU 类型与数量变化

三类互斥计数：

| 变化类型 | workload-1 | workload-5 | workload-10 |
|---|---:|---:|---:|
| GPU 类型与数量同时变化 | 342 | 280 | 383 |
| 仅 GPU 类型变化 | 44 | 109 | 73 |
| 仅 GPU 数量变化 | 273 | 303 | 293 |
| 合计 | 659 | 692 | 749 |

类型变化占比（含“类型+数量”）为 58.57%、56.21%、60.88%；数量变化占比（含“类型+数量”）为 93.32%、84.25%、90.25%。二者可重叠，不能相加为 100%。

### 5.3 固定轮次造成的可观察等待

| 指标 (s) | workload-1 | workload-5 | workload-10 |
|---|---:|---:|---:|
| submission 到首次调度边界，均值 | 29.42 | 28.71 | 31.79 |
| 同上 P95 / P99 | 56 / 58 | 57 / 59 | 56 / 58 |
| submission 到首次分配，均值 | 30.17 | 31.71 | 32.16 |
| completion 到被边界观察，均值 | 25.63 | 26.61 | 28.70 |
| completion 观察延迟 P95 / P99 | 54 / 59 | 55 / 59 | 57 / 59 |

这是周期调度的量化损失，但不是 futile preemption GPU 浪费。把它消除需要到达/完成事件驱动的 scheduler activation，可能改变所有 Job 的决策时刻，不能包装为最小抢占 Guard。

## 6. 可避免 GPU 时间损失的边界

- 可确认的 restart combined 损失：72.94 / 79.18 / 83.85 GPU-hours；
- 可归因于“倒计时未结束又被改配”的候选损失：三条 trace 均为 0；
- 可归因于“未有效训练即再次改配”的候选上界：三条 trace 均为 0；
- 论文严格 futile preemption 损失：原 JSONL 无法直接计算，不能把全部 restart loss 当作其上界或收益空间。

对当前模拟器语义，combined rescale 期间的再次抢占被硬约束禁止。因此候选 Guard 的可测量收益空间为 0；正常重启开销只能通过改变 Sia 的配置稳定性/目标函数来减少，那是另一项算法研究，不是 Lazer 的 deferral 机制验证。

## 7. 只读观测埋点

因为历史 JSONL 缺少精确状态，在原始 commit 上建立独立 worktree：

```text
sia-lazer-observe/  exp/sia-lazer-observe
```

改动：

| 文件 | 行 | 内容 |
|---|---:|---|
| `sia-simulator/cluster.py` | 15–37 | `get_preemption_audit_job_state()`，输出 phase、倒计时、attained/used/wasted GPU-seconds、execution time、配置与 restart。 |
| `sia-simulator/cluster.py` | 53 | 默认 `log_preemption_audit=False`。 |
| `sia-simulator/cluster.py` | 333–375 | 开启时只追加顶层 `preemption_audit`，包含本轮 rescale/migrate events 与每 Job 状态。 |
| `sia-simulator/multi_simulator.py` | 49,156–157 | 新增默认关闭的 `--log_preemption_audit`。 |
| `sia-simulator/tests/test_preemption_audit.py` | 1–77 | 3 项观测与 schema 回归测试。 |

这已经是“不改变执行语义”的最大观测范围。要记录论文严格事件，还需把 Sia 的 combined rescale 拆成 pause/save 和 load 两个阶段，并记录剩余运行时间优先级与到达触发原因；这会新增当前模型不存在的状态，已超出只读埋点范围，因此本轮没有实施。

## 8. 回归测试与观测 smoke

单元测试：

```powershell
D:\科研启航\Sia\.venv-sia-baseline\Scripts\python.exe -m unittest discover -s sia-simulator\tests -p test_preemption_audit.py -v
```

- 3/3 通过，退出码 0；
- 覆盖 phase 分类、默认 schema 不变、开启后仅追加顶层字段；
- 日志：`research_notes/run_logs/sia_lazer_observe_unit_tests.stderr.log`；
- 元数据：`research_notes/run_logs/sia_lazer_observe_unit_tests.run.json`。

使用固定 20 Job trace 顺序运行原始 Sia：

```powershell
# 关闭埋点
python multi_simulator.py <first20.csv> --policy=sia --policy_p_val=-0.5 `
  --mip_lambda_a=0.01 --mip_lambda_n=1.1 --interval=60 `
  --project_throughputs --share_max_replicas --output=<off.log>

# 开启埋点，只多一个参数
python multi_simulator.py <first20.csv> --policy=sia --policy_p_val=-0.5 `
  --mip_lambda_a=0.01 --mip_lambda_n=1.1 --interval=60 `
  --project_throughputs --share_max_replicas --log_preemption_audit `
  --output=<on.log>
```

| 运行 | 退出码 | 耗时 | 轮数 | 结果 |
|---|---:|---:|---:|---|
| audit off | 0 | 79.589 s | 149 | 20/20 完成 |
| audit on | 0 | 77.595 s | 149 | 20/20 完成 |

- off 日志 SHA-256：`FE5B7EF8D5C9D8A759B017F0EE9866D14415DB14B91E8A12572F23E0813D4B9A`；
- 它与既有 LAS 报告中的原始 Sia small-trace 日志字节完全一致；
- on 日志移除每轮新增的 `preemption_audit` 后，149/149 个 JSON 对象与 off 完全一致；
- on 记录 2,359 个 Job snapshot、70 个 rescale event、63 个 migrate event；最终累计 `used_gpu_seconds=205,710`、`wasted_gpu_seconds=27,030`。

两次包装失误被保留说明：第一次 Python 相对路径错误，模拟器未启动；第二次模拟完成但输出目录相对解析错误，写 JSONL 时退出 1。二者均未修改源码或历史日志，也未用于比较；成功运行使用绝对路径。

## 9. 最终判定：NO-GO

按任务给定标准：

- 不是 PASS：没有发现足够多、能被候选 Guard 避免的无效抢占；
- 不是 WEAK：可观察候选不是“少量”，而是三条 trace 全为 0；
- 严格论文计数的数据字段仍是 UNKNOWN，但当前模拟器硬约束和精确 restart/progress 代理共同表明目标机制在此执行模型中不存在；
- 因此对“是否实现最小组合”的决策是 **NO-GO**。

本轮停止于机制验证：

- 不增加 `--policy=sia_lazer` 或 Guard；
- 不创建 `SIA_PREEMPTION_GUARD_SMOKE.md`；
- 不运行完整 trace 新实验；
- 不做参数搜索；
- 不把固定轮次等待问题偷换成抢占 Guard 收益。

若未来转向真实集群或一个显式区分 pause/save 与 load、允许 load 中被抢占的模拟器，应先用本报告列出的最小字段重新审计；只有严格事件数量和可避免 GPU 时间达到 PASS 门槛，才值得设计 Lazer-inspired 组合机制。
