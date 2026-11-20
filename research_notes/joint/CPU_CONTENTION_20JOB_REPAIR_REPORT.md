# B3 20-job 执行失败修复与五策略重跑报告

## 1. 结论先行

本轮已完成根因定位、边界修复、失败审计补齐、8 项新增测试、B3 单策略针对性验证，以及冻结五策略矩阵重跑。最终状态为：

**`REPAIR_VALIDATED — FROZEN_NEUTRAL_5/5_COMPLETE`**

- 历史失败不是 SciPy MILP 的 `result.x is None`，而是可复现的 **`NO_COLUMNS`**。
- 根因是一个未分配作业仍残留 `scaling_underway=True`，联合接入层把它误判成 checkpoint 锁；由于该作业没有当前配置和节点，所有合法 GPU 候选都被“锁定配置”过滤掉。
- 修复只收紧 checkpoint 锁的成立条件，并为 `NO_COLUMNS`、MILP 无可用解和安全保留上一轮分配补充显式审计/回退语义；没有修改 B3 的目标函数、候选 Goodput、资源容量、workload 或响应参数。
- B3 针对性 20-job 运行完成 20/20，专项验证全部通过。
- B1、SY、B4、B5、B3 完整矩阵全部完成 20/20；全矩阵验证器 21 项检查全部通过。
- 完整测试套件为 **38/38 passed**。

本报告中的 JCT 仅是冻结 Neutral 合成资源模型下的稳定性证据。它不构成真实 Saturn 或真实异构 GPU 集群性能结论。

> **SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE**

## 2. 工程保护与版本

- 工作区：`D:\科研启航\Sia`
- 实验仓库：`sia-joint-experiments`
- 分支：`exp/synergy-baselines-cpu-bottleneck`
- HEAD：`aaf7a0de7b6cb5eab06ea020467ede5d2fd18e70`
- 官方仓库：`official-artifacts`，`main...origin/main`，检查时干净。
- 实验仓库在本轮开始前已有未提交的联合调度代码和历史修改；没有 reset、clean、覆盖或提交这些用户工作。

历史证据均保留：

- 初始内存门槛停止：`research_runs/sia_synergy/cpu_contention_20job/`
- 原 B3 失败矩阵：`research_runs/sia_synergy/cpu_contention_20job_run1/`

本轮只新增三个独立目录：

- 根因复现：`research_runs/sia_synergy/b3_20job_failure_diagnostic/`
- B3 针对性运行：`research_runs/sia_synergy/cpu_contention_20job_b3_repair_targeted/`
- 五策略完整重跑：`research_runs/sia_synergy/cpu_contention_20job_repair_full/`

由于联合实验文件在当前 Git HEAD 下仍是未跟踪文件，普通 `git diff --stat` 只显示历史已有的 `job.py` 修改，不能完整表示本轮差异。本报告因此按文件、函数、行号和 SHA-256 明确列出本轮改动。`job.py` 的既有修改不是本轮产生的。

## 3. 失败根因

### 3.1 修复前独立复现

在修改调度语义前，使用原冻结输入单独复现 B3，结果保存在 `b3_20job_failure_diagnostic/`：

| 字段 | 观测值 |
|---|---:|
| 状态 | `REPRODUCED_FAILURE` |
| 模拟时间 | 32880 s |
| policy round | 489 |
| 已完成作业 | 19/20 |
| `result_status` | `NO_COLUMNS` |
| 总候选列 | 0 |
| 活动作业 | `bert-37` |
| `bert-37` 合法 GPU 配置数 | 3 |
| `checkpoint_locked` | `true` |
| `current_config_id` | `null` |
| `current_nodes` | `[]` |

因此可以排除本次故障属于 `result.x is None`。它也不是“部分作业无列、其他作业仍可运行”：该轮只有一个活动作业，且该作业的三种合法 GPU 配置全部被错误锁过滤。

### 3.2 代码链路

1. 原始 `Job.reallocate(())` 在 `job.py:530` 附近清空 `placement` 和 `atomic_bsz`，但不会清零仍在倒计时的 `rescale_time`。
2. `cluster.py:425` 通过 `job.rescale_time > 0` 设置 `JobInfo.scaling_underway`。
3. 修复前 `_job_specs()` 直接把 `scaling_underway` 解释为 `checkpoint_locked`，即使作业已经没有当前配置和节点。
4. 修复前 `build_joint_columns()` 对锁定作业要求候选配置等于 `current_config_id`。此时其值为 `None`，所以三种合法配置全部被过滤。
5. `JointScheduler.solve()` 返回 `NO_COLUMNS`，但旧失败分支的 `audit={}`；原子提交保护正确拒绝了这个无审计结果，最终抛出 `RuntimeError: refusing non-atomic joint allocation: {}`。

这是一项联合接入层的“幽灵 checkpoint 锁”错误，不是原 Sia 的 checkpoint 倒计时本身错误。原子保护拒绝提交空审计是正确行为；错误发生在锁状态构造及失败信息缺失处。

## 4. 修复设计

### 4.1 锁语义

真实配置锁现在必须同时满足：

```text
scaling_underway
AND current_config_id is not None
AND current_nodes is not empty
```

实施位置：

- `joint_prototype/joint_integration.py:222`：从实际 JobInfo 和已有分配构造锁。
- `joint_prototype/joint_scheduler.py:238`：联合候选列过滤。
- `joint_prototype/joint_scheduler.py:328`：B1 候选列过滤使用同一防御条件。

这不会跳过真实 checkpoint 锁；它只拒绝把“无当前分配”的作业解释为必须维持一个不存在的配置。

### 4.2 失败审计

`JointScheduler._failure_audit()`（`joint_scheduler.py:384`）现在为每个失败分支记录：

- `failure_reason`；
- 求解器 status、success、message、objective、MIP gap/node count（若可用）；
- 候选列总数和逐作业候选数；
- 活动作业、无候选作业；
- 当前配置和节点；
- checkpoint 锁定作业及锁定配置；
- 优化维度。

状态已明确区分：

- `NO_COLUMNS` + `NO_CANDIDATE_COLUMNS`（`joint_scheduler.py:438`）；
- `MIP_NO_SOLUTION_STATUS_<n>` + `MIP_NO_USABLE_SOLUTION`（`joint_scheduler.py:455`）。

失败分支不再返回空 audit。

### 4.3 安全回退

`JointSiaPolicy._b3_failure_fallback()`（`joint_integration.py:381`）按以下顺序处理：

1. 对上一轮完整联合分配重新做逐节点 GPU/CPU/RAM 容量审计。
2. 确认所有当前仍分配的活动作业都有上一轮列，且配置、节点未改变。
3. 确认所有真实 checkpoint 锁均由上一轮完整分配保持。
4. 只有全部成立，才返回 `FALLBACK_PRESERVE_PREVIOUS_VALID_ALLOCATION`；新到达或本轮无配置作业继续等待。
5. 若 `NO_COLUMNS` 且当前没有任何分配、没有真实锁，允许 `ALL_WAIT_NO_FEASIBLE_COLUMNS`。
6. 无法证明上一轮状态仍合法时返回失败，原子保护停止该轮，不提交部分解。

成功提交仍在原子容量审计之后发生。B3 两次 20-job 成功运行均没有触发上述 fallback；因此本轮结果不是靠静默复用旧分配跑通的。

### 4.4 异常持久化

- `run_cpu_bottleneck_smoke.py:250`：策略异常附带当前策略、轮次、时间、作业状态、配置、资源和 `last_failure`。
- `run_cpu_contention_20job.py:246`：异常发生时立即写 `<capacity>.<response>.<policy>.failure.json`，并更新 manifest 为 `FAILED_POLICY_EXECUTION`。
- `run_cpu_contention_20job.py:83`：增加严格的 `--targeted-b3` 模式，仅允许冻结的 B3/Neutral/cpu_constrained 组合。

## 5. 修改文件与 diff 摘要

| 文件 | 修改内容 |
|---|---|
| `joint_prototype/joint_scheduler.py` | 防御性真实锁判断；结构化 NO_COLUMNS/MILP 失败审计 |
| `joint_prototype/joint_integration.py` | 修复幽灵锁；保存上一轮完整列；安全回退；`last_failure` |
| `run_cpu_bottleneck_smoke.py` | 逐轮异常 diagnostic snapshot |
| `run_cpu_contention_20job.py` | B3 针对性模式；策略开始前 manifest；即时 failure JSON |
| `tests/test_b3_failure_handling.py` | 新增 8 项失败/回退/容量测试 |
| `tests/test_joint_integration_coverage.py` | 集成路径求解失败测试改为核验安全保留上一轮状态 |
| `diagnose_b3_20job_failure.py` | 修复前根因复现器 |
| `validate_b3_20job_targeted.py` | B3 单策略专项验证器 |

B3 的公平性目标、Sia Goodput、联合列目标系数、资源容量约束及 checkpoint 重配置成本均未修改。

## 6. 测试

最终命令：

```powershell
D:\科研启航\Sia\.venv-sia-baseline\Scripts\python.exe `
  -m unittest discover -s tests -p test_*.py -v
```

结果：**38 tests，38 passed，0 failure，0 error，8.492 s**。

新增 8 项测试分别覆盖：

1. B3 正常求得合法分配；
2. 一个新作业无候选时可行 peer 继续运行；
3. 无当前分配的 `scaling_underway` 不形成幽灵锁；
4. NO_COLUMNS 返回明确 audit；
5. `result.x is None` 返回求解器诊断且不提交；
6. 上一轮合法时安全保留完整分配；
7. 上一轮不合法时拒绝回退；
8. 每个成功选择都通过逐节点三资源容量审计。

既有测试继续覆盖 B1/SY/B4/B5、projected profile、rigid、PMP、多节点、checkpoint 和 Goodput 单次倍率。PMP 测试仍出现既有 autograd `invalid value encountered in log` RuntimeWarning，但测试通过。

## 7. B3 针对性验证

命令：

```powershell
D:\科研启航\Sia\.venv-sia-baseline\Scripts\python.exe run_cpu_contention_20job.py `
  --targeted-b3 `
  --config joint_prototype\cpu_contention_20job_config.json `
  --workload joint_prototype\cpu_contention_20job_workload.csv `
  --synergy-models ..\..\research_sources\Synergy\simulator\models `
  --output ..\..\research_runs\sia_synergy\cpu_contention_20job_b3_repair_targeted `
  --capacities cpu_constrained --responses neutral --policies b3
```

结果：

| 指标 | 值 |
|---|---:|
| 完成作业 | 20/20 |
| 平均 JCT | 3831.05 s |
| 轮数 | 552 |
| 墙钟时间 | 101.189 s |
| rho 范围 | [1.0, 1.0] |
| 求解状态 | `OPTIMAL=494` |
| fallback | 0 |
| 容量违规 | 0 |

专项验证状态为 `PASS`。在历史失败时间 32880 秒，对应的新运行状态为 `OPTIMAL`，活动作业拥有 11 个候选列；由于修复后的调度轨迹已与旧失败轨迹分叉，此时活动作业是 `imagenet-23`，不能把它错误描述为“同一内部状态下的 bert-37”。可核实的结论是：同一冻结输入不再形成空列崩溃，且全过程没有依赖 fallback。

## 8. 五策略完整矩阵

只有专项 B3 验证通过后，才在另一个新目录中按 B1 → SY → B4 → B5 → B3 顺序运行。输入、容量、响应、模板映射和目标均未更改。

### 8.1 运行结果

| 策略 | 完成 | 平均 JCT (s) | 总等待 (s) | 轮数 | 墙钟 (s) | Restart | Wasted GPU-s |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 | 20/20 | 5102.95 | 30000 | 578 | 93.406 | 57 | 13775 |
| SY adapted | 20/20 | 6783.25 | 57180 | 564 | 60.797 | 4 | 10560 |
| B4 | 20/20 | 5056.25 | 28500 | 573 | 94.518 | 70 | 14835 |
| B5 | 20/20 | 5053.30 | 29460 | 590 | 94.647 | 62 | 14095 |
| B3 | 20/20 | 3831.05 | 120 | 552 | 100.358 | 73 | 13485 |

Neutral 下 B3 与各基线的平均 JCT 差值为：B1 -1271.90 秒、SY -2952.20 秒、B4 -1225.20 秒、B5 -1222.25 秒。这里只记录绝对观测值，不将其解释为正式性能提升；该场景仍是合成节点容量与跨硬件响应框架，且本轮目的为修复与稳定性。

B3 与 B1 在共享时间轴上观察到 265 个配置差异事件。由于策略状态会在先前轮次分叉，这一数量证明 B3 确实走了独立联合候选选择路径，但不能逐项解释为同状态下的反事实最优性。

### 8.2 求解与回退

| 策略 | 求解状态摘要 | 真正 fallback |
|---|---|---:|
| B1 | `ORIGINAL_SIA_OPTIMAL=44`, `OPTIMAL=476` | 0 |
| SY | `SYNERGY_TUNE_ADAPTED=506` | 0 |
| B4 | `OPTIMAL=515` | 0（按 status） |
| B5 | `SYNERGY_TUNE_ADAPTED=532` | 0 |
| B3 | `OPTIMAL=494` | 0 |

`comparison_summary.json` 中 B4 的旧 `fallback_count=473` 仍按“存在 `fallback_reason` 字段”累计，并不等同于真正 fallback status；本轮 B4 的 515 个实际求解状态均为 `OPTIMAL`。该旧口径问题未为本次修复而改写，正式性能实验前应重命名或拆分该指标。

### 8.3 资源、等待和 CPU 瓶颈

五策略逐轮和时间加权资源审计均无 GPU/CPU/RAM 超配。节点峰值均不超过：AWS 4 GPU/4 CPU/500 GB、DGX-ext 8/8/1000、RTX 8/6/1000。

- B1/B4/B5/B3 的 RTX 峰值均为 6 GPU、6 CPU，说明 8-GPU RTX 节点继续受 6 CPU floor 约束。
- CPU 饱和且 GPU 空闲事件数：B1=23、SY=0、B4=36、B5=15、B3=53。
- “未分配作业与 CPU 饱和/GPU 空闲同时发生”的轮数：B1=45、SY=0、B4=36、B5=25、B3=2。
- 所有策略都产生实际等待；因此 CPU 约束不是静态配置字段，而是实际影响 admission/placement/执行路径。
- B3 等待总量只有 120 秒，但仍出现 53 个节点级 CPU 饱和/GPU 空闲事件；两者统计对象不同：前者是作业等待秒数，后者是逐节点逐轮资源状态事件。

资源小时：

| 策略 | GPU-hours | CPU-core-hours | RAM-GB-hours |
|---|---:|---:|---:|
| B1 | 91.362 | 91.362 | 11420.278 |
| SY adapted | 119.588 | 119.588 | 14417.784 |
| B4 | 91.773 | 91.773 | 1911.639 |
| B5 | 91.931 | 92.197 | 11474.652 |
| B3 | 90.715 | 90.715 | 1889.593 |

### 8.4 正确性验证

`validation_summary.json` 为 `PASS`，21 项检查全部通过：

- manifest 完整且确为冻结五策略矩阵；
- 每个策略 20/20 完成，无 round limit；
- 瞬时及时间加权 GPU/CPU/RAM 容量合法；
- Neutral `rho` 严格等于 1；
- `effective_goodput = raw_goodput * rho` 且倍率只应用一次；
- B1 admitted config 等于 Sia intent；
- B4/B5 GPU config 冻结；
- SY 使用 adapted TUNE core；
- B3 为作业枚举多个 GPU 候选；
- checkpoint/restore 先消耗时间，完整延迟轮不推进训练；
- 训练进度单调；
- 存在真实排队及 CPU 饱和/GPU 空闲；
- B3 与 B1 的 Sia intent 路径存在差异；
- 原始 Sia 历史回归保持一致。

原始 Sia official/experimental 回归日志 SHA-256 均为：

`F81E0B22C76734E5264742A7231687F366CE77B4D3448155E444A64D42E02DFF`

## 9. 输入、代码与结果 hash

冻结输入 hash 未变化：

| 对象 | SHA-256 |
|---|---|
| config | `2A4E3D7087D0BC66E3EB6987A6E25AC2E04C80B9D8CEF868399F5AAF39E53C11` |
| extracted workload | `46FAB9A9743719B56437E6FC35DADAA1CF397205DCF06483E6A12E7CA488627A` |
| official source workload | `D80DC93782785AAB449D3109AA118F123EA799D891DF0AFE2B9DEE483EDA9468` |
| `alexnet_1.py` | `5BE97BBBABDA711E43903C29EB914C4B45DEC3B7997E3EA60F7ACEAB99943635` |
| `bert_1.py` | `BB3B0DA2CF646A4EF72D8AA4F23D01D1BE23CA21243FC625576FE51BECF44CE2` |
| `deepspeech_1.py` | `CB64753CAB8AA32325588B3E56F0D677379BBD0423678D5404B03295CA381443` |
| `mobilenet_1.py` | `8F6DA649F3A05309A90DF11C8047665BE5CCB53E2DCBDD2D26F990FA4E4A8B7B` |
| `res50_1.py` | `699ADE948408057199B50E6F34B98D71A37BC16638E12FE2EA144DF11D53FD4D` |

关键修复代码 hash：

| 文件 | SHA-256 |
|---|---|
| `joint_scheduler.py` | `65933FFD5BA91F4F4F5855703C2538923B6F040420E127154EF28D44ACE848E7` |
| `joint_integration.py` | `5E1E60F70EE8FD2B347613AA1F7DCB1819A0F018B3DA92253BF863798B8B175B` |
| `run_cpu_bottleneck_smoke.py` | `F125F7F3CFE31BBA89DB6D03708B955C109019AE49E826783A626D56CE8D685E` |
| `run_cpu_contention_20job.py` | `E2E01DC71AF8DBC8DADD51F0520EA58EAB3029BFBCBD207E42BECAC512EAA461` |
| `test_b3_failure_handling.py` | `F9108A208C1F3484FAE13B93312018FF143F0F6161024E41653EDCF80D324871` |
| `validate_b3_20job_targeted.py` | `4C6D1809E903CBAE4E3AE98ED675056F27102BB0DCCEF172713E129E79A1F281` |

关键证据 hash：

| 文件 | SHA-256 |
|---|---|
| diagnostic summary | `8E0E26AE83B128FF1E6B53D408EB6C8E9BD27934FDCE70F161372BE8B16B318C` |
| failure snapshot | `7818A312B558E11B7175FEEE6B00E8CDD2AB53F0E1DFDCF1ACC2F08BE394ABB5` |
| targeted validation | `317A36DC5075E23BE3779345CE02DA5CB20155406A8C582BE7A0EC040AA8A0FA` |
| full run manifest | `58FAF6C4B7D3FCBBBE71BFEAAC55918D07DEC7FD6A01C7F3BE22D39BF7EB006F` |
| comparison summary | `09ECE5326D228C3B9555CD38006BE400EF6A46E537CB781D79B5DA7C26F126AF` |
| validation summary | `A5FE11D6DE51B8DB63AA8D65B4F6238F9563B193F4421D5F0C6F27EC762663F2` |
| targeted B3 payload | `2AEAE537E3D1099C9B5F7E0ABCEBAAAC91DC760396D4500DBF8715FC0D744CCC` |
| full B1 payload | `484F91A900401666BC73A5B2E065DF6423923E128B8D4D1D3477D822750DE2F2` |
| full SY payload | `C344F8A5CD3D1FA1C59D98760C3D6E887AFCC61C1AD3EF5D7C78C18EAC1BA276` |
| full B4 payload | `A9EB4BC75D7BCB5B29496C0A70ABABB0E4A6F1559B02EE34380AD7663639E5AC` |
| full B5 payload | `0C66D5534E730E884B29EED7E8BC6A3F05BE0B41AC358DB43B17B8C7CC3A0212` |
| full B3 payload | `C272BE67F19349AD675CA39E48864C218677BA1BD6642F625751C6B70A8F0FA1` |

完整清单见 `research_runs/sia_synergy/cpu_contention_20job_repair_full/repair_manifest.json`。

## 10. 是否具备正式性能实验条件

从**执行稳定性与正确性门槛**看，修复后的实现已经满足进入下一轮预注册 20-job 对照的必要工程条件：五策略均完成、B3 不再出现空 audit、三资源容量与 checkpoint 语义通过、Neutral 倍率与原始 Sia 回归均通过。

但还不应直接宣称具备真实集群性能结论，原因是：

1. CPU/RAM 响应仍是 Synergy 矩阵到 Sia workload 的 synthetic cross-hardware transfer；
2. 本轮只运行 Neutral，没有运行 Medium，也没有验证真实 CPU/RAM 性能倍率；
3. B4 `fallback_count` 的旧统计口径需要在正式报告前拆分；
4. B4 在相同冻结输入下相对旧 run1 有小幅数值差异（旧 5083.75 s，本轮 5056.25 s），需在正式性能实验前固定/记录求解器确定性和重复次数方案；
5. 工作树尚未形成干净、可引用的实验 commit，正式评估前应由研究者审查并冻结本轮代码。

因此建议的判定是：

**允许在审查并冻结代码后开展预注册的合成 20-job 性能实验；尚不允许把结果外推为真实 Saturn 性能，也不应直接启动 160-job 或参数搜索。**

本轮按要求停止，没有运行 Medium、160-job 或参数搜索。
